"""ROS2 node driving a remote DROID-SLAM server.

Same transport as the VLA node, different payload — and for the same reason. DROID-SLAM
depends on `lietorch` and `droid_backends`, both CUDA-compile-only extensions that will
never build on Apple Silicon, so the SLAM system has to run off-board whatever we do. The
node therefore carries no torch: it forwards frames and republishes poses.

Endpoint contract (`droidSLAM_monocular/slam_server.py` registers these on upstream's PolicyServer):

    ping     -> {"ok": True}
    reset    -> {"ok": True}                      clears the keyframe graph
    track    -> {"translation": [3], "quaternion": [4],   scalar-last, camera-to-world
                 "n_keyframes": int, "tracking_lost": bool, "metric": bool}

**RGB-D is opt-in and changes what the poses mean.** With `use_depth`, each colour frame
is paired with an aligned depth frame and DROID uses it as an inverse-depth prior, which
also disables its scale normalisation — so the trajectory is in metres rather than an
arbitrary scale. `metric` in the reply says which, and it is not cosmetic: every
downstream metre in this pipeline (TSDF fusion, the object frame, the pre-grasp offset)
inherits that gauge.

Two things about it are deliberate:

  * `track` returns the pose as a NAMED translation/quaternion pair rather than a bare
    7-vector, so the scalar-last convention is stated on the wire instead of inferred
    from position. `slam_decode` accepts both, but a server we control should not make
    its client guess;
  * poses are **camera-to-world**, matching `droid.terminate()`, which returns
    `camera_trajectory.inv()` — the internal `video.poses` are world-to-camera. That is
    the direction `nav_msgs/Odometry` wants: where the camera IS in the map frame.

Frames are dropped while a request is in flight. DROID keeps its own keyframe graph and
decides what is a keyframe; queueing every camera frame would only add latency to a pose
that is already superseded.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image

from .latency import LatencyTracker, Throttle, monotonic_ns
from .slam_decode import SlamDecodeError, decode_pose, is_tracking_lost
from .vla_policy_node import image_to_array
from .wire import PolicyClient

# ROS carries depth as uint16 counts or float32 metres; neither states its own units.
_DEPTH_DTYPES = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}


def depth_to_metres(msg: Image, scale: float = 0.0) -> np.ndarray:
    """sensor_msgs/Image -> (H, W) float32 metres, invalid pixels exactly 0.

    `scale` <= 0 means "infer from the encoding": 16UC1/mono16 are millimetre counts
    (0.001 m per unit, the RealSense default) and 32FC1 is already metres. Pass it
    explicitly for a sensor that disagrees — some publish 0.0001 m units — because
    nothing downstream can detect a wrong scale. It does not crash, it resizes the map,
    and every metre in the object-centric frame inherits the error.

    Invalid returns arrive as 0 on uint16 streams and as NaN on float streams. DROID's
    sentinel is 0 (`depth_video.append`: `where(depth > 0, 1/depth, depth)`), so NaN is
    normalised here rather than refused at the server — converting a ROS convention is
    the node's job.
    """
    dtype = _DEPTH_DTYPES.get(msg.encoding)
    if dtype is None:
        raise ValueError(
            f"unsupported depth encoding {msg.encoding!r}; expected one of "
            f"{sorted(_DEPTH_DTYPES)}. An RGB (bgr8/rgb8) topic here means the depth "
            "topic is misconfigured."
        )
    if msg.is_bigendian:
        raise ValueError("big-endian depth is not handled; bytes would be transposed")

    if scale <= 0.0:
        scale = 1.0 if msg.encoding == "32FC1" else 0.001

    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype=dtype)
    arr = arr.reshape(msg.height, msg.step // itemsize)[:, : msg.width]

    depth = arr.astype(np.float32) * scale
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return np.ascontiguousarray(depth)


class SlamNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("droid_slam_node", **kwargs)

        p = self.declare_parameter
        p("host", "localhost")
        p("port", 5557)  # 5555 GR00T, 5556 DiVLA, 5557 SLAM
        p("timeout_ms", 10000)
        p("image_topic", "/camera/color/image_raw")
        p("camera_info_topic", "/camera/color/camera_info")
        # RGB-D is opt-in: it changes the trajectory's gauge from arbitrary-scale to
        # metres, and a stream cannot switch mid-run (the server latches the mode).
        p("use_depth", False)
        p("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        p("depth_scale", 0.0)        # 0 => infer from encoding; see depth_to_metres
        p("max_depth_dt_s", 0.05)    # ~1.5 frames at 30Hz
        p("latency_log_s", 5.0)      # 0 disables the periodic latency summary
        # fx, fy, cx, cy. Left empty to take them from CameraInfo instead, which is the
        # ROS-idiomatic source and avoids a second place for them to go stale.
        p("camera_intrinsics", [0.0])
        p("map_frame", "map")
        p("camera_frame", "camera_link")
        p("publish_tf", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = g("map_frame")
        self.camera_frame = g("camera_frame")

        intr = [float(v) for v in g("camera_intrinsics")]
        self.intrinsics: np.ndarray | None = (
            np.asarray(intr, dtype=np.float32) if len(intr) == 4 else None
        )

        self.use_depth = bool(g("use_depth"))
        self.depth_scale = float(g("depth_scale"))
        self.max_depth_dt_ns = int(float(g("max_depth_dt_s")) * 1e9)

        self.client = PolicyClient(g("host"), int(g("port")), int(g("timeout_ms")))
        self._lock = threading.Lock()
        self._lost_since: int | None = None

        # The tracking request runs on its own thread, NOT in the subscription callback.
        # Called inline it blocks the executor for the whole round trip -- which on a
        # single-threaded executor also stalls the depth and camera_info callbacks, so
        # the depth cache goes stale while we wait for the reply that needs it.
        # `_pending` is latest-wins: a frame that arrives while one is in flight
        # replaces it rather than queueing, because a stale pose is worse than a gap.
        self._pending: tuple | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.latency = LatencyTracker()
        self._log_throttle = Throttle(float(g("latency_log_s")))
        self._worker = threading.Thread(target=self._track_loop, daemon=True,
                                        name="droid_slam_track")
        self._depth: tuple[int, np.ndarray] | None = None
        self._announced_gauge = False

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, g("image_topic"), self._on_image, sensor_qos)
        self.create_subscription(
            CameraInfo, g("camera_info_topic"), self._on_camera_info, sensor_qos
        )
        if self.use_depth:
            self.create_subscription(
                Image, g("depth_topic"), self._on_depth, sensor_qos
            )

        self.odom_pub = self.create_publisher(Odometry, "/droid/odometry", 10)
        self.tf_pub = None
        if g("publish_tf"):
            # Imported here so the node still constructs where tf2_ros is absent.
            from tf2_ros import TransformBroadcaster

            self.tf_pub = TransformBroadcaster(self)

        if not self.client.ping():
            self.get_logger().warn(
                f"SLAM server at {g('host')}:{g('port')} did not answer ping — "
                "will keep retrying on each frame"
            )
        else:
            try:
                self.client.call_endpoint("reset", {"options": None})
            except (TimeoutError, RuntimeError) as exc:
                self.get_logger().warn(f"reset failed: {exc}")

        self._worker.start()
        self.get_logger().info(
            f"droid_slam_node up: {self.map_frame} -> {self.camera_frame}, "
            f"intrinsics {'from parameter' if self.intrinsics is not None else 'from CameraInfo'}"
        )

    # ------------------------------------------------------------- callbacks

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.intrinsics is not None:
            return  # an explicit parameter wins; do not flip-flop mid-run
        k = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
        self.intrinsics = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], np.float32)
        self.get_logger().info(f"intrinsics from CameraInfo: {self.intrinsics.tolist()}")

    def _on_depth(self, msg: Image) -> None:
        """Cache the latest depth frame; the colour stream drives tracking."""
        try:
            depth = depth_to_metres(msg, self.depth_scale)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        stamp = msg.header.stamp
        self._depth = (int(stamp.sec) * 10**9 + int(stamp.nanosec), depth)

    def _paired_depth(self, stamp_ns: int) -> np.ndarray | None:
        """The cached depth iff it is close enough in time to this colour frame.

        Returning a stale depth map would be worse than returning none: it is a wrong
        inverse-depth prior at every pixel where anything moved, fed straight into the
        BA. And "none" is not an option either once the stream has latched RGB-D mode,
        so the caller drops the frame entirely.
        """
        if self._depth is None:
            return None
        depth_stamp, depth = self._depth
        if abs(depth_stamp - stamp_ns) > self.max_depth_dt_ns:
            return None
        return depth

    def _on_image(self, msg: Image) -> None:
        # No in-flight guard here any more: `_pending` is latest-wins, so a frame
        # arriving mid-request replaces the queued one and is counted as dropped. The
        # old guard existed only because the request ran inline on this thread.
        if self.intrinsics is None:
            self.get_logger().warn(
                "no camera intrinsics yet; DROID cannot back-project without them",
                throttle_duration_sec=5.0,
            )
            return
        try:
            frame = image_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return

        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 10**9 + int(stamp.nanosec)
        payload = {
            "image": frame,
            "intrinsics": self.intrinsics,
            "stamp_ns": stamp_ns,
        }

        if self.use_depth:
            depth = self._paired_depth(stamp_ns)
            if depth is None:
                # Drop the frame rather than fall back to monocular: the server latches
                # the mode on frame 0, so a depth-less frame mid-stream is refused, and
                # a mismatched pair would be accepted while being wrong.
                self.get_logger().warn(
                    "no depth frame within max_depth_dt_s of this colour frame; "
                    "dropping it",
                    throttle_duration_sec=2.0,
                )
                return
            payload["depth"] = depth

        # Hand off and return immediately; the executor stays free to service depth and
        # camera_info while the request is in flight.
        with self._lock:
            if self._pending is not None:
                self.latency.dropped += 1
            self._pending = (payload, msg, stamp_ns)
        self._wake.set()

    # ------------------------------------------------------------- worker thread

    def _track_loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.1):
                continue
            self._wake.clear()
            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                continue
            payload, msg, stamp_ns = item

            # `age` is measured against a ROS header stamp, so it uses the ROS clock.
            # Everything below is a duration and uses monotonic -- a ROS-clock step
            # would otherwise show up as a negative round trip.
            age_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1e6
            t_send = monotonic_ns()
            try:
                reply = self.client.call_endpoint("track", payload)
            except (TimeoutError, RuntimeError) as exc:
                self.latency.failed += 1
                self.get_logger().warn(f"track failed: {exc}", throttle_duration_sec=2.0)
                continue
            t_reply = monotonic_ns()

            self._publish(reply, msg)
            t_done = monotonic_ns()

            if age_ms >= 0:      # a backwards ROS-clock step; the durations are still fine
                self.latency.record(
                    age=age_ms,
                    request=(t_reply - t_send) / 1e6,
                    publish=(t_done - t_reply) / 1e6,
                    total=age_ms + (t_done - t_send) / 1e6,
                )
            if self._log_throttle.interval_ns and self._log_throttle.ready():
                self.get_logger().info(self.latency.format())

    # ----------------------------------------------------------------- output

    def _publish(self, reply, msg: Image) -> None:
        if not isinstance(reply, dict):
            self.get_logger().error(
                f"track returned {type(reply).__name__}, expected a dict",
                throttle_duration_sec=5.0,
            )
            return

        if is_tracking_lost(reply):
            # A lost-tracking reply still carries a well-formed pose. Republishing it as
            # current odometry is worse than a gap: downstream cannot tell it is stale.
            self.get_logger().warn(
                "SLAM tracking lost; publishing nothing until it recovers",
                throttle_duration_sec=2.0,
            )
            return

        try:
            translation, quaternion = decode_pose(reply)
        except SlamDecodeError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return

        if not self._announced_gauge:
            self._announced_gauge = True
            metric = bool(reply.get("metric", False))
            self.get_logger().info(
                "tracking; trajectory is "
                + ("METRIC (RGB-D, scale fixed by sensor depth)" if metric
                   else "UP-TO-SCALE (monocular) — translations are not metres")
            )

        odom = Odometry()
        odom.header.stamp = msg.header.stamp  # the frame's stamp, not now()
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.camera_frame
        odom.pose.pose.position.x = float(translation[0])
        odom.pose.pose.position.y = float(translation[1])
        odom.pose.pose.position.z = float(translation[2])
        odom.pose.pose.orientation.x = float(quaternion[0])
        odom.pose.pose.orientation.y = float(quaternion[1])
        odom.pose.pose.orientation.z = float(quaternion[2])
        odom.pose.pose.orientation.w = float(quaternion[3])
        # Twist is left zero: DROID estimates poses, not velocities, and a differentiated
        # pose would be a fabricated number wearing an official-looking field.
        self.odom_pub.publish(odom)

        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header = odom.header
            tf.child_frame_id = self.camera_frame
            tf.transform.translation.x = odom.pose.pose.position.x
            tf.transform.translation.y = odom.pose.pose.position.y
            tf.transform.translation.z = odom.pose.pose.position.z
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf_pub.sendTransform(tf)

    def destroy_node(self) -> bool:
        self._stop.set()
        self._wake.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.client.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

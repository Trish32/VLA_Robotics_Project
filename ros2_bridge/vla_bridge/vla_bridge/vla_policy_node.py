"""ROS2 node driving a VLA policy server (GR00T N1.6 or DiffusionVLA).

The model runs elsewhere — on the GCP L4 or a local CUDA box — behind GR00T's
PolicyServer. This node only gathers observations, calls the server, and turns the
returned action chunk into a JointTrajectory. It carries no torch.

Because the same PolicyServer interface fronts both models, switching between GR00T
and DiVLA is a host/port change, not a code change.

Chunking and latency
--------------------
A VLA predicts a chunk of `H` future actions at once, which is what makes remote
inference viable: one round trip covers H control steps. We execute `execute_k < H`
of them and re-query, so a fresh chunk is always in flight before the current one
runs out. If the reply is late, the arm finishes the actions it already has rather
than stopping dead.

Safety-relevant behaviours, all deliberate:
  * observations older than `max_obs_age_s` are refused — acting on a stale camera
    frame is worse than not acting;
  * a late chunk whose observation has aged past that bound is dropped, not executed;
  * the node publishes nothing at all until every configured input has been seen once.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .action_decode import ActionDecodeError, declared_action_keys, decode_chunk
from .obs_encode import (
    ObservationEncodeError,
    build_observation,
    parse_layout,
    reconcile,
)
from .wire import PolicyClient

_ENCODINGS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def image_to_array(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> HxWxC uint8, RGB. Avoids a cv_bridge dependency."""
    channels = _ENCODINGS.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.step // channels, channels)[:, : msg.width, :]
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


class VLAPolicyNode(Node):
    def __init__(self, **kwargs) -> None:
        # kwargs forwards rclpy's parameter_overrides so tests can configure the node
        # without a YAML file or a running parameter server.
        super().__init__("vla_policy_node", **kwargs)

        p = self.declare_parameter
        p("host", "localhost")
        p("port", 5555)
        p("timeout_ms", 5000)
        # Map ROS image topics onto the server's video modality keys, pairwise.
        p("camera_topics", ["/camera/color/image_raw"])
        p("camera_modality_keys", ["video.ego_view"])
        p("joint_state_topic", "/joint_states")
        # Real embodiments split the state across several keys (GR1: left_arm 7,
        # left_hand 6, right_arm 7, right_hand 6, waist 3). `state_dims` says how to cut
        # `joint_names` between them, in the same order; leave it empty for a single key.
        p("state_modality_keys", ["state.single_arm"])
        p("state_dims", [0])
        p("task_topic", "/vla/task")
        p("task_description", "")
        p("joint_names", [""])
        # N1.6 predicts state-relative chunks for most embodiments; N1.5 and DiVLA
        # predict absolute targets. Getting this wrong sends the arm to the origin.
        p("state_relative", True)
        p("execute_k", 8)
        p("control_dt", 1.0 / 30.0)  # N1.6 System 1 runs at 30Hz
        p("max_obs_age_s", 0.25)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cam_topics = list(g("camera_topics"))
        self.cam_keys = list(g("camera_modality_keys"))
        if len(self.cam_topics) != len(self.cam_keys):
            raise ValueError("camera_topics and camera_modality_keys must be the same length")

        self.joint_names = [n for n in g("joint_names") if n]
        # rclpy infers a parameter's type from its default, so an empty list default is
        # not expressible; the sentinel entries mean "unset" and are dropped here.
        self.state_keys = [k for k in g("state_modality_keys") if k]
        self.state_dims = [int(d) for d in g("state_dims") if int(d) > 0]
        self.state_relative = bool(g("state_relative"))
        self.execute_k = int(g("execute_k"))
        self.control_dt = float(g("control_dt"))
        self.max_obs_age_s = float(g("max_obs_age_s"))
        self.task = g("task_description")

        self.client = PolicyClient(g("host"), int(g("port")), int(g("timeout_ms")))

        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._joint: JointState | None = None
        self._chunk: deque[np.ndarray] = deque()
        self._inflight = False

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        for topic, key in zip(self.cam_topics, self.cam_keys):
            self.create_subscription(
                Image, topic, self._make_image_cb(key), sensor_qos
            )
        self.create_subscription(JointState, g("joint_state_topic"), self._on_joint, 10)
        self.create_subscription(String, g("task_topic"), self._on_task, 10)

        self.pub = self.create_publisher(JointTrajectory, "/vla/joint_trajectory", 10)
        self.create_timer(self.control_dt * self.execute_k, self._tick)

        # One round trip settles BOTH halves of the contract, neither of which is safe to
        # assume. Action key naming is inconsistent across servers (GR00T's raw
        # Gr00tPolicy returns bare "left_arm"/"waist"; its sim wrapper and DiVLA return
        # "action."-prefixed names), and the observation layout the server validates is
        # nested rather than flat. Both were wrong here once, and both were hidden by a
        # fake server that accepted whatever we sent it — bug_log.txt [1] and [3].
        config = self._fetch_modality_config()
        self.action_keys: list[str] | None = self._discover_action_keys(config)
        self.layout = reconcile(self.cam_keys, self.state_keys, self.state_dims,
                                parse_layout(config))
        if self.joint_names and self.layout.state_dims:
            total = sum(self.layout.state_dims)
            if total != len(self.joint_names):
                raise ValueError(
                    f"state_dims sum to {total} but joint_names lists "
                    f"{len(self.joint_names)} joints"
                )
        self.get_logger().info(
            f"observation layout: video={list(self.layout.video_keys)}, "
            f"state={list(self.layout.state_keys)} dims={list(self.layout.state_dims)}, "
            f"language={self.layout.language_key!r}"
        )

        if not self.client.ping():
            self.get_logger().warn(
                f"policy server at {g('host')}:{g('port')} did not answer ping — "
                "will keep retrying on each tick"
            )
        self.get_logger().info(
            f"vla_policy_node up: {len(self.cam_topics)} camera(s), "
            f"state_relative={self.state_relative}, execute_k={self.execute_k}"
        )

    # ------------------------------------------------------------- callbacks

    def _make_image_cb(self, key: str):
        def cb(msg: Image) -> None:
            try:
                arr = image_to_array(msg)
            except ValueError as exc:
                self.get_logger().error(str(exc), throttle_duration_sec=5.0)
                return
            with self._lock:
                self._frames[key] = arr
        return cb

    def _on_joint(self, msg: JointState) -> None:
        with self._lock:
            self._joint = msg

    def _on_task(self, msg: String) -> None:
        self.get_logger().info(f"task -> {msg.data!r}")
        with self._lock:
            self.task = msg.data

    # ------------------------------------------------------------------ loop

    def _tick(self) -> None:
        if self._inflight:
            return
        obs, current_state = self._build_observation()
        if obs is None:
            return

        self._inflight = True
        try:
            action = self.client.get_action(obs)
        except (TimeoutError, RuntimeError) as exc:
            self.get_logger().warn(f"policy call failed: {exc}", throttle_duration_sec=2.0)
            return
        finally:
            self._inflight = False

        chunk = self._extract_chunk(action)
        if chunk is None:
            return
        if self.state_relative:
            chunk = chunk + current_state[None, :]
        self._publish(chunk[: self.execute_k])

    def _build_observation(self):
        """Assemble the server payload, or (None, None) if inputs are missing/stale."""
        with self._lock:
            missing = [k for k in self.cam_keys if k not in self._frames]
            if missing or self._joint is None:
                self.get_logger().info(
                    f"waiting for inputs: {missing or 'joint_states'}",
                    throttle_duration_sec=5.0,
                )
                return None, None
            frames = {k: v.copy() for k, v in self._frames.items()}
            joint = self._joint

        stamp = joint.header.stamp
        age = (self.get_clock().now().nanoseconds - (stamp.sec * 10**9 + stamp.nanosec)) / 1e9
        if stamp.sec and age > self.max_obs_age_s:
            self.get_logger().warn(
                f"joint_states stale by {age:.3f}s (> {self.max_obs_age_s}s); not acting",
                throttle_duration_sec=2.0,
            )
            return None, None

        names = self.joint_names or list(joint.name)
        try:
            idx = [list(joint.name).index(n) for n in names]
        except ValueError as exc:
            self.get_logger().error(f"joint name not in /joint_states: {exc}",
                                    throttle_duration_sec=5.0)
            return None, None
        state = np.asarray([joint.position[i] for i in idx], dtype=np.float32)

        # The nested (B, T, ...) layout lives in obs_encode so it can be handed to a real
        # policy's check_observation on the Mac. This used to build a flat dict with a
        # single leading axis, which every real server rejects — see bug_log.txt [3].
        try:
            obs = build_observation(frames, state, self.task, self.layout)
        except ObservationEncodeError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return None, None
        return obs, state

    def _fetch_modality_config(self) -> dict | None:
        try:
            return self.client.get_modality_config()
        except (TimeoutError, RuntimeError) as exc:
            self.get_logger().warn(f"could not read modality config: {exc}")
            return None

    def _discover_action_keys(self, config) -> list[str] | None:
        """Which keys the server's action dict will use.

        Returns None if the server declares nothing, in which case `_extract_chunk`
        falls back to prefix detection.
        """
        keys = declared_action_keys(config)
        if keys:
            self.get_logger().info(f"server declares action keys: {keys}")
        return keys

    def _extract_chunk(self, action: dict) -> np.ndarray | None:
        """Decode the reply into (H, dof). Logic lives in `action_decode` so it stays
        testable without rclpy — that is how the bare-vs-prefixed mismatch was found."""
        try:
            return decode_chunk(action, self.action_keys)
        except ActionDecodeError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return None

    def _publish(self, chunk: np.ndarray) -> None:
        with self._lock:
            joint = self._joint
        names = self.joint_names or list(joint.name)
        if chunk.shape[1] != len(names):
            self.get_logger().error(
                f"action width {chunk.shape[1]} != {len(names)} joints; not publishing",
                throttle_duration_sec=5.0,
            )
            return

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = names
        for i, step in enumerate(chunk):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in step]
            t = (i + 1) * self.control_dt
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            traj.points.append(pt)
        self.pub.publish(traj)

    def destroy_node(self) -> bool:
        self.client.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

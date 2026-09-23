"""End-to-end: scripted SLAM server -> droid_slam_node -> Odometry + tf.

Runs inside the ROS2 Jazzy container. The same caveat as `test_node_endtoend.py` applies
and is worth repeating: a scripted server proves the node's plumbing, not DROID's. The
server here replies with the field names `droidSLAM_monocular/slam_server.py` actually sends, and
`droidSLAM_monocular/tests/test_slam_server.py` pins those field names on the server side, so the
two cannot drift apart without one of them failing.

What this does establish, none of which had ever been executed:
  * the node subscribes, calls `track`, and publishes Odometry with the frame's own stamp;
  * intrinsics come from CameraInfo, and nothing is sent before they arrive — DROID
    cannot back-project without them;
  * a lost-tracking reply publishes NOTHING rather than a stale pose;
  * the quaternion reaches `geometry_msgs/Quaternion` scalar-last, unpermuted.
"""

import threading
import time

import numpy as np
import pytest
import rclpy
import zmq
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from vla_bridge import wire
from vla_bridge.slam_node import SlamNode

PORT = 5596
FX, FY, CX, CY = 300.0, 300.0, 160.0, 120.0
# 90 degrees about Z, scalar-LAST. Chosen because a scalar-first misread of the same four
# numbers is [0.7071, 0, 0, 0.7071] -- also unit norm, also "valid", a different rotation.
QUAT = [0.0, 0.0, 0.7071068, 0.7071068]


class ScriptedSlamServer(threading.Thread):
    daemon = True

    def __init__(self, reply, delay=0.0):
        super().__init__()
        self.reply = reply
        self.delay = delay
        self.tracked = []
        self.ready = threading.Event()
        self._stop_evt = threading.Event()

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{PORT}")
        sock.setsockopt(zmq.RCVTIMEO, 200)
        self.ready.set()
        while not self._stop_evt.is_set():
            try:
                req = wire.from_bytes(sock.recv())
            except zmq.error.Again:
                continue
            endpoint = req.get("endpoint")
            if endpoint == "track":
                self.tracked.append(req["data"])
                if self.delay:
                    time.sleep(self.delay)
                sock.send(wire.to_bytes(self.reply))
            else:
                sock.send(wire.to_bytes({"ok": True}))
        sock.close(linger=0)
        ctx.term()

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


class Stimulus(Node):
    def __init__(self):
        super().__init__("slam_stimulus")
        self.img_pub = self.create_publisher(Image, "/cam", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/cam_info", 10)
        self.depth_pub = self.create_publisher(Image, "/depth", 10)

    def publish_info(self):
        info = CameraInfo()
        info.header.stamp = self.get_clock().now().to_msg()
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

    def publish_image(self, stamp=None):
        img = Image()
        img.header.stamp = stamp or self.get_clock().now().to_msg()
        img.height, img.width, img.encoding = 4, 4, "rgb8"
        img.step = 4 * 3
        img.data = bytes(np.full((4, 4, 3), 7, np.uint8).tobytes())
        self.img_pub.publish(img)
        return img.header.stamp

    def publish_depth(self, stamp=None, millimetres=1500, encoding="16UC1"):
        img = Image()
        img.header.stamp = stamp or self.get_clock().now().to_msg()
        img.height, img.width, img.encoding = 4, 4, encoding
        if encoding == "32FC1":
            payload = np.full((4, 4), millimetres / 1000.0, np.float32)
        else:
            payload = np.full((4, 4), millimetres, np.uint16)
        img.step = 4 * payload.dtype.itemsize
        img.data = bytes(payload.tobytes())
        self.depth_pub.publish(img)
        return img.header.stamp


class Sink(Node):
    def __init__(self):
        super().__init__("slam_sink")
        self.msgs = []
        self.create_subscription(Odometry, "/droid/odometry", self.msgs.append, 10)


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def overrides(**extra):
    from rclpy.parameter import Parameter

    values = {
        "port": PORT,
        "image_topic": "/cam",
        "camera_info_topic": "/cam_info",
        "camera_intrinsics": [0.0],   # sentinel: take them from CameraInfo
        "map_frame": "map",
        "camera_frame": "camera_link",
        "publish_tf": False,          # no tf2 broadcaster in this test graph
        "timeout_ms": 2000,
    }
    values.update(extra)
    return [Parameter(k, value=v) for k, v in values.items()]


def run_pipeline(reply, send_info=True, spins=40, depth=None, **params):
    """`depth`: None for monocular, else a dict of publish_depth() kwargs."""
    server = ScriptedSlamServer(reply)
    server.start()
    assert server.ready.wait(timeout=5)
    try:
        node = SlamNode(parameter_overrides=overrides(**params))
        stim, sink = Stimulus(), Sink()
        executor = SingleThreadedExecutor()
        for n in (node, stim, sink):
            executor.add_node(n)
        try:
            if send_info:
                for _ in range(5):
                    stim.publish_info()
                    executor.spin_once(timeout_sec=0.05)
            for _ in range(spins):
                if depth is None:
                    stim.publish_image()
                else:
                    # An aligned depth frame carries the colour frame's own stamp, so
                    # pair them on one stamp rather than on wall-clock proximity -- the
                    # spins below would otherwise drift them past max_depth_dt_s. The
                    # stale-depth test overrides `stamp` to exercise the refusal.
                    stamp = stim.get_clock().now().to_msg()
                    kwargs = dict(depth)
                    kwargs.setdefault("stamp", stamp)
                    stim.publish_depth(**kwargs)
                    # Depth must be cached before the colour frame it pairs with.
                    for _ in range(3):
                        executor.spin_once(timeout_sec=0.01)
                    stim.publish_image(stamp=stamp)
                executor.spin_once(timeout_sec=0.05)
            deadline = time.time() + 2
            while not sink.msgs and time.time() < deadline:
                executor.spin_once(timeout_sec=0.05)
            return server, sink.msgs
        finally:
            for n in (node, stim, sink):
                executor.remove_node(n)
                n.destroy_node()
    finally:
        server.stop()


GOOD_REPLY = {"translation": [1.0, 2.0, 3.0], "quaternion": QUAT,
              "n_keyframes": 7, "tracking_lost": False}


def test_pose_reaches_odometry(ros):
    server, msgs = run_pipeline(GOOD_REPLY)

    assert server.tracked, "the node never called track"
    assert msgs, "no odometry published"
    pose = msgs[0].pose.pose
    assert (pose.position.x, pose.position.y, pose.position.z) == (1.0, 2.0, 3.0)


def test_quaternion_is_not_permuted(ros):
    """lietorch and ROS are both scalar-last, so the four numbers must carry across
    untouched. A swapped layout is still unit norm and describes a different rotation."""
    _, msgs = run_pipeline(GOOD_REPLY)

    q = msgs[0].pose.pose.orientation
    assert (q.x, q.y, q.z, q.w) == pytest.approx(tuple(QUAT), abs=1e-6)


def test_frames_and_stamp_come_from_the_image_not_now(ros):
    """Odometry stamped with `now()` instead of the frame's own time desynchronises
    every consumer that interpolates poses."""
    _, msgs = run_pipeline(GOOD_REPLY)

    msg = msgs[0]
    assert msg.header.frame_id == "map"
    assert msg.child_frame_id == "camera_link"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) != (0, 0)


def test_intrinsics_are_taken_from_camera_info(ros):
    server, _ = run_pipeline(GOOD_REPLY)

    sent = np.asarray(server.tracked[0]["intrinsics"], dtype=np.float64)
    assert np.allclose(sent, [FX, FY, CX, CY]), "K was misread"


def test_nothing_is_sent_before_intrinsics_arrive(ros):
    """DROID cannot back-project without them, and a default guess would be a fabricated
    calibration producing a confidently wrong trajectory."""
    server, msgs = run_pipeline(GOOD_REPLY, send_info=False)

    assert not server.tracked, "the node called track with no intrinsics"
    assert not msgs


def test_lost_tracking_publishes_nothing(ros):
    """The reply still carries a well-formed pose; only the flag says it is stale.
    Republishing it as current odometry is worse than a gap, because downstream cannot
    tell the difference."""
    lost = dict(GOOD_REPLY, tracking_lost=True)
    server, msgs = run_pipeline(lost)

    assert server.tracked, "the node should still be calling track"
    assert not msgs, "a stale pose was published as current odometry"


def test_a_pre_keyframe_reply_publishes_nothing(ros):
    """Before the motion filter accepts a first keyframe there is no pose at all."""
    _, msgs = run_pipeline({"n_keyframes": 0, "tracking_lost": True})

    assert not msgs


def test_a_malformed_pose_is_dropped_not_published(ros):
    """A quaternion norm far from 1 is a layout or scaling bug; renormalising it would
    hide exactly the thing worth catching."""
    _, msgs = run_pipeline({"translation": [0.0, 0.0, 0.0],
                            "quaternion": [3.0, 4.0, 0.0, 0.0],   # norm 5
                            "n_keyframes": 3})

    assert not msgs


def test_the_image_is_forwarded_as_rgb_uint8(ros):
    server, _ = run_pipeline(GOOD_REPLY)

    image = np.asarray(server.tracked[0]["image"])
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    assert (image == 7).all()


# ------------------------------------------------------------------------- RGB-D
#
# Depth reaches DROID as a per-pixel inverse-depth prior inside the BA, so a unit error
# or a mispaired frame is a wrong constraint, never an exception. These pin the unit
# conversion and the refusal paths.


def test_monocular_is_the_default_and_sends_no_depth(ros):
    """RGB-D changes the trajectory's gauge, so it must be asked for explicitly."""
    server, _ = run_pipeline(GOOD_REPLY)

    assert "depth" not in server.tracked[0]


def test_millimetre_counts_are_converted_to_metres(ros):
    """16UC1 does not state its own units. RealSense publishes millimetres; sending the
    raw counts would put the map 1000x too far away without raising anything."""
    server, _ = run_pipeline(
        GOOD_REPLY, depth={"millimetres": 1500}, use_depth=True, depth_topic="/depth"
    )

    depth = np.asarray(server.tracked[0]["depth"])
    assert depth.shape == (4, 4) and depth.dtype == np.float32
    assert np.allclose(depth, 1.5)


def test_float_depth_is_already_metres_and_is_not_rescaled(ros):
    """Applying the 16UC1 millimetre factor to a 32FC1 stream shrinks the map 1000x."""
    server, _ = run_pipeline(
        GOOD_REPLY,
        depth={"millimetres": 2000, "encoding": "32FC1"},
        use_depth=True,
        depth_topic="/depth",
    )

    assert np.allclose(np.asarray(server.tracked[0]["depth"]), 2.0)


def test_a_stale_depth_frame_drops_the_colour_frame(ros):
    """Pairing a colour frame with depth from a different moment is a wrong prior at
    every pixel where anything moved. The server has already latched RGB-D mode, so
    sending the frame without depth is not an option either — the frame is dropped."""
    from builtin_interfaces.msg import Time

    server, msgs = run_pipeline(
        GOOD_REPLY,
        depth={"stamp": Time(sec=1, nanosec=0)},   # far from the image's clock stamp
        use_depth=True,
        depth_topic="/depth",
        max_depth_dt_s=0.05,
    )

    assert not server.tracked, "a mispaired frame was forwarded anyway"
    assert not msgs


def test_invalid_depth_pixels_arrive_as_exactly_zero(ros):
    """DROID's sentinel is 0 (`where(depth > 0, 1/depth, depth)`). A NaN from a float
    depth stream would propagate into the BA instead of meaning 'no prior'."""
    from sensor_msgs.msg import Image as ImageMsg

    from vla_bridge.slam_node import depth_to_metres

    msg = ImageMsg()
    msg.height, msg.width, msg.encoding = 2, 2, "32FC1"
    msg.step = 2 * 4
    msg.data = bytes(np.array([[np.nan, 1.0], [-3.0, np.inf]], np.float32).tobytes())

    out = depth_to_metres(msg)
    assert out.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_an_rgb_topic_on_the_depth_input_is_refused(ros):
    """The commonest misconfiguration, and bgr8 bytes reinterpreted as uint16 would be
    well-shaped garbage."""
    from sensor_msgs.msg import Image as ImageMsg

    from vla_bridge.slam_node import depth_to_metres

    msg = ImageMsg()
    msg.height, msg.width, msg.encoding = 2, 2, "bgr8"
    msg.step = 2 * 3
    msg.data = bytes(np.zeros((2, 2, 3), np.uint8).tobytes())

    with pytest.raises(ValueError, match="unsupported depth encoding"):
        depth_to_metres(msg)


def test_the_image_callback_returns_without_waiting_for_the_server(ros):
    """The fix for the real latency bug: the ZMQ round trip used to run INSIDE the
    subscription callback. On a single-threaded executor that stalls every other
    callback for the whole request — including the depth and camera_info ones, so the
    depth cache goes stale while we block waiting for the reply that needs it.

    With the request on a worker thread the callback just hands off. Here the server
    sits on the request for 1.5 s; spinning the executor must not take anywhere near
    that long.
    """
    server = ScriptedSlamServer(GOOD_REPLY, delay=1.5)
    server.start()
    assert server.ready.wait(timeout=5)
    try:
        node = SlamNode(parameter_overrides=overrides())
        stim = Stimulus()
        executor = SingleThreadedExecutor()
        for n in (node, stim):
            executor.add_node(n)
        try:
            for _ in range(5):
                stim.publish_info()
                executor.spin_once(timeout_sec=0.05)

            stim.publish_image()
            start = time.monotonic()
            executor.spin_once(timeout_sec=0.05)   # delivers the image callback
            elapsed = time.monotonic() - start

            assert elapsed < 0.5, (
                f"the image callback blocked for {elapsed:.2f}s against a 1.5s server — "
                "the request is running on the executor thread again"
            )

            # And the executor is still live while the request is outstanding.
            info_start = time.monotonic()
            stim.publish_info()
            executor.spin_once(timeout_sec=0.05)
            assert time.monotonic() - info_start < 0.5
        finally:
            for n in (node, stim):
                executor.remove_node(n)
                n.destroy_node()
    finally:
        server.stop()


def test_a_frame_arriving_mid_request_is_dropped_not_queued(ros):
    """Latest-wins. Queueing would hand DROID a backlog of frames whose poses are
    already superseded, and the queue would grow without bound under load."""
    server, msgs = run_pipeline(GOOD_REPLY, spins=25)
    assert server.tracked, "nothing reached the server at all"
    # Far fewer requests than frames published, and no unbounded growth.
    assert len(server.tracked) < 25

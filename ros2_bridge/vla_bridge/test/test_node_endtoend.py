"""End-to-end test: fake policy server -> vla_policy_node -> JointTrajectory.

Runs inside the ROS2 Jazzy container. Verifies the parts that are easy to get
wrong and expensive to discover on hardware:
  * state-relative chunks are integrated against the CURRENT joint state;
  * only `execute_k` of the H predicted steps are published;
  * time_from_start increments by control_dt.

**What this file cannot do**, and the reason it now says so out loud: a fake server
agrees with whatever you send it. Three protocol bugs have hidden behind this test —
the request envelope, the action key convention, and the observation layout
(`../../bug_log.txt` [1], [3]). The fake below therefore validates the observation it
receives the way a real server would, and the authority on that shape is
`grootN1_Robotics/tests/test_node_observation_against_real_policy.py`, which feeds the node's
own encoder to the genuine policy's `check_observation`.

Run: source /opt/ros/jazzy/setup.bash && python3 -m pytest test/test_node_endtoend.py -q
"""

import threading
import time

import numpy as np
import pytest
import rclpy
import zmq
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

from vla_bridge import wire
from vla_bridge.vla_policy_node import VLAPolicyNode, image_to_array

PORT = 5599
DOF, HORIZON, EXECUTE_K = 3, 16, 4
CONTROL_DT = 0.05
# Deterministic chunk: step i is i+1 in every joint, so integration is easy to read.
CHUNK = np.tile(np.arange(1, HORIZON + 1, dtype=np.float32)[:, None], (1, DOF))

# What the fake declares, in the shape a real PolicyServer serialises.
MODALITY_CONFIG = {
    "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
    "state": {"modality_keys": ["single_arm"], "delta_indices": [0]},
    "language": {"modality_keys": ["annotation.human.task_description"],
                 "delta_indices": [0]},
    "action": {"modality_keys": ["action.arm"], "delta_indices": list(range(HORIZON))},
}


def check_observation(obs) -> None:
    """The subset of `Gr00tPolicy.check_observation` that constrains the wire shape.

    Transcribed from upstream rather than invented, so the fake cannot drift into
    accepting something the real policy would reject — which is exactly how the flat
    payload survived for as long as it did.
    """
    assert set(obs) >= {"video", "state", "language"}, f"not nested: {sorted(obs)}"
    for key in MODALITY_CONFIG["video"]["modality_keys"]:
        arr = obs["video"][key]
        assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
        assert arr.ndim == 5, f"video must be (B, T, H, W, C), got {arr.shape}"
        assert arr.shape[1] == 1 and arr.shape[-1] == 3
    for key in MODALITY_CONFIG["state"]["modality_keys"]:
        arr = obs["state"][key]
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
        assert arr.ndim == 3, f"state must be (B, T, D), got {arr.shape}"
    for key in MODALITY_CONFIG["language"]["modality_keys"]:
        value = obs["language"][key]
        assert isinstance(value, list) and isinstance(value[0], list)
        assert len(value[0]) == 1 and isinstance(value[0][0], str)


class FakeServer(threading.Thread):
    """Minimal PolicyServer: REP socket speaking the same msgpack wire format."""

    daemon = True

    def __init__(self):
        super().__init__()
        self.received = []
        self.rejected = []
        self._stop_evt = threading.Event()  # not _stop: shadows Thread._stop

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{PORT}")
        sock.setsockopt(zmq.RCVTIMEO, 200)
        while not self._stop_evt.is_set():
            try:
                req = wire.from_bytes(sock.recv())
            except zmq.error.Again:
                continue
            endpoint = req.get("endpoint")
            if endpoint == "ping":
                sock.send(wire.to_bytes({"ok": True}))
            elif endpoint == "get_modality_config":
                sock.send(wire.to_bytes(MODALITY_CONFIG))
            elif endpoint == "get_action":
                self.received.append(req["data"])
                try:
                    check_observation(req["data"]["observation"])
                except AssertionError as exc:
                    # Recorded rather than raised: this thread's exception would be
                    # invisible, and the test asserts on `rejected` explicitly.
                    self.rejected.append(str(exc))
                    sock.send(b"ERROR")
                    continue
                # Upstream's BasePolicy.get_action returns (action, info); the wire
                # delivers that as a two-element list, and the client unwraps it.
                sock.send(wire.to_bytes([{"action.arm": CHUNK}, {}]))
            else:
                sock.send(wire.to_bytes({}))
        sock.close(linger=0)
        ctx.term()

    def stop(self):
        self._stop_evt.set()


class Stimulus(Node):
    """Publishes a camera frame and a joint state at a fixed offset."""

    def __init__(self, positions):
        super().__init__("stimulus")
        self.img_pub = self.create_publisher(Image, "/cam", 10)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.positions = positions

    def publish(self):
        now = self.get_clock().now().to_msg()
        img = Image()
        img.header.stamp = now
        img.height, img.width, img.encoding = 4, 4, "rgb8"
        img.step = 4 * 3
        img.data = bytes(np.full((4, 4, 3), 7, dtype=np.uint8).tobytes())
        self.img_pub.publish(img)

        js = JointState()
        js.header.stamp = now
        js.name = [f"j{i}" for i in range(DOF)]
        js.position = list(self.positions)
        self.js_pub.publish(js)


class Sink(Node):
    def __init__(self):
        super().__init__("sink")
        self.msgs = []
        self.create_subscription(JointTrajectory, "/vla/joint_trajectory",
                                 self.msgs.append, 10)


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def overrides(state_relative):
    from rclpy.parameter import Parameter

    values = {
        "port": PORT,
        "camera_topics": ["/cam"],
        "camera_modality_keys": ["video.ego_view"],
        "joint_names": [f"j{i}" for i in range(DOF)],
        "task_description": "test task",
        "state_relative": state_relative,
        "execute_k": EXECUTE_K,
        "control_dt": CONTROL_DT,
        "max_obs_age_s": 5.0,
    }
    return [Parameter(k, value=v) for k, v in values.items()]


def run_pipeline(positions, state_relative):
    server = FakeServer()
    server.start()
    time.sleep(0.3)

    node = VLAPolicyNode(parameter_overrides=overrides(state_relative))
    stim = Stimulus(positions)
    sink = Sink()
    ex = SingleThreadedExecutor()
    for n in (node, stim, sink):
        ex.add_node(n)

    deadline = time.time() + 12.0
    while time.time() < deadline and not sink.msgs:
        stim.publish()
        ex.spin_once(timeout_sec=0.05)

    for n in (node, stim, sink):
        n.destroy_node()
    server.stop()
    server.join(timeout=2)
    return server, sink.msgs


@pytest.mark.parametrize("state_relative", [True, False])
def test_publishes_trajectory(ros, state_relative):
    positions = [0.5, -0.25, 1.0]
    server, msgs = run_pipeline(positions, state_relative)

    assert msgs, "node never published a JointTrajectory"
    traj = msgs[0]

    assert traj.joint_names == [f"j{i}" for i in range(DOF)]
    assert len(traj.points) == EXECUTE_K, "should publish execute_k steps, not the full horizon"

    expected = CHUNK[:EXECUTE_K].copy()
    if state_relative:
        expected = expected + np.asarray(positions, dtype=np.float32)[None, :]

    got = np.array([p.positions for p in traj.points], dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-5)

    times = [p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in traj.points]
    np.testing.assert_allclose(times, [(i + 1) * CONTROL_DT for i in range(EXECUTE_K)], atol=1e-6)


def test_observation_payload_shape(ros):
    """The nested (B, T, ...) layout, checked field by field.

    The node used to send a flat dict with one leading axis. Every assertion below is a
    thing that was wrong about it.
    """
    server, msgs = run_pipeline([0.0] * DOF, True)
    assert server.received, "server got no observation"
    assert not server.rejected, f"server rejected the observation: {server.rejected}"

    data = server.received[0]
    assert set(data) == {"observation", "options"}, (
        "the server dispatches handler(**data), so the envelope is "
        "{'observation': ..., 'options': ...} — see bug_log.txt [1]"
    )
    obs = data["observation"]

    assert set(obs) == {"video", "state", "language"}, "must be nested, not flat"
    assert obs["video"]["ego_view"].shape == (1, 1, 4, 4, 3), "needs BOTH B and T axes"
    assert obs["video"]["ego_view"].dtype == np.uint8
    assert obs["state"]["single_arm"].shape == (1, 1, DOF)
    assert obs["state"]["single_arm"].dtype == np.float32
    assert obs["language"]["annotation.human.task_description"] == [["test task"]]


def test_the_fake_server_would_reject_the_old_flat_payload():
    """Guards the guard: if `check_observation` here stopped constraining anything, the
    test above would pass on any payload again."""
    flat = {"video.ego_view": np.zeros((1, 4, 4, 3), np.uint8),
            "state.single_arm": np.zeros((1, DOF), np.float32),
            "annotation.human.task_description": ["test task"]}

    with pytest.raises((AssertionError, KeyError)):
        check_observation(flat)


def test_image_to_array_handles_row_padding():
    """step > width*channels is legal and common; the extra bytes must be cropped."""
    msg = Image()
    msg.height, msg.width, msg.encoding = 2, 3, "rgb8"
    msg.step = 12  # 4 pixels of stride for 3 pixels of image
    msg.data = bytes(np.arange(2 * 12, dtype=np.uint8).tobytes())
    arr = image_to_array(msg)
    assert arr.shape == (2, 3, 3)


def test_image_to_array_converts_bgr():
    msg = Image()
    msg.height, msg.width, msg.encoding = 1, 1, "bgr8"
    msg.step = 3
    msg.data = bytes(bytearray([10, 20, 30]))
    np.testing.assert_array_equal(image_to_array(msg), np.array([[[30, 20, 10]]], np.uint8))



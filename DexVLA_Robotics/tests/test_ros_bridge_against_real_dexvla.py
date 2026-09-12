"""The ROS bridge against the REAL DexVLA server, not a fake.

The GR00T half of the bridge has been checked this way since
`grootN1_Robotics/tests/test_ros_bridge_against_real_policy.py`, and that check is what found the
request envelope and the missing `options` argument (`ros2_bridge/bug_log.txt` [1], [2]).
The DexVLA half had never been run at all — its server was written by reading upstream,
which is exactly how the observation-layout mismatch survived: two pieces of code, each
plausible, disagreeing about a shape neither one ever saw.

What runs here:

    ros2_bridge/.../obs_encode.build_observation   the payload the NODE builds
      -> vla_bridge.wire.PolicyClient              the torch-free client, over ZMQ
        -> gr00t PolicyServer                      upstream's real server loop
          -> DexVLA_Robotics.policy_server.DivlaPolicy    the real 3.2B model, on CPU
    -> action_decode.decode_chunk                  the node's decoder

Nothing in that chain is stubbed. It is slow (~30s: a Qwen2-VL-2B load plus CPU
generation) and skipped when the backbone is absent.

Note on what this does NOT establish: there is no released DexVLA checkpoint, so the
ScaleDP head is randomly initialised and the actions are meaningless numbers. This is a
protocol test. The values are checked only for shape and finiteness.

Run: PYTHONPATH=grootN1_Robotics/upstream:. pytest DexVLA_Robotics/tests/test_ros_bridge_against_real_divla.py -q
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2_bridge/vla_bridge"))

BACKBONE = ROOT / "DexVLA_Robotics/checkpoints/Qwen2-VL-2B-Instruct"
PORT = 5598  # off DexVLA's 5556 default so a real server is never mistaken for ours
HORIZON, ACTION_DIM, STATE_DIM = 8, 14, 14

pytestmark = pytest.mark.skipif(
    not (BACKBONE / "config.json").exists(),
    reason="needs the Qwen2-VL-2B-Instruct backbone in DexVLA_Robotics/checkpoints",
)


@pytest.fixture(scope="module")
def server():
    """The real PolicyServer wrapping the real DivlaPolicy, in a subprocess."""
    env = dict(os.environ,
               PYTHONPATH=f"{ROOT / 'grootN1_Robotics/upstream'}:{ROOT}")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "DexVLA_Robotics/policy_server.py"),
         "--backbone", str(BACKBONE), "--port", str(PORT),
         "--horizon", str(HORIZON), "--action-dim", str(ACTION_DIM),
         "--state-dim", str(STATE_DIM), "--device", "cpu"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server died during startup:\n{proc.stdout.read()[-3000:]}")
            line = proc.stdout.readline()
            if "[serve] DexVLA on" in line:
                break
        else:
            pytest.fail("server did not bind within 600s")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def client(server):
    from vla_bridge.wire import PolicyClient

    # CPU inference is ~3s and the model load is much longer; a 5s default would time
    # out and leave the REQ socket unusable.
    return PolicyClient("localhost", PORT, timeout_ms=600_000)


@pytest.fixture(scope="module")
def layout(client):
    """Negotiate exactly as the node does, from the server's own modality config."""
    from vla_bridge.obs_encode import parse_layout, reconcile

    declared = parse_layout(client.get_modality_config())
    # DexVLA declares only `action`, so the node falls back to its launch parameters.
    assert declared is None, "DexVLA is expected to declare no video/state/language"
    return reconcile(["video.top"], ["state.joints"], [], declared)


def observation(layout, task="pick up the red block"):
    from vla_bridge.obs_encode import build_observation

    return build_observation({"top": np.zeros((224, 224, 3), np.uint8)},
                             np.zeros(STATE_DIM, np.float32), task, layout)


def test_ping_reaches_the_real_server(client):
    assert client.ping() is True


def test_the_server_declares_only_its_action_key(client):
    config = client.get_modality_config()

    assert set(config) == {"action"}
    assert config["action"]["modality_keys"] == ["action.joints"]
    assert len(config["action"]["delta_indices"]) == HORIZON


def test_the_nodes_observation_is_accepted(client, layout):
    """The mismatch this file exists to catch.

    `DivlaPolicy.get_action` reads `observation["video"]`; the node used to send a flat
    `{"video.top": ...}`, so every call would have raised "observation has no 'video'
    entries" — with a fake server in between, nobody would have found out.
    """
    action = client.get_action(observation(layout))

    assert isinstance(action, dict), f"got {type(action).__name__}, expected a dict"
    assert "action.joints" in action, f"keys were {sorted(action)}"


def test_the_action_decodes_into_a_chunk_the_node_can_publish(client, layout):
    """The other end of the same round trip: what the node does with the reply."""
    from vla_bridge.action_decode import declared_action_keys, decode_chunk

    declared = declared_action_keys(client.get_modality_config())
    chunk = decode_chunk(client.get_action(observation(layout)), declared)

    assert chunk.shape == (HORIZON, ACTION_DIM), (
        "the node publishes one JointTrajectory point per row and one joint per column"
    )
    assert np.isfinite(chunk).all()


def test_a_flat_observation_is_rejected_by_the_real_server(client, layout):
    """Proof the layout requirement is real and not a style preference.

    The server answers `b"ERROR"`, which the client turns into a RuntimeError. Before
    the encoder existed, this is what every tick would have produced.
    """
    obs = observation(layout)
    flat = {"video.top": obs["video"]["top"],
            "state.joints": obs["state"]["joints"],
            "annotation.human.task_description": ["pick up the red block"]}

    with pytest.raises(RuntimeError):
        client.get_action(flat)


def test_the_instruction_reaches_the_model(client, layout):
    """The task string travels ROS -> msgpack -> chat template -> generation.

    Checked through `reset`, which clears the cached reasoning, so a stale value cannot
    make this pass. The reasoning TEXT is not asserted: the head is randomly initialised
    (there is no released DexVLA checkpoint), so the model's words are meaningless.
    """
    client.call_endpoint("reset", {"options": None})
    reply = client.call_endpoint(
        "get_action",
        {"observation": observation(layout, "stack the blue cube on the red one"),
         "options": None},
    )

    assert isinstance(reply, list) and len(reply) == 2, (
        "BasePolicy.get_action returns (action, info); msgpack delivers a 2-element list"
    )
    action, info = reply
    assert "action.joints" in action
    assert isinstance(info.get("reasoning"), str) and info["reasoning"], (
        "the server must return the reasoning it generated, not an empty string"
    )

"""The ROS bridge's wire client against the REAL GR00T policy, not a fake.

`ros2_bridge/.../test_wire_matches_upstream.py` proves our serializer is byte-identical
to upstream's, and `test_node_endtoend.py` drives the node against a hand-written fake
server. Neither proves the genuine 3.29B policy answers in the shape the node expects --
a fake agrees with whatever you wrote it to agree with.

This starts upstream's own `PolicyServer` wrapping the real checkpoint in a subprocess,
then talks to it with the bridge's torch-free `PolicyClient`. Slow (~40s: model load plus
CPU inference) and skipped unless the checkpoint and demo_data are present.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest \
        grootN1_Robotics/tests/test_ros_bridge_against_real_policy.py -q
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
sys.path.insert(0, str(ROOT / "grootN1_Robotics/tools"))

CHECKPOINT = ROOT / "grootN1_Robotics/checkpoints/GR00T-N1.6-3B"
DATASET = ROOT / "grootN1_Robotics/upstream/demo_data/gr1.PickNPlace"
PORT = 5599  # off the default so a stray real server is not mistaken for ours


def lfs_fetched() -> bool:
    parquet = next(DATASET.glob("data/**/*.parquet"), None)
    return parquet is not None and parquet.stat().st_size > 1000


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists() or not DATASET.exists() or not lfs_fetched(),
    reason="needs the N1.6 checkpoint and fetched demo_data git-lfs objects",
)


@pytest.fixture(scope="module")
def server():
    """Real PolicyServer in a subprocess; torn down even if a test fails."""
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'grootN1_Robotics/upstream'}:{ROOT}")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "grootN1_Robotics/tools/serve_policy.py"),
         "--checkpoint", str(CHECKPOINT), "--port", str(PORT)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # Wait for the bind line rather than sleeping a guessed interval.
        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server died during startup:\n{proc.stdout.read()[-2000:]}")
            line = proc.stdout.readline()
            if "listening on" in line or "ready and listening" in line:
                break
        else:
            pytest.fail("server did not start within 300s")
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

    return PolicyClient("localhost", PORT, timeout_ms=300_000)


def observation():
    """Build the same observation the CPU tool builds, then make it wire-safe.

    The bridge sends plain numpy/str over msgpack; anything the policy needs that does
    not survive that round trip is a bug we want to see here rather than on a robot.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    from grootN1_Robotics.policy import LocalGr00tPolicy
    from run_policy_cpu import build_observation

    # A throwaway local policy is only used to learn the modality layout.
    policy = LocalGr00tPolicy(EmbodimentTag.GR1, CHECKPOINT)
    obs = build_observation(DATASET, policy)
    del policy
    return obs


def test_ping_reaches_the_real_server(client):
    assert client.ping() is True


def test_get_action_returns_the_declared_joint_groups(client):
    action = client.get_action(observation())

    assert isinstance(action, dict)
    expected = {"left_arm", "right_arm", "left_hand", "right_hand", "waist"}
    assert expected.issubset(set(action)), f"missing joint groups: {expected - set(action)}"


def test_action_survives_the_msgpack_round_trip_intact(client):
    """Shapes and dtypes must come back as the node expects to index them.

    msgpack has no native ndarray, so the serializer encodes shape/dtype by hand; a
    silent flatten here would leave the node slicing a 1-D buffer as if it were (T, D).
    """
    action = client.get_action(observation())

    for key in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
        arr = np.asarray(action[key])
        assert arr.ndim == 3, f"{key} arrived with ndim={arr.ndim}, expected (B, T, D)"
        assert arr.shape[1] == 16, f"{key} horizon {arr.shape[1]}, expected 16"
        assert np.isfinite(arr).all(), f"{key} contains non-finite values"

    widths = {k: np.asarray(action[k]).shape[-1] for k in
              ("left_arm", "right_arm", "left_hand", "right_hand", "waist")}
    assert widths == {"left_arm": 7, "right_arm": 7, "left_hand": 6,
                      "right_hand": 6, "waist": 3}


def test_modality_config_endpoint_agrees_with_the_action(client):
    """The node sizes its buffers from get_modality_config before any action arrives.

    If that disagrees with what get_action actually returns, the node allocates the wrong
    width and the mismatch only shows up mid-episode.
    """
    config = client.get_modality_config()
    if not config or "action" not in config:
        pytest.skip("server did not expose an action modality config")

    action_cfg = config["action"]
    keys = getattr(action_cfg, "modality_keys", None) or action_cfg["modality_keys"]
    action = client.get_action(observation())
    assert set(keys) == set(action) & set(keys)

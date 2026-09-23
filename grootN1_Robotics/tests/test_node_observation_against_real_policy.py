"""The ROS node's observation, validated by the REAL policy — not by a fake server.

`ros2_bridge/.../test_node_endtoend.py` drives the node against a hand-written fake, and
that fake never looks at the observation it receives. So the node spent its whole life
sending a payload no real server would accept: a FLAT dict with a single leading axis,
where `Gr00tPolicy.check_observation` requires a NESTED one with both a batch and a time
axis. See `ros2_bridge/bug_log.txt` [3].

This closes that hole the cheapest way there is. `check_observation` is the exact gate a
live episode hits first, it is a pure-numpy method, and it needs no inference — so the
authoritative check costs a checkpoint load and no forward pass at all.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest \
        grootN1_Robotics/tests/test_node_observation_against_real_policy.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2_bridge/vla_bridge"))

CHECKPOINT = ROOT / "grootN1_Robotics/checkpoints/GR00T-N1.6-3B"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(),
    reason="needs the GR00T N1.6 checkpoint",
)


@pytest.fixture(scope="module")
def policy():
    from gr00t.data.embodiment_tags import EmbodimentTag

    from grootN1_Robotics.policy import LocalGr00tPolicy

    return LocalGr00tPolicy(EmbodimentTag.GR1, CHECKPOINT)


# GR1's state is split across five keys whose widths are NOT on the wire — see
# statistics.json in the checkpoint. These are the values a launch file must supply.
GR1_STATE_DIMS = {"left_arm": 7, "left_hand": 6, "right_arm": 7, "right_hand": 6,
                  "waist": 3}


@pytest.fixture(scope="module")
def layout(policy):
    """Negotiate the layout exactly as the node does at startup.

    The node calls `get_modality_config` over ZMQ; here the same dict comes from the
    policy directly, which is what the server would have serialised.
    """
    from vla_bridge.obs_encode import parse_layout, reconcile

    config = policy.get_modality_config()
    declared = parse_layout(config)
    assert declared is not None, "GR00T declares video/state/language; it must parse"

    # Feed reconcile the prefixed spelling a user would actually put in a launch file.
    video_params = [f"video.{k}" for k in declared.video_keys]
    state_params = [f"state.{k}" for k in declared.state_keys]
    dims = [GR1_STATE_DIMS[k] for k in declared.state_keys]
    return reconcile(video_params, state_params, dims, declared)


def node_observation(layout) -> dict:
    """What the node would send, built from synthetic camera and joint data."""
    from vla_bridge.obs_encode import build_observation

    frames = {key: np.zeros((224, 224, 3), dtype=np.uint8) for key in layout.video_keys}
    state = np.zeros(sum(layout.state_dims), dtype=np.float32)
    return build_observation(frames, state, "pick up the red block", layout)


def test_gr1_state_really_is_split_across_five_keys(policy):
    """Pinned because it is why `state_dims` exists at all.

    A node with a single `state_modality_key` cannot serve this embodiment, and nothing
    on the wire says so — `get_modality_config` carries names, never widths.
    """
    assert set(policy.modality_configs["state"].modality_keys) == set(GR1_STATE_DIMS)
    assert sum(GR1_STATE_DIMS.values()) == 29


def test_the_node_negotiates_the_real_embodiments_keys(policy, layout):
    """Startup must resolve to the checkpoint's own key names, not our defaults."""
    declared = policy.modality_configs
    assert set(layout.video_keys) == set(declared["video"].modality_keys)
    assert layout.language_key == policy.language_key


def test_real_check_observation_accepts_the_nodes_payload(policy, layout):
    """The gate a live episode hits first. This is the whole point of the file."""
    policy.check_observation(node_observation(layout))


def test_the_old_flat_payload_is_rejected_by_the_real_policy(policy, layout):
    """Proof the bug was real, not a tidiness complaint.

    This is verbatim what the node used to build. If a future refactor reverts to it,
    this test fails with the same error a robot would have produced.
    """
    frames = {f"video.{k}": np.zeros((224, 224, 3), np.uint8) for k in layout.video_keys}
    flat = {k: v[None, ...] for k, v in frames.items()}
    for key, width in zip(layout.state_keys, layout.state_dims):
        flat[f"state.{key}"] = np.zeros((1, width), np.float32)
    flat["annotation.human.task_description"] = ["pick up the red block"]

    with pytest.raises((KeyError, AssertionError, TypeError)):
        policy.check_observation(flat)


def test_a_missing_time_axis_is_rejected(policy, layout):
    """The specific rank error the old payload had, isolated.

    (1, H, W, C) is a plausible-looking batch of one image and passes every shape check a
    human eye applies; upstream asserts ndim == 5 precisely because it is wrong.
    """
    obs = node_observation(layout)
    key = layout.video_keys[0]
    obs["video"][key] = obs["video"][key][0]  # drop the batch axis -> ndim 4

    with pytest.raises(AssertionError, match="ndim|shape|B, T"):
        policy.check_observation(obs)

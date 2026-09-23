"""The bridge must handle BOTH action-key conventions.

Servers disagree on naming, and the disagreement is silent:

    GR00T raw Gr00tPolicy      -> BARE keys      "left_arm", "right_arm", "waist"
    GR00T sim wrapper / DiVLA  -> PREFIXED keys  "action.x", "action.joints"

`_extract_chunk` originally required the `action.` prefix, so every chunk from a raw
GR00T server would have been dropped with "no action.* keys in reply". The existing
end-to-end test did not catch it: its fake server was written to send `{"action.arm": ...}`,
and a fake agrees with whatever you wrote it to agree with. That is the second protocol
mismatch a fake has hidden here — see ../../bug_log.txt [1].

The decoding now lives in `action_decode.py`, free of rclpy, precisely so it can be
tested outside a ROS container.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge.action_decode import (  # noqa: E402
    ActionDecodeError,
    declared_action_keys,
    decode_chunk,
)

HORIZON = 4


def chunk(value: float, dof: int = 1) -> np.ndarray:
    return np.full((HORIZON, dof), value, dtype=np.float32)


# ------------------------------------------------------------------ conventions


def test_prefixed_keys_work_without_discovery():
    """The original convention keeps working when the server declares nothing."""
    got = decode_chunk({"action.x": chunk(1), "action.y": chunk(2)}, declared=None)

    assert got.shape == (HORIZON, 2)
    assert np.allclose(got[:, 0], 1) and np.allclose(got[:, 1], 2)


def test_bare_keys_work_when_the_server_declares_them():
    """The raw-GR00T case, which used to be dropped entirely."""
    got = decode_chunk(
        {"left_arm": chunk(1), "right_arm": chunk(2), "waist": chunk(3)},
        declared=["left_arm", "right_arm", "waist"],
    )
    assert got.shape == (HORIZON, 3)


def test_bare_keys_without_declaration_fail_loudly():
    """No declaration and no prefix: refuse, and say why.

    Silently publishing nothing is how a robot sits still while the logs look clean.
    """
    with pytest.raises(ActionDecodeError, match="no usable action keys"):
        decode_chunk({"left_arm": chunk(1)}, declared=None)


# ------------------------------------------------------------------ ordering


def test_columns_follow_sorted_key_order():
    """Column order maps positionally onto joint_names; a reordering permutes joints."""
    got = decode_chunk({"waist": chunk(9), "left_arm": chunk(1)},
                       declared=["waist", "left_arm"])  # deliberately unsorted input

    assert np.allclose(got[:, 0], 1), "columns not in sorted key order"
    assert np.allclose(got[:, 1], 9)


def test_declared_keys_are_sorted_by_the_helper():
    cfg = {"action": {"modality_keys": ["waist", "left_arm"]}}
    assert declared_action_keys(cfg) == ["left_arm", "waist"]


def test_declared_action_keys_handles_absent_config():
    assert declared_action_keys(None) is None
    assert declared_action_keys({}) is None
    assert declared_action_keys({"action": {}}) is None


# ------------------------------------------------------------------ refusals


def test_partial_reply_is_refused_not_padded():
    """Three declared, two returned: a narrower chunk would shift every joint."""
    with pytest.raises(ActionDecodeError, match="partial chunk"):
        decode_chunk({"a": chunk(1), "b": chunk(2)}, declared=["a", "b", "c"])


def test_non_finite_actions_are_refused():
    bad = chunk(1)
    bad[2, 0] = np.nan
    with pytest.raises(ActionDecodeError, match="non-finite"):
        decode_chunk({"a": bad}, declared=["a"])


def test_inconsistent_horizons_are_refused():
    """Concatenating mismatched horizons would either raise obscurely or truncate."""
    with pytest.raises(ActionDecodeError, match="inconsistent horizons"):
        decode_chunk({"a": chunk(1), "b": np.zeros((HORIZON + 2, 1), np.float32)},
                     declared=["a", "b"])


def test_extra_keys_are_ignored_when_declared():
    """DiVLA returns `reasoning` alongside its action; declared keys make that safe.

    Prefix-guessing or 'take every array-like value' would not.
    """
    got = decode_chunk(
        {"action.joints": chunk(1, dof=7), "reasoning": "pick up the cube",
         "latency_ms": 12.0},
        declared=["action.joints"],
    )
    assert got.shape == (HORIZON, 7)


# ------------------------------------------------------------------ shapes


def test_leading_batch_axis_is_dropped():
    """Servers return (1, H, D); the node wants (H, D)."""
    got = decode_chunk({"a": np.ones((1, HORIZON, 3), np.float32)}, declared=["a"])
    assert got.shape == (HORIZON, 3)


def test_one_dimensional_action_is_reshaped():
    """Some servers return (H,) for a scalar joint."""
    got = decode_chunk(
        {"a": np.arange(HORIZON, dtype=np.float32),
         "b": np.arange(HORIZON, dtype=np.float32) * 2},
        declared=["a", "b"],
    )
    assert got.shape == (HORIZON, 2)


def test_unexpected_rank_is_refused():
    with pytest.raises(ActionDecodeError, match="expected"):
        decode_chunk({"a": np.ones((2, HORIZON, 3, 4), np.float32)}, declared=["a"])


def test_real_gr00t_reply_shape_decodes():
    """The exact shape observed from the real N1.6 server: (1, 16, D) per joint group."""
    reply = {"left_arm": np.zeros((1, 16, 7), np.float32),
             "right_arm": np.zeros((1, 16, 7), np.float32),
             "left_hand": np.zeros((1, 16, 6), np.float32),
             "right_hand": np.zeros((1, 16, 6), np.float32),
             "waist": np.zeros((1, 16, 3), np.float32)}

    got = decode_chunk(reply, declared=sorted(reply))

    assert got.shape == (16, 29), "GR1 uses 29 of the 128 padded action dims"

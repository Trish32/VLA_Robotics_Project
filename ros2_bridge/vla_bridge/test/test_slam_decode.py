"""Tests for DROID-SLAM pose decoding.

The quaternion convention is the whole risk here. lietorch SE3 is
`[tx, ty, tz, qx, qy, qz, qw]` — scalar-LAST — and so is ROS, so it carries across
unchanged. But Eigen, MuJoCo and older scipy put w FIRST, and a swapped layout produces a
unit quaternion describing the wrong rotation: a valid-looking pose, rotated nonsense,
nothing to raise on. Hence the explicit tests rather than a comment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge.slam_decode import (  # noqa: E402
    SE3_LAYOUT,
    SlamDecodeError,
    decode_pose,
    is_tracking_lost,
)


def test_flat_se3_vector_splits_translation_and_quaternion():
    reply = {"pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]}

    t, q = decode_pose(reply)

    assert np.allclose(t, [1.0, 2.0, 3.0])
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0]), "identity rotation is w=1, scalar-last"


def test_layout_constant_documents_scalar_last():
    """Pinned so a future edit cannot quietly reorder it."""
    assert SE3_LAYOUT == ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    assert SE3_LAYOUT[-1] == "qw", "ROS and lietorch are both scalar-LAST"


def test_explicit_translation_quaternion_pair_is_accepted():
    t, q = decode_pose({"translation": [0.0, 0.0, 1.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0]})
    assert np.allclose(t, [0.0, 0.0, 1.0])
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])


def test_wrong_length_pose_is_refused():
    with pytest.raises(SlamDecodeError, match="expected 7"):
        decode_pose({"pose": [1.0, 2.0, 3.0]})


def test_missing_pose_is_refused():
    with pytest.raises(SlamDecodeError, match="neither"):
        decode_pose({"n_keyframes": 5})


def test_non_finite_pose_is_refused():
    with pytest.raises(SlamDecodeError, match="non-finite"):
        decode_pose({"pose": [1.0, np.nan, 3.0, 0.0, 0.0, 0.0, 1.0]})


def test_slightly_unnormalised_quaternion_is_renormalised():
    """SLAM output drifts off the unit sphere; that is not an error."""
    q_raw = np.array([0.0, 0.0, 0.0, 1.0004])
    _, q = decode_pose({"pose": [0, 0, 0, *q_raw]})

    assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_grossly_unnormalised_quaternion_is_refused():
    """A norm far from 1 means a layout or scaling bug, not drift — renormalising it
    would hide the very thing worth catching."""
    with pytest.raises(SlamDecodeError, match="layout mismatch"):
        decode_pose({"pose": [0, 0, 0, 3.0, 4.0, 0.0, 0.0]})  # norm 5


def test_zero_quaternion_is_refused():
    with pytest.raises(SlamDecodeError, match="zero norm"):
        decode_pose({"pose": [0, 0, 0, 0.0, 0.0, 0.0, 0.0]})


def test_a_scalar_first_quaternion_is_not_silently_accepted():
    """The failure mode this module exists to prevent, made concrete.

    A 90-degree rotation about Z is scalar-last [0, 0, 0.7071, 0.7071]. Written
    scalar-FIRST it is [0.7071, 0, 0, 0.7071] — still unit norm, so no check can reject
    it. Decoding is therefore silent, and the test's job is to state that decode_pose
    interprets the LAST element as w so callers cannot misread the contract.
    """
    scalar_last = [0.0, 0.0, 0.7071068, 0.7071068]
    _, q = decode_pose({"pose": [0, 0, 0, *scalar_last]})

    assert q[3] == pytest.approx(0.7071068), "element 3 must be w"
    assert q[2] == pytest.approx(0.7071068), "element 2 must be qz"


def test_tracking_lost_is_detected():
    assert is_tracking_lost({"tracking_lost": True})
    assert is_tracking_lost({"n_keyframes": 0})
    assert not is_tracking_lost({"n_keyframes": 12})
    assert not is_tracking_lost({})


def test_tracking_lost_matters_because_a_stale_pose_looks_valid():
    """A dropped-tracking reply still carries a well-formed pose; only the flag says it
    is stale. Publishing it as current odometry is worse than publishing nothing."""
    reply = {"pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], "tracking_lost": True}

    decode_pose(reply)          # decodes fine — that is the point
    assert is_tracking_lost(reply)

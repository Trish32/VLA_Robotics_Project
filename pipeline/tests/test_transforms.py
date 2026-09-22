"""Rotation round-trips, including the regime the naive formula gets wrong."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.transforms import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402


def rot(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.deg2rad(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


@pytest.mark.parametrize("axis,deg", [
    ((0, 0, 1), 0.0), ((0, 0, 1), 45.0), ((1, 0, 0), 90.0), ((0, 1, 0), 120.0),
    ((1, 1, 0), 179.0),      # the trace-form branch underflows near here
    ((1, 0, 0), 180.0),      # and fails outright at exactly 180
    ((0, 1, 0), 180.0),
    ((0, 0, 1), 180.0),
    ((1, 2, 3), 179.9),
])
def test_matrix_quaternion_round_trip(axis, deg):
    R = rot(axis, deg)
    q = matrix_to_quaternion(R)

    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9), f"not unit norm: {q}"
    assert np.allclose(quaternion_to_matrix(*q), R, atol=1e-9)


def test_a_180_degree_rotation_does_not_collapse():
    """The case that motivates Shepperd's method. `sqrt(1 + trace)` is sqrt(0) here, and
    dividing by it yields inf or NaN — a pose that is wrong rather than absent."""
    q = matrix_to_quaternion(rot((1, 0, 0), 180.0))
    assert np.all(np.isfinite(q))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


def test_identity_is_scalar_last_w_equals_one():
    """A scalar-FIRST reading of this is (1,0,0,0) -> also unit norm, also 'valid', and
    a 180 degree rotation. The convention has to be pinned, not inferred."""
    assert matrix_to_quaternion(np.eye(3)) == pytest.approx((0.0, 0.0, 0.0, 1.0))

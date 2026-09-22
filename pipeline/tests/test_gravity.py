"""Gravity alignment: the frame the whole world model is expressed in.

Built from known rotations so the expected answer is exact. The sign tests carry the
weight here — a flipped `up` rotates the world upside down and every downstream check
still passes, because `on` simply never fires, which is indistinguishable from the
untilted bug this module exists to fix.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.gravity import (  # noqa: E402
    GravityError,
    align_to_z_up,
    estimate_up,
    rotation_to_z_up,
)


def tabletop(up, *, n=4000, seed=0, span=1.5):
    """A planar support with its normal along `up`, plus cameras above it."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up, float) / np.linalg.norm(up)
    basis = np.linalg.svd(up.reshape(1, 3))[2][1:]       # two axes spanning the plane
    uv = rng.uniform(-span, span, (n, 2))
    points = uv @ basis + rng.normal(0, 0.002, (n, 1)) * up
    cameras = np.tile(np.eye(4), (10, 1, 1))
    cameras[:, :3, 3] = up * rng.uniform(0.8, 1.5, (10, 1))
    return points, cameras


# ------------------------------------------------------------------ rotation

@pytest.mark.parametrize("up", [
    [0, 0, 1], [0, 0, -1], [0, 1, 0], [1, 0, 0],
    [0.0, 0.689, -0.722], [-0.008, -0.767, -0.642],     # the real TUM plane normal
])
def test_rotation_puts_up_on_plus_z(up):
    R = rotation_to_z_up(np.array(up, float))
    assert np.allclose(R @ (np.array(up, float) / np.linalg.norm(up)), [0, 0, 1], atol=1e-9)


@pytest.mark.parametrize("up", [[0, 0, 1], [0, 0, -1], [1, 2, 3], [-0.008, -0.767, -0.642]])
def test_rotation_is_proper_not_a_reflection(up):
    """det = -1 fits the plane just as well and mirrors the scene."""
    R = rotation_to_z_up(np.array(up, float))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_a_zero_up_vector_is_refused():
    with pytest.raises(GravityError, match="direction"):
        rotation_to_z_up(np.zeros(3))


# ------------------------------------------------------------------ estimate

@pytest.mark.parametrize("truth", [
    [0, 0, 1], [0, 1, 0], [0.0, 0.6, -0.8], [-0.008, -0.767, -0.642],
])
def test_up_is_recovered_from_a_planar_support(truth):
    points, cameras = tabletop(truth)
    up, fraction = estimate_up(points, camera_positions=cameras[:, :3, 3])
    assert np.allclose(up, np.array(truth, float) / np.linalg.norm(truth), atol=1e-3)
    assert fraction > 0.9


def test_cameras_disambiguate_the_sign():
    """The failure this module is most likely to have, so it is tested directly."""
    truth = np.array([0.0, 0.6, -0.8])
    points, cameras = tabletop(truth)

    above, _ = estimate_up(points, camera_positions=cameras[:, :3, 3])
    below, _ = estimate_up(points, camera_positions=-cameras[:, :3, 3])
    assert np.allclose(above, truth, atol=1e-3)
    assert np.allclose(below, -truth, atol=1e-3), "sign must follow the cameras"


def test_a_weak_plane_is_refused_rather_than_guessed():
    """A wrong gravity looks correct and changes every relation and every grasp."""
    rng = np.random.default_rng(0)
    with pytest.raises(GravityError, match="Refusing to align"):
        estimate_up(rng.normal(0, 1.0, (3000, 3)), min_inlier_fraction=0.5)


@pytest.mark.parametrize("bad,match", [
    (np.zeros((2, 3)), "at least 3 points"),
    (np.zeros((10, 2)), r"\(N, 3\)"),
])
def test_malformed_input_is_refused(bad, match):
    with pytest.raises(GravityError, match=match):
        estimate_up(bad)


# --------------------------------------------------------------------- align

def test_cloud_and_trajectory_take_the_same_rotation():
    """Rotating one without the other decouples the map from the poses that built it."""
    truth = np.array([0.0, 0.6, -0.8])
    points, cameras = tabletop(truth)

    rotated, poses, T, _ = align_to_z_up(points, cameras)
    assert np.allclose(rotated[:, 2], rotated[:, 2].mean(), atol=0.02), "plane not level"
    # Cameras were placed along +up, so after alignment they sit above z=0.
    assert (poses[:, 2, 3] > 0).all()
    assert np.allclose(poses[:, :3, :3], T[:3, :3] @ cameras[:, :3, :3])


def test_relative_geometry_is_preserved():
    """Alignment is a rotation: it may not change any distance or angle."""
    points, cameras = tabletop([0.0, 0.6, -0.8])
    rotated, _, _, _ = align_to_z_up(points, cameras)

    a = points[:200] - points[200:400]
    b = rotated[:200] - rotated[200:400]
    assert np.allclose(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), atol=1e-9)


def test_pose_orientations_rotate_too_not_just_positions():
    """An object's orientation lives in the world frame as much as its position."""
    points, cameras = tabletop([0.0, 0.6, -0.8])
    angle = 0.7
    spin = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    cameras[:, :3, :3] = spin

    _, poses, T, _ = align_to_z_up(points, cameras)
    assert not np.allclose(poses[:, :3, :3], spin), "orientation was left behind"
    assert np.allclose(poses[0, :3, :3], T[:3, :3] @ spin)

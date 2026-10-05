"""The mustard0 mesh control builds a mesh the way the chair's was built.

Tiny synthetic inputs: a box rendered from known poses, fused, thinned and meshed with
the REAL `reject_outliers` and `poisson_mesh`. Nothing here needs CUDA; the job that
registers against these meshes does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from foundationpose_6dof import mesh_control as mc  # noqa: E402

K = np.array([[300.0, 0, 80.0], [0, 300.0, 60.0], [0, 0, 1.0]])
H, W = 120, 160


def _pose(yaw_deg, dist=0.6):
    """ob_in_cam: the box at `dist` in front of the camera, turned by `yaw_deg`."""
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0, 0, dist]
    return T


def _render(ob_in_cam, size=(0.10, 0.07, 0.19)):
    o3d = pytest.importorskip("open3d")
    box = o3d.geometry.TriangleMesh.create_box(*size)
    box.translate(-np.asarray(size) / 2)
    box.transform(ob_in_cam)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(box))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K), o3d.core.Tensor(np.eye(4)), W, H)
    t = scene.cast_rays(rays)["t_hit"].numpy()
    # Pinhole rays have unit z-component, not unit length: depth is t * d_z, and
    # dividing by |d| (as this helper first did) under-reads depth by up to 5%.
    z = t * rays.numpy()[..., 5]
    hit = np.isfinite(z)
    depth = np.where(hit, z * 1000, 0).astype(np.uint16)
    rgb = np.full((H, W, 3), 128, np.uint8)
    return rgb, depth, hit


def _frames(yaws):
    out = []
    for y in yaws:
        P = _pose(y)
        rgb, depth, mask = _render(P)
        out.append((rgb, depth, mask, P))
    return out


def test_relative_scaling_matches_the_chair_at_the_chair():
    assert mc.relative(mc.CHAIR_VOXEL_M, mc.CHAIR_LONGEST_M) == pytest.approx(0.02)
    assert mc.relative(mc.CHAIR_VOXEL_M, 0.19) == pytest.approx(0.02 * 0.19 / 1.58)


def test_view_angles_are_relative_to_frame_zero():
    a = mc.view_angles([_pose(0), _pose(5), _pose(-30)])
    assert a == pytest.approx([0.0, 5.0, 30.0], abs=1e-6)


def test_flip_tolerant_error_forgives_only_the_named_flip():
    R = _pose(0)[:3, :3]
    flip_z = np.diag([-1.0, -1.0, 1.0])
    assert mc.rotation_deg(R @ flip_z, R) == pytest.approx(180.0)
    assert mc.rotation_deg_mod_flip(R @ flip_z, R, axis=2) == pytest.approx(0.0, abs=1e-6)
    flip_x = np.diag([1.0, -1.0, -1.0])
    assert mc.rotation_deg_mod_flip(R @ flip_x, R, axis=2) == pytest.approx(180.0)


def test_fused_points_land_on_the_object_in_the_object_frame():
    """integrate() takes world-to-camera; with the object as world that is ob_in_cam.

    Passing the inverse fuses a mirrored object that still looks plausible, so the test
    checks the points lie on the box surface, not merely that some were produced.
    """
    pytest.importorskip("open3d")
    voxel = mc.relative(mc.CHAIR_VOXEL_M, 0.19)
    pts, _ = mc.fuse_object(_frames([-20, 0, 20]), K, voxel, np.zeros(3), 0.15)
    assert len(pts) > 200
    half = np.array([0.05, 0.035, 0.095])
    # every point within a voxel or two of the box surface
    outside = np.abs(pts) - half
    assert (outside.max(axis=1) < 3 * voxel).all()
    # and on the surface, not filling the interior
    assert (outside.max(axis=1) > -3 * voxel).all()


def test_one_sided_fusion_sees_only_one_side():
    """The chair was seen from ~6 deg of view angle; so is the one-sided arm."""
    pytest.importorskip("open3d")
    voxel = mc.relative(mc.CHAIR_VOXEL_M, 0.19)
    pts, _ = mc.fuse_object(_frames([0, 3]), K, voxel, np.zeros(3), 0.15)
    # camera on -z of the object looking +z: the far face (z = +0.095) is never seen
    assert (pts[:, 2] > 0.08).sum() == 0
    assert (pts[:, 2] < -0.08).sum() > 50


def test_thinning_hits_the_chair_budget():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.1, 0.1, (20000, 3))
    out, cols = mc.thin_to(pts, np.zeros_like(pts), 1128)
    assert abs(len(out) - 1128) <= 0.1 * 1128
    assert len(cols) == len(out)


def test_thinning_never_adds_points():
    pts = np.zeros((10, 3))
    out, _ = mc.thin_to(pts, pts, 1128)
    assert len(out) == 10


def test_ourway_mesh_uses_the_pipeline_functions_it_is_given():
    """Outlier rejection and Poisson are not re-implemented — they are passed in."""
    pytest.importorskip("open3d")
    from pipeline.observations import reject_outliers
    from pipeline.tools.build_pose_bundle import poisson_mesh

    calls = []

    def spy_reject(p):
        calls.append("reject")
        return reject_outliers(p)

    def spy_poisson(pcd, depth):
        calls.append(("poisson", depth))
        return poisson_mesh(pcd, depth=depth)

    voxel = mc.relative(mc.CHAIR_VOXEL_M, 0.19)
    pts, cols = mc.fuse_object(_frames(range(-40, 41, 10)), K, voxel, np.zeros(3), 0.15)
    v, f, vc, centroid, n = mc.ourway_mesh(pts, cols, 0.19, spy_reject, spy_poisson)
    assert calls == ["reject", ("poisson", 7)]
    assert len(f) > 0 and len(vc) == len(v)
    assert n <= 1.1 * mc.CHAIR_POINTS
    # vertices are centroid-relative, as build_pose_bundle makes the chair's
    assert np.abs(v.mean(axis=0)).max() < 0.05
    assert np.abs(centroid).max() < 0.1


def test_view_span_is_the_largest_pairwise_angle():
    assert mc.view_span([_pose(0), _pose(10), _pose(-21)]) == pytest.approx(31.0, abs=1e-6)
    assert mc.view_span([_pose(5)]) == 0.0


def _box_mesh(size=(0.10, 0.07, 0.19)):
    """A centred box as (vertices, faces) — a stand-in for a CAD model."""
    o3d = pytest.importorskip("open3d")
    box = o3d.geometry.TriangleMesh.create_box(*size)
    box.translate(-np.asarray(size) / 2)
    return np.asarray(box.vertices), np.asarray(box.triangles)


def test_track_depth_residual_is_small_at_the_true_pose():
    P = _pose(15)
    _, depth, _ = _render(P)
    r = mc.track_depth_residual(*_box_mesh(), P, K, depth / 1000.0)
    assert r < 0.005


def test_track_depth_residual_catches_a_wrong_pose():
    """A tracked pose 4 cm off in depth — the failure the void test exists to catch."""
    P = _pose(15)
    _, depth, _ = _render(P)
    wrong = P.copy()
    wrong[2, 3] += 0.04
    assert mc.track_depth_residual(*_box_mesh(), wrong, K, depth / 1000.0) > 0.03


def test_track_depth_residual_is_inf_when_nothing_overlaps():
    pytest.importorskip("open3d")
    P = _pose(0)
    P[0, 3] = 5.0
    assert mc.track_depth_residual(*_box_mesh(), P, K, np.ones((H, W))) == float("inf")

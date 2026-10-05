"""Frame selection and mask construction for the FoundationPose input bundle.

These cover the three things that were wrong in the bundle the 175.63 deg rotation
error was measured on, each of which is a silent failure: the builder reported a mask
size and a frame count, and every one of those numbers was true of a bundle showing
roughly a tenth of the object.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.tools.build_pose_bundle import dense_mask, project_instance, select_frames

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def _slab(z=2.0, shape=(480, 640)):
    """A depth image of a fronto-parallel wall at `z`."""
    return np.full(shape, z, np.float32)


def _grid(n=20, z=2.0, half=0.2):
    """A square patch of points on the plane z, in camera coordinates."""
    a = np.linspace(-half, half, n)
    x, y = np.meshgrid(a, a)
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1)


# --------------------------------------------------------------- corroboration
def test_points_on_the_measured_surface_are_corroborated():
    pts = _grid()
    *_, ok, resid = project_instance(pts, np.eye(4), K, _slab(2.0), 0.08)
    assert ok.all()
    assert np.allclose(np.nanmedian(resid), 0.0, atol=1e-6)


def test_the_test_is_two_sided_and_the_sign_says_which_failure():
    """A point behind the surface is occluded; one in front is misplaced.

    The builder used to call this an occlusion test and report only a pass count, which
    hid the distinction. On the real bundle the median residual was -52 cm — the
    instance sitting IN FRONT of the measured surface, which is not occlusion at all.
    """
    wall = _slab(2.0)
    behind = _grid(z=2.5)
    front = _grid(z=1.5)
    *_, ok_b, resid_b = project_instance(behind, np.eye(4), K, wall, 0.08)
    *_, ok_f, resid_f = project_instance(front, np.eye(4), K, wall, 0.08)
    assert not ok_b.any() and not ok_f.any()
    assert np.nanmedian(resid_b) > 0          # occluded
    assert np.nanmedian(resid_f) < 0          # misplaced


def test_points_outside_the_image_are_not_corroborated():
    pts = _grid() + np.array([5.0, 0.0, 0.0])
    *_, ok, _ = project_instance(pts, np.eye(4), K, _slab(), 0.08)
    assert not ok.any()


def test_points_behind_the_camera_do_not_wrap_into_the_image():
    pts = _grid(z=-2.0)
    *_, ok, _ = project_instance(pts, np.eye(4), K, _slab(), 0.08)
    assert not ok.any()


# ------------------------------------------------------------------ dense mask
def test_dense_fill_covers_the_interior_a_splat_leaves_empty():
    """The point cloud is voxel-sparse, so a splat covers a few percent of the area."""
    pts = _grid(n=12, half=0.2)                      # 144 points over ~100x100 px
    ui, vi, z, ok, _ = project_instance(pts, np.eye(4), K, _slab(2.0), 0.08)
    mask = dense_mask(ui, vi, z, ok, _slab(2.0), 0.08)
    ys, xs = np.nonzero(mask)
    box = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    assert (mask > 0).sum() > 10 * ok.sum()
    assert (mask > 0).sum() / box > 0.9              # the interior is filled


def test_dense_fill_does_not_cross_a_depth_discontinuity():
    """Filling is bounded by the depth image, not by a structuring element.

    Half the wall is 60 cm nearer. The fill may claim the half the instance is on and
    must not claim the other, however close the two are in the image.
    """
    depth = _slab(2.0)
    depth[:, 320:] = 1.4
    pts = _grid(n=12, half=0.2)                      # centred, so it straddles x=320
    pts[:, 0] -= 0.25                                # push it entirely onto the far half
    ui, vi, z, ok, _ = project_instance(pts, np.eye(4), K, depth, 0.08)
    mask = dense_mask(ui, vi, z, ok, depth, 0.08)
    assert (mask[:, :320] > 0).sum() > 0
    assert (mask[:, 320:] > 0).sum() == 0


def test_dense_fill_keeps_one_region():
    """A surface elsewhere at the same range is not the instance."""
    depth = _slab(2.0)
    pts = _grid(n=12, half=0.1)
    ui, vi, z, ok, _ = project_instance(pts, np.eye(4), K, depth, 0.08)
    mask = dense_mask(ui, vi, z, ok, depth, 0.08)
    import cv2
    count, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    assert count == 2                                 # background + exactly one region


def test_no_corroborated_points_gives_an_empty_mask():
    pts = _grid(z=1.0)
    ui, vi, z, ok, _ = project_instance(pts, np.eye(4), K, _slab(2.0), 0.08)
    assert not ok.any()
    assert (dense_mask(ui, vi, z, ok, _slab(2.0), 0.08) > 0).sum() == 0


# -------------------------------------------------------------- frame selection
def _scored(fractions, centres=None):
    n = len(fractions)
    centres = centres if centres is not None else [[0.1 * i, 0, 0] for i in range(n)]
    return [{"index": i, "fraction": f, "visible": int(1000 * f), "centre": c}
            for i, (f, c) in enumerate(zip(fractions, centres))]


def test_the_anchor_is_the_best_view_not_the_earliest():
    """`register()` sees only frame 0, so frame 0 must be the best frame available.

    The rule this replaces took frames in arrival order behind a 50-point floor, and on
    the real sequence that chose eight frames at 11-17% visible when 73% was available.
    """
    chosen = select_frames(_scored([0.4, 0.4, 0.9, 0.5, 0.5]), want=3,
                           min_fraction=0.35, min_step=0.05, max_step=0.5)
    assert chosen[0]["index"] == 2


def test_frames_below_the_floor_are_never_selected():
    chosen = select_frames(_scored([0.9, 0.01, 0.02, 0.8, 0.85]), want=5,
                           min_fraction=0.35, min_step=0.05, max_step=0.5)
    assert [c["index"] for c in chosen] == [0, 3, 4]


def test_nothing_above_the_floor_returns_nothing_rather_than_the_best_bad_frame():
    assert select_frames(_scored([0.1, 0.2, 0.05]), want=4, min_fraction=0.35) == []


def test_near_duplicate_views_are_skipped():
    """Eight consecutive frames of a slow camera are one viewpoint recorded eight times."""
    centres = [[0.001 * i, 0, 0] for i in range(6)] + [[0.5, 0, 0]]
    chosen = select_frames(_scored([0.9] * 7, centres), want=3,
                           min_fraction=0.35, min_step=0.05, max_step=1.0)
    assert [c["index"] for c in chosen] == [0, 6]


def test_the_chain_never_jumps_further_than_a_tracker_can_follow():
    """`track_one` refines from the previous frame; a metre-long jump is not tracking."""
    centres = [[0, 0, 0], [0.08, 0, 0], [5.0, 0, 0], [5.08, 0, 0]]
    chosen = select_frames(_scored([0.9, 0.9, 0.95, 0.9], centres), want=4,
                           min_fraction=0.35, min_step=0.03, max_step=0.15)
    steps = [np.linalg.norm(np.array(b["centre"]) - np.array(a["centre"]))
             for a, b in zip(chosen, chosen[1:])]
    assert all(s <= 0.15 for s in steps)


def test_a_short_chain_is_returned_rather_than_nothing():
    centres = [[0, 0, 0], [0.05, 0, 0], [9.0, 0, 0]]
    chosen = select_frames(_scored([0.9, 0.9, 0.99], centres), want=4,
                           min_fraction=0.35, min_step=0.03, max_step=0.15)
    assert len(chosen) == 2


def test_selection_is_chronological_so_tracking_runs_forward():
    chosen = select_frames(_scored([0.5, 0.6, 0.95, 0.7, 0.8]), want=3,
                           min_fraction=0.35, min_step=0.05, max_step=0.5)
    assert [c["index"] for c in chosen] == sorted(c["index"] for c in chosen)


@pytest.mark.parametrize("want", [1, 2, 8, 50])
def test_never_returns_more_than_asked(want):
    assert len(select_frames(_scored([0.9] * 20), want=want, min_fraction=0.35,
                             min_step=0.05, max_step=0.5)) <= want


# ----------------------------------------------------------------- mesh building
def test_poisson_runs_single_threaded():
    """Open3D's default thread count makes Poisson non-deterministic, and it aborts.

    Measured on the real instance: 9,657 / 9,655 / 9,656 triangles over three builds of
    the SAME 1,128 points, and a hard `libc++abi` abort inside 12 runs in a loop — no
    traceback, no cleanup. Single-threaded it is 12/12 identical. A mesh that changes
    between builds cannot be diffed against a saved input and cannot be fingerprinted,
    so this is pinned rather than left to a default.
    """
    o3d = pytest.importorskip("open3d")
    from pipeline.tools import build_pose_bundle

    seen = {}
    real = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson

    def spy(pcd, **kw):
        seen.update(kw)
        return real(pcd, **kw)

    cloud = o3d.geometry.PointCloud()
    phi = np.linspace(0, np.pi, 24)
    theta = np.linspace(0, 2 * np.pi, 24)
    p, t = np.meshgrid(phi, theta)
    sphere = np.stack([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)], -1)
    cloud.points = o3d.utility.Vector3dVector(sphere.reshape(-1, 3))
    cloud.estimate_normals()

    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson = staticmethod(spy)
    try:
        build_pose_bundle.poisson_mesh(cloud, depth=4)
    finally:
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson = real
    assert seen.get("n_threads") == 1


def test_poisson_mesh_is_cropped_to_its_samples():
    """Poisson extrapolates well past the points; uncropped, the mesh invents geometry."""
    o3d = pytest.importorskip("open3d")
    from pipeline.tools import build_pose_bundle

    cloud = o3d.geometry.PointCloud()
    phi = np.linspace(0.2, np.pi - 0.2, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    p, t = np.meshgrid(phi, theta)
    sphere = np.stack([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)], -1)
    cloud.points = o3d.utility.Vector3dVector(sphere.reshape(-1, 3))
    cloud.estimate_normals()

    mesh = build_pose_bundle.poisson_mesh(cloud, depth=5)
    verts = np.asarray(mesh.vertices)
    lo = np.asarray(cloud.points).min(0)
    hi = np.asarray(cloud.points).max(0)
    tol = 1e-6
    assert (verts >= lo - tol).all() and (verts <= hi + tol).all()

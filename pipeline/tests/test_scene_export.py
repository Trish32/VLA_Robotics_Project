"""The stage-1 -> stage-2 handoff, where the failures are geometric and silent.

Nothing here checks that a mask is good. It checks that the five things OpenMask3D reads
together — cloud, images, depths, poses, intrinsics — describe the same world. When they
disagree, the pipeline does not crash: it crops the wrong part of the wrong image and
CLIP embeds it with full confidence.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.scene_export import (  # noqa: E402
    DEPTH_SCALE,
    UINT16_MAX,
    PosedFrame,
    SceneExportError,
    SceneWriter,
    depth_to_uint16,
)

K = np.array([300.0, 300.0, 160.0, 120.0])
HW = (32, 40)


def frame(pose=None, h=HW[0], w=HW[1]) -> PosedFrame:
    return PosedFrame(
        color=np.full((h, w, 3), 7, np.uint8),
        depth_m=np.full((h, w), 1.5, np.float32),
        cam_to_world=np.eye(4) if pose is None else pose,
    )


def writer(tmp_path, metric=True) -> SceneWriter:
    return SceneWriter(tmp_path, "scene0", K, HW, metric=metric)


# ---------------------------------------------------------------------- gauge


def test_a_non_metric_trajectory_is_refused(tmp_path):
    """Monocular SLAM is scale-free. Fusing it produces a scene of the wrong size that
    looks entirely plausible, and the object mesh handed to FoundationPose inherits the
    error — so this has to be refused at the boundary, not discovered later."""
    with pytest.raises(SceneExportError, match="non-metric"):
        SceneWriter(tmp_path, "scene0", K, HW, metric=False)


# ----------------------------------------------------------------------- depth


def test_depth_round_trips_through_uint16_millimetres():
    depth = np.array([[0.0, 1.5], [2.25, 0.001]], np.float32)
    counts = depth_to_uint16(depth)
    assert counts.dtype == np.uint16
    assert np.allclose(counts / DEPTH_SCALE, depth, atol=1e-3)


def test_out_of_range_depth_saturates_instead_of_wrapping():
    """numpy wraps on uint16 overflow, so a 70 m reading would silently become ~4.5 m —
    a confident, wrong, *nearby* surface, which is far worse than a clipped far one."""
    counts = depth_to_uint16(np.array([[70.0]], np.float32))
    assert counts[0, 0] == UINT16_MAX


def test_invalid_pixels_stay_zero_through_the_conversion():
    """0 is the no-return sentinel on both sides of the format."""
    depth = np.array([[0.0, 2.0], [np.nan, np.inf]], np.float32)
    counts = depth_to_uint16(depth)
    assert counts[0, 0] == 0 and counts[1, 0] == 0 and counts[1, 1] == 0
    assert counts[0, 1] == 2000


# ------------------------------------------------------------------- the pose


def test_a_transposed_or_scaled_rotation_is_refused():
    """A pose whose rotation block is not orthonormal fuses into a distorted scene
    without raising anywhere in Open3D."""
    bad = np.eye(4)
    bad[:3, :3] *= 2.0
    with pytest.raises(SceneExportError, match="orthonormal"):
        frame(pose=bad).validate()


def test_a_pose_that_is_not_a_rigid_transform_is_refused():
    bad = np.eye(4)
    bad[3] = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(SceneExportError, match=r"bottom row"):
        frame(pose=bad).validate()


def test_a_non_finite_pose_is_refused():
    bad = np.eye(4)
    bad[0, 3] = np.nan
    with pytest.raises(SceneExportError, match="non-finite"):
        frame(pose=bad).validate()


def test_depth_must_be_aligned_to_colour():
    bad = PosedFrame(color=np.zeros((8, 8, 3), np.uint8),
                     depth_m=np.zeros((4, 4), np.float32),
                     cam_to_world=np.eye(4))
    with pytest.raises(SceneExportError, match="not aligned"):
        bad.validate()


# ------------------------------------------------------------- the on-disk layout


def test_filenames_are_sequential_and_unpadded(tmp_path):
    """Upstream globs and sorts these names. With zero padding absent, "10" must still
    follow "9" — this pins the naming upstream actually documents."""
    w = writer(tmp_path)
    for _ in range(11):
        w.add(frame())

    colors = sorted(p.name for p in (w.scene_dir() / "color").iterdir())
    assert "0.jpg" in colors and "10.jpg" in colors
    assert not any(name.startswith("0") and len(name) > 5 for name in colors)
    assert w.count == 11


def test_every_frame_writes_all_three_files(tmp_path):
    w = writer(tmp_path)
    index = w.add(frame())
    for sub, ext in (("color", "jpg"), ("depth", "png"), ("pose", "txt")):
        assert (w.scene_dir() / f"{sub}/{index}.{ext}").exists()


def test_the_pose_written_is_the_pose_given(tmp_path):
    """Camera-to-world, unmodified. Writing the inverse fuses a mirrored scene."""
    pose = np.eye(4)
    pose[:3, 3] = [1.0, -2.0, 0.5]
    w = writer(tmp_path)
    w.add(frame(pose=pose))

    assert np.allclose(np.loadtxt(w.scene_dir() / "pose/0.txt"), pose, atol=1e-8)


def test_intrinsics_are_expanded_to_the_4x4_upstream_reads(tmp_path):
    w = writer(tmp_path)
    written = np.loadtxt(w.scene_dir() / "intrinsic/intrinsic_color.txt")

    assert written.shape == (4, 4)
    assert written[0, 0] == K[0] and written[1, 1] == K[1]
    assert written[0, 2] == K[2] and written[1, 2] == K[3]


def test_a_frame_that_does_not_match_the_declared_resolution_is_refused(tmp_path):
    """Resizing here without rescaling fx/fy/cx/cy reprojects every mask to the wrong
    place, at a scale factor nothing downstream can detect."""
    w = writer(tmp_path)
    with pytest.raises(SceneExportError, match="intrinsics describe"):
        w.add(frame(h=64, w=80))


# ------------------------------------------------------------------------ TSDF


def test_fusion_places_geometry_in_front_of_the_camera():
    """The direction check that matters: Open3D integrates with WORLD-TO-CAMERA, our
    poses are camera-to-world. Passing ours straight through mirrors the scene about the
    origin, which still yields a plausible-looking cloud.

    A camera at the origin looking down +z at a 1.5 m wall must produce points near
    z = +1.5, not z = -1.5.
    """
    o3d = pytest.importorskip("open3d")
    from pipeline.scene_export import fuse_tsdf

    intr = np.eye(4)
    intr[0, 0] = intr[1, 1] = 300.0
    intr[0, 2], intr[1, 2] = 160.0, 120.0

    pose = np.eye(4)                      # camera at world origin, +z forward
    frames = [PosedFrame(color=np.full((240, 320, 3), 200, np.uint8),
                         depth_m=np.full((240, 320), 1.5, np.float32),
                         cam_to_world=pose)]

    cloud = fuse_tsdf(frames, intr, (240, 320), voxel_length=0.02)
    points = np.asarray(cloud.points)

    assert len(points) > 0, "fusion produced an empty cloud"
    assert points[:, 2].mean() == pytest.approx(1.5, abs=0.1), (
        f"wall fused at z={points[:, 2].mean():.2f}; a negative value means the "
        "extrinsic was not inverted"
    )


# ------------------------------------------------------- gravity-levelled fusion

K3 = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])


def tilted_room(tilt_deg=40.0, n=26):
    """Posed frames of a surface viewed by a pitched-down camera — TUM's geometry.

    Built by ROTATING a level scene, so the tilt to recover is known exactly rather
    than asserted against whatever the fitter happens to return.
    """
    from pipeline.scene_export import PosedFrame

    angle = np.radians(tilt_deg)
    tilt = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), -np.sin(angle)],
                     [0.0, np.sin(angle), np.cos(angle)]])
    yy, _ = np.mgrid[0:HW[0], 0:HW[1]]
    depth = (1.4 + 0.0015 * (yy - HW[0] / 2)).astype(np.float32)
    colour = np.full((*HW, 3), 128, np.uint8)

    frames = []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, :3] = tilt
        pose[:3, 3] = tilt @ np.array([0.04 * i - 0.5, 0.0, 0.0])
        frames.append(PosedFrame(color=colour, depth_m=depth, cam_to_world=pose))
    return frames, tilt


@pytest.mark.parametrize("tilt_deg", [0.0, 25.0, 40.0])
def test_fusion_returns_a_proper_rotation_for_any_tilt(tilt_deg):
    pytest.importorskip("open3d")
    from pipeline.scene_export import fuse_tsdf_levelled

    frames, _ = tilted_room(tilt_deg)
    scene = fuse_tsdf_levelled(frames, K3, HW)

    assert len(np.asarray(scene.cloud.points)) > 0
    # det = -1 fits the plane just as well and mirrors the whole scene.
    assert np.isclose(np.linalg.det(scene.T_level_slam[:3, :3]), 1.0, atol=1e-6)
    assert np.allclose(scene.T_level_slam[3], [0, 0, 0, 1])


def test_cloud_and_poses_come_back_as_one_value():
    """Taking the cloud without the poses is the bug this return type prevents."""
    pytest.importorskip("open3d")
    from pipeline.scene_export import fuse_tsdf_levelled

    frames, _ = tilted_room()
    scene = fuse_tsdf_levelled(frames, K3, HW)

    assert len(scene.frames) == len(frames)
    for original, levelled in zip(frames, scene.frames):
        assert np.allclose(levelled.cam_to_world,
                           scene.T_level_slam @ original.cam_to_world)


def test_levelling_preserves_distances():
    """It is a rotation. Any changed distance means the scene was deformed."""
    pytest.importorskip("open3d")
    from pipeline.scene_export import fuse_tsdf, fuse_tsdf_levelled

    frames, _ = tilted_room()
    raw = np.asarray(fuse_tsdf(frames, K3, HW).points)
    scene = fuse_tsdf_levelled(frames, K3, HW)
    levelled = np.asarray(scene.cloud.points)

    assert len(raw) == len(levelled) > 0
    origin_raw = frames[0].cam_to_world[:3, 3]
    origin_new = scene.frames[0].cam_to_world[:3, 3]
    assert np.allclose(np.sort(np.linalg.norm(raw - origin_raw, axis=1)),
                       np.sort(np.linalg.norm(levelled - origin_new, axis=1)), atol=1e-6)


def test_an_empty_fusion_is_refused_rather_than_levelled():
    """No cloud means no plane; levelling against nothing would invent a frame."""
    pytest.importorskip("open3d")
    from dataclasses import replace

    from pipeline.scene_export import SceneExportError, fuse_tsdf_levelled

    frames, _ = tilted_room()
    blank = [replace(f, depth_m=np.zeros_like(f.depth_m)) for f in frames]
    with pytest.raises(SceneExportError, match="empty cloud|plane"):
        fuse_tsdf_levelled(blank, K3, HW)

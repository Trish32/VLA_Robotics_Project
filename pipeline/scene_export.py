"""Stage 1 -> stage 2: posed RGB-D from SLAM into the scene layout OpenMask3D reads.

`run_openmask3d_single_scene.sh` wants a directory, not a stream:

    <scene>/color/0.jpg          1.jpg ...   sequential, NO zero padding
    <scene>/depth/0.png          1.png ...   uint16, DEPTH_SCALE counts per metre
    <scene>/pose/0.txt           1.txt ...   4x4 camera-to-world
    <scene>/intrinsic/intrinsic_color.txt    4x4
    <scene>/<scene>.ply                      the point cloud the masks are defined on

Everything here is about making the five agree. They are consumed together — a mask is
proposed on the `.ply`, projected into a frame using that frame's pose and the shared
intrinsics, then cropped from the `.jpg` — so any disagreement between them produces
crops of the wrong part of the wrong image, and CLIP will happily embed those.

Four invariants are enforced rather than assumed:

**The trajectory must be metric.** Monocular DROID normalises disparity, so its
translations carry an arbitrary scale while looking perfectly reasonable. Fusing those
into a TSDF gives a scene whose size is wrong by an unknown factor, and every downstream
metre — the object mesh handed to FoundationPose, the pre-grasp offset — inherits it.
`slam_server.py` reports this as `metric`; `SceneWriter` refuses without it.

**Poses are camera-to-world.** `video.poses` is world-to-camera and the server already
inverts. Feeding the un-inverted direction still fuses, into a mirrored scene.

**Intrinsics must match the images actually written.** DROID rescales frames to ~384x512
internally; if we write the original images but DROID's rescaled intrinsics (or the
reverse), reprojection is off by the scale factor with nothing to raise on.

**Depth round-trips through uint16.** We hold metres; the format is integer counts. At
DEPTH_SCALE=1000 that is millimetre quantisation and a 65.5 m ceiling — fine indoors,
and zero still means "no return" on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

DEPTH_SCALE = 1000.0          # counts per metre, matching the single-scene script
UINT16_MAX = 65535


class SceneExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PosedFrame:
    """One RGB-D observation with the pose SLAM assigned to it."""

    color: np.ndarray              # (H, W, 3) uint8, RGB
    depth_m: np.ndarray            # (H, W) float32 metres, 0 = no return
    cam_to_world: np.ndarray       # (4, 4) float64
    stamp_ns: int | None = None

    def validate(self) -> None:
        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise SceneExportError(f"color must be (H, W, 3), got {self.color.shape}")
        if self.depth_m.shape != self.color.shape[:2]:
            raise SceneExportError(
                f"depth {self.depth_m.shape} is not aligned to color "
                f"{self.color.shape[:2]}; a mask projected into this frame would sample "
                "the wrong pixels"
            )
        if self.cam_to_world.shape != (4, 4):
            raise SceneExportError(f"pose must be 4x4, got {self.cam_to_world.shape}")
        if not np.isfinite(self.cam_to_world).all():
            raise SceneExportError("pose contains non-finite values")
        bottom = self.cam_to_world[3]
        if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise SceneExportError(
                f"pose bottom row is {bottom}, not [0,0,0,1] — this is not a rigid "
                "transform, or it is transposed"
            )
        rotation = self.cam_to_world[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
            raise SceneExportError(
                "pose rotation block is not orthonormal; a scaled or transposed "
                "rotation fuses into a distorted scene without raising"
            )


def depth_to_uint16(depth_m: np.ndarray, scale: float = DEPTH_SCALE) -> np.ndarray:
    """Metres -> integer counts, saturating rather than wrapping.

    numpy wraps on overflow, so a 70 m depth reading would silently become ~4.5 m.
    Anything past the ceiling is clipped to the ceiling, which is wrong but monotonic;
    invalid (0) stays 0.
    """
    counts = np.rint(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * scale)
    return np.clip(counts, 0, UINT16_MAX).astype(np.uint16)


class SceneWriter:
    """Accumulate posed frames and emit the directory OpenMask3D expects."""

    def __init__(self, root: Path, name: str, intrinsics: np.ndarray,
                 image_hw: tuple[int, int], metric: bool) -> None:
        if not metric:
            raise SceneExportError(
                "refusing to export a non-metric trajectory. Monocular SLAM is "
                "scale-free, so the fused scene would be the wrong size by an unknown "
                "factor — and the object mesh handed to FoundationPose would be too. "
                "Re-run SLAM with depth (use_depth:=true) so the gauge is fixed."
            )
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if intrinsics.shape == (4,):
            fx, fy, cx, cy = intrinsics
            intrinsics = np.eye(4)
            intrinsics[0, 0], intrinsics[1, 1] = fx, fy
            intrinsics[0, 2], intrinsics[1, 2] = cx, cy
        if intrinsics.shape != (4, 4):
            raise SceneExportError(
                f"intrinsics must be [fx, fy, cx, cy] or 4x4, got {intrinsics.shape}"
            )

        self.root = Path(root) / name
        self.name = name
        self.intrinsics = intrinsics
        self.image_hw = tuple(image_hw)
        self.count = 0
        for sub in ("color", "depth", "pose", "intrinsic"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        np.savetxt(self.root / "intrinsic/intrinsic_color.txt", intrinsics, fmt="%.10f")

    def add(self, frame: PosedFrame) -> int:
        """Write one frame. Returns its index — the filename stem OpenMask3D will use."""
        frame.validate()
        if frame.color.shape[:2] != self.image_hw:
            raise SceneExportError(
                f"frame is {frame.color.shape[:2]} but the scene's intrinsics describe "
                f"{self.image_hw}. Resizing here without rescaling fx/fy/cx/cy would "
                "reproject every mask to the wrong place."
            )
        import cv2

        index = self.count
        # No zero padding: upstream globs and sorts these, and "10" must follow "9".
        cv2.imwrite(str(self.root / f"color/{index}.jpg"),
                    cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.root / f"depth/{index}.png"), depth_to_uint16(frame.depth_m))
        np.savetxt(self.root / f"pose/{index}.txt", frame.cam_to_world, fmt="%.10f")
        self.count += 1
        return index

    def scene_dir(self) -> Path:
        return self.root

    def ply_path(self) -> Path:
        return self.root / f"{self.name}.ply"


def fuse_tsdf(frames: list[PosedFrame], intrinsics: np.ndarray, image_hw: tuple[int, int],
              voxel_length: float = 0.02, sdf_trunc: float = 0.06,
              depth_trunc: float = 4.0):
    """Posed RGB-D -> a fused Open3D point cloud in the world frame.

    This is what replaces a CAD model later: OpenMask3D segments instances on this
    cloud, and the per-instance subset becomes the mesh FoundationPose registers
    against. So its scale is the pipeline's scale, which is why `SceneWriter` refuses
    non-metric input upstream of here.

    `voxel_length` 2cm matches the `voxel_size` Mask3D was configured with; a mismatch
    does not error, it just means the masks are proposed at a different resolution than
    the geometry was built at.
    """
    import open3d as o3d

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    height, width = image_hw
    pinhole = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # UniformTSDFVolume, NOT ScalableTSDFVolume. Scalable is the natural choice for a
    # scene of unknown extent and it is BROKEN in open3d 0.20.0 on arm64 macOS:
    # `integrate()` silently writes nothing, so `extract_voxel_point_cloud()` returns 0
    # points for input that Uniform fuses correctly. It does not raise, it returns an
    # empty cloud — which downstream reads as "the room was empty". See bug_log.txt.
    #
    # The cost is that Uniform needs its bounds up front, so they are derived from the
    # camera track padded by the depth horizon, and the resolution is capped to keep the
    # allocation finite (it is O(resolution^3)).
    centres = np.stack([f.cam_to_world[:3, 3] for f in frames])
    lo = centres.min(axis=0) - depth_trunc
    hi = centres.max(axis=0) + depth_trunc
    length = float(np.max(hi - lo))
    resolution = int(min(round(length / voxel_length), 512))
    if resolution < 8:
        raise SceneExportError(
            f"scene extent {length:.3f} m is too small to voxelise at "
            f"{voxel_length} m — are the poses metric?"
        )

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length, resolution=resolution, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=lo.astype(np.float64),
    )
    for frame in frames:
        frame.validate()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color)),
            o3d.geometry.Image(depth_to_uint16(frame.depth_m)),
            depth_scale=DEPTH_SCALE, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        # Open3D integrates with WORLD-TO-CAMERA (the "extrinsic"), and our poses are
        # camera-to-world. Passing ours directly fuses a mirrored scene that still looks
        # like a plausible room.
        volume.integrate(rgbd, pinhole, np.linalg.inv(frame.cam_to_world))

    return volume.extract_point_cloud()


@dataclass(frozen=True)
class LevelledScene:
    """A fused cloud and the poses that built it, both in a gravity-aligned frame.

    Returned as one value on purpose. Levelling the cloud without levelling the poses
    leaves the map rotated away from the trajectory that produced it, and every
    downstream composition `T_world_cam @ T_cam_obj` then places objects in a frame the
    cloud no longer occupies — with no error raised anywhere, because both halves stay
    individually well-formed. Handing back a pair makes taking one without the other an
    explicit act.
    """

    cloud: object                    # open3d PointCloud, levelled
    frames: list[PosedFrame]         # same rotation applied to every pose
    T_level_slam: np.ndarray         # (4, 4) SLAM frame -> levelled frame
    inlier_fraction: float           # share of the cloud on the supporting plane

    def to_slam_frame(self, points: np.ndarray) -> np.ndarray:
        """Levelled -> SLAM, for comparing against anything saved before levelling."""
        R = self.T_level_slam[:3, :3]
        return np.asarray(points, float).reshape(-1, 3) @ R


def fuse_tsdf_levelled(frames: list[PosedFrame], intrinsics: np.ndarray,
                       image_hw: tuple[int, int], **kwargs) -> LevelledScene:
    """`fuse_tsdf`, then rotate the result into a gravity-aligned frame.

    SLAM's world frame is the first keyframe's camera frame, which is not level — on TUM
    `fr1/xyz` the camera is pitched ~45 degrees down, so the supporting plane's normal
    lands 39.9 degrees from one axis and 50.1 from another. Consumers that assume z-up
    (`infer_relations`' `on` test, place containment, `approach_axis` documented as
    "from above") are wrong by that angle and cannot detect it. See `pipeline/gravity.py`.

    Levelling happens here rather than at display time because it changes results, not
    appearances.
    """
    from pipeline.gravity import align_to_z_up

    cloud = fuse_tsdf(frames, intrinsics, image_hw, **kwargs)
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise SceneExportError(
            "fusion produced an empty cloud; there is no plane to level against"
        )

    poses = np.stack([f.cam_to_world for f in frames])
    levelled, rotated, T, fraction = align_to_z_up(points, poses)

    cloud.points = _o3d().utility.Vector3dVector(levelled)
    if cloud.has_normals():
        # Normals are directions in the same frame; leaving them behind would make a
        # levelled cloud whose normals still point the old way.
        cloud.normals = _o3d().utility.Vector3dVector(
            np.asarray(cloud.normals) @ T[:3, :3].T)

    return LevelledScene(
        cloud=cloud,
        frames=[replace(f, cam_to_world=rotated[i]) for i, f in enumerate(frames)],
        T_level_slam=T,
        inlier_fraction=fraction,
    )


def _o3d():
    import open3d as o3d
    return o3d

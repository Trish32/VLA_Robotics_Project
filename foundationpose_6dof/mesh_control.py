"""Build a mesh for a CAD-annotated object the way our pipeline builds one for the chair.

The question this answers: when FoundationPose cannot fix the chair's orientation, is the
limit our MESH PIPELINE (TSDF fusion -> sparse instance points -> outlier rejection ->
Poisson -> crop) or the OBJECT (a 1.6 m thin-structured chair)? The v15 re-run removed
every input defect upstream of the mesh and the hypotheses still scattered, while
upstream's mustard0 in the same job converged 16/16 on its CAD mesh. Those two cases
differ in mesh and in object at once.

So: give mustard0 a mesh built our way and change nothing else. If that fails, the mesh
pipeline is convicted; if it still lands, the chair is.

Everything here is matched to the chair, RELATIVE to object size, because an absolute
match would hand a 0.19 m bottle ten voxels and fail it for a reason the chair never had:

  * TSDF voxel: the chair was fused at 2 cm over a 1.58 m longest extent, ~79 voxels
    across. mustard0 gets the same count.
  * point budget: the chair instance carried 1,128 points after outlier rejection.
  * normal radius: 0.1 m on the chair, scaled the same way.
  * outlier rejection and Poisson: not matched, IDENTICAL — the caller passes in
    `observations.reject_outliers` and `build_pose_bundle.poisson_mesh` themselves.

This module is pure numpy + open3d so it runs in the Kaggle job and in tests alike; the
pipeline functions are injected rather than imported because the job has no repo.
"""

from __future__ import annotations

import numpy as np

#: The chair, as measured in `pipeline/assets/e2e/pose_bundle/bundle.json`.
CHAIR_LONGEST_M = 1.58
CHAIR_VOXEL_M = 0.02
CHAIR_POINTS = 1128
CHAIR_NORMAL_RADIUS_M = 0.1


def relative(chair_value: float, longest_m: float) -> float:
    """A chair-pipeline length, rescaled to an object whose longest extent is `longest_m`."""
    return chair_value * longest_m / CHAIR_LONGEST_M


def rotation_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle between two rotations."""
    c = (np.trace(np.asarray(A)[:3, :3] @ np.asarray(B)[:3, :3].T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_deg_mod_flip(A: np.ndarray, B: np.ndarray, axis: int) -> float:
    """Rotation error allowing a 180 deg flip about one object axis.

    A mustard bottle is close to 2-fold symmetric about its long axis, so a pose that is
    right up to that flip is reported separately rather than counted as either right or
    wrong. Both numbers are printed; neither replaces the other.
    """
    flip = np.diag([1.0, 1.0, 1.0])
    others = [i for i in range(3) if i != axis]
    flip[others[0], others[0]] = flip[others[1], others[1]] = -1.0
    B = np.asarray(B)[:3, :3]
    return min(rotation_deg(A, B), rotation_deg(A, B @ flip))


def view_angles(ob_in_cam: list[np.ndarray]) -> np.ndarray:
    """How far each frame's view of the object is from frame 0's, in degrees.

    The camera's viewing direction in the OBJECT frame is what decides which side is
    seen, so this is the angle between the frames' object-to-camera rotations — not
    camera motion in the world, which is zero here because the robot moves the bottle.
    """
    R0 = np.asarray(ob_in_cam[0])[:3, :3]
    return np.array([rotation_deg(np.asarray(P)[:3, :3], R0) for P in ob_in_cam])


def fuse_object(frames, K: np.ndarray, voxel: float, centre: np.ndarray,
                half_extent: float, depth_scale: float = 1000.0):
    """TSDF-fuse masked depth into the object frame. Returns (points, colours).

    `frames` is a list of (rgb uint8 HxWx3, depth uint16 HxW, mask bool HxW, ob_in_cam
    4x4). Depth outside the mask is zeroed, which is the 2-D analogue of cutting the
    instance out of the fused scene — and, if anything, kinder than ours, whose instance
    boundary came from a coarse Mask3D proposal.

    A UniformTSDFVolume bounded to the object is used rather than
    `scene_export.fuse_tsdf`, because that function bounds its volume by the camera track
    padded with the depth horizon and caps resolution at 512. At bottle scale that cap
    would make the voxels ~3x coarser RELATIVE to the object than the chair's were — a
    handicap the chair never had. Same integrator, same truncation ratio (3 voxels).
    """
    import open3d as o3d

    resolution = int(np.ceil(2 * half_extent / voxel))
    origin = (np.asarray(centre, float) - half_extent).astype(np.float64)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=2 * half_extent, resolution=resolution, sdf_trunc=3 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=origin)
    h, w = frames[0][1].shape
    pinhole = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    for rgb, depth, mask, ob_in_cam in frames:
        d = np.where(mask, depth, 0).astype(np.uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d),
            depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
        # integrate() takes world-to-camera; with the object as the world, that is
        # exactly ob_in_cam. Passing its inverse fuses a mirrored object.
        volume.integrate(rgbd, pinhole, np.asarray(ob_in_cam, float))
    cloud = volume.extract_point_cloud()
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def thin_to(points: np.ndarray, colours: np.ndarray, target: int):
    """Voxel-downsample to roughly `target` points by bisecting the voxel size.

    The chair instance had 1,128 points; a bottle fused at matched resolution has more,
    because a TSDF surface is denser than a Mask3D instance cut. Matching the count keeps
    "our way" honest in the direction that matters — not giving the bottle a better
    point set than the chair ever had.
    """
    if len(points) <= target:
        return points, colours
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colours)
    span = float(np.max(points.max(0) - points.min(0)))
    lo, hi = 0.0, span
    best = pcd
    for _ in range(30):
        mid = (lo + hi) / 2
        down = pcd.voxel_down_sample(mid) if mid > 0 else pcd
        n = len(down.points)
        best = down
        if abs(n - target) <= 0.03 * target:
            break
        if n > target:
            lo = mid
        else:
            hi = mid
    return np.asarray(best.points), np.asarray(best.colors)


def ourway_mesh(points: np.ndarray, colours: np.ndarray, longest_m: float,
                reject_outliers, poisson_mesh, target_points: int = CHAIR_POINTS):
    """The chair's mesh recipe, applied to another object's fused points.

    Returns (vertices, faces, vertex_colours, centroid, n_points). Vertices are in the
    object frame shifted to the point centroid, exactly as `build_pose_bundle` shifts the
    chair — so the pose FoundationPose returns refers to that centroid.
    """
    import open3d as o3d

    pts, cols = thin_to(points, colours, target_points)
    keep = reject_outliers(pts)
    pts, cols = pts[keep], cols[keep]
    centroid = pts.mean(axis=0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts - centroid))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    r = relative(CHAIR_NORMAL_RADIUS_M, longest_m)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    mesh = poisson_mesh(pcd, depth=7)
    if len(mesh.triangles) == 0:
        raise ValueError("Poisson produced no triangles inside the point box")
    return (np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            np.asarray(mesh.vertex_colors), centroid, len(pts))

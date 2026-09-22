"""Put the world model in a gravity-aligned frame.

SLAM hands back poses in the first keyframe's camera frame. Nothing in that frame is
vertical: for TUM `fr1/xyz` the camera is pitched about 45 degrees down at a desk, so the
fused cloud inherits that tilt and its axes are (lateral, tilted, depth). Measured on the
real run, the supporting plane's normal sits **39.9 degrees from axis 1 and 50.1 degrees
from axis 2**.

Every consumer downstream assumes z-up and none of them can tell:

    scene_graph.infer_relations   `on` compares lo[2] against bhi[2] -- in the SLAM
                                  frame that compares DEPTHS, so `on` never fires. The
                                  end-to-end run inferred zero `on` relations on a desk
                                  scene with a monitor and a box sitting on desks.
    SceneNode.contains / places   axis-aligned boxes in a tilted frame are not the
                                  boxes anyone meant
    world_model_node              `approach_axis` defaults to [0, 0, 1], documented as
                                  "approached from above"; in the SLAM frame that is
                                  roughly FORWARD
    replay_rerun                  logs RIGHT_HAND_Z_UP, asserting a frame the data does
                                  not have

So this is not a display convention -- it decides whether relations and grasps are
right. Alignment belongs at fusion export, applied to the cloud AND the trajectory with
the same rotation, because rotating one without the other silently decouples the map
from the poses that built it.

Gravity is recovered from the scene's dominant plane rather than from an IMU. TUM ships
`accelerometer.txt`, but it is in the Kinect's body frame with its own axis convention,
not the optical frame the poses live in, so using it needs an extrinsic we do not have.
The supporting plane is right there in the data and needs no extra calibration.
"""

from __future__ import annotations

import numpy as np

__all__ = ["GravityError", "estimate_up", "rotation_to_z_up", "align_to_z_up"]


class GravityError(RuntimeError):
    pass


def estimate_up(
    points: np.ndarray,
    *,
    camera_positions: np.ndarray | None = None,
    distance_threshold: float = 0.02,
    min_inlier_fraction: float = 0.05,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Unit "up" vector in the input frame, from the dominant planar support.

    Returns `(up, inlier_fraction)`. The fraction is returned rather than swallowed so
    the caller can refuse a weak fit instead of rotating the world by a plane that was
    never really there.

    **The sign is the part that matters.** A plane normal is defined up to sign, and
    picking wrong flips the world upside down while every downstream check still passes:
    `on` simply never fires, exactly as it does today. It is resolved against the
    cameras — a camera looking at a supporting surface is above it, so `up` must point
    from the plane towards where the cameras were. With no camera positions, the
    fallback is that most of the scene lies BELOW its dominant support, which is weaker
    and is why `camera_positions` should be passed when available.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise GravityError(f"expected (N, 3) points, got {points.shape}")
    if len(points) < 3:
        raise GravityError(f"need at least 3 points to fit a plane, got {len(points)}")

    rng = np.random.default_rng(seed)
    best_normal, best_count, best_offset = None, 0, 0.0

    for _ in range(iterations):
        trio = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:                      # collinear sample, no plane
            continue
        normal = normal / norm
        offset = float(normal @ trio[0])
        count = int((np.abs(points @ normal - offset) <= distance_threshold).sum())
        if count > best_count:
            best_normal, best_count, best_offset = normal, count, offset

    if best_normal is None:
        raise GravityError("no plane could be fitted; are the points degenerate?")

    fraction = best_count / len(points)
    if fraction < min_inlier_fraction:
        raise GravityError(
            f"dominant plane holds only {fraction:.1%} of points (need "
            f"{min_inlier_fraction:.0%}). Refusing to align the world to a plane this "
            "weak — a wrong gravity is worse than a declared tilt, because it looks "
            "correct and silently changes every `on` relation and every grasp."
        )

    # Refit on the inliers: the 3-point sample fixes the plane's identity, not its
    # orientation to full precision, and the normal is about to define vertical.
    inliers = points[np.abs(points @ best_normal - best_offset) <= distance_threshold]
    centred = inliers - inliers.mean(axis=0)
    up = np.linalg.svd(centred, full_matrices=False)[2][-1]
    up = up / np.linalg.norm(up)

    if camera_positions is not None and len(camera_positions):
        cameras = np.asarray(camera_positions, float).reshape(-1, 3)
        if up @ (cameras.mean(axis=0) - inliers.mean(axis=0)) < 0:
            up = -up
    elif up @ (points.mean(axis=0) - inliers.mean(axis=0)) > 0:
        # No cameras: assume the bulk of the scene sits below its dominant support.
        up = -up

    return up, fraction


def rotation_to_z_up(up: np.ndarray) -> np.ndarray:
    """(3, 3) rotation taking `up` onto +z, with no roll beyond what that requires.

    Rodrigues about `up x z`. The rotation is the minimal one, so the horizontal axes
    are left as close to their original directions as possible — a gratuitous yaw would
    be harmless geometrically and confusing in every visualisation.
    """
    up = np.asarray(up, float).reshape(3)
    norm = np.linalg.norm(up)
    if norm < 1e-9:
        raise GravityError("up vector is zero; it is a direction")
    up = up / norm

    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(up @ target)

    if sine < 1e-9:
        # Already vertical: identity, or a 180-degree flip. Any axis perpendicular to
        # `up` will do for the flip, so pick one deterministically.
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])

    axis = axis / sine
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * K + (1 - cosine) * (K @ K)


def align_to_z_up(
    points: np.ndarray,
    poses: np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float]:
    """Rotate cloud and trajectory into a gravity-aligned frame with ONE rotation.

    Returns `(points, poses, T_aligned_slam, inlier_fraction)`. `T_aligned_slam` is kept
    so anything already expressed in the SLAM frame — instance centres, a saved target,
    a pose from an earlier run — can be brought forward instead of silently mixing
    frames.

    Poses are rotated, not just their translations: an object's orientation is expressed
    in the world frame too, and rotating positions alone would leave every orientation
    pointing the old way while the positions moved.
    """
    points = np.asarray(points, float)
    cameras = None if poses is None else np.asarray(poses, float).reshape(-1, 4, 4)[:, :3, 3]
    up, fraction = estimate_up(points, camera_positions=cameras, **kwargs)

    R = rotation_to_z_up(up)
    T = np.eye(4)
    T[:3, :3] = R

    rotated_poses = None
    if poses is not None:
        rotated_poses = T @ np.asarray(poses, float).reshape(-1, 4, 4)
    return points @ R.T, rotated_poses, T, fraction

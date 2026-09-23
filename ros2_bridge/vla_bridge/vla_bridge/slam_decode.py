"""Decoding DROID-SLAM pose replies into ROS-shaped odometry.

DROID-SLAM is not a policy: it consumes images and produces camera poses, so it needs a
different node from the VLA one. What it CAN share is the transport — the same ZMQ
request/reply and the same torch-free client — because `lietorch` and `droid_backends`
are CUDA-only and the SLAM system therefore has to run off-board anyway. That is the same
split the VLA bridge already uses, for the same reason.

Kept free of rclpy so it is testable outside a ROS container.

**Convention, and the reason it is stated here rather than assumed:** DROID-SLAM works in
`lietorch.SE3`, whose 7-vector layout is `[tx, ty, tz, qx, qy, qz, qw]` — translation
first, quaternion scalar-LAST. ROS `geometry_msgs/Quaternion` is also scalar-last (x, y,
z, w), so the quaternion carries across unchanged, but many other libraries (Eigen,
scipy's `as_quat` before 1.14, MuJoCo) put w first. Getting it wrong yields a valid-looking
pose that is rotated nonsense, with nothing to raise on.
"""

from __future__ import annotations

import numpy as np

# lietorch SE3 tangent/data layout, and what ROS expects for each field.
SE3_LAYOUT = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")


class SlamDecodeError(Exception):
    """Raised instead of publishing a pose we cannot vouch for."""


def decode_pose(reply: dict, key: str = "pose") -> tuple[np.ndarray, np.ndarray]:
    """Reply -> (translation (3,), quaternion (4,) as x,y,z,w).

    Accepts either a flat 7-vector in lietorch order, or an explicit
    {"translation": [...], "quaternion": [...]} pair. Rejects anything else rather than
    guessing at the layout.
    """
    if "translation" in reply and "quaternion" in reply:
        t = np.asarray(reply["translation"], dtype=np.float64).reshape(-1)
        q = np.asarray(reply["quaternion"], dtype=np.float64).reshape(-1)
    elif key in reply:
        vec = np.asarray(reply[key], dtype=np.float64).reshape(-1)
        if vec.size != 7:
            raise SlamDecodeError(
                f"{key!r} has {vec.size} elements; expected 7 in lietorch SE3 order "
                f"{SE3_LAYOUT}"
            )
        t, q = vec[:3], vec[3:]
    else:
        raise SlamDecodeError(
            f"reply has neither {key!r} nor translation/quaternion: {sorted(reply)[:6]}"
        )

    if t.size != 3:
        raise SlamDecodeError(f"translation has {t.size} elements, expected 3")
    if q.size != 4:
        raise SlamDecodeError(f"quaternion has {q.size} elements, expected 4")
    if not (np.isfinite(t).all() and np.isfinite(q).all()):
        raise SlamDecodeError("non-finite values in pose")

    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise SlamDecodeError("quaternion has ~zero norm; cannot normalise")
    if abs(norm - 1.0) > 0.1:
        # Small drift off the unit sphere is normal SLAM output and is corrected below.
        # A grossly non-unit norm is a layout or scaling bug, and renormalising it would
        # hide exactly the thing worth catching.
        raise SlamDecodeError(
            f"quaternion norm {norm:.4f} is far from 1; likely a layout mismatch "
            f"(expected scalar-last {SE3_LAYOUT[3:]})"
        )
    q = q / norm  # unconditional: a unit quaternion is unchanged to float precision

    return t, q


def is_tracking_lost(reply: dict) -> bool:
    """SLAM systems drop tracking; a stale pose republished as current is worse than none.

    Treated as lost when the server says so explicitly, or when it reports no keyframes.
    """
    if reply.get("tracking_lost") is True:
        return True
    n_keyframes = reply.get("n_keyframes")
    return n_keyframes is not None and int(n_keyframes) <= 0

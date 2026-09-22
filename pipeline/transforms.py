"""Pose conversions shared by the world model and its ROS publisher.

Not in the ROS package on purpose: these are transform utilities, they have no ROS
dependency, and keeping them here makes them testable without a container. The ROS node
imports them.
"""

from __future__ import annotations

import numpy as np


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Scalar-LAST quaternion -> (3, 3). ROS, lietorch and Sophus all use this order."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """(3, 3) -> (x, y, z, w), scalar-LAST. Shepperd's method.

    Branching on the largest denominator rather than always using the trace form: the
    naive `s = sqrt(1 + trace)` underflows for rotations near 180 degrees and returns a
    quaternion that is not unit norm. That is a rotation subtly wrong rather than
    obviously broken, which is the worst failure mode for a pose.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s, 0.25 / s)

    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k]))
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q[0], q[1], q[2], q[3]

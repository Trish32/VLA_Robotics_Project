"""Closed-form kinematics for the arena's polar arm.

The arm was *chosen* to make this file possible. A 3-link planar chain on a yawing
base — shoulder, elbow, wrist, all hinging about the same horizontal axis — has an
analytic inverse, so a Cartesian command becomes joint targets in a few lines of
trigonometry with no solver, no Jacobian and no iteration budget. A 6-DoF arm with a
spherical wrist would be more realistic and would put a numerical IK loop between the
policy and the robot, which is a second thing to debug every time a grasp misses.

The arm still has real actuators, real contacts and real gravity. What has been removed
is the *redundancy*, not the physics.

Geometry lives here rather than in the MJCF because the IK needs it as numbers;
`test_sim_arm.py` parses `arena.xml` and asserts the two agree, so the duplication
cannot drift silently.

Sign convention, shared with the MJCF: every hinge turns about ``0 -1 0``, so a positive
joint angle raises the link in the (radial, vertical) plane. Link angles are absolute
from the +radial axis for the shoulder and relative thereafter, which is the convention
the cosine rule below is written in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class OutOfReach(ValueError):
    """Requested tip position is outside the arm's annulus."""


@dataclass(frozen=True)
class ArmGeometry:
    """Link lengths in metres. Must match ``arena.xml``."""

    shoulder_height: float = 0.32
    upper: float = 0.26
    fore: float = 0.24
    wrist_to_tip: float = 0.11
    #: Slide-joint targets, per finger, measured from the gripper centreline. The pads
    #: are 6 mm thick, so the jaw gap is ``2 * (q - 0.006)``: 72 mm open, and a closed
    #: command of 20 mm asks for a 28 mm gap around a 48 mm cube. That 10 mm per side of
    #: refused interference is where grip force comes from — a position servo has no
    #: other way to squeeze.
    finger_open: float = 0.042
    finger_closed: float = 0.020

    @property
    def reach_max(self) -> float:
        return self.upper + self.fore

    @property
    def reach_min(self) -> float:
        return abs(self.upper - self.fore)


#: Where the tip is allowed to go. Tighter than the kinematic annulus on purpose: the
#: outer 4 cm of reach is a near-singular fully-extended arm where a position actuator
#: has no authority left, and the floor of 2 cm keeps the fingertips off the table.
WORKSPACE = {
    "radius": (0.18, 0.44),
    "z": (0.020, 0.40),
    "yaw": (-1.2, 1.2),
}


def clamp_to_workspace(xyz, workspace: dict | None = None) -> np.ndarray:
    """Nearest point inside the workspace, in cylindrical coordinates.

    Clamping in cylindrical rather than Cartesian coordinates because that is the shape
    the arm actually has. A Cartesian box clip lets the radius run past full extension
    at the corners, which is exactly where IK fails, so the clip would hand the solver
    the one input it cannot take.
    """
    ws = workspace or WORKSPACE
    x, y, z = (float(v) for v in xyz)
    r = math.hypot(x, y)
    yaw = math.atan2(y, x)
    r = min(max(r, ws["radius"][0]), ws["radius"][1])
    yaw = min(max(yaw, ws["yaw"][0]), ws["yaw"][1])
    z = min(max(z, ws["z"][0]), ws["z"][1])
    return np.array([r * math.cos(yaw), r * math.sin(yaw), z], np.float64)


def inverse_kinematics(xyz, geom: ArmGeometry | None = None,
                       *, tol: float = 1e-4) -> np.ndarray:
    """Joint angles ``[base_yaw, shoulder, elbow, wrist]`` putting the tip at `xyz`.

    The gripper is held vertical — the wrist absorbs whatever the first two joints leave
    over. That costs a degree of freedom and buys a grasp approach that is always
    top-down, which is the only approach a parallel gripper can use on a cube sitting on
    a table.

    Of the two elbow solutions this returns the one with the higher elbow. Both are
    kinematically valid; the elbow-down branch swings the forearm through the table on
    most of the workspace.
    """
    g = geom or ArmGeometry()
    x, y, z = (float(v) for v in xyz)
    yaw = math.atan2(y, x)
    r = math.hypot(x, y)

    # Solve to the WRIST, not the tip: the tip hangs a fixed distance below the wrist
    # once the gripper is vertical, so subtracting that offset turns a 3-link problem
    # into the 2-link one the cosine rule solves.
    wz = z + g.wrist_to_tip - g.shoulder_height
    d = math.hypot(r, wz)
    if d > g.reach_max - tol or d < g.reach_min + tol:
        raise OutOfReach(
            f"tip {xyz} needs wrist distance {d:.3f} m, outside "
            f"[{g.reach_min:.3f}, {g.reach_max:.3f}]")

    cos_elbow = (d * d - g.upper ** 2 - g.fore ** 2) / (2 * g.upper * g.fore)
    elbow_mag = math.acos(min(1.0, max(-1.0, cos_elbow)))
    phi = math.atan2(wz, r)

    best = None
    for elbow in (elbow_mag, -elbow_mag):
        shoulder = phi - math.atan2(g.fore * math.sin(elbow),
                                    g.upper + g.fore * math.cos(elbow))
        elbow_z = g.shoulder_height + g.upper * math.sin(shoulder)
        if best is None or elbow_z > best[0]:
            best = (elbow_z, shoulder, elbow)
    _, shoulder, elbow = best

    # Whatever the first two joints have turned, the wrist undoes, leaving the gripper
    # axis pointing at -z.
    wrist = -math.pi / 2 - shoulder - elbow
    return np.array([yaw, shoulder, elbow, wrist], np.float64)


def forward_kinematics(joints, geom: ArmGeometry | None = None) -> np.ndarray:
    """Tip position for ``[base_yaw, shoulder, elbow, wrist]``. Inverse of the above."""
    g = geom or ArmGeometry()
    yaw, shoulder, elbow, wrist = (float(v) for v in joints)
    a1 = shoulder
    a2 = shoulder + elbow
    a3 = shoulder + elbow + wrist
    r = g.upper * math.cos(a1) + g.fore * math.cos(a2) + g.wrist_to_tip * math.cos(a3)
    z = (g.shoulder_height + g.upper * math.sin(a1) + g.fore * math.sin(a2)
         + g.wrist_to_tip * math.sin(a3))
    return np.array([r * math.cos(yaw), r * math.sin(yaw), z], np.float64)


def reachable(xyz, geom: ArmGeometry | None = None) -> bool:
    try:
        inverse_kinematics(xyz, geom)
    except OutOfReach:
        return False
    return True


#: What `arena.xml` must still say for this file's numbers to be true. Checked by
#: `test_sim_arm.py`, which parses the MJCF: the IK needs link lengths as Python
#: floats and MuJoCo needs them as XML attributes, so the duplication is unavoidable
#: — being told when it drifts is not.
ARM_XML_CHECKS = {
    "upper_arm": ArmGeometry().upper,
    "forearm": ArmGeometry().fore,
    "forearm@pos": [ArmGeometry().upper, 0.0, 0.0],
    "wrist@pos": [ArmGeometry().fore, 0.0, 0.0],
}

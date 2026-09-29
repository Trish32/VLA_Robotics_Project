"""Kinematics of the arena's polar arm — no MuJoCo needed.

These run everywhere, which is the point: the geometry the whole simulator rests on is
checkable on a machine that cannot install a physics engine, and a link-length typo in
the MJCF is caught here rather than as a grasp that mysteriously misses by a centimetre.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pipeline.sim.arm import (ARM_XML_CHECKS, WORKSPACE, ArmGeometry, OutOfReach,
                              clamp_to_workspace, forward_kinematics,
                              inverse_kinematics, reachable)

G = ArmGeometry()


def grid(n=7):
    for r in np.linspace(*WORKSPACE["radius"], n):
        for yaw in np.linspace(*WORKSPACE["yaw"], n):
            for z in np.linspace(*WORKSPACE["z"], n):
                yield np.array([r * math.cos(yaw), r * math.sin(yaw), z])


def test_ik_fk_round_trip_over_the_whole_workspace():
    worst = 0.0
    for p in grid():
        worst = max(worst, float(np.linalg.norm(forward_kinematics(
            inverse_kinematics(p, G), G) - p)))
    # Sub-micron: this is a closed form, so anything larger is an algebra error rather
    # than accumulated numerical noise.
    assert worst < 1e-9, f"round-trip error {worst:.2e} m"


def test_every_workspace_point_is_reachable():
    """The clamp must never hand the solver a point it cannot solve.

    The failure this guards against is specifically at the corners: radius and height
    are clamped independently, so each can be legal while their combination runs past
    full extension.
    """
    bad = [p for p in grid(9) if not reachable(p, G)]
    assert not bad, f"{len(bad)} clamped points are out of reach, e.g. {bad[:3]}"


def test_clamp_is_idempotent_and_projects_outside_points_in():
    far = np.array([1.5, 0.0, 0.9])
    once = clamp_to_workspace(far)
    assert reachable(once, G)
    assert np.allclose(once, clamp_to_workspace(once))
    inside = np.array([0.30, 0.05, 0.12])
    assert np.allclose(inside, clamp_to_workspace(inside))


def test_clamp_preserves_direction_not_just_magnitude():
    """Cylindrical clamping keeps the yaw a caller asked for when yaw was legal.

    A Cartesian box clip would not: clipping x and y independently rotates the target,
    so a command to reach further along one bearing arrives on a different one.
    """
    p = np.array([1.2 * math.cos(0.4), 1.2 * math.sin(0.4), 0.15])
    c = clamp_to_workspace(p)
    assert math.atan2(c[1], c[0]) == pytest.approx(0.4, abs=1e-12)


def test_gripper_is_vertical_at_every_solution():
    for p in grid(5):
        _, sh, el, wr = inverse_kinematics(p, G)
        assert sh + el + wr == pytest.approx(-math.pi / 2, abs=1e-12)


def test_elbow_is_chosen_up():
    """Both elbow branches solve; the down one puts the forearm through the table."""
    p = np.array([0.40, 0.0, 0.05])
    _, shoulder, _, _ = inverse_kinematics(p, G)
    elbow_z = G.shoulder_height + G.upper * math.sin(shoulder)
    assert elbow_z > p[2] + 0.10


def test_out_of_reach_raises_rather_than_clipping_silently():
    with pytest.raises(OutOfReach):
        inverse_kinematics([0.90, 0.0, 0.20], G)
    with pytest.raises(OutOfReach):
        inverse_kinematics([0.0, 0.0, G.shoulder_height - G.wrist_to_tip], G)


def test_jaw_geometry_actually_grips():
    """The closed command must ask for a gap narrower than the cube it holds."""
    pad = 0.006
    gap_open = 2 * (G.finger_open - pad)
    gap_closed = 2 * (G.finger_closed - pad)
    assert gap_open > 0.054, "jaw cannot clear the largest randomised cube"
    assert gap_closed < 0.042, "closed jaw does not interfere with the smallest cube"


@pytest.mark.parametrize("name,expected", sorted(ARM_XML_CHECKS.items()))
def test_mjcf_matches_the_geometry_the_ik_assumes(name, expected):
    """Parse arena.xml and confirm the numbers the IK duplicates are still true.

    The IK needs link lengths as numbers and the MJCF needs them as attributes; there
    is no way to have one source. There is a way to notice when they diverge.
    """
    from pipeline.sim.env import ARENA
    root = ET.parse(ARENA).getroot()
    found = {}
    for geom in root.iter("geom"):
        if geom.get("fromto"):
            v = [float(x) for x in geom.get("fromto").split()]
            found[geom.get("name")] = math.dist(v[:3], v[3:])
    for body in root.iter("body"):
        if body.get("pos"):
            found[body.get("name") + "@pos"] = [float(x)
                                                for x in body.get("pos").split()]
    assert name in found, f"{name} not found in arena.xml"
    got = found[name]
    if isinstance(expected, float):
        assert got == pytest.approx(expected, abs=1e-9)
    else:
        assert np.allclose(got, expected)

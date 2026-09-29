"""Perception-derived reward: the decision rule, not the segmenter.

`pipeline/sim/reward.py` is the piece that would let this stack learn from its own
experience, so its failure modes matter more than its accuracy. The rule must not be
satisfiable by any single condition, and a missing detection must abstain rather than
score a failure — a reward that calls everything it cannot see a failure looks accurate
on a dataset of failures and teaches nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.sim.perception import Perceived
from pipeline.sim.reward import BOWL, CUBE, ShapedReward, perceived_success


def obj(label, centre, extent=(0.15, 0.15, 0.05)):
    return Perceived(label, np.asarray(centre, float), np.asarray(extent, float),
                     500, 0.9)


def placed(dx=0.0, dz=0.0):
    return {CUBE: obj(CUBE, (0.30 + dx, 0.20, 0.03 + dz), (0.05, 0.05, 0.05)),
            BOWL: obj(BOWL, (0.30, 0.20, 0.02))}


def test_a_placed_cube_is_a_success():
    assert perceived_success(placed(), grip=0.0).success


def test_containment_alone_is_not_enough():
    """Directly above the bowl, still held: contained horizontally, not placed."""
    held = placed(dz=0.22)
    assert not perceived_success(held, grip=0.0).success
    assert "height" in perceived_success(held, grip=0.0).reason


def test_release_alone_is_not_enough():
    beside = placed(dx=0.14)
    assert not perceived_success(beside, grip=0.0).success
    assert "planar" in perceived_success(beside, grip=0.0).reason


def test_seating_alone_is_not_enough():
    """Cube at bowl height and position but still gripped is a grasp, not a place."""
    t = perceived_success(placed(), grip=1.0)
    assert not t.success and "grip" in t.reason


def test_a_missing_detection_abstains_rather_than_failing():
    """The distinction that decides whether the reward can be trusted at all.

    Scored as `False`, an undetected goal would make a blind detector look like a
    discriminating one on any set containing failures.
    """
    t = perceived_success({CUBE: obj(CUBE, (0.3, 0.2, 0.03))}, grip=0.0)
    assert not t.success
    assert "not observed" in t.reason and BOWL in t.reason
    assert np.isnan(t.planar), "no geometry should be reported when nothing was seen"


def test_threshold_follows_the_perceived_bowl_not_the_arena():
    """A bigger perceived bowl must accept a wider placement, with no code change."""
    small = {CUBE: obj(CUBE, (0.34, 0.20, 0.03), (0.05, 0.05, 0.05)),
             BOWL: obj(BOWL, (0.30, 0.20, 0.02), (0.10, 0.10, 0.05))}
    big = {**small, BOWL: obj(BOWL, (0.30, 0.20, 0.02), (0.30, 0.30, 0.05))}
    assert not perceived_success(small, grip=0.0).success
    assert perceived_success(big, grip=0.0).success


def test_foreshortening_is_split_between_the_two_horizontal_axes():
    """Radius is the mean of the extents, so one foreshortened axis costs half as much."""
    t = perceived_success(
        {CUBE: obj(CUBE, (0.30, 0.20, 0.03), (0.05, 0.05, 0.05)),
         BOWL: obj(BOWL, (0.30, 0.20, 0.02), (0.16, 0.08, 0.05))}, grip=0.0)
    assert t.radius == pytest.approx(0.06)


def test_shaped_reward_rises_as_the_cube_approaches_the_bowl():
    r = ShapedReward()
    tip = np.array([0.30, 0.20, 0.25])
    far = {CUBE: obj(CUBE, (0.55, -0.10, 0.03)), BOWL: obj(BOWL, (0.30, 0.20, 0.02))}
    near = {CUBE: obj(CUBE, (0.32, 0.18, 0.03)), BOWL: obj(BOWL, (0.30, 0.20, 0.02))}
    assert r(near, tip, 1.0) > r(far, tip, 1.0)


def test_terminal_bonus_dominates_the_shaping():
    """Shaping must not be worth more than finishing, or a policy will farm it."""
    r = ShapedReward()
    tip = np.array([0.30, 0.20, 0.05])
    success = placed()
    hovering = {CUBE: obj(CUBE, (0.30, 0.20, 0.25), (0.05, 0.05, 0.05)),
                BOWL: obj(BOWL, (0.30, 0.20, 0.02))}
    assert r(success, tip, 0.0) - r(hovering, tip, 0.0) > 50

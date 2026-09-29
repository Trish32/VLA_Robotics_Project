"""Grasp and placement regions derived from perceived geometry.

This module exists because Oracle@k measured perturbation-based proposal to have zero
headroom — every candidate led to the same outcome — and measured the obvious remedy,
perturbing harder, to trade 26.7 points of reachability for 3.3 points of headroom. The
properties tested here are the ones that let an affordance proposer avoid that trade:
every candidate must be a *feasible* plan, and the set must still differ.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.sim.affordance import (JAW_HALF_WIDTH, GraspPlan, describe,
                                     graspable_radius, placeable_radius, propose)

CUBE = np.array([0.048, 0.048, 0.048])
BOWL = np.array([0.183, 0.183, 0.058])


def test_graspable_region_shrinks_with_the_jaw_not_the_object():
    """A wide flat object is limited by the pads, a narrow one by its own face."""
    wide = graspable_radius(np.array([0.070, 0.070, 0.02]))
    narrow = graspable_radius(np.array([0.020, 0.020, 0.02]))
    assert narrow < wide
    assert wide <= 0.027


def test_an_object_wider_than_the_jaw_has_no_graspable_region():
    """Zero, not a small positive number: a plan that cannot execute is worse than no
    alternative plan at all."""
    assert graspable_radius(np.array([2 * JAW_HALF_WIDTH + 0.01] * 3)) == 0.0


def test_placeable_region_uses_the_diagonal_not_the_width():
    """A cube released at 45 degrees must clear on its corner, not its face."""
    r = placeable_radius(BOWL, CUBE)
    inner = BOWL[0] / 2
    assert r < inner - CUBE[0] / 2          # stricter than a width-based bound
    assert r == pytest.approx(inner - np.linalg.norm(CUBE[:2]) / 2 - 0.004)


def test_a_container_no_bigger_than_the_object_is_unplaceable():
    assert placeable_radius(CUBE, CUBE) == 0.0


def test_candidate_zero_is_the_nominal_plan():
    """The proposer must never be able to do worse than not proposing."""
    plans = propose(CUBE, BOWL, 8, np.random.default_rng(0))
    assert plans[0].grasp_offset == (0.0, 0.0)
    assert plans[0].release_offset == (0.0, 0.0)


def test_every_plan_stays_inside_the_feasible_region():
    """The property that distinguishes this from perturbation: no candidate is a
    degraded copy, because none of them leaves the region where the plan works."""
    gr, pr = graspable_radius(CUBE), placeable_radius(BOWL, CUBE)
    for seed in range(8):
        for p in propose(CUBE, BOWL, 8, np.random.default_rng(seed)):
            assert np.linalg.norm(p.grasp_offset) <= gr + 1e-9
            assert np.linalg.norm(p.release_offset) <= pr + 1e-9
            assert 0.04 <= p.transit_z <= 0.38


def test_plans_actually_differ():
    """A proposer whose candidates coincide rebuilds the problem it exists to solve."""
    plans = propose(CUBE, BOWL, 8, np.random.default_rng(1))
    g = np.array([p.grasp_offset for p in plans])
    r = np.array([p.release_offset for p in plans])
    assert np.linalg.norm(g, axis=-1).max() > 0.010
    assert np.linalg.norm(r, axis=-1).max() > 0.030
    assert len({round(p.transit_z, 4) for p in plans}) > 2


def test_offsets_spread_around_a_ring_rather_than_clustering_at_zero():
    """A Gaussian would concentrate near the nominal plan and rebuild the failure mode:
    most candidates indistinguishable, a few far enough out to be bad."""
    plans = propose(CUBE, BOWL, 9, np.random.default_rng(2))[1:]
    ang = np.array([np.arctan2(*p.grasp_offset[::-1]) for p in plans])
    # Eight directions spread over the circle; no two should share a bearing.
    gaps = np.diff(np.sort(ang))
    assert gaps.min() > 0.3, "two plans point the same way"
    mags = np.linalg.norm([p.grasp_offset for p in plans], axis=-1)
    assert mags.min() > 0.4 * mags.max(), "some plans collapse toward the nominal one"


def test_k_of_one_returns_only_the_nominal_plan():
    assert propose(CUBE, BOWL, 1, np.random.default_rng(0)) == [
        GraspPlan((0.0, 0.0), (0.0, 0.0), 0.20, pytest.approx(
            min(graspable_radius(CUBE), placeable_radius(BOWL, CUBE))))]


def test_random_plan_is_uniform_in_area_not_in_radius():
    """Sampling the radius uniformly concentrates draws at the centre and leaves the
    outer ring — where the proposer's candidates actually live — barely covered, which
    would defeat the whole point of collecting this data."""
    from pipeline.sim.affordance import random_plan

    rng = np.random.default_rng(0)
    r = np.array([np.linalg.norm(random_plan(CUBE, BOWL, rng).grasp_offset)
                  for _ in range(4000)])
    gr = graspable_radius(CUBE)
    # Under area-uniform sampling, half the draws fall outside r/sqrt(2).
    assert abs((r > gr / np.sqrt(2)).mean() - 0.5) < 0.05
    assert r.max() <= gr + 1e-9


def test_random_plan_stays_feasible():
    from pipeline.sim.affordance import random_plan

    rng = np.random.default_rng(3)
    gr, pr = graspable_radius(CUBE), placeable_radius(BOWL, CUBE)
    for _ in range(500):
        p = random_plan(CUBE, BOWL, rng)
        assert np.linalg.norm(p.grasp_offset) <= gr + 1e-9
        assert np.linalg.norm(p.release_offset) <= pr + 1e-9
        assert 0.04 <= p.transit_z <= 0.38


def test_describe_reports_real_spreads():
    txt = describe(propose(CUBE, BOWL, 8, np.random.default_rng(0)))
    assert "plans" in txt and "grasp spread" in txt

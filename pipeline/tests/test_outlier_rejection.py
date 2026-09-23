"""Outlier rejection before the extent is taken.

An axis-aligned extent is a MAXIMUM over points, so it is set by the single furthest
point in each direction — the statistic most sensitive to a stray. That extent then
becomes the box published to TF2 and the support function the pre-grasp standoff is
measured from, so a flier does not just look wrong, it moves the frame a planner reaches
to. On the TUM run 13 stray points out of 1,142 added 40 cm to a chair's height and
pushed its pre-grasp frame 20 cm too far out.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.identity import Frames, InstanceRegistry
from pipeline.observations import (Observation, from_openmask3d, point_sets_from,
                                   reject_outliers)

NS = 10 ** 9


def blob(n=400, scale=0.1, seed=0):
    return np.random.default_rng(seed).normal(0, scale, (n, 3))


# ------------------------------------------------------------- reject_outliers

def slab(n=20, seed=0):
    """Points on a box surface — what a fused instance actually looks like, as opposed
    to a solid gaussian ball whose tail is genuinely sparse."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(-0.5, 0.5, (n * n, 2)).T
    faces = [np.stack([u, v, np.full_like(u, 0.5)], 1),
             np.stack([u, v, np.full_like(u, -0.5)], 1),
             np.stack([u, np.full_like(u, 0.5), v], 1),
             np.stack([np.full_like(u, -0.5), u, v], 1)]
    return np.vstack(faces) * np.array([1.2, 0.8, 0.6])


def test_a_clean_instance_keeps_essentially_everything():
    pts = slab()
    assert len(reject_outliers(pts)) >= 0.97 * len(pts)


def test_a_gaussian_ball_loses_only_its_sparse_tail():
    """Recorded rather than asserted away: a solid ball's outermost shell really does
    have sparser neighbourhoods, so a few percent go. Real instances are surfaces, where
    the measured loss was 0.4-3.6%."""
    kept = reject_outliers(blob())
    assert 0.93 * 400 <= len(kept) < 400


def test_a_distant_flier_is_dropped():
    pts = np.vstack([blob(), [[5.0, 5.0, 5.0]]])
    kept = reject_outliers(pts)
    assert len(pts) - 1 in set(np.arange(len(pts))) - set(kept)


def test_dropping_a_flier_is_what_shrinks_the_extent():
    """The point of the exercise: one point, a large extent change."""
    pts = np.vstack([blob(), [[0.0, 0.0, 4.0]]])
    raw = pts.max(0) - pts.min(0)
    cleaned = pts[reject_outliers(pts)]
    assert raw[2] > 3.9
    assert (cleaned.max(0) - cleaned.min(0))[2] < 1.0


def test_small_sets_are_returned_untouched():
    """Fewer points than neighbours means no meaningful local density."""
    pts = blob(n=8)
    assert len(reject_outliers(pts, k=20)) == 8


def test_a_set_that_would_mostly_vanish_is_kept_whole():
    """The guard: if the rule would reject half the instance, it rejects nothing.

    A bad extent is visible and recoverable; an instance silently truncated to a
    fragment is neither, and everything downstream would treat the fragment as the whole
    object. Driven here with a cutoff below every point, which is the only way to make
    the density rule reject wholesale.
    """
    pts = slab()
    assert len(reject_outliers(pts, std_ratio=-10.0)) == len(pts)


def test_two_separated_clusters_are_both_kept():
    """Neither half is sparse in its own neighbourhood, so neither is an outlier —
    the rule measures LOCAL density, not distance from the global centroid."""
    pts = np.vstack([blob(n=200, scale=0.01), blob(n=200, scale=0.01, seed=1) + 10.0])
    assert len(reject_outliers(pts)) >= 0.95 * len(pts)


def test_stricter_std_ratio_rejects_more():
    pts = np.vstack([blob(), blob(n=20, scale=1.5, seed=7)])
    assert len(reject_outliers(pts, std_ratio=3.0)) >= len(
        reject_outliers(pts, std_ratio=1.0))


# --------------------------------------------------- integration with observations

def make(indices_list, points, labels):
    return from_openmask3d(indices_list, points, labels, [0.9] * len(labels),
                           anchor_frame=Frames.keyframe(0), stamp_ns=NS,
                           registry=InstanceRegistry())


def test_extent_uses_cleaned_points_and_kept_indices_are_recorded():
    pts = np.vstack([blob(), [[0.0, 0.0, 4.0]]])
    o = make([np.arange(len(pts))], pts, ["cup"])[0]
    assert o.extent[2] < 1.0
    assert o.kept_indices is not None and len(o.kept_indices) < len(pts)


def test_rejection_can_be_turned_off():
    pts = np.vstack([blob(), [[0.0, 0.0, 4.0]]])
    o = from_openmask3d([np.arange(len(pts))], pts, ["cup"], [0.9],
                        anchor_frame=Frames.keyframe(0), stamp_ns=NS,
                        registry=InstanceRegistry(), reject_outliers_=False)[0]
    assert o.extent[2] > 3.9
    assert len(o.kept_indices) == len(pts)


def test_point_sets_follow_the_same_points_the_extent_came_from():
    pts = np.vstack([blob(), [[0.0, 0.0, 4.0]]])
    obs = make([np.arange(len(pts))], pts, ["cup"])
    got = point_sets_from(obs, pts)[obs[0].node_id]
    assert len(got) == len(obs[0].kept_indices)
    assert got[:, 2].max() < 1.0          # the flier is absent here too


def test_point_sets_union_observations_that_merged_into_one_instance():
    """Association collapses duplicate proposals of one object; their points must be
    unioned, not one silently overwriting the other."""
    a = blob(n=300, scale=0.05)
    b = blob(n=300, scale=0.05, seed=2) + np.array([0.02, 0.0, 0.0])
    pts = np.vstack([a, b])
    obs = make([np.arange(300), np.arange(300, 600)], pts, ["cup", "cup"])
    assert len({o.node_id for o in obs}) == 1, "expected the two to merge"
    merged = point_sets_from(obs, pts)[obs[0].node_id]
    assert len(merged) == sum(len(o.kept_indices) for o in obs)


def test_point_sets_skips_observations_without_indices():
    o = Observation(node_id="x", label="cup", pose=np.eye(4), extent=np.ones(3),
                    anchor_frame=Frames.keyframe(0), stamp_ns=NS)
    assert point_sets_from([o], blob()) == {}


def test_pregrasp_standoff_moves_with_the_cleaned_extent():
    """The downstream consequence, computed the way world_model_node computes it."""
    pts = np.vstack([blob(scale=0.15), [[0.0, 0.0, 2.0]]])
    approach, standoff = np.array([0.0, 0.0, 1.0]), 0.12
    raw = from_openmask3d([np.arange(len(pts))], pts, ["cup"], [0.9],
                          anchor_frame=Frames.keyframe(0), stamp_ns=NS,
                          registry=InstanceRegistry(), reject_outliers_=False)[0]
    cleaned = make([np.arange(len(pts))], pts, ["cup"])[0]
    reach = lambda o: float(np.abs(approach) @ (o.extent / 2)) + standoff  # noqa: E731
    assert reach(raw) - reach(cleaned) > 0.4

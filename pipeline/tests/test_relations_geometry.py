"""`inside` and `near` decided from instance geometry rather than bounding boxes.

Both predicates used to be answered by the AABB, and on a real segmentation both
answers were wrong in opposite directions:

  * `inside` fired whenever one coarse proposal's box nested inside another's. On
    freiburg3_walking_xyz that produced `the chair is inside the desk` for a chair only
    4.8% of whose points lie in that desk, and `the chair is inside the person` for a
    4.1 m over-segmented region.
  * `near` compared CENTRES, which for extended objects asks the wrong question: the
    chair sits 7 cm from a desk and 79 cm from its centre, so the true relation was
    missed entirely.

The numbers in `test_measured_*` are from that scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.identity import Frames
from pipeline.observations import convex_hull, downsample
from pipeline.scene_graph import NodeKind, Predicate, SceneGraph, SceneNode

NS = 10 ** 9


def filled_box(centre, extent, n=12, seed=0):
    """A solid box of points — an instance's occupied volume, not just its corners."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, n)] * 3), -1).reshape(-1, 3)
    return np.asarray(centre) + g * np.asarray(extent) + rng.normal(0, 1e-4, g.shape)


def node(node_id, centre, extent, *, kind=NodeKind.OBJECT, pts=None, label=None):
    pose = np.eye(4)
    pose[:3, 3] = centre
    hv, hp = convex_hull(pts) if pts is not None else (None, None)
    return SceneNode(
        node_id=node_id, kind=kind, label=label or node_id,
        tf_frame=Frames.instance(node_id), anchor_frame=Frames.keyframe(0),
        pose=pose, extent=np.asarray(extent, float), stamp_ns=NS,
        hull_vertices=hv, hull_planes=hp,
        sample_points=downsample(pts) if pts is not None else None)


def relations(*nodes):
    g = SceneGraph()
    for n in nodes:
        g.upsert(n)
    return g.infer_relations(NS)


def preds(rels, predicate):
    return {(r.subject, r.object) for r in rels if r.predicate is predicate}


# ------------------------------------------------------------------ inside

def test_genuinely_contained_object_is_inside():
    """A small solid cube well within a large one: hulls agree with the boxes."""
    big = filled_box([0, 0, 0], [2.0, 2.0, 2.0])
    small = filled_box([0, 0, 0], [0.4, 0.4, 0.4], seed=1)
    rels = relations(node("room", [0, 0, 0], [2, 2, 2], kind=NodeKind.SURFACE, pts=big),
                     node("cup", [0, 0, 0], [0.4, 0.4, 0.4], pts=small))
    assert ("cup", "room") in preds(rels, Predicate.INSIDE)


def test_nested_boxes_with_disjoint_hulls_are_not_inside():
    """The chair/desk case in miniature.

    A thin L of points has a big bounding box; another object's box nests inside it
    while its points sit entirely outside the actual material. The box says `inside`,
    the geometry says no, and the geometry wins.
    """
    shell = np.vstack([filled_box([-0.9, 0, 0], [0.2, 2.0, 2.0]),
                       filled_box([0, -0.9, 0], [2.0, 0.2, 2.0], seed=2)])
    other = filled_box([0.5, 0.5, 0], [0.4, 0.4, 0.4], seed=3)
    a = node("shelf", [0, 0, 0], [2, 2, 2], kind=NodeKind.SURFACE, pts=shell)
    b = node("box", [0.5, 0.5, 0], [0.4, 0.4, 0.4], pts=other)
    # The bounding boxes really do nest — this is the prefilter the old rule stopped at.
    blo, bhi = b.aabb()
    alo, ahi = a.aabb()
    assert np.all(blo >= alo - 0.04) and np.all(bhi <= ahi + 0.04)
    assert ("box", "shelf") not in preds(relations(a, b), Predicate.INSIDE)


def test_without_hulls_the_box_rule_still_applies():
    """Backward compatibility: nodes built without point sets keep the old behaviour."""
    rels = relations(node("room", [0, 0, 0], [2, 2, 2], kind=NodeKind.SURFACE),
                     node("cup", [0, 0, 0], [0.4, 0.4, 0.4]))
    assert ("cup", "room") in preds(rels, Predicate.INSIDE)
    assert any("no hull evidence" in r.evidence for r in rels)


def test_hull_fraction_is_nan_without_hulls():
    a, b = node("a", [0, 0, 0], [1, 1, 1]), node("b", [0, 0, 0], [2, 2, 2])
    assert np.isnan(a.hull_fraction_inside(b))


# -------------------------------------------------------------------- near

def test_near_uses_surface_distance_not_centre_distance():
    """Two long slabs 10 cm apart, centres 1.1 m apart.

    Centre distance calls them far; they are adjacent. This is exactly the chair/desk
    geometry that the old rule missed.
    """
    a = filled_box([0, 0, 0], [1.0, 0.3, 0.3])
    b = filled_box([1.1, 0, 0], [1.0, 0.3, 0.3], seed=4)
    na = node("chair", [0, 0, 0], [1.0, 0.3, 0.3], pts=a)
    nb = node("desk", [1.1, 0, 0], [1.0, 0.3, 0.3], kind=NodeKind.SURFACE, pts=b)
    assert np.linalg.norm(na.centre - nb.centre) > 1.0     # centres say "far"
    assert na.separation_from(nb) < 0.2                     # surfaces say "adjacent"
    assert ("chair", "desk") in preds(relations(na, nb), Predicate.NEAR)


def test_separation_is_zero_for_interpenetrating_point_sets():
    a = filled_box([0, 0, 0], [1.0, 1.0, 1.0])
    b = filled_box([0.2, 0, 0], [1.0, 1.0, 1.0], seed=5)
    assert node("a", [0, 0, 0], [1, 1, 1], pts=a).separation_from(
        node("b", [0.2, 0, 0], [1, 1, 1], pts=b)) == pytest.approx(0.0, abs=0.05)


def test_separation_falls_back_to_nan_without_samples():
    assert np.isnan(node("a", [0, 0, 0], [1, 1, 1]).separation_from(
        node("b", [3, 0, 0], [1, 1, 1])))


def test_far_apart_objects_get_no_relation():
    a = filled_box([0, 0, 0], [0.3, 0.3, 0.3])
    b = filled_box([4, 0, 0], [0.3, 0.3, 0.3], seed=6)
    rels = relations(node("a", [0, 0, 0], [0.3] * 3, pts=a),
                     node("b", [4, 0, 0], [0.3] * 3, kind=NodeKind.SURFACE, pts=b))
    assert not rels


# --------------------------------------------------------------- downsample

def test_downsample_caps_and_keeps_small_sets_intact():
    assert len(downsample(filled_box([0, 0, 0], [1, 1, 1], n=20), cap=200)) <= 200
    small = filled_box([0, 0, 0], [1, 1, 1], n=3)      # 27 points
    assert len(downsample(small, cap=200)) == 27


def test_downsample_spreads_rather_than_following_density():
    """A dense blob plus a sparse tail: the sample must keep the tail.

    A uniform random draw would return almost nothing from the tail, which is what makes
    a surface-distance test follow wherever the camera dwelled.
    """
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 0.01, (5000, 3))
    tail = np.stack([np.linspace(0.2, 1.0, 40), np.zeros(40), np.zeros(40)], 1)
    got = downsample(np.vstack([blob, tail]), cap=200)
    assert (got[:, 0] > 0.15).sum() >= 5


def test_convex_hull_returns_none_for_degenerate_input():
    assert convex_hull(np.zeros((3, 3))) == (None, None)          # too few points
    flat = np.stack([np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.zeros(30)], 1)
    assert convex_hull(flat) == (None, None)                       # coplanar


"""The world model, and the refusals that keep a planner from acting on fiction.

Relations are checked against hand-placed geometry so "the cup is on the table" is true
by construction rather than by a threshold happening to fit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.scene_graph import (  # noqa: E402
    NS_PER_S,
    NodeKind,
    Predicate,
    Relation,
    SceneGraph,
    SceneNode,
    StaleFact,
)

NOW = 1_000 * NS_PER_S


def pose_at(x, y, z) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def node(node_id, kind, label, centre, extent, stamp_ns=NOW, **kw) -> SceneNode:
    return SceneNode(
        node_id=node_id, kind=kind, label=label,
        tf_frame=f"object/{node_id}", anchor_frame="keyframe_0",
        pose=pose_at(*centre), extent=extent, stamp_ns=stamp_ns, **kw,
    )


def tabletop_scene() -> SceneGraph:
    """A table with a cup resting on it and a pear beside the cup."""
    graph = SceneGraph()
    # Table top surface spans z in [0.70, 0.75].
    graph.upsert(node("table_0", NodeKind.SURFACE, "table", (0.0, 0.0, 0.725),
                      (1.2, 0.8, 0.05)))
    # Cup base at 0.75 => resting exactly on the table top.
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0.10, 0.0, 0.80),
                      (0.08, 0.08, 0.10), clip_similarity=0.31))
    graph.upsert(node("pear_0", NodeKind.OBJECT, "pear", (0.22, 0.0, 0.79),
                      (0.07, 0.07, 0.08), clip_similarity=0.28))
    return graph


# ------------------------------------------------------------------ relations


def test_a_resting_object_is_on_its_support():
    graph = tabletop_scene()
    graph.infer_relations(NOW)

    on = [(r.subject, r.object) for r in graph.relations if r.predicate is Predicate.ON]
    assert ("cup_0", "table_0") in on
    assert ("pear_0", "table_0") in on


def test_a_floating_object_is_not_on_anything():
    """The vertical band is a tolerance, not a licence. An object 30 cm above the table
    is not on it, and a planner told otherwise will drive the gripper into empty air."""
    graph = tabletop_scene()
    graph.upsert(node("balloon_0", NodeKind.OBJECT, "balloon", (0.0, 0.0, 1.10),
                      (0.2, 0.2, 0.2)))
    graph.infer_relations(NOW)

    on = {r.subject for r in graph.relations if r.predicate is Predicate.ON}
    assert "balloon_0" not in on


def test_an_object_beside_the_table_is_not_on_it_even_at_the_right_height():
    """Height alone is not contact — the footprints have to overlap."""
    graph = tabletop_scene()
    graph.upsert(node("mug_far", NodeKind.OBJECT, "mug", (3.0, 0.0, 0.80),
                      (0.08, 0.08, 0.10)))
    graph.infer_relations(NOW)

    on = {(r.subject, r.object) for r in graph.relations if r.predicate is Predicate.ON}
    assert ("mug_far", "table_0") not in on


def test_nearby_objects_relate_to_each_other():
    graph = tabletop_scene()
    graph.infer_relations(NOW)

    near = {(r.subject, r.object) for r in graph.relations
            if r.predicate is Predicate.NEAR}
    assert ("pear_0", "cup_0") in near or ("cup_0", "pear_0") in near


def test_relations_are_recomputed_not_accumulated():
    """A stale `on` edge pointing at a table the cup has left is worse than no edge: a
    planner will use it."""
    graph = tabletop_scene()
    graph.infer_relations(NOW)
    assert any(r.predicate is Predicate.ON for r in graph.relations)

    lifted = node("cup_0", NodeKind.OBJECT, "cup", (0.10, 0.0, 1.30),
                  (0.08, 0.08, 0.10), stamp_ns=NOW + NS_PER_S)
    graph.upsert(lifted)
    graph.infer_relations(NOW + NS_PER_S)

    on = {r.subject for r in graph.relations if r.predicate is Predicate.ON}
    assert "cup_0" not in on


def test_relations_read_back_as_language():
    graph = tabletop_scene()
    graph.infer_relations(NOW)
    assert "the cup is on the table" in graph.describe()


# ------------------------------------------------------------------ staleness


def test_a_stale_object_pose_is_refused_not_returned():
    """The whole point of the per-kind tolerance. A 2 s-old 6-DoF pose is a pose of
    where the object used to be."""
    graph = tabletop_scene()
    later = NOW + 2 * NS_PER_S

    with pytest.raises(StaleFact, match="past the"):
        graph.resolve("cup", later)


def test_a_label_does_not_spoil_at_the_rate_a_pose_does():
    """One global freshness threshold cannot serve a 30 Hz robot pose and a semantic
    label refreshed once a minute. At the fast rate every label is discarded."""
    graph = SceneGraph()
    graph.upsert(node("table_0", NodeKind.SURFACE, "table", (0, 0, 0.72), (1.2, 0.8, 0.05)))

    still_valid = graph.resolve("table", NOW + 10 * NS_PER_S)
    assert still_valid.node_id == "table_0"


def test_forgetting_drops_the_node_and_its_relations():
    """A relation naming a node that no longer exists is a dangling pointer a planner
    would dereference."""
    graph = tabletop_scene()
    graph.infer_relations(NOW)
    assert graph.relations

    removed = graph.forget_older_than(NOW + 5 * NS_PER_S)

    assert "cup_0" in removed and "pear_0" in removed
    assert "table_0" not in removed, "surfaces outlive object poses"
    assert all(r.subject in graph.nodes and r.object in graph.nodes
               for r in graph.relations)


# ------------------------------------------------------------------- grounding


def test_resolve_picks_the_best_clip_match_not_the_first_insertion():
    graph = SceneGraph()
    graph.upsert(node("mug_a", NodeKind.OBJECT, "mug", (0, 0, 0.8), (0.08,) * 3,
                      clip_similarity=0.11))
    graph.upsert(node("mug_b", NodeKind.OBJECT, "mug", (0.3, 0, 0.8), (0.08,) * 3,
                      clip_similarity=0.42))

    assert graph.resolve("mug", NOW).node_id == "mug_b"


def test_an_unknown_phrase_says_what_the_graph_does_know():
    graph = tabletop_scene()
    with pytest.raises(KeyError, match="table"):
        graph.resolve("stapler", NOW)


def test_two_cups_stay_two_objects():
    """Identity is the node id, not the label. Keying on the label would collapse them
    into one flickering object."""
    graph = SceneGraph()
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0.0, 0, 0.8), (0.08,) * 3))
    graph.upsert(node("cup_1", NodeKind.OBJECT, "cup", (0.5, 0, 0.8), (0.08,) * 3))
    assert len(graph.nodes) == 2


def test_a_late_reply_does_not_roll_the_world_backwards():
    """FoundationPose runs on its own thread; replies can arrive out of order."""
    graph = SceneGraph()
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0.5, 0, 0.8), (0.08,) * 3,
                      stamp_ns=NOW + NS_PER_S))
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0.0, 0, 0.8), (0.08,) * 3,
                      stamp_ns=NOW))

    assert graph.nodes["cup_0"].centre[0] == pytest.approx(0.5)


# ----------------------------------------------------------------- the gauge


def test_a_non_metric_pose_cannot_enter_the_graph():
    """Monocular SLAM is scale-free. A pose in unknown units makes every derived
    distance wrong by an unknown factor, and nothing downstream can detect it."""
    with pytest.raises(ValueError, match="non-metric"):
        node("cup_0", NodeKind.OBJECT, "cup", (0, 0, 0.8), (0.08,) * 3, metric=False)


def test_nodes_name_a_tf_frame_rather_than_owning_the_live_pose():
    """TF2 stays the authority for geometry; duplicating it gives two sources that
    disagree the moment the backend optimiser moves a keyframe."""
    graph = tabletop_scene()
    cup = graph.nodes["cup_0"]
    assert cup.tf_frame == "object/cup_0"
    assert cup.anchor_frame == "keyframe_0", "anchored to a keyframe, not to map"


def test_snapshot_is_flat_and_marks_staleness_without_hiding_it():
    """A 30 Hz consumer reads a cached snapshot; whether a 200 ms-old pose is usable
    depends on what it is about to do, so `stale` is reported, not filtered."""
    graph = tabletop_scene()
    graph.infer_relations(NOW)
    snap = graph.snapshot(NOW + NS_PER_S)

    assert snap["nodes"] and snap["relations"]
    by_id = {n["id"]: n for n in snap["nodes"]}
    assert by_id["cup_0"]["stale"] is True
    assert by_id["table_0"]["stale"] is False
    assert by_id["cup_0"]["tf_frame"] == "object/cup_0"


# ------------------------------------------------------------------ hierarchy


def place(node_id, label, centre, extent, parent=None) -> SceneNode:
    return node(node_id, NodeKind.PLACE, label, centre, extent, place_parent=parent)


def nested_scene() -> SceneGraph:
    """building > floor > kitchen > workspace, with a cup on a counter in the workspace."""
    graph = SceneGraph()
    graph.upsert(place("building", "building", (0, 0, 5.0), (40, 40, 12)))
    graph.upsert(place("floor_1", "floor", (0, 0, 1.5), (40, 40, 3.0), parent="building"))
    graph.upsert(place("kitchen", "kitchen", (2.0, 0, 1.5), (5.0, 4.0, 3.0), parent="floor_1"))
    graph.upsert(place("workspace", "workspace", (2.0, 0, 0.9), (1.4, 1.0, 1.2),
                       parent="kitchen"))
    graph.upsert(node("counter", NodeKind.SURFACE, "counter", (2.0, 0.0, 0.725),
                      (1.2, 0.8, 0.05)))
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (2.0, 0.0, 0.80),
                      (0.08, 0.08, 0.10)))
    return graph


def test_a_thing_belongs_to_the_innermost_place_not_all_of_them():
    """One PART_OF edge, not one per ancestor. Materialising every ancestor edge means a
    room that moves leaves a trail of edges that individually still look valid."""
    graph = nested_scene()
    graph.infer_relations(NOW)

    cup_edges = [r.object for r in graph.relations
                 if r.subject == "cup_0" and r.predicate is Predicate.PART_OF]
    assert cup_edges == ["workspace"]


def test_ancestry_walks_outward_to_the_root():
    graph = nested_scene()
    assert graph.ancestry("cup_0") == ["workspace", "kitchen", "floor_1", "building"]
    assert graph.ancestry("kitchen") == ["floor_1", "building"]
    assert graph.ancestry("building") == []


def test_asking_what_is_in_the_kitchen_reaches_through_the_workspace():
    """A non-recursive answer is almost never the question asked: the cup is in the
    workspace, and the workspace is in the kitchen, so the cup is in the kitchen."""
    graph = nested_scene()

    ids = {n.node_id for n in graph.nodes_in_place("kitchen")}
    assert {"cup_0", "counter", "workspace"} <= ids
    assert "floor_1" not in ids and "building" not in ids

    direct = {n.node_id for n in graph.nodes_in_place("kitchen", recursive=False)}
    assert "cup_0" not in direct, "the cup is in the workspace, not directly the kitchen"


def test_place_of_returns_the_tightest_region():
    graph = nested_scene()
    assert graph.place_of("cup_0").node_id == "workspace"


def test_overlapping_places_resolve_to_the_smaller_one():
    """Two regions at the same depth both containing a node: the tighter box is the more
    specific claim, and an arbitrary pick would flicker as extents are re-estimated."""
    graph = SceneGraph()
    graph.upsert(place("wide", "hall", (0, 0, 1.0), (10, 10, 2.0)))
    graph.upsert(place("tight", "nook", (0, 0, 1.0), (2.0, 2.0, 2.0)))
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0, 0, 1.0), (0.08,) * 3))

    assert graph.place_of("cup_0").node_id == "tight"


def test_a_node_outside_every_place_has_no_ancestry():
    graph = nested_scene()
    graph.upsert(node("stray", NodeKind.OBJECT, "ball", (100.0, 0, 0.5), (0.1,) * 3))
    assert graph.ancestry("stray") == []
    assert graph.place_of("stray") is None


def test_place_nesting_is_declared_not_inferred_from_overlap():
    """Guessing nesting from overlapping boxes would reparent rooms every time an extent
    was re-estimated. The parent is a statement about the map."""
    graph = SceneGraph()
    graph.upsert(place("outer", "hall", (0, 0, 1.0), (10, 10, 2.0)))
    graph.upsert(place("inner", "nook", (0, 0, 1.0), (2.0, 2.0, 2.0)))   # no parent set

    assert graph.ancestry("inner") == [], "overlap alone must not create nesting"


def test_a_cycle_in_the_hierarchy_raises_rather_than_looping():
    """A declared cycle is a wrong map. Spinning forever inside a navigation query is
    the worst way to find that out."""
    graph = SceneGraph()
    graph.upsert(place("a", "a", (0, 0, 1), (4, 4, 2), parent="b"))
    graph.upsert(place("b", "b", (0, 0, 1), (4, 4, 2), parent="a"))

    with pytest.raises(ValueError, match="cycles"):
        graph.ancestry("a")


def test_containment_uses_the_centre_so_a_boundary_object_still_belongs():
    """Requiring the whole box inside would leave a cup at a room edge in no room."""
    graph = SceneGraph()
    graph.upsert(place("room", "room", (0, 0, 1.0), (4.0, 4.0, 2.0)))
    # Centre just inside x=+2.0; the box straddles the wall.
    graph.upsert(node("cup_0", NodeKind.OBJECT, "cup", (1.98, 0, 1.0), (0.2, 0.2, 0.2)))

    assert graph.place_of("cup_0").node_id == "room"

"""Serialising the world model for a VLA's text channel."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.graph_prompt import graph_to_prompt  # noqa: E402
from pipeline.scene_graph import NS_PER_S, NodeKind, SceneGraph, SceneNode  # noqa: E402

NOW = 1_000 * NS_PER_S


def node(nid, kind, label, centre, extent, stamp_ns=NOW):
    T = np.eye(4)
    T[:3, 3] = centre
    return SceneNode(node_id=nid, kind=kind, label=label, tf_frame=f"object/{nid}",
                     anchor_frame="kf", pose=T, extent=extent, stamp_ns=stamp_ns)


def scene():
    g = SceneGraph()
    g.upsert(node("table_0", NodeKind.SURFACE, "table", (0, 0, 0.725), (1.2, 0.8, 0.05)))
    g.upsert(node("cup_0", NodeKind.OBJECT, "cup", (0.10, 0, 0.80), (0.08, 0.08, 0.10)))
    g.upsert(node("pear_0", NodeKind.OBJECT, "pear", (0.22, 0, 0.79), (0.07, 0.07, 0.08)))
    g.infer_relations(NOW)
    return g


def test_relations_become_a_readable_sentence():
    text = graph_to_prompt(scene(), NOW, task="pick up the pear",
                           target_id="pear_0", style="full")
    assert "the pear is on the table" in text
    assert "pick up the pear." in text
    assert text.startswith("Scene:")


def test_terse_style_keeps_only_the_target_relations():
    """GR00T's task_description was trained on short human phrasing. A paragraph of
    unrelated relations is off that distribution, which can degrade rather than help."""
    text = graph_to_prompt(scene(), NOW, target_id="pear_0", style="terse")
    assert "pear" in text
    assert "the cup is on the table" not in text


def test_contact_relations_come_before_proximity():
    """If the cap truncates, it should drop the weakest claims. A grasp plan depends on
    `on`; `near` is decoration."""
    text = graph_to_prompt(scene(), NOW, style="full", max_relations=2)
    near_at = text.index("near") if "near" in text else 10**6
    assert text.index("on the table") < near_at


def test_a_stale_object_is_left_out_entirely():
    """Describing where an object WAS is worse than omitting it: the policy cannot tell
    the difference and will act on it."""
    text = graph_to_prompt(scene(), NOW + 2 * NS_PER_S, style="full")
    assert "cup" not in text and "pear" not in text


def test_symmetric_relations_are_not_repeated():
    text = graph_to_prompt(scene(), NOW, style="full")
    assert text.count("the pear is near the cup") <= 1


def test_an_empty_graph_yields_just_the_task():
    assert graph_to_prompt(SceneGraph(), NOW, task="pick up the pear") == "pick up the pear."


def test_an_unknown_style_is_refused():
    with pytest.raises(ValueError, match="terse"):
        graph_to_prompt(scene(), NOW, style="verbose")


def test_the_terse_prompt_stays_short_enough_for_a_task_description():
    """The whole reason `terse` exists. GR00T's field holds phrases like 'pick up the
    block'; a prompt an order of magnitude longer is a different input distribution."""
    text = graph_to_prompt(scene(), NOW, task="pick up the pear",
                           target_id="pear_0", style="terse")
    assert len(text) < 160, f"terse prompt is {len(text)} chars: {text}"

"""Scene graph -> text, for the channels that already accept text.

Two VLAs, two different openings, and neither needs an architecture change:

**DexVLA** has a reasoning channel built for exactly this. `modeling_qwen2_vla.py`
carries `reasoning_action_proj` and `reasoning_film`, and the reasoning arrives as a
STRING (`robot_data_processor.py`: `answer = sample['reasoning'] + "Next action:"`)
whose embeddings FiLM-condition the ScaleDP denoiser. A serialised graph is the same
modality that channel already consumes.

**GR00T N1.6** has no such channel — its observation is `{video, state, language}` and
its projectors are embodiment-keyed from `statistics.json`. So the only no-retraining
option is the language field, which is a real option but a narrower one (see below).

WHY TEXT RATHER THAN A LEARNED TOKEN ENCODER
--------------------------------------------
A learned encoder producing graph tokens has a higher ceiling and costs a fine-tune plus
a new embodiment tag, which forfeits the 0-missing/0-unexpected checkpoint claim. Text
costs nothing and is testable today. Do the cheap one first and measure it; a learned
encoder is only worth building once there is a number it has to beat.

THE RISK, STATED PLAINLY
------------------------
GR00T's `annotation.human.task_description` was trained on SHORT human task descriptions
("pick up the block"). A paragraph of relations is off that distribution, and off-
distribution prompts can degrade a policy rather than help it. `style="terse"` exists
because of this: it emits only the relations involving the target, which stays close to
the training register. DexVLA's reasoning channel was trained on multi-clause reasoning
text, so it tolerates `style="full"` far better.

None of this is measurable here. DexVLA evaluates on a real robot only, and GR00T needs
LIBERO/SimplerEnv. This module builds the plumbing and makes the prompt inspectable; it
does NOT establish that conditioning on it helps.
"""

from __future__ import annotations

from pipeline.scene_graph import NodeKind, Predicate, SceneGraph


def _phrase(graph: SceneGraph, relation) -> str:
    subject = graph.nodes[relation.subject].label
    obj = graph.nodes[relation.object].label
    if relation.predicate is Predicate.PART_OF:
        return f"the {subject} is in the {obj}"
    return f"the {subject} is {relation.predicate.value} the {obj}"


def graph_to_prompt(
    graph: SceneGraph,
    now_ns: int,
    *,
    task: str = "",
    target_id: str | None = None,
    target_phrase: str | None = None,
    style: str = "terse",
    max_relations: int = 12,
) -> str:
    """Render the graph as a sentence or two of scene description.

    `style="terse"`  only relations involving the target. Closest to the register
                     GR00T's task_description was trained on.
    `style="full"`   every relation, capped. Suited to DexVLA's reasoning channel.

    Stale nodes are excluded: describing where an object *was* is worse than omitting
    it, because the policy cannot tell the difference and will act on it.
    """
    if style not in {"terse", "full"}:
        raise ValueError(f"style must be 'terse' or 'full', got {style!r}")

    fresh = {
        node_id for node_id, node in graph.nodes.items()
        if graph.max_age_s[node.kind] == float("inf")
        or node.age_s(now_ns) <= graph.max_age_s[node.kind]
    }

    relations = [
        r for r in graph.relations
        if r.subject in fresh and r.object in fresh
        and graph.nodes[r.subject].kind is not NodeKind.PLACE
    ]
    if style == "terse" and target_id is not None:
        relations = [r for r in relations if target_id in (r.subject, r.object)]

    # ON before NEAR: contact is what a grasp plan depends on, so if the cap truncates
    # the list it should drop the weakest claims rather than an arbitrary tail.
    order = {Predicate.ON: 0, Predicate.INSIDE: 1, Predicate.PART_OF: 2, Predicate.NEAR: 3}
    relations.sort(key=lambda r: order.get(r.predicate, 9))

    phrases, seen = [], set()
    for relation in relations[:max_relations]:
        text = _phrase(graph, relation)
        if text not in seen:          # `near` is symmetric and arrives twice
            seen.add(text)
            phrases.append(text)

    parts = []
    if phrases:
        parts.append("Scene: " + ", ".join(phrases) + ".")
    if target_id and target_id in graph.nodes:
        # `target_phrase` is the referring expression the CALLER grounded with. When the
        # target was resolved by CLIP feature similarity rather than by label match, the
        # node's own label can disagree with it -- on the TUM run, grounding "the
        # monitor" lands on a node whose argmax label came out "desk", and naming it by
        # that label produced "Target: the desk. pick up the monitor.", which contradicts
        # itself. The user's phrase is the more reliable half: it is what was asked for,
        # whereas the label is an argmax over a fixed vocabulary with a 0.011 median
        # margin on this scene.
        named = target_phrase or f"the {graph.nodes[target_id].label}"
        parts.append(f"Target: {named}.")
    if task:
        parts.append(task if task.endswith((".", "!", "?")) else task + ".")
    return " ".join(parts)

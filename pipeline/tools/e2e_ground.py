#!/usr/bin/env python
"""Stage 5: labelled instances -> scene graph -> a grounded target and a prompt.

Like stages 3 and 4 before it, this existed only as an inline script, so the chain could
not be re-run from stage 1. It is the same gap, closed the same way.

What this stage actually decides: which physical thing a referring expression names, and
what text the policy is handed. Both are graph operations, not string matching —
`resolve` walks nodes by label and freshness, and `graph_to_prompt` renders only the
relations involving the target, which is the register GR00T's `task_description` was
trained on.

Identity comes from `InstanceRegistry`, so several Mask3D proposals of one object
collapse to one node and the id is stable across re-runs. Enumeration order would rename
every instance whenever segmentation changed, and every TF frame and planner target
would then point somewhere else without anything raising.

    conda run -n foundationpose_vl python pipeline/tools/e2e_ground.py --query "the monitor"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "pipeline/assets/e2e"
NS = 1_000_000_000


def resolve_target(args, graph, nodes, observations, keep):
    """Ground a phrase to an instance.

    **By feature, not by label, because the label is lossy.** Stage 4 takes an argmax
    over a fixed 16-word vocabulary and keeps only the winning string, so a query for
    anything that word list did not win destroys the evidence: on this run every
    instance scores 0.21-0.28 against "a monitor" and the single highest scorer is
    `inst_2` at 0.274 — but its argmax came out "a desk", so a label-matched
    `resolve("the monitor")` raises KeyError on a graph that demonstrably contains
    monitor-like evidence. Ranking the saved per-instance CLIP features against the
    query keeps the fixed vocabulary out of the critical path entirely, which is what
    "open-vocabulary" is supposed to buy.

    The margin is returned in the description because it is the part that decides
    whether to believe the answer.
    """
    if args.resolve == "label":
        return graph.resolve(args.query, NS), "label match"

    path = OUT / "instance_features.npz"
    if not path.exists():
        print(f"[5 graph]  {path.name} missing — falling back to label matching")
        return graph.resolve(args.query, NS), "label match (no features)"
    try:
        import clip  # noqa: F401
    except ImportError:
        print("[5 graph]  clip unavailable in this env — falling back to label "
              "matching. Run this stage in openmask3d_vl for feature grounding.")
        return graph.resolve(args.query, NS), "label match (no clip)"

    from pipeline.tools.e2e_label import cosine_similarity

    blob = np.load(path, allow_pickle=True)
    features = blob["features"]
    model, _ = clip.load("ViT-L/14@336px", device="cpu")
    sims = cosine_similarity(features, model, [args.query])[:, 0].numpy()

    # observations[k] was built from labelled[keep[k]], so that is the feature row.
    scored = []
    for k, obs in enumerate(observations):
        row = keep[k]
        if np.abs(features[row]).sum() > 0:
            scored.append((float(sims[row]), obs.node_id))
    if not scored:
        raise SystemExit("no instance carries a CLIP feature to ground against")

    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else float("nan")
    print(f"[5 graph]  feature ranking for {args.query!r}:")
    for score, node_id in scored[:4]:
        print(f"           {node_id:<12}{score:.3f}")
    if margin == margin and margin < 0.01:
        print(f"           WARNING margin {margin:.3f} — the top two are within noise; "
              "this grounding is close to arbitrary")
    return nodes[best_id], f"feature sim {best_score:.3f}, margin {margin:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="the monitor")
    ap.add_argument("--task", default="pick up the monitor")
    ap.add_argument("--style", default="terse", choices=["terse", "full"])
    ap.add_argument("--resolve", default="feature", choices=["feature", "label"],
                    help="feature: rank instances by CLIP similarity to the query "
                         "(needs clip, so run this stage in openmask3d_vl). "
                         "label: string-match the argmax label assigned in stage 4.")
    args = ap.parse_args()

    import open3d as o3d

    from pipeline.graph_prompt import graph_to_prompt
    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d, to_scene_nodes
    from pipeline.scene_graph import SceneGraph

    fuse = json.load(open(OUT / "fuse.json"))
    points = np.asarray(o3d.io.read_point_cloud(fuse["ply"]).points)
    masks = np.load(OUT / "instance_masks.npz")["masks"]
    labelled = json.load(open(OUT / "labelled.json"))["instances"]

    keep = [i for i, r in enumerate(labelled) if r.get("label")]
    if not keep:
        raise SystemExit("no instance carries a label; run e2e_label.py first")
    if masks.shape[1] != len(points):
        raise SystemExit(
            f"masks index {masks.shape[1]} points but the cloud has {len(points)}. "
            "The cloud was re-fused with different settings than the masks were "
            "computed on — re-run e2e_segment.py."
        )

    observations = from_openmask3d(
        [np.flatnonzero(masks[i]) for i in keep], points,
        [labelled[i]["label"] for i in keep],
        [labelled[i]["similarity"] for i in keep],
        anchor_frame=Frames.keyframe(0), stamp_ns=NS, registry=InstanceRegistry())

    nodes = {n.node_id: n for n in to_scene_nodes(observations)}
    graph = SceneGraph()
    for node in nodes.values():
        graph.upsert(node)
    relations = graph.infer_relations(NS)

    merged = len(keep) - len(nodes)
    print(f"[5 graph]  {len(keep)} proposals -> {len(nodes)} instances"
          f"{f' ({merged} merged)' if merged else ''}")
    for node in nodes.values():
        print(f"           {node.node_id:<12} {node.kind.value:<8} {node.label}")
    print(f"[5 graph]  {len(relations)} relations")
    for r in relations:
        print(f"           {r.as_text(nodes)}")

    target, how = resolve_target(args, graph, nodes, observations, keep)
    # Name the target by the phrase that was asked for, not by our argmax label —
    # they disagree whenever grounding is by feature. See graph_prompt.
    prompt = graph_to_prompt(graph, NS, task=args.task, target_id=target.node_id,
                             target_phrase=args.query if how.startswith("feature") else None,
                             style=args.style)
    print(f"[5 graph]  resolve({args.query!r}) -> {target.node_id}   [{how}]")
    print(f"[5 graph]  prompt: {prompt!r}")

    json.dump({
        "ok": True, "target": target.node_id, "query": args.query,
        "prompt": prompt, "style": args.style,
        "nodes": len(nodes), "relations": len(relations),
        "labels": {n.node_id: n.label for n in nodes.values()},
        "relation_text": [r.as_text(nodes) for r in relations],
        "anchor_frame": target.anchor_frame,
        "target_centre": target.centre.tolist(),
        "target_extent": np.asarray(target.extent).tolist(),
    }, open(OUT / "ground.json", "w"), indent=1)

    print(f"\n           -> ground.json")
    print("[next  ]  stage 6: PYTHONPATH=grootN1_Robotics/upstream:. conda run -n groot_vl "
          "python grootN1_Robotics/tools/run_policy_cpu.py  (see pipeline/README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

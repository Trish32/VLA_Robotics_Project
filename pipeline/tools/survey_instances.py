#!/usr/bin/env python
"""Which instances in the map are poseable at all — before a registration is spent.

`build_pose_bundle.py` answers "what is the best view of THIS instance". This asks the
same question of every labelled instance in the scene, because the one that was never
asked is whether the target we picked is the right one to pose.

Three things it reports, none of which needs ground truth:

  * **best visible fraction** — how much of the instance any single frame corroborates
    against measured depth. Below ~35% a bundle cannot be built at all, and an instance
    that never clears it is not a pose candidate however good its label is.
  * **the sign of the residual** — a point BEHIND the measured surface is occluded, which
    is ordinary. A point IN FRONT of it is misplaced, because had the object been there
    it is what the depth camera would have hit. An instance whose median residual is
    negative across its own best frames is not a partial view, it is wrong geometry.
  * **what outlier rejection costs it** — an instance that loses a third of its extent to
    14 stray points was never the size the map said it was, and that extent is also the
    pose gate's depth tolerance.

This is the cheap half of the r14 pre-flight (`foundationpose_6dof/bug_log.txt` [6]):
r14 asks whether the ESTIMATOR can fix an orientation and needs a T4, this asks whether
the INPUT could support one and runs on a laptop in under a minute.

    conda run -n foundationpose_vl python pipeline/tools/survey_instances.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# E2E_DIR lets a second scene run the same stages without overwriting the first
# (the chair bundle and its Kaggle results are keyed to what is in the default).
E2E = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth-tol", type=float, default=0.08)
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="the bar build_pose_bundle.py refuses below")
    ap.add_argument("--scan-stride", type=int, default=3,
                    help="every Nth frame; a survey does not need every one")
    ap.add_argument("--out", type=Path, default=E2E / "poseability.json")
    args = ap.parse_args(argv)

    import cv2
    import open3d as o3d

    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d, reject_outliers
    from pipeline.tools.build_pose_bundle import load_sequence, project_instance

    fuse = json.load(open(E2E / "fuse.json"))
    labelled = json.load(open(E2E / "labelled.json"))["instances"]
    masks = np.load(E2E / "instance_masks.npz")["masks"]
    points = np.asarray(o3d.io.read_point_cloud(fuse["ply"]).points)
    # The survey is how a target gets CHOSEN on a new scene, so it cannot require one.
    ground = E2E / "ground.json"
    target = json.load(open(ground))["target"] if ground.exists() else None

    keep = [i for i, r in enumerate(labelled) if r.get("label")]
    observations = from_openmask3d(
        [np.flatnonzero(masks[i]) for i in keep], points,
        [labelled[i]["label"] for i in keep], [labelled[i]["similarity"] for i in keep],
        anchor_frame=Frames.keyframe(0), stamp_ns=10 ** 9, registry=InstanceRegistry())
    names = {keep[k]: o.node_id for k, o in enumerate(observations)}

    from pipeline.tools.e2e_tum import INTRINSICS
    fx, fy, cx, cy = INTRINSICS[fuse["camera"]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    seq, traj, stamps, rgb, dep, dstamps = load_sequence(fuse)

    # Group by instance id, not by proposal: `InstanceRegistry` merges overlapping
    # proposals under one id, so scoring rows would score half an object twice and
    # report it as two mediocre candidates rather than one good one.
    by_id: dict[str, list[int]] = {}
    for row in keep:
        by_id.setdefault(names[row], []).append(row)
    members = {}
    for node, rows in by_id.items():
        idx = np.unique(np.concatenate([np.flatnonzero(masks[r]) for r in rows]))
        members[node] = (idx, idx[reject_outliers(points[idx])])

    # One pass over the frames, every instance scored per frame. Reading each depth image
    # once rather than once per instance is the whole reason this is a minute not an hour.
    best = {n: {"fraction": 0.0, "index": -1, "residual": float("nan")} for n in by_id}
    seen = {n: [] for n in by_id}
    scanned = 0
    for index, (stamp, _rel) in enumerate(rgb):
        if index % args.scan_stride:
            continue
        j = int(np.argmin(np.abs(stamps - stamp)))
        k = int(np.argmin(np.abs(dstamps - stamp)))
        if abs(stamps[j] - stamp) > 0.02 or abs(dstamps[k] - stamp) > 0.02:
            continue
        depth = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        world_to_cam = np.linalg.inv(traj[stamps[j]])
        scanned += 1
        for node in by_id:
            _, kept = members[node]
            *_, ok, residual = project_instance(points[kept], world_to_cam, K, depth,
                                                args.depth_tol)
            frac = float(ok.mean())
            seen[node].append(frac)
            if frac > best[node]["fraction"]:
                med = float(np.nanmedian(residual)) if np.isfinite(residual).any() else float("nan")
                best[node] = {"fraction": frac, "index": index, "residual": med}

    rows = []
    for node, proposals in by_id.items():
        raw, kept = members[node]
        ext_raw = points[raw].max(0) - points[raw].min(0)
        ext = points[kept].max(0) - points[kept].min(0)
        b = best[node]
        rows.append({
            "instance": node, "label": labelled[proposals[0]].get("label"),
            "proposals": len(proposals),
            "points": int(len(kept)), "rejected": int(len(raw) - len(kept)),
            "extent_m": [round(float(v), 2) for v in ext],
            "shrink": round(float(np.prod(ext) / max(np.prod(ext_raw), 1e-9)), 2),
            "best_fraction": round(b["fraction"], 3),
            "best_frame": b["index"],
            "best_residual_cm": round(b["residual"] * 100, 1) if np.isfinite(b["residual"]) else None,
            "frames_over_bar": int(sum(f >= args.min_visible for f in seen[node])),
            "poseable": bool(b["fraction"] >= args.min_visible),
            "is_target": node == target,
        })
    rows.sort(key=lambda r: -r["best_fraction"])

    print(f"[survey]  {scanned} frames scanned (stride {args.scan_stride}), "
          f"{len(rows)} labelled instances, bar {args.min_visible:.0%}\n")
    head = (f"{'instance':<12}{'prop':>5}{'pts':>6}{'rej':>5}{'extent (m)':>22}{'vol':>6}"
            f"{'best':>7}{'frame':>7}{'resid':>9}{'frames':>8}  verdict")
    print(head)
    print("-" * len(head))
    for r in rows:
        resid = f"{r['best_residual_cm']:+.1f}cm" if r["best_residual_cm"] is not None else "    --"
        verdict = "POSEABLE" if r["poseable"] else "refused"
        if r["is_target"]:
            verdict += "  <- target"
        print(f"{r['instance']:<12}{r['proposals']:5d}{r['points']:6d}{r['rejected']:5d}"
              f"{str(r['extent_m']):>22}{r['shrink']:6.2f}"
              f"{r['best_fraction']:7.2f}{r['best_frame']:7d}{resid:>9}"
              f"{r['frames_over_bar']:8d}  {verdict}")

    json.dump({"ok": True, "sequence": fuse["sequence"], "scanned": scanned,
               "min_visible": args.min_visible, "target": target, "instances": rows},
              open(args.out, "w"), indent=1)
    print(f"\n[survey]  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

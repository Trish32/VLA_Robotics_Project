#!/usr/bin/env python
"""Stage 3, second path: 2-D proposals (SAM) lifted to 3-D with depth and the trajectory.

Why a second path. On TUM fr1/xyz, Mask3D proposes no compact desk object at any score
down to 0.2, and not on a 2.3x finer cloud either (segmentation_score.json, L1): it works at
a fixed 2 cm voxel and was trained on room-scale ScanNet, so a 14 cm mouse is a few voxels
and gets absorbed into the desk it sits on. 2-D proposals see objects at image resolution.

Same outputs as e2e_segment.py — segment.json and instance_masks.npz over the FUSED cloud's
points — so labelling, grounding, the scene graph and the pose bundle run unchanged.

How:
  1. SAM's automatic mask generator on K keyframes spread over the sequence;
  2. each mask's pixels lifted to world points with that frame's depth and ORB-SLAM3 pose,
     the mask eroded first (silhouette-edge depth mixes object and background);
  3. each lifted mask mapped to the fused cloud's points within --snap of it;
  4. proposals from different views merged when their cloud-point sets reach 3-D IoU
     >= --merge-iou. IoU, NOT overlap/min: SAM emits nested masks (a mouse, and the desk
     containing it), and overlap/min would read the mouse as contained in the desk and
     merge it away — which is Mask3D's failure, rebuilt. IoU keeps them apart;
  5. an instance must be seen in >= --min-views keyframes.

    E2E_DIR=pipeline/assets/e2e_fr1_l2 conda run -n openmask3d_vl \\
        python pipeline/tools/e2e_segment_2d.py
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
OUT = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))
SAM_CKPT = ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth"


def merge_proposals(props: list[np.ndarray], views: list[int], iou: float,
                    min_views: int) -> list[tuple[np.ndarray, set]]:
    """Greedy cross-view merge of cloud-index sets by 3-D IoU.

    Returns (indices, set of views) per instance. Pure and small so it can be tested
    without SAM: the nested-mask behaviour is the property that matters.
    """
    insts: list[tuple[set, set]] = []
    order = np.argsort([-len(p) for p in props])
    for k in order:
        P, v = set(props[k].tolist()), views[k]
        best, best_iou = None, 0.0
        for j, (Q, _) in enumerate(insts):
            inter = len(P & Q)
            if not inter:
                continue
            u = inter / len(P | Q)
            if u > best_iou:
                best, best_iou = j, u
        if best is not None and best_iou >= iou:
            insts[best][0].update(P)
            insts[best][1].add(v)
        else:
            insts.append((P, {v}))
    return [(np.array(sorted(Q)), V) for Q, V in insts if len(V) >= min_views]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--points-per-side", type=int, default=24)
    ap.add_argument("--min-area", type=int, default=300, help="pixels")
    ap.add_argument("--snap", type=float, default=0.015,
                    help="m; cloud points within this of a lifted mask belong to it")
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=50)
    a = ap.parse_args(argv)

    import cv2
    import open3d as o3d
    from scipy.spatial import cKDTree
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    from pipeline.observations import reject_outliers
    from pipeline.tools.build_pose_bundle import load_sequence
    from pipeline.tools.build_sam_bundle import lift
    from pipeline.tools.e2e_tum import INTRINSICS

    fuse = json.load(open(OUT / "fuse.json"))
    cloud = np.asarray(o3d.io.read_point_cloud(fuse["ply"]).points)
    tree = cKDTree(cloud)
    fx, fy, cx, cy = INTRINSICS[fuse["camera"]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    seq, traj, stamps, rgb, dep, dstamps = load_sequence(fuse)

    gen = SamAutomaticMaskGenerator(sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT)),
                                    points_per_side=a.points_per_side, min_mask_region_area=a.min_area)
    picks = np.linspace(0, len(rgb) - 1, a.keyframes).round().astype(int)
    props, views, used = [], [], []
    for kf in picks:
        s, rel = rgb[kf]
        j = int(np.argmin(np.abs(stamps - s))); k = int(np.argmin(np.abs(dstamps - s)))
        if abs(stamps[j] - s) > 0.02 or abs(dstamps[k] - s) > 0.02:
            continue
        bgr = cv2.imread(str(seq / rel))
        d = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        T = traj[stamps[j]]
        masks = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        n_kept = 0
        for m in masks:
            if m["area"] < a.min_area:
                continue
            P = lift(m["segmentation"], d, K, T)
            if len(P) < 30:
                continue
            P = P[reject_outliers(P)]
            near = tree.query_ball_point(P, a.snap)
            idx = np.unique(np.concatenate([np.asarray(x, int) for x in near if x] or [np.zeros(0, int)]))
            if len(idx) >= a.min_points:
                props.append(idx); views.append(int(kf)); n_kept += 1
        used.append(int(kf))
        print(f"[seg2d ]  keyframe {kf:4d}: {len(masks)} SAM masks -> {n_kept} lifted proposals", flush=True)

    insts = merge_proposals(props, views, a.merge_iou, a.min_views)
    masks_out = np.zeros((len(insts), len(cloud)), bool)
    records = []
    for i, (idx, V) in enumerate(insts):
        masks_out[i, idx] = True
        P = cloud[idx]
        lo, hi = P.min(0), P.max(0)
        records.append({"id": f"inst_{i}", "n": int(len(idx)), "score": float(len(V) / len(used)),
                        "centre": ((lo + hi) / 2).tolist(), "extent": (hi - lo).tolist(),
                        "views": sorted(V)})
    order = np.argsort([-r["score"] for r in records])
    masks_out = masks_out[order]
    records = [records[i] for i in order]
    for i, r in enumerate(records):
        r["id"] = f"inst_{i}"
    np.savez_compressed(OUT / "instance_masks.npz", masks=masks_out)
    json.dump({"ok": True, "levelled": bool(fuse.get("levelled")), "source": "SAM 2-D proposals lifted",
               "keyframes": used, "points_per_side": a.points_per_side, "snap_m": a.snap,
               "merge_iou": a.merge_iou, "min_views": a.min_views,
               # score = share of keyframes that saw the instance, NOT a detector confidence
               "score_meaning": "fraction of keyframes the instance was proposed in",
               "instances": records}, open(OUT / "segment.json", "w"), indent=1)
    compact = sum(max(r["extent"]) <= 0.30 for r in records)
    print(f"[seg2d ]  {len(props)} proposals over {len(used)} keyframes -> {len(records)} instances "
          f"({compact} compact) -> segment.json, instance_masks.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

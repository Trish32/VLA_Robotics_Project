#!/usr/bin/env python
"""Score Mask3D's class-agnostic instance masks against known ground truth.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is NOT a ScanNet200 benchmark number. ScanNet200 is a gated ~1 TB download and is
not reachable here, so no AP/mIoU figure from this script should be compared against
published results.

What it IS: a controlled precision check on a scene whose ground truth is EXACT because
it was constructed rather than annotated. `pipeline/tools/demo_pipeline.py` raycasts a
room from six known boxes and fuses it with the real TSDF path, so every point's true
instance is computable, not labelled by hand. That makes it a valid instrument for the
question the acceleration work actually raises — *does the speedup cost segmentation
quality?* — because the same scene can be run through both the ViT-H and MobileSAM paths
and compared to each other and to truth.

Two honest limits of the instrument:
  * six large, well-separated, convex instances. Far easier than a real ScanNet room, so
    absolute numbers here will flatter any method.
  * the scene is synthetic, so it lacks the sensor noise and clutter Mask3D was trained
    on. A poor score here is informative; a good one proves less.

Post-processing follows upstream's `trainer.get_mask_and_scores` rather than a
reimplementation, so what is scored is what OpenMask3D would actually consume.

    python openmask3d_semantic/tools/eval_instances.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The boxes demo_pipeline.py raycast the scene from: (name, centre, half-extent).
# Ground truth is derived from these, so it is exact by construction.
PARTS = [
    ("floor",     (0.0, 0.85, 2.2),   (2.0, 0.02, 1.6)),
    ("back wall", (0.0, 0.0, 3.6),    (2.0, 1.2, 0.02)),
    ("left wall", (-1.6, 0.0, 2.2),   (0.02, 1.2, 1.6)),
    ("table",     (0.0, 0.35, 2.2),   (0.55, 0.03, 0.40)),
    ("pear",      (-0.12, 0.24, 2.10), (0.07, 0.08, 0.07)),
    ("mug",       (0.22, 0.26, 2.25), (0.06, 0.07, 0.06)),
]
SLACK = 0.03      # TSDF surfaces sit a voxel or so off the analytic box face

# ScanNet colour statistics, from datasets/semseg.py:163. Mask3D was trained on
# (colour - mean) / std with colour in [0, 1]; feeding raw [0, 1] is a distribution
# shift the network has no way to report.
COLOUR_MEAN = np.array([0.47793125906962, 0.4303257521323044, 0.3749598901421883])
COLOUR_STD = np.array([0.2834475483823543, 0.27566157565723015, 0.27018971370874995])

# demo_pipeline.py builds its room y-DOWN (OpenCV camera convention). OpenMask3D's
# README requires the cloud in Z-UP right-handed coordinates, and Mask3D has a strong
# learned gravity prior — floors are horizontal at low z. A 90-degree rotation is not a
# cosmetic difference to it: (x, y, z) -> (x, z, -y), which is right-handed
# (cross(x_new, y_new) = z_new).
def to_z_up(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def ground_truth(points: np.ndarray):
    """NOTE: `points` must already be in the same frame the boxes are defined in, i.e.
    the ORIGINAL y-down frame, not the rotated one handed to Mask3D."""
    """Exact per-point instance labels from the box definitions.

    Nearest-box assignment among the boxes a point is inside (with slack). Points inside
    none are left unlabelled (-1) and excluded from scoring rather than forced into a
    class — an unlabelled point is not evidence either way.
    """
    labels = np.full(len(points), -1, dtype=int)
    best = np.full(len(points), np.inf)
    for index, (_, centre, half) in enumerate(PARTS):
        centre = np.asarray(centre)
        half = np.asarray(half) + SLACK
        inside = np.all(np.abs(points - centre) <= half, axis=1)
        # Distance to the box surface, so a point on the table is not stolen by the
        # floor simply because the floor's centre is closer in one axis.
        d = np.linalg.norm(np.maximum(np.abs(points - centre) - half, 0.0), axis=1)
        take = inside & (d < best)
        labels[take] = index
        best[take] = d[take]
    return labels


def mask3d_instances(ply: Path, voxel: float = 0.02, score_thresh: float = 0.5,
                     dbscan_eps: float = 0.95, min_points: int = 50,
                     already_z_up: bool = False):
    """Run Mask3D and post-process to binary per-point instance masks.

    `already_z_up` skips `to_z_up`. That helper is a FIXED axis permutation written for
    `demo_pipeline.py`'s synthetic y-down room, not a gravity estimate — applying it to
    a cloud that is already gravity-aligned (see `pipeline/gravity.py`) rotates the
    scene 90 degrees AWAY from vertical, which is precisely the condition Mask3D's
    learned gravity prior is least able to cope with.
    """
    import open3d as o3d
    import torch
    from omegaconf import OmegaConf
    from sklearn.cluster import DBSCAN

    from openmask3d_semantic import me_shim

    me_shim.install_all()
    sys.path.insert(0, str(ROOT / "openmask3d_semantic/upstream/openmask3d"
                                  "/class_agnostic_mask_computation"))
    sys.path.insert(0, str(ROOT / "openmask3d_semantic/tools"))
    import MinkowskiEngine as ME
    from load_checkpoint import MODEL_CFG
    from models.mask3d import Mask3D

    cfg = dict(MODEL_CFG)
    cfg["train_on_segments"] = False       # no supervoxels for an arbitrary scene
    model = Mask3D(**OmegaConf.create(cfg))
    blob = torch.load(ROOT / "openmask3d_semantic/checkpoints/scannet200_model.ckpt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("model."): v
                           for k, v in blob["state_dict"].items()}, strict=True)
    model.eval()

    pcd = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(pcd.points)
    if not already_z_up:
        points = to_z_up(points)
    colours = np.asarray(pcd.colors)

    coords = np.floor(points / voxel).astype(np.int32)
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    feats = np.zeros((len(uniq), 3), np.float64)
    np.add.at(feats, inverse, colours)
    counts = np.bincount(inverse, minlength=len(uniq))[:, None]
    feats = feats / np.maximum(counts, 1)
    feats = ((feats - COLOUR_MEAN) / COLOUR_STD).astype(np.float32)

    batched = np.column_stack([np.zeros(len(uniq), np.int32), uniq])
    x = ME.SparseTensor(features=torch.from_numpy(feats),
                        coordinates=torch.from_numpy(batched))
    with torch.no_grad():
        out = model(x, raw_coordinates=torch.from_numpy((uniq * voxel).astype(np.float32)),
                    is_eval=True)

    # --- upstream's scoring (trainer.get_mask_and_scores), not a reimplementation ----
    logits = out["pred_logits"][0].detach().float()
    mask_pred = out["pred_masks"][0].detach().float()          # (voxels, queries)
    num_queries, num_classes = logits.shape

    probs = logits.softmax(-1)[:, :-1]                         # drop the no-object class
    labels = (torch.arange(num_classes - 1).unsqueeze(0)
              .repeat(num_queries, 1).flatten(0, 1))
    scores_per_query, topk = probs.flatten(0, 1).topk(num_queries, sorted=True)
    query_index = torch.div(topk, num_classes - 1, rounding_mode="trunc")
    mask_pred = mask_pred[:, query_index]

    binary = (mask_pred > 0).float()
    heatmap = mask_pred.sigmoid()
    mask_score = (heatmap * binary).sum(0) / (binary.sum(0) + 1e-6)
    score = (scores_per_query * mask_score).numpy()
    binary = binary.numpy().astype(bool)

    # --- voxel masks -> point masks, then DBSCAN split (general.use_dbscan=true) -----
    instances = []
    for q in np.argsort(-score):
        if score[q] < score_thresh:
            break
        voxel_mask = binary[:, q]
        if voxel_mask.sum() < 4:
            continue
        point_mask = voxel_mask[inverse]
        if point_mask.sum() < min_points:
            continue
        # Upstream splits a query that fired on two disconnected regions; without this a
        # single "query" can span two real objects and score as one bad instance.
        clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(points[point_mask])
        idx = np.flatnonzero(point_mask)
        for cluster in np.unique(clustering.labels_):
            member = idx[clustering.labels_ == cluster]
            if len(member) >= min_points:
                m = np.zeros(len(points), bool)
                m[member] = True
                instances.append((m, float(score[q])))
    return points, instances


def evaluate(points, instances, gt_labels):
    """Greedy best-IoU matching, per ground-truth instance."""
    valid = gt_labels >= 0
    rows = []
    used = set()
    for index, (name, _, _) in enumerate(PARTS):
        truth = (gt_labels == index) & valid
        if truth.sum() == 0:
            rows.append((name, 0, 0.0, 0.0, None))
            continue
        best_iou, best_j = 0.0, None
        for j, (mask, _) in enumerate(instances):
            if j in used:
                continue
            pred = mask & valid
            union = (pred | truth).sum()
            iou = (pred & truth).sum() / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None:
            used.add(best_j)
        rows.append((name, int(truth.sum()), best_iou,
                     instances[best_j][1] if best_j is not None else 0.0, best_j))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path,
                    default=ROOT / "pipeline/assets/demo_scene/demo_scene.ply")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="voxel size in metres; a 7cm pear is ~4 voxels at 0.02")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--iou-thresh", type=float, default=0.25,
                    help="IoU above which a ground-truth instance counts as recalled")
    args = ap.parse_args()

    points, instances = mask3d_instances(args.ply, voxel=args.voxel,
                                         score_thresh=args.score_thresh)
    print(f"[mask3d]  {len(instances)} instances above score {args.score_thresh} "
          f"at voxel {args.voxel} m")

    # Labels come from the boxes, which are defined in the original y-down frame, so
    # undo the rotation rather than rotating the boxes.
    gt = ground_truth(np.column_stack([points[:, 0], -points[:, 2], points[:, 1]]))
    labelled = int((gt >= 0).sum())
    print(f"[gt]      {labelled}/{len(points)} points labelled "
          f"({100 * labelled / len(points):.1f}%); "
          f"{len(np.unique(gt[gt >= 0]))} instances present\n")

    rows = evaluate(points, instances, gt)
    print(f"{'ground truth':<12}{'points':>8}{'best IoU':>10}{'pred score':>12}")
    print("-" * 44)
    for name, n, iou, score, _ in rows:
        print(f"{name:<12}{n:>8}{iou:>10.3f}{score:>12.3f}")
    print("-" * 44)

    ious = [iou for _, n, iou, _, _ in rows if n > 0]
    recalled = sum(1 for i in ious if i >= args.iou_thresh)
    print(f"\nmIoU (over {len(ious)} GT instances)   {100 * np.mean(ious):.1f}%")
    print(f"recall @ IoU>{args.iou_thresh}                 "
          f"{recalled}/{len(ious)} = {100 * recalled / len(ious):.1f}%")
    print("\nSynthetic scene with exact constructed ground truth — NOT a ScanNet200")
    print("number. Six large convex instances is an easy setting; see the docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

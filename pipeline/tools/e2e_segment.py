#!/usr/bin/env python
"""Stage 3: Mask3D class-agnostic instance proposals on the fused cloud.

This stage had no committed tool. `segment.json` and `instance_masks.npz` were produced
by an inline script in an earlier session, which meant the end-to-end chain could not
actually be re-run from stage 1 — the middle was a file nobody could regenerate. That is
the gap this closes.

**The cloud is already gravity-aligned, so `to_z_up` is skipped.** `eval_instances.to_z_up`
is a fixed axis permutation `(x, y, z) -> (x, z, -y)` written for `demo_pipeline.py`'s
synthetic y-down room; it is not a gravity estimate. `pipeline/gravity.py` now levels the
cloud at fusion time by fitting the supporting plane, so applying the permutation on top
would rotate a correctly-levelled scene 90 degrees away from vertical — the one condition
Mask3D's learned gravity prior handles worst, since it was trained on ScanNet where
floors are horizontal at low z.

Masks are stored as indices into the cloud's point array, and that array's order is
load-bearing: `fuse_tsdf` is deterministic and levelling is applied pointwise, so a mask
computed here stays valid against a re-fused cloud. It does NOT stay valid across a
change of voxel size or frame stride, which change the point set itself.

    conda run -n openmask3d_vl python pipeline/tools/e2e_segment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openmask3d_semantic/tools"))

OUT = ROOT / "pipeline/assets/e2e"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, default=None,
                    help="defaults to whatever fuse.json points at")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="must match the voxel_size Mask3D was trained with")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--min-points", type=int, default=50)
    ap.add_argument("--dbscan-eps", type=float, default=0.95)
    args = ap.parse_args()

    from eval_instances import mask3d_instances

    fuse = json.load(open(OUT / "fuse.json")) if (OUT / "fuse.json").exists() else {}
    levelled = bool(fuse.get("levelled"))
    # Follow fuse.json rather than a hardcoded filename: the sequence is now a flag, and
    # segmenting last run's cloud while the graph reads this run's is silent nonsense.
    if args.ply is None:
        if not fuse.get("ply"):
            raise SystemExit("no fuse.json; run pipeline/tools/e2e_tum.py first")
        args.ply = Path(fuse["ply"])
    print(f"[3 seg  ]  cloud {args.ply.name}, "
          f"{'gravity-aligned (skipping to_z_up)' if levelled else 'RAW SLAM frame'}")
    if not levelled:
        print("           warning: cloud is not levelled; Mask3D's gravity prior will "
              "be fed a tilted scene. Re-run pipeline/tools/e2e_tum.py first.")

    points, instances = mask3d_instances(
        args.ply, voxel=args.voxel, score_thresh=args.score_thresh,
        min_points=args.min_points, dbscan_eps=args.dbscan_eps,
        already_z_up=levelled,
    )
    if not instances:
        raise SystemExit(
            f"Mask3D returned no instance above score {args.score_thresh}. Lower "
            "--score-thresh, or check that the cloud is not empty."
        )

    masks = np.stack([m for m, _ in instances])
    scores = [s for _, s in instances]
    records = []
    for i, (mask, score) in enumerate(instances):
        xyz = points[mask]
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        records.append({
            "id": f"inst_{i}",
            "n": int(mask.sum()),
            "score": float(score),
            # Centre and extent are recomputed by every consumer from the mask, because
            # an axis-aligned box is frame-dependent and these go stale the moment the
            # cloud is re-levelled. They are written for inspection, not for use.
            "centre": ((lo + hi) / 2).tolist(),
            "extent": (hi - lo).tolist(),
        })

    np.savez_compressed(OUT / "instance_masks.npz", masks=masks)
    json.dump({"ok": True, "levelled": levelled, "voxel": args.voxel,
               "score_thresh": args.score_thresh, "instances": records},
              open(OUT / "segment.json", "w"), indent=1)

    print(f"           {len(instances)} proposals above score {args.score_thresh}")
    print(f"{'id':<10}{'points':>8}{'score':>8}{'height':>9}")
    for r in sorted(records, key=lambda r: -r["score"]):
        print(f"{r['id']:<10}{r['n']:>8}{r['score']:>8.3f}{r['extent'][2]:>9.2f}")
    print(f"\n           -> segment.json, instance_masks.npz "
          f"({masks.shape[0]} x {masks.shape[1]})")
    print("[next  ]  stage 4: conda run -n openmask3d_vl python "
          "pipeline/tools/e2e_label.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

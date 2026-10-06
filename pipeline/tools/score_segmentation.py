#!/usr/bin/env python
"""Score a segmentation against the click-defined compact targets (rule in Plan.md).

An instance FINDS a target if the two point sets cover each other within --radius at
F1 >= --f1. Precision = share of the instance's points near the target surface; recall =
share of the target surface near the instance. Each target is scored against its best
instance. Also reported: instance count and compact-instance count, so a segmentation that
"finds" everything by shattering the scene into crumbs is visible as such.

Frames. Every fusion levels its own cloud (pipeline/gravity.py fits the support plane), so
two fusions of the same sequence live in DIFFERENT world frames. Targets are carried from
the fusion they were built in to the one being scored through each run's recorded
T_level_slam — skipping that would mis-score every target after a re-fusion, silently.

    E2E_DIR=pipeline/assets/e2e_fr1 conda run -n foundationpose_vl \\
        python pipeline/tools/score_segmentation.py
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
E2E = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))
TARGETS = ROOT / "pipeline/assets/e2e_fr1"


def level_of(fuse_json: Path) -> np.ndarray:
    f = json.load(open(fuse_json))
    return np.asarray(f["T_level_slam"], float) if f.get("levelled") else np.eye(4)


def coverage_f1(inst: np.ndarray, target: np.ndarray, radius: float):
    from scipy.spatial import cKDTree

    if not len(inst) or not len(target):
        return 0.0, 0.0, 0.0
    p = float((cKDTree(target).query(inst)[0] <= radius).mean())
    r = float((cKDTree(inst).query(target)[0] <= radius).mean())
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, default=TARGETS)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--f1", type=float, default=0.5)
    ap.add_argument("--compact", type=float, default=0.30)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)

    import open3d as o3d

    fuse = json.load(open(E2E / "fuse.json"))
    cloud = np.asarray(o3d.io.read_point_cloud(fuse["ply"]).points)
    masks = np.load(E2E / "instance_masks.npz")["masks"]
    from pipeline.observations import reject_outliers
    insts = []
    for m in masks:
        P = cloud[np.flatnonzero(m)]
        if len(P) > 20:
            P = P[reject_outliers(P)]
        insts.append(P)
    ext = [float((P.max(0) - P.min(0)).max()) if len(P) else 0.0 for P in insts]

    # the frame the targets were built in -> the frame this segmentation lives in
    T = level_of(E2E / "fuse.json") @ np.linalg.inv(level_of(a.targets / "fuse.json"))
    rows, found = [], 0
    for d in sorted(a.targets.glob("pose_bundle_*")):
        meta = json.load(open(d / "bundle.json"))
        V = np.asarray(o3d.io.read_triangle_mesh(str(d / "mesh.obj")).vertices)
        tgt = V + np.asarray(meta["mesh_origin_world"])
        tgt = tgt @ T[:3, :3].T + T[:3, 3]
        best = max(((coverage_f1(P, tgt, a.radius), k) for k, P in enumerate(insts)),
                   key=lambda t: t[0][2], default=((0, 0, 0), -1))
        (p, r, f1), k = best
        ok = f1 >= a.f1
        found += ok
        rows.append({"target": meta["target"], "instance": k, "precision": p,
                     "recall": r, "f1": f1, "found": bool(ok),
                     "instance_extent_m": ext[k] if k >= 0 else None})
    n_compact = sum(e <= a.compact for e in ext)
    print(f"[score ]  {a.tag or E2E.name}: {len(insts)} instances "
          f"({n_compact} with longest extent <= {a.compact:.2f} m)")
    print(f"{'target':<9}{'best inst':>10}{'prec':>7}{'recall':>8}{'F1':>7}{'inst extent':>13}  found")
    for r_ in rows:
        print(f"{r_['target']:<9}{r_['instance']:>10}{r_['precision']:7.2f}{r_['recall']:8.2f}"
              f"{r_['f1']:7.2f}{(r_['instance_extent_m'] or 0):12.2f}m  {'YES' if r_['found'] else 'no'}")
    print(f"[score ]  found {found}/{len(rows)}")
    json.dump({"tag": a.tag or E2E.name, "instances": len(insts), "compact_instances": n_compact,
               "found": found, "targets": rows, "radius_m": a.radius, "f1_bar": a.f1},
              open(E2E / "segmentation_score.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

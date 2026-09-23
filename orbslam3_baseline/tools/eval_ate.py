#!/usr/bin/env python
"""Absolute Trajectory Error against TUM ground truth.

THE ALIGNMENT MUST MATCH THE GAUGE, and this is the detail that makes or breaks a
SLAM comparison:

    RGB-D / stereo  -> SE(3), 6-DoF. Rotation and translation only.
    monocular       -> Sim(3), 7-DoF. Scale is a free parameter because the system
                       cannot observe it.

Fitting scale to an RGB-D trajectory lets the optimiser absorb real metric drift into a
scale factor, which flatters the result. Comparing a Sim(3)-aligned monocular number
against an SE(3)-aligned RGB-D one hands one system an unearned win. So `--sim3` exists
but is off by default, and the alignment used is printed with the result.

ORB-SLAM3 is also multithreaded and non-deterministic: the paper reports the MEDIAN of
several runs. A single number from a single run is a sample, not a result, and is
labelled as such below.

    python orbslam3_baseline/tools/eval_ate.py --estimate <traj.txt> --truth <gt.txt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """TUM format: timestamp tx ty tz qx qy qz qw."""
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    stamps = np.array([float(r[0]) for r in rows])
    xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
    return stamps, xyz


def associate(est_t, gt_t, max_dt=0.02):
    """Nearest-in-time pairing, the same rule TUM's own associate.py uses."""
    pairs = []
    for i, t in enumerate(est_t):
        j = int(np.argmin(np.abs(gt_t - t)))
        if abs(gt_t[j] - t) <= max_dt:
            pairs.append((i, j))
    return pairs


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool):
    """Least-squares rigid (or similarity) alignment taking `source` onto `target`."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    S, T = source - mu_s, target - mu_t
    C = T.T @ S / len(source)
    U, D, Vt = np.linalg.svd(C)
    # Reflection guard: without it SVD can return an improper rotation that fits
    # mirrored data beautifully and is not a motion any camera can make.
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    scale = (D * np.diag(W)).sum() / (S ** 2).sum() * len(source) if with_scale else 1.0
    t = mu_t - scale * R @ mu_s
    return scale, R, t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", type=Path,
                    default=HERE / "data/trajectory_orbslam3.txt")
    ap.add_argument("--truth", type=Path,
                    default=HERE / "data/rgbd_dataset_freiburg1_xyz/groundtruth.txt")
    ap.add_argument("--sim3", action="store_true",
                    help="7-DoF alignment. ONLY for monocular; see the module docstring.")
    args = ap.parse_args()

    est_t, est_xyz = read_tum(args.estimate)
    gt_t, gt_xyz = read_tum(args.truth)
    pairs = associate(est_t, gt_t)
    if len(pairs) < 10:
        raise SystemExit(f"only {len(pairs)} associated poses — check the timestamps")

    source = est_xyz[[i for i, _ in pairs]]
    target = gt_xyz[[j for _, j in pairs]]

    scale, R, t = umeyama(source, target, with_scale=args.sim3)
    aligned = scale * (source @ R.T) + t
    errors = np.linalg.norm(aligned - target, axis=1)

    mode = "Sim(3), 7-DoF, scale free" if args.sim3 else "SE(3), 6-DoF, scale FIXED"
    print(f"sequence      {args.truth.parent.name}")
    print(f"associated    {len(pairs)} / {len(est_t)} estimated poses")
    print(f"alignment     {mode}")
    if args.sim3:
        print(f"fitted scale  {scale:.6f}")
    print()
    print(f"ATE RMSE      {np.sqrt((errors ** 2).mean()) * 100:.2f} cm")
    print(f"ATE mean      {errors.mean() * 100:.2f} cm")
    print(f"ATE median    {np.median(errors) * 100:.2f} cm")
    print(f"ATE max       {errors.max() * 100:.2f} cm")
    print()
    print("Single run. ORB-SLAM3 is multithreaded and non-deterministic; the paper")
    print("reports a median over several runs, so treat this as one sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

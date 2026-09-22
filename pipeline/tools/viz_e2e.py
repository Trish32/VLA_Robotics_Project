#!/usr/bin/env python
"""Render what the end-to-end run actually produced.

Every panel is read from an artifact the run wrote — no illustrative data, no redrawn
diagrams. If a stage did not produce something, its panel says so.

    conda run -n foundationpose_vl python pipeline/tools/viz_e2e.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "pipeline/assets/e2e"
SEQ = ROOT / "orbslam3_baseline/data/rgbd_dataset_freiburg1_xyz"

# Colour-blind-safe, and deliberately not a rainbow: instances are categorical.
PALETTE = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f"]


def read_tum(path: Path):
    rows = [l.split() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return (np.array([float(r[0]) for r in rows]),
            np.array([[float(v) for v in r[1:4]] for r in rows]))


def umeyama_se3(src, dst):
    mu_s, mu_t = src.mean(0), dst.mean(0)
    C = (dst - mu_t).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    return R, mu_t - R @ mu_s


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.edgecolor": "#94a3b8"})

    # ---------------------------------------------------------- 1. trajectory
    est_t, est = read_tum(ROOT / "orbslam3_baseline/data/trajectory_orbslam3.txt")
    gt_t, gt = read_tum(SEQ / "groundtruth.txt")
    pairs = [(i, int(np.argmin(np.abs(gt_t - t)))) for i, t in enumerate(est_t)
             if abs(gt_t[np.argmin(np.abs(gt_t - t))] - t) <= 0.02]
    src = est[[i for i, _ in pairs]]
    dst = gt[[j for _, j in pairs]]
    R, t = umeyama_se3(src, dst)
    aligned = src @ R.T + t
    errors = np.linalg.norm(aligned - dst, axis=1) * 100

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), dpi=170)
    ax = axes[0]
    ax.plot(dst[:, 0], dst[:, 1], "-", lw=2.2, color="#94a3b8", label="ground truth")
    ax.plot(aligned[:, 0], aligned[:, 1], "-", lw=1.1, color="#be123c", label="ORB-SLAM3")
    ax.set_aspect("equal"); ax.legend(loc="best", framealpha=0.9)
    ax.set_title(f"1 · localization — ATE RMSE {np.sqrt((errors**2).mean()):.2f} cm")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(est_t[[i for i, _ in pairs]] - est_t[0], errors, lw=0.9, color="#0f766e")
    ax.axhline(np.sqrt((errors**2).mean()), ls="--", lw=1, color="#be123c",
               label=f"RMSE {np.sqrt((errors**2).mean()):.2f} cm")
    ax.set_title("1 · per-frame error"); ax.set_xlabel("time (s)")
    ax.set_ylabel("error (cm)"); ax.legend(); ax.grid(alpha=0.25)

    # ------------------------------------------------------ 2/3. cloud + labels
    cloud = o3d.io.read_point_cloud(str(OUT / "tum_fr1_xyz.ply"))
    pts = np.asarray(cloud.points)
    cols = np.asarray(cloud.colors)
    masks = np.load(OUT / "instance_masks.npz")["masks"]
    labelled = json.load(open(OUT / "labelled.json"))["instances"]

    ax = axes[2]
    step = max(1, len(pts) // 25000)
    ax.scatter(pts[::step, 0], pts[::step, 2], c=np.clip(cols[::step] * 1.4, 0, 1),
               s=1.2, marker=".", linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(f"2 · fusion — {len(pts):,} points from 80 posed frames")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_facecolor("#f1f5f9"); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "e2e_stage12.png", bbox_inches="tight")
    plt.close(fig)

    # instances, coloured by CLIP label
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=170)
    for ax, (a, b, an, bn) in zip(axes, [(0, 2, "x", "z"), (0, 1, "x", "y")]):
        ax.scatter(pts[::step, a], pts[::step, b], c="#cbd5e1", s=0.8,
                   marker=".", linewidths=0)
        for i, r in enumerate(labelled):
            if not r.get("label"):
                continue
            m = masks[i]
            ax.scatter(pts[m, a], pts[m, b], s=2.4, marker=".", linewidths=0,
                       color=PALETTE[i % len(PALETTE)],
                       label=f"{r['id']}: {r['label']} ({r['similarity']:.2f})")
        ax.set_aspect("equal"); ax.set_xlabel(f"{an} (m)"); ax.set_ylabel(f"{bn} (m)")
        ax.set_facecolor("#f8fafc"); ax.grid(alpha=0.2)
    axes[0].set_title("3 · Mask3D instances + CLIP labels (top-down)")
    axes[1].set_title("3 · same, front elevation")
    axes[1].legend(loc="upper right", fontsize=6, framealpha=0.92, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT / "e2e_stage3.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- 6. action chunk
    policy = json.load(open(OUT / "policy.json"))
    values = policy.get("values", {})
    if values:
        fig, axes = plt.subplots(1, len(values), figsize=(3.0 * len(values), 2.9), dpi=170)
        axes = np.atleast_1d(axes)
        for ax, (key, arr) in zip(axes, values.items()):
            a = np.asarray(arr)                      # (H, dof)
            for d in range(a.shape[1]):
                ax.plot(a[:, d], lw=1.1, alpha=0.85)
            ax.set_title(f"{key}  ({a.shape[0]}×{a.shape[1]})")
            ax.set_xlabel("step in chunk"); ax.grid(alpha=0.25)
            ax.set_facecolor("#f8fafc")
        axes[0].set_ylabel("commanded value")
        fig.suptitle(
            f"6 · GR00T N1.6-3B action chunk — H="
            f"{np.asarray(next(iter(values.values()))).shape[0]} steps × "
            f"{sum(np.asarray(v).shape[1] for v in values.values())} DoF, "
            f"{policy['seconds']}s on CPU", y=1.04)
        fig.tight_layout()
        fig.savefig(OUT / "e2e_stage6.png", bbox_inches="tight")
        plt.close(fig)

    summary = {
        "ate_rmse_cm": round(float(np.sqrt((errors ** 2).mean())), 2),
        "poses": len(est_t), "points": len(pts),
        "instances": int(sum(1 for r in labelled if r.get("label"))),
        "labels": [r["label"] for r in labelled if r.get("label")],
        "prompt": json.load(open(OUT / "ground.json"))["prompt"],
        # policy.json's `horizon` is taken from the BATCHED shape (1, 16, 7), so it
        # reads 1. The real chunk length is the unbatched first axis.
        "horizon": int(np.asarray(next(iter(values.values()))).shape[0]) if values
                   else policy["horizon"],
        "dof": int(sum(np.asarray(v).shape[1] for v in values.values())) if values
               else policy["width"],
        "policy_seconds": policy["seconds"],
    }
    (OUT / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

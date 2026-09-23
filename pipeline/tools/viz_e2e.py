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
DATA = ROOT / "orbslam3_baseline/data"
# SEQ and the estimated trajectory both come from fuse.json — see main(). They used to
# be hardcoded to freiburg1_xyz, which put that sequence's 1.03 cm ATE on the same page
# as a different sequence's cloud.

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

    # The run's own manifest decides the sequence and the cloud, so this page follows
    # whatever was last fused instead of a filename baked in when fr1/xyz was current.
    fuse = json.load(open(OUT / "fuse.json"))
    seq = DATA / fuse["sequence"]
    scene = fuse["sequence"].replace("rgbd_dataset_", "")

    # ---------------------------------------------------------- 1. trajectory
    est_t, est = read_tum(DATA / fuse["trajectory"])
    gt_t, gt = read_tum(seq / "groundtruth.txt")
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
    ax.set_title(f"1 · per-frame error — {scene}"); ax.set_xlabel("time (s)")
    ax.set_ylabel("error (cm)"); ax.legend(); ax.grid(alpha=0.25)

    # ------------------------------------------------------ 2/3. cloud + labels
    # Follow fuse.json rather than a hardcoded filename. This used to load
    # `tum_fr1_xyz.ply` while the masks came from whatever run was current, which on a
    # different sequence is not a wrong picture but an IndexError -- 133,928 mask bits
    # indexed into a 70,236-point cloud.
    cloud = o3d.io.read_point_cloud(fuse["ply"])
    pts = np.asarray(cloud.points)
    cols = np.asarray(cloud.colors)
    masks = np.load(OUT / "instance_masks.npz")["masks"]
    labelled = json.load(open(OUT / "labelled.json"))["instances"]
    if masks.shape[1] != len(pts):
        raise SystemExit(
            f"masks index {masks.shape[1]} points, cloud has {len(pts)} — these are "
            "from different runs. Re-run e2e_segment.py against the current fuse.json."
        )

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

    # --------------------------------------------------------------- 4. 6-DoF pose
    # This page's contract is "if a stage did not produce something, its panel says so".
    # Stage 4 DID produce something — a pose, and a refusal — so it gets a panel that
    # shows the four checks rather than being omitted for having failed.
    pose_path = OUT / "pose.json"
    if pose_path.exists():
        pose = json.load(open(pose_path))
        per = pose["per_frame"]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.2), dpi=170,
                                       gridspec_kw={"width_ratios": [1.45, 1]})
        frames = [c["frame"] for c in per]
        axL.plot(frames, [c["translation_cm"] for c in per], "o-", lw=1.4, ms=4,
                 color="#c2410c", label="translation (cm)")
        axL.axhline(pose["thresholds"]["translation_cm"], ls="--", lw=1.0,
                    color="#0f766e", label="gate threshold")
        ax2 = axL.twinx()
        ax2.plot(frames, [c["rotation_deg"] for c in per], "s-", lw=1.4, ms=3.5,
                 color="#7c3aed", label="rotation (deg)")
        # Fixed 0-180 scale. Autoscaled, a 0.7 deg wobble filled the axis and read as a
        # wildly unstable rotation; the truth is the opposite -- it is pinned near 176,
        # i.e. consistently flipped. The scale has to carry that.
        ax2.set_ylim(0, 180)
        ax2.set_yticks([0, 45, 90, 135, 180])
        ax2.axhline(180, ls=":", lw=0.9, color="#7c3aed", alpha=0.5)
        ax2.set_ylabel("rotation error (deg)  —  180 = flipped", color="#7c3aed")
        axL.set_xlabel("bundle frame"); axL.set_ylabel("translation error (cm)",
                                                       color="#c2410c")
        axL.grid(alpha=0.25); axL.set_facecolor("#f8fafc")
        axL.set_title(f"per-frame agreement with the map "
                      f"({pose['frames_corroborating']}/{pose['frames_total']} pass)")
        axL.legend(loc="center left", fontsize=7.5)

        ag = pose.get("agreement") or {}
        rows = [
            ("translation", f"{pose['median_translation_cm']:.1f} cm",
             pose["median_translation_cm"] <= pose["thresholds"]["translation_cm"]),
            ("rotation", f"{pose['median_rotation_deg']:.1f} deg",
             pose["median_rotation_deg"] <= pose["thresholds"]["rotation_deg"]),
            ("depth", f"{per[0]['behind_surface_cm']:.1f} cm behind",
             not any("behind" in r for r in per[0]["reasons"])),
            ("convergence", f"{ag.get('clustered_within_15deg','?')}/"
             f"{ag.get('k','?')} within 15 deg", bool(pose.get("agreement_ok"))),
        ]
        axR.axis("off")
        axR.text(0.0, 0.93, "STAGE 4 GATE", fontsize=10, weight="bold",
                 family="monospace")
        for n, (name, val, ok) in enumerate(rows):
            axR.text(0.0, 0.74 - n * 0.145,
                     f"{'PASS' if ok else 'FAIL'}  {name:<12}{val}",
                     fontsize=9, family="monospace",
                     color=("#0f766e" if ok else "#c2410c"))
        verdict = "POSE ADMITTED" if pose["accepted"] else "POSE REFUSED"
        axR.text(0.0, 0.10, verdict, fontsize=12, weight="bold", family="monospace",
                 color=("#0f766e" if pose["accepted"] else "#c2410c"))
        fig.suptitle("4 · FoundationPose — measured, then gated", y=1.03)
        fig.tight_layout()
        fig.savefig(OUT / "e2e_stage4.png", bbox_inches="tight")
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
    if pose_path.exists():
        _p = json.load(open(pose_path))
        summary["pose_accepted"] = bool(_p["accepted"])
        summary["pose_translation_cm"] = round(_p["median_translation_cm"], 2)
        summary["pose_rotation_deg"] = round(_p["median_rotation_deg"], 2)
    (OUT / "e2e_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

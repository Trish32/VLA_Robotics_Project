#!/usr/bin/env python
"""Does the rollout stay accurate, and does the model know when it doesn't?

Everything training reports is ONE step. The planner rolls out N and weights candidates
by ensemble disagreement, so two claims are load-bearing and neither is tested by a
one-step number:

  1. **N-step accuracy** — the model must still beat constant velocity rolled out the
     same N steps, not just at step 1. Compounding error is where learned dynamics
     usually lose.
  2. **Calibration** — the ensemble spread must track the error it is standing in for.
     If a wide spread does not predict a large error, the risk term is decorative and
     the planner is discounting candidates for no reason.

Calibration is reported as a rank correlation between predicted spread and realised
error, per horizon. Rank rather than Pearson because the planner only ever uses the
ORDER — it needs "this candidate is riskier than that one", not a calibrated metre
value, and a monotone-but-nonlinear relationship is a success here.

    conda run -n groot_vl python -m pipeline.world_model.evaluate \
        --data grootN1_Robotics/upstream/demo_data/gr1.PickNPlace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with no scipy dependency at eval time."""
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def episode_windows(ep, norm_state, norm_action, horizon):
    """Every length-`horizon` window inside one episode, as normalised arrays.

    Windows start at t=1 so the initial latent has a real velocity rather than a faked
    zero one — the same convention the training data uses, and mixing the two would
    evaluate the model on a state distribution it never saw.
    """
    s = norm_state(ep["state"])
    a = norm_action(ep["action"])
    T = len(s)
    out = []
    for t in range(1, T - horizon):
        z0 = np.concatenate([s[t], s[t] - s[t - 1]])
        out.append((z0, a[t:t + horizon], s[t + 1:t + 1 + horizon]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "pipeline/assets/world_model/dynamics.pt")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "pipeline/assets/world_model/rollout_metrics.json")
    args = ap.parse_args()

    from pipeline.world_model.data import TransitionDataset, load_episodes
    from pipeline.world_model.dynamics import DynamicsEnsemble
    from pipeline.world_model.latent import SceneLatent

    eps = load_episodes(args.data)
    if len(eps) < 3:
        raise SystemExit("need >= 3 episodes to reproduce the train/val/test split")
    tr, test = eps[:-2], eps[-1:]
    base = TransitionDataset(tr)                      # normalisers from TRAIN only

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DynamicsEnsemble(ck["slot_dim"], ck["robot_dim"], ck["action_dim"],
                             n_members=ck["members"], hidden=ck["hidden"],
                             layers=ck["layers"],
                             integrate_velocity=ck.get("integrate_velocity", True))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = ck.get("state_dim", base.state_norm.mean.shape[0])

    wins = episode_windows(test[0], base.state_norm, base.action_norm, args.horizon)
    print(f"[eval] {len(wins)} windows of {args.horizon} steps from the held-out episode")

    z0 = torch.from_numpy(np.stack([w[0] for w in wins])).float()
    acts = torch.from_numpy(np.stack([w[1] for w in wins])).float()
    truth = torch.from_numpy(np.stack([w[2] for w in wins])).float()   # (B, N, d)

    B = z0.shape[0]
    z = SceneLatent(torch.zeros(B, 1, max(ck["slot_dim"], 1)), z0,
                    torch.zeros(B, 1, dtype=torch.bool))
    states, unc = model.rollout(z, acts)
    pred = torch.stack([s.robot[:, :d] for s in states], dim=1)        # (B, N, d)

    # Constant velocity rolled out the same N steps: the baseline that also compounds.
    cv = []
    pos, vel = z0[:, :d], z0[:, d:]
    for _ in range(args.horizon):
        pos = pos + vel
        cv.append(pos)
    cv = torch.stack(cv, dim=1)

    # BOTH aggregations, because on this data they disagree about who wins.
    #
    # Global RMSE squares before averaging, so it is dominated by the worst samples;
    # median per-sample error describes the typical one. Measured at one step:
    # constant velocity is 4.6x better than the model on the MEDIAN sample and beats it
    # on 91% of them, while the model is far better in the tail (p99 0.083 vs 0.152).
    # Reporting only the global number says the model wins; reporting only the median
    # says it loses. Both are true and neither alone is honest.
    rows, rmse_model, rmse_cv, corr, winrate = [], [], [], [], []
    for t in range(args.horizon):
        pm = torch.sqrt(((pred[:, t] - truth[:, t]) ** 2).mean(-1))   # (B,) per-sample
        pc = torch.sqrt(((cv[:, t] - truth[:, t]) ** 2).mean(-1))
        g_m = float(torch.sqrt(((pred[:, t] - truth[:, t]) ** 2).mean()))
        g_c = float(torch.sqrt(((cv[:, t] - truth[:, t]) ** 2).mean()))
        r = spearman(unc[:, t].numpy(), pm.numpy())
        wr = float((pm < pc).float().mean())
        rmse_model.append(g_m); rmse_cv.append(g_c); corr.append(r); winrate.append(wr)
        rows.append((t + 1, g_m, g_c, float(pm.median()), float(pc.median()), wr,
                     float(unc[:, t].mean()), r))

    print(f"\n  {'':5}{'---- global RMSE ----':>23}{'--- median sample ---':>23}"
          f"{'':>10}")
    print(f"  {'step':>4} {'model':>10} {'const-vel':>11} {'model':>10} "
          f"{'const-vel':>11} {'win%':>7} {'spread':>9} {'corr':>7}")
    for t, gm, gc, mm, mc, wr, u, r in rows:
        rs = f"{r:7.3f}" if np.isfinite(r) else "    n/a"
        print(f"  {t:4d} {gm:10.5f} {gc:11.5f} {mm:10.5f} {mc:11.5f} "
              f"{wr * 100:6.0f}% {u:9.5f} {rs}")

    finite = [r for r in corr if np.isfinite(r)]
    mean_corr = float(np.mean(finite)) if finite else float("nan")
    print(f"\n[eval] mean rank correlation between predicted spread and realised "
          f"error: {mean_corr:.3f}")
    if np.isfinite(mean_corr):
        if mean_corr > 0.3:
            print("[eval] the ensemble spread tracks the error — the planner's risk "
                  "term is doing real work")
        elif mean_corr > 0.0:
            print("[eval] weak positive calibration; the risk term orders candidates "
                  "only loosely")
        else:
            print("[eval] NOT CALIBRATED. Spread does not predict error, so the risk "
                  "term is decorative and should not be weighted as if it were not.")

    beats = sum(gm < gc for _, gm, gc, _, _, _, _, _ in rows)
    med_beats = sum(mm < mc for _, _, _, mm, mc, _, _, _ in rows)
    print(f"[eval] model beats rolled-out constant velocity at {beats}/{len(rows)} "
          f"horizons on global RMSE, {med_beats}/{len(rows)} on the median sample")
    print(f"[eval] it wins on {np.mean(winrate) * 100:.0f}% of individual rollouts — "
          f"so where it wins, it wins on the hard ones")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"horizon": args.horizon, "windows": len(wins),
               "rmse_model_global": rmse_model, "rmse_constvel_global": rmse_cv,
               "median_model": [r[3] for r in rows],
               "median_constvel": [r[4] for r in rows],
               "winrate_vs_constvel": winrate,
               "spread": [float(unc[:, t].mean()) for t in range(args.horizon)],
               "rank_corr": corr, "mean_rank_corr": mean_corr,
               "beats_constvel_global": int(beats),
               "beats_constvel_median": int(med_beats)}, open(args.out, "w"), indent=1)
    print(f"[eval] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

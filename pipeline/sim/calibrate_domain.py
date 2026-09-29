#!/usr/bin/env python
"""Where does the scorer stop being right? Read the boundary off a curve.

`ranking_metrics.py` established that the model orders affordance candidates well along
the policy's own trajectory (Kendall tau +0.291) and **inverts** along its own
(-0.157). That is a statement about two whole regimes. A gate needs the boundary
between them, per decision, from quantities available without ground truth.

So: sample decisions, record the three candidate signals (state energy, ensemble
spread, per-candidate action deviation), and record the ground truth the gate is not
allowed to see — the per-decision Kendall tau against rewind-and-execute cost, and the
per-candidate error of the scorer. Then report tau **conditioned on** each signal, which
is the curve a threshold is read off.

Driven with `--drive best` on purpose: gating is only interesting where the ranking is
being acted on, and that is the regime where it was measured to invert.

    conda run -n simple_bev_vldrive python -m pipeline.sim.calibrate_domain \
        --candidates-from affordance --drive best --episodes 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _bins(x: np.ndarray, y: np.ndarray, edges: np.ndarray, label: str,
          value: str = "tau") -> None:
    print(f"\n      {label:>22}     n   mean {value}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            print(f"      {lo:9.3f}-{hi:<9.3f} {int(m.sum()):5d}   {y[m].mean():+.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("pipeline/assets/world_model_sim/dynamics.pt"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--candidates-from", default="affordance",
                    choices=("action", "belief", "affordance"))
    ap.add_argument("--drive", default="best", choices=("first", "best"))
    ap.add_argument("--policy-noise", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from pipeline.sim.domain import action_deviation, state_energy
    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.loop import SLOT_ORDER, ClosedLoop, LoopConfig, load_dynamics
    from pipeline.sim.perception import OraclePerception
    from pipeline.sim.ranking_metrics import kendall_tau

    dyn, norms, n_slots, geom = load_dynamics(a.checkpoint)
    cfg = LoopConfig(candidates=a.candidates, candidates_from=a.candidates_from,
                     policy_noise=a.policy_noise, plan=False)
    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(), max_steps=360)
    loop = ClosedLoop(env, OraclePerception(), dyn, norms, cfg=cfg,
                      n_slots=n_slots, slot_geom=geom)

    rows, cands = [], []
    for ep in range(a.episodes):
        obs = env.reset()
        driver = loop.make_driver(a.seed * 1000 + ep)
        boxes, ok = loop._slots(loop.perception(env.render(cfg.camera)))
        prev_boxes, prev_state = boxes.copy(), obs.state.copy()
        due = np.linspace(0, env.max_steps, a.probes + 2)[1:-1]
        rng = np.random.default_rng(a.seed * 31 + ep)
        while not env.done():
            nb, nok = loop._slots(loop.perception(env.render(cfg.camera)))
            boxes = np.where(nok[:, None], nb, boxes)
            ok = nok | ok
            seen = {lab: boxes[i, :3] for i, lab in
                    enumerate(SLOT_ORDER[:n_slots]) if ok[i]}
            step_chunk = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
            if ok[0] and ok[1]:
                raw = loop._candidates(driver, obs.tip, seen, obs.state, rng, boxes)
                z0 = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                scores, _ = loop._score(raw, z0, boxes[1, :3])
                pred = scores.total.detach().numpy().astype(float)
                if any(abs(obs.step - d) < 2 for d in due):
                    true = loop._oracle_costs(raw, env.body_pos("ball").copy())
                    dev = action_deviation(raw, raw[0], norms.action_std)
                    # Scale-matched residual: the scorer's units are arbitrary, so a
                    # raw difference would measure the units gap, not the skill.
                    if np.std(pred) > 0:
                        fit = np.polyfit(pred, true, 1)
                        err = np.abs(np.polyval(fit, pred) - true)
                    else:
                        err = np.abs(true - true.mean())
                    rows.append({"episode": ep, "step": int(obs.step),
                                 "tau": kendall_tau(pred, true),
                                 "energy": state_energy(z0),
                                 "spread": loop._last_spread,
                                 "max_dev": float(dev.max()),
                                 "mean_dev": float(dev.mean()),
                                 # Kept so the threshold sweep can re-rank the
                                 # SURVIVING subset offline. Sweeping a filter without
                                 # re-scoring what it leaves behind measures the
                                 # decisions it drops, not the ranking it produces.
                                 "pred": [float(v) for v in pred],
                                 "true": [float(v) for v in true],
                                 "dev": [float(v) for v in dev]})
                    for k in range(len(raw)):
                        cands.append({"dev": float(dev[k]), "err": float(err[k])})
                if a.drive == "best":
                    step_chunk = raw[int(np.argmin(pred))]
            driver.advance(obs.tip, seen)
            prev_boxes, prev_state = boxes.copy(), obs.state.copy()
            obs = env.step(step_chunk[0])
        print(f"[dom] ep{ep:03d} {len(rows)} decisions", flush=True)
    env.close()

    tau = np.array([r["tau"] for r in rows])
    print(f"\n[dom] {len(rows)} decisions, mean Kendall tau {tau.mean():+.3f}")
    for key, name in (("energy", "state energy"), ("spread", "ensemble spread"),
                      ("max_dev", "max action deviation")):
        v = np.array([r[key] for r in rows])
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 7)))
        _bins(v, tau, edges, name)
        # The decision a threshold actually makes: what is the tau of what survives,
        # and how much is thrown away to get it?
        print(f"      {'keep ' + key + ' <=':>22}  kept   mean tau   (vs {tau.mean():+.3f} ungated)")
        for q in (0.3, 0.5, 0.7, 0.9):
            thr = float(np.quantile(v, q))
            m = v <= thr
            print(f"      {thr:21.3f} {m.mean()*100:5.0f}%   {tau[m].mean():+.3f}")

    print(f"\n[dom] ranking of what SURVIVES a per-candidate deviation filter")
    print(f"      {'keep dev <=':>12} {'decisions kept':>15} {'cands kept':>11} "
          f"{'mean tau':>9}   (vs {tau.mean():+.3f} ungated)")
    alldev = np.concatenate([np.asarray(r["dev"]) for r in rows])
    for thr in np.unique(np.round(np.quantile(alldev, [0.3, 0.5, 0.7, 0.9, 1.0]), 3)):
        taus, kept_c, kept_d = [], 0, 0
        for r in rows:
            dv = np.asarray(r["dev"])
            keep = dv <= thr
            keep[int(np.argmin(dv))] = True     # the policy's own chunk always stays
            kept_c += int(keep.sum())
            if keep.sum() >= 2:
                kept_d += 1
                taus.append(kendall_tau(np.asarray(r["pred"])[keep],
                                        np.asarray(r["true"])[keep]))
        m = float(np.mean(taus)) if taus else float("nan")
        print(f"      {thr:12.3f} {kept_d}/{len(rows):<14d} "
              f"{kept_c}/{len(alldev):<10d} {m:+9.3f}")

    d = np.array([c["dev"] for c in cands])
    e = np.array([c["err"] for c in cands])
    if len(d) > 2 and np.std(d) > 0:
        r = float(np.corrcoef(d, e)[0, 1])
        print(f"\n[dom] per-candidate: correlation between action deviation and the "
              f"scorer's error  r = {r:+.3f}")
        edges = np.unique(np.quantile(d, np.linspace(0, 1, 7)))
        _bins(d, e, edges, "action deviation", value="err")
    if a.json:
        a.json.write_text(json.dumps({"decisions": rows, "candidates": cands},
                                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

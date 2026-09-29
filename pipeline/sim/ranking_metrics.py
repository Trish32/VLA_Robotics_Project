#!/usr/bin/env python
"""Whether a scorer can rank candidates, measured properly rather than by success rate.

Success rate is a blunt instrument for this question: a planner can rank well and still
lose (if the candidates are equivalent, or if acting on the ranking compounds), and it
can rank at chance and still win (if candidate 0 happens to be good). These metrics
separate the ranking itself from what acting on it does.

  * **pairwise accuracy** — over all candidate pairs, how often the predicted order
    matches the true one. 0.5 is chance, and unlike a success rate it is not diluted by
    decisions where the choice did not matter.
  * **Kendall tau / Spearman rho** — rank correlation per decision, averaged. Tau is
    reported because it is the one with a direct reading: (tau + 1) / 2 is the pairwise
    accuracy, so a tau of 0 and an accuracy of 0.5 are the same statement twice.
  * **regret@k** — the true cost given up by taking the model's top-k and keeping the
    best of them, against the true best. This is what a planner actually costs you, and
    it falls to zero as k reaches the candidate count by construction.
  * **signal-to-noise** — the decisive one. Compares how much the true cost *varies
    across candidates* (the signal a scorer would have to resolve) against how far the
    model's predicted cost sits from the true cost (its own error). **Below 1, ranking
    is arithmetically impossible**, and no scorer, objective or proposer changes that.

Truth comes from rewinding the simulator and executing each candidate, so "true cost"
is measured rather than modelled.

    conda run -n simple_bev_vldrive python -m pipeline.sim.ranking_metrics \
        --episodes 12 --candidates-from affordance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Ties are the normal case for near-equal candidates,
    and breaking them by sort order invents an ordering the data does not contain —
    which is how a constant signal once scored AUC 0.764 here (bug_log [S14])."""
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pairwise_accuracy(pred: np.ndarray, true: np.ndarray) -> tuple[float, int]:
    """Fraction of comparable pairs ordered correctly, and how many pairs there were.

    Pairs that are tied in TRUTH are excluded rather than counted as wins or losses:
    when two candidates really do lead to the same place, there is no correct answer
    and scoring it either way measures the tie-breaking, not the ranking.
    """
    n = len(pred)
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true[i] == true[j]:
                continue
            tot += 1
            ok += int((pred[i] < pred[j]) == (true[i] < true[j]))
    return (ok / tot if tot else float("nan")), tot


def kendall_tau(pred: np.ndarray, true: np.ndarray) -> float:
    acc, tot = pairwise_accuracy(pred, true)
    return float("nan") if not tot else 2 * acc - 1


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    a, b = _ranks(pred), _ranks(true)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regret_at_k(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    """True cost of the best of the model's top-k, above the true best."""
    top = np.argsort(pred, kind="mergesort")[:k]
    return float(true[top].min() - true.min())


def summarise(decisions: list[dict]) -> dict:
    """Aggregate per-decision metrics, and the signal-to-noise verdict."""
    acc = [d["pairwise"] for d in decisions if np.isfinite(d["pairwise"])]
    tau = [d["tau"] for d in decisions if np.isfinite(d["tau"])]
    rho = [d["spearman"] for d in decisions if np.isfinite(d["spearman"])]
    ks = sorted({k for d in decisions for k in d["regret"]})
    regret = {k: float(np.mean([d["regret"][k] for d in decisions])) for k in ks}
    signal = float(np.mean([d["signal"] for d in decisions]))
    noise = float(np.mean([d["noise"] for d in decisions]))
    return {"decisions": len(decisions),
            "pairwise_accuracy": float(np.mean(acc)) if acc else None,
            "kendall_tau": float(np.mean(tau)) if tau else None,
            "spearman": float(np.mean(rho)) if rho else None,
            "regret_at_k": regret,
            "signal_std_true_cost": signal,
            "noise_rmse_pred_vs_true": noise,
            "snr": signal / noise if noise > 0 else float("inf")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("pipeline/assets/world_model_sim/dynamics.pt"))
    ap.add_argument("--value", type=Path, default=None)
    ap.add_argument("--also-score", type=Path, default=None,
                    help="a second dynamics checkpoint, scored at the SAME decisions "
                         "on the SAME candidates. Without it, two models compared "
                         "under --drive best are also being asked about different "
                         "states, because each one steered itself there.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8,
                    help="chunk length. The true cost is measured over the whole "
                         "chunk, so this is what 'chunk-level' means here")
    ap.add_argument("--execute", type=int, default=None,
                    help="steps of the chosen chunk to run before re-deciding "
                         "(default: the whole chunk). Only meaningful with "
                         "--drive best; ranking at K=32 while re-deciding every step "
                         "would measure a commitment the metric does not use")
    ap.add_argument("--candidates-from", default="affordance",
                    choices=("action", "belief", "affordance"),
                    help="affordance is the benchmark of record. 'action' "
                         "(perturbation) is INVALID as a ranking test — Oracle@k "
                         "headroom +0.0 — and is kept runnable only as the evidence "
                         "for its own invalidity")
    ap.add_argument("--drive", default="first", choices=("first", "best"),
                    help="whose choices the episode FOLLOWS while metrics are "
                         "collected. 'first' measures ranking along the policy's own "
                         "trajectory; 'best' measures it along the trajectory the "
                         "scorer's own choices produce. If the two disagree, the "
                         "scorer is accurate on-distribution and not off it.")
    ap.add_argument("--policy-noise", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.loop import SLOT_ORDER, ClosedLoop, LoopConfig, load_dynamics
    from pipeline.sim.perception import OraclePerception

    dyn, norms, n_slots, geom = load_dynamics(a.checkpoint)
    execute = a.execute if a.execute is not None else a.horizon
    cfg = LoopConfig(candidates=a.candidates, candidates_from=a.candidates_from,
                     policy_noise=a.policy_noise, plan=False,
                     horizon=a.horizon, execute=execute)
    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(), max_steps=360)
    value = None
    if a.value:
        from pipeline.sim.loop import load_value
        value = load_value(a.value)
    loop = ClosedLoop(env, OraclePerception(), dyn, norms, cfg=cfg, value=value,
                      n_slots=n_slots, slot_geom=geom)
    loop_b = None
    if a.also_score:
        dyn_b, norms_b, nb_slots, geom_b = load_dynamics(a.also_score)
        if (nb_slots, geom_b) != (n_slots, geom):
            raise SystemExit("--also-score must have the same slot layout")
        # Shares the env deliberately: it never steps it, it only scores. The second
        # model's own normaliser is used, so this compares models and not scalings.
        loop_b = ClosedLoop(env, OraclePerception(), dyn_b, norms_b, cfg=cfg,
                            value=value, n_slots=nb_slots, slot_geom=geom_b)

    decisions = []
    decisions_b = []
    for ep in range(a.episodes):
        obs = env.reset()
        driver = loop.make_driver(a.seed * 1000 + ep)
        boxes, ok = loop._slots(loop.perception(env.render(cfg.camera)))
        prev_boxes, prev_state = boxes.copy(), obs.state.copy()
        due = np.linspace(0, env.max_steps, a.probes + 2)[1:-1]
        rng = np.random.default_rng(a.seed * 31 + ep)
        pending: list = []
        while not env.done():
            nb, nok = loop._slots(loop.perception(env.render(cfg.camera)))
            boxes = np.where(nok[:, None], nb, boxes)
            ok = nok | ok
            seen = {lab: boxes[i, :3] for i, lab in
                    enumerate(SLOT_ORDER[:n_slots]) if ok[i]}
            if ok[0] and ok[1] and any(abs(obs.step - d) < 2 for d in due):
                raw = loop._candidates(driver, obs.tip, seen, obs.state, rng, boxes)
                z0 = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                scores, _ = loop._score(raw, z0, boxes[1, :3])
                pred = scores.total.detach().numpy().astype(float)
                true = loop._oracle_costs(raw, env.body_pos("ball").copy())
                acc, npair = pairwise_accuracy(pred, true)
                # Predicted and true costs live on different scales, so the model's
                # error is measured after matching scale — otherwise the SNR would
                # report the units gap rather than the model's skill.
                if np.std(pred) > 0:
                    fit = np.polyfit(pred, true, 1)
                    noise = float(np.sqrt(np.mean((np.polyval(fit, pred) - true) ** 2)))
                else:
                    noise = float(np.std(true))
                decisions.append({
                    "episode": ep, "step": int(obs.step), "pairs": npair,
                    "pairwise": acc, "tau": kendall_tau(pred, true),
                    "spearman": spearman(pred, true),
                    "regret": {k: regret_at_k(pred, true, k) for k in (1, 2, 3)},
                    "signal": float(np.std(true)), "noise": noise})
                if loop_b is not None:
                    zb = loop_b._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                    sb, _ = loop_b._score(raw, zb, boxes[1, :3])
                    pb = sb.total.detach().numpy().astype(float)
                    accb, npb = pairwise_accuracy(pb, true)
                    if np.std(pb) > 0:
                        fb = np.polyfit(pb, true, 1)
                        nb_ = float(np.sqrt(np.mean((np.polyval(fb, pb) - true) ** 2)))
                    else:
                        nb_ = float(np.std(true))
                    decisions_b.append({
                        "episode": ep, "step": int(obs.step), "pairs": npb,
                        "pairwise": accb, "tau": kendall_tau(pb, true),
                        "spearman": spearman(pb, true),
                        "regret": {k: regret_at_k(pb, true, k) for k in (1, 2, 3)},
                        "signal": float(np.std(true)), "noise": nb_})
            # A chunk is committed to for `execute` steps and only then re-decided.
            # Re-deciding every step regardless of the chunk length would measure the
            # ranking of 32-step plans under a 1-step commitment, which is not the
            # thing being asked about.
            if not pending:
                step_chunk = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
                if a.drive == "best" and ok[0] and ok[1]:
                    cand = loop._candidates(driver, obs.tip, seen, obs.state, rng,
                                            boxes)
                    z = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                    sc, _ = loop._score(cand, z, boxes[1, :3])
                    step_chunk = cand[int(torch.argmin(sc.total).item())]
                pending = list(step_chunk[:cfg.execute])
            driver.advance(obs.tip, seen)
            prev_boxes, prev_state = boxes.copy(), obs.state.copy()
            obs = env.step(pending.pop(0))
        print(f"[rank] ep{ep:03d} {len(decisions)} decisions", flush=True)
    env.close()

    out = summarise(decisions)
    if a.candidates_from == "action":
        print("\n[rank] WARNING: perturbation candidates are an INVALID ranking test.\n"
              "       Oracle@k headroom is +0.0 — every candidate leads to the same\n"
              "       outcome — so a perfect ranker and a coin score the same here, and\n"
              "       a result on this set is a property of the test. Kept runnable\n"
              "       because it is the evidence for its own invalidity. Use\n"
              "       --candidates-from affordance for the ranking benchmark of record.")
    print(f"\n[rank] {a.candidates_from} candidates, chunk length {a.horizon}, "
          f"commitment {execute}, {out['decisions']} decisions")
    print(f"       pairwise accuracy  {out['pairwise_accuracy']:.3f}   (0.5 = chance)")
    print(f"       Kendall tau        {out['kendall_tau']:+.3f}")
    print(f"       Spearman rho       {out['spearman']:+.3f}")
    for k, v in out["regret_at_k"].items():
        print(f"       regret@{k}           {v:.5f}")
    print(f"\n       signal (std of true cost across candidates) {out['signal_std_true_cost']:.5f}")
    print(f"       noise  (model's cost error, scale-matched)   {out['noise_rmse_pred_vs_true']:.5f}")
    snr = out["snr"]
    verdict = ("ranking is arithmetically impossible — the model cannot resolve "
               "differences smaller than its own error" if snr < 1.0 else
               "the differences are barely larger than the model's error; ranking is "
               "possible in principle and marginal in practice" if snr < 1.5 else
               "the differences clear the model's error with room to spare"
               if snr < 2.5 else
               "the differences dominate the model's error — ranking is a fair ask")
    print(f"       SNR {snr:.3f} — {verdict}")
    out_b = None
    if decisions_b:
        out_b = summarise(decisions_b)
        print(f"\n[rank] --also-score {a.also_score} — same decisions, same "
              f"candidates, same truth")
        print(f"       pairwise accuracy  {out_b['pairwise_accuracy']:.3f}   "
              f"(driving model: {out['pairwise_accuracy']:.3f})")
        print(f"       Kendall tau        {out_b['kendall_tau']:+.3f}   "
              f"(driving model: {out['kendall_tau']:+.3f})")
        print(f"       regret@1           {out_b['regret_at_k'][1]:.5f}   "
              f"(driving model: {out['regret_at_k'][1]:.5f})")
    if a.json:
        a.json.write_text(json.dumps({"summary": out, "decisions": decisions,
                                      "also_score": {"checkpoint": str(a.also_score),
                                                     "summary": out_b,
                                                     "decisions": decisions_b}
                                      if out_b else None},
                                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

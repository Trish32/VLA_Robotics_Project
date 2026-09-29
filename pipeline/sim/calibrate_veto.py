#!/usr/bin/env python
"""Measure what a veto threshold would actually buy, before choosing one.

A refusal threshold picked by eye is a free parameter that can be tuned until the
result looks good. This collects the two things the gate sees — predicted return drop
and ensemble spread — alongside the ground truth it cannot see, namely whether that
action really did lead to failure, and then reports precision and recall across the
whole sweep. The threshold is then read off a curve rather than guessed.

Ground truth comes from rewinding the simulator: execute the chunk, run the policy to
the end, record success, rewind. That is the same primitive the episode-depth oracle
uses, and it is exact rather than a proxy.

    conda run -n simple_bev_vldrive python -m pipeline.sim.calibrate_veto \
        --dynamics pipeline/assets/world_model_plans/dynamics.pt \
        --value pipeline/assets/world_model_plans/value.pt --episodes 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dynamics", type=Path,
                    default=Path("pipeline/assets/world_model_plans/dynamics.pt"))
    ap.add_argument("--value", type=Path,
                    default=Path("pipeline/assets/world_model_plans/value.pt"))
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--policy-noise", type=float, default=0.006)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", default="spread",
                    choices=("drop", "v_end", "collision", "instability", "spread",
                             "action_dev", "fail_logit"),
                    help="which predicted quantity the threshold sweep uses")
    ap.add_argument("--recoverability", type=int, default=0, metavar="K",
                    help="at each probed decision, roll K affordance candidates to "
                         "completion and count how many succeed. This is what "
                         "separates 'this ACTION was doomed' from 'this STATE was "
                         "already lost', which the failure label alone cannot")
    ap.add_argument("--failure-head", type=Path, default=None,
                    help="a head trained on the FAILURE label by probe_value.py. "
                         "Scored here because this is the only place a per-decision "
                         "label exists: the episode-level label it was fitted on is "
                         "constant within an episode and cannot be scored per decision")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.loop import (ClosedLoop, LoopConfig, load_dynamics,
                                   load_value)
    from pipeline.sim.perception import OraclePerception
    from pipeline.sim.policy import policy_observation
    from pipeline.world_model.scorer import (pairwise_overlap_volume,
                                             unsupported_drop)

    dyn, norms, n_slots, geom = load_dynamics(a.dynamics)
    value = load_value(a.value)
    fail_head = None
    if a.failure_head:
        from pipeline.world_model.scorer import ValueHead
        fck = torch.load(a.failure_head, map_location="cpu", weights_only=False)
        fail_head = ValueHead(fck["slot_dim"], fck["robot_dim"], hidden=fck["hidden"])
        fail_head.load_state_dict(fck["state_dict"])
        fail_head.eval()
    cfg = LoopConfig(policy_noise=a.policy_noise, plan=False)
    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(), max_steps=360)
    loop = ClosedLoop(env, OraclePerception(), dyn, norms, cfg=cfg, value=value,
                      n_slots=n_slots, slot_geom=geom)

    rows = []
    for ep in range(a.episodes):
        obs = env.reset()
        driver = loop.make_driver(a.seed * 1000 + ep)
        frame = env.render(cfg.camera)
        boxes, ok = loop._slots(loop.perception(frame))
        prev_boxes, prev_state = boxes.copy(), obs.state.copy()
        due = set(np.linspace(0, env.max_steps, a.probes + 2)[1:-1].astype(int))
        while not env.done():
            nb, nok = loop._slots(loop.perception(env.render(cfg.camera)))
            boxes = np.where(nok[:, None], nb, boxes)
            ok = nok | ok
            seen = {lab: boxes[i, :3] for i, lab in
                    enumerate(("a small dark cube", "an orange plastic bowl",
                               "a green rubber ball")[:n_slots]) if ok[i]}
            chunk = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
            if ok[0] and ok[1] and any(abs(obs.step - d) < 2 for d in due):
                z0 = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                norm = torch.from_numpy(
                    (chunk - norms.action_mean) / norms.action_std).float()
                with torch.no_grad():
                    states, unc = dyn.rollout(z0, norm.unsqueeze(0))
                    v0 = float(value(z0))
                    vT = float(value(states[-1]))
                    metric = loop._to_metres(states)
                    coll = float(sum(float(pairwise_overlap_volume(z))
                                     for z in metric))
                    inst = float(sum(float(unsupported_drop(z)) for z in metric))
                spread = float(unc.max()) if unc.numel() else 0.0
                # How far the chunk about to be executed sits from the action
                # distribution the dynamics was fitted on. Unlike the per-candidate
                # deviation the domain gate uses, this has no reference chunk to
                # subtract — the policy's own action IS the thing being judged — so it
                # is measured against the training mean.
                act_dev = float(np.sqrt(np.mean(norm.numpy() ** 2)))
                fail_logit = (float(fail_head(z0)) if fail_head is not None
                              else 0.0)
                drop = v0 - vT
                # Ground truth: what this action actually leads to, and what holding
                # instead would lead to. Both, because a veto is only useful when the
                # substitute is better — not merely when the action was doomed.
                failed = not loop._continue_to_end(driver, chunk)
                from pipeline.sim.veto import hold_chunk
                held_ok = loop._continue_to_end(
                    driver, hold_chunk(cfg.horizon, obs.grip))
                # Recoverability: of every plan available here, how many still reach
                # success? A decision where the answer is zero carries no information
                # about the action — the state was lost before it was taken — and
                # counting it as a "doomed action" is the labelling trap this removes.
                recoverable = -1
                plan_mean = plan_std = float("nan")
                plan_vals = plan_vals_short = []
                if a.recoverability:
                    loop.cfg.candidates = a.recoverability
                    loop.cfg.candidates_from = "affordance"
                    cands = loop._candidates(driver, obs.tip, seen, obs.state,
                                             np.random.default_rng(ep * 977 + obs.step),
                                             boxes)
                    recoverable = int(sum(
                        loop._continue_to_end(driver, c) for c in cands))
                    # A run-time proxy for the same quantity, at the cost of ONE
                    # batched forward pass: roll every candidate through the learned
                    # dynamics and read the value head at the end. If the plans agree,
                    # the decision is probably not pivotal — either all of them work or
                    # none do; if they disagree, the choice may matter. No simulator,
                    # no rewind, so this is available to a deployed gate.
                    nc = torch.from_numpy(
                        (cands - norms.action_mean) / norms.action_std).float()
                    with torch.no_grad():
                        st, _ = dyn.rollout(z0.expand_batch(len(cands)), nc)
                        vals = value(st[-1]).reshape(-1).numpy()
                        # The same read at a TRUNCATED horizon. A deployed gate pays
                        # for the rollout it runs, so whether the signal survives being
                        # cut to a quarter of the chunk decides if it is affordable.
                        ns = max(2, nc.shape[1] // 4)
                        st_s, _ = dyn.rollout(z0.expand_batch(len(cands)),
                                              nc[:, :ns])
                        vals_short = value(st_s[-1]).reshape(-1).numpy()
                    plan_mean = float(vals.mean())
                    plan_std = float(vals.std())
                    plan_vals = [float(v) for v in vals]
                    plan_vals_short = [float(v) for v in vals_short]
                rows.append({"episode": ep, "step": int(obs.step),
                             # The waypoint the demonstrator is in. Recorded so a veto
                             # can be evaluated at PHASE granularity as well as
                             # per-decision: the decisive fraction is a property of the
                             # unit you ask at, and a step index cannot recover the
                             # stage boundaries because they move with the episode.
                             "stage": getattr(getattr(driver, "pol", None),
                                              "stage_name", "?"),
                             "drop": drop,
                             "v_end": vT, "collision": coll, "instability": inst,
                             "spread": spread, "action_dev": act_dev,
                             "fail_logit": fail_logit, "recoverable": recoverable,
                             "plan_mean": plan_mean, "plan_std": plan_std,
                             "plan_vals": plan_vals,
                             "plan_vals_short": plan_vals_short,
                             # What a gate can read off the observation with no
                             # rollout at all. The cheapest of the three families and
                             # the floor the other two have to beat.
                             "state_feat": [float(x) for x in policy_observation(
                                 obs.state, obs.tip, seen)],
                             "v0": v0,
                             "failed": bool(failed), "hold_ok": bool(held_ok)})
            driver.advance(obs.tip, seen)
            prev_boxes, prev_state = boxes.copy(), obs.state.copy()
            obs = env.step(chunk[0])
        print(f"[cal] ep{ep:03d} {len(rows)} samples", flush=True)
    env.close()

    d = np.array([r["drop"] for r in rows])
    sp = np.array([r["spread"] for r in rows])
    bad = np.array([r["failed"] for r in rows])
    hold = np.array([r["hold_ok"] for r in rows])
    print(f"\n[cal] {len(rows)} decisions, {bad.sum()} of them doomed")
    print(f"      predicted drop   doomed {d[bad].mean():+.4f}  "
          f"fine {d[~bad].mean():+.4f}" if bad.any() and (~bad).any() else "")
    print(f"      spread range {sp.min():.4f}-{sp.max():.4f}")

    # Which of the signals the model can produce actually predicts disaster? AUC is
    # the right summary because it is threshold-free: a signal with AUC near 0.5 cannot
    # be rescued by any cut-off, and one below 0.5 is pointing the wrong way.
    def auc(x, y):
        """Mann-Whitney AUC with **averaged ranks for ties**.

        The tie handling is not a detail. Breaking ties by argsort order gives a
        constant signal whatever AUC the arbitrary ordering happens to produce — it
        reported 0.764 for a column that was identically zero on every one of 164
        decisions, i.e. it manufactured a strong predictor out of no data at all.
        With averaged ranks a constant signal scores exactly 0.5, which is the truth.
        """
        pos, neg = x[y], x[~y]
        if not len(pos) or not len(neg):
            return float("nan")
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv, kind="mergesort")
        ranks = np.empty(len(allv), float)
        sortv = allv[order]
        i = 0
        while i < len(sortv):
            j = i
            while j + 1 < len(sortv) and sortv[j + 1] == sortv[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2 + 1      # average rank over the tie
            i = j + 1
        return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
            len(pos) * len(neg))

    print("\n[cal] does any available signal predict a doomed action?")
    print(f"      {'signal':>14} {'AUC':>7}   (0.5 = no information)")
    signals = ["drop", "v_end", "collision", "instability", "spread", "action_dev"]
    if a.failure_head:
        signals.append("fail_logit")
    for name in signals:
        v = np.array([r[name] for r in rows], float)
        if np.ptp(v) == 0:
            print(f"      {name:>14} {0.5:7.3f}   (constant at {v[0]:.3g} — carries "
                  "no information by construction)")
            continue
        # `v_end` is a return: higher is better, so the disaster signal is its
        # negation. Reported that way so every row reads "higher means worse".
        if name == "v_end":
            v = -v
        print(f"      {name:>14} {auc(v, bad):7.3f}")

    # Two questions the pooled AUC above cannot answer, and they turn out to matter
    # more than the pooled number does.
    #
    # 1. WITHIN an episode, which is where a veto actually decides. A signal can
    #    separate doomed episodes from healthy ones and still be useless: the gate is
    #    not asked "is this a bad episode", it is asked "is this the decision to refuse".
    # 2. Whether two signals beat the best one. The pair is the natural one to try —
    #    ensemble spread is epistemic ("I do not model this") and action deviation is
    #    geometric ("this is far from what I was fitted on") — so they can disagree.
    #
    # Both are computed per held-out episode and averaged by pair count. Pooling
    # leave-one-out scores instead is an active trap: each fold's logistic has its own
    # intercept, the intercept anti-correlates with the held-out episode's own failure
    # rate, and the pooled AUC comes out INVERTED. It reported 0.188 for a signal whose
    # in-sample AUC is 0.528 — a strong backwards predictor manufactured entirely by
    # comparing scores that were never on a common scale.
    ep_id = np.array([r["episode"] for r in rows])

    def _weighted_auc(score_of):
        num = den = 0.0
        for e in np.unique(ep_id):
            te = ep_id == e
            if bad[te].all() or not bad[te].any():
                continue          # no pair to order inside this episode
            s = score_of(e, te)
            if s is None:
                continue
            n = int(bad[te].sum()) * int((~bad[te]).sum())
            num += auc(s, bad[te]) * n
            den += n
        return num / den if den else float("nan")

    def _fit(X, y, iters=3000, lr=1.0):
        Xt = np.column_stack([X, np.ones(len(X))])
        w = np.zeros(Xt.shape[1])
        for _ in range(iters):
            pr = 1 / (1 + np.exp(-Xt @ w))
            w += lr * Xt.T @ (y - pr) / len(y) - 1e-4 * w
        return w

    def combo_auc(cols):
        X = np.column_stack([np.array([r[c] for r in rows], float) for c in cols])
        X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)

        def score(e, te):
            tr = ~te
            if bad[tr].all() or not bad[tr].any():
                return None
            w = _fit(X[tr], bad[tr].astype(float))
            return np.column_stack([X[te], np.ones(te.sum())]) @ w
        return _weighted_auc(score)

    print("\n[cal] the same signals scored WITHIN episodes, where the gate decides")
    print(f"      {'signals':>28} {'AUC':>7}   (leave-one-episode-out for the combos)")
    within = ["spread", "action_dev", "drop"] + (
        ["fail_logit"] if a.failure_head else [])
    for name in within:
        v = np.array([r[name] for r in rows], float)
        print(f"      {name:>28} {_weighted_auc(lambda e, te, v=v: v[te]):7.3f}")
    for cols in (("spread", "action_dev"), ("spread", "action_dev", "drop")):
        print(f"      {' + '.join(cols):>28} {combo_auc(cols):7.3f}")

    if a.recoverability:
        rec = np.array([r["recoverable"] for r in rows], int)
        k = a.recoverability
        print(f"\n[cal] recoverability — of {k} available plans, how many still reach "
              f"success")
        print(f"      {'plans that work':>16} {'decisions':>10} {'of which the policy '
              'action fails':>38}")
        for n in range(k + 1):
            m = rec == n
            if m.any():
                print(f"      {n:>16} {int(m.sum()):>10} "
                      f"{bad[m].mean() * 100:37.1f}%")
        lost = rec == 0
        pivotal = (rec > 0) & (rec < k)
        print(f"\n      {lost.mean() * 100:.1f}% of probed decisions are already lost "
              f"(no plan works)")
        print(f"      {(bad & lost).sum()} of {int(bad.sum())} 'doomed action' labels "
              f"({(bad & lost).sum() / max(1, bad.sum()) * 100:.1f}%) come from those "
              f"states —\n      the action was not what doomed them, and a veto there "
              f"could not have helped")
        print(f"      {int(pivotal.sum())} decisions are PIVOTAL: some plans work and "
              f"some do not")
        if pivotal.sum() > 4:
            n_piv = int(pivotal.sum())
            n_bad = int(bad[pivotal].sum())
            print(f"\n[cal] the same signals on the {n_piv} PIVOTAL decisions only, "
                  f"where a veto could change the outcome")
            print(f"      ({n_bad} of them doomed, {n_piv - n_bad} fine — "
                  f"{n_bad * (n_piv - n_bad)} orderable pairs)")
            print(f"      {'signal':>28} {'AUC':>7} {'95% CI':>18}")
            eps_piv = ep_id[pivotal]
            uniq_piv = np.unique(eps_piv)
            rs = np.random.default_rng(0)
            for name in ("spread", "action_dev", "drop", "v_end", "fail_logit"):
                if name not in rows[0]:
                    continue
                v = np.array([r[name] for r in rows], float)
                if name == "v_end":
                    v = -v
                vp, bp = v[pivotal], bad[pivotal]
                # Resample EPISODES, not decisions: pivotal decisions cluster inside
                # the few episodes that were genuinely on a knife edge, so resampling
                # decisions would treat one episode's worth of evidence as many.
                draws = []
                for _ in range(2000):
                    pick = rs.choice(uniq_piv, size=len(uniq_piv), replace=True)
                    idx = np.concatenate([np.where(eps_piv == e)[0] for e in pick])
                    a_ = auc(vp[idx], bp[idx])
                    if not np.isnan(a_):
                        draws.append(a_)
                lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws
                          else (float("nan"),) * 2)
                flag = "" if lo > 0.5 or hi < 0.5 else "   (spans 0.5)"
                print(f"      {name:>28} {auc(vp, bp):7.3f} "
                      f"{f'[{lo:.3f}, {hi:.3f}]':>18}{flag}")

            # Can a deployed gate tell it is looking at a pivotal decision at all?
            #
            # Three families, ordered by what they cost. A state classifier reads the
            # observation and rolls nothing. A short rollout pays a quarter of a chunk.
            # The full plan-value distribution pays the whole chunk over every
            # candidate. If the cheap one wins there is nothing to buy; if only the
            # expensive one works, the gate's cost is the rollout, not the threshold.
            pm = np.array([r["plan_mean"] for r in rows], float)
            ps = np.array([r["plan_std"] for r in rows], float)
            if np.isfinite(ps).all():
                def dist_features(key):
                    """Shape of the candidate-value distribution, per decision.

                    Spread is one number off this distribution and it was the only one
                    tried. The others ask different questions: `gap` asks whether one
                    plan stands out, `frac_below` whether the policy's own action is in
                    the good half, `iqr` whether the bulk disagrees or only an outlier.
                    """
                    out = {}
                    V = [np.asarray(r[key], float) for r in rows]
                    v0s = np.array([r["v0"] for r in rows], float)
                    out["std"] = np.array([v.std() for v in V])
                    out["range"] = np.array([v.max() - v.min() for v in V])
                    out["iqr"] = np.array([np.subtract(*np.percentile(v, [75, 25]))
                                           for v in V])
                    srt = [np.sort(v)[::-1] for v in V]
                    out["gap best-2nd"] = np.array(
                        [x[0] - x[1] if len(x) > 1 else 0.0 for x in srt])
                    out["frac below own v0"] = np.array(
                        [float((v < z).mean()) for v, z in zip(V, v0s)])
                    out["-|frac below - .5|"] = -np.abs(
                        out["frac below own v0"] - 0.5)
                    return out

                def loeo_logit(X, y, ids):
                    """Leave-one-episode-out scores from a logistic fit.

                    Held out by EPISODE for the reason the CIs are: decisions inside
                    one episode are near-copies, so a decision-level split would score
                    the classifier on rows it has effectively already seen.
                    """
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    out = np.full(len(y), np.nan)
                    for e in np.unique(ids):
                        tr, te = ids != e, ids == e
                        if len(np.unique(y[tr])) < 2:
                            continue
                        sc = StandardScaler().fit(X[tr])
                        m = LogisticRegression(max_iter=2000, C=0.5).fit(
                            sc.transform(X[tr]), y[tr])
                        out[te] = m.decision_function(sc.transform(X[te]))
                    return out

                print(f"\n[cal] can PIVOTAL itself be spotted at run time? "
                      f"(no rewind, no simulator — {int(pivotal.sum())} of "
                      f"{len(rows)} decisions are pivotal)")
                print(f"      {'proxy':>32} {'AUC':>7} {'95% CI':>18}   "
                      f"{'cost':<24}")
                rs3 = np.random.default_rng(2)
                uniq_all = np.unique(ep_id)

                def show(nm, v, cost):
                    """AUC with an episode-bootstrap CI.

                    The point estimate alone is what let `action_dev 0.586` stand for a
                    round before n=225 put it at 0.500. Every proxy here is a candidate
                    for the same mistake, so none of them is printed without an
                    interval.
                    """
                    ok = np.isfinite(v)
                    if ok.sum() < 10 or len(np.unique(pivotal[ok])) < 2:
                        return
                    vo, po, eo = v[ok], pivotal[ok], ep_id[ok]
                    draws = []
                    for _ in range(2000):
                        pick = rs3.choice(uniq_all, size=len(uniq_all), replace=True)
                        idx = np.concatenate([np.where(eo == e)[0] for e in pick])
                        if len(np.unique(po[idx])) < 2:
                            continue
                        a_ = auc(vo[idx], po[idx])
                        if not np.isnan(a_):
                            draws.append(a_)
                    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws
                              else (float("nan"),) * 2)
                    flag = "" if lo > 0.5 or hi < 0.5 else " *"
                    print(f"      {nm:>32} {auc(vo, po):7.3f} "
                          f"{f'[{lo:.3f}, {hi:.3f}]':>18}{flag:<2} {cost}")

                full = dist_features("plan_vals")
                short = dist_features("plan_vals_short")
                show("mean plan value", pm, f"{a.recoverability} full rollouts")
                for nm, v in full.items():
                    show(nm, v, f"{a.recoverability} full rollouts")
                print(f"      {'':>32} {'':>7}")
                for nm, v in short.items():
                    show(nm + " [1/4 horizon]", v,
                         f"{a.recoverability} short rollouts")
                print(f"      {'':>32} {'':>7}")
                SF = np.array([r["state_feat"] for r in rows], float)
                try:
                    show("state classifier (LOEO)",
                         loeo_logit(SF, pivotal, ep_id), "no rollout")
                    show("state + full-dist (LOEO)",
                         loeo_logit(np.column_stack(
                             [SF] + list(full.values())), pivotal, ep_id),
                         f"{a.recoverability} full rollouts")
                    show("state + short-dist (LOEO)",
                         loeo_logit(np.column_stack(
                             [SF] + list(short.values())), pivotal, ep_id),
                         f"{a.recoverability} short rollouts")
                except ImportError:
                    print("      (scikit-learn not installed; classifiers skipped)")
                print("      * interval spans 0.5 — no information at this n")

            # What the AUC is worth once a threshold is chosen, on pivotal decisions.
            dv = np.array([r["drop"] for r in rows], float)[pivotal]
            hv = hold[pivotal]
            base = bp0 = bad[pivotal]
            print(f"\n[cal] threshold sweep on 'drop', PIVOTAL decisions only "
                  f"(base rate {base.mean():.3f})")
            print(f"      {'drop >':>8} {'fires':>6} {'prec':>6} {'recall':>7} "
                  f"{'F1':>6} {'net/100':>9} {'95% CI':>18}")
            rs2 = np.random.default_rng(1)
            for q in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
                thr = float(np.quantile(dv, q))
                fire = dv > thr
                tp = int((fire & bp0).sum()); fp = int((fire & ~bp0).sum())
                fn = int((~fire & bp0).sum())
                prec = tp / (tp + fp) if tp + fp else float("nan")
                rec_ = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec_ / (prec + rec_) if tp and prec + rec_ > 0 else 0.0

                def net(idx):
                    f = dv[idx] > thr
                    if not f.any():
                        return None
                    return 100.0 * ((f & bp0[idx] & hv[idx]).sum()
                                    - (f & ~bp0[idx] & ~hv[idx]).sum()) / f.sum()
                point = net(np.arange(len(dv)))
                draws = []
                for _ in range(2000):
                    pick = rs2.choice(uniq_piv, size=len(uniq_piv), replace=True)
                    idx = np.concatenate([np.where(eps_piv == e)[0] for e in pick])
                    val = net(idx)
                    if val is not None:
                        draws.append(val)
                lo2, hi2 = (np.percentile(draws, [2.5, 97.5]) if draws
                            else (float("nan"),) * 2)
                span = "" if lo2 > 0 else "   (spans 0)"
                print(f"      {thr:8.4f} {int(fire.sum()):6d} {prec:6.3f} "
                      f"{rec_:7.3f} {f1:6.3f} "
                      f"{('n/a' if point is None else f'{point:+.1f}'):>9} "
                      f"{f'[{lo2:+.1f}, {hi2:+.1f}]':>18}{span}")

    sig = np.array([r[a.signal] for r in rows], float)
    if a.signal == "v_end":
        sig = -sig
    print(f"\n[cal] threshold sweep on '{a.signal}' (higher = worse)")
    print(f"      {'thresh>':>8} {'fires':>6} {'prec':>6} {'recall':>7} {'F1':>6} "
          f"{'rescued':>8}")
    best = None
    for thr in np.unique(np.quantile(sig, np.linspace(0.4, 0.99, 14))):
        fire = sig > thr
        tp = int((fire & bad).sum())
        fp = int((fire & ~bad).sum())
        fn = int((~fire & bad).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if tp and prec + rec > 0 else 0.0
        resc = hold[fire & bad].mean() if (fire & bad).any() else float("nan")
        print(f"      {thr:8.4f} {int(fire.sum()):6d} {prec:6.3f} {rec:7.3f} "
              f"{f1:6.3f} {resc:8.3f}")
        if best is None or f1 > best[1]:
            best = (float(thr), f1)
    if best:
        print(f"\n[cal] best F1 {best[1]:.3f} at {a.signal} > {best[0]:.4f} "
              f"(base rate {bad.mean():.3f} — precision below this is worse than "
              "firing at random)")
    # The number that decides whether a veto is worth having at all: among genuinely
    # doomed actions, how often does holding instead actually recover the episode?
    if bad.any():
        print(f"[cal] holding rescues {hold[bad].mean() * 100:.1f}% of doomed actions; "
              f"if that is near zero, a perfect veto still buys nothing")

    # What the gate is actually worth, with an interval rather than a point.
    #
    # F1 is the wrong objective here: a fire is worthwhile only when the action really
    # was doomed AND holding recovers it, and harmful when the action was fine AND
    # holding ruins it. Net flips per 100 fires is that arithmetic. The counts behind it
    # are single digits, so a point estimate would be indefensible — the bootstrap
    # resamples **episodes**, not decisions, because decisions within one episode share
    # a scene and a policy seed and are not independent draws.
    eps = np.array([r["episode"] for r in rows])
    uniq = np.unique(eps)
    rs = np.random.default_rng(0)

    def net_per_100(mask_sig, thr, idx):
        f = mask_sig[idx] > thr
        if not f.any():
            return None
        saves = int((f & bad[idx] & hold[idx]).sum())
        breaks = int((f & ~bad[idx] & ~hold[idx]).sum())
        return 100.0 * (saves - breaks) / int(f.sum())

    print(f"\n[cal] net episode-outcome flips per 100 fires, "
          f"95% bootstrap CI over {len(uniq)} episodes")
    print(f"      {'signal':>8} {'thresh':>9} {'fires':>6} {'net/100':>9} "
          f"{'95% CI':>18}")
    for name in ("spread", "drop"):
        v = np.array([r[name] for r in rows], float)
        for q in (0.6, 0.7, 0.8, 0.9):
            thr = float(np.quantile(v, q))
            point = net_per_100(v, thr, np.arange(len(rows)))
            if point is None:
                continue
            draws = []
            for _ in range(2000):
                pick = rs.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([np.where(eps == e)[0] for e in pick])
                val = net_per_100(v, thr, idx)
                if val is not None:
                    draws.append(val)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            flag = "" if lo > 0 else "   (CI includes 0)"
            print(f"      {name:>8} {thr:9.4f} {int((v > thr).sum()):6d} "
                  f"{point:+9.1f} {f'[{lo:+.1f}, {hi:+.1f}]':>18}{flag}")
    if a.json:
        # Which model produced these signals, recorded in the file rather than in the
        # launcher's comments. [S22]: two runs with identical decision counts and an
        # identical pivotal set disagreed on return-drop AUC (0.601 vs 0.436) purely
        # because they used different checkpoints, and nothing in either JSON said so.
        # The pivotal LABELS come from simulator rewind and agree across models, which
        # is exactly what makes the mismatch invisible without this block.
        import hashlib

        from pipeline.sim.loop import task_fingerprint

        def sha(path):
            return (hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
                    if path and Path(path).exists() else None)

        a.json.write_text(json.dumps(
            {"config": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(a).items()},
             "checkpoints": {"dynamics": sha(a.dynamics), "value": sha(a.value),
                             "failure_head": sha(a.failure_head)},
             "task": task_fingerprint(env.max_steps),
             "rows": rows, "best_threshold": best},
            indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

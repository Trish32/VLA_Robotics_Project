#!/usr/bin/env python
"""Score a two-stage gate — pivotal proxy, then return-drop — end to end.

The gate this measures is the one a robot could actually run: one batched forward pass
over the K available plans at a quarter horizon gives a spread statistic, and the veto
fires only where that statistic says the decision is pivotal AND the predicted return
drop says the action is bad.

The reason this exists rather than a back-of-envelope: the two stages' AUCs cannot be
multiplied. AUC is a ranking summary over a whole population; a two-stage gate is a
conjunction of two thresholds evaluated on the SAME decisions, and its yield depends on
how the two signals co-vary there. Two proxies at AUC 0.65 that agree on which
decisions they flag buy one gate's worth of filtering, not two.

What is reported is what a gate is for: net episode-outcome flips per 100 fires —
rescues minus breakages — with the interval bootstrapped over episodes.

    conda run -n simple_bev_vldrive python -m pipeline.sim.analyse_gate \
        --rows <scratchpad>/piv6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc(x, y) -> float:
    """Mann-Whitney AUC with averaged ranks for ties."""
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(x)
    xs = x[order]
    ranks = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r = np.empty(len(x), float)
    r[order] = ranks
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    """Rank correlation, so a monotone rescaling of either signal cannot change it."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def proxies(rows) -> dict[str, np.ndarray]:
    """The three run-time pivotal proxies, recomputed from the saved plan values."""
    V = [np.asarray(r["plan_vals"], float) for r in rows]
    Vs = [np.asarray(r["plan_vals_short"], float) for r in rows]
    top2 = lambda v: (np.sort(v)[::-1][0] - np.sort(v)[::-1][1]) if len(v) > 1 else 0.0
    return {
        "range [1/4 H]": np.array([v.max() - v.min() for v in Vs]),
        "std [1/4 H]": np.array([v.std() for v in Vs]),
        "gap best-2nd": np.array([top2(v) for v in V]),
    }


def state_classifier(rows, pivotal, ep) -> np.ndarray | None:
    """Leave-one-episode-out logistic on the raw observation. No rollout at all.

    This is the cheapest candidate for stage 1 — it reads what the robot already has
    and runs no dynamics — so the question it answers is whether the rollout is
    purchasable at zero marginal cost.

    Held out by EPISODE, not by decision: consecutive decisions inside an episode are
    near-duplicates, so a decision-level split would score the classifier on rows it
    has effectively already seen and report a deployment number that does not exist.

    One cost this does NOT avoid: it is fitted on the true pivotal label, which comes
    from rewinding the simulator. The rollout proxies need no labels at all. So "no
    rollout" is a claim about inference cost, never about what it takes to build.
    """
    if "state_feat" not in rows[0]:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    X = np.array([r["state_feat"] for r in rows], float)
    out = np.full(len(rows), np.nan)
    for e in np.unique(ep):
        tr, te = ep != e, ep == e
        if len(np.unique(pivotal[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]),
                                                         pivotal[tr])
        out[te] = m.decision_function(sc.transform(X[te]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3000)
    a = ap.parse_args(argv)

    d = json.loads(a.rows.read_text())
    rows = d["rows"]
    ck = d.get("checkpoints")
    print(f"[gate] {a.rows.name}: {len(rows)} decisions"
          + (f"   checkpoints {ck}" if ck else "   (no checkpoint recorded — [S22])"))

    ep = np.array([r["episode"] for r in rows])
    bad = np.array([r["failed"] for r in rows], bool)
    hold = np.array([r["hold_ok"] for r in rows], bool)
    rec = np.array([r["recoverable"] for r in rows], int)
    K = max(len(r["plan_vals"]) for r in rows)
    pivotal = (rec > 0) & (rec < K)
    drop = np.array([r["drop"] for r in rows], float)
    P = proxies(rows)
    sc_scores = state_classifier(rows, pivotal, ep)
    if sc_scores is not None and np.isfinite(sc_scores).all():
        P["state (no rollout)"] = sc_scores
    uniq = np.unique(ep)
    print(f"       {int(pivotal.sum())} pivotal of {len(rows)}   "
          f"{len(uniq)} episodes   K={K}")

    # --- 2: are the three proxies measuring different things? --------------------
    print("\n[gate] the three proxies — absolute AUC vs the true pivotal label, "
          "on all decisions")
    print(f"       {'proxy':>20} {'AUC':>7} {'95% CI':>18}")
    rs = np.random.default_rng(0)
    for nm, v in P.items():
        draws = []
        for _ in range(a.reps):
            pick = rs.choice(uniq, len(uniq), True)
            idx = np.concatenate([np.where(ep == e)[0] for e in pick])
            val = auc(v[idx], pivotal[idx])
            if not np.isnan(val):
                draws.append(val)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        flag = "" if lo > 0.5 or hi < 0.5 else "   (spans 0.5)"
        print(f"       {nm:>20} {auc(v, pivotal):7.3f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{flag}")

    names = list(P)
    print("\n[gate] pairwise Spearman between the proxies "
          "(1.00 = the same signal twice)")
    print("       " + " " * 20 + "".join(f"{n:>20}" for n in names))
    for i, n1 in enumerate(names):
        cells = "".join(f"{spearman(P[n1], P[n2]):>20.3f}" for n2 in names)
        print(f"       {n1:>20}{cells}")
    print("       ... and against the drop signal the second stage uses:")
    for n1 in names:
        print(f"       {n1:>20} vs drop  {spearman(P[n1], drop):+.3f}")
    # Agreement on the decisions actually flagged matters more than correlation over
    # the whole population: two signals can correlate weakly and still select almost
    # the same top decile, which is the only part a threshold ever sees.
    print("\n[gate] overlap of the top-30% flagged sets (Jaccard)")
    top = {n: v >= np.quantile(v, 0.70) for n, v in P.items()}
    print("       " + " " * 20 + "".join(f"{n:>20}" for n in names))
    for n1 in names:
        cells = "".join(
            f"{(top[n1] & top[n2]).sum() / max(1, (top[n1] | top[n2]).sum()):>20.3f}"
            for n2 in names)
        print(f"       {n1:>20}{cells}")

    # --- 1: the two-stage gate, scored as a gate ---------------------------------
    def net_per_100(fire, idx):
        f = fire[idx]
        if not f.any():
            return None
        saves = int((f & bad[idx] & hold[idx]).sum())
        breaks = int((f & ~bad[idx] & ~hold[idx]).sum())
        return 100.0 * (saves - breaks) / int(f.sum())

    def ci(fire):
        point = net_per_100(fire, np.arange(len(rows)))
        if point is None:
            return None
        rs2 = np.random.default_rng(1)
        draws = []
        for _ in range(a.reps):
            pick = rs2.choice(uniq, len(uniq), True)
            idx = np.concatenate([np.where(ep == e)[0] for e in pick])
            val = net_per_100(fire, idx)
            if val is not None:
                draws.append(val)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return point, lo, hi, int(fire.sum())

    print("\n[gate] net episode-outcome flips per 100 fires — measured under the "
          "combination,\n       never by multiplying the two stages' AUCs")
    print(f"       {'stage 1 (proxy)':>20} {'q1':>5} {'q2':>5} {'fires':>6} "
          f"{'net/100':>9} {'95% CI':>18}")

    def show(tag, fire, q1="—", q2="—"):
        out = ci(fire)
        if out is None:
            print(f"       {tag:>20} {q1:>5} {q2:>5} {'0':>6}   no fires")
            return
        point, lo, hi, n = out
        flag = "" if lo > 0 else "   (CI includes 0)"
        print(f"       {tag:>20} {q1:>5} {q2:>5} {n:6d} {point:+9.1f} "
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}{flag}")

    # Baselines first, so the combination is read against what it has to beat.
    for q in (0.60, 0.70, 0.80):
        show("drop alone", drop > np.quantile(drop, q), "—", f"{q:.2f}")
    show("ORACLE pivotal", pivotal & (drop > np.quantile(drop[pivotal], 0.30)),
         "true", "0.30")
    for nm, v in P.items():
        for q1 in (0.50, 0.70):
            for q2 in (0.30, 0.50):
                sel = v > np.quantile(v, q1)
                if not sel.any():
                    continue
                thr2 = np.quantile(drop[sel], q2)
                show(nm, sel & (drop > thr2), f"{q1:.2f}", f"{q2:.2f}")

    # Can the cheap stage REPLACE the rollout, or only join it? Replacement is the
    # state row above; joining is this. Quantiles are loosened per-signal so the
    # conjunction fires at a comparable rate — a stricter gate that fires half as
    # often is not a fair comparison, it is a different operating point.
    if "state (no rollout)" in P:
        st, rg = P["state (no rollout)"], P["range [1/4 H]"]
        # Matched FIRING RATE, not matched quantile. A gate that fires half as often
        # is a different operating point, and comparing its net/100 to another's is
        # comparing two policies, not two signals.
        def at_rate(sel_fn, target, q2=0.50):
            """Tune stage 1 until the finished gate fires `target` times."""
            best, bestn = None, None
            for q1 in np.linspace(0.02, 0.96, 190):
                sel = sel_fn(q1)
                if not sel.any():
                    continue
                fire = sel & (drop > np.quantile(drop[sel], q2))
                n = int(fire.sum())
                if bestn is None or abs(n - target) < abs(bestn - target):
                    best, bestn = fire, n
            return best, bestn

        target = 158
        cands = {
            "state (no rollout)": lambda q: st > np.quantile(st, q),
            "range [1/4 H]": lambda q: rg > np.quantile(rg, q),
            "state AND range": lambda q: (st > np.quantile(st, q))
                                         & (rg > np.quantile(rg, q * 0.78)),
            "ORACLE pivotal": lambda q: pivotal & (st > np.quantile(st, q * 0.02)),
        }
        fires = {}
        print(f"\n[gate] does the cheap stage REPLACE the rollout, or only join it?")
        print(f"       stage 1 tuned so the finished gate fires ~{target} times")
        print(f"       {'stage 1 (proxy)':>20} {'fires':>6} {'net/100':>9} "
              f"{'95% CI':>18}")
        for nm, fn in cands.items():
            fire, n = at_rate(fn, target)
            if fire is None:
                continue
            fires[nm] = fire
            out = ci(fire)
            point, lo, hi, _ = out
            flag = "" if lo > 0 else "   (CI includes 0)"
            print(f"       {nm:>20} {n:6d} {point:+9.1f} "
                  f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}{flag}")

        # The comparison the question turns on. Differences are PAIRED — each
        # replicate draws one set of episodes and scores both gates on it — because
        # two overlapping marginal intervals say nothing about the sign of a
        # difference, which is the thing being asked.
        print("\n       paired difference vs the rollout proxy, over the same "
              "resampled episodes")
        print(f"       {'comparison':>36} {'Δ net/100':>10} {'95% CI':>18}")
        base_nm = "range [1/4 H]"
        if base_nm in fires:
            rs3 = np.random.default_rng(7)
            for nm in ("state (no rollout)", "state AND range", "ORACLE pivotal"):
                if nm not in fires:
                    continue
                draws = []
                for _ in range(a.reps):
                    pick = rs3.choice(uniq, len(uniq), True)
                    idx = np.concatenate([np.where(ep == e)[0] for e in pick])
                    x1 = net_per_100(fires[nm], idx)
                    x0 = net_per_100(fires[base_nm], idx)
                    if x1 is not None and x0 is not None:
                        draws.append(x1 - x0)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                pt = (net_per_100(fires[nm], np.arange(len(rows)))
                      - net_per_100(fires[base_nm], np.arange(len(rows))))
                flag = "" if lo > 0 or hi < 0 else "   (spans 0)"
                print(f"       {nm + ' − ' + base_nm:>36} {pt:+10.1f} "
                      f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}{flag}")

    # --- is +12.0 a result, or the maximum of a grid I searched? -----------------
    #
    # Every net/100 above was read off a (q1, q2) chosen from a small set, and the
    # best cell of a searched grid is biased upward by construction. The honest
    # summary of a two-threshold gate is the surface, not its argmax: if most of the
    # surface is positive the effect is a property of the signal; if only the peak
    # is, it is a property of the search.
    print("\n[gate] net/100 surface over (q1, q2) — is the headline a peak?")
    for nm in ("range [1/4 H]", "state (no rollout)"):
        if nm not in P:
            continue
        v = P[nm]
        q1s = np.round(np.arange(0.10, 0.91, 0.10), 2)
        q2s = np.round(np.arange(0.10, 0.91, 0.10), 2)
        grid = np.full((len(q1s), len(q2s)), np.nan)
        firegrid = np.zeros_like(grid)
        for i, q1 in enumerate(q1s):
            sel = v > np.quantile(v, q1)
            if not sel.any():
                continue
            for j, q2 in enumerate(q2s):
                fire = sel & (drop > np.quantile(drop[sel], q2))
                val = net_per_100(fire, np.arange(len(rows)))
                # Cells firing fewer than 30 times are dropped rather than plotted:
                # net/100 on 8 fires is a number with no width to it.
                if val is not None and int(fire.sum()) >= 30:
                    grid[i, j] = val
                    firegrid[i, j] = int(fire.sum())
        print(f"\n       {nm}   (rows q1, cols q2; · = under 30 fires)")
        print("       " + "q1\\q2".rjust(7) + "".join(f"{q:>7.2f}" for q in q2s))
        for i, q1 in enumerate(q1s):
            cells = "".join("      ·" if np.isnan(g) else f"{g:>7.1f}"
                            for g in grid[i])
            print(f"       {q1:>7.2f}{cells}")
        ok = grid[~np.isnan(grid)]
        if not len(ok):
            continue
        i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
        print(f"       cells {len(ok)}   positive {100 * (ok > 0).mean():.0f}%   "
              f"median {np.median(ok):+.1f}   IQR "
              f"[{np.percentile(ok, 25):+.1f}, {np.percentile(ok, 75):+.1f}]")
        print(f"       max {grid[i, j]:+.1f} at q1={q1s[i]:.2f} q2={q2s[j]:.2f} "
              f"({int(firegrid[i, j])} fires)")
        # The interval at the MEDIAN cell is the one to quote: it was not chosen for
        # being large, so its CI is not conditioned on a search.
        flat = sorted((grid[i2, j2], i2, j2)
                      for i2 in range(len(q1s)) for j2 in range(len(q2s))
                      if not np.isnan(grid[i2, j2]))
        g, i2, j2 = flat[len(flat) // 2]
        sel = v > np.quantile(v, q1s[i2])
        fire = sel & (drop > np.quantile(drop[sel], q2s[j2]))
        out = ci(fire)
        if out is not None:
            pt, lo, hi, n = out
            flag = "" if lo > 0 else "   (CI includes 0)"
            print(f"       median cell q1={q1s[i2]:.2f} q2={q2s[j2]:.2f}: "
                  f"{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]  {n} fires{flag}")

    # --- what is actually inside the fires ---------------------------------------
    #
    # net/100 is a difference of two counts and hides both of them. A gate at +12
    # could be 12 rescues and 0 breakages out of 100, or 40 and 28 — the same score
    # with completely different risk. And a veto only DOES anything on decisions
    # where the action and the hold disagree; everywhere else it fires and changes
    # nothing, which is a cost in latency and trust but not in outcomes.
    K_ = K
    lost = rec == 0
    allviable = rec == K_

    def anatomy(fire, tag):
        n = int(fire.sum())
        if not n:
            return
        save = fire & bad & hold          # action doomed, holding works  -> rescue
        brk = fire & ~bad & ~hold         # action fine, holding fails    -> breakage
        moot_bad = fire & bad & ~hold     # doomed either way
        moot_ok = fire & ~bad & hold      # fine either way
        print(f"\n       {tag}: {n} fires")
        print(f"         rescues    (action doomed, hold works)   {int(save.sum()):4d}"
              f"   {100 * save.sum() / n:5.1f}%")
        print(f"         breakages  (action fine,  hold fails)    {int(brk.sum()):4d}"
              f"   {100 * brk.sum() / n:5.1f}%")
        print(f"         no effect  (doomed either way)           "
              f"{int(moot_bad.sum()):4d}   {100 * moot_bad.sum() / n:5.1f}%")
        print(f"         no effect  (fine either way)             "
              f"{int(moot_ok.sum()):4d}   {100 * moot_ok.sum() / n:5.1f}%")
        print(f"         -> net {100.0 * (save.sum() - brk.sum()) / n:+.1f} per 100; "
              f"{100 * (moot_bad.sum() + moot_ok.sum()) / n:.0f}% of fires change "
              f"nothing at all")
        # Where the harm lives. A false positive is only costly where holding is
        # worse than acting, and that is a property of the STATE, not of the gate.
        print(f"         {'stratum':>22} {'fires':>6} {'rescue':>7} {'break':>7} "
              f"{'net/100':>9}")
        for sname, m in (("already lost (viable 0)", lost),
                         ("PIVOTAL (0<viable<K)", pivotal),
                         (f"all viable (viable {K_})", allviable)):
            f2 = fire & m
            n2 = int(f2.sum())
            if not n2:
                print(f"         {sname:>22} {0:6d}")
                continue
            s2 = int((f2 & bad & hold).sum())
            b2 = int((f2 & ~bad & ~hold).sum())
            print(f"         {sname:>22} {n2:6d} {s2:7d} {b2:7d} "
                  f"{100.0 * (s2 - b2) / n2:+9.1f}")

    print("\n[gate] anatomy of the fires — rescues, breakages, and fires that "
          "change nothing")
    v = P["range [1/4 H]"]
    for q1, q2, tag in ((0.70, 0.50, "range q1=0.70 q2=0.50  (the headline cell)"),
                        (0.50, 0.70, "range q1=0.50 q2=0.70  (the median cell)")):
        sel = v > np.quantile(v, q1)
        anatomy(sel & (drop > np.quantile(drop[sel], q2)), tag)

    # --- how harmful is a false positive, on its own terms? ----------------------
    #
    # "False positive" for this gate means firing on a decision that is not pivotal.
    # The two non-pivotal strata are not symmetric and must not be pooled: on an
    # already-lost state the veto is free, because nothing was going to work anyway;
    # on an all-viable state it is the only place breakage can come from.
    print("\n[gate] the cost of firing on a NON-pivotal decision, by stratum")
    print(f"       {'stratum':>22} {'decisions':>10} {'P(hold fails | action fine)':>29}")
    for sname, m in (("already lost (viable 0)", lost),
                     ("PIVOTAL (0<viable<K)", pivotal),
                     (f"all viable (viable {K_})", allviable)):
        fine = m & ~bad
        if not fine.any():
            print(f"       {sname:>22} {int(m.sum()):10d}          (no 'fine' rows)")
            continue
        harm = float((~hold[fine]).mean())
        rs4 = np.random.default_rng(11)
        draws = []
        for _ in range(a.reps):
            pick = rs4.choice(uniq, len(uniq), True)
            idx = np.concatenate([np.where(ep == e)[0] for e in pick])
            sub = fine[idx]
            if sub.any():
                draws.append(float((~hold[idx][sub]).mean()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"       {sname:>22} {int(m.sum()):10d}   {harm:9.3f}  "
              f"[{lo:.3f}, {hi:.3f}]")
    print("       (this is the chance a veto BREAKS a working action, given it fired "
          "on one)")

    # --- stage 2, re-aimed ---------------------------------------------------------
    #
    # The second stage has been predicting "will this action fail". That is the wrong
    # target for a veto, and the anatomy above shows why: of 158 fires, 20 landed on
    # actions that were indeed doomed AND on which holding was equally doomed. The
    # signal was right and the fire was pointless.
    #
    # What a veto needs to predict is whether HOLDING BEATS EXECUTING. That label is
    # defined only where the two differ, so it is scored on that subset:
    #
    #     hold better : the action fails and holding succeeds   -> should fire
    #     hold worse  : the action succeeds and holding fails   -> must not fire
    #
    # Everywhere else the veto is a no-op and belongs in neither class. Training on
    # "will it fail" pools the no-ops in with the rescues, which is exactly the
    # confusion that makes 82% of fires inert.
    hold_better = bad & hold
    hold_worse = ~bad & ~hold
    decisive = hold_better | hold_worse
    print("\n[gate] stage 2 re-aimed: 'is holding better than executing', not "
          "'will this fail'")
    print(f"       decisive decisions {int(decisive.sum())} of {len(rows)}   "
          f"hold-better {int(hold_better.sum())}   hold-worse {int(hold_worse.sum())}")
    print(f"       of which pivotal: {int((decisive & pivotal).sum())}")

    def loeo(X, y, mask):
        """Out-of-fold logistic scores for `y`, fitted only on rows in `mask`."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None
        out = np.full(len(y), np.nan)
        for e in uniq:
            tr = (ep != e) & mask
            te = ep == e
            if len(np.unique(y[tr])) < 2 or tr.sum() < 12:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]),
                                                             y[tr])
            out[te] = m.decision_function(sc.transform(X[te]))
        return out

    plan_cols = np.column_stack([P[k] for k in
                                 ("range [1/4 H]", "std [1/4 H]", "gap best-2nd")])
    base = np.column_stack([drop, np.array([r["v_end"] for r in rows], float),
                            np.array([r["spread"] for r in rows], float),
                            np.array([r["action_dev"] for r in rows], float)])
    feats = {"drop (as shipped)": drop}
    if "state_feat" in rows[0]:
        X = np.column_stack([np.array([r["state_feat"] for r in rows], float),
                             base, plan_cols])
        h = loeo(X, hold_better, decisive)
        if h is not None and np.isfinite(h).any():
            feats["head on 'hold better'"] = np.nan_to_num(h, nan=np.nanmin(h))
        h2 = loeo(X, bad, np.ones(len(rows), bool))
        if h2 is not None and np.isfinite(h2).any():
            feats["head on 'will fail'"] = np.nan_to_num(h2, nan=np.nanmin(h2))

    print(f"\n       ranking hold-better above hold-worse "
          f"(the only rows where a veto changes anything)")
    print(f"       {'stage-2 signal':>26} {'AUC':>7} {'95% CI':>18}")
    rs5 = np.random.default_rng(13)
    for nm, v2 in feats.items():
        for sub, lab in ((decisive, "all"), (decisive & pivotal, "pivotal")):
            if len(np.unique(hold_better[sub])) < 2:
                continue
            draws = []
            for _ in range(a.reps):
                pick = rs5.choice(uniq, len(uniq), True)
                idx = np.concatenate([np.where(ep == e)[0] for e in pick])
                m2 = sub[idx]
                if m2.sum() < 5 or len(np.unique(hold_better[idx][m2])) < 2:
                    continue
                val = auc(v2[idx][m2], hold_better[idx][m2])
                if not np.isnan(val):
                    draws.append(val)
            if not draws:
                continue
            lo, hi = np.percentile(draws, [2.5, 97.5])
            flag = "" if lo > 0.5 or hi < 0.5 else "   (spans 0.5)"
            print(f"       {nm + ' [' + lab + ']':>26} "
                  f"{auc(v2[sub], hold_better[sub]):7.3f} "
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}{flag}")

    # The core metric the re-aiming is for: given the gate fires and the fire MATTERS,
    # how often is it the wrong way round? Unlike net/100 this cannot be inflated by
    # firing more often on decisions where nothing was at stake.
    print("\n[gate] conditional breakage rate = breakages / (rescues + breakages)")
    print(f"       {'stage 1':>16} {'stage 2':>24} {'fires':>6} {'resc':>5} "
          f"{'brk':>4} {'cond. breakage':>16} {'net/100':>8}")
    v1 = P["range [1/4 H]"]
    sel = v1 > np.quantile(v1, 0.70)
    for nm, v2 in feats.items():
        fire = sel & (v2 > np.quantile(v2[sel], 0.50))
        n = int(fire.sum())
        r_ = int((fire & hold_better).sum())
        b_ = int((fire & hold_worse).sum())
        dec = r_ + b_
        if not n:
            continue
        rs6 = np.random.default_rng(17)
        draws = []
        for _ in range(a.reps):
            pick = rs6.choice(uniq, len(uniq), True)
            idx = np.concatenate([np.where(ep == e)[0] for e in pick])
            f2 = fire[idx]
            d2 = int((f2 & hold_better[idx]).sum()) + int((f2 & hold_worse[idx]).sum())
            if d2:
                draws.append((f2 & hold_worse[idx]).sum() / d2)
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan,) * 2)
        rate = f"{b_ / dec:.3f} [{lo:.3f}, {hi:.3f}]" if dec else "—"
        print(f"       {'range q1=0.70':>16} {nm:>24} {n:6d} {r_:5d} {b_:4d} "
              f"{rate:>16} {100.0 * (r_ - b_) / n:+8.1f}")

    # --- 2: can a tighter stage-1 shed the inert fires? ---------------------------
    #
    # 89 of 158 fires landed on all-viable states and bought +1.1 between them. If the
    # proxy separates those from pivotal decisions, a higher threshold drops them and
    # keeps the rescues; if it does not, the inert fires are the price of the rescues.
    print("\n[gate] can a tighter stage-1 threshold shed the inert fires?")
    sub = pivotal | allviable
    sep = auc(v1[sub], pivotal[sub])
    draws = []
    rs7 = np.random.default_rng(19)
    for _ in range(a.reps):
        pick = rs7.choice(uniq, len(uniq), True)
        idx = np.concatenate([np.where(ep == e)[0] for e in pick])
        m2 = sub[idx]
        if m2.sum() < 5 or len(np.unique(pivotal[idx][m2])) < 2:
            continue
        val = auc(v1[idx][m2], pivotal[idx][m2])
        if not np.isnan(val):
            draws.append(val)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"       range separates PIVOTAL from ALL-VIABLE: AUC {sep:.3f} "
          f"[{lo:.3f}, {hi:.3f}]"
          + ("" if lo > 0.5 else "   (spans 0.5 — not separable)"))
    print(f"\n       {'q1':>5} {'fires':>6} {'all-viable':>11} {'pivotal':>8} "
          f"{'rescues':>8} {'brk':>4} {'cond.brk':>9} {'net/100':>8}")
    for q1 in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94):
        s1 = v1 > np.quantile(v1, q1)
        if not s1.any():
            continue
        fire = s1 & (drop > np.quantile(drop[s1], 0.50))
        n = int(fire.sum())
        if n < 20:
            continue
        r_ = int((fire & hold_better).sum())
        b_ = int((fire & hold_worse).sum())
        dec = r_ + b_
        print(f"       {q1:5.2f} {n:6d} {int((fire & allviable).sum()):11d} "
              f"{int((fire & pivotal).sum()):8d} {r_:8d} {b_:4d} "
              f"{(b_ / dec if dec else float('nan')):9.3f} "
              f"{100.0 * (r_ - b_) / n:+8.1f}")

    # --- 3: the two conditional rates, which are the symmetric pair ----------------
    #
    # net/100 is a difference of counts and hides both. The pair that does not:
    #
    #   conditional RESCUE   = P(holding works | fired on a doomed action)
    #   conditional BREAKAGE = P(holding fails | fired on a fine action)
    #
    # The first is the ceiling on what firing correctly can buy; the second is what
    # firing incorrectly costs. A veto is worth running when the first is high on the
    # decisions it catches and the second is low on the ones it catches by mistake,
    # and neither can be read off the other.
    def rate_ci(num_mask, den_mask, seed):
        den = int(den_mask.sum())
        if not den:
            return None
        rsx = np.random.default_rng(seed)
        draws = []
        for _ in range(a.reps):
            pick = rsx.choice(uniq, len(uniq), True)
            idx = np.concatenate([np.where(ep == e)[0] for e in pick])
            d2 = int(den_mask[idx].sum())
            if d2:
                draws.append(num_mask[idx].sum() / d2)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(num_mask.sum() / den), lo, hi, den

    print("\n[gate] conditional rescue and conditional breakage — the symmetric pair")
    print(f"       {'gate':>28} {'cond. RESCUE':>26} {'cond. BREAKAGE':>26}")
    v1 = P["range [1/4 H]"]
    for q1, q2, tag in ((0.70, 0.50, "range q1=0.70 q2=0.50"),
                        (0.50, 0.70, "range q1=0.50 q2=0.70"),
                        (None, None, "no gate — every decision")):
        if q1 is None:
            fire = np.ones(len(rows), bool)
        else:
            sel = v1 > np.quantile(v1, q1)
            fire = sel & (drop > np.quantile(drop[sel], q2))
        r = rate_ci(fire & bad & hold, fire & bad, 23)
        b = rate_ci(fire & ~bad & ~hold, fire & ~bad, 29)
        fmt = lambda t: (f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}] n={t[3]}"
                         if t else "—")
        print(f"       {tag:>28} {fmt(r):>26} {fmt(b):>26}")
    print("       rescue is over fires on DOOMED actions; breakage over fires on FINE "
          "ones.\n       A gate that changes neither is selecting decisions, not "
          "changing what they are worth.")

    # --- 2: does coarser granularity raise the decisive fraction? -----------------
    #
    # Only 9% of decisions are decisive, which caps what any per-decision gate can do.
    # The proposal is that a veto asked ONCE per episode faces a different
    # distribution: it no longer has to find the rare decision that matters, it only
    # has to be right about the episode. This bounds that.
    print("\n[gate] does a coarser veto see a denser decisive signal?")
    per_ep_dec, per_ep_n, per_ep_any = [], [], []
    for e in uniq:
        m = ep == e
        per_ep_n.append(int(m.sum()))
        per_ep_dec.append(int((m & decisive).sum()))
        per_ep_any.append(bool((m & decisive).any()))
    per_ep_dec = np.array(per_ep_dec)
    print(f"       per-decision:  {int(decisive.sum())} decisive of {len(rows)} "
          f"= {100 * decisive.mean():.1f}%")
    print(f"       per-episode :  {int(np.sum(per_ep_any))} episodes with at least "
          f"one decisive decision of {len(uniq)} = "
          f"{100 * np.mean(per_ep_any):.1f}%")
    print(f"       decisive decisions per episode: median "
          f"{int(np.median(per_ep_dec))}, mean {per_ep_dec.mean():.2f}, "
          f"max {per_ep_dec.max()}   (decisions per episode "
          f"{np.mean(per_ep_n):.1f})")
    # The bound this buys: fire once per episode, at the decision an oracle picks.
    # It is an upper bound and is labelled as one — no run-time signal is involved.
    ep_rescue = np.array([bool(((ep == e) & hold_better).any()) for e in uniq])
    ep_break = np.array([bool(((ep == e) & hold_worse).any()) for e in uniq])
    print(f"       episodes where SOME decision is a rescue: "
          f"{100 * ep_rescue.mean():.1f}%;  where some is a breakage: "
          f"{100 * ep_break.mean():.1f}%")
    print(f"       episodes carrying both: {100 * (ep_rescue & ep_break).mean():.1f}% "
          f"— a once-per-episode veto must choose WHICH, and at that granularity the "
          f"two are\n       no longer separable by anything the gate observes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

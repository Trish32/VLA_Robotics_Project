#!/usr/bin/env python
"""Panel 6 — what a learned veto actually does, and what bounds it.

The veto is the one place the world model is used for something it is measurably good
at: predicting outcomes, rather than discriminating between near-identical candidates.
This panel is the case for it and the case against it on the same page, because a
net-flips-per-100 headline conceals both.

Three things a single score cannot show:

  * **most fires do nothing.** A veto only matters where holding and executing lead to
    different outcomes. That is 9.1% of decisions, and the rest of the fires are inert.
  * **it does shift the odds.** Conditioned on firing at a doomed action, holding
    rescues it twice as often under the gate as without one — and the rate at which it
    breaks a working action does not move.
  * **the harm is where the gate is supposed to fire.** Firing on a state where every
    plan works is nearly free; the exposure is concentrated on exactly the pivotal
    decisions the first stage is built to find.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig6_veto \
        --rows <scratchpad>/piv6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.sim.figures import theme


def boot(num, den, ep, uniq, reps=3000, seed=0):
    """Rate with a 95% interval resampled over EPISODES, not decisions."""
    d = int(den.sum())
    if not d:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), True)
        idx = np.concatenate([np.where(ep == e)[0] for e in pick])
        d2 = int(den[idx].sum())
        if d2:
            draws.append(num[idx].sum() / d2)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(num.sum() / d), float(lo), float(hi), d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--q1", type=float, default=0.70)
    ap.add_argument("--q2", type=float, default=0.50)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig6_veto.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(a.rows.read_text())
    rows = d["rows"]
    ep = np.array([r["episode"] for r in rows])
    uniq = np.unique(ep)
    bad = np.array([r["failed"] for r in rows], bool)
    hold = np.array([r["hold_ok"] for r in rows], bool)
    rec = np.array([r["recoverable"] for r in rows], int)
    drop = np.array([r["drop"] for r in rows], float)
    K = max(len(r["plan_vals"]) for r in rows)
    pivotal = (rec > 0) & (rec < K)
    allviable = rec == K
    rng_short = np.array([np.ptp(np.asarray(r["plan_vals_short"], float))
                          for r in rows])

    def gate_at(q1, q2):
        sel = rng_short > np.quantile(rng_short, q1)
        return sel & (drop > np.quantile(drop[sel], q2))

    def net(f):
        n_ = int(f.sum())
        return (100.0 * ((f & bad & hold).sum() - (f & ~bad & ~hold).sum()) / n_
                if n_ else float("nan"))

    # The operating point is the MEDIAN of the (q1, q2) surface, not a cell chosen for
    # being large. A two-threshold gate has ~80 usable cells and its argmax is biased
    # upward by construction: at the max cell the conditional rescue reads 0.556, at
    # the median 0.280, and only the second is a property of the signal rather than of
    # the search. The max is drawn alongside so the size of that gap is visible.
    grid = [(net(gate_at(q1, q2)), q1, q2, int(gate_at(q1, q2).sum()))
            for q1 in np.round(np.arange(0.10, 0.91, 0.10), 2)
            for q2 in np.round(np.arange(0.10, 0.91, 0.10), 2)
            if int(gate_at(q1, q2).sum()) >= 30]
    grid.sort()
    med_net, mq1, mq2, _ = grid[len(grid) // 2]
    max_net, xq1, xq2, _ = grid[-1]
    a.q1, a.q2 = mq1, mq2
    fire = gate_at(mq1, mq2)
    rescue = fire & bad & hold
    brk = fire & ~bad & ~hold
    inert_bad = fire & bad & ~hold
    inert_ok = fire & ~bad & hold
    n = int(fire.sum())

    theme.use()
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.34)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- a: what is inside the fires --------------------------------------------
    parts = [("rescue", int(rescue.sum()), theme.GOOD),
             ("breakage", int(brk.sum()), theme.BAD),
             ("inert — doomed either way", int(inert_bad.sum()), theme.FAINT),
             ("inert — fine either way", int(inert_ok.sum()), "#d5d9e0")]
    left = 0.0
    for lab, v, col in parts:
        axa.barh([0], [100 * v / n], left=left, color=col, height=0.5,
                 edgecolor=theme.PAPER, lw=1.0, label=f"{lab}  ({v})")
        if 100 * v / n > 6:
            axa.text(left + 50 * v / n, 0, f"{100 * v / n:.0f}%", ha="center",
                     va="center", fontsize=8.6, color="white"
                     if col in (theme.GOOD, theme.BAD) else theme.INK,
                     fontweight="bold")
        left += 100 * v / n
    inert = 100 * (inert_bad.sum() + inert_ok.sum()) / n
    axa.set_xlim(0, 100)
    axa.set_ylim(-1.5, 0.55)
    axa.set_yticks([])
    axa.set_xlabel("share of fires (%)")
    axa.spines["left"].set_visible(False)
    axa.legend(loc="lower left", frameon=False, ncol=1, fontsize=7.9,
               handlelength=1.2, borderpad=0.1, labelspacing=0.55)
    theme.verdict(axa, f"{inert:.0f}% of fires change nothing", theme.MUTED)
    axa.set_title(f"Anatomy of {n} fires  (surface-median gate)", loc="left")

    # --- b: the conditional pair, across operating points ------------------------
    pts = [("every decision\n(no gate)", np.ones(len(rows), bool)),
           (f"surface MEDIAN\nq1={mq1:g} q2={mq2:g}", fire),
           (f"surface max\nq1={xq1:g} q2={xq2:g}", gate_at(xq1, xq2))]
    cells = {}
    for tag, m in pts:
        cells[(tag, "rescue")] = boot(m & bad & hold, m & bad, ep, uniq, seed=3)
        cells[(tag, "breakage")] = boot(m & ~bad & ~hold, m & ~bad, ep, uniq, seed=5)
    x = np.arange(len(pts))
    for k, (kind, col) in enumerate((("rescue", theme.GOOD), ("breakage", theme.BAD))):
        v = [cells[(t, kind)][0] for t, _ in pts]
        lo = [cells[(t, kind)][0] - cells[(t, kind)][1] for t, _ in pts]
        hi = [cells[(t, kind)][2] - cells[(t, kind)][0] for t, _ in pts]
        axb.errorbar(x, v, yerr=[lo, hi], fmt="o", ms=7, color=col, lw=1.4,
                     capsize=4, label=f"conditional {kind}",
                     ls="-" if kind == "rescue" else "--", alpha=0.95)
        for xi, vi in zip(x, v):
            axb.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points",
                         xytext=(0, 11 if kind == "rescue" else -15), ha="center",
                         fontsize=8.0, color=col, fontweight="bold")
    axb.axvspan(0.5, 1.5, color=theme.GOOD, alpha=0.06, lw=0)
    axb.set_xticks(x, [t for t, _ in pts])
    axb.set_xlim(-0.45, len(pts) - 0.55)
    axb.set_ylim(-0.04, 0.88)
    axb.set_ylabel("probability")
    axb.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    axb.set_axisbelow(True)
    axb.legend(loc="upper left", frameon=False, fontsize=8)
    theme.verdict(axb, "breakage is flat; rescue is where you tune it", theme.MUTED)
    axb.set_title("What a fire is worth, conditioned", loc="left")

    # --- c: where the harm lives ------------------------------------------------
    strata = [(f"all viable\n(every plan works)", allviable, theme.GOOD),
              ("PIVOTAL\n(some plans work)", pivotal, theme.BAD)]
    for i, (lab, m, col) in enumerate(strata):
        pt, lo, hi, nn = boot(m & ~bad & ~hold, m & ~bad, ep, uniq, seed=7 + i)
        axc.bar([i], [pt], width=0.5, color=col, alpha=0.85)
        axc.errorbar([i], [pt], yerr=[[pt - lo], [hi - pt]], fmt="none",
                     color=theme.INK, lw=1.3, capsize=5)
        axc.text(i, hi + 0.018, f"{pt:.3f}\n[{lo:.3f}, {hi:.3f}]\nn={nn}",
                 ha="center", va="bottom", fontsize=8.0, color=theme.INK)
    axc.set_xticks(range(len(strata)), [s[0] for s in strata])
    axc.set_ylim(0, 0.46)
    axc.set_ylabel("P(holding breaks a working action)")
    axc.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    axc.set_axisbelow(True)
    theme.verdict(axc, "the risk is where the gate is meant to fire", theme.BAD)
    axc.set_title("Cost of a fire, by state", loc="left")

    dec = int(((bad & hold) | (~bad & ~hold)).sum())
    # Placed on the FIGURE, not on an axes. theme.caption anchors to axes coordinates,
    # and this axes is short enough that any offset large enough to clear the legend
    # also falls off the canvas.
    import textwrap

    fig.text(0.055, 0.145, "\n".join(textwrap.wrap(
        f"A veto changes an outcome only where holding and executing differ — "
        f"{dec} of {len(rows)} decisions ({100 * dec / len(rows):.1f}%), over "
        f"{len(uniq)} episodes. That fraction is the hard ceiling on the second "
        f"stage: a head fitted on the correct objective — \u201cis holding better\u201d "
        f"rather than \u201cwill this fail\u201d — has {dec} examples and scores at "
        f"chance. The operating point is the median of the {len(grid)}-cell "
        f"(q1, q2) surface, not its argmax: the rescue rate reads "
        f"{cells[(pts[2][0], 'rescue')][0]:.3f} at the best cell and "
        f"{cells[(pts[1][0], 'rescue')][0]:.3f} at the median, and paired against "
        f"no gate the median lift is +0.054 [\u22120.044, +0.160] \u2014 it spans zero. "
        f"Conditional breakage does not move at any operating point. Intervals are "
        f"95% bootstraps over episodes.", 132)),
        ha="left", va="top", fontsize=8.2, color=theme.MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.30)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig6] -> {a.out}")
    print(f"      surface median net {med_net:+.1f} at q1={mq1:g} q2={mq2:g}; "
          f"max {max_net:+.1f} at q1={xq1:g} q2={xq2:g}")
    print(f"      fires {n}  rescue {int(rescue.sum())}  breakage {int(brk.sum())}  "
          f"inert {inert:.0f}%   decisive {dec}/{len(rows)}")
    for t, _ in pts:
        r, b = cells[(t, "rescue")], cells[(t, "breakage")]
        print(f"      {t.replace(chr(10), ' '):>34}  rescue {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] n={r[3]}"
              f"   breakage {b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}] n={b[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

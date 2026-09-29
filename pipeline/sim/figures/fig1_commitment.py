#!/usr/bin/env python
"""Panel 1 — the commitment fracture, and why it sits where it does.

Three rows on one x-axis, because the point is that they share it. The axis is a number
of simulation steps: in the top row that is how long a waypoint stage lasts, and in the
lower two it is how long a chosen plan is held before the controller re-decides. Those
are the same units, and putting them on the same scale is the whole argument — a
commitment shorter than a stage cannot carry a plan through the phase it is in, so the
arm is redirected mid-descent, mid-lift, mid-transfer, every time.

The two marked values are the ones the closed loop was measured at either side of. Every
number here is measured, not modelled: the dwells from 80 demonstrations, the success
rates from 24 episodes per cell with bootstrap CIs over episodes.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig1_commitment \
        --dir <scratchpad>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

MARK = (16, 32)
POLICY_FIRST = 0.958


def load_sweep(d: Path):
    """(K, success, lo, hi) per rule, plus the long-horizon control points.

    K is the commitment — steps executed before re-deciding — and the main curve takes
    the shortest chunk that can supply it, so K and the chunk length move together
    above 8. That would confound commitment with lookahead, which is why the control
    exists: the same commitments measured at a fixed 32-step chunk. If the two agree,
    the curve is about commitment.
    """
    cells: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    for f in sorted(d.glob("k_*.json")):
        m = re.match(r"k_(best|oracle)_(\d+)_(\d+)\.json$", f.name)
        if not m:
            continue
        rule, horizon, execute = m.group(1), int(m.group(2)), int(m.group(3))
        j = json.loads(f.read_text())
        lo, hi = j.get("success_ci", (float("nan"),) * 2)
        cells[(rule, horizon, execute)] = (j["success_rate"], lo, hi)

    main: dict[str, list] = {}
    control: dict[str, list] = {}
    for (rule, horizon, execute), v in cells.items():
        if horizon == max(8, execute):
            main.setdefault(rule, []).append((execute, *v))
        elif horizon == 32:
            control.setdefault(rule, []).append((execute, *v))
    return ({k: sorted(v) for k, v in main.items()},
            {k: sorted(v) for k, v in control.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--dwell", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig1_commitment.png"))
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pipeline.sim.figures import theme
    from pipeline.sim.policy import ScriptedPickPlace

    dwell = json.loads((a.dwell or a.dir / "dwell.json").read_text())["runs"]
    sweep, control = load_sweep(a.dir)
    allv = np.concatenate([np.asarray(v) for v in dwell.values()])
    median = float(np.median(allv))

    theme.use()
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 8.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.25, 0.85, 1.05],
                                              hspace=0.22), dpi=a.dpi)
    ax_d, ax_f, ax_s = axes
    for ax in axes:
        for k in MARK:
            ax.axvline(k, color=theme.INK, lw=1.0, alpha=0.30, ls=(0, (5, 4)),
                       zorder=0)

    # ── top: how long each waypoint stage lasts ──────────────────────────────
    order = [s for s in ScriptedPickPlace.STAGES if s in dwell][::-1]
    for i, s in enumerate(order):
        v = np.asarray(dwell[s])
        lo, hi = np.percentile(v, [10, 90])
        ax_d.plot([lo, hi], [i, i], color=theme.COOL, lw=7, solid_capstyle="round",
                  alpha=0.30, zorder=2)
        ax_d.plot([np.median(v)], [i], "o", color=theme.COOL, ms=6.5, zorder=3)
        ax_d.text(hi * 1.06, i, f"{np.median(v):.0f}", va="center", fontsize=8,
                  color=theme.MUTED)
    ax_d.set_yticks(range(len(order)))
    ax_d.set_yticklabels(order, fontsize=8.5)
    ax_d.set_ylim(-1.15, len(order) - 0.35)
    ax_d.axvline(median, color=theme.WARM, lw=1.4)
    ax_d.text(median * 1.08, -0.55, f"median {median:.0f}", ha="left", va="center",
              fontsize=8.6, color=theme.WARM, fontweight="bold")
    ax_d.set_title("Waypoint stage duration   ·   bar p10–p90, dot median",
                   loc="left", fontsize=10.5)


    # ── middle: what fraction of stages a commitment of K outlasts ───────────
    ks = np.arange(1, 65)
    frac = [(allv <= k).mean() for k in ks]
    ax_f.plot(ks, np.asarray(frac) * 100, color=theme.INK, lw=1.8)
    for k in MARK:
        y = float((allv <= k).mean()) * 100
        ax_f.plot([k], [y], "o", color=theme.INK, ms=6, zorder=4)
        ax_f.annotate(f"{y:.1f}%", xy=(k, y), xytext=(k * 0.60, y + 16),
                      fontsize=9.5, color=theme.INK, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=theme.FAINT, lw=0.9))
    ax_f.set_ylim(-6, 104)
    ax_f.set_ylabel("stages a choice\noutlasts  (%)")
    ax_f.set_title("A plan held for K steps survives this share of the stage it is in",
                   loc="left", fontsize=10.5)

    # ── bottom: what the closed loop actually does ───────────────────────────
    style = {"oracle": dict(color=theme.BAD, marker="s",
                            label="perfect selection (oracle)"),
             "best": dict(color=theme.COOL, marker="o", label="model scorer")}
    for rule in ("oracle", "best"):
        pts = sweep.get(rule, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        err = [[p[1] * 100 - p[2] * 100 for p in pts],
               [p[3] * 100 - p[1] * 100 for p in pts]]
        ax_s.errorbar(xs, ys, yerr=err, capsize=3, lw=1.8, ms=5.5, **style[rule])
        for e, s, _, _ in control.get(rule, []):
            ax_s.plot([e], [s * 100], marker=style[rule]["marker"], ms=7.5,
                      mfc="none", mec=style[rule]["color"], mew=1.4, ls="none",
                      zorder=5)
    ax_s.axhline(POLICY_FIRST * 100, color=theme.GOOD, ls="--", lw=1.4)
    ax_s.text(60, POLICY_FIRST * 100 + 3, "just run the policy — 95.8%",
              color=theme.GOOD, fontsize=9, fontweight="bold", ha="right")
    ax_s.set_ylim(-6, 104)
    ax_s.set_ylabel("episode success  (%)")
    ax_s.set_xlabel("K — steps of the chosen plan executed before re-deciding   "
                    "(top row is stage duration, in the same units)")
    ax_s.set_title("Success is flat at zero until the commitment outlasts a stage",
                   loc="left", fontsize=10.5)
    ax_s.plot([], [], marker="o", ms=7.5, mfc="none", mec=theme.MUTED, mew=1.4,
              ls="none", label="same K at a fixed 32-step chunk (control)")
    ax_s.legend(frameon=False, loc="upper left", fontsize=8.4,
                bbox_to_anchor=(0.0, 0.90), labelspacing=0.35)
    ax_s.set_xscale("log", base=2)
    ax_s.set_xlim(0.9, 70)
    ax_s.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax_s.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for ax in axes:
        ax.grid(axis="x", alpha=0.18, lw=0.6)

    fig.suptitle("The controller re-decides faster than the task changes",
                 fontsize=13.5, fontweight="semibold", y=0.975)
    fig.text(0.5, 0.062,
             "A 32-step chunk committed for only 4 steps also scores 0.0% — the "
             "recovery is the commitment, not the longer lookahead.\n"
             "Even fully committed, perfect selection reaches 37.5% [16.7, 58.3] "
             "against 95.8% for not selecting at all.",
             ha="center", va="top", fontsize=8.8, color=theme.MUTED, linespacing=1.7)
    fig.subplots_adjust(top=0.915, bottom=0.165, left=0.135, right=0.965)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"[fig1] median dwell {median:.0f}; "
          f"K=16 outlasts {(allv <= 16).mean()*100:.1f}%, "
          f"K=32 outlasts {(allv <= 32).mean()*100:.1f}%")
    for rule, pts in sweep.items():
        print(f"[fig1] {rule}: " + "  ".join(f"K={p[0]}:{p[1]*100:.1f}%" for p in pts))
    for rule, pts in control.items():
        print(f"[fig1] {rule} control (chunk 32): "
              + "  ".join(f"K={p[0]}:{p[1]*100:.1f}%" for p in pts))
    print(f"[fig1] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Panel 1 — what selection is worth, measured against its own ceiling.

Every arm below runs the same policy on the same task with the same commitment
schedule. The only thing that changes is who picks the action.

The chart exists because "the scorer ranks badly" and "ranking cannot help here" look
identical from a success rate and have opposite fixes. Putting a perfect selector on
the same axis separates them: `deep-oracle` rewinds every candidate to the end of the
episode and keeps the one the simulator says finishes, so it is the highest score any
ranking rule over this candidate set can reach. It lands on the policy's own number.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig1_selection \
        --runs <scratchpad>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.sim.figures import theme

#: (file, label, sub-label, which family it belongs to)
#: Families decide colour, and they are the reading of the panel: two arms have
#: perfect or no selection and agree; two have a fallible selector and do not.
ARMS = [
    ("s96_ep32_first.json", "no selection",
     "run the policy's own chunk", "ceiling"),
    ("x_ep32_deep.json", "perfect-information selection",
     "rewind all 8 to the end, keep the winner", "ceiling"),
    ("e96_ep32.json", "chunk-depth heuristic",
     "rank by a distance cost on privileged state", "fallible"),
    ("s96_ep32_best.json", "learned model selection",
     "rank by the world model's predicted return", "fallible"),
    ("k4_oracle_1.json", "heuristic, re-decided every step",
     "the same cost, K=1", "fallible"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig1_selection.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = []
    for f, label, sub, fam in ARMS:
        p = a.runs / f
        if not p.exists():
            raise SystemExit(f"missing {p}")
        d = json.loads(p.read_text())
        budget = d["task"].get("max_env_steps",
                               d.get("config", {}).get("max_env_steps"))
        # Same guard as the decomposition panel, for the same reason ([S21]): these
        # arms are only a comparison if the task was the same one.
        if int(budget) != 360:
            raise SystemExit(f"{f}: step budget {budget} != 360, not comparable")
        ci = d.get("success_ci") or (d["success_rate"],) * 2
        eps = d["episodes"]
        over = [e for e in eps if (e.get("chosen_idx") or [0])[0] != 0]
        arms.append({
            "label": label, "sub": sub, "fam": fam,
            "v": d["success_rate"] * 100,
            "lo": ci[0] * 100, "hi": ci[1] * 100,
            "n": len(eps),
            "override": 100 * len(over) / max(1, len(eps)),
        })

    theme.use()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(arms))[::-1]
    ceiling = max(x["v"] for x in arms if x["fam"] == "ceiling")

    ax.axvline(ceiling, color=theme.GOOD, lw=1.1, ls="--", alpha=0.55, zorder=1)
    ax.text(ceiling, len(arms) - 0.33, f"ceiling {ceiling:.1f}%  ", va="bottom",
            ha="right", fontsize=8.2, color=theme.GOOD, fontweight="bold")

    for yi, arm in zip(y, arms):
        col = theme.GOOD if arm["fam"] == "ceiling" else theme.BAD
        # A lollipop rather than a bar: the quantity is a point estimate with an
        # interval, and a filled bar from zero implies a magnitude the interval does
        # not support.
        ax.plot([0, arm["v"]], [yi, yi], color=col, lw=1.4, alpha=0.30, zorder=2)
        ax.plot([arm["lo"], arm["hi"]], [yi, yi], color=col, lw=3.2, alpha=0.30,
                solid_capstyle="round", zorder=3)
        ax.plot([arm["v"]], [yi], "o", color=col, ms=9, zorder=4)
        # Labels flip to the inside once the dot is near the right edge, so a high
        # score never has its own number pushed off the figure.
        inside = arm["v"] > 70
        dx, ha = (-3.0, "right") if inside else (3.0, "left")
        ax.text(arm["v"] + dx, yi + 0.18, f"{arm['v']:.1f}%", va="center",
                ha=ha, fontsize=9.6, color=col, fontweight="bold")
        ax.text(arm["v"] + dx, yi - 0.23,
                f"[{arm['lo']:.1f}, {arm['hi']:.1f}]  n={arm['n']}", va="center",
                ha=ha, fontsize=7.4, color=theme.MUTED)

    # The gap that is the point of the panel, drawn between the ceiling and the best
    # fallible selector.
    worst = max((x for x in arms if x["fam"] == "fallible"), key=lambda x: x["v"])
    yi = y[arms.index(worst)]
    ax.annotate("", xy=(ceiling, yi - 0.42), xytext=(worst["v"], yi - 0.42),
                arrowprops=dict(arrowstyle="<->", color=theme.INK, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(ceiling, yi - 0.58,
            f"{ceiling - worst['v']:.1f} pts LOST to selecting",
            ha="right", va="top", fontsize=8.2, color=theme.INK,
            fontweight="semibold")

    ax.set_yticks(y, [x["label"] for x in arms])
    for tick, arm in zip(ax.get_yticklabels(), arms):
        tick.set_fontsize(9.2)
        tick.set_color(theme.INK if arm["fam"] == "ceiling" else theme.MUTED)
    for yi, arm in zip(y, arms):
        ax.text(-3.0, yi - 0.30, arm["sub"], ha="right", va="center",
                fontsize=7.3, color=theme.FAINT)

    ax.set_xlim(-1, 104)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("episodes solved (%)")
    ax.set_ylim(-0.95, len(arms) - 0.25)
    ax.grid(axis="x", color=theme.FAINT, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    theme.verdict(ax, "selection has no headroom to win")
    ax.set_title("Same policy, same task — only the chooser changes", loc="left")
    theme.caption(ax,
                  f"Perfect episode-depth selection scores {ceiling:.1f}%, identical "
                  f"to running the policy with no selection at all: a candidate that "
                  f"reaches success exists at 100% of decisions, and the policy's own "
                  f"chunk is already one of them. The chunk-depth heuristic's "
                  f"{worst['v']:.1f}% is {ceiling - worst['v']:.1f} points below both "
                  f"— it was not failing to find the best candidate, it was overriding "
                  f"a policy that was already right, which it does on "
                  f"{worst['override']:.1f}% of episodes.", y=-0.26,
                  width=96)

    fig.subplots_adjust(left=0.295, right=0.975, top=0.845, bottom=0.315)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig] -> {a.out}")
    for arm in arms:
        print(f"      {arm['label']:38} {arm['v']:5.1f}% "
              f"[{arm['lo']:5.1f}, {arm['hi']:5.1f}]  override {arm['override']:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

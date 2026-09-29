#!/usr/bin/env python
"""Success against commitment length, for the model scorer and for a perfect one.

The trace comparison said greedy selection fails because the controller re-decides
every `execute` steps and the arm never carries a plan out. If that is the mechanism,
success should be a function of **how long a choice is held**, and should recover as the
commitment approaches the chunk length. If it is flat, the mechanism is wrong.

Both curves are drawn because they answer different halves. The oracle curve is the
ceiling — if even perfect selection stays at zero until commitment grows, the defect is
the control scheme and not the scorer. The model curve says how much of that ceiling a
learned scorer reaches once the scheme is fixed.

Error bars are 95% bootstrap CIs over episodes, because at 24 episodes one flipped
outcome is 4.2 points and a curve drawn without them invites reading noise as a trend.

    conda run -n simple_bev_vldrive python -m pipeline.sim.plot_ksweep \
        --glob 'k_*_*.json' --dir <scratchpad> --out pipeline/sim/assets/ksweep.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/ksweep.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float, float, float, int]]] = {}
    for f in sorted(a.dir.glob("k_*.json")):
        m = re.match(r"k_(best|oracle)_(\d+)_(\d+)\.json$", f.name)
        if not m:
            continue
        rule, horizon, execute = m.group(1), int(m.group(2)), int(m.group(3))
        d = json.loads(f.read_text())
        lo, hi = d.get("success_ci", (float("nan"),) * 2)
        series.setdefault(rule, []).append(
            (execute, d["success_rate"], lo, hi, horizon))
    if not series:
        raise SystemExit(f"no k_*.json under {a.dir}")

    print(f"{'rule':>7} {'execute':>8} {'horizon':>8} {'success':>9} {'95% CI':>16}")
    for rule in sorted(series):
        for e, s, lo, hi, h in sorted(series[rule]):
            print(f"{rule:>7} {e:8d} {h:8d} {s*100:8.1f}% "
                  f"{f'[{lo*100:.1f}, {hi*100:.1f}]':>16}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    style = {"best": dict(color="#2d6cdf", marker="o", label="model scorer"),
             "oracle": dict(color="#c2410c", marker="s",
                            label="perfect selection (oracle)")}
    for rule in sorted(series, reverse=True):
        pts = sorted(series[rule])
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        err = [[p[1] * 100 - p[2] * 100 for p in pts],
               [p[3] * 100 - p[1] * 100 for p in pts]]
        ax.errorbar(xs, ys, yerr=err, capsize=3, lw=1.8, ms=5,
                    **style.get(rule, dict(label=rule)))
    ax.axhline(95.8, color="#16a34a", ls="--", lw=1.4)
    ax.text(1.05, 96.8, "policy-first, 95.8%", color="#16a34a", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([p[0] for p in sorted(series[sorted(series)[0]])])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("steps executed per decision  (commitment length K)")
    ax.set_ylabel("episode success  (%)")
    ax.set_title("Success recovers with commitment, not with a better scorer")
    ax.set_ylim(-4, 104)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

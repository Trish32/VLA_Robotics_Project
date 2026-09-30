#!/usr/bin/env python
"""Panel 7 — the same signal, asked two questions, answers in opposite directions.

Every gate built on this arena fires when a signal is HIGH. That is the right shape only
if the signal rises where the decision matters. Asking each signal two questions instead
of one shows that for the change detectors it does the opposite:

  **(a)** is this decision decisive?          scored over all decisions
  **(b)** is this the START of the run?       scored among decisive decisions only

On (a) the detectors sit at chance — they carry no information about whether a decision
matters. On (b) they sit near 0.22 on all three seeds, which is not "no signal" but
strong information pointing the wrong way: **when observation change is high, you are in
the middle of a decisive stretch, not at its beginning.**

Read backwards, that is usable, and it inverts the gate: to catch onsets, fire when the
detector is LOW. P19–P21 in PREREGISTERED.md test exactly that on a seed that does not
exist yet — this panel is the observation, not the result.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig7_inversion \
        --sets <scratchpad>/piv7.json <scratchpad>/piv8.json <scratchpad>/piv9.json
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

from pipeline.sim.figures import theme

#: Signals that measure a CHANGE between consecutive decisions, versus those read off a
#: single decision. The split is the panel's hypothesis and is drawn, not asserted.
CHANGE = {"|Δ observation|", "|Δ plan spread|", "|Δ value head|", "model surprise"}


def auc(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(x)
    xs = x[o]
    rk = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        rk[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r = np.empty(len(x), float)
    r[o] = rk
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load(path: Path) -> dict:
    r = json.loads(path.read_text())["rows"]
    o = np.lexsort(([x["step"] for x in r], [x["episode"] for x in r]))
    r = [r[i] for i in o]
    n = len(r)
    ep = np.array([x["episode"] for x in r])
    st = np.array([x["step"] for x in r])
    bad = np.array([x["failed"] for x in r], bool)
    hold = np.array([x["hold_ok"] for x in r], bool)
    sf = np.array([x["state_feat"] for x in r], float)
    v0 = np.array([x["v0"] for x in r], float)
    ve = np.array([x["v_end"] for x in r], float)
    ps = np.array([x["plan_std"] for x in r], float)
    dec = (bad & hold) | (~bad & ~hold)
    prv = np.arange(n)
    okp = np.zeros(n, bool)
    for i in range(1, n):
        if ep[i - 1] == ep[i] and st[i] - st[i - 1] <= 2:
            prv[i], okp[i] = i - 1, True
    pos = np.zeros(n, int)
    for i in range(n):
        if dec[i]:
            pos[i] = pos[prv[i]] + 1 if (okp[i] and dec[prv[i]]) else 1
    d = lambda x: np.where(okp, np.abs(x - x[prv]), 0.0)          # noqa: E731
    sig = {
        "|Δ observation|": np.where(okp, np.linalg.norm(sf - sf[prv], axis=1), 0.0),
        "|Δ plan spread|": d(ps),
        "|Δ value head|": d(v0),
        "model surprise": np.where(okp, np.abs(v0 - ve[prv]), 0.0),
        "plan spread ¼-H": np.array([np.ptp(np.asarray(x["plan_vals_short"], float))
                                     for x in r]),
        "return drop": np.array([x["drop"] for x in r], float),
        "near-contact": -sf[:, 22],
    }
    return dict(sig=sig, dec=dec, onset=dec & (pos == 1), later=dec & (pos > 1),
                ep=ep, n=n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig7_inversion.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = [load(p) for p in a.sets]
    names = list(S[0]["sig"])
    A = {k: [auc(d["sig"][k], d["dec"]) for d in S] for k in names}
    B = {}
    for k in names:
        vals = []
        for d in S:
            m = d["onset"] | d["later"]
            vals.append(auc(d["sig"][k][m], d["onset"][m]))
        B[k] = vals

    theme.use()
    fig = plt.figure(figsize=(12.4, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # --- left: the two questions, one against the other -------------------------
    axl.axhline(0.5, color=theme.FAINT, lw=1.0)
    axl.axvline(0.5, color=theme.FAINT, lw=1.0)
    # The quadrant the change detectors land in is the finding, so it is shaded.
    axl.axhspan(0.10, 0.5, xmin=(0.5 - 0.35) / 0.35, xmax=1.0,
                color=theme.BAD, alpha=0.06, lw=0)
    axl.text(0.685, 0.135, "marks decisive stretches,\npoints at their MIDDLE",
             ha="center", va="center", fontsize=8.0, color=theme.BAD,
             fontweight="semibold")
    for k in names:
        col = theme.BAD if k in CHANGE else theme.COOL
        axl.plot(A[k], B[k], "-", color=col, lw=0.9, alpha=0.35)
        axl.plot(A[k], B[k], "o", color=col, ms=4.5, alpha=0.55)
        axl.plot([np.mean(A[k])], [np.mean(B[k])], "o", color=col, ms=11,
                 markeredgecolor=theme.PAPER, markeredgewidth=1.2, zorder=4)
        # The four change detectors land on top of each other, so their labels are
        # placed by hand rather than by a rule that would stack them.
        off = {"|Δ observation|": (13, 10), "|Δ plan spread|": (13, -4),
               "|Δ value head|": (-13, -14), "model surprise": (-13, 9),
               "plan spread ¼-H": (12, -4), "return drop": (10, -12),
               "near-contact": (-11, 9)}.get(k, (11, 4))
        axl.annotate(k, (np.mean(A[k]), np.mean(B[k])),
                     textcoords="offset points", xytext=off,
                     ha="left" if off[0] > 0 else "right",
                     fontsize=8.0, color=col,
                     fontweight="bold" if k in CHANGE else "normal")
    axl.set_xlim(0.35, 0.70)
    axl.set_ylim(0.10, 0.62)
    axl.set_xlabel("(a)  AUC — is this decision decisive?")
    axl.set_ylabel("(b)  AUC — is this the START of the run?")
    axl.grid(color=theme.FAINT, alpha=0.30, lw=0.6)
    axl.set_axisbelow(True)
    theme.verdict(axl, "change detectors invert between the two questions")
    axl.set_title(f"Change signals (red) vs single-decision signals (blue), "
                  f"{len(S)} seeds", loc="left")

    # --- right: how stable each is across seeds ---------------------------------
    order = sorted(names, key=lambda k: -np.std([
        (lambda d: d["dec"][d["sig"][k] > np.quantile(d["sig"][k], 0.90)].mean()
         / d["dec"].mean())(d) for d in S], ddof=1))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        lifts = [d["dec"][d["sig"][k] > np.quantile(d["sig"][k], 0.90)].mean()
                 / d["dec"].mean() for d in S]
        col = theme.BAD if k in CHANGE else theme.COOL
        axr.plot([min(lifts), max(lifts)], [yi, yi], color=col, lw=3.0, alpha=0.30,
                 solid_capstyle="round")
        axr.plot(lifts, [yi] * len(lifts), "o", color=col, ms=5, alpha=0.65)
        axr.plot([np.mean(lifts)], [yi], "|", color=col, ms=14, mew=2.2)
        axr.text(1.72, yi, f"SD {np.std(lifts, ddof=1):.2f}", va="center",
                 ha="right", fontsize=7.8, color=theme.MUTED)
    axr.axvline(1.0, color=theme.INK, lw=1.0, alpha=0.55)
    axr.axvline(1.2, color=theme.BAD, lw=1.0, ls="--", alpha=0.8)
    axr.text(1.21, -0.72, "P12 bar", fontsize=7.6, color=theme.BAD,
             ha="left", va="bottom")
    axr.set_yticks(y, order)
    for tick, k in zip(axr.get_yticklabels(), order):
        tick.set_fontsize(8.2)
        tick.set_color(theme.BAD if k in CHANGE else theme.COOL)
    axr.set_xlim(0.35, 1.75)
    axr.set_ylim(-0.95, len(order) - 0.4)
    axr.set_xlabel("lift in P(decisive) at the 90th percentile, one point per seed")
    axr.grid(axis="x", color=theme.FAINT, alpha=0.30, lw=0.6)
    axr.set_axisbelow(True)
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", length=0)
    theme.verdict(axr, "the two that broke P12 are the two least stable", theme.MUTED)
    axr.set_title("Across-seed spread of the same quantity", loc="left")

    fig.text(0.035, 0.135, "\n".join(textwrap.wrap(
        "Every gate in this project fires when a signal is HIGH, which is the right "
        "shape only if the signal rises where the decision matters. The change "
        "detectors sit at chance on (a) — they say nothing about whether a decision is "
        "decisive — and near 0.22 on (b) on all three seeds, which is not absence of "
        "signal but strong signal pointing the wrong way: high observation change means "
        "mid-burst, not onset. Read backwards that inverts the gate, and the two "
        "triggers that falsified P12 at 1.32× are also the two least stable across "
        "seeds (SD 0.45 and 0.30 against 0.03 for the steadiest). Neither observation "
        "is a result yet: P19–P21 test the inversion and P16–P18 the stability, both "
        "registered before the seeds they need exist.", 148)),
        ha="left", va="top", fontsize=8.1, color=theme.MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.30)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig7] -> {a.out}")
    for k in names:
        print(f"      {k:>18}  (a) {np.mean(A[k]):.3f}   (b) {np.mean(B[k]):.3f}"
              f"   inverted in "
              f"{sum(1 for i in range(len(S)) if A[k][i] > 0.5 and B[k][i] < 0.5)}"
              f"/{len(S)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

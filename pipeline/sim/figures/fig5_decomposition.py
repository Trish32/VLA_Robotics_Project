#!/usr/bin/env python
"""The commitment curve, decomposed into the two failures that make it a curve.

Plotted as one success rate against K, the commitment sweep looks like a tuning
problem with an optimum near K=32. It is not one curve. Every failure in this arena
ends at the step cap, so nothing here "crashes" — but the episodes that time out at
K=1 and the ones that time out at K=64 have stopped in different places, and the
stage the chunker was last committed from says which:

  * **reach**  (hover, descend)   — never arrived at the cube.
  * **grasp**  (close, lift)      — arrived, never got it off the table.
  * **carry**  (transfer, lower, release, retreat) — had the cube, ran out of clock.

Low K thrashes: it re-decides so often that it crawls, and it dies mid-carry with the
cube in the gripper. High K commits the grasp offset sixty-four steps ahead and cannot
correct it, so it dies at the lift. K=32 is not an optimum of one process, it is the
crossing point of two — which is why reporting a single peak hid the mechanism.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig5_decomposition
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.sim.figures import theme

#: Which failure each terminal stage belongs to, and the order they are stacked in.
FAMILY = {
    "hover": "reach", "descend": "reach",
    "close": "grasp", "lift": "grasp",
    "transfer": "carry", "lower": "carry", "release": "carry", "retreat": "carry",
}
ORDER = ("reach", "grasp", "carry")
COLOUR = {"reach": theme.FAINT, "grasp": theme.BAD, "carry": theme.COOL}
LABEL = {
    "reach": "reach — never arrived at the cube",
    "grasp": "grasp — arrived, never lifted it",
    "carry": "carry — had the cube, ran out of clock",
}


def load(path: Path, budget: int) -> dict:
    """One run, with its comparability checked rather than assumed.

    The budget is verified here because it is the field that produced [S21]: two runs
    at 360 and 480 steps printed the same task hash, and the K=16 cell moved from 18.8%
    to 83.3% between them. A figure that silently plots both is the same bug wearing a
    chart.
    """
    d = json.loads(path.read_text())
    got = d["task"].get("max_env_steps", d.get("config", {}).get("max_env_steps"))
    if got is None:
        raise SystemExit(f"{path.name}: no recorded step budget — cannot place it "
                         f"on a curve. Re-run it.")
    if int(got) != budget:
        raise SystemExit(f"{path.name}: budget {got} != {budget}. Not comparable "
                         f"([S21]); re-run rather than plotting it.")
    eps = d["episodes"]
    fails = [e for e in eps if not e["success"]]
    fam = Counter(FAMILY.get((e.get("commit_stage") or ["?"])[-1], "reach")
                  for e in fails)
    # Per-episode family label, kept so the bands can be resampled. Summarising to
    # three shares here would leave nothing to bootstrap from.
    per_ep = [None if e["success"]
              else FAMILY.get((e.get("commit_stage") or ["?"])[-1], "reach")
              for e in eps]
    return {
        "n": len(eps),
        "success": d["success_rate"],
        "ci": d.get("success_ci"),
        # Shares of ALL episodes, so the three families and the success rate sum to 1
        # and the panels can be read against each other.
        "share": {k: fam.get(k, 0) / max(1, len(eps)) for k in ORDER},
        "per_ep": per_ep,
        "steps_ok": [e["steps"] for e in eps if e["success"]],
    }


def zero_between(d0: float, d1: float, a: float, b: float) -> float:
    """Where a straight line from (a, d0) to (b, d1) crosses zero.

    Done by hand rather than with `np.interp`, which requires its sample points to be
    increasing and silently returns the wrong endpoint when they are not. The
    difference carry-minus-grasp is decreasing exactly where the crossing is, so every
    crossing in this figure hits that case: the first version reported the crossover at
    K=64, the right edge of the sweep, when the data put it near 36.
    """
    if d0 == d1:
        return a
    return a + (b - a) * (d0 / (d0 - d1))


def resample(runs, ks, reps=4000, seed=0):
    """Episode bootstrap for the family shares and for where the curves cross.

    Each K is its own run, so each is resampled independently — pooling them would
    imply one sample split across K, which it is not.

    The crossing is the quantity the panel exists to assert, so it gets an interval
    like everything else. It is found per replicate rather than by perturbing the
    point estimate: the crossing is a nonlinear read-off of two noisy curves, and a
    band drawn around each curve separately says nothing about where they meet. Some
    replicates have no crossing at all, and that fraction is reported rather than
    dropped — "the curves cross somewhere in here, 94% of the time" and "the curves
    cross here" are different claims.
    """
    rng = np.random.default_rng(seed)
    xs = np.log2(np.array(ks, float))
    bands = {f: np.zeros((reps, len(ks))) for f in ORDER}
    succ = np.zeros((reps, len(ks)))
    crossings = []
    for r in range(reps):
        carry = np.zeros(len(ks))
        grasp = np.zeros(len(ks))
        for j, k in enumerate(ks):
            lab = runs[k]["per_ep"]
            pick = rng.integers(0, len(lab), len(lab))
            drawn = [lab[i] for i in pick]
            n = len(drawn)
            for f in ORDER:
                bands[f][r, j] = sum(x == f for x in drawn) / n
            succ[r, j] = sum(x is None for x in drawn) / n
            carry[j], grasp[j] = bands["carry"][r, j], bands["grasp"][r, j]
        diff = carry - grasp
        sign = np.where(np.diff(np.sign(diff)))[0]
        if len(sign):
            i = int(sign[0])
            crossings.append(zero_between(diff[i], diff[i + 1], xs[i], xs[i + 1]))
    crossings = np.array(crossings)
    return {
        "bands": {f: np.percentile(v, [2.5, 97.5], axis=0) for f, v in bands.items()},
        "succ": np.percentile(succ, [2.5, 97.5], axis=0),
        "cross_lo": float(2 ** np.percentile(crossings, 2.5)) if len(crossings) else None,
        "cross_hi": float(2 ** np.percentile(crossings, 97.5)) if len(crossings) else None,
        "cross_frac": len(crossings) / reps,
        "cross_log2": crossings,
        "modes": modality(crossings, xs),
    }


def modality(crossings: np.ndarray, xs: np.ndarray) -> dict:
    """Is the crossover one place, or two places being averaged into one?

    A 95% interval summarises a bootstrap distribution as if it had a single centre.
    If the replicates actually pile up at two separate K values — some near 20, some
    near 50 — then the interval spans a gap the data never visits, the midpoint names
    a crossover that no replicate produced, and "the curves cross here" is the wrong
    sentence regardless of how tight the interval looks.

    Detection is deliberately blunt: bin in log2 K, find local maxima, and keep a
    second mode only if the valley between the two peaks drops below 60% of the
    smaller peak and the peaks sit at least half a K-doubling apart. A softer rule
    reports ripple as structure; this one reports a gap.
    """
    if len(crossings) < 50:
        return {"n_modes": 0, "peaks": [], "bimodal": False, "sep": 0.0}
    bins = np.linspace(xs.min(), xs.max(), 25)
    h, edges = np.histogram(crossings, bins=bins)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    peaks = [i for i in range(1, len(h) - 1)
             if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] > 0]
    if h[0] > h[1]:
        peaks = [0] + peaks
    if h[-1] > h[-2]:
        peaks = peaks + [len(h) - 1]
    peaks.sort(key=lambda i: -h[i])
    kept = []
    for i in peaks:
        if all(abs(ctr[i] - ctr[j]) >= 0.5 for j in kept):
            kept.append(i)
    kept.sort()
    bimodal, sep = False, 0.0
    if len(kept) >= 2:
        # The two TALLEST surviving peaks, not the two leftmost. `kept` is sorted by
        # position for reporting, so indexing it directly compared the valley against
        # whichever peak came first on the axis — which let a stray bin holding under
        # 5% of the mass be reported as a second mode.
        tall = sorted(kept, key=lambda i: -h[i])[:2]
        a_, b_ = sorted(tall)
        valley = h[a_:b_ + 1].min()
        smaller = min(h[a_], h[b_])
        sep = 1.0 - (valley / smaller if smaller else 1.0)
        # A real second mode has to hold real mass AND sit behind a real valley.
        bimodal = smaller >= 0.10 * h.max() and sep >= 0.40
    return {"n_modes": len(kept), "bimodal": bool(bimodal), "sep": float(sep),
            "peaks": [float(2 ** ctr[i]) for i in kept],
            "mass": [float(h[i] / h.sum()) for i in kept]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding k4_oracle_<K>.json")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--budget", type=int, default=360)
    ap.add_argument("--episode-commit", type=Path, nargs="*", default=(),
                    help="e96_ep<H>.json — the commit-once-per-episode limit")
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig5_decomposition.png"))
    a = ap.parse_args(argv)

    import matplotlib.pyplot as plt

    runs = {}
    for k in a.ks:
        p = a.runs / f"k4_oracle_{k}.json"
        if p.exists():
            runs[k] = load(p, a.budget)
    if not runs:
        raise SystemExit(f"no k4_oracle_*.json under {a.runs}")
    ks = sorted(runs)
    x = np.arange(len(ks))
    bs = resample(runs, ks)

    theme.use()
    fig = plt.figure(figsize=(8.2, 7.0))
    # Episode-commit gets its own axis rather than a marker on the K axis. K is a
    # re-decision period and H is an open-loop segment length under a single decision;
    # they are different quantities and the peaks sit at different values, which is
    # exactly the thing a shared axis would hide.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                          width_ratios=[2.5, 1.0], hspace=0.42, wspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    axe = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, :])

    # --- top left: the fixed-K curve --------------------------------------------
    succ = np.array([runs[k]["success"] for k in ks]) * 100
    lo = np.array([runs[k]["ci"][0] if runs[k]["ci"] else runs[k]["success"]
                   for k in ks]) * 100
    hi = np.array([runs[k]["ci"][1] if runs[k]["ci"] else runs[k]["success"]
                   for k in ks]) * 100
    blo, bhi = bs["succ"] * 100
    ax0.fill_between(x, np.minimum(lo, blo), np.maximum(hi, bhi),
                     color=theme.GOOD, alpha=0.13, lw=0)
    ax0.plot(x, succ, "-o", color=theme.GOOD, ms=5, lw=1.8, zorder=3)
    best = int(np.argmax(succ))
    ax0.annotate(f"peak K={ks[best]}\n{succ[best]:.1f}%", (x[best], succ[best]),
                 textcoords="offset points", xytext=(-4, 10), ha="right",
                 fontsize=8.5, color=theme.GOOD, fontweight="bold")
    ax0.set_ylabel("episodes solved (%)")
    ax0.set_xlabel("K — steps per re-decision")
    ax0.set_xticks(x, [str(k) for k in ks])
    ax0.set_ylim(-5, 126)
    ax0.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    ax0.set_axisbelow(True)
    ax0.set_title(f"re-decide every K steps  (n={runs[ks[0]]['n']})", loc="left")

    # --- top right: commit once, vary only the open-loop segment ----------------
    pts = []
    for q in a.episode_commit or ():
        q = Path(q)
        if not q.exists():
            continue
        m = re.search(r"ep(\d+)", q.stem)
        if not m:
            raise SystemExit(f"{q.name}: cannot read a horizon from the name; "
                             f"expected ...ep<H>...")
        r = load(q, a.budget)
        pts.append((int(m.group(1)), r))
    pts.sort()
    if pts:
        hx = np.arange(len(pts))
        hv = np.array([r["success"] for _, r in pts]) * 100
        hlo = np.array([r["ci"][0] if r["ci"] else r["success"] for _, r in pts]) * 100
        hhi = np.array([r["ci"][1] if r["ci"] else r["success"] for _, r in pts]) * 100
        axe.fill_between(hx, hlo, hhi, color=theme.WARM, alpha=0.15, lw=0)
        axe.plot(hx, hv, "-D", color=theme.WARM, ms=5, lw=1.8)
        hb = int(np.argmax(hv))
        axe.annotate(f"H={pts[hb][0]}  {hv[hb]:.1f}%", (hx[hb], hv[hb]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8.2, color=theme.WARM, fontweight="bold")
        # The zig-zag IS the result: a genuine commitment-length effect would be
        # monotone or single-peaked. This series tracks how often the selector
        # overrode the policy, and nothing else.
        axe.annotate("not a curve — tracks how often\nthe selector overrode the "
                     "policy\n(r = \u22120.955 across H)", (0.5, 0.06),
                     xycoords="axes fraction", ha="center", va="bottom",
                     fontsize=7.4, color=theme.MUTED)
        axe.set_xticks(hx, [str(h) for h, _ in pts])
        axe.set_xlabel("H — open-loop segment")
        axe.set_ylim(-5, 126)
        axe.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
        axe.set_axisbelow(True)
        axe.set_title(f"decide once  (n={pts[0][1]['n']})", loc="left")
    theme.verdict(ax0, "one curve, two mechanisms")

    # --- bottom: what the failures actually were --------------------------------
    bottom = np.zeros(len(ks))
    for fam in ORDER:
        v = np.array([runs[k]["share"][fam] for k in ks]) * 100
        ax1.bar(x, v, bottom=bottom, width=0.58, color=COLOUR[fam],
                label=LABEL[fam], edgecolor=theme.PAPER, lw=0.8, alpha=0.85)
        bottom += v

    carry = np.array([runs[k]["share"]["carry"] for k in ks]) * 100
    grasp = np.array([runs[k]["share"]["grasp"] for k in ks]) * 100
    for v, col, fam in ((carry, theme.COOL, "carry"), (grasp, theme.BAD, "grasp")):
        b = bs["bands"][fam] * 100
        ax1.fill_between(x, b[0], b[1], color=col, alpha=0.22, lw=0, zorder=3.5)
        ax1.plot(x, v, "--o", color=col, ms=4, lw=1.6, zorder=4)

    cross = np.where(np.diff(np.sign(carry - grasp)))[0]
    xs_log = np.log2(np.array(ks, float))
    if len(cross) and bs["cross_lo"] is not None:
        i = int(cross[0])
        xc = zero_between(carry[i] - grasp[i], carry[i + 1] - grasp[i + 1],
                          float(x[i]), float(x[i + 1]))
        xlo = float(np.interp(np.log2(bs["cross_lo"]), xs_log, x))
        xhi = float(np.interp(np.log2(bs["cross_hi"]), xs_log, x))
        ax1.axvspan(xlo, xhi, color=theme.INK, alpha=0.08, lw=0, zorder=1)
        ax1.axvline(xc, color=theme.INK, lw=1.0, ls=":", alpha=0.75, zorder=5)
        kc = 2 ** float(np.interp(xc, x, xs_log))
        m = bs["modes"]
        ax1.annotate("too slow   |   too committed", (xc, 108), ha="center",
                     fontsize=8.4, color=theme.INK, fontweight="semibold")
        if m["bimodal"]:
            head = ("crossover is BIMODAL — "
                    + " and ".join(f"K\u2248{k:.0f}" for k in m["peaks"][:2]))
            ax1.annotate(head, (xc, 99), ha="center", fontsize=7.9,
                         color=theme.BAD, fontweight="bold")
        else:
            ax1.annotate(f"crossover K\u2248{kc:.0f}   95% CI "
                         f"[{bs['cross_lo']:.0f}, {bs['cross_hi']:.0f}]",
                         (xc, 99), ha="center", fontsize=7.9, color=theme.MUTED)
        q = 2 ** np.percentile(bs["cross_log2"], [5, 25, 50, 75, 95])
        ax1.annotate("deciles  " + "  ".join(f"p{p_}={v:.0f}" for p_, v in
                                             zip((5, 25, 50, 75, 95), q)),
                     (xc, 92), ha="center", fontsize=7.0, color=theme.MUTED)
        ax1.annotate(f"{bs['cross_frac'] * 100:.0f}% of resamples cross, "
                     f"{100 - bs['cross_frac'] * 100:.0f}% do not",
                     (xc, 86), ha="center", fontsize=7.0, color=theme.MUTED)

        print(f"      crossover K~{kc:.1f}  95% CI [{bs['cross_lo']:.1f}, "
              f"{bs['cross_hi']:.1f}]")
        print(f"      resamples containing a crossing {bs['cross_frac'] * 100:.1f}%  "
              f"(no crossing {100 - bs['cross_frac'] * 100:.1f}%)")
        print("      crossing K deciles  "
              + "  ".join(f"p{p_}={v:.1f}" for p_, v in zip((5, 25, 50, 75, 95), q)))
        print(f"      modality: {m['n_modes']} mode(s), peaks at "
              + ", ".join(f"K~{k:.1f}" for k in m["peaks"])
              + f"; valley depth {m['sep']:.2f} -> "
              + ("BIMODAL — the single-crossover narrative does not hold"
                 if m["bimodal"] else "unimodal, single crossover stands"))

    ax1.set_ylabel("episodes failing this way (%)")
    ax1.set_xlabel("K — steps executed per decision")
    ax1.set_xticks(x, [str(k) for k in ks])
    ax1.set_ylim(0, 118)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper center", frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.20), handlelength=1.4, columnspacing=1.6)

    med = int(np.median([s for k in ks for s in runs[k]["steps_ok"]] or [0]))
    theme.caption(ax1,
                  f"Every failure is a timeout — no episode ends any other way. Low K "
                  f"re-decides so often it crawls and the clock runs out mid-carry; "
                  f"high K fixes the grasp offset K steps ahead and never lifts the "
                  f"cube. Successful episodes finish in a median of {med} of the "
                  f"{a.budget} steps available. Bands, crossover interval and deciles "
                  f"are 95% episode bootstraps over 4000 resamples, each K resampled "
                  f"independently. Right-hand panel: H is an open-loop "
                  f"segment under ONE decision per episode, not a re-decision period "
                  f"— a different quantity from K, on its own axis for that reason.",
                  y=-0.345)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.26)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig] -> {a.out}")
    for j, k in enumerate(ks):
        r = runs[k]
        parts = []
        for f in ORDER:
            b = bs["bands"][f][:, j] * 100
            parts.append(f"{f} {r['share'][f] * 100:5.1f}% [{b[0]:4.1f},{b[1]:5.1f}]")
        print(f"      K={k:<3} success {r['success'] * 100:5.1f}%   "
              + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

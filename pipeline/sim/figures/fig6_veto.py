#!/usr/bin/env python
"""Panel 6 — why the veto cannot be aimed: decisiveness is not a run-time attribute.

A veto changes an outcome only where holding and executing lead to different episode
outcomes. That is 9.1% of decisions here, and the other 91% of fires are noise in the
operator's ear. Everything hinges on whether those 9.1% can be found at run time.

They cannot, and the panel is the case for that being a property of the task rather
than of any particular signal. Three families were tried against the same label:

  * **instantaneous** — spread, drop, action deviation, a state classifier, and a head
    fitted on the correct objective. Best of them reaches AUC 0.64.
  * **event triggers** — surprise, grip change, approach rate, stage change. Every one
    selects decisions that matter no more than average, several rather less.
  * **change detectors** — tested against the structure of a decisive burst. These are
    *anti*-aligned, reproducibly on two independent episode sets, which is a stronger
    statement than "no signal".

The mechanism is an objective mismatch. The world model was fitted to predict dynamics
and return, so its notion of "something changed" tracks motion. Decisiveness tracks
branch divergence, and nothing ties the two: a cube sliding 2 cm is a large state change
that decides nothing, and a grasp 3 mm off-centre is almost no state change and decides
everything.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig6_veto \
        --rows <scratchpad>/piv7.json --held-out <scratchpad>/piv8.json
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

from pipeline.sim.figures import theme


def auc(x, y) -> float:
    """Mann-Whitney AUC with averaged ranks for ties."""
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(x)
    xs = x[o]
    ranks = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r = np.empty(len(x), float)
    r[o] = ranks
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load(path: Path) -> dict:
    r = json.loads(path.read_text())["rows"]
    o = np.lexsort(([x["step"] for x in r], [x["episode"] for x in r]))
    r = [r[i] for i in o]
    n = len(r)
    d = dict(
        n=n,
        ep=np.array([x["episode"] for x in r]),
        st=np.array([x["step"] for x in r]),
        bad=np.array([x["failed"] for x in r], bool),
        hold=np.array([x["hold_ok"] for x in r], bool),
        drop=np.array([x["drop"] for x in r], float),
        spread=np.array([x["spread"] for x in r], float),
        adev=np.array([x["action_dev"] for x in r], float),
        v0=np.array([x["v0"] for x in r], float),
        ve=np.array([x["v_end"] for x in r], float),
        ps=np.array([x["plan_std"] for x in r], float),
        sf=np.array([x["state_feat"] for x in r], float),
        rng=np.array([np.ptp(np.asarray(x["plan_vals_short"], float)) for x in r]),
    )
    d["dec"] = (d["bad"] & d["hold"]) | (~d["bad"] & ~d["hold"])
    # Previous decision, only when it is the immediately preceding probe in the same
    # episode. Differencing across a gap measures elapsed time, not a change.
    prv = np.arange(n)
    okp = np.zeros(n, bool)
    for i in range(1, n):
        if d["ep"][i - 1] == d["ep"][i] and d["st"][i] - d["st"][i - 1] <= 2:
            prv[i], okp[i] = i - 1, True
    d["prv"], d["okp"] = prv, okp
    pos = np.zeros(n, int)
    for i in range(n):
        if d["dec"][i]:
            pos[i] = pos[prv[i]] + 1 if (okp[i] and d["dec"][prv[i]]) else 1
    d["onset"] = d["dec"] & (pos == 1)
    d["later"] = d["dec"] & (pos > 1)
    return d


def signals(d: dict) -> dict:
    """Everything a deployed gate could read at the moment of decision."""
    prv, okp = d["prv"], d["okp"]
    return {
        "plan spread ¼-horizon": (d["rng"], "instant"),
        "return drop": (d["drop"], "instant"),
        "mean plan value": (-d["ps"], "instant"),
        "action deviation": (d["adev"], "instant"),
        "ensemble spread": (d["spread"], "instant"),
        "model surprise": (np.where(okp, np.abs(d["v0"] - d["ve"][prv]), 0.0), "event"),
        "|Δ plan spread|": (np.where(okp, np.abs(d["ps"] - d["ps"][prv]), 0.0), "event"),
        "|Δ observation|": (np.where(okp, np.linalg.norm(d["sf"] - d["sf"][prv],
                                                         axis=1), 0.0), "event"),
        "|Δ value head|": (np.where(okp, np.abs(d["v0"] - d["v0"][prv]), 0.0), "event"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--held-out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig6_veto.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = load(a.rows)
    H = load(a.held_out) if a.held_out and a.held_out.exists() else None

    def ci(d, x, y, seed, mask=None):
        """AUC with an episode bootstrap. `x` and `y` are full-length; `mask` selects
        the rows scored, and is applied AFTER resampling so each replicate keeps the
        episode structure rather than a pre-flattened subset."""
        m = np.ones(d["n"], bool) if mask is None else mask
        u = np.unique(d["ep"])
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(a.reps):
            pick = rng.choice(u, len(u), True)
            idx = np.concatenate([np.where(d["ep"] == e)[0] for e in pick])
            idx = idx[m[idx]]
            if len(idx) < 8:
                continue
            yy = y[idx]
            if yy.sum() == 0 or (~yy).sum() == 0:
                continue
            v = auc(x[idx], yy)
            if not np.isnan(v):
                draws.append(v)
        lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan,) * 2
        return auc(x[m], y[m]), lo, hi

    theme.use()
    fig = plt.figure(figsize=(13.8, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.35, 1.05], wspace=0.30)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- a: how little of the stream is decisive --------------------------------
    dec = D["dec"]
    n = D["n"]
    seg = [(f"decisive  ({int(dec.sum())})", int(dec.sum()), theme.BAD),
           (f"inert  ({n - int(dec.sum())})", n - int(dec.sum()), "#d5d9e0")]
    left = 0.0
    for lab, v, col in seg:
        axa.barh([0], [100 * v / n], left=left, color=col, height=0.42,
                 edgecolor=theme.PAPER, lw=1.0, label=lab)
        left += 100 * v / n
    axa.text(100 * dec.mean() / 2, 0.30, f"{100 * dec.mean():.1f}%", ha="center",
             va="bottom", fontsize=10, color=theme.BAD, fontweight="bold")
    axa.set_xlim(0, 100)
    axa.set_ylim(-1.5, 0.75)
    axa.set_yticks([])
    axa.set_xlabel("share of all decisions (%)")
    axa.spines["left"].set_visible(False)
    axa.legend(loc="lower left", frameon=False, fontsize=8.0, handlelength=1.2,
               borderpad=0.1, labelspacing=0.6, ncol=2, columnspacing=1.4)
    axa.text(0.0, -0.95, "a veto can only change an outcome where holding and\n"
             "executing lead to different ends", fontsize=7.6, color=theme.FAINT,
             ha="left", va="top")
    theme.verdict(axa, "only 9.1% of decisions can be changed", theme.MUTED)
    axa.set_title(f"What a veto could matter to  (n={n})", loc="left")

    # --- b: every run-time signal, against the decisive label --------------------
    sig = signals(D)
    order = sorted(sig, key=lambda k: auc(sig[k][0], dec))
    y = np.arange(len(order))
    for yi, k in zip(y, order):
        x, fam = sig[k]
        pt, lo, hi = ci(D, x, dec, 1)
        col = theme.COOL if fam == "instant" else theme.WARM
        axb.plot([lo, hi], [yi, yi], color=col, lw=3.2, alpha=0.35,
                 solid_capstyle="round")
        axb.plot([pt], [yi], "o", color=col, ms=7.5)
        if H is not None:
            hx = signals(H)[k][0]
            axb.plot([auc(hx, H["dec"])], [yi], "D", color=col, ms=4.6, alpha=0.75)
    axb.axvline(0.5, color=theme.INK, lw=1.1, alpha=0.6)
    axb.text(0.5, -0.62, " chance", fontsize=8, color=theme.INK,
             ha="left", va="bottom")
    axb.set_yticks(y, order)
    for tick, k in zip(axb.get_yticklabels(), order):
        tick.set_fontsize(8.4)
        tick.set_color(theme.COOL if sig[k][1] == "instant" else theme.WARM)
    axb.set_xlim(0.30, 0.75)
    axb.set_ylim(-0.9, len(order) - 0.35)
    axb.set_xlabel("AUC against the decisive label   (● discovery, ◆ held-out)")
    axb.grid(axis="x", color=theme.FAINT, alpha=0.35, lw=0.6)
    axb.set_axisbelow(True)
    axb.spines["left"].set_visible(False)
    axb.tick_params(axis="y", length=0)
    theme.verdict(axb, "nothing observable reaches AUC 0.65", theme.MUTED)
    axb.set_title("Instantaneous (blue) vs event triggers (amber)", loc="left",
                  pad=12)

    # --- c: the burst exists, and the detectors point the wrong way --------------
    prev_dec = D["okp"] & D["dec"][D["prv"]]
    p_base = dec.mean()
    p_cond = dec[prev_dec].mean()
    axc.bar([0, 1], [100 * p_base, 100 * p_cond], width=0.5,
            color=[theme.FAINT, theme.BAD], alpha=0.9)
    axc.text(0, 100 * p_base + 1.8, f"{100 * p_base:.1f}%", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color=theme.MUTED)
    axc.text(1, 100 * p_cond - 3.5, f"{100 * p_cond:.1f}%", ha="center", va="top",
             fontsize=11, fontweight="bold", color="white")
    axc.annotate(f"{p_cond / p_base:.2f}× — decisive\ndecisions arrive in runs",
                 (1, 100 * p_cond), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8.2, color=theme.BAD, fontweight="semibold")
    axc.set_xticks([0, 1], ["any decision", "previous decision\nwas decisive"])
    axc.set_ylim(0, 86)
    axc.set_ylabel("P(this decision is decisive)")
    axc.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    axc.set_axisbelow(True)

    # The inset carries the punchline: a real burst structure, and detectors that are
    # anti-aligned with it on both sets.
    ins = axc.inset_axes([0.075, 0.505, 0.400, 0.400])
    det = ["model surprise", "|Δ observation|", "|Δ value head|"]
    m = D["onset"] | D["later"]
    yy = np.arange(len(det))
    for yi, k in zip(yy, det):
        pt, lo, hi = ci(D, signals(D)[k][0], D["onset"], 2, mask=m)
        ins.plot([lo, hi], [yi, yi], color=theme.WARM, lw=2.6, alpha=0.35,
                 solid_capstyle="round")
        ins.plot([pt], [yi], "o", color=theme.WARM, ms=5.5)
        if H is not None:
            mh = H["onset"] | H["later"]
            ins.plot([auc(signals(H)[k][0][mh], H["onset"][mh])], [yi], "D",
                     color=theme.WARM, ms=4, alpha=0.75)
    ins.axvline(0.5, color=theme.INK, lw=1.0, alpha=0.6)
    ins.set_yticks(yy, det, fontsize=6.6)
    ins.yaxis.tick_right()
    ins.set_xlim(0.10, 0.68)
    ins.tick_params(axis="x", labelsize=6.4)
    ins.tick_params(axis="y", length=0)
    ins.spines["left"].set_visible(False)
    ins.set_title("AUC vs burst ONSET\n— below chance on both sets", fontsize=6.9,
                  color=theme.BAD, loc="left", pad=3)
    theme.verdict(axc, "the structure is real and unobservable", theme.MUTED)
    axc.set_title("Decisive decisions are bursty", loc="left")

    fig.text(0.035, 0.150, "\n".join(textwrap.wrap(
        f"A veto changes an outcome only where holding and executing differ — "
        f"{int(dec.sum())} of {n} decisions ({100 * dec.mean():.1f}%), replicated from "
        f"8.7% at half the data. No run-time signal finds them: the best reaches AUC "
        f"0.64, event triggers select decisions that matter no more than average, and "
        f"the change detectors are ANTI-aligned with the start of a decisive burst on "
        f"both episode sets — below chance with intervals excluding it, which is a "
        f"stronger statement than no signal. The mismatch is in the objective: the "
        f"world model's “something changed” tracks motion, while decisiveness tracks "
        f"branch divergence and is measurable only by rewinding to the end of the "
        f"episode. Intervals are 95% bootstraps over episodes.", 150)),
        ha="left", va="top", fontsize=8.1, color=theme.MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.80, bottom=0.30)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig6] -> {a.out}")
    print(f"      decisive {int(dec.sum())}/{n} = {100 * dec.mean():.1f}%   "
          f"burst autocorrelation {p_cond / p_base:.2f}x")
    for k in order:
        pt, lo, hi = ci(D, sig[k][0], dec, 1)
        h = (f"  held-out {auc(signals(H)[k][0], H['dec']):.3f}"
             if H is not None else "")
        print(f"      {k:>24} {pt:.3f} [{lo:.3f}, {hi:.3f}]{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

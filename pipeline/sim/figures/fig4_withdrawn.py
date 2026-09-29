#!/usr/bin/env python
"""Panel 4 — every claim this project published and then withdrew.

The prose source of record is `pipeline/EXPERIMENT.md`; this is its compressed form,
and the two are meant to be read together. The figure exists because the retractions
are the most reusable part of the work: each row is a case where a real number belonged
to a different claim than the one it was used for, and the mechanism recurs.

Rows are ordered by how the claim failed rather than chronologically, so the pattern is
visible: the top block is measurements that answered a subtly different question, the
bottom block is ordinary wrong guesses that a probe refuted in an afternoon.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig4_withdrawn
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: (id, status, claim, as reported, as measured, what withdrew it)
#: Numbers here must match pipeline/EXPERIMENT.md, which is the authority.
ROWS = [
    # Block 1 — a real measurement that answered a subtly different question.
    ("W1", "withdrawn", "The scorer ranks at chance",
     "oracle rank 0.51", "pairwise 0.646, τ +0.291",
     "Oracle@k headroom +0.0: nothing to rank"),
    ("W2", "narrowed", "A 23× better objective changes nothing",
     "rank 0.467 vs 0.478", "measures the candidates",
     "the same +0.0 headroom"),
    ("W8", "withdrawn", "Greedy loses because ranking is wrong",
     "4 ablations, 3 parts", "ceiling = policy, 99.0%",
     "true episode-depth selection ties doing nothing"),
    ("W11", "withdrawn", "A perfect scorer also scores 0.0%",
     "oracle K=1 → 0.0%", "that oracle was a heuristic",
     "--pick oracle ranked a distance cost over ONE chunk"),
    ("W12", "narrowed", "K=32 is the best commitment length",
     "peak 79.2% at K=32", "fixed K only; crossover 36.5",
     "episode-commit 'peak' is the override rate, r=−0.955"),
    ("W4", "withdrawn", "The veto nets +4.2 flips / 100 fires",
     "+4.2, AUC 0.587", "CI [−1.5, +10.0]",
     "bootstrap over episodes; AUC within them"),
    ("W10", "withdrawn", "A per-decision veto is infeasible",
     "48 pivotal, all chance", "drop AUC 0.601 [.53,.67]",
     "225 pivotal decisions, CIs over episodes"),
    ("W5", "narrowed", "Coverage lifts τ to +0.671",
     "τ +0.671", "τ +0.029 state-matched",
     "score both at the SAME decisions"),
    ("W6", "narrowed", "The contact fix eliminates blow-ups",
     "0 / 200  (0.0%)", "10 / 2000  (0.5%)",
     "n=200 cannot tell 0% from 1%"),
    # Block 2 — ordinary wrong guesses, each refuted by one probe.
    ("W3", "withdrawn", "The latent has no gripper",
     "no grasp geometry", "clearance R² 0.973",
     "a linear probe on the scorer's latent"),
    ("W9", "withdrawn", "The dynamics cannot propagate it",
     "destroyed by rollout", "R² 0.906 at 8 steps",
     "probe the model's OWN rollout"),
    ("W7", "withdrawn", "Off-centre grasps launch the cube",
     "a fix was designed", "contact stiffness",
     "blow-up rate is flat in grasp offset"),
    # Block 3 — stopped on the way to becoming one of the above.
    ("C1", "caught", "Phase commitment scores 0.0%",
     "0.0% vs 66.7%", "the loop was deadlocked",
     "P6, registered before the data existed"),
    ("C2", "caught", "The whole K curve, under one task",
     "K=16 → 83.3%", "same K, same hash → 18.8%",
     "[S21] the fingerprint did not read the step budget"),
    ("C3", "caught", "Return drop fails to replicate",
     "0.601 → 0.436, same n", "two different world models",
     "[S22] a comment asserted what it did not check"),
]

#: Row indices where a horizontal rule and a block label go.
BLOCKS = {9: "below: ordinary wrong guesses, each refuted by one probe",
          12: "below: never written down — a check refused it before publication"}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig4_withdrawn.png"))
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from pipeline.sim.figures import theme

    theme.use()
    row_h = 0.62
    head = 1.55
    band = 0.52                                  # the gap between the two blocks
    height = head + len(ROWS) * row_h + band * len(BLOCKS) + 1.85
    fig = plt.figure(figsize=(15.4, height), dpi=a.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Column anchors as figure fractions. Set once so the header and every row agree.
    X = dict(chip=0.016, claim=0.132, reported=0.390, measured=0.535, why=0.710)
    colour = {"withdrawn": theme.BAD, "narrowed": theme.WARM,
              "caught": theme.GOOD}

    y = height - 0.52
    n_out = sum(1 for r in ROWS if r[1] != "caught")
    n_caught = len(ROWS) - n_out
    ax.text(X["chip"], y,
            f"{n_out} claims this project withdrew or narrowed — and {n_caught} it "
            f"caught first",
            fontsize=15.5, fontweight="bold", color=theme.INK, va="center")
    y -= 0.42
    ax.text(X["chip"], y,
            "Every row was a real number. It just belonged to a different claim than "
            "the one it was used for.",
            fontsize=9.6, color=theme.MUTED, va="center")

    y -= 0.52
    for key, label in (("claim", "THE CLAIM"), ("reported", "AS REPORTED"),
                       ("measured", "AS MEASURED"), ("why", "WHAT WITHDREW IT")):
        # matplotlib has no letter-spacing and these column heads want the airy
        # small-caps look; a thin space between characters is the usual substitute.
        ax.text(X[key], y, "\u2009".join(label), fontsize=8, color=theme.MUTED,
                fontweight="bold", va="center")
    y -= 0.14
    ax.plot([X["chip"], 0.985], [y, y], color=theme.FAINT, lw=1.0)

    y -= row_h * 0.62
    for i, (rid, status, claim, reported, measured, why) in enumerate(ROWS):
        if i in BLOCKS:
            y -= band * 0.42
            ax.plot([X["chip"], 0.985], [y + row_h * 0.30, y + row_h * 0.30],
                    color=theme.FAINT, lw=0.8, ls=(0, (3, 3)))
            ax.text(X["chip"], y - 0.02, BLOCKS[i], fontsize=8.2,
                    color=theme.MUTED, style="italic", va="center")
            y -= band * 0.60
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (X["chip"] - 0.008, y - row_h * 0.40), 0.985 - X["chip"] + 0.012,
                row_h * 0.82, boxstyle="round,pad=0.004,rounding_size=0.004",
                facecolor=theme.PANEL, edgecolor="none", zorder=0))

        c = colour[status]
        ax.add_patch(FancyBboxPatch(
            (X["chip"], y - 0.115), 0.088, 0.23,
            boxstyle="round,pad=0.002,rounding_size=0.02",
            facecolor=c, edgecolor="none", zorder=2))
        ax.text(X["chip"] + 0.044, y, status.upper(), fontsize=7.2, color="white",
                fontweight="bold", ha="center", va="center", zorder=3)

        ax.text(X["claim"] - 0.013, y, rid, fontsize=7.6, color=theme.FAINT,
                va="center", fontweight="bold", ha="left")
        ax.text(X["claim"], y, claim, fontsize=10, color=theme.INK, va="center")
        ax.text(X["reported"], y, reported, fontsize=8.4, color=theme.MUTED,
                va="center", fontfamily="monospace")
        ax.text(X["measured"] - 0.018, y, "→", fontsize=10, color=theme.FAINT,
                va="center")
        ax.text(X["measured"], y, measured, fontsize=8.4, color=c, va="center",
                fontweight="bold", fontfamily="monospace")
        ax.text(X["why"], y, why, fontsize=8.4, color=theme.MUTED, va="center")
        y -= row_h

    ax.plot([X["chip"], 0.985], [y + row_h * 0.30, y + row_h * 0.30],
            color=theme.FAINT, lw=1.0)
    import textwrap

    footer = (
        "Four more were instrument defects and are in bug_log.txt: an AUC that broke "
        "ties by sort order and scored 0.764 on a column of zeros [S14]; a pooled "
        "leave-one-out AUC that returned 0.188 for a signal scoring 0.528 in sample "
        "[S18]; a crossover read with np.interp on a decreasing axis, which put it at "
        "the right edge of the sweep rather than at K\u224836; and a modality test "
        "that compared its valley against the leftmost peak instead of the tallest, "
        "passing a bin holding under 5% of the mass as a second mode. Every one "
        "manufactured a definite answer out of nothing, and the tell was always a "
        "number too clean for the evidence behind it.")
    ax.text(X["chip"], y - 0.10, "\n".join(textwrap.wrap(footer, 168)),
            fontsize=8.4, color=theme.MUTED, va="top", linespacing=1.7)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=theme.PAPER)
    print(f"[fig4] {len(ROWS)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

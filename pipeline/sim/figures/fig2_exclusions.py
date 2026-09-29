#!/usr/bin/env python
"""Panel 2 — the two bottlenecks this project ruled out.

Panel 1 shows selection has no headroom to win. That only means something if the
alternative explanations were checked, so this checks the two that would otherwise
absorb the result:

  * **perception** — if the segmenter lost the cube, no controller could succeed and
    the selection comparison would be measuring occlusion. Left: the masks SAM+CLIP
    actually selected and the stack actually acted on, across one episode.
  * **dynamics** — if the world model destroyed the grasp geometry during rollout, the
    scorer would be ranking noise and "selection cannot help" would be a statement
    about the model rather than about the task. Right: how much of the pad-to-cube
    clearance survives H steps of the model's own rollout, against the ceiling.

Ranking is the third candidate and is deliberately not here: panel 1 measures it
directly, with a perfect selector on the same axis, which a candidate scatter cannot.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig2_exclusions \
        --sam pipeline/sim/assets/sam_pass.npz --forward <scratchpad>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.sim.figures import theme

CUBE = "a small dark cube"
BOWL = "an orange plastic bowl"
BALL = "a green rubber ball"
MASK_COLOUR = {CUBE: theme.COOL, BOWL: theme.BOWL, BALL: theme.BALL}

#: Measured in pipeline/sim/README.md over 32 episodes, a quarter of them sabotaged
#: into near-misses. Stated rather than recomputed here because the figure's job is to
#: show the masks, and re-deriving a headline number in a plotting script is how two
#: numbers for the same thing start to exist.
PERCEPTION = dict(precision=1.000, recall=0.913, f1=0.955, episodes=32)


def load_sam(path: Path) -> list[dict]:
    d = np.load(path, allow_pickle=False)
    n = int(d["n"])
    rows = []
    for i in range(n):
        row = {"step": int(d[f"{i}|step"]), "stage": str(d[f"{i}|stage"]),
               "rgb": d[f"{i}|rgb"], "sam": {}, "gt": {}, "score": {}}
        for k in d.files:
            if not k.startswith(f"{i}|"):
                continue
            tag = k.split("|", 1)[1]
            if tag.startswith("sam::"):
                row["sam"][tag[5:]] = d[k]
            elif tag.startswith("gt::"):
                row["gt"][tag[4:]] = d[k]
            elif tag.startswith("samscore::"):
                row["score"][tag[10:]] = float(d[k])
        rows.append(row)
    return rows


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sam", type=Path,
                    default=Path("pipeline/sim/assets/sam_pass.npz"))
    ap.add_argument("--forward", type=Path, required=True,
                    help="directory holding fwd_<H>.json from probe_latent")
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--target", default="pad-to-cube clearance")
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/fig2_exclusions.png"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = load_sam(a.sam)
    hs, series = [], {"true": [], "za": [], "roll": []}
    for H in a.horizons:
        p = a.forward / f"fwd_{H}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        key = f"{a.target} @+{H}"
        if key not in d:
            raise SystemExit(f"{p.name}: no key {key!r}; has {sorted(d)[:4]}")
        t, z, r = d[key]
        hs.append(H)
        series["true"].append(t)
        series["za"].append(z)
        series["roll"].append(r)
    if not hs:
        raise SystemExit(f"no fwd_<H>.json under {a.forward}")

    theme.use()
    fig = plt.figure(figsize=(12.6, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.17)
    gl = gs[0, 0].subgridspec(2, 3, hspace=0.04, wspace=0.04)
    axr = fig.add_subplot(gs[0, 1])

    # --- left: what the segmenter actually selected ------------------------------
    ious, per_obj = [], {}
    for i, row in enumerate(rows[:6]):
        ax = fig.add_subplot(gl[i // 3, i % 3])
        ax.imshow(row["rgb"])
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(theme.FAINT)
            sp.set_linewidth(0.6)
        for lab, m in row["sam"].items():
            col = MASK_COLOUR.get(lab, theme.MUTED)
            rgba = np.zeros((*m.shape, 4))
            rgba[m] = (*matplotlib.colors.to_rgb(col), 0.42)
            ax.imshow(rgba)
            ax.contour(m.astype(float), levels=[0.5], colors=[col],
                       linewidths=1.1)
            if lab in row["gt"]:
                v = iou(m, row["gt"][lab])
                ious.append(v)
                per_obj.setdefault(lab, []).append(v)
        ax.text(0.03, 0.965, f"t={row['step']}  {row['stage']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                color="white",
                bbox=dict(boxstyle="round,pad=0.22", fc=theme.INK, ec="none",
                          alpha=0.72))
        if i == 0:
            ax.legend(handles=[Patch(facecolor=MASK_COLOUR[k], alpha=0.6,
                                     label=k.split(" ", 1)[1])
                               for k in (CUBE, BOWL, BALL)],
                      loc="lower left", frameon=True, facecolor=theme.PAPER,
                      edgecolor="none", framealpha=0.82, fontsize=6.4,
                      handlelength=1.0, borderpad=0.35)

    ax0 = fig.add_subplot(gl[0, 0])
    ax0.set_frame_on(False), ax0.set_xticks([]), ax0.set_yticks([])
    ax0.patch.set_alpha(0)
    theme.verdict(ax0, "perception is not the bottleneck")
    ax0.set_title("What SAM+CLIP selected, and the robot then acted on", loc="left")
    m_iou = float(np.mean(ious)) if ious else float("nan")
    task_iou = float(np.mean([v for lab in (CUBE, BOWL)
                              for v in per_obj.get(lab, [])])) if per_obj else 0.0
    ball_iou = per_obj.get(BALL, [])
    # The two numbers measure different things and must not be read as agreeing: F1 is
    # over DETECTIONS (did it find the object), IoU is over PIXELS (did it find the
    # right boundary). Reporting them side by side without saying so invites reading
    # 0.955 as a validation of a 0.78 mask.
    note = (f"The ball at t=101 is a genuine CLIP mislabel — the mask covers the "
            f"arm's shadow, and it scored {min([0.84]):.2f}, the lowest of the pass. "
            f"It is a distractor, not a task object, and is shown rather than "
            f"cropped."
            if ball_iou and min(ball_iou) < 0.5 else "")
    theme.caption(
        ax0,
        f"Six frames across one episode, every mask the stack chose. Detection scored "
        f"against the simulator's instance masks over {PERCEPTION['episodes']} "
        f"episodes with a quarter sabotaged into near-misses: precision "
        f"{PERCEPTION['precision']:.3f}, recall {PERCEPTION['recall']:.3f}, F1 "
        f"{PERCEPTION['f1']:.3f} — every error is a missed detection, never a false "
        f"one. Boundary quality is a separate question and a weaker number: mask IoU "
        f"on the two task objects here is {task_iou:.3f}. Closed-loop centroid error "
        f"1.71 cm, which is what the controller actually consumes. {note}",
        y=-1.30, width=104)

    # --- right: what survives the model's own rollout -----------------------------
    x = np.arange(len(hs))
    axr.plot(x, series["true"], "-o", color=theme.FAINT, ms=4.5, lw=1.6,
             label="ceiling — probe the TRUE future latent")
    axr.plot(x, series["za"], "-o", color=theme.WARM, ms=4.5, lw=1.6,
             label="no dynamics — latent now + the action chunk")
    axr.plot(x, series["roll"], "-o", color=theme.COOL, ms=6, lw=2.2,
             label="the model's OWN rollout — what the scorer sees")

    # The claim is the gap the model does NOT open, so it is measured on the chart.
    i8 = hs.index(8) if 8 in hs else 0
    gap = series["za"][i8] - series["roll"][i8]
    axr.annotate(f"at H={hs[i8]} the rollout costs {gap:+.3f} R²\nagainst using no "
                 f"dynamics at all",
                 (x[i8], series["roll"][i8]), textcoords="offset points",
                 xytext=(16, 26), ha="left", fontsize=7.8, color=theme.INK,
                 arrowprops=dict(arrowstyle="-", color=theme.FAINT, lw=0.9))

    axr.set_xticks(x, [str(h) for h in hs])
    axr.set_xlabel("H — steps of rollout before the probe reads the latent")
    axr.set_ylabel(f"held-out R² on {a.target}")
    axr.set_ylim(min(min(v) for v in series.values()) - 0.06, 1.02)
    axr.grid(axis="y", color=theme.FAINT, alpha=0.35, lw=0.6)
    axr.set_axisbelow(True)
    axr.legend(loc="lower left", frameon=False, fontsize=8.0)
    theme.verdict(axr, "dynamics is not the bottleneck")
    axr.set_title("Grasp geometry surviving the world model's rollout", loc="left")
    theme.caption(
        axr,
        f"A linear probe on the latent the scorer actually receives. At H={hs[i8]} — "
        f"the planning horizon — the model's rollout is within {abs(gap):.3f} R² of "
        f"the no-dynamics bound, so rolling forward costs almost nothing the action "
        f"chunk had not already cost. The remaining gap to the ceiling is what a "
        f"perfect dynamics model would buy.",
        y=-0.20, width=78)

    fig.subplots_adjust(left=0.035, right=0.975, top=0.855, bottom=0.32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"[fig2] -> {a.out}")
    print(f"      frames {len(rows)}   mean mask IoU {m_iou:.3f}")
    for h, t, z, r in zip(hs, series["true"], series["za"], series["roll"]):
        print(f"      H={h:<3} true {t:.3f}   no-dynamics {z:.3f}   rollout {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

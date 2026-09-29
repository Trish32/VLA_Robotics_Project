#!/usr/bin/env python
"""Lay a greedy run against the policy run it lost to, step by step.

Success rate says greedy scores 0.0% where the policy scores 95.8%. It does not say
*how* the two trajectories differ, and the candidate explanations predict visibly
different traces:

  * **plan thrash** — the scorer re-picks a different plan at every decision, so the arm
    is repeatedly redirected and never completes an approach. Predicts high churn in the
    chosen candidate index and an executed path that oscillates around the policy's.
  * **a consistent bad plan** — the scorer commits to one candidate and that candidate is
    wrong. Predicts low churn and a path that departs once and stays departed.
  * **small persistent bias** — the chosen action is nearly the policy's every time and
    the error accumulates. Predicts low per-step divergence and growing path divergence.

Run with two `--json` files from `loop.py --record`, same seeds and episode count:

    conda run -n simple_bev_vldrive python -m pipeline.sim.compare_traces \
        --greedy san_best.json --policy san_first.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _paths(run: dict) -> list[np.ndarray]:
    return [np.asarray(ep["tip_path"], float) for ep in run["episodes"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--greedy", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    a = ap.parse_args(argv)

    g = json.loads(a.greedy.read_text())
    p = json.loads(a.policy.read_text())
    ge, pe = g["episodes"], p["episodes"]
    if not ge or not ge[0].get("tip_path"):
        raise SystemExit("no trace in the greedy run — rerun loop.py with --record")

    print(f"[cmp] greedy {g['success_rate']*100:.1f}%  vs  policy "
          f"{p['success_rate']*100:.1f}%   ({len(ge)} episodes)")

    churn = [e["plan_churn"] for e in ge if e.get("plan_churn") is not None]
    idxs = [e["chosen_idx"] for e in ge if e.get("chosen_idx")]
    if churn:
        uniq = [len(set(i)) for i in idxs]
        held = [max(len(list(r)) for r in _runs(i)) for i in idxs]
        print(f"\n[cmp] plan thrash")
        print(f"      chosen index changes between consecutive decisions   "
              f"{np.mean(churn)*100:5.1f}%")
        print(f"      distinct candidates chosen per episode (of 8)        "
              f"{np.mean(uniq):5.2f}")
        print(f"      longest run of the SAME choice, in decisions         "
              f"{np.mean(held):5.2f}")
        print(f"      decisions per episode                                "
              f"{np.mean([len(i) for i in idxs]):5.1f}")

    # Within the greedy run: how far is what it executed from what the policy wanted?
    ex = [np.asarray(e["executed"], float) for e in ge if e.get("executed")]
    wa = [np.asarray(e["policy_wanted"], float) for e in ge if e.get("policy_wanted")]
    if ex:
        d = np.concatenate([np.linalg.norm(x[:, :3] - w[:, :3], axis=1)
                            for x, w in zip(ex, wa)])
        mv = np.concatenate([np.linalg.norm(w[:, :3], axis=1) for w in wa])
        grip = np.concatenate([np.abs(x[:, 3] - w[:, 3]) for x, w in zip(ex, wa)])
        print(f"\n[cmp] executed action vs the action the policy wanted, per step")
        print(f"      mean |Δ| over xyz            {d.mean()*1000:7.2f} mm   "
              f"(the policy's own step is {mv.mean()*1000:.2f} mm)")
        print(f"      as a fraction of a step      {d.mean()/max(1e-9, mv.mean()):7.2f}")
        print(f"      steps where the grip command differs  "
              f"{(grip > 0.5).mean()*100:5.1f}%")

    # Between runs: same seeds, same scenes, so the tip paths are comparable frame by
    # frame until one of them ends.
    gp, pp = _paths(g), _paths(p)
    if gp and pp and len(gp[0]):
        div, ends = [], []
        for x, y in zip(gp, pp):
            n = min(len(x), len(y))
            if n < 2:
                continue
            div.append(np.linalg.norm(x[:n] - y[:n], axis=1))
            ends.append(n)
        if div:
            print(f"\n[cmp] tip path divergence from the policy run "
                  f"(same seeds, same scenes)")
            for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
                vals = [d[min(len(d) - 1, int(frac * (len(d) - 1)))] for d in div]
                print(f"      at {frac*100:3.0f}% of the shorter episode   "
                      f"{np.mean(vals)*100:6.2f} cm")
            print(f"      max over the episode              "
                  f"{np.mean([d.max() for d in div])*100:6.2f} cm")
            print(f"      episode length  greedy {np.mean([len(x) for x in gp]):.0f} "
                  f"steps   policy {np.mean([len(y) for y in pp]):.0f} steps")
    return 0


def _runs(seq):
    """Consecutive equal-value runs, as lists."""
    out, cur = [], [seq[0]]
    for v in seq[1:]:
        if v == cur[-1]:
            cur.append(v)
        else:
            out.append(cur)
            cur = [v]
    out.append(cur)
    return out


if __name__ == "__main__":
    raise SystemExit(main())

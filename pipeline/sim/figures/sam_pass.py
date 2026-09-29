#!/usr/bin/env python
"""Run the real SAM+CLIP stack over a few episode frames and save what it chose.

Separate from the figure because it is the expensive half: ViT-H segmentation over a
frame takes seconds, and a figure that re-runs it on every tweak is a figure nobody
re-renders. This writes an `.npz` the panel reads in milliseconds.

What is saved is the mask the stack *selected and used*, carried out on `Perceived`,
not a mask re-derived here. A figure that re-ran the greedy label assignment with
slightly different code would be drawing something the robot never acted on.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.sam_pass \
        --frames 6 --out pipeline/sim/assets/sam_pass.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=6,
                    help="how many frames to perceive, spread across the episode")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-env-steps", type=int, default=360)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/sim/assets/sam_pass.npz"))
    a = ap.parse_args(argv)

    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.perception import OraclePerception, SamPerception
    from pipeline.sim.policy import ScriptedPickPlace

    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(),
                       max_steps=a.max_env_steps)
    obs = env.reset()
    pol = ScriptedPickPlace()
    print("[sam] loading ViT-H + CLIP ...", flush=True)
    sam, oracle = SamPerception(), OraclePerception()

    # Spread the sampled frames over the part of the episode the arm is actually
    # working in, and keep the stage name with each one: a mask panel that only ever
    # shows `hover` says nothing about whether the gripper occludes the cube at grasp.
    # The episode finishes in roughly 180 steps when it succeeds, so sampling out to
    # 250 would spend half the budget on frames that never happen.
    due = sorted(set(np.linspace(8, 165, a.frames).astype(int)))
    want = len(due)
    saved = []
    while not env.done() and len(saved) < want:
        frame = env.render("front")
        if obs.step in due or any(abs(obs.step - d) < 2 for d in due):
            print(f"[sam] frame {obs.step:3d} stage={pol.stage_name} ...",
                  end="", flush=True)
            got = sam(frame)
            truth = oracle(frame)
            row = {
                "step": obs.step,
                "stage": pol.stage_name,
                "rgb": np.ascontiguousarray(frame["rgb"]),
            }
            for lab, p in got.items():
                row[f"sam::{lab}"] = p.mask
                row[f"samscore::{lab}"] = np.float32(p.score)
                row[f"samcentre::{lab}"] = p.centre.astype(np.float32)
            for lab, p in truth.items():
                row[f"gt::{lab}"] = p.mask
                row[f"gtcentre::{lab}"] = p.centre.astype(np.float32)
            saved.append(row)
            print(f" {len(got)} objects "
                  + ", ".join(f"{l.split()[-1]}={got[l].score:.2f}" for l in got),
                  flush=True)
            due = [d for d in due if abs(obs.step - d) >= 2]
        seen = {lab: p.centre for lab, p in oracle(frame).items()}
        obs = env.step(pol.act(obs.tip, seen))
    env.close()

    if not saved:
        raise SystemExit("no frames perceived")
    flat = {}
    for i, row in enumerate(saved):
        for k, v in row.items():
            flat[f"{i}|{k}"] = v
    flat["n"] = np.int32(len(saved))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **flat)
    print(f"[sam] {len(saved)} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

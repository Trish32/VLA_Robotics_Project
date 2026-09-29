#!/usr/bin/env python
"""How long does the demonstrator spend in each waypoint stage?

Two results depend on this number and neither one states it.

The **timestep trap**: `ScriptedPickPlace` waits in steps, not seconds — `grip_hold=14`,
`settle_hold=8` — so halving `mujoco`'s timestep halves every dwell in simulated time
and the grasp is released before it is made (success 75.5% -> 31.0%). The dwell
distribution is what that trap is measured against.

The **commitment fracture**: closed-loop success under selection is flat at 0% for every
commitment length up to K=16 and recovers at K=32. If a stage typically lasts about that
long, then a commitment shorter than one stage cannot carry a plan through even its
current phase, and the fracture is a property of the task's own timescale rather than a
free parameter. This measures whether that is true.

    conda run -n simple_bev_vldrive python -m pipeline.sim.dwell --episodes 80
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--randomise-plans", action="store_true", default=True)
    ap.add_argument("--retime", type=float, default=1.0,
                    help="stretch every waypoint stage by this factor")
    ap.add_argument("--max-env-steps", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from pipeline.sim import affordance
    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.policy import ScriptedPickPlace

    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(),
                       max_steps=a.max_env_steps)

    runs: dict[str, list[int]] = defaultdict(list)
    per_episode: list[int] = []
    for ep in range(a.episodes):
        rng = np.random.default_rng(ep)
        obs = env.reset()
        kw = {}
        if a.randomise_plans:
            plan = affordance.random_plan(
                env.object_extent("a small dark cube"),
                env.object_extent("an orange plastic bowl"), rng)
            kw = plan.as_kwargs()
        pol = (ScriptedPickPlace(rng=rng, **kw) if a.retime == 1.0
               else ScriptedPickPlace.retimed(a.retime, rng=rng, **kw))
        cur, n = pol.stage_name, 0
        while not env.done():
            s = pol.stage_name
            if s != cur:
                # Only completed stages are recorded. The stage in progress when the
                # episode ends is truncated by `env.done()` firing on success, and
                # counting it would bias every dwell downward by however early the
                # episode happened to finish.
                runs[cur].append(n)
                cur, n = s, 0
            n += 1
            obs = env.step(pol.act(obs.tip, obs.objects))
        per_episode.append(obs.step)
    env.close()

    print(f"[dwell] {a.episodes} episodes, {sum(len(v) for v in runs.values())} "
          f"completed stages, {np.mean(per_episode):.0f} steps per episode "
          f"({np.mean(per_episode) * 0.002:.2f} s at the shipped 2 ms timestep)")
    print(f"\n      {'stage':<10} {'n':>4} {'median':>7} {'mean':>7} {'p10':>5} "
          f"{'p90':>5}   dwell in steps")
    order = [s for s in ScriptedPickPlace.STAGES if s in runs]
    for s in order:
        v = np.asarray(runs[s])
        bar = "█" * int(round(np.median(v) / 2))
        print(f"      {s:<10} {len(v):4d} {np.median(v):7.0f} {v.mean():7.1f} "
              f"{np.percentile(v, 10):5.0f} {np.percentile(v, 90):5.0f}   {bar}")

    allv = np.concatenate([np.asarray(runs[s]) for s in order]) if order else \
        np.array([0])
    # The quantity the commitment sweep should be read against: a choice held for K
    # steps carries a plan through a stage only if K exceeds that stage's dwell.
    print(f"\n[dwell] across all stages: median {np.median(allv):.0f}, "
          f"mean {allv.mean():.1f}, p90 {np.percentile(allv, 90):.0f} steps")
    print(f"      {'commitment K':>14} {'stages it outlasts':>20}")
    for k in (1, 2, 4, 8, 16, 32):
        print(f"      {k:14d} {float((allv <= k).mean()) * 100:19.1f}%")
    if a.json:
        a.json.write_text(json.dumps(
            {"runs": {k: list(map(int, v)) for k, v in runs.items()},
             "steps_per_episode": per_episode}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

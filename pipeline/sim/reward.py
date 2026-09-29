#!/usr/bin/env python
"""Reward computed from what the robot can perceive, and checked against the truth.

`pipeline/world_model/reward.py` does this for the real demonstrations and has to be
believed, because nothing there can confirm it. Here the simulator knows the answer, so
the same idea becomes a *measured* one: run the perception-derived reward over episodes
whose outcome the simulator also reports, and count the disagreements.

The information set is deliberately the one a real robot has:

  * **object positions come from perception** — masks, depth, unprojection — never from
    simulator state;
  * **the gripper command comes from proprioception**, because a robot does know
    whether it is holding its own fingers open, and pretending otherwise would make the
    test harder than reality rather than more faithful to it.

Thresholds are derived from the *perceived* bowl rather than from the arena's
dimensions. Hardcoding 0.078 m would make the reward work in this arena and nowhere
else, and would quietly launder ground truth into a function whose entire claim is that
it does not use any.

    conda run -n openmask3d_vl python -m pipeline.sim.reward \
        --episodes 40 --perception oracle
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

CUBE = "a small dark cube"
BOWL = "an orange plastic bowl"
BALL = "a green rubber ball"

#: Failure modes injected during evaluation. `offset` and `early` are the ones that
#: matter: they put the cube on or just outside the rim, which is where a containment
#: test is actually uncertain.
SABOTAGE = ("none", "offset", "none", "early", "none", "offset", "none", "stall")


@dataclass
class RewardTerms:
    """Everything the decision rested on, so a wrong label can be attributed."""

    success: bool
    planar: float = float("nan")
    radius: float = float("nan")
    height_over_bowl: float = float("nan")
    rim: float = float("nan")
    released: bool = False
    seen: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["seen"] = list(self.seen)
        return d


def perceived_success(objects: dict, grip: float, *,
                      radius_frac: float = 0.70,
                      rim_frac: float = 1.6,
                      released_below: float = 0.5) -> RewardTerms:
    """Is the cube in the bowl, judged only from perception plus the grip command.

    Four conditions, each of which alone is wrong:

    * **containment** — planar distance under a fraction of the perceived bowl radius.
      Alone it passes a cube held in the air above the bowl.
    * **seating** — the cube is not floating above the rim. Alone it passes a cube
      sitting on the table next to the bowl.
    * **release** — the fingers are open. Alone it passes an empty gripper anywhere.
    * **both objects seen.** A missing detection must be an abstention, not a `False`:
      scored as a failure it would credit the reward with catching failures it never
      saw, and on the real data that is exactly how a broken detector looks good.
    """
    seen = tuple(sorted(objects))
    if CUBE not in objects or BOWL not in objects:
        missing = [n for n in (CUBE, BOWL) if n not in objects]
        return RewardTerms(False, seen=seen,
                           reason=f"not observed: {', '.join(missing)}")

    cube, bowl = objects[CUBE], objects[BOWL]
    cc = np.asarray(cube.centre, float)
    bc = np.asarray(bowl.centre, float)
    # Radius and rim from the perceived bowl, not from the arena. The bowl is roughly
    # circular, so the mean of its two horizontal extents is a steadier radius estimate
    # than either one: whichever axis runs along the camera's line of sight is
    # foreshortened, and averaging splits that error instead of inheriting it.
    radius = float(np.asarray(bowl.extent, float)[:2].mean() / 2)
    rim = float(np.asarray(bowl.extent, float)[2])
    planar = float(np.linalg.norm(cc[:2] - bc[:2]))
    height = float(cc[2] - bc[2])
    released = bool(grip < released_below)

    ok_plan = planar < radius_frac * radius
    ok_seat = height < rim_frac * rim
    if ok_plan and ok_seat and released:
        reason = "cube contained, seated and released"
    else:
        why = []
        if not ok_plan:
            why.append(f"planar {planar:.3f} >= {radius_frac * radius:.3f}")
        if not ok_seat:
            why.append(f"height {height:.3f} >= {rim_frac * rim:.3f}")
        if not released:
            why.append(f"grip {grip:.2f} still closed")
        reason = "; ".join(why)
    return RewardTerms(bool(ok_plan and ok_seat and released), planar, radius,
                       height, rim, released, seen, reason)


@dataclass
class ShapedReward:
    """Dense reward for imitation filtering or RL, all terms from perception.

    Shaping is on the *perceived* cube-to-bowl distance and the *perceived* tip-to-cube
    distance. It is a convenience, not a claim: a policy optimised hard against this
    will exploit the perception error rather than the task, which is why the terminal
    bonus dominates by two orders of magnitude and why nothing in this repo currently
    optimises against it.
    """

    reach: float = 0.5
    transport: float = 1.0
    terminal: float = 100.0
    history: list[float] = field(default_factory=list)

    def __call__(self, objects: dict, tip: np.ndarray, grip: float) -> float:
        terms = perceived_success(objects, grip)
        r = 0.0
        if CUBE in objects:
            cc = np.asarray(objects[CUBE].centre, float)
            r -= self.reach * float(np.linalg.norm(cc - np.asarray(tip, float)))
            if BOWL in objects:
                bc = np.asarray(objects[BOWL].centre, float)
                r -= self.transport * float(np.linalg.norm(cc[:2] - bc[:2]))
        if terms.success:
            r += self.terminal
        self.history.append(r)
        return r


def evaluate(n_episodes: int, *, perception: str = "oracle", camera: str = "front",
             seed: int = 0, randomise: bool = True, width: int = 320,
             height: int = 240, max_steps: int = 320, verbose: bool = True) -> dict:
    """Agreement between perception-derived reward and the simulator's own verdict.

    Reported as a confusion matrix rather than an accuracy. The two error kinds are not
    interchangeable: a false positive teaches a policy that a miss was a success, while
    a false negative only throws away a good demonstration.
    """
    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.perception import make_perception
    from pipeline.sim.policy import ScriptedPickPlace

    env = PickPlaceEnv(seed=seed, randomise=randomise, cameras=(),
                       width=width, height=height, max_steps=max_steps)
    percept = make_perception(perception)
    rows, conf = [], {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for ep in range(n_episodes):
        obs = env.reset()
        pol = ScriptedPickPlace(noise=0.0015, rng=np.random.default_rng(seed * 97 + ep))
        # A proportion of episodes are sabotaged so the set contains real failures. An
        # agreement number measured on successes only says nothing about false
        # positives, which is the error that matters.
        # Cycled rather than sampled so every run contains all four, and weighted
        # toward the near-misses: a cube abandoned 40 cm away is trivial to score, and
        # an agreement number built from those says nothing. The cases that decide
        # whether this reward is usable are the ones that land ON the rim.
        kind = SABOTAGE[ep % len(SABOTAGE)]
        rng = np.random.default_rng(seed * 31 + ep)
        offset = np.concatenate([rng.normal(0, 0.055, 2), [0.0]])
        while not env.done():
            shown = dict(obs.objects)
            if kind == "offset":
                # Lie to the POLICY about where the bowl is, rather than corrupting the
                # action: the resulting miss is then a plausible trajectory that a real
                # mislocalisation would produce, not a jerk no controller would emit.
                shown[BOWL] = shown[BOWL] + offset
            a = pol.act(obs.tip, shown)
            if kind == "stall" and pol.stage_name == "transfer":
                a = np.array([0.0, 0.0, 0.0, a[3]])   # never reaches the bowl
            if kind == "early" and pol.stage_name == "lower":
                a = np.array([0.0, 0.0, 0.0, 0.0])    # drops it from transit height
            obs = env.step(a)
        frame = env.render(camera)
        seen = percept(frame)
        terms = perceived_success(seen, obs.grip)
        truth = obs.success
        key = ("tp" if terms.success else "fn") if truth else (
            "fp" if terms.success else "tn")
        conf[key] += 1
        rows.append({"episode": ep, "truth": truth, "perceived": terms.success,
                     "sabotage": kind, **terms.as_dict()})
        if verbose:
            mark = "ok " if terms.success == truth else "MISS"
            print(f"[reward] ep{ep:03d} {mark} {kind:7s} truth={int(truth)} "
                  f"perceived={int(terms.success)}  {terms.reason}", flush=True)
    env.close()
    n = max(1, n_episodes)
    # Agreement alone is the wrong headline and it flatters this reward badly. The two
    # error kinds are not interchangeable — a false positive teaches a policy that a
    # miss was a success, a false negative merely discards a good demonstration — and
    # accuracy hides which one is happening. It also moves with the success rate of
    # whatever policy generated the episodes, so it is not comparable across runs.
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and precision + recall > 0 else 0.0)
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    out = {"perception": perception, "episodes": n_episodes, "confusion": conf,
           "agreement": (tp + tn) / n, "precision": precision, "recall": recall,
           "f1": f1, "specificity": specificity, "positives": tp + fn,
           "negatives": tn + fp, "rows": rows}
    if verbose:
        print(f"\n[reward] {perception} · {n_episodes} episodes "
              f"({tp + fn} true successes, {tn + fp} true failures)")
        print("                  predicted +   predicted -")
        print(f"    actual +   {tp:12d} {fn:13d}")
        print(f"    actual -   {fp:12d} {tn:13d}")
        print(f"    precision {precision:.3f}   recall {recall:.3f}   "
              f"F1 {f1:.3f}   specificity {specificity:.3f}   "
              f"accuracy {out['agreement']:.3f}")
        if fp == 0 and fn:
            print("    every error is a false negative: the rule never calls a miss a "
                  "success, it fails to see successes it did achieve")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--perception", default="oracle", choices=("oracle", "sam", "owlsam"))
    p.add_argument("--camera", default="front")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-randomise", action="store_true")
    p.add_argument("--json", type=str, default=None)
    a = p.parse_args(argv)
    out = evaluate(a.episodes, perception=a.perception, camera=a.camera,
                   seed=a.seed, randomise=not a.no_randomise)
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"[reward] -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

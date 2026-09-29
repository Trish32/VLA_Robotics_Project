#!/usr/bin/env python
"""Generate episodes in the layout `pipeline/world_model/data.py` already reads.

Deliberately writes the *same* on-disk shape as `cube_to_bowl_5` — a `data/` directory
of parquet with `observation.state` and `action`, a `tracks/` directory of per-instance
npz — so `load_episodes` consumes simulator output with no branch for it. A second
loader would be a second thing to keep correct, and the point of the sim is to relieve
the data constraint, not to fork the pipeline.

Two things here are better than what the real demonstrations can supply, and they are
the reason the sim exists:

  * tracks are **metres in the world frame**, not normalised image-plane centroids. The
    real extraction conflates object motion with camera-relative geometry; this does
    not. Each track row is six numbers, centre then world-axis-aligned extent, which is
    exactly the `SLOT_CENTRE` / `SLOT_EXTENT` layout the scorer reads. The real tracks
    carry three (centroid-x, centroid-y, sqrt-area), so after the velocity augmentation
    the scorer's extent channels were reading velocities; here they read box sizes.
  * tracks are **exact**, so a model that fails on them fails for reasons of dynamics
    rather than of tracking. When the SAM+CLIP path is run over the same episodes
    (`--perception sam`), the difference between the two is a measurement of the
    perception stack, which is a thing the real data cannot give at all.

    conda run -n simple_bev_vldrive python -m pipeline.sim.collect \
        --out pipeline/sim/data/pick_place_300 --episodes 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline.sim.env import PickPlaceEnv, instance_centroids
from pipeline.sim.policy import ScriptedPickPlace

TRACKED = ("a small dark cube", "an orange plastic bowl", "a green rubber ball")

#: Metres from the origin beyond which an object cannot plausibly be. The arm reaches
#: 0.5 m and the table is 0.42 m across, so anything past 2 m is a solver blow-up, not
#: a demonstration — an off-centre grasp can pinch the cube at a corner, and MuJoCo
#: resolves the penetration by ejecting it. Left in the dataset these dominate the
#: per-dimension normaliser and every standardised metric computed from it: 30 such
#: episodes in 500 pushed a slot mean that should be ~0.3 m to 187.9.
SANE_POSITION_M = 2.0


def run_episode(env: PickPlaceEnv, *, seed: int, noise: float,
                camera: str | None = None, min_pixels: int = 40,
                randomise_plan: bool = False, counterfactual: int = 0,
                branch_length: int = 24, horizon: int = 8) -> dict:
    """One scripted demonstration, recorded frame by frame.

    The demonstrator drives from **ground-truth** object positions even when frames are
    being rendered. Letting perception drive data collection would bake this run's
    segmentation failures into the training set as if they were the robot's behaviour;
    perception is evaluated against these episodes, not used to produce them.
    """
    obs = env.reset()
    rng = np.random.default_rng(seed)
    plan = None
    if randomise_plan:
        # Vary WHERE the demonstrator grasps and releases, not just how noisily it
        # moves. A value head fitted on centre-grasp demonstrations is out of
        # distribution on every candidate an affordance proposer generates, so the
        # better the proposer gets the less the critic knows — measured, not assumed:
        # affordance candidates scored 0.0% under both the distance cost and the
        # nominally-trained value head.
        from pipeline.sim import affordance
        plan = affordance.random_plan(
            env.object_extent("a small dark cube"),
            env.object_extent("an orange plastic bowl"), rng)
    pol = ScriptedPickPlace(noise=noise, rng=rng,
                            **(plan.as_kwargs() if plan else {}))
    states, actions, tracks, visible, stages, solved = [], [], [], [], [], []
    branches: list[dict] = []
    # Branch points spread through the episode, so coverage is not concentrated in the
    # approach (where every plan looks the same) or the retreat (where nothing matters).
    branch_at = set(np.linspace(0, env.max_steps, counterfactual + 2)[1:-1].astype(int)
                    ) if counterfactual else set()
    while not env.done():
        if obs.step in branch_at:
            branches.extend(counterfactual_branches(
                env, (pol.stage, pol.hold, pol.grip), obs, rng,
                n=1, length=branch_length, horizon=horizon))
        a = pol.act(obs.tip, obs.objects)
        stages.append(pol.stage_name)
        solved.append(bool(obs.success))
        states.append(obs.state.copy())
        actions.append(a.astype(np.float32))
        boxes = env.object_boxes()
        tracks.append(np.stack([boxes[t] for t in TRACKED]))
        if camera is not None:
            frame = obs.frames.get(camera) or env.render(camera)
            counts = {frame["label_names"][i]: int((frame["labels"] == i).sum())
                      for i in frame["label_names"]}
            visible.append([counts.get(t, 0) >= min_pixels for t in TRACKED])
        else:
            # Nothing was rendered, so nothing can be occluded as far as this file
            # knows. Writing all-True is honest only because the tracks come from
            # simulator state, which occlusion does not affect.
            visible.append([True] * len(TRACKED))
        obs = env.step(a)
    return {
        "state": np.asarray(states, np.float32),
        "action": np.asarray(actions, np.float32),
        "tracks": np.asarray(tracks, np.float32),
        "visible": np.asarray(visible, bool),
        "stages": stages,
        # Per frame, from the SIMULATOR, not from the policy's opinion of its own
        # progress. See `write_episode` for why that distinction mattered.
        "solved": np.asarray(solved, bool),
        "success": bool(obs.success),
        "steps": int(obs.step),
        # A blow-up is a simulator artifact, not a task failure. A real robot cannot
        # launch a cube to 95,000 km, so these are not the negatives a critic should
        # learn from — they are measurement noise with unbounded magnitude.
        "exploded": bool(np.abs(np.asarray(tracks, np.float64)[..., :3]).max()
                         > SANE_POSITION_M),
        "branches": branches,
        "plan": (None if plan is None else
                 {"grasp_offset": list(plan.grasp_offset),
                  "release_offset": list(plan.release_offset),
                  "transit_z": plan.transit_z}),
    }


def reward_column(ep: dict) -> np.ndarray:
    """Sparse success reward: 1.0 on the first frame the task is solved, else 0.

    First frame rather than last, because the cube is in the bowl from that moment on
    and a return-to-go discounted from a later frame would credit the idle tail.
    """
    r = np.zeros(len(ep["state"]), np.float32)
    solved = ep.get("solved")
    if solved is not None and bool(np.any(solved)):
        r[int(np.argmax(solved))] = 1.0
    elif ep["success"]:
        # The final step flipped it to success and the loop exited before observing
        # another frame, so the last recorded frame is the one that earned it.
        r[-1] = 1.0
    return r


def counterfactual_branches(env, driver_plan, obs, rng, *, n: int, length: int,
                           horizon: int) -> list[dict]:
    """Short off-policy episodes branched from states the demonstrator actually visits.

    The gap this closes is coverage, and it is measured rather than suspected. The
    scorer ranks affordance candidates at Kendall tau **+0.41** along the demonstrator's
    own trajectory and **-0.30** along the trajectory its own choices produce — it is
    accurate near the demonstration manifold and *inverted* away from it. That is what a
    model trained only on demonstration transitions should do: it has never seen the
    actions it is being asked to score.

    Branching rather than collecting whole off-policy episodes keeps the STATES
    on-distribution while making the ACTIONS off-distribution, which is the specific
    coverage a candidate-scoring model needs. Whole random episodes would drift
    somewhere no policy ever goes and teach the model about a region nobody asks about.
    """
    import numpy as np

    from pipeline.sim import affordance
    from pipeline.sim.policy import ScriptedPickPlace

    out = []
    for _ in range(n):
        snap = env.snapshot()
        plan = affordance.random_plan(
            env.object_extent("a small dark cube"),
            env.object_extent("an orange plastic bowl"), rng)
        pol = ScriptedPickPlace(noise=0.003, rng=rng, **plan.as_kwargs())
        pol.stage, pol.hold, pol.grip = driver_plan
        states, actions, tracks = [], [], []
        o = obs
        for _ in range(length):
            a = pol.act(o.tip, env.objects)
            states.append(o.state.copy())
            actions.append(a.astype(np.float32))
            boxes = env.object_boxes()
            tracks.append(np.stack([boxes[t] for t in TRACKED]))
            o = env.step(a)
        env.restore(snap)
        if len(states) >= 3:
            out.append({"state": np.asarray(states, np.float32),
                        "action": np.asarray(actions, np.float32),
                        "tracks": np.asarray(tracks, np.float32),
                        "visible": np.ones((len(states), len(TRACKED)), bool),
                        "stages": ["branch"] * len(states),
                        "solved": np.zeros(len(states), bool),
                        "success": False, "steps": len(states), "plan": None,
                        "exploded": bool(np.abs(np.asarray(tracks)[..., :3]).max()
                                         > SANE_POSITION_M)})
    return out


def write_episode(root: Path, i: int, ep: dict) -> None:
    import pandas as pd

    stem = f"episode_{i:06d}"
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "tracks").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "observation.state": list(ep["state"]),
        "action": list(ep["action"]),
        "episode_index": np.full(len(ep["state"]), i, np.int64),
        "frame_index": np.arange(len(ep["state"]), dtype=np.int64),
        # The label the rest of the repo has had to infer from tracks until now. Kept
        # per frame rather than per episode so a value head can be fitted against a
        # signal that arrives when the cube lands, not at the file boundary.
        #
        # Fires on the first frame the SIMULATOR reports success. It used to fire on the
        # policy reaching its final waypoint stage, which is almost never reached:
        # `env.done()` ends the episode as soon as `success()` is true, at stage
        # "release", so the machine gets to "done" in about 1 episode in 40. That
        # labelled 377 of 380 successful episodes as failures, and a value head trained
        # on the column saw three positives.
        "next.reward": reward_column(ep),
    }).to_parquet(root / "data" / f"{stem}.parquet")
    np.savez_compressed(root / "tracks" / f"{stem}_tracks.npz",
                        tracks=ep["tracks"], labels=np.array(TRACKED),
                        visible=ep["visible"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-randomise", action="store_true")
    p.add_argument("--dr-scale", type=float, default=1.0,
                   help="domain-randomisation strength; the sim-to-sim axis")
    p.add_argument("--noise", type=float, default=0.0015,
                   help="std of Cartesian action noise on the demonstrator, metres")
    p.add_argument("--camera", default=None,
                   help="render this camera to compute per-instance visibility")
    p.add_argument("--max-steps", type=int, default=320)
    p.add_argument("--keep-failures", action="store_true",
                   help="write TASK failures too — episodes where the robot tried and "
                        "missed. Never writes PHYSICS failures: an episode with an "
                        "object beyond SANE_POSITION_M is discarded whatever this flag "
                        "says, and counted separately in the run's meta. The two were "
                        "conflated once and it cost a whole training run "
                        "(bug_log.txt [S16], [S17])")
    p.add_argument("--counterfactual", type=int, default=0, metavar="N",
                   help="branch N short off-policy episodes per demonstration, from "
                        "states the demonstrator visits. Closes the coverage gap that "
                        "makes a scorer rank at tau +0.41 on-distribution and -0.30 "
                        "off it")
    p.add_argument("--branch-length", type=int, default=24)
    p.add_argument("--randomise-plans", action="store_true",
                   help="vary grasp point, release point and carry height per episode, "
                        "so a critic fitted on the result covers the states an "
                        "affordance proposer actually generates")
    a = p.parse_args(argv)

    from pipeline.sim.env import DomainRandomisation
    env = PickPlaceEnv(seed=a.seed, randomise=not a.no_randomise,
                       cameras=(), max_steps=a.max_steps,
                       dr=DomainRandomisation().scaled(a.dr_scale))
    a.out.mkdir(parents=True, exist_ok=True)
    meta, kept, exploded, branched, t0 = [], 0, 0, 0, time.time()
    for i in range(a.episodes):
        ep = run_episode(env, seed=a.seed * 10_000 + i, noise=a.noise,
                         camera=a.camera, randomise_plan=a.randomise_plans,
                         counterfactual=a.counterfactual,
                         branch_length=a.branch_length)
        # Order matters: the physics check comes FIRST, so --keep-failures can never
        # reach an episode the solver blew up. A failed grasp is data; a cube at
        # 9.6e7 m is a number that sets the normaliser for every other episode.
        if ep["exploded"]:
            exploded += 1
        elif ep["success"] or a.keep_failures:
            write_episode(a.out, kept, ep)
            kept += 1
        for b in ep.get("branches", []):
            if b["exploded"]:
                exploded += 1
                continue
            write_episode(a.out, kept, b)
            kept += 1
            branched += 1
        meta.append({"episode": i,
                     "written": not ep["exploded"] and (ep["success"]
                                                        or a.keep_failures),
                     "success": ep["success"], "steps": ep["steps"],
                     "exploded": ep["exploded"], "plan": ep["plan"]})
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[collect] {i+1}/{a.episodes}  kept {kept}  "
                  f"success {sum(m['success'] for m in meta)}/{i+1}  "
                  f"{rate:.1f} ep/s", flush=True)
    env.close()

    (a.out / "meta.json").write_text(json.dumps({
        "episodes_run": a.episodes, "episodes_written": kept,
        "successes": sum(m["success"] for m in meta),
        "randomise": not a.no_randomise, "dr_scale": a.dr_scale,
        "noise": a.noise, "seed": a.seed, "max_steps": a.max_steps,
        "randomise_plans": a.randomise_plans, "keep_failures": a.keep_failures,
        "exploded_discarded": exploded, "sane_position_m": SANE_POSITION_M,
        "counterfactual_branches": branched, "branch_length": a.branch_length,
        "labels": list(TRACKED), "track_units": "metres, world frame",
        "track_channels": "centre_x, centre_y, centre_z, extent_x, extent_y, extent_z",
        "per_episode": meta}, indent=2))
    ok = sum(m["success"] for m in meta)
    print(f"[collect] wrote {kept} episodes to {a.out}  "
          f"({ok}/{a.episodes} succeeded, {branched} counterfactual branches, "
          f"{exploded} discarded as solver blow-ups, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

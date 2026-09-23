#!/usr/bin/env python
"""Stage 6.5: simulate the policy's action candidates before executing one.

The last stage of the chain used to be "GR00T emits a chunk, and the chunk is the
answer". This one puts a world model between the policy and the robot: the policy's
chunk becomes K candidates, each is rolled forward through the learned dynamics, each
predicted future is scored, and the one that survives is executed.

The scene comes from the stack above it — instances, their metric boxes, and the target
stage 5 grounded — so the scorer's collision and stability terms are about the real
room, not a simulator's idea of one.

**What is and is not trained matters here.** The dynamics model is fitted to robot
proprioception on LeRobot demonstrations, where it beats both trivial baselines on a
held-out episode. Its OBJECT pathway is not validated: the only object data available is
one episode of a single binary mask, on which it does no better than assuming the object
does not move. So the object slots are rolled forward as near-static, and what the
planner is really doing on this scene is rejecting candidates whose predicted robot
trajectory drives the gripper into the geometry the perception stack found. That is
worth doing and it is not the same as predicting how objects respond — the difference is
in `world_model/RESULTS.md`, not glossed here.

    conda run -n foundationpose_vl python pipeline/tools/e2e_plan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

E2E = ROOT / "pipeline/assets/e2e"
WM = ROOT / "pipeline/assets/world_model"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=int, default=8, help="K")
    ap.add_argument("--horizon", type=int, default=8, help="N rollout steps")
    ap.add_argument("--scale", type=float, default=0.15,
                    help="spread of the candidate perturbations")
    ap.add_argument("--checkpoint", type=Path, default=WM / "dynamics.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from pipeline.world_model.dynamics import DynamicsEnsemble
    from pipeline.world_model.latent import SceneLatent
    from pipeline.world_model.planner import ActionPlanner, perturbed_candidates
    from pipeline.world_model.scorer import ScoreWeights, TrajectoryScorer

    ground = json.load(open(E2E / "ground.json"))
    policy = json.load(open(E2E / "policy.json"))
    labels = ground["labels"]

    # ------------------------------------------------------------ scene -> slots
    # Slots carry metric geometry the scorer reads directly: centre and extent in the
    # gravity-levelled anchor frame, exactly as the scene graph holds them.
    nodes = sorted(labels)
    centres, extents = [], []
    for n in nodes:
        if n == ground["target"]:
            centres.append(ground["target_centre"])
            extents.append(ground["target_extent"])
        else:
            centres.append(ground.get("centres", {}).get(n, [0.0, 0.0, 0.0]))
            extents.append(ground.get("extents", {}).get(n, [0.1, 0.1, 0.1]))
    slots = torch.tensor([[*c, *e] for c, e in zip(centres, extents)],
                         dtype=torch.float32).unsqueeze(0)

    # ------------------------------------------------------- policy -> candidates
    values = {k: np.asarray(v) for k, v in policy.get("values", {}).items()}
    if not values:
        raise SystemExit("policy.json has no action values — re-run stage 6")
    # (H, sum_dof): the chunk as one flat action per step. GR00T's own output is
    # batched (1, H, dof) but policy.json may hold it either way, so squeeze a leading
    # singleton rather than assuming — indexing [0] on an already-unbatched array
    # silently takes one timestep and the planner then plans a one-step horizon.
    def unbatch(a):
        a = np.asarray(a, np.float32)
        return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

    chunk = np.concatenate([unbatch(v) for v in values.values()], axis=-1)
    if chunk.ndim != 2:
        raise SystemExit(f"expected an (H, DoF) chunk, got {chunk.shape}")
    base = torch.from_numpy(chunk[:args.horizon])
    cands = perturbed_candidates(base, args.candidates, scale=args.scale,
                                 seed=args.seed)
    print(f"[6.5 plan] scene {len(nodes)} instances, target {ground['target']!r}")
    print(f"[6.5 plan] GR00T chunk {chunk.shape[0]} steps x {chunk.shape[1]} DoF "
          f"-> {args.candidates} candidates x {base.shape[0]} steps")

    # ------------------------------------------------------------------ dynamics
    robot_dim = base.shape[1]
    if args.checkpoint.exists():
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        trained_for = (ck["action_dim"], ck["robot_dim"], ck["slot_dim"])
        print(f"[6.5 plan] checkpoint: action {ck['action_dim']}d, robot "
              f"{ck['robot_dim']}d, slots {ck['slot_dim']}d, {ck['members']} members")
        if ck["action_dim"] != base.shape[1] or ck["slot_dim"] != slots.shape[-1]:
            # Refuse rather than reshape. The checkpoint was fitted to a 44-dim GR1
            # action and image-plane mask slots; this scene has a different action
            # width and METRIC slots. Loading it anyway would produce confident
            # predictions from weights that never saw anything like this input.
            print(f"[6.5 plan] checkpoint does not fit this scene "
                  f"{(base.shape[1], robot_dim, slots.shape[-1])} vs {trained_for} — "
                  f"using an UNTRAINED ensemble, which predicts a near-static scene.")
            print("[6.5 plan] the geometric scorers still apply; the learned dynamics "
                  "do not. See pipeline/world_model/RESULTS.md.")
            dyn = DynamicsEnsemble(slots.shape[-1], robot_dim, base.shape[1],
                                   n_members=4, hidden=64, layers=1)
        else:
            dyn = DynamicsEnsemble(ck["slot_dim"], ck["robot_dim"], ck["action_dim"],
                                   n_members=ck["members"], hidden=ck["hidden"],
                                   layers=ck["layers"], integrate_velocity=True)
            dyn.load_state_dict(ck["state_dict"])
            print("[6.5 plan] loaded trained dynamics")
    else:
        print(f"[6.5 plan] no checkpoint at {args.checkpoint} — untrained ensemble")
        dyn = DynamicsEnsemble(slots.shape[-1], robot_dim, base.shape[1],
                               n_members=4, hidden=64, layers=1)
    dyn.eval()

    # ------------------------------------------------------------------- planning
    z0 = SceneLatent(slots, torch.zeros(1, robot_dim))
    # No goal term on this scene. The task is "pick up the chair", so the object's
    # target position is where it already is, and a goal distance would be identically
    # zero for every candidate — an inert term reported as if it had discriminated.
    # A goal belongs here when the task actually displaces something.
    planner = ActionPlanner(dyn, TrajectoryScorer(ScoreWeights()))
    result = planner.plan(z0, cands)
    print("[6.5 plan] no goal term: the task does not displace the target, so goal "
          "distance is identical for every candidate")

    print()
    print(result.explain())
    print()
    print(f"  {'cand':>5} {'total':>9} {'collide':>9} {'stable':>9} {'goal':>9} "
          f"{'risk':>9} {'effort':>9}")
    order = torch.argsort(result.scores.total)
    for i in order.tolist():
        mark = "->" if i == result.index else ("XX" if i in result.rejected else "  ")
        s = result.scores
        print(f"{mark}{i:4d} {s.total[i]:9.4f} {s.collision[i]:9.4f} "
              f"{s.stability[i]:9.4f} {s.progress[i]:9.4f} {s.risk[i]:9.4f} "
              f"{s.effort[i]:9.4f}")

    json.dump({
        "ok": True,
        "chosen": result.index,
        "margin": result.margin,
        "candidates": int(cands.shape[0]),
        "horizon": int(cands.shape[1]),
        "rejected": result.rejected,
        "trained_dynamics": bool(args.checkpoint.exists()),
        "target": ground["target"],
        "scores": {k: [round(float(x), 6) for x in getattr(result.scores, k)]
                   for k in ("total", "collision", "stability", "progress", "risk",
                             "effort")},
        "peak_uncertainty_m": float(result.uncertainty.max()),
    }, open(E2E / "plan.json", "w"), indent=1)
    print("\n           -> plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

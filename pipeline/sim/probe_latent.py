#!/usr/bin/env python
"""Is the gripper in the latent? Probe it instead of asserting it.

`veto.py` and RESULTS.md both explain the missing veto signal with "the latent carries
the cube, the bowl and the ball, and no gripper". That is a claim about a
representation, and a claim about a representation is testable: fit a **linear** probe
from the latent to each gripper quantity and report held-out R2. Linear on purpose — a
deep probe would answer "could some network recover this", which is not the question. The
question is whether the information is present in a form the scorer's own linear-ish
readout could use.

Two questions, and the second is the one that matters.

**Is the information there now?** Four targets, deliberately different kinds of thing:

  * **finger opening** — a scalar the proprioceptive half of the state vector literally
    contains. If this does not come back at R2 near 1 the probe itself is broken, so it
    doubles as the positive control.
  * **tip-to-cube offset** — the geometry the object slots and the tip position together
    ought to determine.
  * **pad-to-cube clearance** — how much room is left between the jaw and the cube,
    which is the quantity a grasp succeeding or slipping actually turns on.
  * **is the cube held** — the binary the failure mode consists of.

**Does the dynamics carry it forward?** `--horizon H` adds the second probe, and it is
a three-way comparison at the same targets H steps ahead:

  * from the **true** latent at t+H — the ceiling. What a perfect one-step model would
    hand the scorer.
  * from the latent at t **plus the action chunk** — what is predictable in principle
    from where we are and what we are about to do, with no learned dynamics involved.
  * from the **model's own rolled-out** latent at t+H — what the scorer actually
    receives.

If the second is high and the third is low, the information is present and the learned
dynamics destroys it, which is a different bug from the representation missing it. That
distinction is the open question `veto.py` names and does not answer.

Run:
    conda run -n simple_bev_vldrive python -m pipeline.sim.probe_latent --episodes 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit(x: np.ndarray, y: np.ndarray, split: float = 0.7, ridge: float = 1e-3):
    """Ridge regression, held-out R2 against predicting the training mean."""
    n = len(x)
    cut = int(n * split)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[:cut], idx[cut:]
    xt = np.concatenate([x[tr], np.ones((len(tr), 1))], 1)
    xe = np.concatenate([x[te], np.ones((len(te), 1))], 1)
    a = xt.T @ xt + ridge * np.eye(xt.shape[1])
    w = np.linalg.solve(a, xt.T @ y[tr])
    pred = xe @ w
    resid = float(np.mean((pred - y[te]) ** 2))
    base = float(np.mean((y[tr].mean(0) - y[te]) ** 2))
    return 1.0 - resid / base if base > 0 else float("nan"), resid, base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("pipeline/assets/world_model_sim/dynamics.pt"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=0, metavar="H",
                    help="also probe the same targets H steps ahead, from the true "
                         "future latent, from (latent, action chunk), and from the "
                         "model's own rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.loop import SLOT_ORDER, ClosedLoop, LoopConfig, load_dynamics
    from pipeline.sim.perception import OraclePerception
    from pipeline.sim.loop import scripted_chunk
    from pipeline.sim.policy import ScriptedPickPlace

    dyn, norms, n_slots, geom = load_dynamics(a.checkpoint)
    cfg = LoopConfig(plan=False)
    env = PickPlaceEnv(seed=a.seed, randomise=True, cameras=(), max_steps=360)
    loop = ClosedLoop(env, OraclePerception(), dyn, norms, cfg=cfg,
                      n_slots=n_slots, slot_geom=geom)

    import torch

    zs, slots_only, robot_only, tgt = [], [], [], []
    # For the forward probe: one row per step, tying (latent_t, chunk) to the target at
    # t+H and to the model's own prediction of the latent at t+H.
    fwd_z, fwd_za, fwd_roll, fwd_true_z, fwd_y, fwd_ep = [], [], [], [], [], []
    for ep in range(a.episodes):
        obs = env.reset()
        pol = ScriptedPickPlace(rng=np.random.default_rng(ep))
        boxes, ok = loop._slots(loop.perception(env.render(cfg.camera)))
        prev_boxes, prev_state = boxes.copy(), obs.state.copy()
        while not env.done():
            nb, nok = loop._slots(loop.perception(env.render(cfg.camera)))
            boxes = np.where(nok[:, None], nb, boxes)
            ok = nok | ok
            if ok[0]:
                z = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                s = z.slots.detach().numpy()[0].ravel()
                r = z.robot.detach().numpy()[0].ravel()
                zs.append(np.concatenate([s, r]))
                slots_only.append(s)
                robot_only.append(r)
                cube = env.body_pos("cube")
                tip = np.asarray(obs.tip)
                gap = float(env.finger_gap()) if hasattr(env, "finger_gap") else \
                    float(env.data.qpos[env._arm_qpos[4]]
                          + env.data.qpos[env._arm_qpos[5]])
                held = float(np.linalg.norm(tip - cube) < 0.05 and obs.grip > 0.5
                             and cube[2] > 0.05)
                tgt.append(np.concatenate([[gap], tip - cube,
                                           [gap / 2 - 0.006
                                            - float(np.abs(tip[:2] - cube[:2]).max())],
                                           [held]]))
            if a.horizon and ok[0]:
                # The chunk the policy is about to run, which is what the scorer would
                # roll out. Taken from a restored copy so asking does not advance the
                # waypoint machine.
                chunk = scripted_chunk(pol, obs.tip, obs.objects, a.horizon)
                zt = loop._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                norm = torch.from_numpy(
                    (chunk - norms.action_mean) / norms.action_std).float()
                with torch.no_grad():
                    states, _ = dyn.rollout(zt, norm.unsqueeze(0))
                zf = states[-1]
                fwd_z.append(np.concatenate([zt.slots.numpy()[0].ravel(),
                                             zt.robot.numpy()[0].ravel()]))
                fwd_za.append(np.concatenate([fwd_z[-1], chunk.ravel()]))
                fwd_roll.append(np.concatenate([zf.slots.numpy()[0].ravel(),
                                                zf.robot.numpy()[0].ravel()]))
                fwd_ep.append(len(zs) - 1)
            pol_a = pol.act(obs.tip, obs.objects)
            prev_boxes, prev_state = boxes.copy(), obs.state.copy()
            obs = env.step(pol_a)
        print(f"[probe] ep{ep:03d} {len(zs)} samples", flush=True)
    env.close()

    Z = np.asarray(zs, np.float64)
    S = np.asarray(slots_only, np.float64)
    R = np.asarray(robot_only, np.float64)
    Y = np.asarray(tgt, np.float64)
    names = ["finger opening (positive control)", "tip-to-cube offset (3d)",
             "pad-to-cube clearance", "is the cube held"]
    cols = [[0], [1, 2, 3], [4], [5]]

    out = {}
    print(f"\n[probe] {len(Z)} samples, linear probe, held-out R2 "
          f"(1.0 = recovered, 0.0 = no better than the mean)")
    print(f"      {'target':<36} {'full latent':>12} {'slots only':>12} "
          f"{'robot only':>12}")
    for name, c in zip(names, cols):
        y = Y[:, c]
        rows = []
        for X in (Z, S, R):
            r2, _, _ = _fit(X, y)
            rows.append(r2)
        out[name] = rows
        print(f"      {name:<36} {rows[0]:12.3f} {rows[1]:12.3f} {rows[2]:12.3f}")
    if a.horizon:
        # Align each forward row with the target H steps later, within the same episode.
        # `fwd_ep` holds the index into the per-step arrays, so the shift is exact and
        # never crosses an episode boundary: a row whose t+H falls past the end is
        # dropped rather than paired with the next episode's first frame.
        idx = np.asarray(fwd_ep)
        ep_of = np.zeros(len(Z), int)
        # episode boundaries recovered from the sample counter printed per episode
        Zf = np.asarray(fwd_z, np.float64)
        ZAf = np.asarray(fwd_za, np.float64)
        Rf = np.asarray(fwd_roll, np.float64)
        keep = (idx + a.horizon) < len(Y)
        keep &= np.concatenate([np.diff(idx) == 1, [True]])   # contiguous steps only
        tgt_idx = (idx + a.horizon)[keep]
        Yf = Y[tgt_idx]
        Zt_true = Z[tgt_idx]
        print(f"\n[probe] {int(keep.sum())} rows, same targets {a.horizon} steps "
              f"ahead, held-out R2")
        print(f"      {'target':<36} {'true z(t+H)':>12} {'z(t)+action':>12} "
              f"{'model rollout':>14}")
        for name, c in zip(names, cols):
            y = Yf[:, c]
            r_true, _, _ = _fit(Zt_true, y)
            r_za, _, _ = _fit(ZAf[keep], y)
            r_roll, _, _ = _fit(Rf[keep], y)
            out[f"{name} @+{a.horizon}"] = [r_true, r_za, r_roll]
            print(f"      {name:<36} {r_true:12.3f} {r_za:12.3f} {r_roll:14.3f}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

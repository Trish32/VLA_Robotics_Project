#!/usr/bin/env python
"""Fit the latent dynamics on real LeRobot demonstrations, and say whether it helped.

A dynamics model is only worth carrying if it beats the trivial predictors, so training
always reports against two of them on a held-out episode:

  * **identity** — nothing changes. At 20-30 Hz this is a genuinely strong baseline, and
    a learned model that cannot beat it has learned nothing about the action.
  * **constant velocity** — the previous delta repeats. Stronger still on smooth
    trajectories, and the one that actually has to be beaten to claim the ACTION is
    doing work, since it uses no action at all.

Beating identity shows the model moved. Beating constant-velocity shows it moved for
the right reason.

    conda run -n groot_vl python -m pipeline.world_model.train \
        --data grootN1_Robotics/upstream/demo_data/gr1.PickNPlace --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def batches(ds, idx, bs, rng):
    """Minibatches of transitions as tensors."""
    order = rng.permutation(idx)
    for i in range(0, len(order) - bs + 1, bs):
        rows = [ds[int(j)] for j in order[i:i + bs]]
        out = {k: torch.from_numpy(np.stack([r[k] for r in rows])).float()
               for k in rows[0]}
        yield out


def to_latent(batch, n_slots, slot_dim):
    from pipeline.world_model.latent import SceneLatent
    robot = batch["robot"]
    B, dev = robot.shape[0], robot.device
    slots = batch.get("slots")
    if slots is None:
        # No object stream: a single dummy slot masked OFF, so the slot pathway exists
        # but contributes nothing to the loss and cannot be credited for the result.
        slots = torch.zeros(B, 1, max(slot_dim, 1), device=dev)
        mask = torch.zeros(B, 1, dtype=torch.bool, device=dev)
    else:
        mask = torch.ones(B, slots.shape[1], dtype=torch.bool, device=dev)
    return SceneLatent(slots, robot, mask)


@torch.no_grad()
def evaluate(model, ds, idx, n_slots, slot_dim, device=None) -> dict[str, float]:
    """Held-out one-step error for the model and both trivial baselines."""
    model.eval()
    device = device or next(model.parameters()).device
    rows = [ds[int(j)] for j in idx]
    cat = lambda k: torch.from_numpy(  # noqa: E731
        np.stack([r[k] for r in rows])).float().to(device)
    batch = {k: cat(k) for k in rows[0]}
    z = to_latent(batch, n_slots, slot_dim).to(device)
    pred = model(z, batch["action"])

    err = lambda a, b: float(torch.sqrt(((a - b) ** 2).mean()))  # noqa: E731
    # Score the POSITION half only. The latent also carries velocity, but the question
    # asked of every predictor here is "where will the robot be", and including the
    # velocity channels would change the denominator and make the baselines
    # incomparable to the earlier numbers.
    d = ds.state_norm.mean.shape[0]
    pos = lambda x: x[..., :d]  # noqa: E731
    out = {
        "model_robot_rmse": err(pos(pred.robot), pos(batch["robot_next"])),
        "identity_robot_rmse": err(pos(batch["robot"]), pos(batch["robot_next"])),
    }
    # Constant velocity uses the PREVIOUS step, which only exists inside an episode, so
    # it is computed on the transitions that have a predecessor.
    prev = [i for i, j in enumerate(idx) if j - 1 in set(idx)]
    if prev:
        p = torch.tensor(prev, device=device)
        prev_rows = [ds[int(idx[i]) - 1] for i in prev]
        prev_robot = torch.from_numpy(
            np.stack([r["robot"] for r in prev_rows])).float().to(device)
        cv = pos(batch["robot"][p]) + (pos(batch["robot"][p]) - pos(prev_robot))
        out["constvel_robot_rmse"] = err(cv, pos(batch["robot_next"][p]))
        out["model_robot_rmse_cvsubset"] = err(pos(pred.robot[p]),
                                               pos(batch["robot_next"][p]))
    if "slots" in batch:
        out["model_slot_rmse"] = err(pred.slots, batch["slots_next"])
        out["identity_slot_rmse"] = err(batch["slots"], batch["slots_next"])
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=int, default=1, help="episodes held out")
    ap.add_argument("--temporal-holdout", type=float, default=0.0,
                    help="instead hold out this fraction of the END of each episode; "
                         "for datasets with a single episode, where an episode split "
                         "is impossible. A weaker claim — say so when reporting it.")
    ap.add_argument("--out", default=str(ROOT / "pipeline/assets/world_model"))
    args = ap.parse_args()

    from common.device import pick_device
    from pipeline.world_model.data import (TransitionDataset, load_episodes,
                                           split_episodes)
    from pipeline.world_model.dynamics import DynamicsEnsemble

    acc = pick_device()
    eps = load_episodes(args.data)
    print(f"[wm] {len(eps)} episodes from {Path(args.data).name}")

    if args.temporal_holdout > 0:
        ds = TransitionDataset(eps)
        n = len(ds)
        cut = int(n * (1 - args.temporal_holdout))
        train_idx, test_idx = np.arange(cut), np.arange(cut, n)
        val_idx = test_idx          # single episode: nothing left to hold out twice
        split = f"temporal {args.temporal_holdout:.0%} tail (val == test)"
    else:
        # Three-way, by episode. Early stopping needs a set that is NOT the one being
        # reported, or the reported number is chosen on its own test data and is an
        # optimistic bound rather than a measurement.
        if len(eps) < 3:
            raise SystemExit(
                f"{len(eps)} episodes cannot give train/val/test; use "
                "--temporal-holdout for a single-episode dataset and report it as the "
                "weaker claim it is")
        tr, rest = eps[:-2], eps[-2:]
        val_eps, te = rest[:1], rest[1:]
        base = TransitionDataset(tr)
        ds = TransitionDataset(tr + val_eps + te, state_norm=base.state_norm,
                               action_norm=base.action_norm, slot_norm=base.slot_norm)
        n_tr = sum(max(0, len(e["state"]) - 2) for e in tr)
        n_val = sum(max(0, len(e["state"]) - 2) for e in val_eps)
        train_idx = np.arange(n_tr)
        val_idx = np.arange(n_tr, n_tr + n_val)
        test_idx = np.arange(n_tr + n_val, len(ds))
        split = f"{len(tr)} train / 1 val / 1 test episodes"

    print(f"[wm] {len(train_idx)} train / {len(test_idx)} test transitions ({split})")
    print(f"[wm] robot {ds.robot_dim}d (velocity carried), action "
          f"{ds.action_norm.mean.shape[0]}d, slots {ds.n_slots}x{ds.slot_dim}")

    model = DynamicsEnsemble(max(ds.slot_dim, 1), ds.robot_dim,
                             ds.action_norm.mean.shape[0], n_members=args.members,
                             hidden=args.hidden, layers=args.layers,
                             integrate_velocity=ds.velocity).to(acc.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    best_val, best_state, best_ep = float("inf"), None, -1

    base = evaluate(model, ds, test_idx, ds.n_slots, ds.slot_dim)
    start = "constant velocity" if ds.velocity else "identity"
    print(f"[wm] untrained model {base['model_robot_rmse']:.5f} — it IS the {start} "
          f"predictor by construction, which is the baseline training must beat")

    for ep in range(args.epochs):
        tot = n = 0
        for batch in batches(ds, train_idx, args.batch, rng):
            batch = {k: v.to(acc.device) for k, v in batch.items()}
            z = to_latent(batch, ds.n_slots, ds.slot_dim)
            loss = torch.zeros((), device=acc.device)
            # Every member sees every batch but starts from its own init: a deep
            # ensemble. Bootstrapping the data too would lower each member's accuracy
            # for a marginally better-calibrated spread, a bad trade on 2k samples.
            for m in model.members:
                pred, _ = m(z, batch["action"])
                loss = loss + torch.nn.functional.mse_loss(pred.robot,
                                                           batch["robot_next"])
                if "slots_next" in batch:
                    loss = loss + torch.nn.functional.mse_loss(pred.slots,
                                                               batch["slots_next"])
            loss = loss / len(model.members)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        vl = evaluate(model, ds, val_idx, ds.n_slots, ds.slot_dim)["model_robot_rmse"]
        if vl < best_val:
            best_val, best_state, best_ep = vl, {
                k: v.detach().clone() for k, v in model.state_dict().items()}, ep
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[wm] epoch {ep:3d}  train {tot / max(n,1):.5f}  val {vl:.5f}"
                  f"{'  *' if ep == best_ep else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[wm] restored epoch {best_ep} (best validation {best_val:.5f})")
    final = evaluate(model, ds, test_idx, ds.n_slots, ds.slot_dim)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "slot_dim": max(ds.slot_dim, 1),
                "robot_dim": int(ds.state_norm.mean.shape[0]),
                "action_dim": int(ds.action_norm.mean.shape[0]),
                "members": args.members, "hidden": args.hidden,
                "layers": args.layers}, out / "dynamics.pt")

    print("\n[wm] held-out one-step RMSE (standardised units)")
    id_, md = final["identity_robot_rmse"], final["model_robot_rmse"]
    print(f"       identity          {id_:.5f}")
    if "constvel_robot_rmse" in final:
        print(f"       constant velocity {final['constvel_robot_rmse']:.5f}")
    cv = final.get("constvel_robot_rmse")
    gain_cv = f", {(1 - md / cv) * 100:+.1f}% vs constant velocity" if cv else ""
    print(f"       model             {md:.5f}   "
          f"({(1 - md / id_) * 100:+.1f}% vs identity{gain_cv})")
    if cv and md >= cv:
        print("       NOTE: the model does NOT beat constant velocity. It has not "
              "learned anything the action explains; do not plan with it.")
    if "model_slot_rmse" in final:
        print(f"       slots: model {final['model_slot_rmse']:.5f} vs identity "
              f"{final['identity_slot_rmse']:.5f}")
    final["split"] = split
    final["train_transitions"] = int(len(train_idx))
    final["test_transitions"] = int(len(test_idx))
    json.dump(final, open(out / "metrics.json", "w"), indent=1)
    print(f"[wm] -> {out}/dynamics.pt, metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

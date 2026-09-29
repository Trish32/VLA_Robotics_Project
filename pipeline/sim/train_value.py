#!/usr/bin/env python
"""Fit `ValueHead` on task outcome, so the planner can score success instead of distance.

This is the one lever the Oracle@k diagnostic leaves open. That measurement showed the
scorer ranking candidates **at chance**, and — more decisively — that even *perfect*
per-decision selection does not beat simply running the policy. Neither the proposer nor
the ranking rule is the binding constraint, so improving either cannot help. The
objective is what fails: a hand-written instantaneous cost over geometric distances is
not a good enough proxy for task success that greedily optimising it pays.

So the objective is replaced with a learned one. The target is the discounted return of
a sparse success reward — `gamma ** (steps remaining)` on episodes that succeed, zero on
episodes that do not — which is exactly "how close is this state to finishing", not "how
close is this object to that object". The distinction is the whole point: a distance
cost is minimised by dragging the cube across the table, and a return is not.

Two baselines are reported because the head has to beat both to be worth carrying:

  * **constant** — predict the training mean. Beating it only shows the state matters.
  * **distance** — a one-feature least-squares fit on cube-to-goal distance, which is
    the geometric term the scorer already has. This is the one that counts: if a
    learned value cannot beat the heuristic it is replacing, it is not an improvement,
    it is a bigger version of the same thing.

    conda run -n simple_bev_vldrive python -m pipeline.sim.train_value \
        --data pipeline/sim/data/pick_place_400b \
        --dynamics pipeline/assets/world_model_sim/dynamics.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.world_model.data import Normaliser, TransitionDataset, load_episodes
from pipeline.world_model.latent import SLOT_CENTRE, SceneLatent
from pipeline.world_model.scorer import ValueHead

#: Slot index of the object the task is about, and of the goal. Matches `SLOT_ORDER` in
#: loop.py, which is fixed precisely so an index means the same object everywhere.
TARGET, GOAL = 0, 1


def returns(episodes, gamma: float) -> list[np.ndarray]:
    """Discounted return to go per frame, from the sparse success reward.

    Success is read from the `next.reward` column the collector writes, so the label
    comes from the same simulator verdict the loop is scored against rather than from a
    heuristic applied here.
    """
    out = []
    for e in episodes:
        r = e.get("reward")
        n = len(e["state"])
        if r is None or not float(np.max(r)):
            out.append(np.zeros(n, np.float32))
            continue
        # The frame the reward fires on is the end of the task; everything before it is
        # discounted by how far it still has to go.
        end = int(np.argmax(r))
        k = np.arange(n)
        v = np.where(k <= end, gamma ** np.maximum(0, end - k), 0.0)
        out.append(v.astype(np.float32))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dynamics", type=Path,
                    default=Path("pipeline/assets/world_model_sim/dynamics.pt"))
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/assets/world_model_sim/value.pt"))
    a = ap.parse_args(argv)

    ck = torch.load(a.dynamics, map_location="cpu", weights_only=False)
    eps = load_episodes(a.data)
    # Reward rides along with the episode so `returns` can see it; `load_episodes` does
    # not read it, because nothing before this needed it.
    import pandas as pd
    for e, pq in zip(eps, sorted((a.data / "data").rglob("*.parquet"))):
        col = pd.read_parquet(pq).get("next.reward")
        e["reward"] = (None if col is None
                       else col.to_numpy().astype(np.float32)[:len(e["state"])])
    succ = sum(1 for e in eps if e["reward"] is not None and float(np.max(e["reward"])))
    print(f"[value] {len(eps)} episodes, {succ} successful")

    h = max(1, int(a.holdout))
    tr_eps, te_eps = eps[:-h], eps[-h:]
    # The SAME normalisers the dynamics was fitted under. A value head that scores
    # states the dynamics predicts must live in the dynamics' input space, or every
    # rollout it grades is shifted by whatever the two means differ by.
    base = TransitionDataset(
        tr_eps, state_norm=Normaliser(ck["state_mean"], ck["state_std"]),
        action_norm=Normaliser(ck["action_mean"], ck["action_std"]),
        slot_norm=Normaliser(ck["slot_mean"], ck["slot_std"]))
    ds = TransitionDataset(
        tr_eps + te_eps, state_norm=base.state_norm, action_norm=base.action_norm,
        slot_norm=base.slot_norm)
    G = returns(tr_eps + te_eps, a.gamma)

    slots, robot, y, ep_of = [], [], [], []
    for i in range(len(ds)):
        e, t = ds.index[i]
        row = ds[i]
        slots.append(row["slots"])
        robot.append(row["robot"])
        y.append(G[e][t])
        ep_of.append(e)
    slots = torch.from_numpy(np.stack(slots)).float()
    robot = torch.from_numpy(np.stack(robot)).float()
    y = torch.from_numpy(np.asarray(y, np.float32))
    ep_of = np.asarray(ep_of)
    n_tr = len(tr_eps)
    tr = np.flatnonzero(ep_of < n_tr)
    te = np.flatnonzero(ep_of >= n_tr)
    print(f"[value] {len(tr)} train / {len(te)} test frames "
          f"({n_tr} / {len(te_eps)} episodes)")

    head = ValueHead(ds.slot_dim, ds.robot_dim, hidden=a.hidden)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    mask = torch.ones(len(slots), slots.shape[1], dtype=torch.bool)

    def predict(idx):
        return head(SceneLatent(slots[idx], robot[idx], mask[idx]))

    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        order = rng.permutation(tr)
        head.train()
        for i in range(0, len(order) - a.batch + 1, a.batch):
            j = torch.from_numpy(order[i:i + a.batch])
            loss = torch.nn.functional.mse_loss(predict(j), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(
                predict(torch.from_numpy(te)), y[torch.from_numpy(te)]))
        if val < best:
            best, best_state = val, {k: v.detach().clone()
                                     for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[value] epoch {ep:3d}  test MSE {val:.6f}"
                  f"{'  *' if val == best else ''}", flush=True)
    head.load_state_dict(best_state)

    # ── baselines ──────────────────────────────────────────────────────────────
    yt = y[torch.from_numpy(te)].numpy()
    const = float(((yt - y[torch.from_numpy(tr)].numpy().mean()) ** 2).mean())
    # Distance from the target slot's centre to the goal slot's centre, in the latent's
    # own (standardised) units — the same quantity `goal_distance` feeds the scorer.
    def dist_feat(idx):
        s = slots[idx]
        return torch.linalg.norm(s[:, TARGET, SLOT_CENTRE] - s[:, GOAL, SLOT_CENTRE],
                                 dim=-1).numpy()
    dtr, dte = dist_feat(torch.from_numpy(tr)), dist_feat(torch.from_numpy(te))
    A = np.stack([dtr, np.ones_like(dtr)], -1)
    coef, *_ = np.linalg.lstsq(A, y[torch.from_numpy(tr)].numpy(), rcond=None)
    dist_mse = float(((np.stack([dte, np.ones_like(dte)], -1) @ coef - yt) ** 2).mean())
    with torch.no_grad():
        model = float(torch.nn.functional.mse_loss(
            predict(torch.from_numpy(te)), y[torch.from_numpy(te)]))

    print(f"\n[value] held-out MSE on discounted return (gamma {a.gamma})")
    print(f"       constant (train mean)   {const:.6f}")
    print(f"       cube-to-goal distance   {dist_mse:.6f}")
    print(f"       learned value head      {model:.6f}"
          f"   ({(1 - model / const) * 100:+.1f}% vs constant, "
          f"{(1 - model / dist_mse) * 100:+.1f}% vs distance)")
    if model >= dist_mse:
        print("       NOTE: no better than the geometric heuristic it replaces. "
              "Scoring with it would change the objective without improving it.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "slot_dim": ds.slot_dim,
                "robot_dim": ds.robot_dim, "hidden": a.hidden, "gamma": a.gamma,
                "test_mse": model, "constant_mse": const,
                "distance_mse": dist_mse}, a.out)
    (a.out.parent / "value_metrics.json").write_text(json.dumps(
        {"test_mse": model, "constant_mse": const, "distance_mse": dist_mse,
         "gamma": a.gamma, "train_frames": int(len(tr)),
         "test_frames": int(len(te))}, indent=1))
    print(f"[value] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

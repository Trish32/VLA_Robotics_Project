#!/usr/bin/env python
"""Behaviour-clone the demonstrator into the small policy the closed loop runs.

The scripted waypoint machine is a data source, not a policy: it reads exact object
positions and executes a fixed plan, so a loop driven by it measures the simulator
rather than anything learned. `ChunkPolicy` is what replaces it — a small MLP mapping
proprioception plus **perceived** object offsets to an H-step action chunk.

Chunked rather than single-step for the reason it is standard in manipulation: a
per-frame policy re-decides the whole plan every frame, and the action distribution is
multi-modal exactly at grasp and release, so the re-decision jitters. It is also what
makes the world model useful — a chunk *is* the candidate the planner scores, so one
rollout per chunk rather than one per frame.

Observations use object positions **relative to the tip**, which is what makes the
policy survive the layout randomisation: handed absolute coordinates a small MLP
memorises the table region the cube was sampled from.

    conda run -n simple_bev_vldrive python -m pipeline.sim.train_bc \
        --data pipeline/sim/data/pick_place_400 --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.sim.collect import TRACKED
from pipeline.sim.policy import ChunkPolicy, policy_observation

#: Where the fingertip sits inside `observation.state`; see `PickPlaceEnv._observe`.
TIP = slice(12, 15)


def build(root: Path, horizon: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(N, obs) inputs, (N, H, 4) targets, and each row's episode index.

    Chunks are clipped at the episode boundary by repeating the final action rather
    than by dropping the last H frames. Dropping them removes precisely the release and
    retreat — the part of the task with the fewest examples and the highest chance of
    going wrong.
    """
    import pandas as pd

    X, Y, ep_of = [], [], []
    for e, pq in enumerate(sorted((root / "data").rglob("*.parquet"))):
        df = pd.read_parquet(pq)
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        z = np.load(root / "tracks" / f"{pq.stem}_tracks.npz", allow_pickle=True)
        tracks = z["tracks"].astype(np.float32)
        labels = [str(x) for x in z["labels"]]
        n = min(len(state), len(tracks))
        for t in range(n):
            objects = {lab: tracks[t, i, :3] for i, lab in enumerate(labels)}
            X.append(policy_observation(state[t], state[t][TIP], objects))
            idx = np.clip(np.arange(t, t + horizon), 0, len(action) - 1)
            Y.append(action[idx])
            ep_of.append(e)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), ep_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=int, default=40, help="episodes, not frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("pipeline/assets/sim_policy/chunk_policy.pt"))
    a = ap.parse_args(argv)

    X, Y, ep_of = build(a.data, a.horizon)
    ep_of = np.asarray(ep_of)
    n_eps = int(ep_of.max()) + 1
    if n_eps <= a.holdout:
        raise SystemExit(f"{n_eps} episodes cannot give a {a.holdout}-episode holdout")
    # By episode, never by frame: adjacent frames at 42 Hz are nearly identical, so a
    # frame-wise holdout measures interpolation and reports it as generalisation.
    tr = np.flatnonzero(ep_of < n_eps - a.holdout)
    te = np.flatnonzero(ep_of >= n_eps - a.holdout)
    print(f"[bc] {len(X)} frames, {n_eps} episodes -> "
          f"{len(tr)} train / {len(te)} test")

    pol = ChunkPolicy(X.shape[1], horizon=a.horizon, hidden=a.hidden, seed=a.seed)
    pol.obs_mean = X[tr].mean(0)
    pol.obs_std = np.maximum(X[tr].std(0), 1e-3)
    xs = torch.from_numpy((X - pol.obs_mean) / pol.obs_std)
    ys = torch.from_numpy(Y.reshape(len(Y), -1))

    opt = torch.optim.Adam(pol.net.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        order = rng.permutation(tr)
        pol.net.train()
        for i in range(0, len(order) - a.batch + 1, a.batch):
            j = torch.from_numpy(order[i:i + a.batch])
            loss = torch.nn.functional.mse_loss(pol.net(xs[j]), ys[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pol.net.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(
                pol.net(xs[te]), ys[te]))
        if val < best:
            best, best_state = val, {k: v.detach().clone()
                                     for k, v in pol.net.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[bc] epoch {ep:3d}  val {val:.6f}{'  *' if val == best else ''}",
                  flush=True)
    pol.net.load_state_dict(best_state)

    # Against the trivial predictor, as everywhere else in this repo. "Do nothing" is a
    # strong baseline on a chunked action space: most frames of most episodes are
    # small, smooth moves, so a policy that cannot beat zero has learned nothing.
    with torch.no_grad():
        model = float(torch.nn.functional.mse_loss(pol.net(xs[te]), ys[te]))
    zero = float((ys[te] ** 2).mean())
    print(f"\n[bc] held-out chunk MSE   zero-action {zero:.6f}   model {model:.6f}"
          f"   ({(1 - model / zero) * 100:+.1f}%)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": pol.net.state_dict(), "obs_dim": int(X.shape[1]),
                "horizon": a.horizon, "hidden": a.hidden,
                "obs_mean": pol.obs_mean, "obs_std": pol.obs_std,
                "labels": list(TRACKED)}, a.out)
    (a.out.parent / "metrics.json").write_text(json.dumps(
        {"held_out_chunk_mse": model, "zero_action_mse": zero,
         "train_frames": len(tr), "test_frames": len(te),
         "test_episodes": a.holdout, "horizon": a.horizon}, indent=1))
    print(f"[bc] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

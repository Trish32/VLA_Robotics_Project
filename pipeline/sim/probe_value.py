#!/usr/bin/env python
"""Two questions about the value head, both answered against the same held-out frames.

**Is the value head the bottleneck?** It is fitted on the discounted return of a sparse
success reward and beats a distance heuristic by 84%, and none of that says whether its
*output* tracks the quantity a planner needs — value-to-go at the state it is actually
looking at. So: correlate the shipped head's prediction with ground-truth value-to-go,
and fit a fresh probe on the same latents to find the ceiling. If the probe is far above
the head, the head is undertrained or mis-fitted; if the probe is also low, value-to-go
is not linearly present in this latent and no amount of head-tuning reaches it.

**Is return the wrong target for a veto?** The veto does not need to know how good a
state is, only whether it is doomed. Those are different learning problems: return is a
regression dominated by the long middle of an episode, while failure is a classification
whose signal lives in the few frames where it becomes inevitable. A head trained on the
**failure label** is fitted here and scored by AUC against the same held-out episodes,
alongside the return head thresholded — which is what the current veto does.

    conda run -n simple_bev_vldrive python -m pipeline.sim.probe_value \
        --data pipeline/sim/data/plans_600fix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _r2(pred, y_tr_mean, y):
    resid = float(np.mean((pred - y) ** 2))
    base = float(np.mean((y_tr_mean - y) ** 2))
    return 1.0 - resid / base if base > 0 else float("nan")


def _ridge(x, y, ridge=1e-3):
    xt = np.concatenate([x, np.ones((len(x), 1))], 1)
    a = xt.T @ xt + ridge * np.eye(xt.shape[1])
    return np.linalg.solve(a, xt.T @ y)


def _auc(x, y):
    """Mann-Whitney AUC with averaged ranks for ties (see bug_log.txt [S14])."""
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dynamics", type=Path,
                    default=Path("pipeline/assets/world_model_plans/dynamics.pt"))
    ap.add_argument("--value", type=Path,
                    default=Path("pipeline/assets/world_model_plans/value.pt"))
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--holdout", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--save-failure-head", type=Path, default=None,
                    help="write the failure-label head so calibrate_veto can score it "
                         "against per-decision rewind ground truth, which is the only "
                         "labelling that can say WHEN an episode became doomed")
    a = ap.parse_args(argv)

    import pandas as pd

    from pipeline.world_model.data import (Normaliser, TransitionDataset,
                                           load_episodes)
    from pipeline.world_model.latent import SceneLatent
    from pipeline.world_model.scorer import ValueHead
    from pipeline.sim.train_value import returns

    ck = torch.load(a.dynamics, map_location="cpu", weights_only=False)
    eps = load_episodes(a.data)
    for e, pq in zip(eps, sorted((a.data / "data").rglob("*.parquet"))):
        col = pd.read_parquet(pq).get("next.reward")
        e["reward"] = (None if col is None
                       else col.to_numpy().astype(np.float32)[:len(e["state"])])

    h = max(1, int(a.holdout))
    tr_eps, te_eps = eps[:-h], eps[-h:]
    norms = dict(state_norm=Normaliser(ck["state_mean"], ck["state_std"]),
                 action_norm=Normaliser(ck["action_mean"], ck["action_std"]),
                 slot_norm=Normaliser(ck["slot_mean"], ck["slot_std"]))
    ds = TransitionDataset(tr_eps + te_eps, **norms)
    G = returns(tr_eps + te_eps, a.gamma)
    # Failure label: this episode never reaches the success reward. Applied to every
    # frame of it, which is the label a veto would be trained on — "from here, does
    # this end badly" — and deliberately NOT a per-frame judgement, because nothing in
    # the data says at which frame an episode became doomed.
    failed = [not (e["reward"] is not None and float(np.max(e["reward"])))
              for e in tr_eps + te_eps]

    slots, robot, y, lab, ep_of = [], [], [], [], []
    for i in range(len(ds)):
        e, t = ds.index[i]
        row = ds[i]
        slots.append(row["slots"])
        robot.append(row["robot"])
        y.append(G[e][t])
        lab.append(failed[e])
        ep_of.append(e)
    slots = torch.from_numpy(np.stack(slots)).float()
    robot = torch.from_numpy(np.stack(robot)).float()
    y = np.asarray(y, np.float32)
    lab = np.asarray(lab, bool)
    ep_of = np.asarray(ep_of)
    n_tr = len(tr_eps)
    tr = np.flatnonzero(ep_of < n_tr)
    te = np.flatnonzero(ep_of >= n_tr)
    mask = torch.ones(len(slots), slots.shape[1], dtype=torch.bool)
    flat = np.concatenate([slots.numpy().reshape(len(slots), -1),
                           robot.numpy()], 1).astype(np.float64)
    print(f"[pv] {len(tr)} train / {len(te)} test frames "
          f"({n_tr} / {len(te_eps)} episodes, {lab[te].mean()*100:.1f}% of test frames "
          f"from failed episodes)")

    # ── 1. value-to-go ────────────────────────────────────────────────────────
    # Dimensions and width come from the checkpoint, never from this script's
    # defaults: silently constructing a differently-shaped head and letting
    # load_state_dict fail is the good case — quietly loading a head of the wrong
    # width would be the bad one.
    vck = torch.load(a.value, map_location="cpu", weights_only=False)
    head = ValueHead(vck["slot_dim"], vck["robot_dim"], hidden=vck["hidden"])
    head.load_state_dict(vck["state_dict"])
    head.eval()
    with torch.no_grad():
        vhat = np.concatenate([
            head(SceneLatent(slots[i:i + 4096], robot[i:i + 4096],
                             mask[i:i + 4096])).numpy().ravel()
            for i in range(0, len(slots), 4096)])

    w = _ridge(flat[tr], y[tr])
    probe = np.concatenate([flat, np.ones((len(flat), 1))], 1) @ w
    ymean = y[tr].mean()
    print("\n[pv] value-to-go on held-out frames")
    print(f"      {'predictor':<34} {'R2':>7} {'Pearson r':>10} {'Spearman':>9}")
    for name, v in (("shipped value head", vhat), ("linear probe on the same latent",
                                                   probe)):
        r = float(np.corrcoef(v[te], y[te])[0, 1])
        rs = float(np.corrcoef(np.argsort(np.argsort(v[te])),
                               np.argsort(np.argsort(y[te])))[0, 1])
        print(f"      {name:<34} {_r2(v[te], ymean, y[te]):7.3f} {r:10.3f} {rs:9.3f}")
    agree = float(np.corrcoef(vhat[te], probe[te])[0, 1])
    print(f"      head vs probe, on their own outputs          r = {agree:+.3f}")

    # ── 2. a head trained on the failure label instead ────────────────────────
    clf = ValueHead(ds.slot_dim, ds.robot_dim, hidden=vck["hidden"])
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    rng = np.random.default_rng(a.seed)
    yt = torch.from_numpy(lab[tr].astype(np.float32))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), 256):
            b = tr[perm[i:i + 256]]
            logit = clf(SceneLatent(slots[b], robot[b], mask[b])).ravel()
            loss = lossf(logit, torch.from_numpy(lab[b].astype(np.float32)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        flog = np.concatenate([
            clf(SceneLatent(slots[i:i + 4096], robot[i:i + 4096],
                            mask[i:i + 4096])).numpy().ravel()
            for i in range(0, len(slots), 4096)])

    print("\n[pv] predicting FAILURE on held-out episodes (AUC, 0.5 = no information)")
    print(f"      {'signal':<40} {'pooled':>8} {'within-episode':>15}")

    def within(v):
        num = den = 0.0
        for e in np.unique(ep_of[te]):
            m = ep_of == e
            if lab[m].all() or not lab[m].any():
                continue
            n = int(lab[m].sum()) * int((~lab[m]).sum())
            num += _auc(v[m], lab[m]) * n
            den += n
        return num / den if den else float("nan")

    rows = {}
    for name, v in (("return head, negated (what the veto uses)", -vhat),
                    ("linear probe on value-to-go, negated", -probe),
                    ("head trained on the failure label", flog)):
        p_auc = _auc(v[te], lab[te])
        rows[name] = [p_auc, within(v)]
        w_auc = within(v)
        print(f"      {name:<40} {p_auc:8.3f} "
              f"{'n/a' if np.isnan(w_auc) else f'{w_auc:15.3f}'}")
    print("      within-episode is n/a here by construction: the label is per-EPISODE,")
    print("      so inside one episode every frame carries the same class and there is")
    print("      no pair to order. That is itself the finding — this data cannot say")
    print("      WHEN an episode became doomed, only THAT it was.")
    if a.save_failure_head:
        a.save_failure_head.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": clf.state_dict(), "slot_dim": ds.slot_dim,
                    "robot_dim": ds.robot_dim, "hidden": vck["hidden"],
                    "target": "failure", "pooled_auc": rows[
                        "head trained on the failure label"][0]},
                   a.save_failure_head)
        print(f"[pv] -> {a.save_failure_head}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

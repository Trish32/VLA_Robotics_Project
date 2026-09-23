"""Scoring predicted futures: would this action have been a good idea?

The terms are deliberately GEOMETRIC and readable rather than a single learned scalar.
A learned value trained on five demonstrations would be a confident fiction, and a
planner that rejects an action needs to be able to say why — "35 cm^3 of predicted
interpenetration" is actionable, "value 0.31" is not. `ValueHead` exists for when real
reward data does, and the scorer blends it in only if one is supplied.

Every term is a COST: lower is better, zero is neutral. They are combined by weighted
sum and the breakdown is always returned alongside the total, so a selection can be
explained and a bad weight can be found.

Costs are in physical units where possible — cubic metres of overlap, metres of
unsupported drop — so the weights carry the unit conversion explicitly instead of
hiding it inside an arbitrary normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from pipeline.world_model.latent import SceneLatent


@dataclass
class ScoreWeights:
    """What the planner is trying to avoid, and how much it cares.

    `risk` defaults to **zero, on measured grounds**. The ensemble spread was checked
    against realised rollout error on a held-out episode (`evaluate.py`) and the rank
    correlation is **-0.071** — it does not predict which rollout will be wrong. What it
    does track is how error grows with HORIZON, and closely: over 8 steps the realised
    error grows 14.2x while the spread grows 14.1x. So it is a good horizon discount and
    a useless per-candidate discriminator, and since every candidate in a plan shares the
    same horizon, weighting it changes nothing except to look principled.

    Raise it when the dynamics model is good enough for the spread to mean something —
    the term is correct in principle and the measurement, not the idea, is what zeroed
    it. `evaluate.py` reports the correlation to check against.
    """

    collision: float = 40.0     # per m^3 of predicted interpenetration
    stability: float = 8.0      # per metre an object is left unsupported
    progress: float = 1.0       # per metre of remaining distance to the goal
    risk: float = 0.0           # measured uncorrelated with error; see above
    effort: float = 0.05        # per unit of action magnitude; breaks ties toward calm


@dataclass
class ScoreBreakdown:
    """Per-candidate costs. Every field is (K,), so candidates can be compared term by
    term rather than only on the total."""

    total: torch.Tensor
    collision: torch.Tensor
    stability: torch.Tensor
    progress: torch.Tensor
    risk: torch.Tensor
    effort: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def best(self) -> int:
        return int(torch.argmin(self.total).item())

    def explain(self, i: int) -> str:
        return (f"total {self.total[i]:.4f}  = collision {self.collision[i]:.4f}"
                f" + stability {self.stability[i]:.4f} + progress {self.progress[i]:.4f}"
                f" + risk {self.risk[i]:.4f} + effort {self.effort[i]:.4f}")


def pairwise_overlap_volume(z: SceneLatent) -> torch.Tensor:
    """Total interpenetration volume between distinct real slots, (B,) in m^3.

    Axis-aligned, matching how extents are carried everywhere else in the stack. This
    is a proxy for collision, not a contact model: it answers "does the prediction put
    two solid things in the same place", which is the question that should veto an
    action.
    """
    lo, hi = z.aabb()                                        # (B, M, 3)
    inter = (torch.minimum(hi[:, :, None], hi[:, None, :])
             - torch.maximum(lo[:, :, None], lo[:, None, :])).clamp(min=0)
    vol = inter.prod(-1)                                     # (B, M, M)
    real = z.mask.float()
    pair = real[:, :, None] * real[:, None, :]
    eye = torch.eye(z.n_slots, device=vol.device, dtype=vol.dtype)
    # Halve: each unordered pair appears twice. Self-overlap is excluded, not halved.
    return ((vol * pair * (1 - eye)).sum((-1, -2)) / 2.0)


def vol_like(x: torch.Tensor) -> torch.Tensor:
    """(B, M) -> (B, M, M) filler, so `torch.where` has a shape to broadcast against."""
    return x[:, :, None].expand(x.shape[0], x.shape[1], x.shape[1])


def unsupported_drop(z: SceneLatent, *, ground_z: float | torch.Tensor = 0.0,
                     tol: float = 0.05) -> torch.Tensor:
    """How far each object would fall if released, summed over the scene, (B,) metres.

    An object is supported if its underside is within `tol` of the ground or of the top
    of another object whose footprint it overlaps. This is only meaningful because the
    world frame is gravity-aligned upstream (`pipeline/gravity.py`) — on the raw SLAM
    frame the supporting plane's normal sat 50 deg off z and "up" was not up, which
    would make every stability verdict here noise.
    """
    lo, hi = z.aabb()
    bottom, top = lo[..., 2], hi[..., 2]                     # (B, M)
    gz = torch.as_tensor(ground_z, device=lo.device, dtype=lo.dtype)
    drop = (bottom - gz).clamp(min=0)                        # distance to the floor

    # Footprint overlap in x/y with every other slot.
    over = ((torch.minimum(hi[:, :, None, :2], hi[:, None, :, :2])
             - torch.maximum(lo[:, :, None, :2], lo[:, None, :, :2])).clamp(min=0)
            .prod(-1) > 0)                                   # (B, M, M)
    real = z.mask
    below = top[:, None, :] <= bottom[:, :, None] + tol      # j's top at/under i's base
    eye = torch.eye(z.n_slots, device=lo.device, dtype=torch.bool)
    cand = over & below & real[:, None, :] & ~eye
    # Gap to the highest surface under each object; +inf where nothing qualifies, which
    # `torch.where` then replaces with the drop to the floor.
    gaps = torch.where(cand, bottom[:, :, None] - top[:, None, :],
                       torch.full_like(vol_like(bottom), float("inf")))
    nearest = gaps.min(-1).values                            # (B, M)
    supported_gap = torch.where(torch.isfinite(nearest), nearest.clamp(min=0), drop)
    unsupported = (supported_gap - tol).clamp(min=0)
    return (unsupported * real.float()).sum(-1)


def goal_distance(z: SceneLatent, target: int,
                  goal_xyz: torch.Tensor) -> torch.Tensor:
    """Distance from the target slot's centre to the goal point, (B,) metres."""
    if goal_xyz.dim() == 1:
        goal_xyz = goal_xyz.unsqueeze(0).expand(z.batch, 3)
    return torch.linalg.norm(z.centre[:, target] - goal_xyz, dim=-1)


class ValueHead(nn.Module):
    """A learned cost on top of the geometric terms.

    Deliberately small and deliberately optional. It is only meaningful with real
    reward or preference data; trained on a handful of demonstrations it would mostly
    memorise them, so the planner runs without it by default and the geometric terms
    carry the decision.
    """

    def __init__(self, slot_dim: int, robot_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim + robot_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z: SceneLatent) -> torch.Tensor:
        m = z.mask.float().unsqueeze(-1)
        pooled = (z.slots * m).sum(1) / m.sum(1).clamp(min=1)
        return self.net(torch.cat([pooled, z.robot], -1)).squeeze(-1)


class TrajectoryScorer:
    """Turns a set of predicted rollouts into one cost per candidate."""

    def __init__(self, weights: ScoreWeights | None = None, *,
                 value_head: ValueHead | None = None,
                 discount: float = 0.9, ground_z: float = 0.0) -> None:
        self.w = weights or ScoreWeights()
        self.value_head = value_head
        self.discount = discount
        self.ground_z = ground_z

    def __call__(self, states: list[SceneLatent], uncertainty: torch.Tensor,
                 actions: torch.Tensor, *, target: int | None = None,
                 goal_xyz: torch.Tensor | None = None,
                 baseline: SceneLatent | None = None) -> ScoreBreakdown:
        """`states` is the N predicted steps; every tensor is batched over candidates.

        Costs are discounted over the horizon: a collision predicted at step 2 is more
        credible than one at step 20, because the rollout that produced it has
        accumulated less error. The same discount therefore does double duty as a
        horizon-confidence weighting, which is why it is not 1.0 by default.

        `baseline` is the scene BEFORE acting. Supplying it makes collision and
        stability measure what the action CAUSED rather than what was already true, and
        that distinction decides whether the terms can discriminate at all: on the real
        TUM scene every instance box already overlaps its neighbours, so absolute
        collision cost was identical (0.2400) for all eight candidates and vetoed all of
        them. A planner choosing among actions cannot be charged for overlap that is
        common to every choice — that is a fact about the segmentation, not about any
        action.
        """
        if not states:
            raise ValueError("no predicted states to score")
        K = states[0].batch
        dev, dt = states[0].slots.device, states[0].slots.dtype
        zero = torch.zeros(K, device=dev, dtype=dt)
        coll = stab = prog = zero.clone()
        norm = 0.0
        for t, z in enumerate(states):
            g = self.discount ** t
            norm += g
            coll = coll + g * pairwise_overlap_volume(z)
            stab = stab + g * unsupported_drop(z, ground_z=self.ground_z)
            if target is not None and goal_xyz is not None:
                prog = prog + g * goal_distance(z, target, goal_xyz)
        coll, stab, prog = coll / norm, stab / norm, prog / norm
        if baseline is not None:
            # Only the INDUCED part is chargeable. Clamped at zero: an action that
            # happens to reduce pre-existing overlap gets no negative cost, because the
            # dynamics is not trusted enough to pay a bonus for a predicted improvement.
            coll = (coll - pairwise_overlap_volume(baseline)).clamp(min=0)
            stab = (stab - unsupported_drop(baseline,
                                            ground_z=self.ground_z)).clamp(min=0)

        risk = uncertainty.mean(-1) if uncertainty.numel() else zero.clone()
        effort = actions.flatten(1).abs().mean(-1)

        total = (self.w.collision * coll + self.w.stability * stab
                 + self.w.progress * prog + self.w.risk * risk + self.w.effort * effort)
        terms = {}
        if self.value_head is not None:
            v = self.value_head(states[-1])
            terms["value"] = v
            total = total + v
        return ScoreBreakdown(total=total, collision=self.w.collision * coll,
                              stability=self.w.stability * stab,
                              progress=self.w.progress * prog,
                              risk=self.w.risk * risk,
                              effort=self.w.effort * effort, terms=terms)

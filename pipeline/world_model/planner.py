"""Simulate each candidate action before executing any of them.

The loop the rest of this package exists to serve:

    1. the VLA proposes K candidate action chunks;
    2. the ensemble rolls each one N steps through the latent dynamics;
    3. the scorer costs every predicted future;
    4. the lowest-cost candidate is executed — and the runner-up, the margin and the
       per-term breakdown come back with it.

Returning the losers matters. A planner that emits only its winner cannot be audited:
you cannot tell a decisive choice from a coin flip between two near-identical costs, and
"the model picked it" is not a reason. `PlanResult.margin` and `.explain()` exist so a
selection can be argued with.

**This selects among the policy's own proposals; it does not optimise actions.** There is
no gradient through the dynamics into the action here, deliberately: with a dynamics
model fit to a handful of demonstrations, optimising against it finds its errors rather
than good actions. Ranking what the policy already considered reasonable is the honest
use of a weak world model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pipeline.world_model.dynamics import DynamicsEnsemble
from pipeline.world_model.latent import SceneLatent
from pipeline.world_model.scorer import ScoreBreakdown, TrajectoryScorer


@dataclass
class PlanResult:
    """The chosen action and everything needed to second-guess the choice."""

    index: int                      # which candidate won
    action: torch.Tensor            # (N, A) the winning chunk
    scores: ScoreBreakdown          # per-candidate costs, (K,)
    states: list[SceneLatent]       # predicted rollout of the WINNER, length N
    uncertainty: torch.Tensor       # (K, N) ensemble disagreement, metres
    rejected: list[int]             # candidate indices vetoed outright

    @property
    def margin(self) -> float:
        """Cost gap to the runner-up. Near zero means the choice was arbitrary and the
        scorer did not actually discriminate — worth knowing before trusting it."""
        if self.scores.total.numel() < 2:
            return float("inf")
        ordered = torch.sort(self.scores.total).values
        return float(ordered[1] - ordered[0])

    def explain(self) -> str:
        lines = [f"chose candidate {self.index} of {self.scores.total.numel()} "
                 f"(margin {self.margin:.4f})",
                 f"  winner   {self.scores.explain(self.index)}"]
        if self.rejected:
            lines.append(f"  vetoed   {self.rejected}")
        peak = float(self.uncertainty[self.index].max()) if self.uncertainty.numel() else 0.0
        lines.append(f"  rollout uncertainty peaks at {peak * 100:.2f} cm")
        return "\n".join(lines)


class ActionPlanner:
    """Roll out candidate actions and pick one."""

    def __init__(self, dynamics: DynamicsEnsemble, scorer: TrajectoryScorer | None = None,
                 *, veto_collision_m3: float | None = 1e-3,
                 veto_uncertainty_m: float | None = None) -> None:
        """`veto_*` are hard constraints, applied before the weighted sum.

        A veto is not the same as a large weight: a weight lets a big enough gain
        elsewhere buy a collision, which is exactly what should never happen. Set to
        None to disable and let everything be traded off.
        """
        self.dynamics = dynamics
        self.scorer = scorer or TrajectoryScorer()
        self.veto_collision_m3 = veto_collision_m3
        self.veto_uncertainty_m = veto_uncertainty_m

    @torch.no_grad()
    def plan(self, z0: SceneLatent, candidates: torch.Tensor, *,
             target: int | None = None,
             goal_xyz: torch.Tensor | None = None) -> PlanResult:
        """`z0` is a single scene (batch 1); `candidates` is (K, N, A)."""
        if z0.batch != 1:
            raise ValueError(f"plan() takes one scene, got batch {z0.batch}")
        if candidates.dim() != 3:
            raise ValueError(
                f"candidates must be (K, N, A), got {tuple(candidates.shape)}")
        K = candidates.shape[0]
        if K == 0:
            raise ValueError("no candidate actions to choose between")

        states, unc = self.dynamics.rollout(z0.expand_batch(K), candidates)
        # The pre-action scene is the baseline every candidate is charged against, so
        # the costs describe what each action would CAUSE.
        scores = self.scorer(states, unc, candidates, target=target, goal_xyz=goal_xyz,
                             baseline=z0.expand_batch(K))

        # Hard constraints. Pushing the cost to +inf rather than dropping the rows keeps
        # every candidate's index and breakdown intact for inspection.
        total = scores.total.clone()
        rejected: list[int] = []
        if self.veto_collision_m3 is not None:
            bad = scores.collision / max(self.scorer.w.collision, 1e-9) > self.veto_collision_m3
            rejected += torch.nonzero(bad).flatten().tolist()
        if self.veto_uncertainty_m is not None and unc.numel():
            bad_u = unc.max(-1).values > self.veto_uncertainty_m
            rejected += torch.nonzero(bad_u).flatten().tolist()
        rejected = sorted(set(rejected))
        if rejected and len(rejected) < K:
            total[torch.tensor(rejected, device=total.device)] = float("inf")
        # If EVERY candidate is vetoed, fall back to ranking them normally rather than
        # returning nothing: the caller still has to do something, and it should be the
        # least bad option with the veto reported, not an arbitrary one.

        index = int(torch.argmin(total).item())
        chosen = [SceneLatent(s.slots[index:index + 1], s.robot[index:index + 1],
                              s.mask[index:index + 1]) for s in states]
        return PlanResult(index=index, action=candidates[index], scores=scores,
                          states=chosen, uncertainty=unc, rejected=rejected)


def perturbed_candidates(base: torch.Tensor, k: int, *, scale: float = 0.05,
                         seed: int = 0) -> torch.Tensor:
    """K variations on one action chunk, the first of which is the chunk itself.

    A stand-in for sampling the policy K times where re-running it is too expensive.
    Keeping the original as candidate 0 means the planner can only do at least as well
    as the unmodified policy on the scorer's own terms — it never has to discard the
    policy's answer to have something to choose.
    """
    if base.dim() != 2:
        raise ValueError(f"base must be (N, A), got {tuple(base.shape)}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(k - 1, *base.shape, generator=g) * scale
    return torch.cat([base.unsqueeze(0), base.unsqueeze(0) + noise.to(base.device)], 0)

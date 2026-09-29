"""Where an object can be grasped and where it can be placed, from perception alone.

Built because the Oracle@k measurement said this is the only remaining option. Ranking
candidates is worth doing exactly when the candidates differ in **outcome**, and
perturbation-based proposal does not produce candidates that do: at the default scale
all eight lead to the same result (headroom +0.0 points), and widening the perturbation
only makes them differ by being worse (reachability 96.7% -> 70.0% -> 36.7% as the
spread grows). Diversity and quality are the same knob turned in opposite directions.

That also makes perturbation an **invalid ranking benchmark**, not merely a weak
proposer, and the distinction matters for what the earlier ablations mean. A test where
every candidate leads to the same outcome scores a perfect ranker and a coin the same,
so the run of negative results measured on it is a property of the test and cannot be
cited against the scorer. **This module supplies the valid one**: on affordance
candidates the same scorer reaches pairwise 0.646 / Kendall tau +0.291 along the
policy's trajectory, well above the chance it was previously reported at.

An affordance proposer breaks that coupling. Every candidate here is a **valid plan** —
a real grasp point on the object and a real release point inside the container — so the
set can differ substantially without any member being a degraded copy of another.

Everything is derived from the **perceived** box: the graspable region comes from the
object's own extent minus the jaw's needs, and the placement region from the
container's extent minus the object's. Hardcoding the arena's dimensions would make
this work here and nowhere else, and would smuggle ground truth into a module whose
whole claim is that it uses none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Half the jaw opening the gripper can close on, metres. A grasp point must leave the
#: object narrow enough here to be pinched at all.
JAW_HALF_WIDTH = 0.036
#: Pad half-length along the approach axis. A grasp offset larger than this slides the
#: pads off the object's face.
PAD_HALF = 0.027


@dataclass(frozen=True)
class GraspPlan:
    """One task-meaningful way to do the task."""

    grasp_offset: tuple[float, float]     # on the object's top face, world xy
    release_offset: tuple[float, float]   # within the container, world xy
    transit_z: float
    margin: float                          # metres of clearance the plan retains

    def as_kwargs(self) -> dict:
        return {"grasp_offset": self.grasp_offset,
                "release_offset": self.release_offset,
                "transit_z": self.transit_z}


def graspable_radius(extent: np.ndarray) -> float:
    """How far off-centre a top-down grasp can land and still hold.

    Limited by two different things and the smaller wins: the pads must still overlap
    the object's face (`PAD_HALF`), and the object must still fit the jaw across the
    grasp axis. An object wider than the jaw has no graspable region at all, which is
    reported as zero rather than as a small positive number — a plan that cannot be
    executed is worse than no alternative plan.
    """
    e = np.asarray(extent, float)
    if float(np.min(e[:2])) > 2 * JAW_HALF_WIDTH:
        return 0.0
    return max(0.0, min(PAD_HALF, float(np.min(e[:2])) / 2) - 0.004)


def placeable_radius(container_extent: np.ndarray, object_extent: np.ndarray) -> float:
    """How far off-centre the object can be released and still land inside.

    Uses the object's diagonal rather than its width: a cube released at 45 degrees
    needs its corner to clear, and half the width would let the corner overhang.
    """
    c = np.asarray(container_extent, float)
    o = np.asarray(object_extent, float)
    inner = float(np.min(c[:2])) / 2
    half_diag = float(np.linalg.norm(o[:2])) / 2
    return max(0.0, inner - half_diag - 0.004)


def propose(cube_extent, bowl_extent, k: int, rng: np.random.Generator, *,
            base_transit: float = 0.20, transit_span: float = 0.05,
            workspace_z: tuple[float, float] = (0.020, 0.40)) -> list[GraspPlan]:
    """K distinct plans, the first of which is the nominal centre-grasp centre-place.

    Candidate 0 is the unmodified plan for the same reason `perturbed_candidates` keeps
    the policy's own chunk: the proposer must never be able to do worse than not
    proposing, and a planner that has discarded the obvious plan cannot recover it.

    Offsets are drawn on a **deterministic ring** rather than from a Gaussian. A
    Gaussian concentrates near zero, which would rebuild the same failure mode this
    module exists to escape — most candidates indistinguishable from the nominal one,
    a few far enough out to be bad. A ring spends the budget on genuinely different
    plans and keeps every one of them inside the feasible region.
    """
    gr = graspable_radius(cube_extent)
    pr = placeable_radius(bowl_extent, cube_extent)
    plans = [GraspPlan((0.0, 0.0), (0.0, 0.0), base_transit, min(gr, pr))]
    if k <= 1:
        return plans
    phase = float(rng.uniform(0, 2 * np.pi))
    for i in range(k - 1):
        a = phase + 2 * np.pi * i / (k - 1)
        # Alternate between the inner and outer part of the feasible annulus so the set
        # covers magnitude as well as direction; radius 0 would duplicate candidate 0.
        f = 0.55 if i % 2 else 0.95
        g = (gr * f * np.cos(a), gr * f * np.sin(a))
        r = (pr * f * np.cos(a + np.pi), pr * f * np.sin(a + np.pi))
        z = float(np.clip(base_transit + transit_span * np.cos(a),
                          workspace_z[0] + 0.02, workspace_z[1] - 0.02))
        plans.append(GraspPlan(g, r, z, min(gr, pr) * (1 - f)))
    return plans


def random_plan(cube_extent, bowl_extent, rng: np.random.Generator, *,
                base_transit: float = 0.20, transit_span: float = 0.05,
                workspace_z: tuple[float, float] = (0.020, 0.40)) -> GraspPlan:
    """One plan sampled uniformly over the feasible region, for data collection.

    Uniform over the **disc**, not over (radius, angle) independently — sampling the
    radius uniformly would concentrate the draws near the centre and leave the outer
    ring, which is exactly where the proposer's candidates live, barely covered. The
    point of this sampler is coverage of the candidate manifold, so it has to be
    uniform in area.

    `propose` uses a deterministic ring instead, because a proposer wants spread across
    K candidates while a collector wants coverage across many episodes.
    """
    gr = graspable_radius(cube_extent)
    pr = placeable_radius(bowl_extent, cube_extent)
    ga, pa = rng.uniform(0, 2 * np.pi, 2)
    grad = gr * np.sqrt(rng.uniform())
    prad = pr * np.sqrt(rng.uniform())
    z = float(np.clip(base_transit + transit_span * rng.uniform(-1, 1),
                      workspace_z[0] + 0.02, workspace_z[1] - 0.02))
    return GraspPlan((float(grad * np.cos(ga)), float(grad * np.sin(ga))),
                     (float(prad * np.cos(pa)), float(prad * np.sin(pa))), z,
                     float(min(gr - grad, pr - prad)))


def describe(plans: list[GraspPlan]) -> str:
    g = np.array([p.grasp_offset for p in plans])
    r = np.array([p.release_offset for p in plans])
    return (f"{len(plans)} plans · grasp spread "
            f"{np.linalg.norm(g, axis=-1).max() * 100:.1f} cm · release spread "
            f"{np.linalg.norm(r, axis=-1).max() * 100:.1f} cm · transit "
            f"{min(p.transit_z for p in plans):.3f}-"
            f"{max(p.transit_z for p in plans):.3f} m")

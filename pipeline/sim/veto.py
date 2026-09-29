"""High-confidence disaster veto: the policy acts, the world model only refuses.

Built while selection looked unsalvageable. Across two scorers, three candidate
sources, two optimisers and four horizons, executing the policy's own chunk matched or
beat ranking candidates.

**Most of that evidence has since been withdrawn.** All but one of those arms ranked
perturbation candidates, and Oracle@k at episode depth showed those to be
outcome-equivalent (headroom +0.0) — an invalid ranking test, on which every ranker
scores alike. What survives is narrower and still motivates this module: the affordance
candidates that *do* differ get exploited by a greedy objective (0.0% success under
distance-optimal picking), because the dynamics inverts off the policy manifold. The
answer to that is `domain.py`, which gates *where* ranking is allowed; refusal is the
complementary job, and the two are independent.

What the world model is demonstrably good at is a different job. The value head
predicts discounted return at held-out MSE 0.025 against 0.080 for the cube-to-goal
distance term — which on plan-randomised data is indistinguishable from predicting the
mean. So the model can say *whether things are going badly*. It cannot say *which of
eight near-identical actions is best*. This module asks it only the first question.

Two modes, and the one that was designed is not the one that was measured to work.

``mode="drop"`` is the design. Two conditions must BOTH hold before an action is
refused:

  * **magnitude** — the predicted return must drop by more than `drop`, i.e. the action
    is predicted to destroy progress rather than merely fail to add any;
  * **confidence** — the dynamics ensemble must agree, with rollout spread under
    `max_spread`. A prediction the ensemble disagrees about is not evidence of
    disaster, it is evidence of ignorance, and refusing on it converts model
    uncertainty into robot paralysis.

``mode="spread"`` is the default, because `calibrate_veto.py` measured the design to be
backwards. Across 492 rewound decisions, every signal built from the value head is at
or below chance at predicting a genuinely doomed action — predicted drop 0.457,
instability 0.396, predicted return at horizon 0.353 — and the **ensemble's
disagreement with itself is the only one above it, at 0.587**. So the quantity the
design treats as a disqualifier is the detector, and the quantity it treats as the
detector is noise.

**The obvious explanation for that is wrong, and it was checked rather than assumed.**
This docstring previously said the latent "carries the cube, the bowl and the ball, and
no gripper", so the model could not represent the mechanism it was being asked to
predict. `probe_latent.py` tested it: a **linear** probe on z0 recovers the finger
opening at R2 1.000, the tip-to-cube offset at 0.999, the pad-to-cube clearance at
0.973, and whether the cube is currently held at 0.903. The gripper is in the latent,
linearly, and the grasp geometry with it — the proprioceptive half of the state vector
carries both finger joints and the tip, and the object slots carry the cube.

So the missing piece is not the *representation*. It is one of the two things built on
top of it: the dynamics cannot predict how that geometry evolves under an action (it
never sees a slip coming), or the value head, fitted on a sparse success signal, cannot
map the geometry to an outcome. Which of those it is has not been isolated and is not
claimed here. What can be said is that "add the gripper to the latent" is **not** the
fix, and would have been a month of work aimed at a hypothesis one probe refuted.

The edge is real and small: precision peaks at 0.404 against a base rate of 0.285.
`SPREAD_FIRE` is the threshold read off the sweep, and it is chosen on **net episode
outcomes flipped** rather than on F1, because the two disagree — the highest-F1 operating
point nets slightly negative. See RESULTS.md for the full sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pipeline.world_model.latent import SceneLatent


@dataclass
class VetoDecision:
    allowed: bool
    predicted_drop: float
    spread: float
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


#: Ensemble spread above which `mode="spread"` refuses. Read off the calibration
#: sweep at the operating point with the best *net* effect on episode outcomes
#: (+2.6 flips per 100 fires at precision 0.376), not the best F1.
SPREAD_FIRE = 0.1767


@dataclass
class VetoGate:
    """Refuses an action chunk the model is confident will make things much worse."""

    #: Which measured quantity fires the gate. ``"spread"`` is the calibrated default;
    #: ``"drop"`` is the original design, kept because it is the hypothesis the
    #: calibration refuted and rerunning it is how that stays checkable.
    mode: str = "spread"
    #: Predicted return loss, in the value head's own units, that counts as a disaster.
    drop: float = 0.15
    #: Maximum ensemble rollout spread (metres) at which the prediction is trusted.
    #: Only consulted in ``mode="drop"``.
    max_spread: float = 0.05
    #: Ensemble spread above which ``mode="spread"`` refuses.
    spread_fire: float = SPREAD_FIRE
    #: Vetoes fired, for reporting. A gate that never fires and a gate that always
    #: fires are both useless, and only the count distinguishes them from a working one.
    fired: int = 0
    checked: int = 0
    history: list[VetoDecision] = field(default_factory=list)

    @torch.no_grad()
    def __call__(self, dynamics, value, z0: SceneLatent,
                 chunk: torch.Tensor) -> VetoDecision:
        """`chunk` is (N, A), normalised the way the dynamics was trained."""
        self.checked += 1
        states, unc = dynamics.rollout(z0, chunk.unsqueeze(0))
        spread = float(unc.max()) if unc.numel() else 0.0
        # `mode="spread"` needs no value head at all — which is the point, since the
        # value head is the part the calibration found to be uninformative here. The
        # drop is still recorded when one is available so a spread-mode run stays
        # re-scorable against the refuted rule.
        if value is None:
            drop = float("nan")
        else:
            drop = float(value(z0)) - float(value(states[-1]))

        if self.mode == "spread":
            # Fires on ignorance, not on predicted magnitude. `drop` is still recorded
            # so a run in this mode can be re-scored against the other rule offline.
            if spread <= self.spread_fire:
                d = VetoDecision(True, drop, spread,
                                 f"ensemble spread {spread:.3f} within "
                                 f"{self.spread_fire:.3f}")
            else:
                d = VetoDecision(False, drop, spread,
                                 f"VETO: ensemble spread {spread:.3f} exceeds "
                                 f"{self.spread_fire:.3f} — the model does not model "
                                 "this state")
                self.fired += 1
            self.history.append(d)
            return d

        if drop <= self.drop:
            d = VetoDecision(True, drop, spread,
                             f"predicted drop {drop:+.3f} within {self.drop:.3f}")
        elif spread > self.max_spread:
            # The model thinks this is bad and does not agree with itself. Allowing it
            # is the honest call: refusing here would let an unreliable model veto
            # anything it happens to be uncertain about, which is most of what a robot
            # does near contact.
            d = VetoDecision(True, drop, spread,
                             f"drop {drop:+.3f} exceeds {self.drop:.3f} but ensemble "
                             f"spread {spread:.3f} > {self.max_spread:.3f} — not "
                             "confident enough to refuse")
        else:
            d = VetoDecision(False, drop, spread,
                             f"VETO: predicted return drops {drop:+.3f} with ensemble "
                             f"spread {spread:.3f}")
            self.fired += 1
        self.history.append(d)
        return d

    def summary(self) -> dict:
        return {"checked": self.checked, "fired": self.fired,
                "rate": self.fired / max(1, self.checked),
                "mean_drop": float(np.mean([h.predicted_drop for h in self.history]))
                if self.history else None}


def hold_chunk(horizon: int, grip: float) -> np.ndarray:
    """What to do instead of the refused action: stop moving, keep the grip.

    Holding rather than retreating or re-planning. A veto is a statement that the model
    does not want *this* action executed, not that it knows a better one — it was
    explicitly not trained to rank — so substituting an alternative would smuggle
    selection back in through the fallback. Holding also keeps the substitution
    measurable: whatever changes in the outcome is attributable to not acting.
    """
    out = np.zeros((horizon, 4), np.float32)
    out[:, 3] = grip
    return out

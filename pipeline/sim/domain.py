"""Applicable-domain gate: the scorer may rank only where it was fitted.

Built because the ranking failure was localised rather than merely observed. Scored
along the policy's own trajectory the model orders affordance candidates at Kendall
tau **+0.291**; scored along the trajectory its own choices produce, the same model on
the same candidates orders them at **-0.157**. It does not degrade off the policy
manifold, it **inverts** — and acting on an inverted ranking is what carries the
episode further off the manifold, which is the compounding loop behind reachability
falling 91.7% to 40% under `--pick best`.

A model that is right in one region and wrong outside it does not need to be trusted
less everywhere. It needs a boundary. This module is that boundary, and the policy is
what lies on the safe side of it: **the policy's own chunk always executes unless the
scorer is inside its applicable domain**, which keeps the policy-first baseline intact
by construction rather than by tuning.

**The load-bearing signal is the action, and it is per-candidate.** Three were
calibrated against per-decision Kendall tau by `calibrate_domain.py`, and only one
survived:

  * **action deviation** — how far a candidate's chunk is from the policy's own, in
    standardised action units. Filtering on it is what works. Re-ranking only the
    candidates that survive the filter takes the ordering from **tau -0.134 (inverted)
    to +0.062 (correct)** at a threshold of 0.68, keeping half the candidates and
    losing none of the 144 decisions. It also predicts the scorer's own error per
    candidate (r = +0.417, binned error 0.004 -> 0.009), which is the mechanism: the
    model is measurably less accurate on actions further from the ones it was fitted on.
  * **state energy** — DOWNGRADED TO AN UNARMED BACKSTOP, because it was measured not
    to fire. Across 144 decisions in the regime this gate exists for, it ranged
    0.38-1.43 against an in-distribution expectation of ~1, and no threshold on it
    recovered a positive tau (best -0.095 against -0.133 ungated). The states the
    scorer steers into are **ordinary scenes**; what leaves the manifold is the action.
    That is a real finding rather than a tuning failure, and it is why the
    counterfactual collector branches off-policy *actions* from on-policy *states*.
    The threshold defaults to infinity — the caller must arm it — so it can still catch
    a genuinely novel scene without silently doing nothing in the meantime.
  * **ensemble spread** — likewise unarmed. It is the one signal that predicts a doomed
    action (AUC 0.587 against 0.457 for the value head's predicted drop), but predicting
    *disaster* and predicting *a bad ranking* are different jobs, and on the ranking job
    no threshold on it helped (best -0.106 against -0.133).

The gate is deliberately asymmetric. Excluding a good candidate costs at most the gain
that candidate would have bought; including a bad one costs an inverted ranking acted
on, which is the failure this exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.world_model.latent import SceneLatent


@dataclass(frozen=True)
class DomainThresholds:
    """Where the applicable domain ends. Read off a calibration, never guessed.

    Only `action` is armed. The other two are retained as backstops a caller can arm
    deliberately, and default to infinity because `calibrate_domain.py` measured them
    not to separate good rankings from bad ones — shipping a plausible-looking number
    for a signal that does not work is how a gate comes to look calibrated while doing
    nothing.
    """

    #: Per-candidate. 0.68 is the knee of the measured curve: re-ranking the survivors
    #: gives tau +0.062 against -0.134 unfiltered, and it is the loosest threshold that
    #: still turns the ordering positive.
    action: float = 0.68
    #: Unarmed. Measured range 0.38-1.43 with no useful threshold; see the module
    #: docstring.
    state: float = float("inf")
    #: Unarmed. Predicts disaster, not ranking quality; see the module docstring.
    spread: float = float("inf")
    #: How many candidates must survive before ranking is allowed at all. Two, because
    #: ranking one candidate is not ranking.
    min_candidates: int = 2


@dataclass
class DomainVerdict:
    in_domain: np.ndarray          # (K,) bool, per candidate
    state_energy: float
    spread: float
    abstain: bool
    reason: str

    def __bool__(self) -> bool:
        return not self.abstain


def state_energy(z: SceneLatent) -> float:
    """Mean squared standardised coordinate over the real slots and proprioception.

    The mask matters: padded slots are exact zeros, and averaging them in would pull
    the statistic toward zero in proportion to how many objects were *not* detected,
    which is the opposite of the intended reading.
    """
    slots = z.slots.detach().cpu().numpy()[0]
    robot = z.robot.detach().cpu().numpy()[0]
    mask = (z.mask.detach().cpu().numpy()[0] if z.mask is not None
            else np.ones(slots.shape[0], bool))
    vals = np.concatenate([slots[mask].ravel(), robot.ravel()]) if mask.any() \
        else robot.ravel()
    return float(np.mean(vals ** 2))


def action_deviation(candidates: np.ndarray, reference: np.ndarray,
                     action_std: np.ndarray) -> np.ndarray:
    """(K,) RMS distance from the policy's own chunk, in standardised action units.

    Standardised, because the chunk mixes metres of Cartesian delta with a dimensionless
    grip command; an unstandardised norm would be almost entirely the grip channel.
    """
    c = np.asarray(candidates, np.float64)
    r = np.asarray(reference, np.float64)
    s = np.asarray(action_std, np.float64)
    s = np.where(s > 1e-8, s, 1.0)
    d = (c - r[None]) / s
    return np.sqrt(np.mean(d.reshape(len(c), -1) ** 2, axis=1))


@dataclass
class DomainGate:
    """Decides whether the scorer is allowed to rank, and which candidates it may rank.

    Returns a verdict rather than a filtered list so the caller can report *why* it
    abstained. A gate that silently rolls back is indistinguishable from a scorer that
    happens to keep choosing candidate 0, and those two have very different fixes.
    """

    thresholds: DomainThresholds = DomainThresholds()
    #: Counters, for reporting. A gate that never abstains and one that always abstains
    #: are both useless and only these distinguish them from a working one.
    decisions: int = 0
    abstained: int = 0
    dropped: int = 0
    offered: int = 0

    def __call__(self, z0: SceneLatent, candidates: np.ndarray,
                 reference: np.ndarray, action_std: np.ndarray,
                 spread: float | None = None) -> DomainVerdict:
        t = self.thresholds
        self.decisions += 1
        energy = state_energy(z0)
        dev = action_deviation(candidates, reference, action_std)
        ok = dev <= t.action
        # The policy's own chunk is in-domain by definition — its deviation from itself
        # is zero — and forcing it in keeps the rollback target inside the ranked set
        # rather than making abstention the only way to reach it.
        if len(ok):
            ok[int(np.argmin(dev))] = True
        self.offered += len(ok)
        self.dropped += int((~ok).sum())

        sp = float(spread) if spread is not None else 0.0
        if energy > t.state:
            v = DomainVerdict(ok, energy, sp, True,
                              f"state energy {energy:.2f} > {t.state:.2f} — this scene "
                              "is outside the region the dynamics was fitted on")
        elif spread is not None and sp > t.spread:
            v = DomainVerdict(ok, energy, sp, True,
                              f"ensemble spread {sp:.3f} > {t.spread:.3f} — the model "
                              "does not agree with itself about this rollout")
        elif int(ok.sum()) < t.min_candidates:
            v = DomainVerdict(ok, energy, sp, True,
                              f"only {int(ok.sum())} of {len(ok)} candidates are on the "
                              "policy manifold — nothing left to rank")
        else:
            v = DomainVerdict(ok, energy, sp, False,
                              f"{int(ok.sum())}/{len(ok)} candidates in domain, state "
                              f"energy {energy:.2f}")
        self.abstained += int(v.abstain)
        return v

    def summary(self) -> dict:
        return {"decisions": self.decisions, "abstained": self.abstained,
                "abstain_rate": self.abstained / max(1, self.decisions),
                "candidates_offered": self.offered, "candidates_dropped": self.dropped,
                "drop_rate": self.dropped / max(1, self.offered)}

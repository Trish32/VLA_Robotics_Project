"""The applicable-domain gate.

No MuJoCo here on purpose: the gate is arithmetic over a latent and a candidate set,
and it should be testable without a physics engine so a failure in it is never confused
with a failure in the arena.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.sim.domain import (DomainGate, DomainThresholds, action_deviation,
                                 state_energy)
from pipeline.world_model.latent import SceneLatent


def latent(slot_value: float = 0.0, robot_value: float = 0.0, n_real: int = 2,
           n_slots: int = 3):
    mask = torch.zeros(1, n_slots, dtype=torch.bool)
    mask[0, :n_real] = True
    slots = torch.zeros(1, n_slots, 12)
    slots[0, :n_real] = slot_value          # padding stays exactly zero, as in the loop
    return SceneLatent(slots, torch.full((1, 32), robot_value), mask)


def chunks(k: int = 4, horizon: int = 8, offsets=None):
    out = np.zeros((k, horizon, 4), np.float32)
    for i, off in enumerate(offsets or []):
        out[i + 1] += off
    return out


def test_padded_slots_do_not_dilute_the_state_energy():
    """The bug this guards: averaging exact-zero padding in makes the statistic read
    LOW exactly when perception found fewer objects, which is backwards."""
    # Same two real objects; the only difference is how many blank slots trail them.
    tight = state_energy(latent(slot_value=2.0, n_real=2, n_slots=2))
    padded = state_energy(latent(slot_value=2.0, n_real=2, n_slots=6))
    assert tight == pytest.approx(padded, rel=1e-6)
    # 24 slot entries at 2.0 and 32 robot entries at 0.0.
    assert tight == pytest.approx(24 * 4.0 / 56, rel=1e-6)


def test_action_deviation_is_standardised_not_raw():
    """A chunk mixes metres of Cartesian delta with a dimensionless grip command, so a
    raw norm would be almost entirely whichever channel happens to have the larger
    numbers."""
    c = np.zeros((2, 4, 4), np.float32)
    c[1, :, 0] = 0.01            # 1 cm of x, in a channel whose std is 0.01
    c[1, :, 3] = 0.5             # half a grip unit, in a channel whose std is 0.5
    dev = action_deviation(c, c[0], np.array([0.01, 0.01, 0.01, 0.5]))
    # Both channels contribute one standardised unit over a quarter of the entries.
    assert dev[0] == 0.0
    assert dev[1] == pytest.approx(np.sqrt(2 / 4), rel=1e-6)


def test_a_zero_std_channel_does_not_divide_by_zero():
    c = np.zeros((2, 4, 4), np.float32)
    c[1, :, 2] = 1.0
    dev = action_deviation(c, c[0], np.array([1.0, 1.0, 0.0, 1.0]))
    assert np.isfinite(dev).all()


def test_the_policys_own_chunk_is_always_in_domain():
    """Abstaining must mean 'do what the policy wanted', so the rollback target has to
    survive its own filter even when the threshold is set below every deviation."""
    gate = DomainGate(DomainThresholds(action=0.0, min_candidates=1))
    c = chunks(3, offsets=[5.0, 7.0])
    v = gate(latent(), c, c[0], np.ones(4))
    assert v.in_domain[0] and not v.in_domain[1:].any()
    assert not v.abstain


def test_abstains_when_too_few_candidates_survive():
    gate = DomainGate(DomainThresholds(action=0.1, min_candidates=2))
    c = chunks(3, offsets=[5.0, 7.0])
    v = gate(latent(), c, c[0], np.ones(4))
    assert v.abstain and "nothing left to rank" in v.reason
    assert bool(v) is False


def test_an_out_of_distribution_scene_abstains_before_any_candidate_matters():
    """State energy is the one signal available before candidates exist, and it must
    short-circuit: a scene the dynamics never saw makes every ranking meaningless
    regardless of how on-manifold the actions are."""
    gate = DomainGate(DomainThresholds(state=1.5))
    c = chunks(4)                                   # all identical, deviation 0
    v = gate(latent(slot_value=4.0), c, c[0], np.ones(4))
    assert v.abstain and "state energy" in v.reason


def test_spread_is_only_consulted_when_it_is_supplied():
    """Callers that have not paid for a rollout pass None, and None must not read as 0
    — silently treating 'unknown' as 'perfectly confident' is how a gate stops gating."""
    gate = DomainGate(DomainThresholds(spread=0.01))
    c = chunks(4)
    assert not gate(latent(), c, c[0], np.ones(4)).abstain
    assert gate(latent(), c, c[0], np.ones(4), spread=0.5).abstain


def test_counters_separate_abstention_from_candidate_filtering():
    """A gate that abstains often and one that quietly drops candidates have different
    fixes, so the summary has to tell them apart."""
    gate = DomainGate(DomainThresholds(action=1.0, min_candidates=2))
    c = chunks(4, offsets=[0.1, 5.0, 5.0])
    gate(latent(), c, c[0], np.ones(4))
    s = gate.summary()
    assert s["decisions"] == 1 and s["abstained"] == 0
    assert s["candidates_dropped"] == 2 and s["candidates_offered"] == 4

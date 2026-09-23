"""Action-conditioned world model: latent, dynamics, scorer, planner.

The properties here are the ones the design rests on, so a change that quietly breaks
one of them should fail a test rather than show up as a worse rollout three stages later:

  * the untrained model is EXACTLY the baseline it claims to start from;
  * padded slots never influence a prediction or a score;
  * ensemble uncertainty grows with horizon, since that is the only thing stopping the
    planner trusting step 20 as much as step 1;
  * the planner refuses a predicted collision rather than trading it off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.world_model.dynamics import DynamicsEnsemble, LatentDynamics
from pipeline.world_model.latent import SceneLatent
from pipeline.world_model.planner import ActionPlanner, perturbed_candidates
from pipeline.world_model.scorer import (ScoreWeights, TrajectoryScorer,
                                         pairwise_overlap_volume, unsupported_drop)

torch.manual_seed(0)


def latent(b=2, m=3, d=8, r=6):
    return SceneLatent(torch.randn(b, m, d), torch.randn(b, r))


# ------------------------------------------------------------------ SceneLatent

def test_shape_errors_name_the_offending_tensor():
    with pytest.raises(ValueError, match="slots must be"):
        SceneLatent(torch.randn(2, 8), torch.randn(2, 6))
    with pytest.raises(ValueError, match="robot must be"):
        SceneLatent(torch.randn(2, 3, 8), torch.randn(2, 3, 6))
    with pytest.raises(ValueError, match="batch mismatch"):
        SceneLatent(torch.randn(2, 3, 8), torch.randn(5, 6))


def test_extent_is_non_negative_even_if_predicted_negative():
    """A predicted negative side length would silently invert every overlap test."""
    z = SceneLatent(torch.tensor([[[0., 0, 0, -2, -2, -2]]]), torch.zeros(1, 2))
    assert torch.all(z.extent == 2.0)
    lo, hi = z.aabb()
    assert torch.all(hi >= lo)


def test_expand_batch_is_interleaved_so_candidate_i_is_row_b_times_k_plus_i():
    z = SceneLatent(torch.arange(6.).reshape(2, 1, 3), torch.zeros(2, 2))
    e = z.expand_batch(3)
    assert e.batch == 6
    assert torch.equal(e.slots[0], e.slots[2])       # scene 0, candidates 0 and 2
    assert not torch.equal(e.slots[0], e.slots[3])   # scene 1 starts at row 3


# -------------------------------------------------------------------- dynamics

def test_untrained_model_is_exactly_the_identity():
    z = latent()
    out, gate = LatentDynamics(8, 6, 4, hidden=16, layers=1)(z, torch.randn(2, 4))
    assert torch.allclose(out.slots, z.slots, atol=1e-7)
    assert torch.allclose(out.robot, z.robot, atol=1e-7)
    assert float(gate.mean()) == pytest.approx(0.04743, abs=1e-3)   # sigmoid(-3)


def test_untrained_model_is_exactly_constant_velocity_when_integrating():
    """The measured reason this option exists: identity scores 0.0554 held-out on
    gr1.PickNPlace and constant velocity scores 0.0262."""
    pos, vel = torch.tensor([[1., 2, 3]]), torch.tensor([[0.1, 0.2, 0.3]])
    z = SceneLatent(torch.zeros(1, 1, 2), torch.cat([pos, vel], -1))
    out, _ = LatentDynamics(2, 6, 4, hidden=16, layers=1,
                            integrate_velocity=True)(z, torch.randn(1, 4))
    assert torch.allclose(out.robot[:, :3], pos + vel, atol=1e-6)
    assert torch.allclose(out.robot[:, 3:], vel, atol=1e-6)


def test_integrate_velocity_rejects_odd_dims():
    with pytest.raises(ValueError, match="even dims"):
        LatentDynamics(8, 7, 4, integrate_velocity=True)


def test_action_dimension_is_checked():
    with pytest.raises(ValueError, match="action has"):
        LatentDynamics(8, 6, 4, hidden=16, layers=1)(latent(), torch.randn(2, 9))


def test_padded_slots_are_returned_untouched():
    z = SceneLatent(torch.randn(1, 4, 8), torch.randn(1, 6),
                    torch.tensor([[True, True, False, False]]))
    f = LatentDynamics(8, 6, 4, hidden=16, layers=1)
    with torch.no_grad():                       # move the heads off their zero init
        f.slot_out.weight.normal_(0, 0.5)
        f.robot_out.weight.normal_(0, 0.5)
    out, _ = f(z, torch.randn(1, 4))
    assert torch.allclose(out.slots[0, 2:], z.slots[0, 2:], atol=1e-7)
    assert not torch.allclose(out.slots[0, :2], z.slots[0, :2])


def test_padding_does_not_change_the_prediction_for_real_slots():
    """The same two real objects must predict the same whether or not blanks trail
    them, or a prediction depends on how the batch was assembled."""
    f = LatentDynamics(8, 6, 4, hidden=16, layers=1)
    with torch.no_grad():
        f.slot_out.weight.normal_(0, 0.5)
    real = torch.randn(1, 2, 8)
    a = torch.randn(1, 4)
    tight = SceneLatent(real, torch.zeros(1, 6))
    padded = SceneLatent(torch.cat([real, torch.randn(1, 3, 8)], 1), torch.zeros(1, 6),
                         torch.tensor([[True, True, False, False, False]]))
    assert torch.allclose(f(tight, a)[0].slots, f(padded, a)[0].slots[:, :2], atol=1e-5)


def test_gradients_reach_every_parameter():
    f = LatentDynamics(8, 6, 4, hidden=16, layers=1)
    out, _ = f(latent(), torch.randn(2, 4))
    (out.slots.sum() + out.robot.sum()).backward()
    missing = [n for n, p in f.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, f"no finite gradient for {missing}"


def test_ensemble_members_are_not_identical():
    e = DynamicsEnsemble(8, 6, 4, n_members=3, hidden=16, layers=1)
    a = torch.cat([p.flatten() for p in e.members[0].parameters()])
    b = torch.cat([p.flatten() for p in e.members[1].parameters()])
    assert (a - b).abs().max() > 1e-6


def test_ensemble_needs_a_member():
    with pytest.raises(ValueError, match="at least one"):
        DynamicsEnsemble(8, 6, 4, n_members=0)


def test_rollout_uncertainty_grows_with_horizon():
    """The planner discounts long rollouts using exactly this signal."""
    e = DynamicsEnsemble(8, 6, 4, n_members=4, hidden=16, layers=1)
    with torch.no_grad():                        # a trained model has non-zero heads
        for m in e.members:
            m.slot_out.weight.normal_(0, 0.3)
            m.slot_gate.bias.fill_(0.0)
    _, unc = e.rollout(latent(b=1), torch.randn(1, 8, 4))
    assert unc.shape == (1, 8)
    assert unc[0, -1] > unc[0, 0]


def test_single_member_reports_no_uncertainty_rather_than_nan():
    e = DynamicsEnsemble(8, 6, 4, n_members=1, hidden=16, layers=1)
    _, unc = e.rollout(latent(b=1), torch.randn(1, 3, 4))
    assert torch.all(unc == 0)


def test_rollout_rejects_unbatched_actions():
    e = DynamicsEnsemble(8, 6, 4, n_members=2, hidden=16, layers=1)
    with pytest.raises(ValueError, match=r"\(B, N, A\)"):
        e.rollout(latent(b=1), torch.randn(3, 4))


# ---------------------------------------------------------------------- scorer

def test_overlap_volume_matches_a_hand_computed_value():
    z = SceneLatent(torch.tensor([[[0., 0, 0, 1, 1, 1], [0.5, 0, 0, 1, 1, 1]]]),
                    torch.zeros(1, 2))
    assert float(pairwise_overlap_volume(z)) == pytest.approx(0.5, abs=1e-6)


def test_disjoint_objects_do_not_collide():
    z = SceneLatent(torch.tensor([[[0., 0, 0, 1, 1, 1], [5., 0, 0, 1, 1, 1]]]),
                    torch.zeros(1, 2))
    assert float(pairwise_overlap_volume(z)) == pytest.approx(0.0)


def test_padded_slots_never_count_as_collisions():
    """Padding sits at the origin; scored naively, every blank collides with every
    other blank and the cost is dominated by objects that do not exist."""
    z = SceneLatent(torch.zeros(1, 4, 6), torch.zeros(1, 2),
                    torch.tensor([[True, False, False, False]]))
    z.slots[0, :, 3:] = 1.0
    assert float(pairwise_overlap_volume(z)) == pytest.approx(0.0)


def test_object_on_the_floor_is_supported_and_a_floating_one_is_not():
    on_floor = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1]]]), torch.zeros(1, 2))
    floating = SceneLatent(torch.tensor([[[0., 0, 2.5, 1, 1, 1]]]), torch.zeros(1, 2))
    assert float(unsupported_drop(on_floor)) == pytest.approx(0.0, abs=1e-5)
    assert float(unsupported_drop(floating)) == pytest.approx(1.95, abs=0.01)


def test_an_object_stacked_on_another_is_supported():
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1], [0., 0, 1.5, 1, 1, 1]]]),
                    torch.zeros(1, 2))
    assert float(unsupported_drop(z)) == pytest.approx(0.0, abs=1e-5)


def test_support_requires_footprint_overlap_not_just_height():
    """An object at the right height but off to the side is not resting on anything."""
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1], [9., 0, 1.5, 1, 1, 1]]]),
                    torch.zeros(1, 2))
    assert float(unsupported_drop(z)) > 0.5


def test_scorer_returns_a_breakdown_that_sums_to_the_total():
    z = SceneLatent(torch.randn(3, 2, 6), torch.randn(3, 4))
    sc = TrajectoryScorer(ScoreWeights())
    b = sc([z, z], torch.zeros(3, 2), torch.zeros(3, 2, 5))
    parts = b.collision + b.stability + b.progress + b.risk + b.effort
    assert torch.allclose(b.total, parts, atol=1e-6)
    assert 0 <= b.best() < 3


def test_risk_weight_penalises_an_uncertain_rollout():
    z = SceneLatent(torch.zeros(2, 1, 6), torch.zeros(2, 4))
    z.slots[..., 3:] = 1.0
    unc = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    b = TrajectoryScorer(ScoreWeights())([z, z], unc, torch.zeros(2, 2, 3))
    assert b.best() == 0, "the certain rollout must win when all else is equal"


def test_scorer_rejects_an_empty_rollout():
    with pytest.raises(ValueError, match="no predicted states"):
        TrajectoryScorer()([], torch.zeros(1, 1), torch.zeros(1, 1, 3))


# --------------------------------------------------------------------- planner

def static_ensemble(slot_dim=6, robot_dim=4, action_dim=3):
    """An ensemble frozen at its identity init: predictions equal the input, so a test
    can control the predicted future exactly by choosing the input."""
    return DynamicsEnsemble(slot_dim, robot_dim, action_dim, n_members=2,
                            hidden=16, layers=1)


def test_planner_requires_a_single_scene_and_batched_candidates():
    p = ActionPlanner(static_ensemble())
    with pytest.raises(ValueError, match="one scene"):
        p.plan(SceneLatent(torch.zeros(2, 1, 6), torch.zeros(2, 4)),
               torch.zeros(3, 2, 3))
    with pytest.raises(ValueError, match=r"\(K, N, A\)"):
        p.plan(SceneLatent(torch.zeros(1, 1, 6), torch.zeros(1, 4)), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="no candidate"):
        p.plan(SceneLatent(torch.zeros(1, 1, 6), torch.zeros(1, 4)),
               torch.zeros(0, 2, 3))


def test_planner_prefers_the_candidate_that_moves_the_target_toward_the_goal():
    """With a frozen dynamics, the only thing separating candidates is effort — so a
    goal term must be what decides, and the cheapest action must win the tie."""
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1]]]), torch.zeros(1, 4))
    cands = torch.stack([torch.zeros(3, 3), torch.full((3, 3), 5.0)])
    r = ActionPlanner(static_ensemble()).plan(z, cands, target=0,
                                              goal_xyz=torch.tensor([0., 0, 0.5]))
    assert r.index == 0
    assert r.scores.effort[1] > r.scores.effort[0]


class SlidingDynamics:
    """Test double: slot 0 slides along +x in proportion to the action.

    The planner only ever calls `.rollout`, so a stand-in makes "what the action caused"
    exactly controllable — which a learned model, frozen at its identity init, cannot be.
    """

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def rollout(self, z, actions):
        states, x = [], z.slots.clone()
        for t in range(actions.shape[1]):
            x = x.clone()
            x[:, 0, 0] = x[:, 0, 0] + self.gain * actions[:, t, 0]
            states.append(SceneLatent(x.clone(), z.robot, z.mask))
        return states, torch.zeros(z.batch, actions.shape[1])


def test_planner_vetoes_a_collision_the_action_causes():
    """A veto is not a large weight: no gain elsewhere should buy a collision."""
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1], [3., 0, 0.5, 1, 1, 1]]]),
                    torch.zeros(1, 4))
    # Candidate 0 stays put; candidate 1 drives slot 0 into slot 1.
    cands = torch.zeros(2, 3, 3)
    cands[1, :, 0] = 1.0
    r = ActionPlanner(SlidingDynamics()).plan(z, cands)
    assert r.rejected == [1]
    assert r.index == 0


def test_pre_existing_overlap_is_not_charged_to_any_candidate():
    """Measured on the real scene: every Mask3D proposal already overlaps its
    neighbours, which gave all eight candidates an identical 0.2400 collision cost and
    vetoed every one of them. Overlap common to all choices is a fact about the
    segmentation, not about an action."""
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1], [0.2, 0, 0.5, 1, 1, 1]]]),
                    torch.zeros(1, 4))
    r = ActionPlanner(SlidingDynamics(gain=0.0)).plan(z, torch.zeros(3, 2, 3))
    assert r.rejected == []
    assert torch.all(r.scores.collision == 0)


def test_planner_returns_the_losers_so_a_choice_can_be_audited():
    z = SceneLatent(torch.tensor([[[0., 0, 0.5, 1, 1, 1]]]), torch.zeros(1, 4))
    r = ActionPlanner(static_ensemble()).plan(z, perturbed_candidates(
        torch.zeros(3, 3), k=5, scale=0.3))
    assert r.scores.total.numel() == 5
    assert r.margin >= 0
    assert "chose candidate" in r.explain()
    assert len(r.states) == 3 and r.states[0].batch == 1


def test_perturbed_candidates_keep_the_original_first():
    """The planner must never have to discard the policy's own answer to have a set to
    choose from."""
    base = torch.randn(4, 3)
    c = perturbed_candidates(base, k=6, scale=0.1)
    assert c.shape == (6, 4, 3)
    assert torch.allclose(c[0], base)
    assert not torch.allclose(c[1], base)


def test_perturbed_candidates_are_deterministic_for_a_seed():
    base = torch.randn(3, 2)
    assert torch.allclose(perturbed_candidates(base, 4, seed=7),
                          perturbed_candidates(base, 4, seed=7))


def test_perturbed_candidates_reject_a_batched_base():
    with pytest.raises(ValueError, match=r"\(N, A\)"):
        perturbed_candidates(torch.zeros(2, 3, 4), k=3)

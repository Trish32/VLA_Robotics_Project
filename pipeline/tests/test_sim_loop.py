"""The wiring between perception, the dynamics and the arm.

Every assertion here is about a seam that failed silently at least once while this was
being built: a lookahead that advanced the policy it was only supposed to interrogate,
a latent whose slots were in detection order, and a rollout scored in standardised
units against thresholds in cubic metres.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from pipeline.sim.env import PickPlaceEnv  # noqa: E402
from pipeline.sim.loop import (SLOT_ORDER, ClosedLoop, LoopConfig,  # noqa: E402
                               Normalisers, scripted_chunk)
from pipeline.sim.perception import OraclePerception, Perceived  # noqa: E402
from pipeline.sim.policy import ScriptedPickPlace  # noqa: E402
from pipeline.world_model.dynamics import DynamicsEnsemble  # noqa: E402

GEOM, NSLOT, STATE, ACT = 6, 3, 16, 4


def make_loop(env, norms=None):
    dyn = DynamicsEnsemble(2 * GEOM, 2 * STATE, ACT, n_members=2, hidden=16, layers=1)
    return ClosedLoop(env, OraclePerception(), dyn,
                      norms or Normalisers.identity(GEOM, STATE, ACT),
                      cfg=LoopConfig(horizon=4, execute=2, candidates=4),
                      n_slots=NSLOT, slot_geom=GEOM)


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(seed=1, randomise=False, cameras=(), width=160, height=120,
                     max_steps=40)
    yield e
    e.close()


def test_lookahead_does_not_advance_the_policy(env):
    """The bug this catches ran the waypoint machine at H times its intended rate."""
    obs = env.reset()
    pol = ScriptedPickPlace()
    before = (pol.stage, pol.hold, pol.grip)
    chunk = scripted_chunk(pol, obs.tip, obs.objects, 8)
    assert (pol.stage, pol.hold, pol.grip) == before
    assert chunk.shape == (8, 4)


def test_lookahead_does_not_consume_the_policys_noise(env):
    """The RNG is state too. A lookahead that draws from it makes the number of
    candidates change the noise the policy actually executes, so two arms that differ
    only in candidate count silently run different policies."""
    obs = env.reset()
    pol = ScriptedPickPlace(noise=0.004, rng=np.random.default_rng(3))
    before = pol.rng.bit_generator.state["state"]["state"]
    for _ in range(5):
        scripted_chunk(pol, obs.tip, obs.objects, 8)
    assert pol.rng.bit_generator.state["state"]["state"] == before
    a = pol.act(obs.tip, obs.objects)
    assert pol.rng.bit_generator.state["state"]["state"] != before, (
        "acting for real must still advance it")
    assert a.shape == (4,)


def test_lookahead_actually_moves_toward_the_cube(env):
    obs = env.reset()
    chunk = scripted_chunk(ScriptedPickPlace(), obs.tip, obs.objects, 12)
    cube = obs.objects["a small dark cube"]
    assert (np.linalg.norm(obs.tip + chunk[:, :3].sum(0) - cube)
            < np.linalg.norm(obs.tip - cube))


def test_slots_keep_a_fixed_order_regardless_of_detection_order(env):
    """A slot index is the planner's `target`; reordering re-aims the goal term."""
    loop = make_loop(env)
    obs = env.reset()
    seen = OraclePerception()(env.render("front"))
    shuffled = {k: seen[k] for k in reversed(list(seen))}
    a, ok_a = loop._slots(seen)
    b, ok_b = loop._slots(shuffled)
    assert np.allclose(a, b) and (ok_a == ok_b).all()
    assert np.allclose(a[0, :3], seen[SLOT_ORDER[0]].centre)


def test_a_missing_object_is_marked_invalid_not_placed_at_the_origin(env):
    loop = make_loop(env)
    env.reset()
    boxes, ok = loop._slots({})
    assert not ok.any()
    assert np.allclose(boxes, 0.0), "unset slots are zero, and the mask says so"


def test_latent_carries_position_then_velocity(env):
    loop = make_loop(env)
    obs = env.reset()
    boxes, ok = loop._slots(OraclePerception()(env.render("front")))
    prev = boxes - 0.01
    z = loop._latent(boxes, prev, ok, obs.state, obs.state - 0.5)
    assert z.slots.shape == (1, NSLOT, 2 * GEOM)
    assert z.robot.shape == (1, 2 * STATE)
    assert torch.allclose(z.slots[0, :, GEOM:], torch.full((NSLOT, GEOM), 0.01),
                          atol=1e-5)


def test_normalisation_round_trips_through_the_latent(env):
    """With a real normaliser, metres in must be the metres that come back out."""
    rng = np.random.default_rng(0)
    norms = Normalisers(rng.normal(0, 0.3, GEOM), rng.uniform(0.5, 2.0, GEOM),
                        np.zeros(STATE), np.ones(STATE),
                        np.zeros(ACT), np.ones(ACT))
    loop = make_loop(env, norms)
    obs = env.reset()
    boxes, ok = loop._slots(OraclePerception()(env.render("front")))
    boxes[:, 3:6] = np.abs(boxes[:, 3:6]) + 0.01
    z = loop._latent(boxes, boxes, ok, obs.state, obs.state)
    back = loop._to_metres([z])[0]
    assert torch.allclose(back.slots[0, :, :GEOM],
                          torch.from_numpy(boxes).float(), atol=1e-4)


def test_denormalisation_keeps_extents_non_negative(env):
    """A predicted negative extent would give the overlap term a negative volume."""
    norms = Normalisers(np.zeros(GEOM), np.ones(GEOM), np.zeros(STATE),
                        np.ones(STATE), np.zeros(ACT), np.ones(ACT))
    loop = make_loop(env, norms)
    from pipeline.world_model.latent import SceneLatent
    z = SceneLatent(torch.full((1, NSLOT, 2 * GEOM), -5.0),
                    torch.zeros(1, 2 * STATE))
    assert (loop._to_metres([z])[0].slots[..., 3:6] >= 0).all()


def test_loop_runs_and_reports_ground_truth_it_never_planned_on(env):
    loop = make_loop(env)
    rep = loop.run(seed=0)
    assert rep.steps > 0
    assert isinstance(rep.success, bool)
    assert rep.plans > 0, "the planner never ran"
    assert 0.0 <= rep.ball_displacement < 1.0
    assert rep.perception_error and np.mean(rep.perception_error) < 0.05


def test_the_waypoint_machine_advances_during_a_run(env):
    """The regression test for a loop that hovered for a whole episode.

    `scripted_chunk` restores whatever state the lookahead borrowed, so if nothing
    else advances the machine it never leaves stage 0 — and the symptom is not a crash
    but a perfectly smooth robot that reaches for the cube and then waits forever.
    """
    e = PickPlaceEnv(seed=1, randomise=False, cameras=(), width=160, height=120,
                     max_steps=300)
    loop = make_loop(e)
    seen_stages = set()
    real_act = ScriptedPickPlace.act

    def spy(self, tip, objects):
        seen_stages.add(self.stage_name)
        return real_act(self, tip, objects)

    ScriptedPickPlace.act = spy
    try:
        loop.run(seed=0)
    finally:
        ScriptedPickPlace.act = real_act
        e.close()
    assert {"hover", "descend", "close", "lift"} <= seen_stages, (
        f"machine never got past {sorted(seen_stages)}")


def test_no_plan_mode_executes_the_policys_own_chunk(env):
    loop = make_loop(env)
    loop.cfg.plan = False
    rep = loop.run(seed=0)
    assert rep.plans == 0 and rep.replaced == 0


def test_belief_candidates_are_all_real_policy_outputs(env):
    """Every candidate must be something the policy would actually have produced.

    This is the whole difference from the `action` source, and it is the difference the
    measurement turned on: additive noise makes every candidate but one strictly worse
    than the policy's own answer, so ranking can only claw back what the perturbation
    destroyed. A belief candidate is the policy's answer to a scene it might be in.
    """
    from pipeline.sim.loop import ScriptedDriver
    loop = make_loop(env)
    loop.cfg.candidates_from = "belief"
    obs = env.reset()
    seen = {k: v.centre for k, v in OraclePerception()(env.render("front")).items()}
    rng = np.random.default_rng(0)
    cands = loop._candidates(ScriptedDriver(noise=0.0, rng=np.random.default_rng(1)),
                             obs.tip, seen, obs.state, rng)
    assert cands.shape == (loop.cfg.candidates, loop.cfg.horizon, 4)
    # Grip is a command, and a belief about where the bowl is cannot change whether
    # the fingers should be shut.
    assert set(np.unique(cands[..., 3])) <= {0.0, 1.0}
    assert not np.allclose(cands[0], cands[1]), "perturbing belief changed nothing"


def test_candidate_zero_is_always_the_unperturbed_policy_chunk(env):
    from pipeline.sim.loop import ScriptedDriver
    obs = env.reset()
    seen = {k: v.centre for k, v in OraclePerception()(env.render("front")).items()}
    for source in ("action", "belief"):
        loop = make_loop(env)
        loop.cfg.candidates_from = source
        d = ScriptedDriver(noise=0.0, rng=np.random.default_rng(1))
        base = d.chunk(obs.tip, seen, obs.state, loop.cfg.horizon)
        cands = loop._candidates(d, obs.tip, seen, obs.state,
                                 np.random.default_rng(0))
        assert np.allclose(cands[0], base), source


def test_action_candidates_never_jitter_the_grip_channel(env):
    """A displacement in metres applied to a 0-1 command opens the hand mid-carry."""
    from pipeline.sim.loop import ScriptedDriver
    loop = make_loop(env)
    obs = env.reset()
    seen = {k: v.centre for k, v in OraclePerception()(env.render("front")).items()}
    cands = loop._candidates(ScriptedDriver(noise=0.0, rng=np.random.default_rng(1)),
                             obs.tip, seen, obs.state, np.random.default_rng(0))
    assert np.allclose(cands[:, :, 3], cands[0, :, 3])


def test_unknown_candidate_source_raises(env):
    from pipeline.sim.loop import ScriptedDriver
    loop = make_loop(env)
    loop.cfg.candidates_from = "vibes"
    obs = env.reset()
    with pytest.raises(ValueError, match="unknown candidate source"):
        loop._candidates(ScriptedDriver(noise=0.0, rng=np.random.default_rng(1)),
                         obs.tip, {}, obs.state, np.random.default_rng(0))


def test_bc_driver_refuses_a_horizon_it_was_not_trained_for(env):
    """Truncating a chunk changes what the policy committed to; refuse, do not clip."""
    from pipeline.sim.loop import ChunkDriver
    from pipeline.sim.policy import ChunkPolicy
    d = ChunkDriver(ChunkPolicy(obs_dim=26, horizon=8))
    with pytest.raises(SystemExit, match="8-step chunk"):
        d.chunk(np.zeros(3), {}, np.zeros(16), 4)


def test_oracle_costs_leave_the_episode_untouched(env):
    """Oracle@k executes every candidate for real. If the rewind leaked, the episode
    the loop is actually running would be corrupted by its own diagnostics."""
    from pipeline.sim.loop import ScriptedDriver
    loop = make_loop(env)
    obs = env.reset()
    seen = {k: v.centre for k, v in OraclePerception()(env.render("front")).items()}
    raw = loop._candidates(ScriptedDriver(noise=0.0, rng=np.random.default_rng(1)),
                           obs.tip, seen, obs.state, np.random.default_rng(0))
    before = (env.data.qpos.copy(), env.target.copy(), env.step_count)
    costs = loop._oracle_costs(raw, env.body_pos("ball").copy())
    assert costs.shape == (len(raw),)
    assert np.allclose(env.data.qpos, before[0])
    assert np.allclose(env.target, before[1]) and env.step_count == before[2]


def test_oracle_cost_prefers_moving_the_cube_toward_the_bowl(env):
    """The oracle's objective must actually encode the task, or its ranking is noise."""
    loop = make_loop(env)
    env.reset()
    ball0 = env.body_pos("ball").copy()
    adr = env._free_qpos["cube"]
    bowl = env.body_pos("bowl")
    far = loop._true_cost(ball0)
    env.data.qpos[adr:adr + 3] = [bowl[0], bowl[1], 0.03]
    mujoco.mj_forward(env.model, env.data)
    assert loop._true_cost(ball0) < far


def test_oracle_cost_penalises_disturbing_the_distractor(env):
    """Otherwise the oracle can look good by swatting the ball out of the way."""
    loop = make_loop(env)
    env.reset()
    ball0 = env.body_pos("ball").copy()
    quiet = loop._true_cost(ball0)
    adr = env._free_qpos["ball"]
    env.data.qpos[adr:adr + 3] += [0.10, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    assert loop._true_cost(ball0) > quiet


def test_mppi_returns_one_chunk_and_respects_the_grip_range(env):
    from pipeline.sim.loop import ScriptedDriver
    loop = make_loop(env)
    loop.cfg.optimiser = "mppi"
    obs = env.reset()
    percept = OraclePerception()(env.render("front"))
    seen = {k: v.centre for k, v in percept.items()}
    boxes, ok = loop._slots(percept)
    z0 = loop._latent(boxes, boxes, ok, obs.state, obs.state)
    out = loop._mppi(ScriptedDriver(noise=0.0, rng=np.random.default_rng(1)),
                     obs.tip, seen, obs.state, z0, boxes[1, :3],
                     np.random.default_rng(0))
    assert out.shape == (loop.cfg.horizon, 4)
    assert np.isfinite(out).all()
    assert (out[:, 3] >= 0).all() and (out[:, 3] <= 1).all()


def test_mppi_weights_survive_large_costs(env):
    """exp(-cost) underflows every weight to zero unless the costs are shifted first,
    and the mean update then becomes 0/0. Costs here are unnormalised, so this is not
    hypothetical."""
    cost = np.array([1e4, 1e4 + 1.0, 1e4 + 2.0])
    w = np.exp(-(cost - cost.min()) / 0.05)
    w = w / max(1e-12, w.sum())
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_value_is_negated_into_a_cost(env):
    """The head predicts RETURN (higher is better); the scorer minimises its total.

    Handed the return unchanged, the planner steers toward the states it should avoid —
    and does it quietly, because the number stays finite, well-scaled, and every
    diagnostic keeps working.
    """
    from pipeline.sim.loop import ValueCost
    from pipeline.world_model.latent import SceneLatent
    from pipeline.world_model.scorer import ValueHead

    head = ValueHead(2 * GEOM, 2 * STATE, hidden=8)
    wrapped = ValueCost(head, scale=1.0)
    z = SceneLatent(torch.randn(4, NSLOT, 2 * GEOM), torch.randn(4, 2 * STATE))
    ret = head(z)
    cost = wrapped(z)
    assert torch.allclose(cost, -ret)
    # The state with the highest predicted return must be the cheapest.
    assert int(ret.argmax()) == int(cost.argmin())


def test_load_value_refuses_a_head_that_loses_to_the_distance_heuristic(tmp_path):
    """A value head that cannot beat the geometric term it replaces is not an
    improvement, it is a slower restatement of it."""
    from pipeline.sim.loop import load_value
    from pipeline.world_model.scorer import ValueHead

    head = ValueHead(2 * GEOM, 2 * STATE, hidden=8)
    p = tmp_path / "value.pt"
    torch.save({"state_dict": head.state_dict(), "slot_dim": 2 * GEOM,
                "robot_dim": 2 * STATE, "hidden": 8, "gamma": 0.99,
                "test_mse": 0.0070, "distance_mse": 0.0069}, p)
    with pytest.raises(SystemExit, match="no better than"):
        load_value(p)
    torch.save({"state_dict": head.state_dict(), "slot_dim": 2 * GEOM,
                "robot_dim": 2 * STATE, "hidden": 8, "gamma": 0.99,
                "test_mse": 0.0050, "distance_mse": 0.0069}, p)
    assert load_value(p) is not None


def test_load_dynamics_refuses_a_checkpoint_with_no_slot_normaliser(tmp_path):
    """Reshaping to fit is how a silently mis-scaled model gets deployed."""
    from pipeline.sim.loop import load_dynamics
    dyn = DynamicsEnsemble(2 * GEOM, 2 * STATE, ACT, n_members=2, hidden=16, layers=1)
    p = tmp_path / "dynamics.pt"
    torch.save({"state_dict": dyn.state_dict(), "slot_dim": 2 * GEOM,
                "robot_dim": 2 * STATE, "action_dim": ACT, "members": 2,
                "hidden": 16, "layers": 1, "integrate_velocity": True,
                "integrate_slot_velocity": True, "slot_mean": None}, p)
    with pytest.raises(SystemExit, match="slot normaliser"):
        load_dynamics(p)


# ── the veto gate ──────────────────────────────────────────────────────────────

class _Spread:
    """Dynamics stub whose ensemble disagreement is whatever the test asks for."""

    def __init__(self, spread):
        self.spread = spread

    def rollout(self, z0, chunk):
        return [z0], torch.tensor([self.spread])


def test_spread_mode_fires_on_disagreement_and_needs_no_value_head():
    """The calibrated rule. `value=None` is the point: the head it would consult was
    measured at AUC 0.457, i.e. below chance, so the gate must not depend on one."""
    from pipeline.sim.veto import SPREAD_FIRE, VetoGate

    gate = VetoGate(mode="spread")
    quiet = gate(_Spread(SPREAD_FIRE - 0.05), None, object(), torch.zeros(4, ACT))
    loud = gate(_Spread(SPREAD_FIRE + 0.05), None, object(), torch.zeros(4, ACT))
    assert quiet.allowed and not loud.allowed
    assert gate.fired == 1 and gate.checked == 2
    assert np.isnan(quiet.predicted_drop)      # recorded as absent, not as zero


def test_drop_mode_still_treats_spread_as_a_disqualifier():
    """The refuted design, kept runnable. It must keep behaving the way it did when
    the calibration refuted it, or the comparison stops being reproducible."""
    from pipeline.sim.veto import VetoGate

    values = iter([1.0, 0.0] * 4)
    value = lambda _z: next(values)                                    # noqa: E731
    confident = VetoGate(mode="drop", drop=0.15, max_spread=0.05)
    assert not confident(_Spread(0.01), value, object(), torch.zeros(4, ACT)).allowed
    unsure = VetoGate(mode="drop", drop=0.15, max_spread=0.05)
    d = unsure(_Spread(0.50), value, object(), torch.zeros(4, ACT))
    assert d.allowed and "not confident enough" in d.reason


def test_hold_chunk_keeps_the_grip_and_smuggles_in_no_selection():
    """The fallback must be inaction. Substituting a *better* action would make the
    veto a selector again, which is the thing that was measured not to work."""
    from pipeline.sim.veto import hold_chunk

    held = hold_chunk(6, 1.0)
    assert held.shape == (6, 4)
    assert np.all(held[:, :3] == 0.0) and np.all(held[:, 3] == 1.0)


# ── reporting a rate with an interval ──────────────────────────────────────────

def test_bootstrap_ci_is_degenerate_when_every_episode_agrees():
    """All-success must report [1, 1], not a band. A CI that widens on unanimous data
    would make the 95.8% policy-first arm look uncertain when it is not."""
    from pipeline.sim.loop import bootstrap_ci

    lo, hi = bootstrap_ci([True] * 12)
    assert lo == 1.0 and hi == 1.0
    lo, hi = bootstrap_ci([False] * 12)
    assert lo == 0.0 and hi == 0.0


def test_bootstrap_ci_stays_inside_the_unit_interval():
    """The reason for the percentile method rather than mean +- 1.96 sigma: at 23/24
    the normal interval runs past 1.0, and a success rate above 100% is not a result."""
    from pipeline.sim.loop import bootstrap_ci

    lo, hi = bootstrap_ci([True] * 23 + [False])
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == 1.0                       # the upper end is reachable and is the cap
    assert lo < 23 / 24


def test_bootstrap_ci_brackets_the_point_estimate():
    from pipeline.sim.loop import bootstrap_ci

    vals = [True] * 12 + [False] * 12
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.5 < hi
    assert hi - lo > 0.1                   # 24 episodes cannot pin a rate tighter


def test_bootstrap_ci_is_reproducible_and_handles_no_episodes():
    from pipeline.sim.loop import bootstrap_ci

    vals = [True, False, True, True, False]
    assert bootstrap_ci(vals) == bootstrap_ci(vals)
    assert all(np.isnan(v) for v in bootstrap_ci([]))


def test_one_candidate_records_no_margin_instead_of_crashing(env):
    """`--candidates 1` is the control that must reproduce the policy exactly: with
    nothing to choose between, greedy selection has to be a no-op. It used to raise
    IndexError reading the runner-up score, which made the control unrunnable."""
    from pipeline.sim.loop import ClosedLoop, LoopConfig, Normalisers

    dyn = DynamicsEnsemble(2 * GEOM, 2 * STATE, ACT, n_members=2, hidden=16, layers=1)
    loop = ClosedLoop(env, OraclePerception(), dyn,
                      Normalisers.identity(GEOM, STATE, ACT),
                      cfg=LoopConfig(horizon=4, execute=2, candidates=1,
                                     candidates_from="affordance", pick="best"),
                      n_slots=NSLOT, slot_geom=GEOM)
    rep = loop.run(seed=0)
    assert rep.plans > 0
    assert rep.margins == []
    assert rep.replaced == 0          # index 0 is the only choice


def test_task_fingerprint_separates_two_step_budgets():
    """[S21]: the budget is part of the task, so it must change the hash.

    The bug this guards was not a wrong answer, it was a confident one — two runs at
    360 and 480 steps printed the same hash while K=16 moved from 18.8% to 83.3%. A
    fingerprint that omits a field certifies the wrong answer rather than staying
    silent, so the field being *present* is the thing worth asserting.
    """
    from pipeline.sim.loop import task_fingerprint

    a, b = task_fingerprint(360), task_fingerprint(480)
    assert a["max_env_steps"] == 360 and b["max_env_steps"] == 480
    assert a["task_sha"] != b["task_sha"]
    assert task_fingerprint(360)["task_sha"] == a["task_sha"]


def test_task_fingerprint_tracks_the_demonstrator_constants():
    """[S20]: and the constants that started all this still move the hash."""
    from pipeline.sim.loop import task_fingerprint
    from pipeline.sim.policy import ScriptedPickPlace

    before = task_fingerprint(360)["task_sha"]
    old = ScriptedPickPlace.stage_timeout
    try:
        ScriptedPickPlace.stage_timeout = old + 1
        assert task_fingerprint(360)["task_sha"] != before
    finally:
        ScriptedPickPlace.stage_timeout = old
    assert task_fingerprint(360)["task_sha"] == before

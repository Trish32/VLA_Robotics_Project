"""Interventions must really replan and obey the original absolute deadline."""
from copy import deepcopy
import numpy as np
import pytest

from grootN1_Robotics.candidate_outcomes import PolicyStream, run_continuation, summarize_points
from grootN1_Robotics.tests.test_branch_replay import replay
from grootN1_Robotics.tests.test_baseline import ToyEnv, rollout


def labels(env):
    return {'task_success': len(env.unwrapped.actions) == env.unwrapped.stop and env.unwrapped.success}


def state(env):
    return np.asarray(env.unwrapped.actions, np.float64)


class ReactivePolicy:
    def get_action(self, obs):
        # Respond to the actual altered history, rather than following the old actions.
        value = self.base.actions[-1] + 1
        return {'action.joint': np.full((1, 4, 1), value, np.float32)}, {}


def test_candidate_replans_from_altered_history_and_keeps_absolute_budget():
    original = rollout(ToyEnv(), budget=8)
    cold = replay(original); cold.advance(3)
    policy = ReactivePolicy(); policy.base = cold.env.unwrapped
    result = run_continuation(cold, {'action.joint': np.full((1, 4, 1), 50, np.float32)},
        policy, max_steps=8, action_space=cold.env.unwrapped.action_space,
        label_reader=labels, state_reader=state)
    assert cold.env.unwrapped.actions == [10, 11, 12, 50, 50, 50, 51, 51]
    assert result['sim_steps'] == 8 and result['continuation_policy_calls'] == 1
    assert [r['sim_step'] for r in result['trace']] == list(range(3, 9))


def test_reference_checks_hidden_state_on_every_frame():
    original = rollout(ToyEnv(), budget=5)
    control = replay(original, label_reader=labels); control.advance(5)
    cold = replay(original, label_reader=labels); cold.advance(3)
    result = run_continuation(cold, cold.actions[1], None, max_steps=5,
        action_space=cold.env.unwrapped.action_space, label_reader=labels, state_reader=state,
        expected_trace=control.trace, reference_episode=original)
    assert result['trace'] == control.trace[3:]
    corrupted = deepcopy(control.trace); corrupted[4]['integration_sha256'] = '0' * 64
    cold = replay(original, label_reader=labels); cold.advance(3)
    with pytest.raises(ValueError, match='reference trace'):
        run_continuation(cold, cold.actions[1], None, max_steps=5,
            action_space=cold.env.unwrapped.action_space, label_reader=labels, state_reader=state,
            expected_trace=corrupted)


def test_success_stops_inside_the_intervention_chunk():
    original = rollout(ToyEnv(stop=4, success=True), budget=5)
    # Build the same stop-enabled environment without changing the recorded prefix.
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from grootN1_Robotics.branch_replay import RecordedReplay
    base = ToyEnv(stop=4, success=True)
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    cold = RecordedReplay(env, original, execute=3, horizon=4, action_space=base.action_space,
                          state_reader=state, label_reader=labels)
    cold.advance(3)
    r = run_continuation(cold, cold.actions[1], None, max_steps=5,
        action_space=base.action_space, label_reader=labels, state_reader=state)
    assert r['success'] and r['sim_steps'] == 4 and len(base.actions) == 4


@pytest.mark.parametrize('bad', ['dtype', 'nan', 'boundary'])
def test_invalid_intervention_never_reaches_simulator(bad):
    cold = replay(rollout(ToyEnv())); cold.advance(1 if bad == 'boundary' else 3)
    actions = deepcopy(cold.actions[0])
    if bad == 'dtype': actions['action.joint'] = actions['action.joint'].astype(np.float64)
    if bad == 'nan': actions['action.joint'][0, 0, 0] = np.nan
    before = list(cold.env.unwrapped.actions)
    with pytest.raises(ValueError):
        run_continuation(cold, actions, None, max_steps=5,
            action_space=cold.env.unwrapped.action_space, label_reader=labels, state_reader=state)
    assert cold.env.unwrapped.actions == before


def test_policy_rng_is_paired_and_leaves_simulator_rng_unchanged():
    import torch
    from grootN1_Robotics.decision_capture import capture_rng, rng_equal
    class RandomPolicy:
        def reset(self): pass
        def get_action(self, obs): return torch.rand(3), {}
    saved = capture_rng('cpu')
    a, b = PolicyStream(RandomPolicy(), saved, 'cpu'), PolicyStream(RandomPolicy(), saved, 'cpu')
    x, _ = a.get_action({}); torch.rand(100)
    before = capture_rng('cpu')
    y, _ = b.get_action({})
    assert torch.equal(x, y) and rng_equal(before, capture_rng('cpu'))
    assert torch.equal(a.get_action({})[0], b.get_action({})[0])


def test_oracle_opportunity_is_separate_from_success_breakage():
    def point(seed, outcomes):
        return {'seed': seed, 'sim_step': 0, 'arms': [dict(arm=name, success=ok) for name, ok in
            zip(['reference', *[f'candidate_{i}' for i in range(4)]], outcomes)]}
    r = summarize_points([point(0, [False, False, True, False, False]),
                          point(3, [True, False, True, True, True])])
    assert r['rescued_failure_points'] == 1 and r['broken_success_alternatives'] == 1
    assert not r['selection_rule_evaluated']
    with pytest.raises(ValueError, match='incomplete'):
        summarize_points([{'seed': 0, 'sim_step': 0, 'arms': []}])

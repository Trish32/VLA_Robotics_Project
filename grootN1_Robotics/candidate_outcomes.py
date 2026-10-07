"""One action-chunk intervention, then the unchanged closed-loop policy.

Cold history replay owns reconstruction; this module never restores partial physics
state or follows frozen reference actions after a counterfactual.
"""
from __future__ import annotations

import math
import numpy as np

from grootN1_Robotics.baseline import batch_observation, validate_chunk
from grootN1_Robotics.branch_replay import integration_state, observation_hash


# Fixed before candidate outcomes are measured: four failure stages, two controls.
PILOT_POINTS = [(0, 256), (2, 160), (7, 160), (8, 224), (3, 160), (5, 160)]


class PolicyStream:
    """Keep continuation draws paired and independent of simulator RNG consumption."""
    def __init__(self, policy, rng, device):
        self.policy, self.rng, self.device = policy, rng, device
        policy.reset()

    def get_action(self, observation):
        from grootN1_Robotics.decision_capture import capture_rng, isolated_rng, restore_rng
        with isolated_rng(self.device):
            restore_rng(self.rng)
            actions, info = self.policy.get_action(observation)
            self.rng = capture_rng(self.device)
        return actions, info


def step_trace(env, observation, step, reward, success, terminated, truncated,
               *, state_reader=integration_state, label_reader):
    state = np.asarray(state_reader(env))
    labels = label_reader(env)
    if (not math.isfinite(reward) or not np.isfinite(state).all()
            or bool(labels['task_success']) != success):
        raise ValueError('invalid branch reward/state or inconsistent official success')
    row = {'sim_step': step, 'observation_sha256': observation_hash(observation),
           'policy_observation_sha256': observation_hash(batch_observation(observation)),
           'integration_sha256': observation_hash({'integration': state}),
           'integration_size': state.size, 'labels': labels, 'reward': reward,
           'success': success, 'terminated': terminated, 'truncated': truncated}
    import json
    json.dumps(row, allow_nan=False)
    return row


def run_continuation(replay, first_actions, policy, *, max_steps, action_space,
                     label_reader, state_reader=integration_state,
                     expected_trace=None, reference_episode=None, observer=None):
    """Budget is absolute episode time, including the already replayed prefix.

The first chunk is the sole intervention. All subsequent chunks must be predicted
from the new observations. A reference arm must reproduce every saved control row
and every original decoded action, not just the final success flag.
    """
    start = replay.step
    if (start % replay.execute or start >= max_steps or
            replay.success or replay.terminated or replay.truncated):
        raise ValueError('branch needs a live decision boundary before its deadline')
    if expected_trace is not None and replay.trace != expected_trace[:start + 1]:
        raise ValueError('branch prefix differs from verified oracle')
    env, observation = replay.env, replay.observation
    step, success, terminated, truncated = start, False, False, False
    trace, decisions = [replay.trace[-1]], []
    actions = first_actions
    if observer:
        observer(env, observation, step, success)
    while step < max_steps:
        if decisions:
            actions, _ = policy.get_action(batch_observation(observation))
        validate_chunk(actions, action_space, replay.horizon)
        decision = {'sim_step': step,
                    'state': {k: np.asarray(v).tolist() for k, v in observation.items()
                              if k.startswith('state.')},
                    'action_chunk': {k: np.asarray(v)[0].tolist() for k, v in actions.items()}}
        if reference_episode is not None:
            index = step // replay.execute
            if (index >= len(reference_episode['decisions']) or
                    decision != reference_episode['decisions'][index]):
                raise ValueError(f'closed-loop reference action/state differs at step {step}')
        decision['policy_observation_sha256'] = observation_hash(batch_observation(observation))
        decisions.append(decision)
        for offset in range(min(replay.execute, max_steps - step)):
            observation, _, terminated, truncated, info = env.step(
                {k: v[0, offset:offset + 1] for k, v in actions.items()})
            if 'success' not in info:
                raise ValueError('branch environment lacks official success')
            step += 1
            success = bool(np.asarray(info['success']).any())
            terminated, truncated = bool(terminated), bool(truncated)
            row = step_trace(env, observation, step, float(np.asarray(info['rewards']).sum()),
                             success, terminated, truncated,
                             state_reader=state_reader, label_reader=label_reader)
            if expected_trace is not None and (
                    step >= len(expected_trace) or row != expected_trace[step]):
                raise ValueError(f'closed-loop reference trace differs at step {step}')
            trace.append(row)
            if observer:
                observer(env, observation, step, success)
            if success or terminated or truncated:
                break
        if success or terminated or truncated:
            break
    reason = 'success' if success else 'truncated' if truncated else 'terminated' if terminated else 'budget'
    if expected_trace is not None and len(expected_trace) != step + 1:
        raise ValueError('closed-loop reference stops before the verified control')
    return {'start_step': start, 'sim_steps': step, 'success': success,
            'stop_reason': reason, 'continuation_policy_calls': len(decisions) - 1,
            'decisions': decisions, 'trace': trace}


def summarize_points(points):
    """Descriptive oracle opportunity only; these selected scenes are not a test set."""
    rows = []
    for point in points:
        arms = point['arms']
        if [a['arm'] for a in arms] != ['reference', 'candidate_0', 'candidate_1',
                                       'candidate_2', 'candidate_3']:
            raise ValueError('incomplete or reordered candidate arms')
        baseline = arms[0]['success']
        alternatives = arms[1:]
        rows.append({'seed': point['seed'], 'sim_step': point['sim_step'],
            'reference_success': baseline,
            'successful_alternatives': sum(a['success'] for a in alternatives),
            'rescued_failure': not baseline and any(a['success'] for a in alternatives),
            'broken_success_alternatives': sum(not a['success'] for a in alternatives) if baseline else 0})
    return {'points': rows, 'rescued_failure_points': sum(r['rescued_failure'] for r in rows),
            'failed_reference_points': sum(not r['reference_success'] for r in rows),
            'broken_success_alternatives': sum(r['broken_success_alternatives'] for r in rows),
            'selection_rule_evaluated': False, 'generalization_evaluated': False}

"""An independent audit must reject misleading branch outcomes and interventions."""
from copy import deepcopy
import numpy as np
import pytest

from grootN1_Robotics.candidate_outcomes import run_continuation
from grootN1_Robotics.tests.test_candidate_outcomes import labels, state
from grootN1_Robotics.tests.test_branch_replay import replay
from grootN1_Robotics.tests.test_baseline import ToyEnv, rollout
from grootN1_Robotics.tools.analyze_candidate_outcomes import verify_arm


def example():
    original = rollout(ToyEnv(), budget=5)
    cold = replay(original, label_reader=labels); cold.advance(3)
    first = cold.actions[1]
    result = run_continuation(cold, first, None, max_steps=5,
        action_space=cold.env.unwrapped.action_space, label_reader=labels, state_reader=state)
    return result, first


def test_complete_branch_is_auditable():
    result, first = example()
    verify_arm(result, start=3, deadline=5, execute=3, first_actions=first)


@pytest.mark.parametrize('change', ['frame', 'success', 'stop', 'action', 'observation',
                                   'nan', 'budget', 'oracle', 'continued'])
def test_changed_branch_is_rejected(change):
    result, first = example(); result = deepcopy(result)
    control = None
    if change == 'frame': result['trace'].pop(1)
    elif change == 'success': result['success'] = True
    elif change == 'stop': result['stop_reason'] = 'success'
    elif change == 'action': result['decisions'][0]['action_chunk']['action.joint'][0][0] += 1
    elif change == 'observation': result['decisions'][0]['policy_observation_sha256'] = '0' * 64
    elif change == 'nan': result['trace'][-1]['reward'] = float('nan')
    elif change == 'budget': result['sim_steps'] = 6
    elif change == 'oracle':
        control = [{}] * 3 + deepcopy(result['trace']); control[-1]['integration_sha256'] = '0' * 64
    else: result['trace'][0]['terminated'] = True
    with pytest.raises(ValueError):
        verify_arm(result, start=3, deadline=5, execute=3, first_actions=first, control=control)

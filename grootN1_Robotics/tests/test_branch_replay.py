"""Branch history and wrapper state must agree, not just the joint observation."""
from copy import deepcopy

import numpy as np
import pytest

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from grootN1_Robotics.baseline import batch_observation
from grootN1_Robotics.branch_replay import RecordedReplay, observation_hash, require_same_observation
from grootN1_Robotics.tests.test_baseline import ToyEnv, rollout


def fresh():
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    return base, env


def replay(episode, **kwargs):
    base, env = fresh()
    return RecordedReplay(env, episode, execute=3, horizon=4, action_space=base.action_space,
        state_reader=lambda _: np.asarray(base.actions, np.float64), **kwargs)


def test_cold_prefix_pause_and_continuation_match_continuous_control():
    episode = rollout(ToyEnv(), budget=5)
    continuous = replay(episode)
    continuous.advance(5)
    cold = replay(episode, expected_trace=continuous.trace)
    cold.advance(1)  # pause inside a chunk; the next action must retain offset 1
    cold.advance(3)  # pause on the next policy decision
    cold.advance(5)
    assert cold.trace == continuous.trace
    assert cold.env.unwrapped.actions == [10, 11, 12, 20, 21]


def test_capture_checks_pixels_and_container_dtype():
    episode = rollout(ToyEnv(), budget=5)
    control = replay(episode)
    observation = batch_observation(control.advance(3))
    cold = replay(episode, captures={3: observation})
    cold.advance(5)
    changed = deepcopy(observation)
    changed["video.camera"][0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="video.camera"):
        replay(episode, captures={3: changed}).advance(3)


def test_hidden_integration_state_difference_is_rejected():
    episode = rollout(ToyEnv())
    control = replay(episode)
    control.advance(5)
    cold = replay(episode, expected_trace=control.trace)
    cold.state_reader = lambda _: np.zeros(1)
    with pytest.raises(ValueError, match="trace differs"):
        cold.advance(1)


def test_recorded_contact_labels_must_match_at_every_step():
    episode = rollout(ToyEnv())
    labels = {i: {"contact": False} for i in range(6)}
    cold = replay(episode, label_reader=lambda _: {"contact": False}, recorded_labels=labels)
    labels[2]["contact"] = True
    with pytest.raises(ValueError, match="task labels differ at step 2"):
        cold.advance(5)


@pytest.mark.parametrize("mutation", ["initial", "cadence", "state", "stop", "action"])
def test_corrupt_reference_is_rejected(mutation):
    episode = rollout(ToyEnv())
    if mutation == "initial":
        episode["initial_observation_sha256"] = "changed"
    elif mutation == "cadence":
        episode["decisions"].pop()
    elif mutation == "state":
        episode["decisions"][1]["state"]["state.joint"] = [[-1.]]
    elif mutation == "stop":
        episode["success"] = True
    else:
        episode["decisions"][0]["action_chunk"]["action.joint"][0][0] = float("nan")
    with pytest.raises(ValueError):
        replay(episode).advance(5)


def test_premature_stop_is_not_counted_as_a_failed_candidate():
    episode = rollout(ToyEnv())
    base = ToyEnv(stop=1, truncate=True)
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    cold = RecordedReplay(env, episode, execute=3, horizon=4, action_space=base.action_space,
                          state_reader=lambda _: np.asarray(base.actions))
    with pytest.raises(ValueError, match="prematurely"):
        cold.advance(5)


def test_observation_dtype_nan_and_missing_keys_are_rejected():
    with pytest.raises(ValueError, match="differs"):
        require_same_observation({"x": np.zeros(1, np.float64)}, {"x": np.zeros(1, np.float32)})
    with pytest.raises(ValueError, match="keys"):
        require_same_observation({"x": np.zeros(1)}, {})
    with pytest.raises(ValueError, match="non-finite"):
        observation_hash({"x": np.array([float("nan")])})

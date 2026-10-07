"""The independent artifact gate rejects altered signals, inputs and continuations."""
from copy import deepcopy

import numpy as np
import pytest

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from grootN1_Robotics.baseline import batch_observation
from grootN1_Robotics.branch_replay import RecordedReplay
from grootN1_Robotics.tests.test_baseline import ToyEnv, rollout
from grootN1_Robotics.tools.analyze_branch_replay import verify_episode


def artifacts():
    original = rollout(ToyEnv())
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    control = RecordedReplay(env, original, execute=3, horizon=4, action_space=base.action_space,
        state_reader=lambda _: np.array([len(base.actions), sum(base.actions)], np.float64),
        label_reader=lambda _: {"task_success": False, "contact": len(base.actions) > 1})
    captures = {0: batch_observation(control.observation)}
    captures[3] = batch_observation(control.advance(3))
    control.advance(5)
    result = {**{k: original[k] for k in ["seed", "sim_steps", "success", "stop_reason"]},
        "exact_reconstruction_verified": True, "exact_step_traces": 6,
        "captured_observations_verified": 2}
    signals = [{"sim_step": row["sim_step"], "success": row["success"], **row["labels"]}
               for row in control.trace]
    return original, result, [control.trace, deepcopy(control.trace)], signals, captures


def test_complete_cold_branch_artifacts():
    verify_episode(*artifacts())


@pytest.mark.parametrize("change", ["future", "contact", "pixels", "missing", "nan", "early_stop"])
def test_changed_branch_artifacts_are_refused(change):
    original, result, traces, signals, captures = artifacts()
    if change == "future":
        traces[1][4]["integration_sha256"] = "f" * 64
    elif change == "contact":
        signals[2]["contact"] = False
    elif change == "pixels":
        captures[3]["video.camera"][0, 0, 0, 0, 0] = 1
    elif change == "missing":
        traces[1].pop()
    elif change == "nan":
        traces[0][1]["reward"] = traces[1][1]["reward"] = float("nan")
    else:
        traces[0][2]["truncated"] = traces[1][2]["truncated"] = True
    with pytest.raises(ValueError):
        verify_episode(original, result, traces, signals, captures)

"""Reconstruct simulator branches by replaying history in a fresh seeded environment.

No partial MuJoCo restore: controllers, observation windows and task bookkeeping
are reconstructed by their official reset/step paths. This is a correctness oracle,
not a fast snapshot implementation or a candidate-quality evaluator.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from grootN1_Robotics.baseline import batch_observation, validate_chunk


def observation_hash(observation):
    digest = hashlib.sha256()
    for key, value in sorted(observation.items()):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{key}: object array in replay observation")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{key}: non-finite replay observation")
        digest.update(json.dumps([key, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_same_observation(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("reconstructed policy observation keys differ")
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"reconstructed policy observation differs: {key}")


def integration_state(env):
    """Read the native full integration state, including solver warmstart and control."""
    import mujoco
    task = env.unwrapped.env
    model, data = task.sim.model._model, task.sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
    mujoco.mj_getState(model, data, state, spec)
    if not np.isfinite(state).all():
        raise ValueError("non-finite MuJoCo integration state")
    return state


class RecordedReplay:
    """Pause at arbitrary steps and continue the recorded actions without resampling.

    Each instance owns a fresh official environment. Compare a continuous control
    with a second instance paused at captured decisions before admitting branches.
    """
    def __init__(self, env, episode, *, execute, horizon, action_space,
                 state_reader=integration_state, label_reader=None,
                 captures=None, expected_trace=None, recorded_labels=None):
        if execute < 1 or execute > horizon or episode["sim_steps"] < 1:
            raise ValueError("invalid replay horizon or episode length")
        self.env, self.episode = env, episode
        self.execute, self.horizon = execute, horizon
        self.state_reader, self.label_reader = state_reader, label_reader
        self.captures = captures or {}
        self.expected_trace = expected_trace
        self.recorded_labels = recorded_labels
        self.step = 0
        self.trace = []
        self.success = self.terminated = self.truncated = False
        expected_steps = list(range(0, episode["sim_steps"], execute))
        if [d["sim_step"] for d in episode["decisions"]] != expected_steps:
            raise ValueError("recorded decision cadence is incomplete or changed")
        if not set(self.captures) <= set(expected_steps):
            raise ValueError("capture is not a recorded decision")
        self.actions = []
        for decision in episode["decisions"]:
            actions = {k: np.asarray(v, np.float32)[None] for k, v in decision["action_chunk"].items()}
            validate_chunk(actions, action_space, horizon)
            self.actions.append(actions)
        self.observation, self.info = env.reset(seed=episode["seed"])
        if observation_hash(self.observation) != episode["initial_observation_sha256"]:
            raise ValueError("reconstructed initial observation differs")
        self._record(0.0)

    def _record(self, reward):
        if self.step < self.episode["sim_steps"] and self.step % self.execute == 0:
            decision = self.episode["decisions"][self.step // self.execute]
            actual = {k: np.asarray(v).tolist() for k, v in self.observation.items() if k.startswith("state.")}
            if actual != decision["state"]:
                raise ValueError(f"recorded decision state differs at step {self.step}")
        if self.step in self.captures:
            require_same_observation(batch_observation(self.observation), self.captures[self.step])
        state = np.asarray(self.state_reader(self.env))
        if state.dtype.hasobject or not np.isfinite(state).all():
            raise ValueError("invalid replay integration state")
        labels = self.label_reader(self.env) if self.label_reader else None
        if labels is not None and "task_success" in labels and bool(labels["task_success"]) != self.success:
            raise ValueError("replay task predicate differs from the official success flag")
        if self.recorded_labels is not None and labels != self.recorded_labels.get(self.step):
            raise ValueError(f"recorded task labels differ at step {self.step}")
        row = {"sim_step": self.step, "observation_sha256": observation_hash(self.observation),
               "policy_observation_sha256": observation_hash(batch_observation(self.observation)),
               "integration_sha256": observation_hash({"integration": state}),
               "integration_size": state.size, "labels": labels, "reward": float(reward),
               "success": self.success, "terminated": self.terminated, "truncated": self.truncated}
        # Also reject NaN/Inf in labels/rewards rather than allowing ambiguous JSON.
        json.dumps(row, allow_nan=False)
        if self.expected_trace is not None and (
                self.step >= len(self.expected_trace) or row != self.expected_trace[self.step]):
            raise ValueError(f"cold replay trace differs at step {self.step}")
        self.trace.append(row)

    def advance(self, target):
        if not isinstance(target, int) or not self.step <= target <= self.episode["sim_steps"]:
            raise ValueError("target must be a forward step within the recorded episode")
        while self.step < target:
            if self.success or self.terminated or self.truncated:
                raise ValueError("recorded replay stopped prematurely")
            index, offset = divmod(self.step, self.execute)
            actions = {k: v[0, offset:offset + 1] for k, v in self.actions[index].items()}
            self.observation, _, terminated, truncated, self.info = self.env.step(actions)
            if "success" not in self.info:
                raise ValueError("replay environment does not provide success")
            self.step += 1
            self.success |= bool(np.asarray(self.info["success"]).any())
            self.terminated, self.truncated = bool(terminated), bool(truncated)
            self._record(float(np.asarray(self.info["rewards"]).sum()))
        if self.step == self.episode["sim_steps"]:
            reason = ("success" if self.success else "truncated" if self.truncated else
                      "terminated" if self.terminated else "budget")
            if self.success != self.episode["success"] or reason != self.episode["stop_reason"]:
                raise ValueError("recorded replay stopping outcome differs")
            if self.expected_trace is not None and len(self.trace) != len(self.expected_trace):
                raise ValueError("cold replay trace length differs")
        return self.observation

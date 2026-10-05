"""No-selection rollout bookkeeping; environments and policy decoding stay upstream.

The upstream MultiStepWrapper is used with one action per step. This lets the
driver stop on base-env truncation and count simulator steps, even mid-chunk.
Imports of upstream, torch and simulator packages belong in the CLI/use sites.
"""
from __future__ import annotations

import time
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_contract(env, configs, execute: int) -> None:
    horizon = list(configs["action"].delta_indices)
    if horizon != list(range(len(horizon))):
        raise ValueError("baseline requires contiguous action indices starting at zero")
    if not 1 <= execute <= len(horizon):
        raise ValueError(f"execute must be in [1, {len(horizon)}]")
    for modality, space in (("video", env.observation_space),
                            ("state", env.observation_space),
                            ("action", env.action_space)):
        expected = {f"{modality}.{key}" for key in configs[modality].modality_keys}
        actual = {key for key in space.spaces if key.startswith(f"{modality}.")}
        if (modality == "action" and expected != actual) or not expected <= actual:
            raise ValueError(f"{modality} interface mismatch: expected {expected}, got {actual}")
    for key in configs["language"].modality_keys:
        # Preserve the official DC-environment compatibility alias.
        if key not in env.observation_space.spaces and not (
                key == "task" and "annotation.human.coarse_action" in env.observation_space.spaces):
            raise ValueError(f"missing language observation {key}")


def batch_observation(observation):
    # MultiStepWrapper already adds the temporal axis; only batch is missing.
    return {key: [str(np.asarray(value).reshape(-1)[0])]
            if isinstance(value, str) or np.asarray(value).dtype.kind in "US"
            else np.asarray(value)[None]
            for key, value in observation.items()}


def validate_chunk(actions, action_space, horizon):
    if set(actions) != set(action_space.spaces):
        raise ValueError("policy action keys do not match environment")
    for key, space in action_space.spaces.items():
        value = np.asarray(actions[key])
        expected = (1, horizon, *space.shape)
        if value.shape != expected or value.dtype != np.float32:
            raise ValueError(f"{key}: expected float32 {expected}, got {value.dtype} {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: non-finite action")
    # Do not clip or reinterpret units: upstream owns embodiment action semantics.


def budget_outcomes(episodes, budgets, control_freq):
    """Success-by-deadline from one rollout; late successes never enter earlier budgets."""
    return {str(b): {
        "simulation_seconds": b / control_freq,
        "successful_seeds": [e["seed"] for e in episodes if e["success"] and e["sim_steps"] <= b],
        "success_rate": sum(e["success"] and e["sim_steps"] <= b for e in episodes) / len(episodes)}
        for b in budgets}


def run_episode(env, policy, *, seed: int, execute: int,
                horizon: int, max_steps: int, action_space,
                capture_dir: Path | None = None, observer=None,
                reference_episode: dict | None = None, reference_mode="prefix",
                decision_capture=None) -> dict:
    """Consume the policy's first `execute` steps, then observe and replan.

    env is the official MultiStepWrapper configured with n_action_steps=1.
    Success must be the simulator's flag, never a reward or policy-stage proxy.
    """
    if max_steps < 1 or not 1 <= execute <= horizon:
        raise ValueError("invalid step budget or execution horizon")
    if reference_mode not in {"prefix", "initial"}:
        raise ValueError("reference mode must be prefix or initial")
    if reference_episode is not None and not reference_episode.get("decisions"):
        raise ValueError("reference episode has no initial decision")
    obs, info = env.reset(seed=seed)
    digest = hashlib.sha256()
    for key, value in sorted(obs.items()):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{key}: object array cannot be captured reproducibly")
        digest.update(json.dumps([key, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes())
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(capture_dir / f"episode_{seed}_first_observation.npz", **obs)
    if reference_episode is not None:
        if reference_episode["seed"] != seed or reference_episode["initial_observation_sha256"] != digest.hexdigest():
            raise ValueError(f"seed {seed}: initial observation differs from reference")
    if observer is not None:
        observer(env, obs, 0, False)
    policy.reset()
    steps, calls, total_reward = 0, 0, 0.0
    latencies = []
    decisions = []
    success = False
    reason = "budget"
    started = time.perf_counter()
    while steps < max_steps:
        start = time.perf_counter()
        observation = batch_observation(obs)
        if decision_capture is None:
            actions, _ = policy.get_action(observation)
        else:
            actions, _ = decision_capture.predict(policy, observation, seed=seed,
                                                  sim_step=steps, env=env)
        latencies.append(time.perf_counter() - start if decision_capture is None
                         else decision_capture.inference_seconds)
        calls += 1
        validate_chunk(actions, action_space, horizon)
        decision = {"sim_step": steps,
            "state": {k: np.asarray(v).tolist() for k, v in obs.items() if k.startswith("state.")},
            "action_chunk": {k: np.asarray(v)[0].tolist() for k, v in actions.items()}}
        compare = (reference_episode is not None and
                   ((reference_mode == "prefix" and len(decisions) < len(reference_episode["decisions"]))
                    or (reference_mode == "initial" and not decisions)))
        if compare:
            if decision != reference_episode["decisions"][len(decisions)]:
                raise ValueError(f"seed {seed}: state/action prefix differs at simulator step {steps}")
        decisions.append(decision)
        for offset in range(min(execute, max_steps - steps)):
            step_action = {key: value[0, offset:offset + 1] for key, value in actions.items()}
            obs, reward, terminated, truncated, info = env.step(step_action)
            steps += 1
            # Wrapper rewards are cumulative; its per-step rewards are not.
            total_reward += float(np.asarray(info["rewards"]).sum())
            if "success" not in info:
                raise ValueError("environment does not provide a success flag")
            success |= bool(np.asarray(info["success"]).any())
            if observer is not None:
                observer(env, obs, steps, success)
            if success or terminated or truncated:
                reason = "success" if success else "truncated" if truncated else "terminated"
                break
        if success or terminated or truncated:
            break
    if reference_episode is not None and reference_mode == "prefix":
        if len(decisions) < len(reference_episode["decisions"]):
            raise ValueError(f"seed {seed}: stopped before reference prefix was reproduced")
        if reference_episode["success"] and (not success or steps != reference_episode["sim_steps"]):
            raise ValueError(f"seed {seed}: previous success was not reproduced at the same step")
    return {"seed": seed, "success": success, "stop_reason": reason,
            "sim_steps": steps, "policy_calls": calls, "reward": total_reward,
            "seconds": time.perf_counter() - started,
            "inference_seconds": latencies, "initial_observation_sha256": digest.hexdigest(),
            "decisions": decisions,
            "reference_prefix_verified": reference_episode is not None and reference_mode == "prefix",
            "reference_initial_decision_verified": reference_episode is not None}

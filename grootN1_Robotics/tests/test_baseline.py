"""Exercise action commitment against the real upstream temporal wrapper."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from grootN1_Robotics.baseline import run_episode, validate_contract
from grootN1_Robotics.tools.eval_baseline import checkpoint_contract
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper


class ToyEnv(gym.Env):
    def __init__(self, stop=99, truncate=False, success=False):
        self.observation_space = gym.spaces.Dict({
            "video.camera": gym.spaces.Box(0, 255, (2, 2, 3), np.uint8),
            "state.joint": gym.spaces.Box(-100, 100, (1,), np.float32),
            "task": gym.spaces.Text(100)})
        self.action_space = gym.spaces.Dict({
            "action.joint": gym.spaces.Box(-100, 100, (1,), np.float32)})
        self.stop, self.truncate, self.success = stop, truncate, success
        self.actions = []

    def observation(self):
        return {"video.camera": np.zeros((2, 2, 3), np.uint8),
                "state.joint": np.array([len(self.actions)], np.float32), "task": "move"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actions = []
        return self.observation(), {"success": False}

    def step(self, action):
        self.actions.append(float(action["action.joint"][0]))
        stop = len(self.actions) == self.stop
        return self.observation(), 1., stop and not self.truncate, stop and self.truncate, {
            "success": stop and self.success}


class Policy:
    def reset(self):
        self.calls = 0

    def get_action(self, observation):
        assert observation["state.joint"].shape == (1, 1, 1)
        assert observation["video.camera"].shape == (1, 1, 2, 2, 3)
        assert observation["task"] == ["move"]
        self.calls += 1
        return {"action.joint": (np.arange(4, dtype=np.float32) + self.calls * 10)[None, :, None]}, {}


def rollout(base, budget=5):
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    return run_episode(env, Policy(), seed=3, execute=3, horizon=4,
                       max_steps=budget, action_space=base.action_space)


def test_chunk_prefix_and_exact_budget():
    base = ToyEnv()
    result = rollout(base)
    assert base.actions == [10, 11, 12, 20, 21]
    assert result["policy_calls"] == 2
    assert result["sim_steps"] == 5
    assert result["reward"] == 5
    assert result["stop_reason"] == "budget"
    assert not result["success"]


def test_saved_inputs_reproduce_digest_and_decision_trace(tmp_path):
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    result = run_episode(env, Policy(), seed=3, execute=3, horizon=4,
                         max_steps=5, action_space=base.action_space, capture_dir=tmp_path)
    with np.load(tmp_path / "episode_3_first_observation.npz", allow_pickle=False) as saved:
        assert saved["state.joint"].shape == (1, 1)
        np.testing.assert_array_equal(saved["state.joint"], [[0]])
        assert saved["task"].item() == "move"
    assert result["initial_observation_sha256"] == rollout(ToyEnv())["initial_observation_sha256"]
    assert [d["sim_step"] for d in result["decisions"]] == [0, 3]
    assert result["decisions"][1]["state"]["state.joint"] == [[3.0]]
    assert result["decisions"][0]["action_chunk"]["action.joint"] == [[10.0], [11.0], [12.0], [13.0]]


def test_extended_budget_keeps_reference_prefix_with_observer():
    reference = rollout(ToyEnv())
    base = ToyEnv()
    seen = []
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    extended = run_episode(env, Policy(), seed=3, execute=3, horizon=4,
        max_steps=10, action_space=base.action_space, reference_episode=reference,
        observer=lambda env, obs, step, success: seen.append(step))
    assert extended["reference_prefix_verified"]
    assert extended["decisions"][:2] == reference["decisions"]
    assert base.actions[:5] == [10, 11, 12, 20, 21]
    assert seen == list(range(11))


def test_reference_divergence_stops_before_applying_different_chunk():
    reference = rollout(ToyEnv())
    reference["decisions"][1]["action_chunk"]["action.joint"][0][0] += 1
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    with pytest.raises(ValueError, match="prefix differs"):
        run_episode(env, Policy(), seed=3, execute=3, horizon=4,
                    max_steps=10, action_space=base.action_space, reference_episode=reference)
    assert base.actions == [10, 11, 12]


def test_late_success_is_not_counted_at_earlier_budget():
    from grootN1_Robotics.baseline import budget_outcomes
    episodes = [rollout(ToyEnv(stop=7, success=True), budget=10), rollout(ToyEnv(), budget=10)]
    episodes[1]["seed"] = 4
    curve = budget_outcomes(episodes, [5, 10], 20)
    assert curve["5"]["success_rate"] == 0
    assert curve["10"]["success_rate"] == 0.5
    assert curve["10"]["successful_seeds"] == [3]
    assert curve["10"]["simulation_seconds"] == 0.5


def test_initial_comparison_allows_new_decisions_and_changed_success_step():
    class ActionSuccessEnv(ToyEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            success = self.actions[-1] >= 30
            return obs, reward, success, False, {**info, "success": success}
    reference = rollout(ActionSuccessEnv(), budget=10)
    base = ActionSuccessEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    changed = run_episode(env, Policy(), seed=3, execute=2, horizon=4,
        max_steps=10, action_space=base.action_space, reference_episode=reference,
        reference_mode="initial")
    assert reference["sim_steps"] == 7 and changed["sim_steps"] == 5
    assert changed["decisions"][0] == reference["decisions"][0]
    assert base.actions == [10, 11, 20, 21, 30]
    assert changed["reference_initial_decision_verified"]
    assert not changed["reference_prefix_verified"]


def test_initial_comparison_rejects_changed_first_prediction_before_execution():
    reference = rollout(ToyEnv())
    reference["decisions"][0]["action_chunk"]["action.joint"][0][0] += 1
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    with pytest.raises(ValueError, match="prefix differs"):
        run_episode(env, Policy(), seed=3, execute=2, horizon=4,
            max_steps=5, action_space=base.action_space, reference_episode=reference,
            reference_mode="initial")
    assert not base.actions


def test_execution_change_is_allowed_only_for_initial_reference_mode():
    from copy import deepcopy
    from grootN1_Robotics.tools.eval_baseline import validate_reference_provenance
    reference = {k: {} for k in ["env", "embodiment", "attention", "upstream_commit",
        "gr1_sim_commit", "checkpoint_sha256", "contract", "control_freq", "packages",
        "action_shapes", "asset_manifests", "accelerator", "code_sha256"]}
    reference["execute"] = 8
    changed = deepcopy(reference)
    changed["execute"] = 4
    validate_reference_provenance(changed, reference, "initial")
    with pytest.raises(ValueError, match="execute"):
        validate_reference_provenance(changed, reference, "prefix")


@pytest.mark.parametrize("key", ["checkpoint_sha256", "packages", "asset_manifests"])
def test_initial_reference_still_rejects_changed_provenance(key):
    from copy import deepcopy
    from grootN1_Robotics.tools.eval_baseline import validate_reference_provenance
    reference = {k: {} for k in ["env", "embodiment", "attention", "upstream_commit",
        "gr1_sim_commit", "checkpoint_sha256", "contract", "control_freq", "packages",
        "action_shapes", "asset_manifests", "accelerator", "code_sha256"]}
    changed = deepcopy(reference)
    changed[key] = {"changed": True}
    with pytest.raises(ValueError, match=key):
        validate_reference_provenance(changed, reference, "initial")


@pytest.mark.parametrize("truncate,success,reason", [
    (True, False, "truncated"), (False, False, "terminated"), (False, True, "success")])
def test_stop_mid_chunk(truncate, success, reason):
    base = ToyEnv(stop=2, truncate=truncate, success=success)
    result = rollout(base)
    assert base.actions == [10, 11]
    assert result["stop_reason"] == reason
    assert result["success"] == success


def test_nonfinite_never_reaches_simulator():
    class BadPolicy(Policy):
        def get_action(self, obs):
            return {"action.joint": np.full((1, 4, 1), np.nan, np.float32)}, {}
    base = ToyEnv()
    env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
    with pytest.raises(ValueError, match="non-finite"):
        run_episode(env, BadPolicy(), seed=0, execute=3, horizon=4,
                    max_steps=5, action_space=base.action_space)
    assert not base.actions


def test_wrong_embodiment_fails_before_loading_model(tmp_path):
    (tmp_path / "processor_config.json").write_text(
        '{"processor_kwargs": {"modality_configs": {"gr1": {}}}}')
    with pytest.raises(ValueError, match="does not declare"):
        checkpoint_contract(tmp_path, "libero_panda")


def test_contract_rejects_missing_camera_and_excess_commitment():
    configs = {m: SimpleNamespace(modality_keys=[key], delta_indices=indices)
               for m, key, indices in [("video", "camera", [0]), ("state", "joint", [0]),
                                        ("action", "joint", [0, 1, 2, 3]), ("language", "task", [0])]}
    validate_contract(ToyEnv(), configs, 3)
    with pytest.raises(ValueError, match="execute"):
        validate_contract(ToyEnv(), configs, 5)
    configs["video"].modality_keys = ["wrong_camera"]
    with pytest.raises(ValueError, match="interface mismatch"):
        validate_contract(ToyEnv(), configs, 3)


def test_official_sim_policy_keeps_language_and_action_mapping():
    from gr00t.data.types import ModalityConfig
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from grootN1_Robotics.baseline import batch_observation

    class NestedPolicy:
        modality_configs = {m: ModalityConfig(modality_keys=[key], delta_indices=indices)
                            for m, key, indices in [
                                ("video", "camera", [0]), ("state", "joint", [0]),
                                ("language", "task", [0]), ("action", "joint", [0, 1, 2, 3])]}

        def get_modality_config(self):
            return self.modality_configs

        def get_action(self, obs, options=None):
            assert obs["language"]["task"] == [["unlocked_waist: move"]]
            assert obs["state"]["joint"].shape == (1, 1, 1)
            return {"joint": np.full((1, 4, 1), 0.25, np.float32)}, {}

    observation = {"video.camera": np.zeros((1, 2, 2, 3), np.uint8),
                   "state.joint": np.zeros((1, 1), np.float32),
                   "annotation.human.coarse_action": "unlocked_waist: move"}
    actions, _ = Gr00tSimPolicyWrapper(NestedPolicy()).get_action(batch_observation(observation))
    assert set(actions) == {"action.joint"}
    np.testing.assert_array_equal(actions["action.joint"], np.full((1, 4, 1), 0.25, np.float32))

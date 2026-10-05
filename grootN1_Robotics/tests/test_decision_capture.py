"""Passive capture, exact stochastic replay and corruption/provenance gates."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grootN1_Robotics.decision_capture import (
    DecisionCapture, capture_rng, file_hash, load_capture, replay_decision,
    rng_equal, seed_rng,
)


class Encoder(torch.nn.Module):
    def forward(self, actions, timesteps, embodiment):
        return actions


class TinyFlowPolicy:
    def __init__(self, device="cpu"):
        self.device = device
        self.encoder = Encoder()
        self.policy = SimpleNamespace(model=SimpleNamespace(
            action_head=SimpleNamespace(action_encoder=self.encoder)))

    def reset(self):
        pass

    def get_action(self, observation):
        assert isinstance(observation["task"], (list, tuple))
        # Exercise all captured generators and the same upstream noise boundary.
        offset = random.random() + np.random.random() + torch.rand(1).item()
        noise = torch.randn((1, 4, 1), device=self.device)
        encoded = self.encoder(noise, torch.zeros(1, device=self.device), None)
        state = float(np.asarray(observation["state.joint"]).reshape(-1)[0])
        return {"action.joint": (encoded + offset + state).cpu().numpy().astype(np.float32)}, {}


def observation():
    return {"video.camera": np.arange(12, dtype=np.uint8).reshape(1, 1, 2, 2, 3),
            "state.joint": np.array([[[0.25]]], np.float32),
            "task": ["move 无损"]}


def capture(tmp_path, step=0, policy=None):
    policy = policy or TinyFlowPolicy()
    recorder = DecisionCapture(tmp_path, [step], policy.device, {"test": "tiny"})
    seed_rng(37, policy.device)
    actions, _ = recorder.predict(policy, observation(), seed=3, sim_step=step, env=None)
    return policy, recorder, actions, tmp_path / "episode_3" / f"step_{step}"


def test_recording_is_lossless_and_does_not_change_action_or_rng(tmp_path):
    policy = TinyFlowPolicy()
    seed_rng(37, "cpu")
    expected, _ = policy.get_action(observation())
    expected_rng = capture_rng("cpu")
    _, _, actual, path = capture(tmp_path, policy=policy)
    assert rng_equal(capture_rng("cpu"), expected_rng)
    np.testing.assert_array_equal(actual["action.joint"], expected["action.joint"])
    manifest, inputs, actions, saved = load_capture(path)
    for key, value in observation().items():
        np.testing.assert_array_equal(inputs[key], np.asarray(value))
        assert np.asarray(inputs[key]).dtype == np.asarray(value).dtype
    assert isinstance(inputs["task"], list)
    assert manifest["sim_step"] == 0
    assert saved["initial_noise"].shape == (1, 4, 1)
    assert rng_equal(saved["after"], expected_rng)
    np.testing.assert_array_equal(actions["action.joint"], expected["action.joint"])
    assert not policy.encoder._forward_pre_hooks


def test_exact_replay_and_alternatives_leave_callers_rng_untouched(tmp_path):
    policy, _, _, path = capture(tmp_path)
    seed_rng(999, "cpu")
    before = capture_rng("cpu")
    result = replay_decision(policy, path, "cpu", candidates=3)
    assert result["exact_replay_verified"]
    assert not result["candidate_quality_evaluated"]
    assert rng_equal(capture_rng("cpu"), before)
    assert len(result["candidate_differences"]) == 3
    assert all(row["groups"]["action.joint"]["rms_difference"] > 0 for row in result["candidate_differences"])
    assert result == replay_decision(policy, path, "cpu", candidates=3)


def test_unselected_decision_is_not_saved(tmp_path):
    recorder = DecisionCapture(tmp_path, [8], "cpu", {})
    recorder.predict(TinyFlowPolicy(), observation(), seed=0, sim_step=0, env=None)
    assert not recorder.records and not list(tmp_path.iterdir())


def test_alternative_action_chunks_are_saved_losslessly(tmp_path):
    policy, _, _, path = capture(tmp_path / "reference")
    target = tmp_path / "alternatives"
    result = replay_decision(policy, path, "cpu", candidates=2, candidate_dir=target)
    _, _, expected, _ = load_capture(path)
    for row in result["candidate_differences"]:
        file = target / row["actions_file"]
        assert row["actions_sha256"] == file_hash(file)
        with np.load(file, allow_pickle=False) as saved:
            delta = saved["action.joint"].astype(np.float64) - expected["action.joint"]
            assert row["groups"]["action.joint"]["rms_difference"] == float(np.sqrt(np.mean(delta**2)))


def test_cuda_seed_initializes_generators_before_indexing(monkeypatch):
    # Exercise cold-start ordering on CPU; the numerical oracle below uses real CUDA.
    calls = []
    generator = SimpleNamespace(manual_seed=lambda seed: calls.append(seed))
    monkeypatch.setattr(torch.cuda, "default_generators", ())
    def initialize():
        calls.append("initialize")
        monkeypatch.setattr(torch.cuda, "default_generators", (generator,))
    monkeypatch.setattr(torch.cuda, "init", initialize)
    seed_rng(37, "cuda:0")
    assert calls == ["initialize", 37]


def test_refuses_to_overwrite_capture(tmp_path):
    policy, recorder, _, _ = capture(tmp_path)
    before = capture_rng("cpu")
    with pytest.raises(FileExistsError):
        recorder.predict(policy, observation(), seed=3, sim_step=0, env=None)
    assert rng_equal(capture_rng("cpu"), before)


@pytest.mark.parametrize("name", ["observation.npz", "actions.npz", "rng.pt"])
def test_corrupted_artifact_is_rejected(tmp_path, name):
    _, _, _, path = capture(tmp_path)
    with (path / name).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash differs"):
        load_capture(path)


def test_changed_policy_fails_exact_replay_and_restores_rng(tmp_path):
    policy, _, _, path = capture(tmp_path)
    original = policy.get_action
    def changed(obs):
        result, info = original(obs)
        result["action.joint"] += 1
        return result, info
    policy.get_action = changed
    before = capture_rng("cpu")
    with pytest.raises(ValueError, match="action chunk differs"):
        replay_decision(policy, path, "cpu")
    assert rng_equal(capture_rng("cpu"), before)
    assert not policy.encoder._forward_pre_hooks


def test_missing_noise_boundary_fails_and_removes_hook(tmp_path):
    policy = TinyFlowPolicy()
    policy.get_action = lambda obs: ({"action.joint": np.ones((1, 4, 1), np.float32)}, {})
    with pytest.raises(ValueError, match="noise was not captured"):
        capture(tmp_path, policy=policy)
    assert not policy.encoder._forward_pre_hooks


def test_object_inputs_are_rejected_before_prediction(tmp_path):
    recorder = DecisionCapture(tmp_path, [0], "cpu", {})
    bad = observation()
    bad["task"] = np.array([object()], dtype=object)
    with pytest.raises(ValueError, match="object arrays"):
        recorder.predict(TinyFlowPolicy(), bad, seed=0, sim_step=0, env=None)


def test_capture_matches_completed_rollout_inventory_and_trace(tmp_path):
    from grootN1_Robotics.tools.replay_decisions import validate_capture_rollout
    _, _, _, path = capture(tmp_path)
    manifest, inputs, actions, _ = load_capture(path)
    report = {"status": "complete", "test": "tiny", "episodes": [{"seed": 3,
        "decision_captures": ["episode_3/step_0"], "decisions": [{"sim_step": 0,
            "state": {"state.joint": [[0.25]]},
            "action_chunk": {k: v[0].tolist() for k, v in actions.items()}}]}]}
    validate_capture_rollout(path, manifest, inputs, actions, report)
    report["episodes"][0]["decisions"][0]["state"]["state.joint"][0][0] += 1
    with pytest.raises(ValueError, match="state differs"):
        validate_capture_rollout(path, manifest, inputs, actions, report)


def test_capture_during_rollout_keeps_every_reference_decision(tmp_path):
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from grootN1_Robotics.baseline import run_episode
    from grootN1_Robotics.tests.test_baseline import ToyEnv
    def run(recorder=None, reference=None):
        seed_rng(55, "cpu")
        base = ToyEnv()
        env = MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=1)
        return run_episode(env, TinyFlowPolicy(), seed=3, execute=2, horizon=4,
            max_steps=6, action_space=base.action_space,
            decision_capture=recorder, reference_episode=reference)
    reference = run()
    recorder = DecisionCapture(tmp_path, [0, 4], "cpu", {})
    result = run(recorder, reference)
    assert result["reference_prefix_verified"]
    assert result["decisions"] == reference["decisions"]
    assert recorder.records == ["episode_3/step_0", "episode_3/step_4"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only RNG/noise numerical oracle")
def test_cuda_capture_replay_preserves_policy_and_other_device_rng(tmp_path):
    policy = TinyFlowPolicy("cuda:0")
    _, _, _, path = capture(tmp_path, policy=policy)
    before = capture_rng("cuda:0")
    other = torch.cuda.get_rng_state(1) if torch.cuda.device_count() > 1 else None
    replay_decision(policy, path, "cuda:0", candidates=2)
    assert rng_equal(capture_rng("cuda:0"), before)
    if other is not None:
        assert torch.equal(other, torch.cuda.get_rng_state(1))

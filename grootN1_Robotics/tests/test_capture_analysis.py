"""Independent passive-capture gates reject missing records and trajectory changes."""
from copy import deepcopy

import pytest

from grootN1_Robotics.tools.analyze_decision_captures import SEEDS, STEPS, verify_run


def reports():
    reference = {key: {} for key in ["env", "embodiment", "attention", "upstream_commit",
        "gr1_sim_commit", "checkpoint_sha256", "contract", "control_freq", "packages",
        "action_shapes", "asset_manifests", "accelerator", "code_sha256"]}
    reference.update(status="complete", execute=8, max_steps=1440, episodes=[])
    for seed in range(10):
        steps = {3: 284, 5: 283}.get(seed, 1440)
        reference["episodes"].append({"seed": seed, "success": seed in [3, 5],
            "sim_steps": steps, "stop_reason": "success" if seed in [3, 5] else "budget",
            "initial_observation_sha256": str(seed), "decisions": [{"sim_step": 0}],
            "policy_calls": (steps + 7)//8})
    capture = deepcopy(reference)
    capture.update(reference_mode="prefix", diagnostic_seeds=SEEDS,
                   requested_episodes=6, decision_capture={"steps": STEPS})
    capture["episodes"] = [e for e in capture["episodes"] if e["seed"] in SEEDS]
    replay = {"status": "complete", "candidate_quality_evaluated": False, "decisions": []}
    for episode in capture["episodes"]:
        episode["reference_prefix_verified"] = True
        episode["decision_captures"] = [f"episode_{episode['seed']}/step_{step}"
                                        for step in STEPS if step < episode["sim_steps"]]
        for step in STEPS:
            if step < episode["sim_steps"]:
                replay["decisions"].append({"seed": episode["seed"], "sim_step": step,
                    "exact_replay_verified": True, "candidate_quality_evaluated": False,
                    "candidate_differences": [{}, {}, {}, {}]})
    return reference, capture, replay


def test_complete_fixed_capture_inventory():
    assert len(verify_run(*reports())) == 46


@pytest.mark.parametrize("key", ["decisions", "initial_observation_sha256", "sim_steps", "success", "policy_calls"])
def test_recording_must_not_change_reference(key):
    reference, capture, replay = reports()
    capture["episodes"][0][key] = "changed"
    with pytest.raises(ValueError, match="reference"):
        verify_run(reference, capture, replay)


def test_missing_capture_is_rejected():
    reference, capture, replay = reports()
    capture["episodes"][0]["decision_captures"].pop()
    with pytest.raises(ValueError, match="capture decision inventory"):
        verify_run(reference, capture, replay)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_replay_is_rejected(duplicate):
    reference, capture, replay = reports()
    if duplicate:
        replay["decisions"].append(replay["decisions"][0])
    else:
        replay["decisions"].pop()
    with pytest.raises(ValueError, match="replay decision inventory"):
        verify_run(reference, capture, replay)


def test_prediction_variation_cannot_be_marked_as_quality():
    reference, capture, replay = reports()
    replay["candidate_quality_evaluated"] = True
    with pytest.raises(ValueError, match="candidate scope"):
        verify_run(reference, capture, replay)


def test_nonexact_replay_is_rejected():
    reference, capture, replay = reports()
    replay["decisions"][0]["exact_replay_verified"] = False
    with pytest.raises(ValueError, match="replay verification"):
        verify_run(reference, capture, replay)


def test_capture_provenance_must_match_control():
    reference, capture, replay = reports()
    capture["packages"] = {"torch": "changed"}
    with pytest.raises(ValueError, match="packages"):
        verify_run(reference, capture, replay)

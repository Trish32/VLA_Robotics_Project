from copy import deepcopy
import math

import pytest

from grootN1_Robotics.tools.analyze_execution_scan import verify_execution_pair, deadline_changes


def episode(seed, execute, success=False, steps=1440):
    return {"seed": seed, "success": success, "sim_steps": steps,
        "stop_reason": "success" if success else "budget",
        "policy_calls": math.ceil(steps / execute), "initial_observation_sha256": str(seed),
        "decisions": [{"sim_step": step, "state": {"joint": [step]},
            "action_chunk": {"joint": [[seed]]}} for step in range(0, steps, execute)]}


def pair():
    reference = {key: {} for key in ["env", "embodiment", "checkpoint_sha256", "upstream_commit",
        "gr1_sim_commit", "contract", "packages", "action_shapes", "asset_manifests", "accelerator", "attention"]}
    reference.update(status="complete", execute=8, max_steps=1440, seed=0,
        requested_episodes=10, budget_scan=[360, 720, 1080, 1440], control_freq=20,
        selection="none", purpose="task_baseline",
        code_sha256={"grootN1_Robotics/policy.py": "policy", "patches/0001.patch": "patch"},
        episodes=[episode(seed, 8, seed==0, 8 if seed==0 else 1440) for seed in range(10)])
    scan = deepcopy(reference)
    scan.update(execute=4, reference_mode="initial",
        episodes=[{**episode(seed, 4, seed==1, 900 if seed==1 else 1440),
            "reference_initial_decision_verified": True, "reference_prefix_verified": False}
            for seed in range(10)])
    return reference, scan


def test_changed_trajectories_and_success_regression_remain_valid():
    reference, scan = pair()
    verify_execution_pair(reference, scan)
    assert deadline_changes(reference, scan, 720)["regressed_seeds"] == [0]
    assert deadline_changes(reference, scan, 720)["improved_seeds"] == []
    later = deadline_changes(reference, scan, 1440)
    assert later["improved_seeds"] == [1] and later["regressed_seeds"] == [0]
    assert later["success_rate_difference"] == 0


@pytest.mark.parametrize("change", ["first_prediction", "initial_observation", "checkpoint",
    "policy", "patch", "duplicate_seed", "horizon", "budget", "early_stop", "call_count", "decision_steps", "gate"])
def test_invalid_execution_comparison_is_rejected(change):
    reference, scan = pair()
    e = scan["episodes"][0]
    if change == "first_prediction":
        e["decisions"][0]["action_chunk"]["joint"][0][0] += 1
    elif change == "initial_observation":
        e["initial_observation_sha256"] = "different"
    elif change == "checkpoint":
        scan["checkpoint_sha256"] = {"weights": "changed"}
    elif change in {"policy", "patch"}:
        path = next(k for k in scan["code_sha256"] if k.endswith("policy.py" if change=="policy" else ".patch"))
        scan["code_sha256"][path] = "different"
    elif change == "duplicate_seed":
        e["seed"] = 1
    elif change == "horizon":
        scan["execute"] = 2
    elif change == "budget":
        scan["max_steps"] = 720
    elif change == "early_stop":
        e.update(sim_steps=100, stop_reason="terminated")
    elif change == "call_count":
        e["policy_calls"] -= 1
    elif change == "decision_steps":
        e["decisions"][1]["sim_step"] += 1
    else:
        e["reference_prefix_verified"] = True
    with pytest.raises(ValueError):
        verify_execution_pair(reference, scan)

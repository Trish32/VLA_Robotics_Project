from copy import deepcopy

import pytest

from grootN1_Robotics.tools.analyze_budget_scan import verify_pair, verify_diagnostics
from grootN1_Robotics.diagnostics import EpisodeRecorder
from grootN1_Robotics.tests.test_diagnostics import signals
import numpy as np


def pair():
    reference = {key: {} for key in ["env", "embodiment", "execute", "checkpoint_sha256",
        "upstream_commit", "gr1_sim_commit", "contract", "control_freq", "packages",
        "action_shapes", "asset_manifests", "accelerator", "attention"]}
    reference.update(status="complete", max_steps=720, episodes=[{
        "seed": 0, "success": False, "sim_steps": 720,
        "initial_observation_sha256": "same", "decisions": [{"sim_step": 0, "action": 1}]}])
    reference["code_sha256"] = {"grootN1_Robotics/policy.py": "policy",
                                "grootN1_Robotics/patches/0001.patch": "patch"}
    scan = deepcopy(reference)
    scan["episodes"][0].update(reference_prefix_verified=True, success=True, sim_steps=900)
    scan["episodes"][0]["decisions"].append({"sim_step": 720, "action": 2})
    return reference, scan


def test_later_success_preserves_old_prefix():
    verify_pair(*pair())


@pytest.mark.parametrize("change", ["prefix", "early_success", "seed", "provenance",
                                    "policy", "patch", "short_rollout"])
def test_mismatched_pair_is_rejected(change):
    reference, scan = pair()
    if change == "prefix":
        scan["episodes"][0]["decisions"][0]["action"] = 2
    elif change == "early_success":
        scan["episodes"][0]["sim_steps"] = 700
    elif change == "seed":
        scan["episodes"][0]["seed"] = 1
    elif change == "provenance":
        scan["checkpoint_sha256"] = {"weights": "changed"}
    elif change in {"policy", "patch"}:
        suffix = "policy.py" if change == "policy" else ".patch"
        key = next(k for k in scan["code_sha256"] if k.endswith(suffix))
        scan["code_sha256"][key] = "changed"
    else:
        scan["episodes"][0].update(success=False, sim_steps=719)
    with pytest.raises(ValueError):
        verify_pair(reference, scan)


def test_video_stage_alignment_is_verified(tmp_path):
    recorder = EpisodeRecorder(tmp_path, 0, 20, reader=signals)
    try:
        for step in range(3):
            recorder(None, {"video.camera": np.zeros((1, 16, 16, 3), np.uint8)}, step, False)
    finally:
        recorder.close()
    episode = {"sim_steps": 2, "success": False, "diagnostics": {
        "stages": recorder.stages_path.name, "video": recorder.video_path.name}}
    assert len(verify_diagnostics(episode, tmp_path, 20)) == 3
    with pytest.raises(ValueError, match="missing or duplicated"):
        verify_diagnostics({**episode, "sim_steps": 3}, tmp_path, 20)

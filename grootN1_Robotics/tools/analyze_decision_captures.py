"""Independently verify passive capture, exact reference trajectories and replay inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEEDS = [0, 2, 3, 5, 7, 8]
STEPS = [0, 96, 160, 224, 256, 360, 720, 1080, 1432]


def verify_run(reference, capture, replay):
    from grootN1_Robotics.tools.eval_baseline import validate_reference_provenance
    if any(r.get("status") != "complete" for r in [reference, capture, replay]):
        raise ValueError("all reports must be complete")
    validate_reference_provenance(capture, reference, "prefix")
    if (capture["execute"] != 8 or capture["max_steps"] != 1440
            or capture.get("reference_mode") != "prefix"
            or capture.get("diagnostic_seeds") != SEEDS
            or [e["seed"] for e in capture["episodes"]] != SEEDS
            or capture.get("requested_episodes") != len(SEEDS)
            or capture.get("decision_capture", {}).get("steps") != STEPS):
        raise ValueError("capture design differs")
    expected = set()
    for episode in capture["episodes"]:
        old = next(e for e in reference["episodes"] if e["seed"] == episode["seed"])
        for key in ["success", "sim_steps", "stop_reason", "initial_observation_sha256", "decisions", "policy_calls"]:
            if episode[key] != old[key]:
                raise ValueError(f"capture changes reference {key}: seed {episode['seed']}")
        if not episode.get("reference_prefix_verified"):
            raise ValueError("capture did not verify the complete reference prefix")
        inventory = [f"episode_{episode['seed']}/step_{step}" for step in STEPS if step < episode["sim_steps"]]
        if episode.get("decision_captures") != inventory:
            raise ValueError("capture decision inventory differs")
        expected.update((episode["seed"], step) for step in STEPS if step < episode["sim_steps"])
    actual = [(r["seed"], r["sim_step"]) for r in replay["decisions"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("replay decision inventory differs")
    if replay.get("candidate_quality_evaluated") is not False or any(
            not r.get("exact_replay_verified") or r.get("candidate_quality_evaluated") is not False
            or len(r.get("candidate_differences", [])) != 4 for r in replay["decisions"]):
        raise ValueError("replay verification or candidate scope differs")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from grootN1_Robotics.decision_capture import load_capture, file_hash
    from grootN1_Robotics.tools.replay_decisions import validate_capture_rollout
    from grootN1_Robotics.tools.analyze_budget_scan import verify_diagnostics
    reference = json.loads(args.reference.read_text())
    rollout_path = args.artifacts / "gr1_baseline.json"
    capture = json.loads(rollout_path.read_text())
    replay = json.loads((args.artifacts / "decision_replay.json").read_text())
    if capture.get("reference_report_sha256") != file_hash(args.reference):
        raise ValueError("capture reference hash differs")
    if replay.get("rollout_report_sha256") != file_hash(rollout_path):
        raise ValueError("replay rollout hash differs")
    expected = verify_run(reference, capture, replay)
    paths = list((args.artifacts / "decisions").glob("episode_*/step_*/manifest.json"))
    actual = set()
    diagnostics = {e["seed"]: verify_diagnostics(e, args.artifacts / "diagnostics", capture["control_freq"])
                   for e in capture["episodes"]}
    stages = {"before_contact": 0, "contact_without_placement": 0,
              "inside_without_closure": 0, "after_contact_outside_drawer": 0}
    for path in paths:
        manifest, observation, actions, rng = load_capture(path.parent)
        validate_capture_rollout(path.parent, manifest, observation, actions, capture)
        if replay.get("provenance") != manifest["provenance"]:
            raise ValueError("replay/capture provenance differs")
        seed, step = manifest["seed"], manifest["sim_step"]
        actual.add((seed, step))
        row = diagnostics[seed][step]
        labels = {k: row[k] for k in manifest["labels"]}
        if manifest["labels"] != labels:
            raise ValueError("capture labels disagree with diagnostic step")
        result = next(r for r in replay["decisions"] if (r["seed"], r["sim_step"]) == (seed, step))
        if result["labels"] != manifest["labels"]:
            raise ValueError("replay labels differ")
        import numpy as np
        for index, candidate in enumerate(result["candidate_differences"]):
            name = f"candidate_{index}.npz"
            file = args.artifacts / "candidate_predictions" / path.parent.relative_to(args.artifacts / "decisions") / name
            identity = f"candidate-v1:{seed}:{step}:{index}"
            candidate_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            if candidate.get("actions_file") != name or candidate.get("actions_sha256") != file_hash(file) or candidate["seed"] != candidate_seed:
                raise ValueError("candidate identity or action hash differs")
            with np.load(file, allow_pickle=False) as predictions:
                if set(predictions.files) != set(actions):
                    raise ValueError("candidate action keys differ")
                groups = {}
                for key, value in actions.items():
                    proposed = predictions[key]
                    if proposed.shape != value.shape or proposed.dtype != value.dtype or not np.isfinite(proposed).all():
                        raise ValueError("candidate action contract differs")
                    delta = proposed.astype(np.float64) - value
                    groups[key] = {"rms_difference": float(np.sqrt(np.mean(delta**2))),
                                   "max_abs_difference": float(np.abs(delta).max())}
                if groups != candidate["groups"]:
                    raise ValueError("candidate differences disagree with saved actions")
        if labels["object_inside_drawer"]:
            stages["inside_without_closure"] += 1
        elif any(labels["grasp_contact"].values()):
            stages["contact_without_placement"] += 1
        elif not any(any(r["grasp_contact"].values()) for r in diagnostics[seed][:step + 1]):
            stages["before_contact"] += 1
        else:
            stages["after_contact_outside_drawer"] += 1
        import torch
        if not torch.isfinite(rng["initial_noise"]).all():
            raise ValueError("nonfinite captured flow noise")
    if actual != expected or len(paths) != len(expected):
        raise ValueError("lossless capture files differ from expected inventory")
    summary = {"reference_trajectories_unchanged": True, "exact_replays": len(expected),
               "alternative_predictions": len(expected) * 4,
               "candidate_quality_evaluated": False, "captured_stages": stages,
               "video_frames_verified": sum(len(rows) for rows in diagnostics.values()),
               "seeds": SEEDS, "capture_steps": STEPS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

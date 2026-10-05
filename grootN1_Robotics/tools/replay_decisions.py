"""Verify captured GR00T decisions and describe candidate variation on identical inputs."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_capture_rollout(directory, manifest, observation, actions, report):
    """Link each artifact to the completed rollout, not just its own manifest."""
    from grootN1_Robotics.baseline import batch_observation
    if report.get("status") != "complete":
        raise ValueError("capture rollout must be complete")
    for key, value in manifest["provenance"].items():
        if report.get(key) != value:
            raise ValueError(f"capture/rollout provenance differs: {key}")
    episode = next((e for e in report["episodes"] if e["seed"] == manifest["seed"]), None)
    decision = next((d for d in episode["decisions"] if d["sim_step"] == manifest["sim_step"]), None) if episode else None
    relative = f"episode_{manifest['seed']}/step_{manifest['sim_step']}"
    if decision is None or relative not in episode.get("decision_captures", []):
        raise ValueError("capture is not a recorded rollout decision")
    # Capture stores the batched policy input; rollout trace stores unbatched state.
    expected_state = batch_observation(decision["state"])
    actual_state = {k: v for k, v in observation.items() if k.startswith("state.")}
    if set(actual_state) != set(expected_state) or any(
            not np.array_equal(actual_state[k], expected_state[k]) for k in actual_state):
        raise ValueError("capture state differs from rollout")
    if set(actions) != set(decision["action_chunk"]) or any(
            not np.array_equal(actions[k][0], np.asarray(decision["action_chunk"][k], np.float32)) for k in actions):
        raise ValueError("capture actions differ from rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--candidate-dir", type=Path, help="save alternative action chunks losslessly")
    parser.add_argument("--smoke", action="store_true", help="local gate: replay only one initial decision")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.candidates < 0:
        parser.error("output must be new and candidate count nonnegative")
    from common.device import pick_device, describe
    from grootN1_Robotics.decision_capture import load_capture, replay_decision
    from grootN1_Robotics.tools.eval_baseline import save_report, sha256
    accelerator = pick_device(args.device)
    if not accelerator.is_cuda and not args.smoke:
        parser.error("CPU/MPS replay is a smoke test; use --smoke")
    rollout = json.loads(args.rollout_report.read_text())
    directories = sorted(path.parent for path in args.capture_dir.glob("episode_*/step_*/manifest.json"))
    expected = {p for e in rollout["episodes"] for p in e.get("decision_captures", [])}
    if not directories or {str(p.relative_to(args.capture_dir)) for p in directories} != expected:
        parser.error("capture files do not match rollout inventory")
    if args.smoke:
        directories = directories[:1]
    provenance = None
    for directory in directories:
        manifest, observation, actions, _ = load_capture(directory)
        validate_capture_rollout(directory, manifest, observation, actions, rollout)
        if provenance is not None and manifest["provenance"] != provenance:
            parser.error("mixed capture provenance")
        provenance = manifest["provenance"]
    if args.smoke and (len(directories) != 1 or manifest["sim_step"] != 0):
        parser.error("local smoke requires one initial decision")
    if provenance["accelerator"] != describe(accelerator):
        parser.error("exact replay requires the recorded accelerator and dtype")
    hashes = {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir())
              if p.suffix in {".json", ".safetensors"}}
    if hashes != provenance["checkpoint_sha256"]:
        parser.error("replay checkpoint differs")
    upstream = ROOT / "grootN1_Robotics/upstream"
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != provenance["upstream_commit"]:
        parser.error("replay upstream commit differs")
    for key, value in provenance["code_sha256"].items():
        if key.endswith("policy.py") or key.endswith(".patch") or key.endswith("decision_capture.py"):
            if sha256(ROOT / key) != value:
                parser.error(f"replay source differs: {key}")
    for package, expected_version in provenance["packages"].items():
        if version(package) != expected_version:
            parser.error(f"replay package differs: {package}")
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from grootN1_Robotics.policy import LocalGr00tPolicy
    policy = Gr00tSimPolicyWrapper(LocalGr00tPolicy(
        EmbodimentTag(provenance["embodiment"]), args.checkpoint,
        device=str(accelerator.device), dtype=accelerator.amp_dtype or torch.float32))
    report = {"status": "running", "purpose": "local_smoke" if args.smoke else "decision_replay",
              "rollout_report_sha256": sha256(args.rollout_report), "provenance": provenance,
              "candidate_quality_evaluated": False, "decisions": []}
    save_report(args.out, report)
    try:
        for directory in directories:
            target = args.candidate_dir / directory.relative_to(args.capture_dir) if args.candidate_dir else None
            result = replay_decision(policy, directory, accelerator.device, args.candidates, target)
            report["decisions"].append(result)
            save_report(args.out, report)
            print(json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_report(args.out, report)


if __name__ == "__main__":
    main()

"""Gate cold simulator reconstruction against captured inputs and continuous controls.

Run one continuous action replay and one fresh replay paused at captured decisions.
No checkpoint allocation, policy sampling, selection or candidate quality claims.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_simulator_provenance(rollout):
    for name in ["torch", "numpy", "gymnasium", "robosuite", "robocasa", "mujoco"]:
        if version(name) != rollout["packages"][name]:
            raise ValueError(f"simulator replay package differs: {name}")
    upstream = ROOT / "grootN1_Robotics/upstream"
    sim = upstream / "external_dependencies/robocasa-gr1-tabletop-tasks"
    for path, key in [(upstream, "upstream_commit"), (sim, "gr1_sim_commit")]:
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != rollout[key]:
            raise ValueError(f"simulator replay source differs: {key}")
    manifests = {p.name: json.loads(p.read_text()) for p in sorted(
        (sim / "robocasa/models/assets").glob(".*.manifest.json"))}
    if manifests != rollout["asset_manifests"]:
        raise ValueError("simulator replay asset manifests differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="only a recorded two-step local smoke")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing replay report")
    rollout = json.loads(args.rollout_report.read_text())
    if rollout.get("status") != "complete" or rollout.get("execute") != 8:
        parser.error("requires a completed execute-8 control")
    if args.smoke:
        if (rollout.get("purpose") != "local_smoke" or rollout["max_steps"] != 2
                or len(rollout["episodes"]) != 1 or rollout["episodes"][0]["sim_steps"] != 2):
            parser.error("local branch gate requires the real two-step smoke")
    else:
        from common.device import pick_device
        if not pick_device("cuda").is_cuda:
            parser.error("full simulator replay runs on the cloud; use --smoke locally")
    validate_simulator_provenance(rollout)
    from grootN1_Robotics.branch_replay import RecordedReplay
    from grootN1_Robotics.decision_capture import load_capture, file_hash
    from grootN1_Robotics.diagnostics import drawer_signals
    from grootN1_Robotics.tools.eval_baseline import create_seeded_env, save_report
    from grootN1_Robotics.tools.replay_decisions import validate_capture_rollout
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from types import SimpleNamespace
    import torch

    configs = {k: SimpleNamespace(**v) for k, v in rollout["contract"].items()}
    captures = {}
    actual = set()
    for manifest_path in args.capture_dir.glob("episode_*/step_*/manifest.json"):
        manifest, observation, actions, _ = load_capture(manifest_path.parent)
        validate_capture_rollout(manifest_path.parent, manifest, observation, actions, rollout)
        identity = (manifest["seed"], manifest["sim_step"])
        if identity in actual:
            raise ValueError("duplicate simulator replay capture")
        actual.add(identity)
        captures[identity] = (observation, manifest["labels"])
    expected = {tuple(map(int, (name.split('/')[0].removeprefix('episode_'),
                               name.split('/')[1].removeprefix('step_'))))
                for e in rollout["episodes"] for name in e.get("decision_captures", [])}
    if not expected or actual != expected:
        parser.error("capture inventory differs from the recorded control")
    report = {"status": "running", "purpose": "local_smoke" if args.smoke else "branch_replay_gate",
              "restoration": "fresh_seeded_environment_and_recorded_action_prefix",
              "candidate_quality_evaluated": False, "published_metric_reproduction": False,
              "rollout_report_sha256": file_hash(args.rollout_report), "episodes": [],
              "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                [Path(__file__), ROOT / "grootN1_Robotics/branch_replay.py"]},
              "simulator_provenance": {k: rollout[k] for k in
                ["env", "execute", "contract", "packages", "upstream_commit", "gr1_sim_commit", "asset_manifests"]}}
    save_report(args.out, report)
    try:
        for episode in rollout["episodes"]:
            seed = episode["seed"]
            diagnostic = args.capture_dir.parent / "diagnostics" / episode["diagnostics"]["stages"]
            rows = [json.loads(line) for line in diagnostic.read_text().splitlines()]
            if [r["sim_step"] for r in rows] != list(range(episode["sim_steps"] + 1)):
                raise ValueError("reference diagnostic step inventory differs")
            label_keys = next(labels.keys() for (s, _), (_, labels) in captures.items() if s == seed)
            recorded_labels = {r["sim_step"]: {k: r[k] for k in label_keys} for r in rows}
            observations = {step: obs for (s, step), (obs, _) in captures.items() if s == seed}
            if not observations:
                raise ValueError("episode has no recorded capture")
            traces, trace_files = [], []
            pause_steps = [1] if args.smoke else sorted(observations)
            for pass_index in range(2):
                # Fresh constructor rebuilds independently seeded robosuite generators.
                env = create_seeded_env(rollout["env"], seed)
                try:
                    action_space = env.action_space
                    task = env.unwrapped.env
                    if task.control_freq != rollout["control_freq"] or {
                            k: list(v.shape) for k, v in action_space.spaces.items()} != rollout["action_shapes"]:
                        raise ValueError("simulator frequency or action interface differs")
                    env = MultiStepWrapper(env, np.asarray(configs["video"].delta_indices),
                        np.asarray(configs["state"].delta_indices), n_action_steps=1, max_episode_steps=None)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    replay = RecordedReplay(env, episode, execute=rollout["execute"],
                        horizon=len(configs["action"].delta_indices), action_space=action_space,
                        label_reader=drawer_signals, captures=observations,
                        expected_trace=traces[0] if pass_index else None, recorded_labels=recorded_labels)
                    if pass_index:
                        for step in pause_steps:
                            replay.advance(step)
                    replay.advance(episode["sim_steps"])
                    for step in observations:
                        if replay.trace[step]["labels"] != captures[(seed, step)][1]:
                            raise ValueError(f"captured task labels differ: seed {seed} step {step}")
                    trace_path = args.out.parent / f"{args.out.stem}_seed_{seed}_pass_{pass_index}.json"
                    if trace_path.exists():
                        raise FileExistsError(trace_path)
                    save_report(trace_path, {"trace": replay.trace})
                    trace_files.append({"file": trace_path.name, "sha256": file_hash(trace_path)})
                    traces.append(replay.trace)
                    print(f"seed {seed} pass {pass_index}: {replay.step} steps verified", flush=True)
                finally:
                    env.close()
            report["episodes"].append({"seed": seed, "sim_steps": episode["sim_steps"],
                "captured_observations_verified": len(observations), "pause_steps": pause_steps,
                "exact_step_traces": len(traces[0]), "success": episode["success"],
                "stop_reason": episode["stop_reason"], "trace_files": trace_files,
                "reference_diagnostics_sha256": file_hash(diagnostic),
                "exact_reconstruction_verified": True})
            save_report(args.out, report)
        report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_report(args.out, report)


if __name__ == "__main__":
    main()

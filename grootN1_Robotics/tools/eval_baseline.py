"""Evaluate the actual GR00T policy on its official simulator, without planning.

Run with PYTHONPATH=grootN1_Robotics/upstream from the workspace root.
Use --preflight to validate checkpoint embodiment before loading model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def checkpoint_contract(checkpoint: Path, embodiment: str):
    data = json.loads((checkpoint / "processor_config.json").read_text())
    configs = data["processor_kwargs"]["modality_configs"]
    if embodiment not in configs:
        raise ValueError(f"checkpoint does not declare {embodiment}; available: {sorted(configs)}")
    return configs[embodiment]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def create_seeded_env(name, seed):
    # The official reset(seed) only reseeds legacy np.random; object/layout
    # sampling uses a default_rng created in the robosuite constructor.
    # Construct each episode with its seed, keeping all upstream mappings.
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    return gym.make(name, enable_render=True, seed=seed)


def validate_reference_provenance(report, reference, mode):
    keys = ["env", "embodiment", "attention", "upstream_commit", "gr1_sim_commit",
            "checkpoint_sha256", "contract", "control_freq", "packages", "action_shapes",
            "asset_manifests", "accelerator"]
    if mode == "prefix":
        keys.append("execute")
    elif mode != "initial":
        raise ValueError("invalid reference mode")
    for key in keys:
        if report[key] != reference[key]:
            raise ValueError(f"reference provenance differs: {key}")
    for key, value in reference["code_sha256"].items():
        if key.endswith("policy.py") or key.endswith(".patch"):
            if report["code_sha256"].get(key) != value:
                raise ValueError(f"reference policy/patch source differs: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env", default="gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="explicit diagnostic seeds instead of a consecutive episode range")
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--out", type=Path, default=ROOT / "grootN1_Robotics/data/baseline.json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one episode, two simulator steps; never report task success rate")
    parser.add_argument("--capture-dir", type=Path,
                        help="save first episode observations for same-input activation comparisons")
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="record existing camera frames and read-only drawer task signals")
    parser.add_argument("--decision-capture-dir", type=Path,
                        help="lossless selected policy inputs, RNG and actual flow noise")
    parser.add_argument("--decision-capture-steps", nargs="+", type=int,
                        default=[0, 96, 160, 224, 256, 360, 720, 1080, 1432])
    parser.add_argument("--reference-report", type=Path,
                        help="require exact initial observations and state/action prefixes from this report")
    parser.add_argument("--reference-mode", choices=["prefix", "initial"], default="prefix",
                        help="initial compares the first decision only, for execution-horizon contrasts")
    parser.add_argument("--budgets", nargs="+", type=int, default=[],
                        help="measure success by these simulator-step budgets within one longer rollout")
    args = parser.parse_args()
    if args.smoke:
        args.episodes, args.max_steps = 1, 2
        if args.seeds:
            parser.error("--seeds is for CUDA diagnostics, not a smoke test")
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error("explicit seeds must be unique and nonnegative")
    args.episodes, args.seed = len(seeds), seeds[0]
    if args.execute < 1:
        parser.error("execute must be positive")
    if args.decision_capture_dir and (min(args.decision_capture_steps) < 0 or
            any(step % args.execute for step in args.decision_capture_steps)):
        parser.error("capture steps must be nonnegative multiples of execute")
    if args.episodes < 1 or args.max_steps < 1 or args.seed < 0:
        parser.error("episodes and max-steps must be positive; seed must be nonnegative")
    if args.budgets and (min(args.budgets) < 1 or max(args.budgets) > args.max_steps):
        parser.error("budgets must lie within the rollout step limit")
    reference = json.loads(args.reference_report.read_text()) if args.reference_report else None
    if args.reference_mode == "initial" and reference is None:
        parser.error("initial comparison requires --reference-report")
    if reference is not None and reference.get("status") != "complete":
        parser.error("reference report must be complete")
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name

    tag = get_embodiment_tag_from_env_name(args.env)
    contract = checkpoint_contract(args.checkpoint, tag.value)
    if args.env.split("/")[0] not in {"gr1", "gr1_unified", "robocasa_panda_omron"}:
        parser.error("this baseline driver supports official RoboCasa GR1/Panda environments only")
    horizon = len(contract["action"]["delta_indices"])
    if not 1 <= args.execute <= horizon:
        parser.error(f"execute must be in [1, {horizon}]")
    print(json.dumps({"env": args.env, "embodiment": tag.value, "horizon": horizon,
                      "action_keys": contract["action"]["modality_keys"]}), flush=True)
    if args.preflight:
        print("Checkpoint contract checked. Simulator and model execution not tested.")
        return
    if args.out.exists():
        parser.error(f"refusing to overwrite existing results: {args.out}")

    import torch
    from common.device import pick_device, describe
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from grootN1_Robotics.baseline import run_episode, validate_contract, budget_outcomes
    from grootN1_Robotics.policy import LocalGr00tPolicy

    accelerator = pick_device(args.device)
    if not accelerator.is_cuda and not args.smoke:
        parser.error("local CPU/MPS runs are smoke tests; pass --smoke, or evaluate on CUDA")
    report = {"status": "running", "kind": "SIM", "selection": "none",
              "purpose": "local_smoke" if args.smoke else "task_baseline",
              "published_metric_reproduction": False,
              "env": args.env, "embodiment": tag.value,
              "execute": args.execute, "max_steps": args.max_steps,
              "requested_episodes": args.episodes, "seed": args.seed,
              "accelerator": describe(accelerator), "attention": "sdpa",
              "upstream_commit": subprocess.check_output(
                  ["git", "-C", str(ROOT / "grootN1_Robotics/upstream"), "rev-parse", "HEAD"],
                  text=True).strip(), "episodes": []}
    report["checkpoint_sha256"] = {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir())
                                   if p.suffix in {".json", ".safetensors"}}
    sim_repo = ROOT / "grootN1_Robotics/upstream/external_dependencies/robocasa-gr1-tabletop-tasks"
    if sim_repo.exists() and (sim_repo / "setup.py").exists():
        report["gr1_sim_commit"] = subprocess.check_output(
            ["git", "-C", str(sim_repo), "rev-parse", "HEAD"], text=True).strip()
    report["contract"] = contract
    report["budget_scan"] = sorted(set(args.budgets))
    if args.seeds is not None:
        report["diagnostic_seeds"] = seeds
    if reference is not None:
        report["reference_report_sha256"] = sha256(args.reference_report)
        report["reference_mode"] = args.reference_mode
    report["code_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in [
        Path(__file__), ROOT / "grootN1_Robotics/baseline.py", ROOT / "grootN1_Robotics/policy.py",
        *sorted((ROOT / "grootN1_Robotics/patches").glob("*.patch"))]}
    if args.decision_capture_dir:
        source = ROOT / "grootN1_Robotics/decision_capture.py"
        report["code_sha256"][str(source.relative_to(ROOT))] = sha256(source)
        report["decision_capture"] = {"steps": sorted(set(args.decision_capture_steps)),
                                     "format_version": 1}
    env = None
    save_report(args.out, report)
    try:
        # Fail on missing simulator/interface before allocating the 3B model.
        base_env = create_seeded_env(args.env, args.seed)
        env = base_env
        from types import SimpleNamespace
        configs = {key: SimpleNamespace(**value) for key, value in contract.items()}
        validate_contract(base_env, configs, args.execute)
        action_space = base_env.action_space
        report["packages"] = {name: version(name) for name in
                              ["torch", "numpy", "gymnasium", "robosuite", "robocasa", "mujoco"]}
        report["control_freq"] = getattr(getattr(base_env.unwrapped, "env", None), "control_freq", None)
        report["action_shapes"] = {key: list(space.shape) for key, space in action_space.spaces.items()}
        report["asset_manifests"] = {p.name: json.loads(p.read_text()) for p in
            sorted((sim_repo / "robocasa/models/assets").glob(".*.manifest.json"))}
        if reference is not None:
            validate_reference_provenance(report, reference, args.reference_mode)
        report["task_fingerprint"] = hashlib.sha256(json.dumps({key: report[key] for key in
            ["env", "embodiment", "execute", "max_steps", "upstream_commit", "code_sha256",
             "contract", "action_shapes", "control_freq", "packages", "asset_manifests",
             "gr1_sim_commit"]}, sort_keys=True).encode()).hexdigest()
        env = MultiStepWrapper(base_env,
                               video_delta_indices=np.asarray(configs["video"].delta_indices),
                               state_delta_indices=np.asarray(configs["state"].delta_indices),
                               n_action_steps=1, max_episode_steps=None)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        policy = Gr00tSimPolicyWrapper(LocalGr00tPolicy(
            tag, args.checkpoint, device=str(accelerator.device),
            dtype=accelerator.amp_dtype or torch.float32))
        capture = None
        if args.decision_capture_dir:
            from grootN1_Robotics.decision_capture import DecisionCapture
            from grootN1_Robotics.diagnostics import drawer_signals
            keys = ["env", "embodiment", "attention", "accelerator", "upstream_commit",
                    "gr1_sim_commit", "checkpoint_sha256", "code_sha256", "packages",
                    "contract", "execute", "max_steps", "asset_manifests", "control_freq", "action_shapes"]
            capture = DecisionCapture(args.decision_capture_dir, args.decision_capture_steps,
                accelerator.device, {key: report[key] for key in keys}, reader=drawer_signals)
        for index, seed in enumerate(seeds):
            if index:
                env.close()
                base_env = create_seeded_env(args.env, seed)
                env = base_env
                validate_contract(base_env, configs, args.execute)
                action_space = base_env.action_space
                env = MultiStepWrapper(base_env,
                    video_delta_indices=np.asarray(configs["video"].delta_indices),
                    state_delta_indices=np.asarray(configs["state"].delta_indices),
                    n_action_steps=1, max_episode_steps=None)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            previous = None
            if reference is not None:
                previous = next((e for e in reference["episodes"] if e["seed"] == seed), None)
                if previous is None or (args.reference_mode == "prefix" and previous["sim_steps"] > args.max_steps):
                    raise ValueError(f"reference seed {seed} is absent or exceeds current budget")
            recorder = None
            if args.diagnostics_dir:
                from grootN1_Robotics.diagnostics import EpisodeRecorder, stage_summary
                recorder = EpisodeRecorder(args.diagnostics_dir, seed, report["control_freq"])
            try:
                episode = run_episode(env, policy, seed=seed, execute=args.execute,
                                      horizon=horizon, max_steps=args.max_steps,
                                      action_space=action_space, capture_dir=args.capture_dir,
                                      observer=recorder, reference_episode=previous,
                                      reference_mode=args.reference_mode, decision_capture=capture)
            finally:
                if recorder is not None:
                    recorder.close()
            if recorder is not None:
                episode["diagnostics"] = {"video": recorder.video_path.name,
                    "stages": recorder.stages_path.name, "frames": recorder.frames,
                    "stage_at_budget": {str(b): stage_summary(recorder.rows, b)
                        for b in (report["budget_scan"] or [args.max_steps])}}
            if capture is not None:
                episode["decision_captures"] = [path for path in capture.records
                                                if path.startswith(f"episode_{seed}/")]
            report["episodes"].append(episode)
            save_report(args.out, report)
            print(json.dumps({k: v for k, v in episode.items() if k != "decisions"}), flush=True)
        report["status"] = "complete"
        if args.smoke:
            return  # finally saves status; a two-step smoke is not a success-rate measurement.
        report["success_rate"] = sum(e["success"] for e in report["episodes"]) / args.episodes
        # Wilson interval over independent episodes, not correlated decisions.
        n, p, z = args.episodes, report["success_rate"], 1.959963984540054
        centre = (p + z*z / (2*n)) / (1 + z*z/n)
        radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        report["success_rate_ci95_wilson"] = [max(0., centre-radius), min(1., centre+radius)]
        if report["budget_scan"]:
            report["success_by_budget"] = budget_outcomes(
                report["episodes"], report["budget_scan"], report["control_freq"])
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_report(args.out, report)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

"""Verify execute-4 versus execute-8 on identical scenes and initial predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_execution_pair(reference, scan):
    if reference.get("status") != "complete" or scan.get("status") != "complete":
        raise ValueError("comparison requires complete reports")
    if reference["execute"] != 8 or scan["execute"] != 4:
        raise ValueError("comparison requires execute 8 versus 4")
    if scan.get("reference_mode") != "initial":
        raise ValueError("execution contrast requires initial-only reference mode")
    for key in ["env", "embodiment", "checkpoint_sha256", "upstream_commit", "gr1_sim_commit",
                "contract", "control_freq", "packages", "action_shapes", "asset_manifests",
                "accelerator", "attention", "selection", "purpose", "seed", "requested_episodes",
                "max_steps", "budget_scan"]:
        if reference[key] != scan[key]:
            raise ValueError(f"comparison provenance differs: {key}")
    if scan["max_steps"] != 1440 or scan["requested_episodes"] != 10 or scan["seed"] != 0:
        raise ValueError("comparison requires ten seeds 0–9 and a 1440-step cap")
    if scan["budget_scan"] != [360, 720, 1080, 1440]:
        raise ValueError("comparison deadlines differ from experiment specification")
    for path, digest in reference["code_sha256"].items():
        if path.endswith("policy.py") or path.endswith(".patch"):
            if scan["code_sha256"].get(path) != digest:
                raise ValueError(f"comparison policy/patch source differs: {path}")
    old = {e["seed"]: e for e in reference["episodes"]}
    new = {e["seed"]: e for e in scan["episodes"]}
    if len(reference["episodes"]) != 10 or len(scan["episodes"]) != 10 or old.keys() != new.keys() or set(new) != set(range(10)):
        raise ValueError("comparison requires unique paired seeds 0–9")
    for report in (reference, scan):
        for episode in report["episodes"]:
            steps = episode["sim_steps"]
            if not 1 <= steps <= report["max_steps"]:
                raise ValueError("episode step count exceeds the experiment budget")
            if not episode["success"] and (steps != report["max_steps"] or episode["stop_reason"] != "budget"):
                raise ValueError("unsuccessful episode ended before the fixed budget")
            if episode["success"] and episode["stop_reason"] != "success":
                raise ValueError("successful episode has inconsistent stop reason")
            if episode["policy_calls"] != math.ceil(steps / report["execute"]):
                raise ValueError("policy call count differs from execution horizon")
            if [d["sim_step"] for d in episode["decisions"]] != list(range(0, steps, report["execute"])):
                raise ValueError("policy decision steps differ from execution horizon")
    for seed, previous in old.items():
        episode = new[seed]
        if not episode.get("reference_initial_decision_verified") or episode.get("reference_prefix_verified"):
            raise ValueError(f"seed {seed}: incorrect reference verification gate")
        if previous["initial_observation_sha256"] != episode["initial_observation_sha256"]:
            raise ValueError(f"seed {seed}: initial observation differs")
        if previous["decisions"][0] != episode["decisions"][0]:
            raise ValueError(f"seed {seed}: first state/action prediction differs")


def deadline_changes(reference, scan, budget):
    old = {e["seed"] for e in reference["episodes"] if e["success"] and e["sim_steps"] <= budget}
    new = {e["seed"] for e in scan["episodes"] if e["success"] and e["sim_steps"] <= budget}
    return {"improved_seeds": sorted(new - old), "regressed_seeds": sorted(old - new),
            "retained_success_seeds": sorted(old & new),
            "both_unsuccessful_seeds": sorted(set(range(10)) - old - new),
            "success_rate_difference": (len(new) - len(old)) / 10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    scan = json.loads(args.scan.read_text())
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != scan["reference_report_sha256"]:
        raise ValueError("mounted reference hash differs")
    verify_execution_pair(reference, scan)
    from grootN1_Robotics.baseline import budget_outcomes
    from grootN1_Robotics.diagnostics import stage_summary
    from grootN1_Robotics.tools.analyze_budget_scan import verify_diagnostics, make_contact_sheet
    curves = {}
    for key, report in (("execute_8", reference), ("execute_4", scan)):
        curve = budget_outcomes(report["episodes"], report["budget_scan"], report["control_freq"])
        if curve != report["success_by_budget"]:
            raise ValueError(f"{key}: independent budget curve differs")
        curves[key] = curve
    args.out.mkdir(parents=True, exist_ok=False)
    episodes = []
    old = {e["seed"]: e for e in reference["episodes"]}
    for episode in scan["episodes"]:
        rows = verify_diagnostics(episode, args.diagnostics, scan["control_freq"])
        stage = {str(b): stage_summary(rows, b) for b in scan["budget_scan"]}
        if stage != episode["diagnostics"]["stage_at_budget"]:
            raise ValueError("independent stage summaries differ")
        previous = old[episode["seed"]]
        episodes.append({"seed": episode["seed"], "execute_8_success": previous["success"],
            "execute_8_steps": previous["sim_steps"], "execute_4_success": episode["success"],
            "execute_4_steps": episode["sim_steps"], "execute_4_stages": stage})
        make_contact_sheet(episode, rows, args.diagnostics, args.out)
    summary = {"initial_scenes_and_predictions_verified": True, "curves": curves,
        "paired_changes": {str(b): deadline_changes(reference, scan, b) for b in scan["budget_scan"]},
        "episodes": episodes}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

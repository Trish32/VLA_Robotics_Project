"""Independently audit saved cold-replay traces, captures and original task signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_episode(original, result, traces, signals, captured_observations):
    from grootN1_Robotics.branch_replay import observation_hash
    if any(result.get(k) != original[k] for k in ["seed", "sim_steps", "success", "stop_reason"]):
        raise ValueError("branch episode outcome differs from original")
    if (result.get("exact_reconstruction_verified") is not True or len(traces) != 2
            or traces[0] != traces[1] or len(traces[0]) != original["sim_steps"] + 1
            or result.get("exact_step_traces") != len(traces[0])):
        raise ValueError("cold trace equality or inventory differs")
    if [r["sim_step"] for r in signals] != list(range(len(traces[0]))):
        raise ValueError("original signal inventory differs")
    for step, (row, signal) in enumerate(zip(traces[0], signals)):
        if row["sim_step"] != step or row["success"] != signal["success"]:
            raise ValueError("trace step/success differs from original")
        if row["labels"] != {k: signal[k] for k in row["labels"]}:
            raise ValueError("trace labels differ from original")
        if not isinstance(row["integration_size"], int) or row["integration_size"] < 1:
            raise ValueError("invalid native integration inventory")
        for key in ["observation_sha256", "policy_observation_sha256", "integration_sha256"]:
            if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError("invalid replay hash")
        if step < original["sim_steps"] and any(row[k] for k in ["success", "terminated", "truncated"]):
            raise ValueError("trace stopped before original episode")
    final = traces[0][-1]
    reason = ("success" if final["success"] else "truncated" if final["truncated"] else
              "terminated" if final["terminated"] else "budget")
    if reason != original["stop_reason"]:
        raise ValueError("final replay flags differ from original")
    if result.get("captured_observations_verified") != len(captured_observations):
        raise ValueError("verified capture count differs")
    for step, observation in captured_observations.items():
        if traces[0][step]["policy_observation_sha256"] != observation_hash(observation):
            raise ValueError("trace policy observation differs from original capture")
    # Parsing/serializing strictly also rejects non-finite numbers hidden in a trace.
    json.dumps(traces, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite existing analysis")
    from grootN1_Robotics.decision_capture import file_hash, load_capture
    from grootN1_Robotics.tools.eval_baseline import save_report
    from grootN1_Robotics.tools.replay_decisions import validate_capture_rollout
    report = json.loads(args.report.read_text())
    rollout = json.loads(args.rollout_report.read_text())
    if (report.get("status") != "complete" or report.get("candidate_quality_evaluated") is not False
            or report.get("rollout_report_sha256") != file_hash(args.rollout_report)
            or [e["seed"] for e in report["episodes"]] != [e["seed"] for e in rollout["episodes"]]):
        parser.error("incomplete, changed or mislabelled replay report")
    count, frames = 0, 0
    for original, result in zip(rollout["episodes"], report["episodes"]):
        traces = []
        if len(result["trace_files"]) != 2:
            raise ValueError("requires both independent replay trace files")
        for index, entry in enumerate(result["trace_files"]):
            expected = f"{args.report.stem}_seed_{original['seed']}_pass_{index}.json"
            if entry["file"] != expected:
                raise ValueError("replay trace filename differs")
            path = args.report.parent / expected
            if file_hash(path) != entry["sha256"]:
                raise ValueError("replay trace hash differs")
            traces.append(json.loads(path.read_text())["trace"])
        diagnostic = args.capture_dir.parent / "diagnostics" / original["diagnostics"]["stages"]
        if file_hash(diagnostic) != result["reference_diagnostics_sha256"]:
            raise ValueError("original diagnostics hash differs")
        signals = [json.loads(line) for line in diagnostic.read_text().splitlines()]
        captured = {}
        for name in original["decision_captures"]:
            directory = args.capture_dir / name
            manifest, obs, actions, _ = load_capture(directory)
            validate_capture_rollout(directory, manifest, obs, actions, rollout)
            captured[manifest["sim_step"]] = obs
        verify_episode(original, result, traces, signals, captured)
        if report["purpose"] != "local_smoke" and result["pause_steps"] != sorted(captured):
            raise ValueError("registered branch pause inventory differs")
        count += len(captured)
        frames += len(traces[0])
    save_report(args.out, {"status": "verified", "report_sha256": file_hash(args.report),
        "episodes": len(report["episodes"]), "captured_inputs": count,
        "exact_control_frames": frames, "candidate_quality_evaluated": False})
    print(f"independent gate: {count} captures, {frames} exact control frames verified")


if __name__ == "__main__":
    main()

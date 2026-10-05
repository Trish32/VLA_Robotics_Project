"""Independently verify the paired scan, stage/video alignment, then summarise results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_pair(reference, scan):
    if reference.get("status") != "complete" or scan.get("status") != "complete":
        raise ValueError("comparison requires two complete reports")
    for key in ["env", "embodiment", "execute", "checkpoint_sha256", "upstream_commit",
                "gr1_sim_commit", "contract", "control_freq", "packages", "action_shapes",
                "asset_manifests", "accelerator", "attention"]:
        if reference[key] != scan[key]:
            raise ValueError(f"comparison provenance differs: {key}")
    for path, digest in reference["code_sha256"].items():
        if path.endswith("policy.py") or path.endswith(".patch"):
            if scan["code_sha256"].get(path) != digest:
                raise ValueError(f"comparison policy/patch source differs: {path}")
    old = {e["seed"]: e for e in reference["episodes"]}
    new = {e["seed"]: e for e in scan["episodes"]}
    if len(old) != len(reference["episodes"]) or len(new) != len(scan["episodes"]) or old.keys() != new.keys():
        raise ValueError("duplicate or different episode seeds")
    for seed, previous in old.items():
        episode = new[seed]
        if not episode.get("reference_prefix_verified"):
            raise ValueError(f"seed {seed}: missing prefix gate")
        if previous["initial_observation_sha256"] != episode["initial_observation_sha256"]:
            raise ValueError(f"seed {seed}: initial observation mismatch")
        if previous["decisions"] != episode["decisions"][:len(previous["decisions"])]:
            raise ValueError(f"seed {seed}: state/action prefix mismatch")
        if episode["sim_steps"] < previous["sim_steps"]:
            raise ValueError(f"seed {seed}: rollout ended before the reference prefix")
        if previous["success"]:
            if not episode["success"] or episode["sim_steps"] != previous["sim_steps"]:
                raise ValueError(f"seed {seed}: previous success changed")
        elif episode["success"] and episode["sim_steps"] <= reference["max_steps"]:
            raise ValueError(f"seed {seed}: changed success within original budget")


def verify_diagnostics(episode, folder, control_freq):
    import cv2
    diagnostic = episode["diagnostics"]
    rows = [json.loads(line) for line in (folder / diagnostic["stages"]).read_text().splitlines()]
    if [r["sim_step"] for r in rows] != list(range(episode["sim_steps"] + 1)):
        raise ValueError("missing or duplicated diagnostic steps")
    for row in rows:
        if row["simulation_seconds"] != row["sim_step"] / control_freq or row["success"] != row["task_success"]:
            raise ValueError("stage times or task success disagree")
    first = next((r["sim_step"] for r in rows if r["success"]), None)
    if bool(first is not None) != episode["success"] or (first is not None and first != episode["sim_steps"]):
        raise ValueError("stage success disagrees with rollout result")
    video = cv2.VideoCapture(str(folder / diagnostic["video"]))
    try:
        if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
            raise ValueError("video frame count disagrees with stage steps")
        if video.get(cv2.CAP_PROP_FPS) != control_freq:
            raise ValueError("video FPS disagrees with simulator frequency")
    finally:
        video.release()
    return rows


def make_contact_sheet(episode, rows, folder, output):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = {0, min(720, episode["sim_steps"]), episode["sim_steps"]}
    for predicate in (lambda r: any(r["grasp_contact"].values()),
                      lambda r: r["object_inside_drawer"]):
        step = next((r["sim_step"] for r in rows if predicate(r)), None)
        if step is not None:
            selected.add(step)
    steps = sorted(selected)
    figure, axes = plt.subplots(len(steps), 1, figsize=(8, 4 * len(steps)), squeeze=False)
    video = cv2.VideoCapture(str(folder / episode["diagnostics"]["video"]))
    try:
        for step, axis in zip(steps, axes[:, 0]):
            video.set(cv2.CAP_PROP_POS_FRAMES, step)
            ok, frame = video.read()
            if not ok:
                raise ValueError(f"cannot decode video frame {step}")
            row = rows[step]
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.set_title(f"seed {episode['seed']}, step {step}, success={row['success']}\n"
                f"grasp={any(row['grasp_contact'].values())}, inside={row['object_inside_drawer']}, "
                f"drawer={row['drawer_open_fraction']:.3f}")
            axis.axis("off")
    finally:
        video.release()
    figure.tight_layout()
    figure.savefig(output / f"episode_{episode['seed']}_contact_sheet.png", dpi=120)
    plt.close(figure)


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
        raise ValueError("reference file hash differs from scan's mounted reference")
    verify_pair(reference, scan)
    from grootN1_Robotics.baseline import budget_outcomes
    from grootN1_Robotics.diagnostics import stage_summary
    curve = budget_outcomes(scan["episodes"], scan["budget_scan"], scan["control_freq"])
    if curve != scan["success_by_budget"]:
        raise ValueError("independently computed budget curve differs from report")
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for episode in scan["episodes"]:
        rows = verify_diagnostics(episode, args.diagnostics, scan["control_freq"])
        stage = {str(b): stage_summary(rows, b) for b in scan["budget_scan"]}
        if stage != episode["diagnostics"]["stage_at_budget"]:
            raise ValueError("independently computed stage summary differs from report")
        summaries.append({"seed": episode["seed"], "success": episode["success"],
                          "sim_steps": episode["sim_steps"], "stages": stage})
        make_contact_sheet(episode, rows, args.diagnostics, args.out)
    newly_successful = [e["seed"] for e in scan["episodes"]
                        if e["success"] and e["sim_steps"] > reference["max_steps"]]
    summary = {"paired_prefix_verified": True, "success_by_budget": curve,
               "newly_successful_seeds": newly_successful, "episodes": summaries}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

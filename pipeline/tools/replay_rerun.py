#!/usr/bin/env python
"""Replay the completed end-to-end run with all six stages on one timeline.

The earlier logger put stages 1 and 6 on the timeline and everything else as static
geometry, which shows a finished map with a camera moving through it. That is a diagram,
not a replay: you cannot see the map being built, and you cannot see which stage produced
what or when.

Here every stage occupies real time on a single `wall` clock in nanoseconds:

    stage 1-2   over the sequence's own 26 s. The camera moves, and the cloud GROWS —
                each frame's depth is unprojected and accumulated, so the map assembles
                the way fusion actually assembles it.
    stage 3-5   after the sequence, at their measured durations. Instances appear,
                labels attach to them, FoundationPose's pose is drawn against the
                gate's verdict, then the graph resolves a target.
    stage 6     the action chunk, laid out at the 30 Hz a controller would execute it.

Stage 4 is included even though its answer is currently "refuse". A replay that dropped
a stage because it failed would show a stack that is one stage shorter and one claim
stronger than the real one; the gap between FoundationPose's pose and the instance our
map holds is drawn, because that gap is the result.

`world/stage` carries the active stage as text, so scrubbing tells you what the system
was doing, not just what it had.

One clock throughout, from `pipeline.identity.Stamp`. That is the whole reason this is
coherent: stage 1 ingests TUM's float seconds, everything after is integer nanoseconds,
and float seconds cannot hold nanosecond precision at epoch scale — a mixed pipeline
would place the cloud and the trajectory on timelines disagreeing by a factor of 1e9 and
Rerun would render it without complaint.

    conda run -n foundationpose_vl python pipeline/tools/replay_rerun.py [--serve]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.identity import NS_PER_S, Frames, InstanceRegistry, Stamp  # noqa: E402

OUT = ROOT / "pipeline/assets/e2e"
SEQ = ROOT / "orbslam3_baseline/data/rgbd_dataset_freiburg1_xyz"
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989

PALETTE = [(15, 118, 110), (180, 83, 9), (124, 58, 237),
           (190, 18, 60), (3, 105, 161), (77, 124, 15)]

# Measured durations, so the replay's later stages take as long as they really took.
# "pose" is a T4 measurement (register + an 8-frame track), not a local one — the
# stage cannot run on this machine at all. Kept in the same units so the timeline
# stays honest about which stage costs what.
STAGE_SECONDS = {"segment": 34.0, "label": 210.0, "pose": 12.0, "graph": 0.2,
                 "policy": 3.0}


def read_trajectory(path: Path):
    from pipeline.transforms import quaternion_to_matrix

    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        out.append((Stamp.from_seconds(v[0]), T))     # lossy once, at ingestion only
    return out


def unproject(depth, T, stride=6):
    """Depth image + camera pose -> world points. This is what fusion accumulates."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = (z > 0.1) & (z < 4.0)
    if not ok.any():
        return np.empty((0, 3))
    z = z[ok]
    x = (us[ok] - CX) * z / FX
    y = (vs[ok] - CY) * z / FY
    cam = np.column_stack([x, y, z])
    return cam @ T[:3, :3].T + T[:3, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    import cv2
    import open3d as o3d
    import rerun as rr

    rr.init("vlaprojects_replay", spawn=args.serve)
    if not args.serve:
        rr.save(str(OUT / "e2e_replay.rrd"))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    traj = read_trajectory(ROOT / "orbslam3_baseline/data/trajectory_orbslam3.txt")
    traj_ns = np.array([s.ns for s, _ in traj])

    rgb = [(Stamp.from_seconds(float(l.split()[0])), l.split()[1])
           for l in (SEQ / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(Stamp.from_seconds(float(l.split()[0])), l.split()[1])
           for l in (SEQ / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    dep_ns = np.array([s.ns for s, _ in dep])

    def at(stamp: Stamp):
        rr.set_time("wall", timestamp=np.datetime64(stamp.ns, "ns"))

    def stage(text: str):
        rr.log("world/stage", rr.TextLog(text, level=rr.TextLogLevel.INFO))

    # ================================================ stages 1-2, over real time
    at(rgb[0][0])
    stage("STAGE 1-2 · ORB-SLAM3 localization + incremental fusion")

    track, kept, total = [], 0, 0
    for k, (stamp, rel) in enumerate(rgb):
        if k % args.stride:
            continue
        j = int(np.argmin(np.abs(traj_ns - stamp.ns)))
        if abs(traj_ns[j] - stamp.ns) > 0.02 * NS_PER_S:
            continue
        T = traj[j][1]
        at(stamp)

        rr.log(f"world/{Frames.CAMERA_OPTICAL}",
               rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image",
               rr.Pinhole(image_from_camera=[[FX, 0, CX], [0, FY, CY], [0, 0, 1]],
                          resolution=[640, 480]))
        bgr = cv2.imread(str(SEQ / rel))
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image/rgb",
               rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=75))

        m = int(np.argmin(np.abs(dep_ns - stamp.ns)))
        depth = cv2.imread(str(SEQ / dep[m][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image/depth", rr.DepthImage(depth, meter=1.0))

        track.append(T[:3, 3])
        rr.log("world/trajectory",
               rr.LineStrips3D([np.array(track)], colors=[(190, 18, 60)], radii=0.004))

        # The map GROWS -- the difference between a replay and a diagram. Each frame's
        # NEW points go to their own entity path and are logged once. Re-logging the
        # whole accumulated cloud every frame is quadratic: it produced a 1.6 GB
        # recording for 133 frames before this was fixed. Rerun keeps each entity
        # visible from its timestamp onward, so they accumulate without resending.
        fresh = unproject(depth, T)
        total += len(fresh)
        rr.log(f"world/cloud/live/f{kept:04d}",
               rr.Points3D(fresh, colors=(150, 160, 172), radii=0.006))
        rr.log("progress/points", rr.Scalars(float(total)))
        rr.log("progress/frames", rr.Scalars(float(len(track))))
        kept += 1

    end_of_sequence = rgb[-1][0]
    print(f"[replay]  stages 1-2: {kept} frames over "
          f"{(end_of_sequence.ns - rgb[0][0].ns) / NS_PER_S:.1f} s of sequence time")

    # Swap the running accumulation for the real TSDF output at the moment fusion ends.
    cloud = o3d.io.read_point_cloud(str(OUT / "tum_fr1_xyz.ply"))
    points = np.asarray(cloud.points)
    colours = (np.asarray(cloud.colors) * 255).astype(np.uint8)
    at(end_of_sequence)
    stage(f"STAGE 2 done · TSDF fused {len(points):,} points")
    # Retire the live accumulation and replace it with the real TSDF output.
    rr.log("world/cloud/live", rr.Clear(recursive=True))
    rr.log("world/cloud/fused", rr.Points3D(points, colors=colours, radii=0.004))

    # ===================================================== stage 3: segmentation
    cursor = Stamp(end_of_sequence.ns + int(STAGE_SECONDS["segment"] * NS_PER_S))
    masks = np.load(OUT / "instance_masks.npz")["masks"]
    labelled = json.load(open(OUT / "labelled.json"))["instances"]

    at(cursor)
    stage(f"STAGE 3 · Mask3D — {len(labelled)} class-agnostic proposals")
    for i, r in enumerate(labelled):
        rr.log(f"world/proposal/{r['id']}",
               rr.Points3D(points[masks[i]], colors=(120, 130, 145), radii=0.007))

    # ========================================== stage 3.5: labels + association
    registry = InstanceRegistry()
    grouped: dict[str, list] = {}
    for i, r in enumerate(labelled):
        if not r.get("label"):
            continue
        instance_id, _ = registry.associate(r["centre"], r["extent"], r["label"], cursor)
        grouped.setdefault(instance_id, []).append((i, r))

    per_label = STAGE_SECONDS["label"] / max(1, len(grouped))
    for n, (instance_id, members) in enumerate(grouped.items()):
        cursor = Stamp(cursor.ns + int(per_label * NS_PER_S))
        at(cursor)
        colour = PALETTE[n % len(PALETTE)]
        tracked = registry.tracked[instance_id]
        note = "" if len(members) == 1 else f" [{len(members)} proposals merged]"
        stage(f"STAGE 3.5 · SAM+CLIP -> {instance_id}{note}")

        # Retire the grey proposals this instance absorbed, so the merge is visible
        # rather than leaving two overlapping renderings of one desk.
        for i, _ in members:
            rr.log(f"world/proposal/{labelled[i]['id']}", rr.Clear(recursive=True))

        path = f"world/{Frames.instance(instance_id)}"
        union = np.any([masks[i] for i, _ in members], axis=0)
        rr.log(path, rr.Points3D(points[union], colors=colour, radii=0.007))
        rr.log(f"{path}/box", rr.Boxes3D(
            centers=[tracked.centre], half_sizes=[tracked.extent / 2], colors=[colour],
            labels=[f"{instance_id} ({members[0][1]['similarity']:.2f}){note}"]))

    merged = sum(len(m) - 1 for m in grouped.values())
    print(f"[replay]  stage 3: {len(labelled)} proposals -> {len(grouped)} instances "
          f"({merged} merged)")

    # ==================================================== stage 4: the 6-DoF pose
    # The stage that refuses. Drawing the rejected pose next to the instance our map
    # holds is the point: the gap IS the result, and a replay that skipped a stage
    # because its answer was "no" would misrepresent what the stack does.
    pose_path = OUT / "pose.json"
    if pose_path.exists():
        pose = json.load(open(pose_path))
        cursor = Stamp(cursor.ns + int(STAGE_SECONDS["pose"] * NS_PER_S))
        at(cursor)
        verdict = "ADMITTED" if pose["accepted"] else "REFUSED"
        stage(f"STAGE 4 · FoundationPose -> {verdict}")
        Tp = pose.get("pose_world") or pose.get("rejected_pose_world")
        if Tp is not None:
            Tp = np.asarray(Tp, float)
            colour = [0x0f, 0x76, 0x6e] if pose["accepted"] else [0xc2, 0x41, 0x0c]
            rr.log("world/pose/foundationpose", rr.Transform3D(
                translation=Tp[:3, 3], mat3x3=Tp[:3, :3], axis_length=0.3))
            rr.log("world/pose/marker", rr.Points3D(
                [Tp[:3, 3]], colors=[colour], radii=0.045,
                labels=[f"FoundationPose ({verdict})"]))
            tgt = registry.tracked[pose["target"]].centre
            rr.log("world/pose/disagreement", rr.LineStrips3D(
                [[Tp[:3, 3], tgt]], colors=[colour], radii=0.004,
                labels=[f"{np.linalg.norm(Tp[:3, 3] - tgt) * 100:.0f} cm"]))
        rr.log("world/pose/gate", rr.TextDocument(
            "\n".join([
                f"**stage 4 — {verdict}**", "",
                f"- translation `{pose['median_translation_cm']:.1f} cm`",
                f"- rotation `{pose['median_rotation_deg']:.1f} deg`",
                f"- agreement `{pose.get('agreement_detail', 'n/a')}`",
                f"- frames corroborating `{pose['frames_corroborating']}/"
                f"{pose['frames_total']}`", "",
                ("6-DoF pose entered the graph." if pose["accepted"] else
                 "The position-only pose stands. A pose that says nothing about "
                 "rotation beats one that says something wrong."),
            ]), media_type=rr.MediaType.MARKDOWN))
        print(f"[replay]  stage 4: pose {verdict.lower()}")

    # ======================================================= stage 5: the graph
    ground = json.load(open(OUT / "ground.json"))
    cursor = Stamp(cursor.ns + int(STAGE_SECONDS["graph"] * NS_PER_S))
    at(cursor)
    stage(f"STAGE 5 · scene graph -> target {ground['target']}")
    rr.log("world/graph", rr.TextDocument(
        "\n".join([f"**target** `{ground['target']}`", "",
                   f"**prompt** `{ground['prompt']}`", "", "**instances**"] +
                  [f"- `{i}` — {registry.tracked[i].label}" for i in grouped]),
        media_type=rr.MediaType.MARKDOWN))

    # ====================================================== stage 6: the policy
    policy = json.load(open(OUT / "policy.json"))
    values = policy.get("values", {})
    if values:
        cursor = Stamp(cursor.ns + int(STAGE_SECONDS["policy"] * NS_PER_S))
        at(cursor)
        stage(f"STAGE 6 · GR00T N1.6-3B -> {policy['seconds']}s, chunk of "
              f"{len(next(iter(values.values())))} steps")
        for step in range(len(next(iter(values.values())))):
            at(Stamp(cursor.ns + step * NS_PER_S // 30))       # executed at 30 Hz
            for group, arr in values.items():
                for d, value in enumerate(np.asarray(arr)[step]):
                    rr.log(f"policy/{group}/dof_{d}", rr.Scalars(float(value)))
        span = (cursor.ns - rgb[0][0].ns) / NS_PER_S
        print(f"[replay]  stages 5-6 logged; full replay spans {span:.1f} s")

    if not args.serve:
        print(f"\n[replay]  wrote {OUT / 'e2e_replay.rrd'}")
        print("          rerun pipeline/assets/e2e/e2e_replay.rrd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

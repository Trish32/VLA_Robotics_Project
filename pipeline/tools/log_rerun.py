#!/usr/bin/env python
"""Replay the end-to-end run into Rerun, time-aligned on one clock.

Rerun is the right first layer for this stack because the stack is mostly non-ROS
Python: rqt_graph would show two nodes, RViz needs a display and cannot express `on` or
`inside`, but Rerun puts the camera track, the fused cloud, per-instance masks, the
source images and the policy's action chunk on a SINGLE scrubbable timeline.

This is also the first real consumer of `pipeline/identity.py`, and it is what that
module was for:

  * **one clock.** Every `rr.set_time` call takes `Stamp.ns`. Mixing float seconds and
    nanoseconds across stages would put the cloud and the trajectory on timelines that
    silently disagree by a factor of 1e9, and Rerun would render it without complaint.
  * **one frame authority.** Entity paths mirror `Frames`, so the transform hierarchy in
    the viewer is the same hierarchy TF publishes: `world/camera`, `world/object/<id>`.
  * **one identity.** Instances are logged under their STABLE id from `InstanceRegistry`.
    Under the old rank-ordered `inst_0`, re-running segmentation would relabel every
    entity path and the recording would show an object teleporting between identities.

    conda run -n foundationpose_vl python pipeline/tools/log_rerun.py [--serve]
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
        # from_seconds is lossy at epoch scale and that is fine HERE: it happens once,
        # at ingestion, because TUM stores float seconds. Everything downstream uses ns.
        out.append((Stamp.from_seconds(v[0]), T))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="open the viewer instead of saving")
    ap.add_argument("--stride", type=int, default=8, help="log every Nth camera frame")
    args = ap.parse_args()

    import cv2
    import open3d as o3d
    import rerun as rr

    rr.init("vlaprojects_e2e", spawn=args.serve)
    if not args.serve:
        rr.save(str(OUT / "e2e.rrd"))

    # Right-handed, Z up: matches the convention the scene graph and Frames assume.
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    traj = read_trajectory(ROOT / "orbslam3_baseline/data/trajectory_orbslam3.txt")
    t0 = traj[0][0]
    print(f"[rerun ]  {len(traj)} poses; timeline starts at {t0.ns} ns")

    # ------------------------------------------------------------ static scene
    cloud = o3d.io.read_point_cloud(str(OUT / "tum_fr1_xyz.ply"))
    points = np.asarray(cloud.points)
    colours = (np.asarray(cloud.colors) * 255).astype(np.uint8)
    rr.log("world/cloud", rr.Points3D(points, colors=colours, radii=0.004), static=True)

    masks = np.load(OUT / "instance_masks.npz")["masks"]
    labelled = json.load(open(OUT / "labelled.json"))["instances"]

    # Stable identity, minted once. Under rank-ordered ids these entity paths would
    # change meaning on every re-run of segmentation.
    registry = InstanceRegistry()
    stamp_scene = Stamp(t0.ns)

    # Several Mask3D proposals can resolve to ONE stable instance -- on this scene
    # inst_2 and inst_5 overlap at IoU 0.498, i.e. one desk segmented twice. Grouping
    # by stable id rather than logging per proposal is what turns that into a single
    # entity; logging per proposal would have the second silently overwrite the first
    # at the same entity path.
    grouped: dict[str, list] = {}
    for i, r in enumerate(labelled):
        if not r.get("label"):
            continue
        instance_id, _ = registry.associate(
            r["centre"], r["extent"], r["label"], stamp_scene)
        grouped.setdefault(instance_id, []).append((i, r))

    for n, (instance_id, members) in enumerate(grouped.items()):
        colour = PALETTE[n % len(PALETTE)]
        path = f"world/{Frames.instance(instance_id)}"
        union = np.any([masks[i] for i, _ in members], axis=0)
        rr.log(path, rr.Points3D(points[union], colors=colour, radii=0.007), static=True)

        tracked = registry.tracked[instance_id]
        note = "" if len(members) == 1 else f" [{len(members)} proposals merged]"
        rr.log(f"{path}/box", rr.Boxes3D(
            centers=[tracked.centre], half_sizes=[tracked.extent / 2],
            colors=[colour],
            labels=[f"{instance_id} ({members[0][1]['similarity']:.2f}){note}"]),
            static=True)

    merged = sum(len(m) - 1 for m in grouped.values())
    print(f"[rerun ]  {len(labelled)} proposals -> {len(grouped)} instances "
          f"({merged} merged by association): {list(grouped)}")

    # --------------------------------------------------------- scene relations
    ground = json.load(open(OUT / "ground.json"))
    rr.log("world/graph", rr.TextDocument(
        "\n".join([f"target: {ground['target']}", f"prompt: {ground['prompt']}",
                   "", "labels:"] +
                  [f"  {k} -> {v}" for k, v in ground.get("labels", {}).items()]),
        media_type=rr.MediaType.MARKDOWN), static=True)

    # ------------------------------------------------------------- the camera
    rgb = [(Stamp.from_seconds(float(l.split()[0])), l.split()[1])
           for l in (SEQ / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(Stamp.from_seconds(float(l.split()[0])), l.split()[1])
           for l in (SEQ / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    dep_ns = np.array([s.ns for s, _ in dep])
    traj_ns = np.array([s.ns for s, _ in traj])

    track = []
    for k, (stamp, rel) in enumerate(rgb):
        if k % args.stride:
            continue
        j = int(np.argmin(np.abs(traj_ns - stamp.ns)))
        if abs(traj_ns[j] - stamp.ns) > 0.02 * NS_PER_S:
            continue
        T = traj[j][1]
        # ONE timeline, in nanoseconds. Every stage below sets the same clock.
        rr.set_time("wall", timestamp=np.datetime64(stamp.ns, "ns"))

        rr.log(f"world/{Frames.CAMERA_OPTICAL}", rr.Transform3D(
            translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image", rr.Pinhole(
            image_from_camera=[[FX, 0, CX], [0, FY, CY], [0, 0, 1]],
            resolution=[640, 480]))

        bgr = cv2.imread(str(SEQ / rel))
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image/rgb",
               rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        m = int(np.argmin(np.abs(dep_ns - stamp.ns)))
        depth = cv2.imread(str(SEQ / dep[m][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        rr.log(f"world/{Frames.CAMERA_OPTICAL}/image/depth",
               rr.DepthImage(depth, meter=1.0))

        track.append(T[:3, 3])
        rr.log("world/trajectory", rr.LineStrips3D([np.array(track)],
                                                   colors=[(190, 18, 60)], radii=0.004))
    print(f"[rerun ]  {len(track)} camera frames logged with RGB + depth")

    # -------------------------------------------------------- the action chunk
    policy = json.load(open(OUT / "policy.json"))
    values = policy.get("values", {})
    if values:
        # The chunk is the policy's FUTURE, so it is laid out on the timeline after the
        # observation that produced it, at the 30 Hz the controller would execute it.
        base = rgb[0][0].ns
        for step in range(len(next(iter(values.values())))):
            rr.set_time("wall", timestamp=np.datetime64(base + step * NS_PER_S // 30, "ns"))
            for group, arr in values.items():
                for d, value in enumerate(np.asarray(arr)[step]):
                    rr.log(f"policy/{group}/dof_{d}", rr.Scalars(float(value)))
        print(f"[rerun ]  action chunk: {len(next(iter(values.values())))} steps x "
              f"{sum(np.asarray(v).shape[1] for v in values.values())} DoF")

    if not args.serve:
        print(f"\n[rerun ]  wrote {OUT / 'e2e.rrd'}")
        print("          view with:  rerun pipeline/assets/e2e/e2e.rrd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

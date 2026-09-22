#!/usr/bin/env python
"""End-to-end on REAL data: ORB-SLAM3 poses -> TSDF -> Mask3D -> scene graph -> prompt.

Why TUM rather than the synthetic demo room. Two independent failures pointed at the
same cause:

  * ORB-SLAM3 tracked **0 / 12** frames of the synthetic scene. Measured reason: **95**
    ORB keypoints per frame versus **1200** on TUM. Six flat-shaded analytic boxes have
    corners only at their edges, and a feature-based tracker cannot initialise on that.
  * Mask3D scored **0.000 IoU** on the synthetic pear and mug at every voxel size from
    0.005 to 0.02 m. It was trained on real ScanNet rooms — noisy normals, clutter,
    texture — and an analytic cube is out of that distribution.

So the synthetic scene was a bad instrument for BOTH stages, and in the same way. TUM
fr1/xyz is a real desk scene: ORB-SLAM3 tracks 798/798 on it at ATE 1.03 cm, and its
geometry is the kind Mask3D was trained on.

This fuses the sequence using the poses ORB-SLAM3 ESTIMATED — not ground truth — so the
run tests the pipeline rather than the TSDF.

    conda run -n foundationpose_vl python pipeline/tools/e2e_tum.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEQ = ROOT / "orbslam3_baseline/data/rgbd_dataset_freiburg1_xyz"
TRAJ = ROOT / "orbslam3_baseline/data/trajectory_orbslam3.txt"
OUT = ROOT / "pipeline/assets/e2e"

# Intrinsics are PER CAMERA, not per dataset: freiburg1 and freiburg3 were recorded on
# different Kinects and upstream ships TUM1.yaml and TUM3.yaml separately. Fusing fr3
# with fr1's numbers warps the cloud by a few percent of range -- enough to bend a flat
# desk and to move every instance centroid, and nothing raises.
INTRINSICS = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}
DEPTH_SCALE = 5000.0


def intrinsics_for(sequence: Path):
    for key, values in INTRINSICS.items():
        if key in sequence.name:
            return key, values
    raise SystemExit(
        f"no intrinsics known for {sequence.name!r}; add them to INTRINSICS rather "
        "than defaulting, because the wrong ones warp the map silently."
    )


def load_trajectory(path: Path) -> dict[float, np.ndarray]:
    from pipeline.transforms import quaternion_to_matrix

    poses = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        poses[v[0]] = T
    return poses


def read_index(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        stamp, rel = line.split()
        rows.append((float(stamp), rel))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, default=None,
                    help="TUM sequence dir; defaults to fr1/xyz")
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="TUM-format poses; defaults to the fr1 ORB-SLAM3 run. For "
                         "fr3/walking_xyz use traj_walking_filtered.txt -- the "
                         "dynamic-rejected run (ATE 18.70 cm vs 80.92 baseline); "
                         "fusing the baseline smears the map with the walkers' drift.")
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth frame; 798 frames is far more than a TSDF needs")
    ap.add_argument("--query", default="monitor")
    args = ap.parse_args()

    import cv2
    import open3d as o3d

    global SEQ
    if args.sequence:
        SEQ = args.sequence
    traj_path = args.trajectory or TRAJ
    camera, (fx, fy, cx, cy) = intrinsics_for(SEQ)
    print(f"[0 setup]  {SEQ.name}  ({camera}: fx={fx}, fy={fy})")
    print(f"[0 setup]  poses {traj_path.name}")

    from pipeline.scene_export import PosedFrame, fuse_tsdf_levelled

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- 1. estimated trajectory
    poses = load_trajectory(traj_path)
    rgb = read_index(SEQ / "rgb.txt")
    depth_index = read_index(SEQ / "depth.txt")
    depth_stamps = np.array([s for s, _ in depth_index])
    pose_stamps = np.array(sorted(poses))
    print(f"[1 slam ]  {len(poses)} estimated poses from ORB-SLAM3")

    frames = []
    for stamp, rgb_rel in rgb[:: args.stride]:
        j = int(np.argmin(np.abs(pose_stamps - stamp)))
        if abs(pose_stamps[j] - stamp) > 0.02:
            continue
        k = int(np.argmin(np.abs(depth_stamps - stamp)))
        if abs(depth_stamps[k] - stamp) > 0.02:
            continue
        colour = cv2.cvtColor(cv2.imread(str(SEQ / rgb_rel)), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(SEQ / depth_index[k][1]), cv2.IMREAD_ANYDEPTH)
        frames.append(PosedFrame(
            color=np.ascontiguousarray(colour),
            depth_m=(raw.astype(np.float32) / DEPTH_SCALE),
            cam_to_world=poses[pose_stamps[j]],
            stamp_ns=int(stamp * 1e9),
        ))
    print(f"[2 fuse ]  {len(frames)} posed frames (stride {args.stride})")

    intr = np.eye(4)
    intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2] = fx, fy, cx, cy
    # Levelled, not raw. SLAM's world frame is the first keyframe's camera frame, which
    # on this sequence is pitched ~45 degrees down; every consumer below assumes z-up
    # and none of them can detect the tilt. See pipeline/gravity.py.
    scene = fuse_tsdf_levelled(frames, intr, frames[0].color.shape[:2],
                               voxel_length=0.02, sdf_trunc=0.06, depth_trunc=4.0)
    cloud, points = scene.cloud, np.asarray(scene.cloud.points)
    ply = OUT / f"{SEQ.name.replace('rgbd_dataset_', '')}.ply"
    o3d.io.write_point_cloud(str(ply), cloud)
    print(f"           {len(points)} points, extent "
          f"{(points.max(0) - points.min(0)).round(2).tolist()} m -> {ply.name}")
    print(f"           levelled: {scene.inlier_fraction:.1%} of points on the support "
          f"plane; cameras now at z "
          f"{min(f.cam_to_world[2, 3] for f in scene.frames):.2f}..."
          f"{max(f.cam_to_world[2, 3] for f in scene.frames):.2f}")

    # The rotation is saved because anything produced in the SLAM frame by an earlier
    # run -- instance centres, a target pose -- is in a DIFFERENT frame to this cloud,
    # and mixing them silently misplaces objects.
    json.dump({"ok": True, "frames": len(frames), "points": int(len(points)),
               "ply": str(ply), "query": args.query, "sequence": SEQ.name,
               "trajectory": traj_path.name, "camera": camera,
               "levelled": True,
               "plane_inlier_fraction": scene.inlier_fraction,
               "T_level_slam": scene.T_level_slam.tolist()},
              open(OUT / "fuse.json", "w"))
    print("\n[next  ]  stage 3 (Mask3D) runs in openmask3d_vl — different torch/env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

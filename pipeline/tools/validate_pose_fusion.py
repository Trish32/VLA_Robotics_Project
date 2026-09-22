#!/usr/bin/env python
"""How much does SLAM drift displace a fused object pose?

The world model's central claim is that camera-frame object poses fuse with camera
trajectories into a consistent world frame:

    T_anchor_obj = T_anchor_cam @ T_cam_obj

The unit tests check that composition on synthetic single frames, which proves the
algebra and nothing else. The question that actually decides whether the world model is
good enough to grasp with is different: the camera pose on the left is an ESTIMATE, and
it carries ORB-SLAM3's 1.03 cm ATE into every object it places. How far does an object
land from where it really is?

Measuring that needs two independent trajectories, or the test is circular. Synthesising
the detection from the same estimated pose used to fuse it makes
`T_est @ inv(T_est) @ T_wo` collapse to `T_wo` exactly, for any trajectory, including a
wrong one. So:

    detection   synthesised from TUM's GROUND-TRUTH camera pose — a perfect
                FoundationPose looking from where the camera really was
    fusion      performed with the ORB-SLAM3 ESTIMATED pose, through the real
                `refine_with_pose`

The residual is then exactly the object-placement error the world model inherits from
localization, and nothing else. Objects are the real instances from the end-to-end run,
and only the frames where an object was actually observable are counted — averaging in
frames where it sat behind the camera would measure drift at moments no detector could
have fired.

    conda run -n foundationpose_vl python pipeline/tools/validate_pose_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "orbslam3_baseline/tools"))

E2E = ROOT / "pipeline/assets/e2e"
SEQ = ROOT / "orbslam3_baseline/data/rgbd_dataset_freiburg1_xyz"
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989
W, H = 640, 480
NS_PER_S = 1_000_000_000


def read_poses(path: Path):
    """TUM format -> (stamps, (N,4,4)). Full poses, unlike eval_ate's read_tum."""
    from pipeline.transforms import quaternion_to_matrix

    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    stamps = np.array([float(r[0]) for r in rows])
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, r in enumerate(rows):
        v = [float(x) for x in r[1:]]
        poses[i, :3, :3] = quaternion_to_matrix(v[3], v[4], v[5], v[6])
        poses[i, :3, 3] = v[0:3]
    return stamps, poses


def visible(point_world, T_world_cam, *, near=0.3, far=5.0):
    """Would a detector plausibly have seen this point from this camera?"""
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    cam = R.T @ (point_world - t)
    if not near < cam[2] < far:
        return False
    u = FX * cam[0] / cam[2] + CX
    v = FY * cam[1] / cam[2] + CY
    return 0 <= u < W and 0 <= v < H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the measurement here")
    args = ap.parse_args()

    import open3d as o3d
    from eval_ate import associate, umeyama

    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d, refine_with_pose

    est_t, est_T = read_poses(ROOT / "orbslam3_baseline/data/trajectory_orbslam3.txt")
    gt_t, gt_T = read_poses(SEQ / "groundtruth.txt")
    pairs = associate(est_t, gt_t)
    if not pairs:
        raise SystemExit("no estimate/ground-truth pairs within 20 ms")

    # Rigid alignment only. RGB-D is metric, so fitting a scale would let the optimiser
    # absorb real metric drift into a scale factor and flatter the result.
    ei = np.array([i for i, _ in pairs])
    gi = np.array([j for _, j in pairs])
    _, R, t = umeyama(est_T[ei, :3, 3], gt_T[gi, :3, 3], with_scale=False)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, t
    aligned_T = A @ est_T                       # estimated trajectory, in the GT frame

    ate = np.linalg.norm(aligned_T[ei, :3, 3] - gt_T[gi, :3, 3], axis=1)

    # ATE is a translation metric and object placement is not. Measure the camera's
    # ROTATION error too, as the geodesic angle between estimate and truth: an object
    # at range d is displaced by roughly d*theta, so this is the term that decides
    # world-model accuracy and the one ATE never reports.
    dR = np.einsum("nij,nkj->nik", aligned_T[ei, :3, :3], gt_T[gi, :3, :3])
    angle = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))
    mean_angle = float(angle.mean())
    print(f"[fusion]  {len(pairs)} associated poses · ATE RMSE "
          f"{np.sqrt((ate ** 2).mean()) * 100:.2f} cm · rotation error "
          f"{np.degrees(mean_angle):.2f}°")

    # The real instances from the end-to-end run. Their centres live in the estimated
    # frame (the cloud was fused with the estimated trajectory), so they take the same
    # alignment as the trajectory that built them.
    points = np.asarray(o3d.io.read_point_cloud(str(E2E / "tum_fr1_xyz.ply")).points)
    masks = np.load(E2E / "instance_masks.npz")["masks"]
    labelled = json.load(open(E2E / "labelled.json"))["instances"]
    keep = [i for i, r in enumerate(labelled) if r.get("label")]
    observations = from_openmask3d(
        [np.flatnonzero(masks[i]) for i in keep], points,
        [labelled[i]["label"] for i in keep], [labelled[i]["similarity"] for i in keep],
        anchor_frame=Frames.keyframe(0), stamp_ns=NS_PER_S, registry=InstanceRegistry())

    by_id = {}
    for obs in observations:
        by_id.setdefault(obs.node_id, obs)

    # The cloud is LEVELLED (gravity-aligned) but `A` was fitted from the SLAM-frame
    # trajectory onto ground truth, so instance centres have to come back to the SLAM
    # frame before `A` is applied. Skipping this does not raise -- it silently places
    # every object somewhere else, and the only symptom is that nothing is ever
    # "observable", which is how this was caught.
    fuse = json.load(open(E2E / "fuse.json"))
    R_level = np.asarray(fuse.get("T_level_slam", np.eye(4)), float)[:3, :3]
    if not fuse.get("levelled"):
        R_level = np.eye(3)

    report, every_error = {}, []
    for node_id, obs in by_id.items():
        in_slam = R_level.T @ obs.pose[:3, 3]          # levelled -> SLAM
        truth = (A @ np.append(in_slam, 1.0))[:3]      # SLAM -> ground truth
        T_world_obj = np.eye(4)
        T_world_obj[:3, 3] = truth

        errors, placed, ranges, rot_only, trans_only = [], [], [], [], []
        for i, j in pairs:
            if not visible(truth, gt_T[j]):
                continue
            # Perfect detector, TRUE camera: what FoundationPose would report.
            T_cam_obj = np.linalg.inv(gt_T[j]) @ T_world_obj
            # Real fusion path, ESTIMATED camera.
            stamp = int(est_t[i] * NS_PER_S)
            fused = refine_with_pose(obs, T_cam_obj, aligned_T[i],
                                     stamp_ns=stamp, camera_stamp_ns=stamp)
            errors.append(np.linalg.norm(fused.pose[:3, 3] - truth))
            placed.append(fused.pose[:3, 3])
            ranges.append(np.linalg.norm(T_cam_obj[:3, 3]))

            # Which half of the camera pose is responsible? Swap one at a time.
            # Rotation error acts on the object's RANGE (a small angle at 2 m is a
            # large displacement); translation error is range-independent.
            hybrid = aligned_T[i].copy()
            hybrid[:3, 3] = gt_T[j][:3, 3]
            rot_only.append(np.linalg.norm((hybrid @ T_cam_obj)[:3, 3] - truth))
            hybrid = gt_T[j].copy()
            hybrid[:3, 3] = aligned_T[i][:3, 3]
            trans_only.append(np.linalg.norm((hybrid @ T_cam_obj)[:3, 3] - truth))

        if not errors:
            print(f"[fusion]  {node_id:<11} never observable — skipped")
            continue
        errors = np.array(errors)
        every_error.append(errors)
        # Real multi-view fusion: average the PLACED POSITIONS, then measure how far
        # that average sits from truth. (Averaging the error magnitudes instead would
        # just restate the mean and could never show a gain.)
        multiview = float(np.linalg.norm(np.mean(placed, axis=0) - truth))
        report[node_id] = dict(
            label=obs.label, views=len(errors),
            mean_cm=float(errors.mean() * 100), median_cm=float(np.median(errors) * 100),
            p95_cm=float(np.percentile(errors, 95) * 100),
            max_cm=float(errors.max() * 100), multiview_cm=multiview * 100,
            mean_range_m=float(np.mean(ranges)),
            rotation_only_cm=float(np.mean(rot_only) * 100),
            translation_only_cm=float(np.mean(trans_only) * 100))
        # d*theta is the UPPER BOUND on the displacement, not a point prediction: it is
        # attained only when the rotation axis is perpendicular to the object's bearing,
        # and the component of rotation about the viewing axis displaces nothing. The
        # measured error running 70-92% of it is the mechanism confirming itself.
        bound = float(np.mean(ranges) * mean_angle)
        report[node_id]["range_x_angle_bound_cm"] = bound * 100
        print(f"[fusion]  {node_id:<11} {obs.label:<9} {len(errors):>4} views  "
              f"range {np.mean(ranges):.2f} m  mean {errors.mean() * 100:5.2f} cm  "
              f"(rot {np.mean(rot_only) * 100:5.2f} / trans {np.mean(trans_only) * 100:4.2f})  "
              f"d·θ bound {bound * 100:5.2f}  multi-view {multiview * 100:5.2f} cm")

    if not report:
        raise SystemExit("no instance was observable in any associated frame")

    pooled = np.concatenate(every_error)
    summary = dict(
        ate_rmse_cm=float(np.sqrt((ate ** 2).mean()) * 100),
        associated_poses=len(pairs), instances=len(report),
        pooled_mean_cm=float(pooled.mean() * 100),
        pooled_median_cm=float(np.median(pooled) * 100),
        pooled_p95_cm=float(np.percentile(pooled, 95) * 100),
        pooled_max_cm=float(pooled.max() * 100),
        per_instance=report)
    print(f"\n[fusion]  pooled over {len(pooled)} observations: "
          f"mean {pooled.mean() * 100:.2f} cm · p95 {np.percentile(pooled, 95) * 100:.2f} cm "
          f"· max {pooled.max() * 100:.2f} cm")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"[fusion]  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

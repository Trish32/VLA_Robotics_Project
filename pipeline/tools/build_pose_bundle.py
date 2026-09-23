#!/usr/bin/env python
"""Stage 4.5 inputs: cut a mesh + per-frame masks for FoundationPose out of OUR map.

FoundationPose is model-based — `register(K, rgb, depth, ob_mask, mesh)` needs a mesh of
the object. This pipeline deliberately does not use CAD models (see
`foundationpose_6dof/bug_log.txt`): DROID/ORB-SLAM3 in RGB-D gives METRIC poses, those
fuse into a TSDF, and OpenMask3D cuts a per-instance subset out of the fused scene. So
the mesh comes from stages 1-3 and the 2-D `ob_mask` is that instance projected into the
frame. No CAD, no BundleSDF NeRF.

That only holds because the poses are metric. Monocular SLAM carries an arbitrary scale,
which would make the mesh the wrong SIZE and the resulting pose confidently wrong — the
failure mode nothing downstream can catch, which is why `SceneWriter` refuses non-metric
input upstream.

Two things this writes that are easy to get wrong:

  * **`ob_mask` is a 2-D image mask, not the 3-D instance.** It is produced by projecting
    the instance's points into the frame with that frame's camera pose and intrinsics,
    with an occlusion test against measured depth — without which the mask covers
    whatever is on the far side of the room too.
  * **The mesh origin is the instance's point centroid, and that is what the returned
    pose refers to.** `estimater.reset_object` does subtract `(min_xyz + max_xyz) / 2`
    from the vertices, but `estimater.py:233` undoes it on the way out
    (`poses[0] @ get_tf_to_centered_mesh()`), so `register()` returns the pose of the
    mesh AS SUPPLIED. Re-centring the mesh here to "compensate" therefore compensates
    for nothing — it only moves the point the residual is measured at. Measured: doing
    so moved the origin 52.9 cm and made the residual worse, 72.1 -> 112.8 cm.

    conda run -n foundationpose_vl python pipeline/tools/build_pose_bundle.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "orbslam3_baseline/data"
E2E = ROOT / "pipeline/assets/e2e"
OUT = E2E / "pose_bundle"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=None, help="defaults to ground.json's target")
    ap.add_argument("--frames", type=int, default=8, help="frames to export")
    ap.add_argument("--depth-tol", type=float, default=0.08,
                    help="metres; projected vs measured depth for the occlusion test")
    args = ap.parse_args()

    import cv2
    import open3d as o3d

    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d
    from pipeline.tools.e2e_tum import INTRINSICS
    from pipeline.transforms import quaternion_to_matrix

    fuse = json.load(open(E2E / "fuse.json"))
    ground = json.load(open(E2E / "ground.json"))
    labelled = json.load(open(E2E / "labelled.json"))["instances"]
    masks = np.load(E2E / "instance_masks.npz")["masks"]

    cloud = o3d.io.read_point_cloud(fuse["ply"])
    points = np.asarray(cloud.points)
    colours = np.asarray(cloud.colors)
    if masks.shape[1] != len(points):
        raise SystemExit(f"masks index {masks.shape[1]} points, cloud has {len(points)}")

    seq = DATA / fuse["sequence"]
    fx, fy, cx, cy = INTRINSICS[fuse["camera"]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T_level = np.asarray(fuse["T_level_slam"], float) if fuse.get("levelled") else np.eye(4)

    # Rebuild identity the same way every other stage does, so `target` means the same
    # thing here as in ground.json.
    keep = [i for i, r in enumerate(labelled) if r.get("label")]
    observations = from_openmask3d(
        [np.flatnonzero(masks[i]) for i in keep], points,
        [labelled[i]["label"] for i in keep], [labelled[i]["similarity"] for i in keep],
        anchor_frame=Frames.keyframe(0), stamp_ns=10**9, registry=InstanceRegistry())
    target = args.target or ground["target"]
    row = next((keep[k] for k, o in enumerate(observations) if o.node_id == target), None)
    if row is None:
        raise SystemExit(f"{target!r} not among {[o.node_id for o in observations]}")

    member = masks[row]
    inst_pts, inst_col = points[member], colours[member]
    centroid = inst_pts.mean(axis=0)
    print(f"[bundle]  target {target}: {len(inst_pts)} points, extent "
          f"{np.round(inst_pts.max(0) - inst_pts.min(0), 2).tolist()} m")

    # ------------------------------------------------------------------ mesh
    OUT.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inst_pts - centroid)   # provisional origin
    pcd.colors = o3d.utility.Vector3dVector(inst_col)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
    # Poisson extrapolates a watertight surface well beyond the samples; clip it back to
    # the instance's own bounding box or the mesh includes invented geometry that
    # FoundationPose would happily align against.
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise SystemExit(
            "Poisson produced no triangles inside the instance box — the instance is "
            "probably too sparse to mesh. Try a denser fusion (smaller --stride)."
        )
    # The mesh keeps the point-centroid origin it was built with. FoundationPose centres
    # it internally and un-centres the answer (`estimater.py:233`), so the returned pose
    # is the centroid's position and no compensation belongs here.
    #
    # `bbox_centre` is recorded but NOT applied: it is how far a bbox-centre origin would
    # sit from this one, which is worth knowing because the Poisson surface over a
    # one-sided observed shell is lopsided (here 52.9 cm). Supplying the mesh at both
    # origins is a free rotation probe — the returned translations must differ by
    # `R_est @ bbox_centre`, so the angle against `R_world_to_cam @ bbox_centre` measures
    # the rotation error without any ground truth. That is what exposed a >= 99 deg
    # rotation error on this instance.
    verts = np.asarray(mesh.vertices)
    bbox_centre = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    mesh_origin_world = centroid
    o3d.io.write_triangle_mesh(str(OUT / "mesh.obj"), mesh)
    print(f"[bundle]  mesh {len(mesh.vertices)} verts / {len(mesh.triangles)} tris")
    print(f"[bundle]  origin = point centroid; a bbox-centre origin would sit "
          f"{np.linalg.norm(bbox_centre)*100:.1f} cm away")

    # ---------------------------------------------------------------- frames
    traj = {}
    for line in (DATA / fuse["trajectory"]).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        traj[v[0]] = T_level @ T
    stamps = np.array(sorted(traj))

    rgb = [(float(l.split()[0]), l.split()[1])
           for l in (seq / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(float(l.split()[0]), l.split()[1])
           for l in (seq / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    dstamps = np.array([s for s, _ in dep])

    written, records = 0, []
    for stamp, rel in rgb:
        if written >= args.frames:
            break
        j = int(np.argmin(np.abs(stamps - stamp)))
        k = int(np.argmin(np.abs(dstamps - stamp)))
        if abs(stamps[j] - stamp) > 0.02 or abs(dstamps[k] - stamp) > 0.02:
            continue
        T = traj[stamps[j]]
        depth = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        bgr = cv2.imread(str(seq / rel))
        h, w = depth.shape

        world_to_cam = np.linalg.inv(T)
        cam = inst_pts @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        z = cam[:, 2]
        front = z > 1e-3
        u = np.full(len(cam), -1.0); v = np.full(len(cam), -1.0)
        u[front] = fx * cam[front, 0] / z[front] + cx
        v[front] = fy * cam[front, 1] / z[front] + cy
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if inside.sum() < 50:
            continue
        measured = np.zeros(len(cam))
        measured[inside] = depth[vi[inside], ui[inside]]
        # Occlusion test: a projected point only counts if the depth map agrees. Without
        # it the mask covers whatever sits on the far side of the room along that ray.
        vis = inside & (measured > 0) & (np.abs(measured - z) < args.depth_tol)
        if vis.sum() < 50:
            continue

        ob_mask = np.zeros((h, w), np.uint8)
        ob_mask[vi[vis], ui[vis]] = 255
        ob_mask = cv2.morphologyEx(ob_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        cv2.imwrite(str(OUT / f"rgb_{written:03d}.png"), bgr)
        cv2.imwrite(str(OUT / f"depth_{written:03d}.png"),
                    np.clip(depth * 5000.0, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(OUT / f"mask_{written:03d}.png"), ob_mask)
        records.append({
            "frame": written, "stamp": stamp, "rgb": rel,
            # The trajectory stamp. On this sequence it EQUALS the image stamp, because
            # ORB-SLAM3 emits a pose per rgb frame — so `refine_with_pose`'s staleness
            # guard is trivially satisfied here and this field proves nothing today. It
            # is kept separate anyway so a source whose poses arrive on their own clock
            # is carried correctly rather than silently passing one stamp twice.
            "cam_stamp": float(stamps[j]),
            "cam_to_world": T.tolist(),
            "mask_pixels": int((ob_mask > 0).sum()),
            "visible_points": int(vis.sum()),
            # Where our own map says the object is, in the camera frame — the position
            # FoundationPose's answer is compared against. Not ground truth: it is the
            # segmentation's opinion, which is exactly what makes the comparison
            # informative when the two disagree.
            "mesh_origin_cam": (world_to_cam[:3, :3] @ mesh_origin_world
                                + world_to_cam[:3, 3]).tolist(),
            # The mesh is cut from the world cloud unrotated, so its frame IS the world
            # frame up to translation. The pose's rotation should therefore reproduce
            # this, which makes rotation error measurable without ground truth.
            "R_world_to_cam": world_to_cam[:3, :3].tolist(),
        })
        written += 1

    if not written:
        raise SystemExit("no frame saw the target with enough visible points")

    json.dump({
        "ok": True, "target": target, "label": labelled[row].get("label"),
        "sequence": fuse["sequence"], "camera": fuse["camera"],
        "K": K.tolist(), "depth_scale": 5000.0,
        "mesh": "mesh.obj", "point_centroid_world": centroid.tolist(),
        "mesh_origin_world": mesh_origin_world.tolist(),
        "bbox_centre_instance": bbox_centre.tolist(),
        "instance_points": int(len(inst_pts)),
        # The pose stage's depth check needs to know how deep the object is: a centroid
        # legitimately sits behind the visible face, but only by so much.
        "extent_m": (inst_pts.max(0) - inst_pts.min(0)).tolist(),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
        "frames": records,
    }, open(OUT / "bundle.json", "w"), indent=1)

    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"[bundle]  {written} frames, {total / 1e6:.1f} MB -> {OUT}")
    print(f"[bundle]  mask pixels per frame: "
          f"{[r['mask_pixels'] for r in records]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

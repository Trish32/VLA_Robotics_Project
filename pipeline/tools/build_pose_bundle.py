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
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "orbslam3_baseline/data"
# E2E_DIR lets a second scene run the same stages without overwriting the first
# (the chair bundle and its Kaggle results are keyed to what is in the default).
E2E = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))
OUT = E2E / "pose_bundle"



def project_instance(inst_pts, world_to_cam, K, depth, depth_tol):
    """Project the instance into one frame and test every point against measured depth.

    Returns `(ui, vi, z, corroborated)`. A point is corroborated only if the camera
    measured something along that ray AND measured it at the point's own range.

    This is usually described as an occlusion test, which undersells it: the test is
    two-sided. A point BEHIND the measurement is occluded, which is ordinary and
    expected. A point IN FRONT of the measurement is not occluded by anything — it is
    in the wrong place, because had the object really been there it is what the depth
    camera would have hit. Reporting only the pass rate hides which of the two is
    happening, so the caller gets the signed residual as well.
    """
    import numpy as np

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    cam = inst_pts @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam[:, 2]
    front = z > 1e-3
    safe = np.where(front, z, 1.0)
    ui = np.round(fx * cam[:, 0] / safe + cx).astype(int)
    vi = np.round(fy * cam[:, 1] / safe + cy).astype(int)
    inside = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    measured = np.zeros(len(cam))
    measured[inside] = depth[vi[inside], ui[inside]]
    has_depth = inside & (measured > 0)
    corroborated = has_depth & (np.abs(measured - z) < depth_tol)
    residual = np.where(has_depth, z - measured, np.nan)
    return ui, vi, z, corroborated, residual


def dense_mask(ui, vi, z, corroborated, depth, depth_tol, radius=11):
    """A depth-verified REGION mask, rather than a splat of projected points.

    The instance cloud is voxel-sparse: a 1.58 m object carries ~1,100 points, so
    splatting them covers well under 1% of the area the object subtends and a
    morphological close can only guess at the rest. Instead, rasterise the instance's
    own depth into a sparse z-buffer, spread it to a local nearest-surface estimate,
    and keep every pixel whose MEASURED depth agrees with that estimate.

    The fill is therefore bounded by the depth image rather than by a structuring
    element: interiors between projected points get filled because the camera measured
    the same surface there, and background does not, because it did not.
    """
    import cv2
    import numpy as np

    h, w = depth.shape
    zbuf = np.full((h, w), np.inf, np.float32)
    np.minimum.at(zbuf, (vi[corroborated], ui[corroborated]), z[corroborated])
    known = np.isfinite(zbuf)
    kernel = np.ones((radius, radius), np.uint8)
    # Nearest-surface estimate = min-filter over the neighbourhood, which OpenCV spells
    # as a dilation of the negated image. Unknown pixels are padded far away so they
    # never win the minimum.
    padded = np.where(known, zbuf, 1e3).astype(np.float32)
    surface = -cv2.dilate(-padded, kernel)
    support = cv2.dilate(known.astype(np.uint8), kernel) > 0
    mask = (support & (depth > 0) & (np.abs(depth - surface) < depth_tol))
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    # One object, one region. The depth test can corroborate a far-away surface that
    # happens to sit at the same range, and that blob is not the instance.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    return mask


def select_frames(scored, want, min_fraction, min_step=0.03, max_step=0.15):
    """Pick frames by what the depth image says is visible, not by arrival order.

    The rule this replaces — take the first frames clearing a floor of 50 projected
    points — is close to the worst available rule on a SLAM-posed sequence. 50 of 1,142
    points is a 4% floor, so it is cleared by frames in which the object is barely in
    view, and the earliest frames are also where the trajectory has had least time to
    converge. Measured on fr3/walking_xyz it chose eight frames showing 11-17% of the
    instance, against a sequence best of 73%.

    The selection is shaped by how the frames are consumed, which is not symmetric:
    `register()` sees ONLY frame 0, and `track_one()` refines frame to frame from there.
    So

      * frame 0 is the anchor, and it goes to the best-corroborated view available;
      * the rest form a forward chain, each within `max_step` of the last, because a
        tracker handed a metre-long jump is not tracking;
      * subject to that, each step is at least `min_step`, because eight consecutive
        frames of a slow handheld camera span 3.4 cm and 1.1 deg. That is one viewpoint
        recorded eight times: it gives the tracker nothing to resolve, and it makes the
        world-frame-spread consistency check nearly vacuous, since a tracker ignoring
        its input entirely would also hold still across those frames.

    Anchors are tried best-first and the first that can fill the chain wins; if none
    can, the longest chain found is returned rather than nothing.
    """
    import numpy as np

    usable = sorted((s for s in scored if s["fraction"] >= min_fraction),
                    key=lambda s: s["index"])
    if not usable:
        return []
    centres = {s["index"]: np.asarray(s["centre"], float) for s in usable}

    def chain_from(anchor_pos):
        chain = [usable[anchor_pos]]
        for cand in usable[anchor_pos + 1:]:
            if len(chain) >= want:
                break
            step = float(np.linalg.norm(centres[cand["index"]]
                                        - centres[chain[-1]["index"]]))
            if step < min_step:
                continue            # too similar to the view already held
            if step > max_step:
                break               # the camera has left; tracking would not survive
            chain.append(cand)
        return chain

    order = sorted(range(len(usable)), key=lambda i: -usable[i]["visible"])
    best_chain = []
    for pos in order:
        chain = chain_from(pos)
        if len(chain) >= want:
            return chain
        if len(chain) > len(best_chain):
            best_chain = chain
    return best_chain



def bundle_fingerprint(meta: dict) -> str:
    """A hash of everything about this bundle that a pose result depends on.

    `pose_result.json` is produced on a T4, hours later, and copied back by hand. Nothing
    else connected the two: the pose stage checked only that the target NAME matched, so
    a rebuilt bundle and an old result compared equal and the stale pose was read as
    current. That is [S22] from the simulator track, in a different file — comparability
    has to be a property of the recorded artefacts, not of remembering which run produced
    what.

    Covers the mesh, the frame identities and their poses. Deliberately not the image
    bytes: re-encoding a PNG should not invalidate a result, moving a camera should.
    """
    import hashlib

    payload = json.dumps({
        "target": meta["target"],
        "sequence": meta["sequence"],
        "mesh": [meta["mesh_vertices"], meta["mesh_triangles"]],
        "origin": [round(v, 6) for v in meta["mesh_origin_world"]],
        "frames": [[f["source_index"], f["mask_pixels"],
                    [round(v, 6) for row in f["cam_to_world"] for v in row]]
                   for f in meta["frames"]],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]




def poisson_mesh(pcd, depth: int = 7):
    """A watertight surface over the instance's points, cropped back to the instance.

    Two things here are load-bearing and neither is obvious.

    **`n_threads=1` is a correctness choice, not a performance one.** Open3D's default
    (-1, all cores) makes Poisson NON-DETERMINISTIC: the same 1,128 points gave 9,657 /
    9,655 / 9,656 triangles on three consecutive builds, and run in a tight loop it does
    not merely vary — it ABORTS the process, "Failed to close loop" out of
    `FEMTree.IsoSurface` through `libc++abi`, so there is no traceback and no cleanup.
    Measured: 12/12 identical vertex hashes single-threaded, against an abort inside 12
    runs multi-threaded. A non-reproducible mesh also defeats the Debug Rule, which turns
    on diffing against pristine upstream on the SAME saved inputs, and it makes the
    bundle fingerprint meaningless. The instance is ~1k points; the thread is not missed.

    **The crop is not cosmetic.** Poisson extrapolates a watertight surface well beyond
    its samples, so without clipping back to the instance's own bounding box the mesh
    carries invented geometry that FoundationPose will happily align against.
    """
    import open3d as o3d

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    return mesh


def load_sequence(fuse: dict):
    """(posed-camera lookup, rgb index, depth index) for the fused sequence.

    Shared with `survey_instances.py`, which asks the same question of every instance
    that this file asks of one.
    """
    import numpy as np

    from pipeline.transforms import quaternion_to_matrix

    seq = DATA / fuse["sequence"]
    T_level = np.asarray(fuse["T_level_slam"], float) if fuse.get("levelled") else np.eye(4)
    traj = {}
    for line in (DATA / fuse["trajectory"]).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        traj[v[0]] = T_level @ T
    rgb = [(float(l.split()[0]), l.split()[1])
           for l in (seq / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(float(l.split()[0]), l.split()[1])
           for l in (seq / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    return seq, traj, np.array(sorted(traj)), rgb, dep, np.array([s for s, _ in dep])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=None, help="defaults to ground.json's target")
    ap.add_argument("--frames", type=int, default=8, help="frames to export")
    ap.add_argument("--depth-tol", type=float, default=0.08,
                    help="metres; projected vs measured depth corroboration tolerance")
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="refuse a frame unless this fraction of the instance's points "
                         "are corroborated by measured depth. The old rule was an "
                         "absolute floor of 50 points (4%% here), which admitted frames "
                         "showing almost none of the object")
    ap.add_argument("--min-step", type=float, default=0.03,
                    help="metres of camera motion between kept frames; raises the "
                         "baseline so multi-view consistency means something")
    ap.add_argument("--max-step", type=float, default=0.15,
                    help="metres; `track_one` refines frame to frame, so the chain "
                         "must not jump further than a tracker can follow")
    ap.add_argument("--exclude-fused-stride", type=int, default=0,
                    help="skip frames whose index is a multiple of N — the frames "
                         "e2e_tum.py fused at --stride N — so every bundle frame is HELD "
                         "OUT from building the mesh. 0 (default) keeps the old behaviour")
    ap.add_argument("--scan-stride", type=int, default=1,
                    help="scan every Nth frame when ranking visibility; the scan reads "
                         "depth only, so 1 is affordable on an 859-frame sequence")
    ap.add_argument("--mask-source", choices=["depth", "sam"], default="depth",
                    help="depth: the depth-verified dense fill (default, unchanged). sam: SAM "
                         "on each bundle frame, prompted by the box of the projected points — "
                         "NOT built from depth agreement, so the gate's depth check is not "
                         "partly true by construction. Needs segment_anything (openmask3d_vl)")
    ap.add_argument("--splat-mask", action="store_true",
                    help="fall back to the old splat+close mask instead of the "
                         "depth-verified dense fill, for A/B against a recorded run")
    args = ap.parse_args()

    import cv2
    import open3d as o3d

    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d, reject_outliers
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
    # An instance can have MORE than one proposal: `InstanceRegistry.associate` merges
    # overlapping ones under a single id (on the TUM run two desk proposals at IoU 0.498
    # became one `desk_1`). Taking `next(...)` kept the first and silently dropped the
    # rest — for `desk_1` that is 4,641 of 9,841 points, so the mesh would be cut from
    # half the object while every printed number looked ordinary. `chair_4` has one
    # proposal, so this changes nothing today; it changes everything for a target that
    # does not.
    rows = [keep[k] for k, o in enumerate(observations) if o.node_id == target]
    if not rows:
        raise SystemExit(f"{target!r} not among {[o.node_id for o in observations]}")
    row = rows[0]
    if len(rows) > 1:
        print(f"[bundle]  {target} has {len(rows)} merged proposals; using their union")

    member = np.unique(np.concatenate([np.flatnonzero(masks[r]) for r in rows]))
    raw_extent = (points[member].max(0) - points[member].min(0))
    # The same density-based rejection every other stage applies. It was missing here,
    # so the mesh was cut from the raw member set: 13 strays out of 1,142 added 40 cm to
    # this instance's height, and Poisson is cropped to the instance's bounding box, so
    # those strays bought the mesh a slab of invented surface in empty space for
    # FoundationPose to align against. `extent_m` is also the pose gate's depth
    # tolerance, which the same strays inflated.
    proposal_points = len(member)
    member = member[reject_outliers(points[member])]
    inst_pts, inst_col = points[member], colours[member]
    centroid = inst_pts.mean(axis=0)
    extent = inst_pts.max(0) - inst_pts.min(0)
    print(f"[bundle]  target {target}: {len(inst_pts)} points "
          f"({proposal_points - len(member)} rejected), extent "
          f"{np.round(extent, 2).tolist()} m (raw {np.round(raw_extent, 2).tolist()})")

    # ------------------------------------------------------------------ mesh
    OUT.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inst_pts - centroid)   # provisional origin
    pcd.colors = o3d.utility.Vector3dVector(inst_col)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    mesh = poisson_mesh(pcd, depth=7)
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
    _, traj, stamps, rgb, dep, dstamps = load_sequence(fuse)

    # -- pass 1: rank every frame by how much of the instance the depth image
    # corroborates. This is the measurement the old builder never took: it accepted
    # frames in arrival order and so paid no attention to whether the object was in
    # view. Reading depth only keeps the scan cheap.
    scored, aligned = [], []
    for index, (stamp, rel) in enumerate(rgb):
        if index % args.scan_stride:
            continue
        # A frame the mesh was fused from cannot test the mesh: its depth is already in it.
        if args.exclude_fused_stride and index % args.exclude_fused_stride == 0:
            continue
        j = int(np.argmin(np.abs(stamps - stamp)))
        k = int(np.argmin(np.abs(dstamps - stamp)))
        if abs(stamps[j] - stamp) > 0.02 or abs(dstamps[k] - stamp) > 0.02:
            continue
        depth = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        T = traj[stamps[j]]
        _, _, _, ok, residual = project_instance(
            inst_pts, np.linalg.inv(T), K, depth, args.depth_tol)
        aligned.append((index, rel, stamp, float(stamps[j]), k, T))
        scored.append({
            "index": index, "visible": int(ok.sum()),
            "fraction": float(ok.mean()), "centre": T[:3, 3].tolist(),
            "median_residual_m": float(np.nanmedian(residual)) if np.isfinite(residual).any() else float("nan"),
        })

    if not scored:
        raise SystemExit("no frame had both a pose and a depth image within 20 ms")
    best = max(s["fraction"] for s in scored)
    chosen = select_frames(scored, args.frames, args.min_visible,
                           args.min_step, args.max_step)
    print(f"[bundle]  scanned {len(scored)} frames: best {best:.0%} of the instance "
          f"corroborated, median {np.median([s['fraction'] for s in scored]):.0%}")
    if not chosen:
        # Refusing beats emitting a bundle nobody can tell is bad. A pose estimated
        # from a frame that shows a tenth of the object is wrong in a way the pose
        # stage's own self-consistency check cannot detect.
        raise SystemExit(
            f"no frame reaches --min-visible {args.min_visible:.0%}; the best is "
            f"{best:.0%}. Either this instance is never properly in view, or the "
            f"trajectory does not agree with the map. Do not lower the threshold "
            f"without looking at the signed residuals first — a point IN FRONT of "
            f"the measured surface is a placement error, not an occlusion.")

    by_index = {a[0]: a for a in aligned}
    span = 0.0
    if len(chosen) > 1:
        centres = np.array([c["centre"] for c in chosen])
        span = float(max(np.linalg.norm(a - b) for a in centres for b in centres))
    steps = [float(np.linalg.norm(np.asarray(b["centre"]) - np.asarray(a["centre"])))
             for a, b in zip(chosen, chosen[1:])]
    print(f"[bundle]  selected {len(chosen)} frames at "
          f"{[round(c['fraction'], 2) for c in chosen]} visible")
    print(f"[bundle]  anchor (register) frame {chosen[0]['index']} at "
          f"{chosen[0]['fraction']:.0%}; baseline span {span * 100:.1f} cm, "
          f"max tracking step {max(steps) * 100 if steps else 0:.1f} cm")
    if len(chosen) < args.frames:
        print(f"[bundle]  WARNING: only {len(chosen)} of {args.frames} frames; the "
              f"usable run ends before the chain fills")

    # -- pass 2: write the selected frames.
    predictor = None
    if args.mask_source == "sam":
        from segment_anything import SamPredictor, sam_model_registry
        predictor = SamPredictor(sam_model_registry["vit_h"](
            checkpoint=str(ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth")))
    written, records = 0, []
    for slot, pick in enumerate(chosen):
        index, rel, stamp, cam_stamp, k, T = by_index[pick["index"]]
        depth = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        bgr = cv2.imread(str(seq / rel))
        h, w = depth.shape
        world_to_cam = np.linalg.inv(T)
        ui, vi, z, vis, _ = project_instance(
            inst_pts, world_to_cam, K, depth, args.depth_tol)

        if predictor is not None:
            pu, pv = ui[vis], vi[vis]
            box = np.array([pu.min() - 8, pv.min() - 8, pu.max() + 8, pv.max() + 8], float)
            predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            sm, _, _ = predictor.predict(box=box, multimask_output=False)
            ob_mask = sm[0].astype(np.uint8) * 255
        elif args.splat_mask:
            ob_mask = np.zeros((h, w), np.uint8)
            ob_mask[vi[vis], ui[vis]] = 255
            ob_mask = cv2.morphologyEx(ob_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        else:
            ob_mask = dense_mask(ui, vi, z, vis, depth, args.depth_tol)

        cv2.imwrite(str(OUT / f"rgb_{written:03d}.png"), bgr)
        cv2.imwrite(str(OUT / f"depth_{written:03d}.png"),
                    np.clip(depth * 5000.0, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(OUT / f"mask_{written:03d}.png"), ob_mask)
        records.append({
            "frame": written, "stamp": stamp, "rgb": rel,
            "source_index": index,
            # The trajectory stamp. On this sequence it EQUALS the image stamp, because
            # ORB-SLAM3 emits a pose per rgb frame — so `refine_with_pose`'s staleness
            # guard is trivially satisfied here and this field proves nothing today. It
            # is kept separate anyway so a source whose poses arrive on their own clock
            # is carried correctly rather than silently passing one stamp twice.
            "cam_stamp": cam_stamp,
            "cam_to_world": T.tolist(),
            "mask_pixels": int((ob_mask > 0).sum()),
            "visible_points": int(vis.sum()),
            # What fraction of the instance this frame actually shows. Recorded so a
            # later reader can tell a pose estimated from a good view from one
            # estimated from a sliver, which `mask_pixels` alone does not say.
            "visible_fraction": float(vis.mean()),
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

    meta = {
        "ok": True, "target": target, "label": labelled[row].get("label"),
        "sequence": fuse["sequence"], "camera": fuse["camera"],
        "K": K.tolist(), "depth_scale": 5000.0,
        "mesh": "mesh.obj", "point_centroid_world": centroid.tolist(),
        "mesh_origin_world": mesh_origin_world.tolist(),
        "bbox_centre_instance": bbox_centre.tolist(),
        "instance_points": int(len(inst_pts)),
        # The pose stage's depth check needs to know how deep the object is: a centroid
        # legitimately sits behind the visible face, but only by so much.
        "extent_m": extent.tolist(),
        "extent_m_raw": raw_extent.tolist(),
        "points_rejected": int(proposal_points - len(member)),
        "proposals_merged": len(rows),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
        # How the frames were chosen, so a reader can audit the selection rather than
        # trusting it. `scanned` vs `selected` is the ratio the old builder hid.
        "selection": {
            "rule": "best-corroborated anchor, then a forward chain within --max-step",
            "scanned": len(scored), "selected": len(chosen),
            "min_visible": args.min_visible,
            "anchor_index": chosen[0]["index"],
            "anchor_fraction": chosen[0]["fraction"],
            "max_tracking_step_m": max(steps) if steps else 0.0,
            "best_fraction": best,
            "baseline_span_m": span,
            "mask": ("SAM per frame, box prompt" if args.mask_source == "sam"
                     else "splat+close" if args.splat_mask else "depth-verified dense fill"),
            "held_out_from_fusion": bool(args.exclude_fused_stride),
            "excluded_fused_stride": args.exclude_fused_stride,
        },
        "frames": records,
    }
    meta["fingerprint"] = bundle_fingerprint(meta)
    json.dump(meta, open(OUT / "bundle.json", "w"), indent=1)

    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"[bundle]  {written} frames, {total / 1e6:.1f} MB -> {OUT}")
    print(f"[bundle]  fingerprint {meta['fingerprint']} — a pose result must carry "
          f"this to be read as current")
    print(f"[bundle]  mask pixels per frame: "
          f"{[r['mask_pixels'] for r in records]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

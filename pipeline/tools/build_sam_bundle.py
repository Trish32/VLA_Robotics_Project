#!/usr/bin/env python
"""A FoundationPose bundle for an object defined by a SAM click, not by our segmentation.

Why this exists. On TUM fr1/xyz, Mask3D proposes no compact desk object at any score
threshold down to 0.2 — its small instances are fragments of monitor edges and chairs;
the mouse, cup, books and boxes are never segmented. To ask whether FoundationPose works on
a COMPACT object with our trajectory and our mesh recipe, the target has to be defined
another way. This tool does that, and labels every output as SAM-defined so nobody reads
it as a result of our segmentation.

What is ours and what is not:
  * target       — a SAM mask from one click on one reference frame (NOT our segmentation)
  * 3-D points   — that mask's depth, lifted with our ORB-SLAM3 trajectory
  * mesh         — TSDF-fused along the trajectory from the FUSION frames only, then the
                   chair's recipe at the chair's relative resolution (mesh_control.py:
                   ~79 voxels across, ~1.1k points, our reject_outliers and poisson_mesh)
  * bundle frames— HELD OUT: none of them was fused into the mesh
  * bundle masks — SAM on each bundle frame, prompted by the box of the projected points.
                   NOT built from depth agreement. This answers a review point: the chair
                   bundle's masks were made from depth-agreeing points, so a pose's depth
                   residual on them was partly agreement-by-construction.

    E2E_DIR=pipeline/assets/e2e_fr1 conda run -n openmask3d_vl \\
        python pipeline/tools/build_sam_bundle.py --name mouse --click 535 400
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

# E2E_DIR lets a second scene run the same stages without overwriting the first.
E2E = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))
SAM_CKPT = ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth"


def lift(mask, depth_m, K, cam_to_world, erode_px=2):
    """Masked depth -> world points. The mask is eroded first: depth at a silhouette edge
    mixes object and background, and those pixels become points floating between them."""
    import cv2

    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
    v, u = np.nonzero(m & (depth_m > 0))
    z = depth_m[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--click", type=int, nargs=2, required=True, metavar=("U", "V"))
    ap.add_argument("--frame", type=int, default=90, help="reference frame index")
    ap.add_argument("--fused-stride", type=int, default=10,
                    help="the stride e2e_tum.py fused at; those frames build the mesh and "
                         "are excluded from the bundle")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-extent", type=float, default=0.30,
                    help="compact = longest extent at most this (m); registered criterion")
    ap.add_argument("--min-visible", type=float, default=0.35)
    ap.add_argument("--depth-tol", type=float, default=0.08)
    a = ap.parse_args(argv)

    import cv2
    import open3d as o3d
    from segment_anything import SamPredictor, sam_model_registry

    from foundationpose_6dof import mesh_control as mc
    from pipeline.observations import reject_outliers
    from pipeline.tools.build_pose_bundle import (bundle_fingerprint, dense_mask,
                                                  load_sequence, poisson_mesh,
                                                  project_instance, select_frames)
    from pipeline.tools.e2e_tum import INTRINSICS

    fuse = json.load(open(E2E / "fuse.json"))
    fx, fy, cx, cy = INTRINSICS[fuse["camera"]]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    seq, traj, stamps, rgb, dep, dstamps = load_sequence(fuse)

    def frame(i):
        s, rel = rgb[i]
        j = int(np.argmin(np.abs(stamps - s))); k = int(np.argmin(np.abs(dstamps - s)))
        if abs(stamps[j] - s) > 0.02 or abs(dstamps[k] - s) > 0.02:
            return None
        bgr = cv2.imread(str(seq / rel))
        d = cv2.imread(str(seq / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        return bgr, d, traj[stamps[j]], s, float(stamps[j])

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    pred = SamPredictor(sam)

    # -- 1. the target: one click, SAM's three nested masks, the smallest that lifts to a
    # plausible object. SAM's "whole" level readily swallows the desk the object sits on.
    ref = frame(a.frame)
    if ref is None:
        raise SystemExit(f"frame {a.frame} has no pose/depth within 20 ms")
    bgr, d, T, _, _ = ref
    pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = pred.predict(point_coords=np.array([a.click], float),
                                    point_labels=np.array([1]), multimask_output=True)
    chosen = None
    for m, sc in sorted(zip(masks, scores), key=lambda t: t[0].sum()):
        P = lift(m, d, K, T)
        if len(P) < 30:
            continue
        P = P[reject_outliers(P)]
        ext = P.max(0) - P.min(0)
        print(f"[sam   ]  mask {int(m.sum()):6d} px, score {sc:.3f}, lifts to "
              f"{len(P)} pts, extent {np.round(ext, 3).tolist()} m")
        if chosen is None and ext.max() <= a.max_extent:
            chosen = (m, P, ext, float(sc))
    if chosen is None:
        raise SystemExit(f"REFUSED: no SAM mask at {a.click} lifts to an object within "
                         f"{a.max_extent} m — not compact by the registered criterion")
    m0, P0, ext0, sc0 = chosen
    longest = float(ext0.max())
    print(f"[target]  {a.name}: {int(m0.sum())} px, {len(P0)} pts, longest {longest:.3f} m")

    # -- 2. mesh from the FUSION frames only, the chair's recipe at matched resolution
    fused, fuse_ids = [], []
    for i in range(0, len(rgb), a.fused_stride):
        f = frame(i)
        if f is None:
            continue
        b, dd, Ti, _, _ = f
        ui, vi, z, ok, _ = project_instance(P0, np.linalg.inv(Ti), K, dd, a.depth_tol)
        if ok.mean() < a.min_visible:
            continue
        mk = dense_mask(ui, vi, z, ok, dd, a.depth_tol) > 0
        fused.append((cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                      (dd * 1000).astype(np.uint16), mk, np.linalg.inv(Ti)))
        fuse_ids.append(i)
    if len(fused) < 3:
        raise SystemExit(f"REFUSED: only {len(fused)} fusion frames see the target")
    voxel = mc.relative(mc.CHAIR_VOXEL_M, longest)
    pts, cols = mc.fuse_object(fused, K, voxel, P0.mean(0), max(0.6 * longest, 0.1))
    v, f_, vc, centroid, n = mc.ourway_mesh(pts, cols, longest, reject_outliers, poisson_mesh)
    inst = pts[reject_outliers(pts)] if len(pts) > 30 else pts
    ext = inst.max(0) - inst.min(0)
    print(f"[mesh  ]  {len(fused)} fusion frames, voxel {voxel*1000:.2f} mm "
          f"({longest/voxel:.0f} across), {len(pts)} TSDF pts -> {n}, "
          f"mesh {len(v)} v / {len(f_)} f, extent {np.round(ext, 3).tolist()} m")
    if ext.max() > a.max_extent * 1.5:
        raise SystemExit(f"REFUSED: fused extent {ext.max():.2f} m — the mask leaked")

    # -- 3. held-out bundle frames, chosen by visibility exactly as the chair's were
    scored, cache = [], {}
    for i in range(len(rgb)):
        if i % a.fused_stride == 0:
            continue
        f = frame(i)
        if f is None:
            continue
        _, dd, Ti, _, _ = f
        *_, ok, _ = project_instance(inst, np.linalg.inv(Ti), K, dd, a.depth_tol)
        scored.append({"index": i, "visible": int(ok.sum()), "fraction": float(ok.mean()),
                       "centre": Ti[:3, 3].tolist()})
    chosen_frames = select_frames(scored, a.frames, a.min_visible)
    if len(chosen_frames) < a.frames:
        raise SystemExit(f"REFUSED: {len(chosen_frames)} held-out frames clear "
                         f"{a.min_visible:.0%}; the registered criterion needs {a.frames}")

    out = E2E / f"pose_bundle_{a.name}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f_))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vc, 0, 1))
    o3d.io.write_triangle_mesh(str(out / "mesh.obj"), mesh)

    records = []
    for slot, c in enumerate(chosen_frames):
        b, dd, Ti, stamp, cam_stamp = frame(c["index"])
        w2c = np.linalg.inv(Ti)
        ui, vi, z, ok, _ = project_instance(inst, w2c, K, dd, a.depth_tol)
        # SAM mask from the projected box — independent of depth agreement.
        pu, pv = ui[ok], vi[ok]
        box = np.array([pu.min() - 8, pv.min() - 8, pu.max() + 8, pv.max() + 8], float)
        pred.set_image(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        ms, ss, _ = pred.predict(box=box, multimask_output=False)
        sam_m = ms[0]
        dep_m = dense_mask(ui, vi, z, ok, dd, a.depth_tol) > 0
        iou = float((sam_m & dep_m).sum() / max((sam_m | dep_m).sum(), 1))
        cv2.imwrite(str(out / f"rgb_{slot:03d}.png"), b)
        cv2.imwrite(str(out / f"depth_{slot:03d}.png"),
                    np.clip(dd * 5000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / f"mask_{slot:03d}.png"), sam_m.astype(np.uint8) * 255)
        records.append({
            "frame": slot, "stamp": stamp, "rgb": rgb[c["index"]][1],
            "source_index": c["index"], "cam_stamp": cam_stamp,
            "cam_to_world": Ti.tolist(), "mask_pixels": int(sam_m.sum()),
            "visible_points": int(ok.sum()), "visible_fraction": float(ok.mean()),
            "sam_score": float(ss[0]), "sam_vs_depth_mask_iou": iou,
            "mesh_origin_cam": (w2c[:3, :3] @ centroid + w2c[:3, 3]).tolist(),
            "R_world_to_cam": w2c[:3, :3].tolist()})
        print(f"[frame ]  {c['index']:4d}  visible {ok.mean():.0%}  SAM {int(sam_m.sum())} px "
              f"(score {ss[0]:.2f}), IoU with the depth-built mask {iou:.2f}")

    span = max(float(np.linalg.norm(np.asarray(x["centre"]) - np.asarray(y["centre"])))
               for x in chosen_frames for y in chosen_frames)
    meta = {
        "ok": True, "target": a.name, "label": a.name,
        "target_source": f"SAM click {a.click} on frame {a.frame} — NOT our segmentation",
        "mask_source": "SAM per bundle frame, box prompt — not built from depth agreement",
        "sequence": fuse["sequence"], "camera": fuse["camera"], "K": K.tolist(),
        "depth_scale": 5000.0, "mesh": "mesh.obj",
        "point_centroid_world": centroid.tolist(), "mesh_origin_world": centroid.tolist(),
        "instance_points": int(len(inst)), "extent_m": ext.tolist(),
        "mesh_vertices": int(len(v)), "mesh_triangles": int(len(f_)),
        "fusion_frames": fuse_ids, "voxel_m": voxel,
        "selection": {"rule": "best-corroborated anchor, then a forward chain",
                      "held_out_from_fusion": True, "excluded_fused_stride": a.fused_stride,
                      "scanned": len(scored), "selected": len(chosen_frames),
                      "min_visible": a.min_visible, "baseline_span_m": span},
        "frames": records,
    }
    meta["fingerprint"] = bundle_fingerprint(meta)
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "foundationpose-bundle", "id": "trishli/foundationpose-bundle",
         "licenses": [{"name": "CC0-1.0"}]}))
    json.dump(meta, open(out / "bundle.json", "w"), indent=1)
    print(f"[bundle]  {out.relative_to(ROOT)}  fingerprint {meta['fingerprint']}, "
          f"baseline {span*100:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

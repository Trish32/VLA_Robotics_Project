#!/usr/bin/env python
"""Stage 4: fold FoundationPose's 6-DoF pose into the world model — or refuse to.

Stages 1-3 give every instance a position-only pose: an axis-aligned box centre, with
identity rotation, because segmentation measures *where* something is and not *how it is
turned*. This stage replaces that with a real 6-DoF pose via `refine_with_pose`.

It is a gate, not a pipe. A pose that is wrong by a metre is worse than no pose at all:
the position-only fallback is honestly uninformative about rotation, whereas a confident
wrong rotation is acted on. So the pose is admitted only if it agrees with what the rest
of the stack already believes, and the checks below need no ground truth:

  * **translation** — the returned pose refers to the mesh origin, which is the
    instance's point centroid (`estimater.py:233` un-centres the answer, so the origin we
    supply is the origin we get back). Our own map knows where that centroid is.
  * **rotation** — the mesh is cut out of the world cloud unrotated, so its frame is the
    world frame up to translation, and the pose's rotation should therefore reproduce
    `R_world_to_cam`. This is the check that caught the real failure on `chair_4`: the
    translation residual alone was ambiguous, but the rotation was >= 99 deg off.
  * **depth** — the object's origin cannot sit further behind the measured surface than
    the object is deep. Independent of both of the above, and of the segmentation.

Disagreement does not say which side is wrong; it says the two do not corroborate, which
is the most an ungrounded comparison can say and enough to withhold the pose.

    conda run -n foundationpose_vl python pipeline/tools/e2e_pose.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# E2E_DIR lets a second scene run the same stages without overwriting the first
# (the chair bundle and its Kaggle results are keyed to what is in the default).
E2E = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e"))
# POSE_BUNDLE points the gate at a bundle other than the scene's default one — e.g. a
# SAM-defined target from build_sam_bundle.py.
BUNDLE = Path(os.environ.get("POSE_BUNDLE", E2E / "pose_bundle"))
NS_PER_S = 10 ** 9


def check_agreement(agreement: dict | None, *, min_clustered_frac: float = 0.5,
                    max_pairwise_deg: float = 15.0) -> tuple[bool, str]:
    """Did the estimator's own top hypotheses agree on an orientation?

    The only check here that needs neither our map nor ground truth — it asks whether
    `register()` converged or merely picked. `register()` refines a grid spanning the
    whole rotation group; if the top-k land on one orientation from different starts,
    the pose is determined. If they are scattered, the scorer is choosing between
    orientations it cannot distinguish and the winner is close to arbitrary.

    Measured, same code path, same weights:

        mustard0 (correct pose)   16/16 within 15 deg, median pairwise   0.29 deg
        chair_4  (175.63 deg off)  2/16 within 15 deg, median pairwise 127.31 deg

    Note this INVERTS the score-based reading: mustard0's scores are flatter (an exact
    tie at the top, 91/252 within 1%) precisely because its hypotheses converged onto
    the same pose. Ties among converged hypotheses mean agreement, not ambiguity.

    Being map-free, this is also the check that can run on a scene with no prior — i.e.
    a pre-flight test of whether an instance is poseable at all, before a registration
    is spent on it.
    """
    if not agreement:
        return True, "no agreement data (pre-r14 pose result) — check skipped"
    k = agreement.get("k") or 0
    clustered = agreement.get("clustered_within_15deg") or 0
    pairwise = agreement.get("median_pairwise_deg")
    frac = clustered / k if k else 0.0
    ok = frac >= min_clustered_frac and (pairwise is None or pairwise <= max_pairwise_deg)
    detail = (f"{clustered}/{k} of the top hypotheses within 15 deg of the best, "
              f"median pairwise {pairwise:.2f} deg" if pairwise is not None else
              f"{clustered}/{k} clustered")
    return ok, detail


def rotation_error_deg(R_est: np.ndarray, R_ref: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    R_err = np.asarray(R_est, float) @ np.asarray(R_ref, float).T
    # arccos is defined on [-1, 1]; a trace a hair outside it is float noise, not a bug.
    cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


@dataclass
class PoseCheck:
    """One frame's verdict. `accepted` is the AND of the individual checks."""

    frame: int
    translation_cm: float
    rotation_deg: float
    behind_surface_cm: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    depth_residual_cm: float | None = None
    depth_coverage: float | None = None

    def as_row(self) -> str:
        mark = "ok  " if self.accepted else "FAIL"
        dz = ("   n/a" if self.depth_residual_cm is None
              else f"{self.depth_residual_cm:6.2f}")
        cov = "  n/a" if self.depth_coverage is None else f"{100 * self.depth_coverage:4.0f}%"
        return (f"  {mark} frame {self.frame}:  t {self.translation_cm:7.2f} cm   "
                f"R {self.rotation_deg:6.2f} deg   behind {self.behind_surface_cm:7.2f} cm   "
                f"|dz| {dz} cm  cover {cov}")


def check_pose(
    T_cam_obj: np.ndarray,
    *,
    frame: int,
    origin_cam: np.ndarray,
    R_world_to_cam: np.ndarray,
    depth_median_m: float | None,
    depth_span_m: float,
    max_translation_cm: float,
    max_rotation_deg: float,
    depth_margin_cm: float = 10.0,
    depth_residual_cm: float | None = None,
    max_depth_residual_cm: float = 8.0,
    depth_coverage: float | None = None,
    min_depth_coverage: float = 0.5,
    require_depth_evidence: bool = False,
) -> PoseCheck:
    """Corroborate one pose against the map and the depth image.

    Thresholds are grasp-relevant tolerances, not statistical ones: they answer "is this
    pose good enough to reach for" rather than "is this pose significantly different".
    """
    T = np.asarray(T_cam_obj, float).reshape(4, 4)

    # Validity first. Every comparison below is a `>` against a threshold, and NaN fails
    # every comparison — so a NaN pose used to pass every check. And the rotation metric
    # clips its arccos argument, so a matrix that is not a rotation at all (2I) read as
    # 0 deg from the reference. Both were found by review, not by a run.
    invalid = []
    if not np.isfinite(T).all():
        invalid.append("pose is not finite")
    else:
        R = T[:3, :3]
        if (np.abs(R.T @ R - np.eye(3)).max() > 1e-3
                or abs(np.linalg.det(R) - 1.0) > 1e-3):
            invalid.append("rotation block is not a rotation (R^T R != I or det != 1)")
    if invalid:
        return PoseCheck(frame=frame, translation_cm=float("nan"),
                         rotation_deg=float("nan"), behind_surface_cm=float("nan"),
                         accepted=False, reasons=invalid,
                         depth_residual_cm=depth_residual_cm, depth_coverage=depth_coverage)

    translation_cm = float(np.linalg.norm(T[:3, 3] - np.asarray(origin_cam, float)) * 100)
    rotation_deg = rotation_error_deg(T[:3, :3], R_world_to_cam)

    # How far behind the measured surface the origin sits, bounded by how deep the
    # surface the camera ACTUALLY SEES is (5th-95th percentile of masked depth).
    #
    # This used to be bounded by half the instance's 3-D extent, which is nearly inert on
    # a coarse proposal: `chair_4` spans 1.58 m, so the tolerance was 79 cm and a 60.8 cm
    # error passed. The visible surface is a 13.4 cm slab. Using the observation rather
    # than the segmentation's opinion of the object's size also keeps this check
    # independent of the segmentation, which is the only reason it is worth having
    # alongside the translation check.
    if depth_median_m is None:
        behind_cm, depth_bad = 0.0, False
    else:
        behind_cm = float((T[2, 3] - depth_median_m) * 100)
        depth_bad = behind_cm > depth_span_m * 100 + depth_margin_cm

    reasons = []
    if translation_cm > max_translation_cm:
        reasons.append(f"translation {translation_cm:.1f} cm > {max_translation_cm:.0f} cm")
    if rotation_deg > max_rotation_deg:
        reasons.append(f"rotation {rotation_deg:.1f} deg > {max_rotation_deg:.0f} deg")
    if depth_bad:
        reasons.append(f"origin {behind_cm:.1f} cm behind the measured surface, which is "
                       f"only {depth_span_m*100:.1f} cm deep")
    # The one check here that needs NEITHER our map NOR the segmentation's centroid: does
    # the mesh, rendered at this pose, reproduce the depth the camera measured under the
    # mask? A pose can sit at the right distance and still be turned so that the surface
    # it predicts is nowhere near the one observed. On chair_4 the map pose reads
    # 1.3-6.0 cm and FoundationPose's register() pose 74-166 cm on the SAME pixels.
    # 8 cm is the bundle's existing corroboration tolerance (`--depth-tol`), not a value
    # tuned to those numbers.
    # A residual is evidence only about the pixels it was computed on. Without a floor on
    # how many, a pose covering 1 of 25 valid pixels read 0 cm. The floor — half the
    # observed object — was fixed before it was run on any recorded result.
    if depth_coverage is not None and depth_coverage < min_depth_coverage:
        reasons.append(f"rendered mesh covers {100 * depth_coverage:.0f}% of the measured "
                       f"pixels under the mask < {100 * min_depth_coverage:.0f}%")
    if require_depth_evidence and (depth_residual_cm is None or depth_coverage is None):
        reasons.append("depth-consistency evidence unavailable, and it is required")
    if depth_residual_cm is not None and depth_residual_cm > max_depth_residual_cm:
        reasons.append(
            "rendered mesh covers no measured pixel under the mask"
            if not np.isfinite(depth_residual_cm) else
            f"rendered mesh disagrees with measured depth by {depth_residual_cm:.1f} cm "
            f"(median under the mask) > {max_depth_residual_cm:.0f} cm")

    return PoseCheck(frame=frame, translation_cm=translation_cm, rotation_deg=rotation_deg,
                     behind_surface_cm=behind_cm, accepted=not reasons, reasons=reasons,
                     depth_residual_cm=depth_residual_cm, depth_coverage=depth_coverage)


def depth_residuals_cm(poses, frames, bundle: Path, depth_scale: float, K
                       ) -> list[tuple[float | None, float | None]]:
    """Per frame: median |rendered mesh depth - measured depth| under the mask, in cm.

    None ONLY where it cannot be computed (no open3d, missing files) — the check is then
    skipped and that is printed. A pose whose rendered mesh lands on no measured pixel is
    NOT that case: it is the largest disagreement there is, and it is returned as inf so
    it fails. The first version mapped it to None, which would have let a pose rendered
    entirely off the object pass this check silently; v18's prior-tracked frames 5-7
    were the first to land there.
    """
    try:
        import cv2
        import open3d as o3d

        sys.path.insert(0, str(ROOT))
        from foundationpose_6dof.mesh_control import depth_agreement
    except ImportError:
        return [(None, None)] * len(poses)
    mesh = o3d.io.read_triangle_mesh(str(bundle / "mesh.obj"))
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    out: list[tuple[float | None, float | None]] = []
    for T, rec in zip(poses, frames):
        i = rec["frame"]
        d = cv2.imread(str(bundle / f"depth_{i:03d}.png"), cv2.IMREAD_ANYDEPTH)
        m = cv2.imread(str(bundle / f"mask_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
        if d is None or m is None:
            out.append((None, None))
            continue
        if not np.isfinite(np.asarray(T, float)).all():
            out.append((float("inf"), 0.0))          # nothing to render; check_pose refuses
            continue
        dm = np.where(m > 0, d.astype(np.float32) / depth_scale, 0.0)
        r, cov = depth_agreement(V, F, np.asarray(T, float), np.asarray(K, float), dm)
        out.append((float("inf") if not np.isfinite(r) else r * 100, cov))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=Path, default=BUNDLE / "pose_result.json",
                    help="FoundationPose output; written by the Kaggle kernel")
    ap.add_argument("--max-translation-cm", type=float, default=10.0)
    ap.add_argument("--max-rotation-deg", type=float, default=30.0)
    ap.add_argument("--max-depth-residual-cm", type=float, default=8.0,
                    help="median |rendered mesh depth - measured depth| under the mask; "
                         "8 cm is the bundle's existing --depth-tol")
    ap.add_argument("--min-depth-coverage", type=float, default=0.5,
                    help="fraction of measured pixels under the mask the rendered mesh must "
                         "cover; fixed before being run on any recorded result")
    ap.add_argument("--out", type=Path, default=E2E / "pose.json")
    ap.add_argument("--force", action="store_true",
                    help="write the refined pose even if the checks fail (records that "
                         "it was forced, so nothing downstream can mistake it for clean)")
    args = ap.parse_args()

    from pipeline.identity import Frames, InstanceRegistry
    from pipeline.observations import from_openmask3d, refine_with_pose

    meta = json.load(open(BUNDLE / "bundle.json"))
    if not args.poses.exists():
        print(f"[4 pose ]  no pose result at {args.poses}")
        print("[4 pose ]  FoundationPose needs CUDA; this stage consumes what the "
              "Kaggle kernel writes.\n"
              "           run:  foundationpose_6dof/.kaggle/fp.py  (see its header)\n"
              "           then: place pose_result.json beside bundle.json")
        return 2

    result = json.load(open(args.poses))
    poses = [np.asarray(p, float).reshape(4, 4) for p in result["poses_cam_obj"]]
    frames = meta["frames"][:len(poses)]
    if not poses:
        raise SystemExit("pose_result.json has no poses")
    if result.get("target") != meta["target"]:
        raise SystemExit(f"pose result is for {result.get('target')!r}, bundle is for "
                         f"{meta['target']!r} — rebuild one of them")

    # The pose is computed on a T4 and copied back by hand, so nothing but this check
    # ties the result to the input it was computed from. Matching target names is not
    # enough: a rebuilt bundle keeps its target and changes every frame under it. See
    # `build_pose_bundle.bundle_fingerprint`.
    want = meta.get("fingerprint")
    got = result.get("bundle_fingerprint")
    if want and got != want:
        print(f"[4 pose ]  STALE: this result came from bundle "
              f"{got or 'an un-fingerprinted build'}; the bundle on disk is {want}.")
        print("[4 pose ]  refusing to read it as current. Re-run the Kaggle kernel "
              "against the rebuilt bundle, or check out the bundle it was run on.")
        return 2

    print(f"[4 pose ]  {meta['target']} ({meta['label']}), {len(poses)} frames "
          f"from {result.get('source', 'unknown source')}")

    # Depth medians under the mask, if the bundle's depth images are still on disk. They
    # are what makes the third check independent of the segmentation.
    depth_stats: list[tuple[float | None, float]] = []
    try:
        import cv2
        for rec in frames:
            i = rec["frame"]
            d = cv2.imread(str(BUNDLE / f"depth_{i:03d}.png"), cv2.IMREAD_ANYDEPTH)
            m = cv2.imread(str(BUNDLE / f"mask_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
            z = (d.astype(np.float32) / meta["depth_scale"])[(m > 0) & (d > 0)]
            if not z.size:
                depth_stats.append((None, 0.0))
                continue
            p5, p95 = np.percentile(z, [5, 95])
            depth_stats.append((float(np.median(z)), float(p95 - p5)))
    except ImportError:
        depth_stats = [(None, 0.0)] * len(frames)
        print("[4 pose ]  no cv2 — skipping the depth check")

    residuals = depth_residuals_cm(poses, frames, BUNDLE, meta["depth_scale"], meta["K"])
    if all(r is None for r, _ in residuals):
        print("[4 pose ]  depth-consistency check unavailable (needs open3d and the bundle "
              "images) — it is REQUIRED, so every frame is refused")
    checks = [
        check_pose(T, frame=rec["frame"],
                   origin_cam=rec["mesh_origin_cam"],
                   R_world_to_cam=np.asarray(rec["R_world_to_cam"], float),
                   depth_median_m=dm, depth_span_m=span,
                   max_translation_cm=args.max_translation_cm,
                   max_rotation_deg=args.max_rotation_deg,
                   depth_residual_cm=r, depth_coverage=cov,
                   max_depth_residual_cm=args.max_depth_residual_cm,
                   min_depth_coverage=args.min_depth_coverage,
                   require_depth_evidence=True)
        for T, rec, (dm, span), (r, cov) in zip(poses, frames, depth_stats, residuals)
    ]
    for c in checks:
        print(c.as_row())

    n_ok = sum(c.accepted for c in checks)
    print(f"\n[4 pose ]  {n_ok}/{len(checks)} frames corroborate the map")
    print(f"[4 pose ]  median translation {np.nanmedian([c.translation_cm for c in checks]):.2f} cm"
          f"   median rotation {np.nanmedian([c.rotation_deg for c in checks]):.2f} deg")

    # A consistent tracker that is consistently wrong still fails. Self-consistency was
    # already measured upstream (world-frame spread); it is not evidence of accuracy.
    if "world_spread_cm" in result:
        print(f"[4 pose ]  tracker self-consistency {result['world_spread_cm']:.2f} cm "
              f"(says it is steady, not that it is right)")

    # A pose initialised FROM our map cannot corroborate the map: translation and rotation
    # agreeing with it is partly built in. Only the depth-consistency check is evidence
    # then, so it must be available, and there are no register() hypotheses to check.
    prior = result.get("initialization") == "map_prior"
    if prior:
        agree_ok = all(c.depth_residual_cm is not None for c in checks)
        agree_detail = ("initialised from the map: translation/rotation agreement is NOT "
                        "independent evidence; only the depth-consistency check is"
                        + ("" if agree_ok else " — and it is unavailable, so refused"))
        print(f"[4 pose ]  {agree_detail}")
    else:
        # The estimator's own self-agreement, which needs neither our map nor ground truth.
        agree_ok, agree_detail = check_agreement(result.get("agreement"))
        print(f"[4 pose ]  hypothesis agreement: {'ok' if agree_ok else 'SCATTERED'} — "
              f"{agree_detail}")
    rs = [c.depth_residual_cm for c in checks if c.depth_residual_cm is not None]
    n_off = sum(1 for r in rs if not np.isfinite(r))
    if n_off:
        print(f"[4 pose ]  {n_off} frame(s) where the rendered mesh covers no measured pixel")
    if rs:
        print(f"[4 pose ]  depth consistency: median {np.median(rs):.2f} cm over "
              f"{len(rs)} frames (limit {args.max_depth_residual_cm:.0f} cm)"
              + (f", {n_off} of them off the object entirely" if n_off else ""))

    accepted = n_ok > len(checks) // 2 and agree_ok
    refined = None

    # The best frame's pose in world coordinates, computed whether or not it is admitted
    # — a refused pose still has to be inspectable, and the demo draws it next to where
    # our map puts the object so the disagreement is visible rather than only asserted.
    # Deliberately NOT called `pose_world`: that field stays null when refused, so
    # nothing downstream can read a rejected pose by forgetting to check `accepted`.
    best_any = min(checks, key=lambda c: (np.isnan(c.translation_cm), c.translation_cm))
    rec_any = next(r for r in frames if r["frame"] == best_any.frame)
    T_world_any = (np.asarray(rec_any["cam_to_world"], float)
                   @ poses[[r["frame"] for r in frames].index(best_any.frame)])
    if accepted or args.force:
        # Rebuild the target observation exactly as the other stages do, so the id and
        # label this pose attaches to are the same ones the scene graph already holds.
        labelled = json.load(open(E2E / "labelled.json"))["instances"]
        masks = np.load(E2E / "instance_masks.npz")["masks"]
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(json.load(open(E2E / "fuse.json"))["ply"]).points)
        keep = [i for i, r in enumerate(labelled) if r.get("label")]
        observations = from_openmask3d(
            [np.flatnonzero(masks[i]) for i in keep], points,
            [labelled[i]["label"] for i in keep], [labelled[i]["similarity"] for i in keep],
            anchor_frame=Frames.keyframe(0), stamp_ns=10 ** 9, registry=InstanceRegistry())
        obs = next((o for o in observations if o.node_id == meta["target"]), None)

        best = min((c for c in checks if c.accepted or args.force),
                   key=lambda c: c.translation_cm)
        rec = next(r for r in frames if r["frame"] == best.frame)
    if (accepted or args.force) and obs is None:
        # A target defined outside our segmentation (a SAM click) has no scene-graph node
        # to attach to. The accepted pose is recorded; nothing is refined in the graph.
        print(f"[4 pose ]  {meta['target']} is not a segmentation instance "
              f"({meta.get('target_source', 'unknown source')}); pose recorded, "
              f"not attached to the scene graph")
    elif accepted or args.force:
        refined = refine_with_pose(
            obs, poses[[r["frame"] for r in frames].index(best.frame)],
            np.asarray(rec["cam_to_world"], float),
            stamp_ns=int(rec["stamp"] * NS_PER_S),
            camera_stamp_ns=int(rec["cam_stamp"] * NS_PER_S))
        print(f"[4 pose ]  refined from frame {best.frame} -> 6-DoF pose in "
              f"{refined.anchor_frame}")
    else:
        worst = sorted(checks, key=lambda c: -c.translation_cm)[0]
        why = list(worst.reasons)
        if not agree_ok:
            why.append(f"hypotheses did not converge ({agree_detail})")
        print(f"[4 pose ]  REFUSED — {'; '.join(why)}")
        print("[4 pose ]  the position-only pose from stage 3 stands. It says nothing "
              "about rotation, which is better than saying something wrong.")

    json.dump({
        "ok": True,
        "accepted": bool(accepted),
        "forced": bool(args.force and not accepted),
        "target": meta["target"], "label": meta["label"],
        "source": result.get("source"),
        "frames_corroborating": n_ok, "frames_total": len(checks),
        "median_translation_cm": float(np.nanmedian([c.translation_cm for c in checks])),
        "median_rotation_deg": float(np.nanmedian([c.rotation_deg for c in checks])),
        "thresholds": {"translation_cm": args.max_translation_cm,
                       "rotation_deg": args.max_rotation_deg,
                       "depth_residual_cm": args.max_depth_residual_cm,
                       "depth_coverage": args.min_depth_coverage},
        "initialization": result.get("initialization", "register"),
        "median_depth_residual_cm": float(np.median(rs)) if rs else None,
        "frames_off_object": n_off if rs else None,
        "agreement_ok": bool(agree_ok), "agreement_detail": agree_detail,
        "agreement": result.get("agreement"),
        "per_frame": [vars(c) for c in checks],
        "pose_world": refined.pose.tolist() if refined is not None else None,
        "anchor_frame": refined.anchor_frame if refined is not None else None,
        # For inspection and for the demo only. Never a substitute for `pose_world`.
        "rejected_pose_world": None if accepted else T_world_any.tolist(),
        # An ACCEPTED pose for a target with no scene-graph node (SAM-defined) — kept,
        # but under its own name so nothing mistakes it for a graph-attached `pose_world`.
        "accepted_pose_world_unattached": (
            T_world_any.tolist() if accepted and refined is None else None),
        "target_source": meta.get("target_source"),
        "mask_source": meta.get("mask_source"),
        "best_frame": int(best_any.frame),
    }, open(args.out, "w"), indent=1)

    print(f"\n           -> {args.out.name}")
    print("[next  ]  stage 5: conda run -n openmask3d_vl python pipeline/tools/e2e_ground.py")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

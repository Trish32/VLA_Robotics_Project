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
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

E2E = ROOT / "pipeline/assets/e2e"
BUNDLE = E2E / "pose_bundle"
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

    def as_row(self) -> str:
        mark = "ok  " if self.accepted else "FAIL"
        return (f"  {mark} frame {self.frame}:  t {self.translation_cm:7.2f} cm   "
                f"R {self.rotation_deg:6.2f} deg   behind {self.behind_surface_cm:7.2f} cm")


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
) -> PoseCheck:
    """Corroborate one pose against the map and the depth image.

    Thresholds are grasp-relevant tolerances, not statistical ones: they answer "is this
    pose good enough to reach for" rather than "is this pose significantly different".
    """
    T = np.asarray(T_cam_obj, float).reshape(4, 4)
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

    return PoseCheck(frame=frame, translation_cm=translation_cm, rotation_deg=rotation_deg,
                     behind_surface_cm=behind_cm, accepted=not reasons, reasons=reasons)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=Path, default=BUNDLE / "pose_result.json",
                    help="FoundationPose output; written by the Kaggle kernel")
    ap.add_argument("--max-translation-cm", type=float, default=10.0)
    ap.add_argument("--max-rotation-deg", type=float, default=30.0)
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

    checks = [
        check_pose(T, frame=rec["frame"],
                   origin_cam=rec["mesh_origin_cam"],
                   R_world_to_cam=np.asarray(rec["R_world_to_cam"], float),
                   depth_median_m=dm, depth_span_m=span,
                   max_translation_cm=args.max_translation_cm,
                   max_rotation_deg=args.max_rotation_deg)
        for T, rec, (dm, span) in zip(poses, frames, depth_stats)
    ]
    for c in checks:
        print(c.as_row())

    n_ok = sum(c.accepted for c in checks)
    print(f"\n[4 pose ]  {n_ok}/{len(checks)} frames corroborate the map")
    print(f"[4 pose ]  median translation {np.median([c.translation_cm for c in checks]):.2f} cm"
          f"   median rotation {np.median([c.rotation_deg for c in checks]):.2f} deg")

    # A consistent tracker that is consistently wrong still fails. Self-consistency was
    # already measured upstream (world-frame spread); it is not evidence of accuracy.
    if "world_spread_cm" in result:
        print(f"[4 pose ]  tracker self-consistency {result['world_spread_cm']:.2f} cm "
              f"(says it is steady, not that it is right)")

    # The estimator's own self-agreement, which needs neither our map nor ground truth.
    agree_ok, agree_detail = check_agreement(result.get("agreement"))
    print(f"[4 pose ]  hypothesis agreement: {'ok' if agree_ok else 'SCATTERED'} — "
          f"{agree_detail}")

    accepted = n_ok > len(checks) // 2 and agree_ok
    refined = None

    # The best frame's pose in world coordinates, computed whether or not it is admitted
    # — a refused pose still has to be inspectable, and the demo draws it next to where
    # our map puts the object so the disagreement is visible rather than only asserted.
    # Deliberately NOT called `pose_world`: that field stays null when refused, so
    # nothing downstream can read a rejected pose by forgetting to check `accepted`.
    best_any = min(checks, key=lambda c: c.translation_cm)
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
        obs = next(o for o in observations if o.node_id == meta["target"])

        best = min((c for c in checks if c.accepted or args.force),
                   key=lambda c: c.translation_cm)
        rec = next(r for r in frames if r["frame"] == best.frame)
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
        "median_translation_cm": float(np.median([c.translation_cm for c in checks])),
        "median_rotation_deg": float(np.median([c.rotation_deg for c in checks])),
        "thresholds": {"translation_cm": args.max_translation_cm,
                       "rotation_deg": args.max_rotation_deg},
        "agreement_ok": bool(agree_ok), "agreement_detail": agree_detail,
        "agreement": result.get("agreement"),
        "per_frame": [vars(c) for c in checks],
        "pose_world": refined.pose.tolist() if refined is not None else None,
        "anchor_frame": refined.anchor_frame if refined is not None else None,
        # For inspection and for the demo only. Never a substitute for `pose_world`.
        "rejected_pose_world": None if accepted else T_world_any.tolist(),
        "best_frame": int(best_any.frame),
    }, open(E2E / "pose.json", "w"), indent=1)

    print("\n           -> pose.json")
    print("[next  ]  stage 5: conda run -n openmask3d_vl python pipeline/tools/e2e_ground.py")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

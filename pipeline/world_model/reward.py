#!/usr/bin/env python
"""Task success, labelled by the perception stack rather than by hand.

The missing ingredient for anything closed-loop. Reinforcement learning needs a reward;
filtering demonstrations for imitation learning needs a success label; and neither
exists in `cube_to_bowl_5`, which ships no reward column at all.

The stack can supply it. `extract_tracks.py` already says where the cube is in every
frame; this adds where the *goal* is, and tests containment.

**Why the bowl is detected once instead of tracked.** It never moves, so a track buys
nothing and costs a great deal: the arm is orange, CLIP calls it a bowl, and the bowl's
track was dropped from four of five episodes by the trust filter for exactly that
reason. One confident detection of a static object beats a fragile track of it.

**Why displacement alone is not the signal.** On episode 2 the cube's track drifts onto
the retreating arm after the place, so the cube "moves" 0.304 in image units and ends at
the bottom of the frame — a net displacement indistinguishable from the 0.318 of a real
success. Testing *where* the cube ended, against an independently detected bowl, tells
those apart. Testing only *how far* it went scores a corrupted track as a success.

    conda run -n openmask3d_vl python -m pipeline.world_model.reward \
        --data grootN1_Robotics/upstream/demo_data/cube_to_bowl_5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOAL_LABEL = "an orange plastic bowl"
SUBJECT_LABEL = "a small dark cube"


def detect_static_object(frames, label, gen, clip_model, preprocess, device, text_feats,
                         vocab, *, probes=None, min_area=600, max_area_frac=0.25):
    """Median detection of a non-moving object across a few early frames.

    Median rather than best-scoring: a single frame can have the arm occluding or
    overlapping the goal, and one bad detection would move the goal region for the whole
    episode. Agreement across probes is also the only evidence available that the thing
    found is actually static.
    """
    import cv2

    from pipeline.world_model.extract_tracks import label_masks

    H, W = frames[0].shape[:2]
    # Probe across the WHOLE episode, not just the opening frames. The goal is static,
    # so any frame where it is unoccluded serves — and the arm sits over it at the start
    # of some episodes, which is why probing only frames 0-2 found no bowl on two of
    # five.
    if probes is None:
        probes = np.linspace(0, len(frames) - 1, 6).astype(int).tolist()
    hits = []
    for t in probes:
        if t >= len(frames):
            continue
        rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
        cand = [m["segmentation"] for m in gen.generate(rgb)
                if min_area <= m["area"] <= max_area_frac * H * W]
        if not cand:
            continue
        labels = label_masks(rgb, cand, clip_model, preprocess, device, text_feats)
        best, best_sc = None, -1e9
        for m, (lab, sc) in zip(cand, labels):
            if lab == label and sc > best_sc:
                best, best_sc = m, sc
        if best is not None:
            ys, xs = np.nonzero(best)
            hits.append((xs.mean() / W, ys.mean() / H,
                         np.sqrt(len(xs) / (H * W)), float(best_sc)))
    if not hits:
        return None
    a = np.array(hits)
    spread = float(np.linalg.norm(a[:, :2].std(0))) if len(a) > 1 else 0.0
    return {"x": float(np.median(a[:, 0])), "y": float(np.median(a[:, 1])),
            "radius": float(np.median(a[:, 2])), "detections": len(a),
            "spread": spread, "score": float(np.median(a[:, 3]))}


def success_from_tracks(track_xy, visible, goal, *, hold: int = 20,
                        slack: float = 1.15) -> dict:
    """Did the subject end up inside the goal region, and was that actually observed?

    Requires the last `hold` frames to be MEASURED, not forward-filled. Occlusion carries
    the last known position forward, so a cube that vanished behind the gripper at the
    far side of the table would otherwise "hold" whatever position it had when it
    disappeared — and a reward that fires on an unobserved frame is a reward for being
    hidden.
    """
    idx = np.flatnonzero(visible)
    if len(idx) < hold:
        return {"success": False, "observed_tail": False, "fraction_inside": 0.0,
                "final_distance": float("nan"), "goal_radius": 0.0,
                "reason": f"only {len(idx)} observed frames, need {hold}"}
    # The last OBSERVED frames, not the last frames.
    #
    # Bracketed trust needs an anchor after a flow segment, so the frames following the
    # final anchor are never confirmed and every episode's tail reads as unobserved.
    # Demanding the literal last N frames be measured therefore fails every episode,
    # including the ones that plainly succeed. Taking the last N *measured* frames uses
    # the visibility data instead of fighting it — and still never scores a
    # forward-filled position, which is the property that matters.
    tail = idx[-hold:]
    seen = True
    d = np.linalg.norm(track_xy[tail] - np.array([goal["x"], goal["y"]]), axis=1)
    inside = d <= goal["radius"] * slack
    frac = float(inside.mean())
    return {
        "success": bool(seen and frac >= 0.8),
        "observed_tail": seen,
        "observed_frames": int(len(idx)),
        "total_frames": int(len(track_xy)),
        "fraction_inside": frac,
        "final_distance": float(d[-1]),
        "goal_radius": float(goal["radius"] * slack),
        "reason": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--hold", type=int, default=20)
    args = ap.parse_args()

    import clip
    import torch

    from common.device import pick_device
    from pipeline.world_model.extract_tracks import (VOCAB, patch_mps_float64,
                                                     read_frames)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    acc = pick_device()
    patch_mps_float64()
    sam = sam_model_registry["vit_h"](
        checkpoint=str(ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth")
    ).to(acc.device).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=12, pred_iou_thresh=0.88,
                                    stability_score_thresh=0.92,
                                    min_mask_region_area=600)
    gen.point_grids = [g.astype(np.float32) for g in gen.point_grids]
    clip_model, preprocess = clip.load("ViT-B/32", device=acc.device)
    with torch.no_grad():
        tf = clip_model.encode_text(clip.tokenize(VOCAB).to(acc.device)).float()
        tf /= tf.norm(dim=-1, keepdim=True)

    data = Path(args.data)
    out = []
    for vid in sorted((data / "videos").rglob(f"*images.{args.camera}/*.mp4")):
        tp = data / "tracks" / f"{vid.stem}_tracks.npz"
        if not tp.exists():
            print(f"[reward] {vid.stem}: no tracks — run extract_tracks.py first")
            continue
        z = np.load(tp, allow_pickle=True)
        labels = [str(x) for x in z["labels"]]
        if SUBJECT_LABEL not in labels:
            print(f"[reward] {vid.stem}: no {SUBJECT_LABEL!r} track")
            continue
        i = labels.index(SUBJECT_LABEL)

        frames = read_frames(vid)
        goal = detect_static_object(frames, GOAL_LABEL, gen, clip_model, preprocess,
                                    acc.device, tf, VOCAB)
        if goal is None:
            print(f"[reward] {vid.stem}: goal {GOAL_LABEL!r} not detected")
            continue
        r = success_from_tracks(z["tracks"][:, i, :2], z["visible"][:, i], goal,
                                hold=args.hold)
        r.update(episode=vid.stem, goal=goal)
        out.append(r)
        mark = "SUCCESS" if r["success"] else "no     "
        print(f"[reward] {vid.stem}  {mark}  goal ({goal['x']:.3f},{goal['y']:.3f}) "
              f"r={goal['radius']:.3f} from {goal['detections']} probes "
              f"(spread {goal['spread']:.3f})")
        print(f"[reward]     inside {r['fraction_inside'] * 100:5.1f}% of the last "
              f"{args.hold} frames, final distance {r['final_distance']:.3f}"
              f"{'' if r['observed_tail'] else '  [TAIL NOT OBSERVED]'}")

    dst = data / "tracks" / "success.json"
    json.dump(out, open(dst, "w"), indent=1)
    n = sum(r["success"] for r in out)
    print(f"\n[reward] {n}/{len(out)} episodes succeed -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Per-instance object tracks from RGB video, using this repo's own SAM + CLIP.

`cube_to_bowl_5` ships video and actions but no per-instance masks — only one episode of
a single binary foreground mask, on which the object pathway measured *worse* than
assuming nothing moves. This produces the missing half: which objects are present, and
where each one is in every frame.

**Every frame's position is measured, never interpolated.** That constraint drives the
whole design. Interpolating between sparse keyframes would make the tracks piecewise
linear *by construction*, a constant-velocity predictor would then be near-perfect on
them, and any dynamics model trained against that would be learning an artefact of the
extraction. So:

  * **SAM at keyframes** (every `--keyframe-every` frames) proposes instances and
    re-anchors the tracks, correcting drift;
  * **Lucas-Kanade optical flow every frame in between** moves each instance's points
    using that frame's own pixels.

SAM ViT-H costs ~4 s per call on MPS, which is why it anchors rather than runs
per-frame; flow costs milliseconds.

CLIP labels each instance open-vocabulary, so the track for "the cube" is identified by
what it looks like rather than by a hardcoded index that would silently point at the
bowl on a different episode.

    conda run -n openmask3d_vl python -m pipeline.world_model.extract_tracks \
        --data grootN1_Robotics/upstream/demo_data/cube_to_bowl_5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# What we expect to see. Open-vocabulary: change this list and the labels change, with
# no other edit. The prompts are phrases because CLIP was trained on captions.
VOCAB = ["an orange robotic arm", "a robot gripper claw", "an orange plastic bowl",
         "a small dark cube", "a green rubber ball", "a plain table surface", "a cable"]
# Instances we want tracks for. The arm and table are segmented but are not objects the
# dynamics model should predict: the arm IS the action, and the table never moves.
KEEP = {"an orange plastic bowl", "a small dark cube", "a green rubber ball"}


def patch_mps_float64() -> None:
    """SAM's coordinate transforms upcast to float64, which MPS cannot hold.

    Patched at the source rather than at each call site — `apply_coords` is reached from
    several paths inside the automatic mask generator, and casting the point grids alone
    is not enough (that was the first attempt and it still failed inside `_process_batch`).
    """
    from segment_anything.utils.transforms import ResizeLongestSide
    for name in ("apply_coords", "apply_boxes"):
        orig = getattr(ResizeLongestSide, name)
        if getattr(orig, "_mps_patched", False):
            continue

        def wrap(self, *a, _o=orig, **k):
            out = _o(self, *a, **k)
            return out.astype(np.float32) if isinstance(out, np.ndarray) else out

        wrap._mps_patched = True
        setattr(ResizeLongestSide, name, wrap)


def read_frames(path: Path) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def mask_text_similarity(img_rgb, masks, clip_model, preprocess, device, text_feats):
    """(n_masks, n_vocab) CLIP cosine similarities, from background-suppressed crops.

    Suppressed rather than left in: a crop of a small cube on a large table is mostly
    table, and CLIP would confidently describe the table.

    Returned as the whole matrix, not as a per-mask argmax, because the two directions
    disagree and the argmax is the worse one. Measured in the simulator, where the
    right answer is known: SAM segments the bowl at IoU 0.977 and CLIP still argmaxes
    that exact mask to "a small dark cube" (0.295 against the bowl prompt's 0.289).
    Reading the matrix down its columns — for each label, which mask suits it best —
    recovers the bowl; reading it across rows loses it.
    """
    import torch
    from PIL import Image
    crops = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 8
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(img_rgb.shape[0], y1 + pad), min(img_rgb.shape[1], x1 + pad)
        crop = img_rgb[y0:y1, x0:x1].copy()
        sub = m[y0:y1, x0:x1]
        crop[~sub] = (0.25 * crop[~sub]).astype(np.uint8)
        crops.append(preprocess(Image.fromarray(crop)))
    if not crops:
        return np.zeros((0, text_feats.shape[0]), np.float32)
    with torch.no_grad():
        feats = clip_model.encode_image(torch.stack(crops).to(device)).float()
        feats /= feats.norm(dim=-1, keepdim=True)
        sim = feats @ text_feats.T
    return sim.cpu().numpy()


def label_probabilities(img_rgb, masks, clip_model, preprocess, device, text_feats,
                        logit_scale: float = 100.0):
    """(n_masks, n_vocab) probabilities — the COLUMN-wise read, and the correct one.

    Softmax across the vocabulary at CLIP's own logit scale. The raw cosine
    similarities sit in a narrow 0.21-0.32 band whatever the crop is, so a threshold on
    them is a threshold on nothing; the softmax turns "which prompt fits best, and by
    how much" into a number a threshold can use.

    Read this down its columns — for each label, which mask suits it — rather than
    across its rows. Measured in the simulator where the answer is known: SAM segments
    the bowl at IoU 0.977 and the row-wise argmax of that same mask is "a small dark
    cube". A label is a claim about a mask; a mask is not a claim about a label.
    """
    sim = mask_text_similarity(img_rgb, masks, clip_model, preprocess, device,
                               text_feats)
    if not len(sim):
        return sim
    z = np.exp(logit_scale * (sim - sim.max(-1, keepdims=True)))
    return z / z.sum(-1, keepdims=True)


def label_masks(img_rgb, masks, clip_model, preprocess, device, text_feats,
                vocab=None):
    """Best label per mask. The row-wise read, kept for callers that want it.

    Prefer `label_probabilities` — this loses a correctly-segmented object whenever its
    own best prompt happens to be another object's.
    """
    sim = mask_text_similarity(img_rgb, masks, clip_model, preprocess, device,
                               text_feats)
    if not len(sim):
        return []
    # `vocab` must be the list the text features were encoded from; defaulting to the
    # module list keeps every existing caller unchanged, and the sim passes its own
    # because its arm is grey rather than orange.
    words = VOCAB if vocab is None else vocab
    idx = sim.argmax(-1)
    return [(words[i], float(sim[r, i])) for r, i in enumerate(idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--camera", default="front",
                    help="front is the scene view; the wrist camera moves with the arm, "
                         "so apparent object motion there is mostly ego-motion")
    ap.add_argument("--keyframe-every", type=int, default=12,
                    help="SAM re-anchor interval. Lower corrects drift sooner at linear "
                         "cost; 30 let the cube track ride the arm away after a grasp")
    ap.add_argument("--bracket-slack", type=float, default=2.0,
                    help="a flow segment is trusted only if its two bracketing anchors "
                         "are within this multiple of the keyframe interval")
    ap.add_argument("--points-per-side", type=int, default=12)
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--max-area-frac", type=float, default=0.25,
                    help="drop masks larger than this fraction of the image: the table "
                         "and background are segmented too and are not objects")
    ap.add_argument("--label-read", default="row", choices=("row", "column"),
                    help="how to read the CLIP similarity matrix. MEASURED, and the "
                         "answer differs by dataset, which is why this is a flag and "
                         "not a fix: 'column' (for each label, the best mask) is "
                         "clearly right in simulation, where it recovers a bowl "
                         "segmented at IoU 0.977 that the row-wise argmax assigns to "
                         "'a small dark cube'. On the real video it REGRESSES — the "
                         "arm is orange, wins the bowl prompt, and the static bowl's "
                         "track travels 0.68-0.73 normalised units. The difference is "
                         "whether the competing label is something we want: in sim it "
                         "is another tracked object, on real video it is the arm.")
    ap.add_argument("--min-prob", type=float, default=0.25,
                    help="minimum P(label | mask) under the column-wise read; a "
                         "threshold on raw cosine similarity would be meaningless, "
                         "since those sit in a 0.21-0.32 band whatever the crop is")
    ap.add_argument("--gate", type=float, default=0.18,
                    help="max normalised image distance a keyframe re-anchor may move "
                         "an existing track; beyond it the detection is a different "
                         "object that happens to share a label")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="render a track overlay for the first N episodes; the only way "
                         "to tell a healthy-looking track from one on the wrong object")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    import clip
    import torch

    from common.device import pick_device
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    acc = pick_device()
    patch_mps_float64()
    print(f"[tracks] device {acc.device}")

    sam = sam_model_registry["vit_h"](
        checkpoint=str(ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth")
    ).to(acc.device).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side,
                                    pred_iou_thresh=0.88,
                                    stability_score_thresh=0.92,
                                    min_mask_region_area=args.min_area)
    gen.point_grids = [g.astype(np.float32) for g in gen.point_grids]

    clip_model, preprocess = clip.load("ViT-B/32", device=acc.device)
    with torch.no_grad():
        tf = clip_model.encode_text(clip.tokenize(VOCAB).to(acc.device)).float()
        tf /= tf.norm(dim=-1, keepdim=True)

    data = Path(args.data)
    out_dir = Path(args.out) if args.out else data / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted((data / "videos").rglob(f"*images.{args.camera}/*.mp4"))
    if not videos:
        raise SystemExit(f"no {args.camera} videos under {data}/videos")

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    summary = []

    for vid in videos:
        t0 = time.time()
        frames = read_frames(vid)
        H, W = frames[0].shape[:2]
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        T = len(frames)
        print(f"\n[tracks] {vid.stem} · {T} frames")

        # tracks[label] = (T, 3) observables, and a visibility flag per frame
        tracks: dict[str, np.ndarray] = {}
        vis: dict[str, np.ndarray] = {}
        pts: dict[str, np.ndarray] = {}          # live LK points per instance
        anchors: dict[str, list[int]] = {}       # keyframes where SAM confirmed a track

        for t in range(T):
            if t % args.keyframe_every == 0:
                rgb = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
                raw = gen.generate(rgb)
                cand = [m["segmentation"] for m in raw
                        if args.min_area <= m["area"] <= args.max_area_frac * H * W]
                prob = label_probabilities(rgb, cand, clip_model, preprocess,
                                           acc.device, tf)
                if args.label_read == "row":
                    # Keep only each mask's OWN best prompt. Measured to be the right
                    # default here and the wrong one in simulation — see the note on
                    # --label-read.
                    keep_row = prob.argmax(-1)
                    mask_ok = np.array([VOCAB[i] in KEEP for i in keep_row])
                else:
                    mask_ok = np.ones(len(cand), bool)
                # Associate by LABEL AND POSITION, not by label argmax alone.
                #
                # The arm is orange and CLIP happily calls it "an orange plastic bowl",
                # so a pure argmax let the arm steal the bowl's identity every time it
                # swung past: the bowl's track travelled 0.811 in normalised image units
                # when the bowl never moves, alternating between the real bowl and the
                # arm. Requiring a re-anchor to be near where the track already is fixes
                # it for the same reason `InstanceRegistry` associates by IoU in 3-D —
                # objects do not teleport between observations.
                #
                # Scored down the COLUMNS: every mask is considered for every label we
                # want, rather than only the masks whose own best prompt happens to be
                # one of them. The row-wise version discarded a correctly-segmented
                # object outright whenever CLIP's argmax for it landed elsewhere, which
                # is a near-tie away at these similarity margins.
                # Which masks are something we are NOT looking for. Reading down the
                # columns alone regressed the real data badly: the arm is orange, so
                # asking "which mask best suits 'an orange plastic bowl'" lets an arm
                # mask win, and the bowl's track travelled 0.68-0.73 in normalised
                # units for an object that never moves.
                #
                # The sim failure and the real failure have different shapes, and this
                # separates them. In the sim the competing label was "a small dark
                # cube" — also an object we want — so the column read is right there.
                # On the real video the competitor is "an orange robotic arm", which we
                # do not want at all, and a mask that CLIP thinks is mostly arm should
                # never be handed to the bowl however well it scores on the bowl prompt.
                other = [i for i, w in enumerate(VOCAB) if w not in KEEP]
                dominated = prob[:, other].max(-1) if other else np.zeros(len(cand))

                best: dict[str, tuple[float, np.ndarray]] = {}
                for lab in KEEP:
                    col = VOCAB.index(lab)
                    for mi, m in enumerate(cand):
                        if not mask_ok[mi]:
                            continue
                        if args.label_read == "row" and VOCAB[
                                int(prob[mi].argmax())] != lab:
                            continue
                        sc = float(prob[mi, col])
                        if sc < args.min_prob or sc < dominated[mi]:
                            continue
                        ys, xs = np.nonzero(m)
                        cx, cy = xs.mean() / W, ys.mean() / H
                        if lab in tracks and pts.get(lab) is not None:
                            p_now = pts[lab]
                            d = float(np.hypot(p_now[:, 0].mean() / W - cx,
                                               p_now[:, 1].mean() / H - cy))
                            if d > args.gate:
                                continue                 # too far to be the same object
                            sc = sc - d                  # prefer the nearer candidate
                        if sc > best.get(lab, (-1e9, None))[0]:
                            best[lab] = (sc, m)
                # One mask cannot serve two labels. Resolved by score so the more
                # confident claim wins, rather than by dictionary order.
                claimed: dict[int, str] = {}
                for lab, (sc, m) in sorted(best.items(), key=lambda kv: -kv[1][0]):
                    key = int(m.sum()) * 1000003 + int(np.flatnonzero(m.ravel())[0])
                    if key in claimed:
                        del best[lab]
                    else:
                        claimed[key] = lab
                for lab, (_, m) in best.items():
                    ys, xs = np.nonzero(m)
                    if lab not in tracks:
                        tracks[lab] = np.full((T, 3), np.nan, np.float32)
                        vis[lab] = np.zeros(T, bool)
                    # Re-anchor: SAM's mask replaces whatever flow had drifted to.
                    sel = np.random.default_rng(0).choice(
                        len(xs), min(120, len(xs)), replace=False)
                    pts[lab] = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
                    tracks[lab][t] = (xs.mean() / W, ys.mean() / H,
                                      np.sqrt(len(xs) / (H * W)))
                    vis[lab][t] = True
                    anchors.setdefault(lab, []).append(t)
            if t % args.keyframe_every == 0:
                # Tracks with no accepted re-anchor this keyframe keep flowing, so a
                # rejected association is a missed correction rather than a lost object.
                for lab in [k for k in tracks if k not in best]:
                    if pts.get(lab) is not None and t > 0:
                        p = pts[lab]
                        tracks[lab][t] = (p[:, 0].mean() / W, p[:, 1].mean() / H,
                                          tracks[lab][t - 1, 2])
                        vis[lab][t] = True
            else:
                for lab, p in list(pts.items()):
                    if p is None or len(p) < 6:
                        continue
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                        gray[t - 1], gray[t], p.reshape(-1, 1, 2), None, **lk)
                    st = st.reshape(-1).astype(bool)
                    if st.sum() < 6:
                        pts[lab] = None          # lost; wait for the next keyframe
                        continue
                    p = nxt.reshape(-1, 2)[st]
                    pts[lab] = p
                    # Area is carried from the last anchor: optical flow tracks points,
                    # not extent, and inventing a scale from point spread would add a
                    # signal the pixels did not provide.
                    last = np.nanmax(np.where(vis[lab][:t + 1])[0], initial=-1)
                    area = tracks[lab][last, 2] if last >= 0 else np.nan
                    tracks[lab][t] = (p[:, 0].mean() / W, p[:, 1].mean() / H, area)
                    vis[lab][t] = True

        # Trust only flow that is BRACKETED by two SAM confirmations.
        #
        # Optical flow follows whatever texture it latched onto, and when the gripper
        # occludes the cube it follows the ARM away — episode 2 ended with both the cube
        # and bowl markers riding the retreating arm while the cube sat in the bowl. A
        # flow segment that no later detection confirms is a guess, and a guess recorded
        # as a measurement is how a dynamics model learns motion that never happened.
        for lab, a in anchors.items():
            trusted = np.zeros(T, bool)
            for lo, hi in zip(a, a[1:]):
                if hi - lo <= args.keyframe_every * args.bracket_slack:
                    trusted[lo:hi + 1] = True
            trusted[a] = True                       # anchors themselves are measured
            vis[lab] &= trusted

        keep = {k: v for k, v in tracks.items() if vis[k].mean() > 0.5}
        for lab in keep:
            # Gaps are forward-filled: an occluded object did not move to NaN.
            a = keep[lab]
            for i in range(1, T):
                if np.isnan(a[i]).any():
                    a[i] = a[i - 1]
            if np.isnan(a[0]).any():
                first = np.where(~np.isnan(a).any(1))[0]
                a[:first[0]] = a[first[0]] if len(first) else 0.0
        order = sorted(keep)
        arr = np.stack([keep[k] for k in order], 1) if order else np.zeros((T, 0, 3),
                                                                          np.float32)
        np.savez_compressed(out_dir / f"{vid.stem}_tracks.npz",
                            tracks=arr.astype(np.float32),
                            labels=np.array(order),
                            visible=np.stack([vis[k] for k in order], 1) if order
                            else np.zeros((T, 0), bool))
        moved = {k: float(np.linalg.norm(keep[k][:, :2].max(0) - keep[k][:, :2].min(0)))
                 for k in order}
        print(f"[tracks]   {len(order)} instances, {time.time() - t0:.0f}s")
        for k in order:
            print(f"[tracks]     {k:24} visible {vis[k].mean() * 100:5.1f}%  "
                  f"travel {moved[k]:.3f} (normalised image units)")
        summary.append({"episode": vid.stem, "frames": T, "labels": order,
                        "visible": {k: float(vis[k].mean()) for k in order},
                        "travel": moved})

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)

    # A track file looks fine whatever it contains; the only way to know an instance is
    # on the right object is to draw it on the frames. This caught the arm stealing the
    # bowl's identity, which every summary statistic had reported as a healthy 99.5%
    # visible track.
    if args.check:
        for vid in videos[:args.check]:
            z = np.load(out_dir / f"{vid.stem}_tracks.npz", allow_pickle=True)
            tr, lab = z["tracks"], [str(x) for x in z["labels"]]
            vz = z["visible"]
            frames = read_frames(vid)
            H, W = frames[0].shape[:2]
            cols = {"a small dark cube": (0, 255, 0),
                    "an orange plastic bowl": (0, 165, 255),
                    "a green rubber ball": (255, 0, 0)}
            picks = np.linspace(0, len(frames) - 1, 5).astype(int)
            tiles = []
            for t in picks:
                f = frames[t].copy()
                for i, l in enumerate(lab):
                    x, y = tr[t, i, 0] * W, tr[t, i, 1] * H
                    c = cols.get(l, (255, 255, 255))
                    seen = bool(vz[t, i])
                    # Hollow + "?" where the position is forward-filled rather than
                    # measured. Drawing both the same way made a correctly-flagged
                    # occlusion look like a tracking failure.
                    cv2.circle(f, (int(x), int(y)), 11, c, 3 if seen else 1)
                    cv2.putText(f, l.split()[-1] + ("" if seen else "?"),
                                (int(x) + 13, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2 if seen else 1)
                cv2.putText(f, f"t={t}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
                tiles.append(cv2.resize(f, (320, 240)))
            png = out_dir / f"{vid.stem}_check.png"
            cv2.imwrite(str(png), np.hstack(tiles))
            print(f"[tracks] overlay -> {png}")

    print(f"\n[tracks] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

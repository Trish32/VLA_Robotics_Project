#!/usr/bin/env python
"""Stage 3.5: give Mask3D's class-agnostic instances real open-vocabulary labels.

Mask3D proposes WHERE instances are and never WHAT they are — `pred_logits` is trained
on ScanNet200's closed set and OpenMask3D discards it. The label comes from projecting
each 3-D instance into the views that see it best, refining the 2-D region with SAM,
cropping, and embedding with CLIP. Querying that embedding against arbitrary text is the
"open-vocabulary" part; nothing here is restricted to a fixed class list.

This replaces the geometry-derived placeholders the first end-to-end run used (labelling
by bounding-box diagonal), which were honest stand-ins but were not recognition.

Runs the accelerated path from `openmask3d_semantic/fast_features.py` — view-major, so
SAM's image encoder fires once per distinct view rather than once per (instance, view)
pair. On CPU that is the difference between minutes and tens of minutes.

    conda run -n openmask3d_vl python pipeline/tools/e2e_label.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "pipeline/assets/e2e"
DATA = ROOT / "orbslam3_baseline/data"

# Sequence, trajectory and INTRINSICS all follow fuse.json rather than being hardcoded.
# freiburg1 and freiburg3 were recorded on different Kinects, so projecting fr3 with
# fr1's intrinsics puts every instance's crop in the wrong place and SAM is then prompted
# on the wrong pixels -- with no error anywhere, just worse labels.
SEQ = DATA / "rgbd_dataset_freiburg1_xyz"
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989

# An open vocabulary: nothing in the pipeline is tied to this list, it is just what we
# ask about. A desk scene, so these are plausible; any other string would work.
VOCAB = [
    "a monitor", "a computer screen", "a desk", "a table", "a chair", "a keyboard",
    "a plant", "a book", "a cup", "a lamp", "a wall", "a floor", "a person",
    "a teddy bear", "a box", "a poster",
]


# CLIP was trained on caption-like text, and its zero-shot accuracy is markedly better
# through the prompt templates it was benchmarked with than on bare nouns. The ensemble
# is averaged in TEXT space before normalising, which is how the CLIP authors report it.
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {} in a room.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
)


def cosine_similarity(features, clip_model, vocabulary):
    """(n_instances, n_words) true cosine similarities, both sides unit-normalised."""
    import clip
    import torch

    image = torch.from_numpy(np.asarray(features)).float()
    image = image / image.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    with torch.no_grad():
        columns = []
        for word in vocabulary:
            bare = word.removeprefix("a ").removeprefix("the ")
            tokens = clip.tokenize([t.format(bare) for t in PROMPT_TEMPLATES])
            embeddings = clip_model.encode_text(tokens).float()
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            pooled = embeddings.mean(dim=0)
            columns.append(pooled / pooled.norm().clamp_min(1e-8))
        text = torch.stack(columns)
    return image @ text.T


def project_visibility(points, masks, poses, depths, image_hw, vis_tol=0.08):
    """(n_views, n_masks, H, W) bool — which instance each pixel sees, per view.

    A point counts as visible only if its projected depth agrees with the measured depth
    at that pixel. Without that test every instance is "visible" through every wall, and
    SAM gets prompted on the far side of the room.
    """
    height, width = image_hw
    n_views, n_masks = len(poses), len(masks)
    out = np.zeros((n_views, n_masks, height, width), dtype=bool)

    for v, (T, depth) in enumerate(zip(poses, depths)):
        world_to_cam = np.linalg.inv(T)
        cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        z = cam[:, 2]
        front = z > 1e-3
        u = np.full(len(points), -1.0)
        vv = np.full(len(points), -1.0)
        u[front] = FX * cam[front, 0] / z[front] + CX
        vv[front] = FY * cam[front, 1] / z[front] + CY
        # Round FIRST, then bounds-check: np.round(479.6) is 480, which is out of
        # range for a 480-row image even though the float passed a `< height` test.
        ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
        inside = front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        if not inside.any():
            continue
        measured = np.zeros(len(points))
        measured[inside] = depth[vi[inside], ui[inside]]
        visible = inside & (measured > 0) & (np.abs(measured - z) < vis_tol)

        for m, mask in enumerate(masks):
            sel = visible & mask
            if sel.any():
                out[v, m, vi[sel], ui[sel]] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=int, default=12, help="views sampled from the track")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--sam", default="vit_h", help="vit_h, or vit_t for MobileSAM")
    args = ap.parse_args()

    import clip
    import cv2
    import open3d as o3d
    import torch
    from PIL import Image

    from openmask3d_semantic.fast_features import build_sam_predictor, extract_mask_features

    sys.path.insert(0, str(ROOT / "openmask3d_semantic/upstream/openmask3d"))
    from mask_features_computation.utils import mask2box_multi_level

    from pipeline.transforms import quaternion_to_matrix

    seg = json.load(open(OUT / "segment.json"))
    fuse = json.load(open(OUT / "fuse.json"))

    global SEQ, FX, FY, CX, CY
    if fuse.get("sequence"):
        SEQ = DATA / fuse["sequence"]
    from pipeline.tools.e2e_tum import INTRINSICS
    FX, FY, CX, CY = INTRINSICS[fuse.get("camera", "freiburg1")]
    traj_file = DATA / fuse.get("trajectory", "trajectory_orbslam3.txt")
    print(f"[label ]  {SEQ.name} · {traj_file.name} · fx={FX}")
    points = np.asarray(o3d.io.read_point_cloud(fuse["ply"]).points)

    # Masks come from the .npz stage 3 wrote, NOT re-derived from centre/extent.
    # Re-deriving was a real bug: those boxes are computed on the Z-UP ROTATED points
    # inside mask3d_instances, so testing them against the original cloud matched
    # nothing and every instance silently produced zero crops.
    membership = np.load(OUT / "instance_masks.npz")["masks"]
    masks = [membership[i] for i in range(membership.shape[0])]
    print(f"[label ]  {len(masks)} instances, {len(points)} points")

    # --- posed frames -------------------------------------------------------------
    # The cloud is LEVELLED but the trajectory file is in the raw SLAM frame, so the
    # poses take the same rotation before they are used to project it. Without this the
    # projection is wrong by the levelling angle, nothing is visible in any view, and
    # every instance silently yields zero crops and therefore no label.
    T_level = (np.asarray(fuse["T_level_slam"], float)
               if fuse.get("levelled") else np.eye(4))

    traj = {}
    for line in traj_file.read_text().splitlines():
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        traj[v[0]] = T_level @ T
    stamps = np.array(sorted(traj))

    rgb = [(float(l.split()[0]), l.split()[1])
           for l in (SEQ / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(float(l.split()[0]), l.split()[1])
           for l in (SEQ / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    dstamps = np.array([s for s, _ in dep])

    chosen = rgb[:: max(1, len(rgb) // args.views)][: args.views]
    poses, depths, images_np, images_pil = [], [], [], []
    for stamp, rel in chosen:
        j = int(np.argmin(np.abs(stamps - stamp)))
        k = int(np.argmin(np.abs(dstamps - stamp)))
        bgr = cv2.imread(str(SEQ / rel))
        poses.append(traj[stamps[j]])
        depths.append(cv2.imread(str(SEQ / dep[k][1]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0)
        images_np.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        images_pil.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    print(f"[label ]  {len(poses)} views")

    visible = project_visibility(points, masks, poses, depths, images_np[0].shape[:2])
    counts = visible.reshape(len(poses), len(masks), -1).sum(-1)
    topk = np.argsort(-counts, axis=0)[: args.topk].T
    print(f"[label ]  visibility computed; top-{args.topk} views per instance selected")

    predictor = build_sam_predictor(
        "cpu", args.sam,
        str(ROOT / "openmask3d_semantic/checkpoints/sam_vit_h_4b8939.pth"))
    clip_model, preprocess = clip.load("ViT-L/14@336px", device="cpu")
    print(f"[label ]  SAM({args.sam}) + CLIP ViT-L/14 on CPU — this is the slow part")

    features = extract_mask_features(
        topk_indices_per_mask=topk, visible_points_in_view_in_mask=visible,
        images_np=images_np, images_pil=images_pil, predictor_sam=predictor,
        clip_model=clip_model, clip_preprocess=preprocess,
        crop_box_fn=mask2box_multi_level, device="cpu",
        num_levels=2, num_random_rounds=5, num_selected_points=5, feature_dim=768,
    )

    # OpenMask3D's actual product is this per-instance feature, not the label: compute
    # once, query any text later. Saving it turns a 10-minute SAM+CLIP pass into a
    # millisecond dot product, which is what makes the vocabulary genuinely open.
    np.savez_compressed(OUT / "instance_features.npz", features=features,
                        ids=np.array([r["id"] for r in seg["instances"]]))

    # Normalise the IMAGE feature at query time. `extract_mask_features` normalises each
    # crop and then AVERAGES them, and the mean of unit vectors is not a unit vector --
    # its norm falls as the crops disagree across views. Without this the dot product is
    # |f|*cos(theta), not cos(theta): the argmax per instance is unchanged (the scale is
    # constant within a row) but every reported "similarity" is scaled by a per-instance
    # factor, so the numbers are not comparable between instances and are not cosine
    # similarities at all. The stored feature is left faithful to upstream.
    sims = cosine_similarity(features, clip_model, VOCAB)

    print(f"\n{'instance':<10}{'points':>8}{'label':>22}{'sim':>8}   runner-up")
    print("-" * 68)
    labelled = []
    for i, r in enumerate(seg["instances"]):
        if np.abs(features[i]).sum() == 0:
            print(f"{r['id']:<10}{r['n']:>8}{'(no crops)':>22}")
            labelled.append({**r, "label": None})
            continue
        order = torch.argsort(-sims[i])
        best, second = int(order[0]), int(order[1])
        print(f"{r['id']:<10}{r['n']:>8}{VOCAB[best]:>22}{sims[i, best]:>8.3f}   "
              f"{VOCAB[second]} ({sims[i, second]:.3f})")
        labelled.append({**r, "label": VOCAB[best].removeprefix("a "),
                         "similarity": float(sims[i, best])})

    json.dump({"ok": True, "instances": labelled}, open(OUT / "labelled.json", "w"))
    print(f"\n[label ]  -> {OUT / 'labelled.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

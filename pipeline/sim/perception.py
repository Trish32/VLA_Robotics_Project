"""Rendered frame -> world-frame object observations, two ways.

The two backends exist to be *differenced*. `OraclePerception` uses the simulator's own
instance masks; `SamPerception` runs the same SAM + CLIP stack the real
`cube_to_bowl_5` extraction runs. Both end at the same type, so a policy, a reward or a
world model can be driven by either and the gap between the two runs is a measurement
of the perception stack rather than an opinion about it. On real demonstrations that
measurement is unavailable at any price, because there is no ground truth to difference
against.

Both backends deliberately go **mask, then depth, then unproject** — never
`data.xpos`. Reading the simulator's state would make the oracle a perfect observer
rather than a perfect *segmenter*, and every projection and depth-quantisation effect
that the SAM path also suffers would then be charged to SAM.

One honest limitation applies to both. A single camera sees one side of an object, so
the centroid of the visible surface sits toward the camera and the extent along the
viewing axis is a lower bound. For the cube that bias is about 1 cm — measured, not
assumed, in `test_sim_perception.py`. It is a property of monocular RGB-D, not of the
segmenter, and it does not cancel when comparing the two backends because both have it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

#: CLIP prompts for the simulated scene. The three task-object phrases are character
#: for character the ones the real extraction uses, so a sim track and a real track
#: mean the same thing; only the arm's description differs, because this arm is grey
#: and the real one is orange. Getting that wrong is not cosmetic — an orange prompt
#: over a grey arm sends the arm's masks to whichever orange thing is in the vocabulary,
#: which is the bowl.
SIM_VOCAB = ["a grey robotic arm", "a metal gripper claw", "an orange plastic bowl",
             "a small dark cube", "a green rubber ball", "a plain table surface",
             "a checkered floor"]
KEEP = ("an orange plastic bowl", "a small dark cube", "a green rubber ball")


@dataclass
class Perceived:
    """One object as the stack actually saw it."""

    label: str
    centre: np.ndarray      # (3,) world frame, metres
    extent: np.ndarray      # (3,) world-axis-aligned size, metres
    pixels: int
    score: float            # CLIP similarity; 1.0 for the oracle
    #: The boolean image mask this reading came from, when the backend kept it.
    #: Optional and unused by the control path — it exists so a figure can draw the
    #: mask the stack ACTUALLY selected rather than re-running the selection with
    #: slightly different code and showing something the robot never saw.
    mask: np.ndarray | None = None

    def as_slot(self) -> np.ndarray:
        """(6,) centre + extent — the geometry channels of a `SceneLatent` slot."""
        return np.concatenate([self.centre, self.extent]).astype(np.float32)


def mask_to_points(mask: np.ndarray, depth: np.ndarray, K: dict,
                   T_world_cam: np.ndarray, *, max_range: float = 5.0) -> np.ndarray:
    """Masked pixels to an (N, 3) world-frame point cloud."""
    m = mask & np.isfinite(depth) & (depth > 1e-4) & (depth < max_range)
    vs, us = np.nonzero(m)
    if not len(vs):
        return np.zeros((0, 3))
    z = depth[vs, us]
    pts = np.stack([(us + 0.5 - K["cx"]) * z / K["fx"],
                    (vs + 0.5 - K["cy"]) * z / K["fy"], z], -1)
    return pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def summarise(points: np.ndarray, *, trim: float = 2.0) -> tuple[np.ndarray,
                                                                 np.ndarray]:
    """Robust centre and extent of a point cloud.

    Percentiles rather than min/max: one stray pixel at a depth discontinuity — the
    silhouette edge, where a ray grazes the object and lands on the table behind it —
    moves a min/max box by tens of centimetres, and the extent feeds the collision term
    of the scorer.
    """
    if len(points) < 4:
        c = points.mean(0) if len(points) else np.zeros(3)
        return c, np.zeros(3)
    lo = np.percentile(points, trim, axis=0)
    hi = np.percentile(points, 100 - trim, axis=0)
    return (lo + hi) / 2, np.maximum(hi - lo, 1e-4)


class OraclePerception:
    """Simulator instance masks, real depth, real unprojection."""

    name = "oracle"

    def __init__(self, *, min_pixels: int = 40, keep: tuple[str, ...] = KEEP) -> None:
        self.min_pixels = min_pixels
        self.keep = set(keep)

    def __call__(self, frame: dict) -> dict[str, Perceived]:
        out = {}
        for i, name in frame["label_names"].items():
            if name not in self.keep:
                continue
            m = frame["labels"] == i
            n = int(m.sum())
            if n < self.min_pixels:
                continue
            pts = mask_to_points(m, frame["depth"], frame["K"], frame["T_world_cam"])
            if len(pts) < self.min_pixels:
                continue
            c, e = summarise(pts)
            out[name] = Perceived(name, c, e, n, 1.0, mask=m)
        return out


class SamPerception:
    """SAM automatic masks, CLIP labels, depth unprojection.

    The same code path as the real extraction, minus the cross-frame association: in a
    closed loop each frame is perceived on its own, because an association error made
    at frame 40 would otherwise steer the robot for the rest of the episode with no way
    to notice. `extract_tracks.py` can associate because it works offline and can look
    ahead; a controller cannot.
    """

    name = "sam"

    def __init__(self, *, device=None, vocab: list[str] | None = None,
                 keep: tuple[str, ...] = KEEP, min_pixels: int = 40,
                 min_prob: float = 0.25, points_per_side: int = 16,
                 max_area_frac: float = 0.25, logit_scale: float = 100.0,
                 checkpoint: Path | None = None) -> None:
        import clip
        import torch
        from segment_anything import (SamAutomaticMaskGenerator,
                                      sam_model_registry)

        from pipeline.world_model.extract_tracks import patch_mps_float64

        patch_mps_float64()
        if device is None:
            from common.device import pick_device
            # pick_device returns an Accelerator, not a torch.device; `.device` is the
            # part torch.nn.Module.to() accepts. Passing the wrapper fails only at the
            # first `.to()`, which is after the ViT-H checkpoint has been read off disk.
            device = pick_device().device
        self.device = device
        self.vocab = vocab or SIM_VOCAB
        self.keep = set(keep)
        self.min_pixels = min_pixels
        self.min_prob = min_prob
        self.max_area_frac = max_area_frac
        self.logit_scale = logit_scale

        ckpt = checkpoint or (ROOT / "openmask3d_semantic/checkpoints"
                                     "/sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt)).to(device).eval()
        # A coarse point grid on purpose. The scene holds four objects on a plain
        # table; the default 32x32 grid spends most of its 1024 prompts re-segmenting
        # the tablecloth and costs about four times as long per frame, which is the
        # difference between a closed loop that runs and one that does not.
        self.gen = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, pred_iou_thresh=0.85,
            stability_score_thresh=0.90, min_mask_region_area=60)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        with torch.no_grad():
            tf = self.clip_model.encode_text(
                clip.tokenize(self.vocab).to(device)).float()
            self.text_feats = tf / tf.norm(dim=-1, keepdim=True)

    def __call__(self, frame: dict) -> dict[str, Perceived]:
        from pipeline.world_model.extract_tracks import label_probabilities

        rgb = np.ascontiguousarray(frame["rgb"])
        h, w = rgb.shape[:2]
        masks = [m["segmentation"] for m in self.gen.generate(rgb)]
        # Drop the whole-scene masks SAM always produces. They are not wrong — the
        # background IS a region — but a crop of the entire image matches every prompt
        # about equally, and one of those near-ties wins a label often enough to matter.
        masks = [m for m in masks
                 if self.min_pixels <= m.sum() <= self.max_area_frac * h * w]
        if not masks:
            return {}
        # Shared with the real extraction path, so the two cannot drift: whichever
        # reading is correct, both use it.
        prob = label_probabilities(rgb, masks, self.clip_model, self.preprocess,
                                   self.device, self.text_feats,
                                   logit_scale=self.logit_scale)

        keep_cols = [(lab, self.vocab.index(lab)) for lab in self.vocab
                     if lab in self.keep]
        # Assign down the columns — for each label, the mask that best suits it — and
        # resolve greedily so no mask is handed to two labels. Row-wise argmax is the
        # obvious alternative and is measurably worse here: it loses a bowl segmented at
        # IoU 0.977 because that mask's own best prompt happens to be "a small dark
        # cube". A label is a claim about a mask; a mask is not a claim about a label.
        claims = sorted(((float(prob[r, c]), lab, r) for lab, c in keep_cols
                         for r in range(len(masks))), reverse=True)
        taken_mask: set[int] = set()
        chosen: dict[str, tuple[float, int]] = {}
        for pr, lab, r in claims:
            if pr < self.min_prob or lab in chosen or r in taken_mask:
                continue
            chosen[lab] = (pr, r)
            taken_mask.add(r)

        out = {}
        for lab, (pr, r) in chosen.items():
            m = masks[r]
            pts = mask_to_points(m, frame["depth"], frame["K"],
                                 frame["T_world_cam"])
            if len(pts) < self.min_pixels:
                continue
            c, e = summarise(pts)
            out[lab] = Perceived(lab, c, e, int(m.sum()), pr, mask=m)
        return out


#: Short noun phrases for the detector, mapped to the canonical labels the rest of the
#: stack uses. OWL-style detectors are trained on detection queries, not on captions,
#: and they localise better from a terse phrase than from the descriptive sentence CLIP
#: prefers — so the query and the label are deliberately different strings.
OWL_QUERIES = {
    "a dark cube": "a small dark cube",
    "an orange bowl": "an orange plastic bowl",
    "a green ball": "a green rubber ball",
}


class OwlSamPerception:
    """Open-vocabulary DETECTION, then SAM prompted by the box it found.

    Built because the measurement said to. `SamPerception` fails on the bowl 12 times
    in 32, and dumping every mask showed SAM segmenting it at IoU 0.977 while CLIP
    argmaxed that same mask to "a small dark cube". The weak component is CLIP being
    used as a classifier of crops — a job it was not trained for and is barely better
    than chance at on small low-texture objects, where its cosine similarities all sit
    in a 0.21-0.32 band.

    An open-vocabulary detector is trained for exactly the job being asked: given a text
    query, where is it. Handing its box to SAM as a prompt then recovers the precise
    mask, which SAM was always good at. Each model does the thing it is good at instead
    of one model doing both.

    It is also cheaper. `SamAutomaticMaskGenerator` segments the whole image and labels
    every region; this segments three boxes.
    """

    name = "owlsam"

    def __init__(self, *, device=None, queries: dict[str, str] | None = None,
                 threshold: float = 0.12, min_pixels: int = 40,
                 checkpoint: Path | None = None,
                 model_id: str = "google/owlv2-base-patch16-ensemble") -> None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        from pipeline.world_model.extract_tracks import patch_mps_float64

        patch_mps_float64()
        if device is None:
            from common.device import pick_device
            device = pick_device().device
        self.device = device
        self.queries = queries or OWL_QUERIES
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.torch = torch

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.detector = Owlv2ForObjectDetection.from_pretrained(model_id).to(
            device).eval()
        ckpt = checkpoint or (ROOT / "openmask3d_semantic/checkpoints"
                                     "/sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt)).to(device).eval()
        self.predictor = SamPredictor(sam)

    def detect(self, rgb: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
        """label -> (score, xyxy box). One box per label, the most confident."""
        from PIL import Image

        phrases = list(self.queries)
        inputs = self.processor(text=[phrases], images=Image.fromarray(rgb),
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.detector(**inputs)
        sizes = self.torch.tensor([[rgb.shape[0], rgb.shape[1]]]).to(self.device)
        res = self.processor.post_process_grounded_object_detection(
            outputs=out, target_sizes=sizes, threshold=self.threshold)[0]
        best: dict[str, tuple[float, np.ndarray]] = {}
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            name = self.queries[phrases[int(label)]]
            sc = float(score)
            if name not in best or sc > best[name][0]:
                best[name] = (sc, box.detach().cpu().numpy())
        return best

    def __call__(self, frame: dict) -> dict[str, Perceived]:
        rgb = np.ascontiguousarray(frame["rgb"])
        found = self.detect(rgb)
        if not found:
            return {}
        self.predictor.set_image(rgb)
        out = {}
        for label, (score, box) in found.items():
            masks, quality, _ = self.predictor.predict(
                box=box[None, :].astype(np.float32), multimask_output=False)
            m = masks[0].astype(bool)
            if int(m.sum()) < self.min_pixels:
                continue
            pts = mask_to_points(m, frame["depth"], frame["K"],
                                 frame["T_world_cam"])
            if len(pts) < self.min_pixels:
                continue
            c, e = summarise(pts)
            out[label] = Perceived(label, c, e, int(m.sum()), score,
                                   mask=m)
        return out


def make_perception(kind: str, **kw):
    if kind == "oracle":
        return OraclePerception(**{k: v for k, v in kw.items()
                                   if k in ("min_pixels", "keep")})
    if kind == "sam":
        return SamPerception(**kw)
    if kind == "owlsam":
        return OwlSamPerception(**{k: v for k, v in kw.items()
                                   if k in ("device", "queries", "threshold",
                                            "min_pixels", "checkpoint", "model_id")})
    raise ValueError(
        f"unknown perception backend {kind!r}; use 'oracle', 'sam' or 'owlsam'")

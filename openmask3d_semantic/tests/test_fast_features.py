"""The optimisation is only worth anything if it changes nothing.

Loop inversion is a reordering, so the test is equality against a mask-major reference
that mirrors upstream's nesting. Both run through the same seeded prompt generator, which
is the part that makes the comparison meaningful: upstream draws from the global numpy
RNG, so its own output depends on traversal order and is not even reproducible between
two runs of itself.

Stub SAM and CLIP throughout — no weights, no GPU, no upstream import.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmask3d_semantic.fast_features import (  # noqa: E402
    GraspableCriteria,
    _prompt_rounds,
    extract_mask_features,
    graspable_candidates,
)

torch = pytest.importorskip("torch")

H, W = 24, 32
FEATURE_DIM = 8


class StubTransform:
    def apply_coords_torch(self, coords, original_size):
        return coords


class StubPredictor:
    """Records set_image calls; produces masks that depend on the prompts."""

    def __init__(self):
        self.transform = StubTransform()
        self.device = "cpu"
        self.set_image_calls: list[int] = []
        self._image = None

    def set_image(self, image):
        self._image = image
        self.set_image_calls.append(int(image[0, 0, 0]))

    def predict_torch(self, point_coords, point_labels, multimask_output=False):
        rounds = point_coords.shape[0]
        masks = torch.zeros(rounds, 1, H, W, dtype=torch.bool)
        scores = torch.zeros(rounds, 1)
        for r in range(rounds):
            pts = point_coords[r]
            # Score depends on the prompt, so which round "wins" is a function of the
            # seed — which is exactly the order-dependence we need to pin down.
            scores[r, 0] = float((pts.sum().item() * 37) % 991) / 991.0
            cx = int(pts[:, 0].mean().item()) % W
            cy = int(pts[:, 1].mean().item()) % H
            masks[r, 0, max(0, cy - 3):cy + 3, max(0, cx - 3):cx + 3] = True
        return masks, scores, None


class StubImage:
    def __init__(self, tag: int):
        self.tag = tag

    def crop(self, box):
        return (self.tag, *box)


def stub_preprocess(crop):
    return np.asarray(crop, dtype=np.float32)


class StubClip:
    def encode_image(self, batch):
        # Deterministic, and varies with the crop contents so masks differ from each other.
        base = batch.sum(dim=1, keepdim=True)
        ramp = torch.arange(FEATURE_DIM, dtype=torch.float32).unsqueeze(0)
        return base + ramp


def stub_crop_box(mask_tensor, level, ratio):
    rows = torch.nonzero(mask_tensor.sum(axis=1))[:, 0]
    cols = torch.nonzero(mask_tensor.sum(axis=0))[:, 0]
    if len(rows) == 0:
        return 0, 0, 1, 1
    pad = level
    return (int(cols[0]) - pad, int(rows[0]) - pad, int(cols[-1]) + pad, int(rows[-1]) + pad)


def make_scene(num_masks=6, num_views=4, topk=3, seed=1):
    rng = np.random.default_rng(seed)
    topk_indices = np.stack([
        rng.choice(num_views, size=topk, replace=False) for _ in range(num_masks)
    ])
    visible = np.zeros((num_views, num_masks, H, W), dtype=bool)
    for v in range(num_views):
        for m in range(num_masks):
            cy, cx = rng.integers(5, H - 5), rng.integers(5, W - 5)
            visible[v, m, cy - 3:cy + 3, cx - 3:cx + 3] = True
    images_np = [np.full((H, W, 3), v, dtype=np.uint8) for v in range(num_views)]
    images_pil = [StubImage(v) for v in range(num_views)]
    return topk_indices, visible, images_np, images_pil


def run_view_major(predictor, **overrides):
    topk_indices, visible, images_np, images_pil = overrides.pop("scene")
    return extract_mask_features(
        topk_indices_per_mask=topk_indices,
        visible_points_in_view_in_mask=visible,
        images_np=images_np, images_pil=images_pil,
        predictor_sam=predictor, clip_model=StubClip(),
        clip_preprocess=stub_preprocess, crop_box_fn=stub_crop_box,
        feature_dim=FEATURE_DIM, num_levels=2, num_random_rounds=4,
        num_selected_points=3, **overrides,
    )


def run_mask_major(predictor, scene, candidate_mask=None, seed=0):
    """Upstream's nesting — view inside mask — with the same seeded prompts."""
    from openmask3d_semantic.fast_features import _best_mask_for

    topk_indices, visible, images_np, images_pil = scene
    num_masks = len(topk_indices)
    out = np.zeros((num_masks, FEATURE_DIM))
    clip_model = StubClip()

    for mask_index in range(num_masks):
        if candidate_mask is not None and not candidate_mask[mask_index]:
            continue
        crops = []
        for view in topk_indices[mask_index]:
            view = int(view)
            coords = np.transpose(np.where(visible[view][mask_index]))
            if coords.shape[0] == 0:
                continue
            predictor.set_image(images_np[view])
            best = _best_mask_for(
                predictor, coords, 4, 3,
                seed=seed + mask_index * 100_003 + view, image_hw=images_np[view].shape[:2],
            )
            for level in range(2):
                box = stub_crop_box(torch.from_numpy(best), level, 0.1)
                crops.append(stub_preprocess(images_pil[view].crop(box)))
        if crops:
            batch = torch.tensor(np.stack(crops))
            with torch.no_grad():
                feats = clip_model.encode_image(batch).float()
                feats /= feats.norm(dim=-1, keepdim=True)
            out[mask_index] = feats.mean(axis=0).numpy()
    return out


# ------------------------------------------------------------------- equality


def test_view_major_matches_mask_major_exactly():
    """The whole justification for the refactor."""
    scene = make_scene()

    fast = run_view_major(StubPredictor(), scene=scene)
    reference = run_mask_major(StubPredictor(), scene)

    assert np.array_equal(fast, reference), (
        f"max abs difference {np.abs(fast - reference).max():.3e} — the reordering "
        "changed the result, so it is not the optimisation it claims to be"
    )


def test_the_encoder_runs_once_per_distinct_view_not_once_per_pair():
    scene = make_scene()
    topk_indices = scene[0]

    fast_predictor = StubPredictor()
    run_view_major(fast_predictor, scene=scene)

    reference_predictor = StubPredictor()
    run_mask_major(reference_predictor, scene)

    distinct = len(np.unique(topk_indices))
    assert len(fast_predictor.set_image_calls) == distinct
    assert len(set(fast_predictor.set_image_calls)) == distinct, "a view was re-encoded"
    assert len(reference_predictor.set_image_calls) > len(fast_predictor.set_image_calls)


def test_prompts_do_not_depend_on_traversal_order():
    """The property that makes the two loops comparable at all. Upstream's global-RNG
    shuffle does not have it, which is also why upstream cannot reproduce itself."""
    coords = np.transpose(np.where(np.ones((6, 6), dtype=bool)))

    first = _prompt_rounds(coords, 4, 3, seed=42)
    np.random.shuffle(np.arange(1000))            # disturb the global RNG
    second = _prompt_rounds(coords, 4, 3, seed=42)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, _prompt_rounds(coords, 4, 3, seed=43))


def test_prompt_points_are_xy_not_row_column():
    """`np.where` yields (row, col); SAM's prompt encoder takes (x, y). Getting this
    backwards still returns a mask — of the transposed region."""
    coords = np.array([[2, 9]])                   # row 2, col 9
    rounds = _prompt_rounds(coords, 1, 1, seed=0)
    assert rounds[0, 0].tolist() == [9.0, 2.0]


# ------------------------------------------------------------------- gating


def test_gating_skips_work_and_leaves_the_row_zero():
    scene = make_scene()
    candidates = np.array([True, False, True, False, True, True])

    predictor = StubPredictor()
    out = run_view_major(predictor, scene=scene, candidate_mask=candidates)

    assert np.array_equal(out[~candidates], np.zeros((2, FEATURE_DIM)))
    assert (np.abs(out[candidates]).sum(axis=1) > 0).all()
    assert np.array_equal(out, run_mask_major(StubPredictor(), scene, candidates))


def test_gating_keeps_a_graspable_object_and_drops_the_floor():
    rng = np.random.default_rng(0)
    floor = rng.uniform([-3, -3, 0.0], [3, 3, 0.01], (4000, 3))
    pear = rng.uniform([0.1, 0.1, 0.75], [0.18, 0.18, 0.83], (300, 3))
    speck = rng.uniform([0, 0, 0.8], [0.004, 0.004, 0.804], (80, 3))

    points = np.vstack([floor, pear, speck])
    groups = [np.arange(4000), np.arange(4000, 4300), np.arange(4300, 4380)]

    keep, reasons = graspable_candidates(groups, points)

    assert keep.tolist() == [False, True, False]
    assert "gripper can span" in reasons[0]   # the diagonal check fires before footprint
    assert reasons[1] == ""
    assert "gripper" in reasons[2]


def test_gating_reports_a_reason_for_everything_it_drops():
    """Over-rejection removes the object the user asked for, and the symptom appears
    three stages later as 'not found'. Every rejection has to be greppable."""
    points = np.zeros((10, 3))
    keep, reasons = graspable_candidates([np.arange(5)], points)

    assert not keep[0]
    assert "points" in reasons[0] and len(reasons) == 1


def test_an_object_outside_the_workspace_is_out_of_reach():
    rng = np.random.default_rng(1)
    near = rng.uniform([0, 0, 0.8], [0.08, 0.08, 0.88], (200, 3))
    far = rng.uniform([4, 4, 0.8], [4.08, 4.08, 0.88], (200, 3))
    points = np.vstack([near, far])

    keep, reasons = graspable_candidates(
        [np.arange(200), np.arange(200, 400)], points,
        GraspableCriteria(workspace_centre=(0.0, 0.0, 0.8), workspace_radius_m=1.0),
    )

    assert keep.tolist() == [True, False]
    assert "out of reach" in reasons[1]

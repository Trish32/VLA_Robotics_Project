"""Regression tests for Eagle's image-packing mask on the dense attention paths.

Upstream communicated the packed-image layout only via `seq_len_list`, which only the
flash varlen wrapper reads. The stock sdpa/eager interfaces discarded it and attended
across image boundaries. patches/0002 builds the equivalent block-diagonal mask.

These are the guard against a re-pull silently restoring the leak. The rationale and a
standalone runner live in tools/check_attention_packing.py.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_attention_packing.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from check_attention_packing import (  # noqa: E402
    HIDDEN,
    IMG_TOKENS,
    build_attention,
    check_equivalence,
    check_leak,
    check_single_image,
    reference,
    run,
    siglip2_module,
)

IMPLS = ("eager", "sdpa")


@pytest.mark.parametrize("impl", IMPLS)
def test_no_cross_image_attention(impl):
    """Perturbing image B must not move image A. This is what packing guarantees."""
    assert check_leak(impl) <= 1e-6, f"{impl} attends across image boundaries"


@pytest.mark.parametrize("impl", IMPLS)
def test_matches_block_diagonal_reference(impl):
    """Stronger than the leak test: a mask that over-masks also leaks nothing.

    The reference is written out independently of the patched code path.
    """
    assert check_equivalence(impl) <= 1e-5


def test_single_image_is_unmasked():
    """One image means nothing to separate, so the mask must be None.

    Masking anyway would be harmless numerically but would cost an O(N^2) allocation on
    the common single-image path.
    """
    assert check_single_image() <= 1e-6


def test_mask_is_none_for_a_single_image():
    m = siglip2_module()
    assert m._build_packing_mask([12], 12, torch.device("cpu"), torch.float32) is None
    assert m._build_packing_mask([], 0, torch.device("cpu"), torch.float32) is None


def test_mask_is_block_diagonal():
    m = siglip2_module()
    mask = m._build_packing_mask([2, 3], 5, torch.device("cpu"), torch.float32)

    allowed = (mask[0, 0] == 0.0)
    expected = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(allowed, expected)


def test_inconsistent_layout_raises_rather_than_mismasking():
    """If seq_len_list does not sum to the sequence length we do not understand the
    layout. Masking the wrong spans would be silently wrong, so refuse instead."""
    m = siglip2_module()
    with pytest.raises(ValueError, match="cannot build the packing mask"):
        m._build_packing_mask([2, 2], 7, torch.device("cpu"), torch.float32)


def test_three_images_of_unequal_length():
    """The realistic case: images differ in token count."""
    attn = build_attention("sdpa")
    lens = [3, 7, 5]
    torch.manual_seed(4)
    x = torch.randn(1, sum(lens), HIDDEN)

    got = run(attn, x, lens)

    total = sum(lens)
    m = siglip2_module()
    mask = torch.full((total, total), float("-inf"))
    start = 0
    for n in lens:
        mask[start:start + n, start:start + n] = 0.0
        start += n
    with torch.no_grad():
        q = attn.q_proj(x).view(1, total, 4, -1).transpose(1, 2)
        k = attn.k_proj(x).view(1, total, 4, -1).transpose(1, 2)
        v = attn.v_proj(x).view(1, total, 4, -1).transpose(1, 2)
        out, _ = m.eager_attention_forward(attn, q, k, v, mask[None, None],
                                           scaling=attn.scale)
        want = attn.out_proj(out.reshape(1, total, HIDDEN))

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_reference_helper_agrees_with_itself():
    """Sanity: the reference used by the other tests is deterministic."""
    attn = build_attention("eager")
    torch.manual_seed(1)
    x = torch.randn(1, sum(IMG_TOKENS), HIDDEN)
    torch.testing.assert_close(
        reference(attn, x, IMG_TOKENS), reference(attn, x, IMG_TOKENS), atol=0, rtol=0
    )

"""Verify that the dense attention paths honour Eagle's image packing.

Background (the bug this guards against)
----------------------------------------
Eagle packs several images into one sequence and communicates the layout ONLY through
`seq_len_list`. Upstream's `Siglip2Attention.forward` hardcoded `attention_mask = None`
and passed `seq_len_list` down. Only `flash_attention_forward_for_packing` reads it --
it becomes `cu_seqlens` for `flash_attn_varlen_func`, giving block-diagonal attention,
one block per image. The stock `sdpa_attention_forward` / `eager_attention_forward`
accept `**kwargs`, silently discard `seq_len_list`, and with a null mask attend densely
across the whole packed sequence.

So substituting sdpa for flash was not a kernel swap: image A's tokens attended to image
B. patches/0002 builds the equivalent block-diagonal mask for the dense paths.

What this checks
----------------
1. LEAK -- perturbing image B must not change image A's output.
2. EQUIVALENCE -- the dense paths must match an explicit block-diagonal reference
   numerically. (1) alone is weak: a mask that over-masks also leaks nothing.
3. SINGLE IMAGE -- with one image there is nothing to separate, so the mask must be
   None and behaviour must be byte-identical to the unmasked path.

Runs on CPU. Requires neither flash-attn nor a GPU.

Usage:
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/check_attention_packing.py
"""

from __future__ import annotations

import sys
import warnings

import torch

warnings.filterwarnings("ignore")

IMG_TOKENS = [6, 6]  # two images, 6 tokens each
HEADS, HIDDEN = 4, 32
TOL = 1e-6


def siglip2_module():
    """Import the vendored modeling_siglip2 by path.

    The directory is `Eagle-Block2A-2B-v2` -- hyphens, so it is not importable as a
    package name.
    """
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "upstream/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_siglip2.py"
    )
    spec = importlib.util.spec_from_file_location("vendored_siglip2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vendored_siglip2"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_attention(impl: str):
    m = siglip2_module()
    cfg = m.Siglip2VisionConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        attention_dropout=0.0,
        use_rope=False,
        use_windows_attn=False,
    )
    cfg._attn_implementation = impl
    torch.manual_seed(0)
    return m.Siglip2Attention(cfg).eval()


def win_meta_for(seq_len_list):
    """One window per image, shaped (1, n) so win_hw[0]*win_hw[1] == n tokens."""
    return [{"img_idx": i, "win_hw": (1, n)} for i, n in enumerate(seq_len_list)]


def run(attn, x, seq_len_list):
    with torch.no_grad():
        return attn(x, win_meta_list=win_meta_for(seq_len_list))[0]


def reference(attn, x, seq_len_list):
    """Explicit block-diagonal attention, written out independently of the patch."""
    m = siglip2_module()
    total = sum(seq_len_list)
    mask = torch.full((total, total), float("-inf"))
    start = 0
    for n in seq_len_list:
        mask[start:start + n, start:start + n] = 0.0
        start += n

    with torch.no_grad():
        q = attn.q_proj(x).view(1, total, HEADS, -1).transpose(1, 2)
        k = attn.k_proj(x).view(1, total, HEADS, -1).transpose(1, 2)
        v = attn.v_proj(x).view(1, total, HEADS, -1).transpose(1, 2)
        out, _ = m.eager_attention_forward(attn, q, k, v, mask[None, None],
                                           scaling=attn.scale)
        return attn.out_proj(out.reshape(1, total, HIDDEN))


def check_leak(impl: str) -> float:
    """Max change in image A's output caused by perturbing image B only."""
    attn = build_attention(impl)
    torch.manual_seed(1)
    x = torch.randn(1, sum(IMG_TOKENS), HIDDEN)

    a = run(attn, x, IMG_TOKENS)
    x2 = x.clone()
    x2[:, IMG_TOKENS[0]:] = torch.randn_like(x2[:, IMG_TOKENS[0]:])  # image B only
    b = run(attn, x2, IMG_TOKENS)

    return float((a[:, :IMG_TOKENS[0]] - b[:, :IMG_TOKENS[0]]).abs().max())


def check_equivalence(impl: str) -> float:
    """Max abs difference between the dense path and the block-diagonal reference."""
    attn = build_attention(impl)
    torch.manual_seed(1)
    x = torch.randn(1, sum(IMG_TOKENS), HIDDEN)
    return float((run(attn, x, IMG_TOKENS) - reference(attn, x, IMG_TOKENS)).abs().max())


def check_single_image() -> float:
    """One image => nothing to separate => must equal plain dense attention."""
    attn = build_attention("sdpa")
    n = sum(IMG_TOKENS)
    torch.manual_seed(1)
    x = torch.randn(1, n, HIDDEN)

    packed = run(attn, x, [n])          # single image, mask must be None
    m = siglip2_module()
    with torch.no_grad():
        q = attn.q_proj(x).view(1, n, HEADS, -1).transpose(1, 2)
        k = attn.k_proj(x).view(1, n, HEADS, -1).transpose(1, 2)
        v = attn.v_proj(x).view(1, n, HEADS, -1).transpose(1, 2)
        out, _ = m.eager_attention_forward(attn, q, k, v, None, scaling=attn.scale)
        dense = attn.out_proj(out.reshape(1, n, HIDDEN))
    return float((packed - dense).abs().max())


def main() -> None:
    print(f"packing: {len(IMG_TOKENS)} images of {IMG_TOKENS} tokens\n")

    rows = []
    for impl in ("eager", "sdpa"):
        rows.append((f"{impl}: leak into image A", check_leak(impl), TOL))
        rows.append((f"{impl}: vs block-diagonal ref", check_equivalence(impl), 1e-5))
    rows.append(("single image: vs dense", check_single_image(), TOL))

    print(f"{'check':<40}{'value':>14}  verdict")
    print("-" * 64)
    ok = True
    for name, value, tol in rows:
        good = value <= tol
        ok &= good
        print(f"{name:<40}{value:>14.3e}  {'PASS' if good else 'FAIL'}")
    print("-" * 64)

    if ok:
        print("\nPacking honoured on the dense paths: no cross-image attention, and\n"
              "numerically equal to an independent block-diagonal reference.")
        sys.exit(0)
    print("\nFAILED -- the dense paths do not reproduce flash's packing semantics.")
    sys.exit(1)


if __name__ == "__main__":
    main()

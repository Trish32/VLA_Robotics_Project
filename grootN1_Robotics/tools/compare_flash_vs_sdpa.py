"""Diff the flash-attn path against our SDPA path on the real Eagle backbone. CUDA only.

This is the check that the local work cannot do. `tools/check_attention_packing.py`
proves the dense paths now reproduce block-diagonal packing semantics, but it compares
against a reference *we* wrote. Only a GPU with flash-attn can compare against the kernel
NVIDIA actually shipped and trained with.

Requires sm_80+ (Ampere/Ada/Hopper): FlashAttention-2 does not support Turing, so a T4
cannot run the reference half of this comparison at all. That is precisely why the
substitution exists, and precisely why it has to be validated somewhere else. An L4
(sm_89) is the cheapest box that can.

Method
------
Dispatch in `Siglip2Attention.forward` reads `self.config._attn_implementation` at
*forward* time, so one built model can be run both ways. That matters: the two paths then
share weights by construction, and any difference is attributable to attention alone
rather than to a reload.

Reports max/mean absolute difference on the backbone features, and — because raw max-rel
is meaningless near zero (see diffusiondrive_planner/bug_log.txt) — relative error restricted
to elements above the median magnitude.

Usage (on the GPU box):
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/compare_flash_vs_sdpa.py \
        --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B --images 2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — this comparison cannot run on CPU or MPS.")
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"[gpu] {name} (sm_{major}{minor})")
    if major < 8:
        raise SystemExit(
            f"FlashAttention-2 needs sm_80+; {name} is sm_{major}{minor}. "
            "The flash half of this comparison cannot run here — use an L4/A100/A10G."
        )
    try:
        import flash_attn  # noqa: F401
        print(f"[gpu] flash_attn {flash_attn.__version__}")
    except ImportError:
        raise SystemExit("flash_attn not installed: pip install flash-attn --no-build-isolation")


def set_impl(backbone, impl: str) -> None:
    """Flip attention implementation on both towers of an already-built model."""
    backbone.model.config._attn_implementation = impl
    backbone.model.vision_model.config._attn_implementation = impl
    backbone.model.language_model.config._attn_implementation = impl


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    mag = b.abs()
    median = mag.median()
    big = mag > median  # relative error is only meaningful away from zero
    rel_big = (diff[big] / mag[big]).max() if big.any() else torch.tensor(0.0)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel_all": float((diff / mag.clamp_min(1e-12)).max()),
        "max_rel_above_median": float(rel_big),
        "median_magnitude": float(median),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", type=int, default=2,
                    help="images packed per sample; >1 is where packing semantics bite")
    ap.add_argument("--select-layer", type=int, default=None,
                    help="override to shrink the LLM for a fast smoke run")
    args = ap.parse_args()

    require_cuda()

    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    ckpt = Path(args.checkpoint)
    cfg = Gr00tN1d6Config(**json.loads((ckpt / "config.json").read_text()))
    if args.select_layer is not None:
        cfg.select_layer = args.select_layer
    # Build once on the reference settings, then flip at forward time.
    cfg.use_flash_attention = True
    cfg.load_bf16 = True

    print(f"[build] Gr00tN1d6 from {ckpt}")
    model = Gr00tN1d6.from_pretrained(ckpt, config=cfg).cuda().eval()
    backbone = model.backbone

    # Synthetic but correctly-shaped VL input. The point is the attention path, not the
    # pixels: identical inputs both ways, only the kernel differs.
    torch.manual_seed(0)
    vision_cfg = backbone.model.config.vision_config
    per_image = 64
    total = per_image * args.images
    hidden = vision_cfg.hidden_size
    feats = torch.randn(1, total, hidden, device="cuda", dtype=torch.bfloat16)
    win_meta = [{"img_idx": i, "win_hw": (1, per_image)} for i in range(args.images)]

    layer = backbone.model.vision_model.vision_model.encoder.layers[0].self_attn

    outs = {}
    for impl in ("flash_attention_2", "sdpa"):
        set_impl(backbone, impl)
        with torch.no_grad():
            outs[impl] = layer(feats, win_meta_list=win_meta)[0].float()
        print(f"[run] {impl}: {tuple(outs[impl].shape)}")

    s = stats(outs["sdpa"], outs["flash_attention_2"])
    print(f"\n{'=' * 64}\nvision attention, {args.images} image(s) packed, "
          f"{per_image} tokens each\n{'=' * 64}")
    for k, v in s.items():
        print(f"  {k:<24}{v:>18.6e}")

    ok = s["max_rel_above_median"] < 1e-2
    print("=" * 64)
    print("\nSDPA reproduces the flash path." if ok else
          "\nDIVERGENCE — the substitution changes the function, not just the kernel.")
    if args.images > 1:
        print("Note: with >1 image this also exercises the packing mask from patches/0002.\n"
              "Re-run with --images 1 to separate kernel noise from packing semantics.")


if __name__ == "__main__":
    main()

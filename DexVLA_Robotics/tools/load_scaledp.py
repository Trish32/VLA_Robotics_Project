"""Load the released ScaleDP-H/L weights and report missing / unexpected keys.

This is the ONLY strict-load check available in this project. There is no released
DexVLA VLA checkpoint -- `lesjie/scale_dp_h` and `scale_dp_l` are stage-1 action-head
weights, and the backbone is stock Qwen2-VL-2B with no VLA post-training. So the house
Fidelity Rule ("official checkpoint loads 0/0, then reproduce the published metric")
can be satisfied for the head and for nothing else. See ../README.md.

The checkpoint is a raw torch save, not an HF model directory: the tensors live at
`ckpt["nets"]["nets"]` under a `noise_pred_net.` prefix, and there is no config.json --
so the architecture has to be reconstructed from the factory functions rather than read.

Usage:
    cd DexVLA_Robotics/upstream
    PYTHONPATH=. python ../tools/load_scaledp.py \
        --checkpoint ../checkpoints/scale_dp_h/open_scale_dp_h_backbone.ckpt --size ScaleDP-H
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

# action/state dims are embodiment-specific and NOT encoded in the released file.
# These are the repo's own defaults; they change the input/output adapters only, which
# is exactly the part stage-1 weights are not expected to carry.
DEFAULTS = dict(action_dim=14, state_dim=14, cond_dim=1536, prediction_horizon=16)


def checkpoint_tensors(path: Path) -> dict[str, torch.Size]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj
    # Unwrap the nested {"nets": {"nets": state_dict}} layout.
    while isinstance(sd, dict) and len(sd) == 1 and isinstance(next(iter(sd.values())), dict):
        inner = next(iter(sd.values()))
        if all(torch.is_tensor(v) for v in list(inner.values())[:5]):
            sd = inner
            break
        sd = inner
    return {k: v.shape for k, v in sd.items() if torch.is_tensor(v)}


def build(size: str, **overrides):
    from policy_heads.models.transformer_diffusion.configuration_scaledp import (
        ScaleDPPolicyConfig,
    )
    from policy_heads.models.transformer_diffusion.modeling_scaledp import ScaleDP

    cfg = ScaleDPPolicyConfig(model_size=size, **{**DEFAULTS, **overrides})
    return ScaleDP(cfg)


def report(name: str, keys: list[str], limit: int = 10) -> None:
    print(f"\n{name}: {len(keys)}")
    for k in keys[:limit]:
        print(f"    {k}")
    if len(keys) > limit:
        print(f"    ... and {len(keys) - limit} more")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    # NOTE the underscore. configuration_scaledp.MODEL_STRUCTURE is keyed
    # "ScaleDP_H"/"ScaleDP_L", while modeling_scaledp.ScaleDP_models is keyed
    # "ScaleDP-H"/"ScaleDP-L" -- the two files disagree, and the hyphen form raises
    # KeyError here.
    ap.add_argument("--size", default="ScaleDP_H", choices=["ScaleDP_H", "ScaleDP_L"])
    ap.add_argument("--prefix", default="noise_pred_net.",
                    help="stripped from checkpoint keys before comparison")
    ap.add_argument("--prediction-horizon", type=int, default=None,
                    help="override; the released pos_embed encodes the real value")
    ap.add_argument("--action-dim", type=int, default=None)
    ap.add_argument("--state-dim", type=int, default=None)
    ap.add_argument("--cond-dim", type=int, default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in (
        ("prediction_horizon", args.prediction_horizon),
        ("action_dim", args.action_dim),
        ("state_dim", args.state_dim),
        ("cond_dim", args.cond_dim),
    ) if v is not None}

    ckpt_raw = checkpoint_tensors(Path(args.checkpoint))
    ckpt = {k[len(args.prefix):] if k.startswith(args.prefix) else k: v
            for k, v in ckpt_raw.items()}
    print(f"[ckpt] {len(ckpt)} tensors, "
          f"{sum(s.numel() for s in ckpt.values()) / 1e9:.3f}B params "
          f"(prefix {args.prefix!r} stripped)")

    model = build(args.size, **overrides)
    model_sd = {k: v.shape for k, v in model.state_dict().items()}
    print(f"[model] {args.size}: {len(model_sd)} tensors, "
          f"{sum(p.numel() for p in model.state_dict().values()) / 1e9:.3f}B params")

    missing = sorted(set(model_sd) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model_sd))
    mismatched = sorted(k for k in set(ckpt) & set(model_sd) if ckpt[k] != model_sd[k])

    report("MISSING (model wants, checkpoint lacks)", missing)
    report("UNEXPECTED (checkpoint has, model lacks)", unexpected)
    print(f"\nSHAPE MISMATCH: {len(mismatched)}")
    for k in mismatched[:10]:
        print(f"    {k}: ckpt {tuple(ckpt[k])} vs model {tuple(model_sd[k])}")

    print("\n" + "=" * 66)
    if not (missing or unexpected or mismatched):
        print("0 missing / 0 unexpected / 0 shape mismatch")
    else:
        print(f"{len(missing)} missing / {len(unexpected)} unexpected / "
              f"{len(mismatched)} mismatched")
        print("\nNOTE: the released file is named *_backbone.ckpt. Adapters whose shapes")
        print("depend on action_dim/state_dim/cond_dim are embodiment-specific and are")
        print("not expected to be present — check the lists above before calling this a")
        print("failure, and re-run with matching dims if the mismatch is only in those.")
    print("=" * 66)


if __name__ == "__main__":
    main()

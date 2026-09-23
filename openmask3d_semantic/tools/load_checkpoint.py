#!/usr/bin/env python
"""Fidelity gate, step 1: does Mask3D load its released weights 0 missing / 0 unexpected?

This is the first thing that had to be true before any of `minkowski_compat.py` could be
believed. It is a real test of the substitution, not a formality: every convolution's
parameter shape is decided by our `kernel_offsets` and our ME-layout choices, so a wrong
kernel volume, a transposed layout or a missed special case shows up here as a shape
mismatch rather than as a silently worse metric.

Three ME conventions were found exactly this way, each of which had loaded "fine" in
isolation:

  * kernels are `(K, C_in, C_out)`, and the released file has no torch-style 5-D weight;
  * at kernel volume 1 ME drops the kernel axis entirely and stores `(C_in, C_out)` --
    Res16UNet34C's nine residual `downsample` projections and `mask_features_head`;
  * bias is `(1, C_out)`, not `(C_out,)`.

What this does NOT establish is that the kernel OFFSET ORDER matches MinkowskiEngine.
Shapes are order-independent, so this gate passes either way. See bug_log.txt [2].

    conda run -n openmask3d_vl python openmask3d_semantic/tools/load_checkpoint.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
UPSTREAM = PROJECT / "upstream/openmask3d/class_agnostic_mask_computation"

# Read off conf/model/mask3d.yaml, with the hydra interpolations resolved from the
# released checkpoint itself rather than guessed:
#   in_channels  3   <- backbone.conv0p1s1.kernel is (125, 3, 32)
#   out_channels 200 <- backbone.final.kernel is (96, 200)
#   num_classes  201 <- class_embed_head.weight is (201, 128), i.e. 200 + no-object
MODEL_CFG = {
    "hidden_dim": 128, "dim_feedforward": 1024, "num_queries": 100, "num_heads": 8,
    "num_decoders": 3, "dropout": 0.0, "pre_norm": False, "use_level_embed": False,
    "normalize_pos_enc": True, "positional_encoding_type": "fourier", "gauss_scale": 1.0,
    "hlevels": [0, 1, 2, 3],
    "non_parametric_queries": True, "random_query_both": False, "random_normal": False,
    "random_queries": False, "use_np_features": False,
    "sample_sizes": [200, 800, 3200, 12800, 51200], "max_sample_size": False,
    "shared_decoder": True, "num_classes": 201, "train_on_segments": True,
    "scatter_type": "mean", "voxel_size": 0.02,
    "config": {
        "backbone": {
            "_target_": "models.Res16UNet34C",
            "config": {"dialations": [1, 1, 1, 1], "conv1_kernel_size": 5,
                       "bn_momentum": 0.02},
            "in_channels": 3, "out_channels": 200, "out_fpn": True,
        }
    },
}


def build(ckpt: Path):
    sys.path.insert(0, str(ROOT))
    from openmask3d_semantic import me_shim

    me_shim.install_all()
    sys.path.insert(0, str(UPSTREAM))

    from omegaconf import OmegaConf

    from models.mask3d import Mask3D

    cfg = OmegaConf.create(MODEL_CFG)
    return Mask3D(**cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=PROJECT / "checkpoints/scannet200_model.ckpt")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"missing {args.ckpt}\nRun scripts/fetch_checkpoints.sh first.")

    import torch

    model = build(args.ckpt)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    # Lightning wraps the network as `model.` inside the LightningModule.
    state = {k.removeprefix("model."): v for k, v in state.items()}

    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    mismatch = [(k, tuple(state[k].shape), tuple(own[k].shape))
                for k in state if k in own and tuple(state[k].shape) != tuple(own[k].shape)]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build] Mask3D — {n_params/1e6:.2f}M params, {len(own)} tensors")
    print(f"[ckpt]  {len(state)} tensors")
    print(f"[diff]  missing {len(missing)}  unexpected {len(unexpected)}  "
          f"shape-mismatch {len(mismatch)}")
    for key in missing[:8]:
        print(f"    missing:    {key}  {tuple(own[key].shape)}")
    for key in unexpected[:8]:
        print(f"    unexpected: {key}  {tuple(state[key].shape)}")
    for key, got, want in mismatch[:8]:
        print(f"    mismatch:   {key}  ckpt {got} vs code {want}")

    ok = not (missing or unexpected or mismatch)
    if ok:
        model.load_state_dict(state, strict=True)
        print("\n[gate]  PASSED — Mask3D loads 0 missing / 0 unexpected / "
              "0 shape-mismatch (strict=True)")
        print("        Still open: kernel offset order vs real MinkowskiEngine "
              "(bug_log.txt [2]), and the published ScanNet200 metric.")
    else:
        print("\n[gate]  NOT PASSED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

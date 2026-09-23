"""Load the official GR00T N1.6-3B checkpoint and report missing / unexpected keys.

This is the Fidelity Rule gate for week 2: until the released weights land on the
adapted model at 0 missing / 0 unexpected, nothing downstream is measurable.

It builds the model with `use_flash_attention=False` / `load_bf16=False` so the check
runs on CPU. That substitution is attention-kernel and dtype only -- it changes no
parameter name or shape, so the key comparison is exactly the one that would run on an
H100. Shapes are compared too, since a renamed-but-same-shape key and a same-name-but-
reshaped key are different bugs.

Usage:
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/load_checkpoint.py \
        --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def checkpoint_tensors(ckpt_dir: Path) -> dict[str, torch.Size]:
    """Key -> shape for every tensor in the sharded safetensors checkpoint."""
    from safetensors import safe_open

    index = ckpt_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = [p.name for p in sorted(ckpt_dir.glob("*.safetensors"))]

    out: dict[str, torch.Size] = {}
    for shard in shards:
        with safe_open(str(ckpt_dir / shard), framework="pt") as f:
            for k in f.keys():
                out[k] = torch.Size(f.get_slice(k).get_shape())
    return out


def build_model(ckpt_dir: Path):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    cfg = Gr00tN1d6Config(**json.loads((ckpt_dir / "config.json").read_text()))
    # CPU/MPS build. Same parameters, different attention kernel and dtype.
    cfg.use_flash_attention = False
    cfg.load_bf16 = False
    cfg.model_dtype = "float32"
    cfg.torch_dtype = torch.float32
    return Gr00tN1d6(cfg), cfg


def report(name: str, keys: list[str], limit: int = 12) -> None:
    print(f"\n{name}: {len(keys)}")
    for k in keys[:limit]:
        print(f"    {k}")
    if len(keys) > limit:
        print(f"    ... and {len(keys) - limit} more")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    ckpt_dir = Path(args.checkpoint).resolve()

    print(f"[ckpt] {ckpt_dir}")
    ckpt = checkpoint_tensors(ckpt_dir)
    print(f"[ckpt] {len(ckpt)} tensors, {sum(s.numel() for s in ckpt.values()) / 1e9:.2f}B params")

    model, cfg = build_model(ckpt_dir)
    model_sd = {k: v.shape for k, v in model.state_dict().items()}
    print(f"[model] {len(model_sd)} tensors, "
          f"{sum(p.numel() for p in model.state_dict().values()) / 1e9:.2f}B params")

    missing = sorted(set(model_sd) - set(ckpt))      # model wants, checkpoint lacks
    unexpected = sorted(set(ckpt) - set(model_sd))   # checkpoint has, model lacks
    mismatched = sorted(k for k in set(ckpt) & set(model_sd) if ckpt[k] != model_sd[k])

    report("MISSING (model wants, checkpoint lacks)", missing)
    report("UNEXPECTED (checkpoint has, model lacks)", unexpected)
    print(f"\nSHAPE MISMATCH: {len(mismatched)}")
    for k in mismatched[:12]:
        print(f"    {k}: ckpt {tuple(ckpt[k])} vs model {tuple(model_sd[k])}")

    print("\n" + "=" * 64)
    if not (missing or unexpected or mismatched):
        print("0 missing / 0 unexpected / 0 shape mismatch  -- FIDELITY GATE PASSED")
    else:
        print(f"{len(missing)} missing / {len(unexpected)} unexpected / "
              f"{len(mismatched)} mismatched  -- GATE FAILED")
    print("=" * 64)


if __name__ == "__main__":
    main()

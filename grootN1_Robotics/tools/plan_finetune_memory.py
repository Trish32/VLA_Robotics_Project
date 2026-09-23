"""What actually fits: preset x optimiser vs the GPUs in our compute profile.

Builds the real model (weights not needed — parameter counts are structural) and reports
the steady-state memory floor for each combination.

Usage:
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/plan_finetune_memory.py \
        --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# name -> (VRAM GB, bf16?) from CLAUDE.md's compute profile.
GPUS = [("T4 (Kaggle/Colab free)", 16, False), ("2xT4 FSDP (Kaggle)", 32, False),
        ("L4 (GCP spot)", 24, True), ("A100 40GB", 40, True)]


def build(checkpoint: Path):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    cfg = Gr00tN1d6Config(**json.loads((checkpoint / "config.json").read_text()))
    cfg.use_flash_attention = False
    cfg.load_bf16 = False
    cfg.torch_dtype = torch.float32
    return Gr00tN1d6(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    from grootN1_Robotics.finetune import (
        PRESETS,
        configure_trainable,
        estimate_memory,
        parameter_census,
    )

    model = build(Path(args.checkpoint))
    census = parameter_census(model)
    print(f"\nmodel: {census['total'] / 1e9:.3f}B parameters total\n")

    print(f"{'preset':<14}{'optimiser':<13}{'trainable':>12}{'floor GB':>11}   fits (75% of)")
    print("-" * 78)
    for preset_name in ("released", "action_head", "projectors"):
        configure_trainable(model, preset_name)
        for opt in ("adamw", "adamw8bit"):
            est = estimate_memory(model, optimizer=opt)
            ok = [n for n, vram, _ in GPUS if est["total_gb"] < vram * 0.75]
            print(f"{preset_name:<14}{opt:<13}{est['trainable_params'] / 1e9:>11.3f}B"
                  f"{est['total_gb']:>11.1f}   {', '.join(ok) if ok else 'NOTHING'}")
        print()

    print("-" * 78)
    print("Floor excludes activations (batch/image/checkpointing dependent), so the")
    print("'fits' column already discounts to 75% of VRAM. Treat it as necessary, not")
    print("sufficient — confirm with a real step before trusting a session budget.")
    print(f"\nPreset notes:")
    for name, p in PRESETS.items():
        print(f"  {name:<14}{p.note}")


if __name__ == "__main__":
    main()

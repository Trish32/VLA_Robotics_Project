"""Build the DexVLA model from stock Qwen2-VL-2B + a ScaleDP head, and report the load.

There is no released DexVLA checkpoint, so this cannot be a 0/0 fidelity check (see
../README.md). What it CAN establish is the thing that actually blocks progress: that the
VLA config delta is right, that the two towers assemble, and that the stock Qwen2-VL
weights land on the backbone with nothing missing or unexpected on that side.

The config construction mirrors `train_vla.py::parse_param` exactly rather than being
reinvented — `using_film`, `policy_head_config` and `policy_head_size` are read by
`modeling_qwen2_vla.py` but are NOT declared on `Qwen2VLAConfig`, so they only exist if
the caller injects them. Reading the config class alone tells you nothing about them.

Usage:
    cd DexVLA_Robotics/upstream
    PYTHONPATH=. python ../tools/build_vla.py \
        --backbone ../checkpoints/Qwen2-VL-2B-Instruct --policy-head ScaleDP_H
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def build_config(backbone: Path, head_size: str, action_dim: int, state_dim: int,
                 chunk_size: int, using_film: bool):
    from transformers import AutoConfig

    import policy_heads  # noqa: F401  registers scale_dp_policy / unet_diffusion_policy

    # Registers model_type "qwen2_vla" (configuration_qwen2_vla.py, bottom of file).
    # Without this, AutoConfig.from_pretrained on the swapped config.json raises
    # "Transformers does not recognize this architecture" — which reads like a version
    # problem but is really a missing import.
    import qwen2_vla.models.configuration_qwen2_vla  # noqa: F401

    cfg = AutoConfig.from_pretrained(backbone, policy_head_type="scale_dp_policy")
    cfg.policy_head_size = head_size
    cfg.policy_head_config = AutoConfig.for_model(
        model_type=cfg.policy_head_type,
        model_size=head_size,
        cond_dim=cfg.hidden_size,      # 1536 for Qwen2-VL-2B; ScaleDP's default too
        action_dim=action_dim,
        prediction_horizon=chunk_size,
        state_dim=state_dim,
        is_tinyvla=False,
    )
    # train_vla.py sets these AFTER construction; they are not constructor args.
    setattr(cfg.policy_head_config, "input_dim", action_dim)
    setattr(cfg.policy_head_config, "state_dim", state_dim)
    cfg.using_film = using_film
    cfg.concat = "None"
    cfg.llm_loss_weight = 1.0
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--policy-head", default="ScaleDP_H",
                    choices=["ScaleDP_H", "ScaleDP_L"])
    ap.add_argument("--action-dim", type=int, default=14)
    ap.add_argument("--state-dim", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="ScaleDP prediction_horizon; the released head uses 50")
    ap.add_argument("--using-film", action="store_true", default=True)
    args = ap.parse_args()

    backbone = Path(args.backbone)
    cfg = build_config(backbone, args.policy_head, args.action_dim, args.state_dim,
                       args.chunk_size, args.using_film)
    print(f"[config] hidden_size={cfg.hidden_size} policy_head={cfg.policy_head_type}"
          f"/{cfg.policy_head_size} using_film={cfg.using_film}")
    # ScaleDPPolicyConfig stores the action dim as input_dim/output_dim, not action_dim.
    print(f"[config] head cond_dim={cfg.policy_head_config.cond_dim} "
          f"input_dim={cfg.policy_head_config.input_dim} "
          f"state_dim={cfg.policy_head_config.state_dim} "
          f"horizon={cfg.policy_head_config.prediction_horizon}")

    from qwen2_vla.models.modeling_qwen2_vla import (
        Qwen2VLForConditionalGenerationForVLA,
    )

    print("[build] assembling VLA (stock Qwen2-VL weights + randomly-init head) ...")
    model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(
        backbone, config=cfg, torch_dtype=torch.float32
    )
    model.eval()

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"\n{'component':<26}{'params':>14}")
    print("-" * 42)
    print(f"{'TOTAL':<26}{count(model) / 1e9:>13.3f}B")
    for name, mod in (("visual", getattr(model, "visual", None)),
                      ("language model", getattr(model, "model", None)),
                      ("policy_head", getattr(model, "policy_head", None))):
        if mod is not None:
            print(f"{name:<26}{count(mod) / 1e9:>13.3f}B")
    for name in ("input_action_proj", "reasoning_action_proj", "reasoning_film"):
        mod = getattr(model, name, None)
        if mod is not None:
            print(f"{name:<26}{count(mod) / 1e6:>13.2f}M")
    print("-" * 42)
    print("\nVLA assembles. NOTE: no released DexVLA checkpoint exists, so the policy head "
          "and\nthe film/projector modules are randomly initialised — this is an "
          "architecture check,\nnot a fidelity check.")


if __name__ == "__main__":
    main()

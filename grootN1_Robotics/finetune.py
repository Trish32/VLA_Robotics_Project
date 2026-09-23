"""Fine-tuning recipe for GR00T N1.6 under a 16GB free-tier budget.

The headline constraint, measured rather than assumed:

    total                      ~2.98B parameters
      backbone                  1.557B   (vision 428M + Qwen3 1.116B, truncated to 16 layers)
      action head               1.419B   (AlternateVLDiT 1.092B + projectors 327M)
    trainable with the SHIPPED config    1.620B
      (= the whole action head, PLUS LLM layers 12-15 via tune_top_llm_layers=4)

Adam needs roughly 16 bytes per trainable parameter (fp32 master + grad + two moments).
So the released recipe wants ~26GB of optimiser state alone, and even
"freeze the backbone, train the action head" wants ~23GB. Neither fits a 16GB T4 —
freezing the VLM is necessary but NOT sufficient, which is the thing the plan in
README.md originally got wrong.

What does fit is set out in `PRESETS` and checked by `estimate_memory`. Nothing here
guesses: `bytes_per_trainable_param` is explicit per optimiser, and the estimator refuses
to report a budget it cannot justify.

Activations are deliberately NOT estimated — they depend on batch size, image count and
gradient checkpointing, and a fabricated number would be worse than none. Treat the
reported total as a floor and leave headroom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# fp32 master weights (4) + fp32 grad (4) + optimiser state.
# Adam keeps two fp32 moments (8); 8-bit Adam quantises them to one byte each (2);
# SGD with momentum keeps one fp32 buffer (4); plain SGD keeps none.
OPTIMIZER_BYTES_PER_PARAM = {
    "adamw": 4 + 4 + 8,
    "adamw8bit": 4 + 4 + 2,
    "sgd_momentum": 4 + 4 + 4,
    "sgd": 4 + 4,
}

FROZEN_BYTES_PER_PARAM = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}


@dataclass(frozen=True)
class Preset:
    """A choice of what to train. `note` records why it exists, not what it does."""

    tune_projector: bool
    tune_diffusion_model: bool
    tune_vlln: bool
    tune_top_llm_layers: int
    note: str


PRESETS = {
    # The shipped recipe. Recorded so deviations are measured against something real.
    "released": Preset(True, True, True, 4,
                       "NVIDIA's own config; needs ~40GB, listed for reference only"),
    # Frozen VLM, whole action head. The obvious free-tier plan -- and it does not fit.
    "action_head": Preset(True, True, True, 0,
                          "frozen VLM; still ~23GB with AdamW because the DiT is 1.09B"),
    # The one that actually fits 16GB.
    "projectors": Preset(True, False, True, 0,
                         "state/action encoders + decoder + vlln only; fits a single T4"),
}


def configure_trainable(model, preset: str | Preset = "projectors") -> nn.Module:
    """Apply a preset via upstream's own `set_trainable_parameters` hooks.

    Deliberately routed through upstream rather than by setting `requires_grad` here:
    the action head also flips frozen submodules to eval() during training, which
    matters for its dropout, and reimplementing that would silently diverge.
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset

    model.action_head.set_trainable_parameters(
        tune_projector=p.tune_projector,
        tune_diffusion_model=p.tune_diffusion_model,
        tune_vlln=p.tune_vlln,
    )
    model.backbone.set_trainable_parameters(
        tune_llm=False, tune_visual=False, tune_top_llm_layers=p.tune_top_llm_layers
    )
    return model


def parameter_census(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def estimate_memory(
    model,
    optimizer: str = "adamw",
    frozen_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    """Floor on steady-state memory, in GB. Excludes activations -- see module docstring.

    Raises on an unknown optimiser rather than defaulting: a wrong bytes-per-parameter
    silently turns "does not fit" into "fits", which is the expensive direction to be
    wrong in.
    """
    if optimizer not in OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; known: {sorted(OPTIMIZER_BYTES_PER_PARAM)}"
        )
    if frozen_dtype not in FROZEN_BYTES_PER_PARAM:
        raise ValueError(f"unsupported frozen dtype {frozen_dtype}")

    census = parameter_census(model)
    gb = 1024 ** 3
    trainable_gb = census["trainable"] * OPTIMIZER_BYTES_PER_PARAM[optimizer] / gb
    frozen_gb = census["frozen"] * FROZEN_BYTES_PER_PARAM[frozen_dtype] / gb
    return {
        "trainable_params": census["trainable"],
        "frozen_params": census["frozen"],
        "trainable_gb": trainable_gb,
        "frozen_gb": frozen_gb,
        "total_gb": trainable_gb + frozen_gb,
    }


def fits(model, budget_gb: float, **kw) -> bool:
    """Whether the floor fits, with headroom for activations we do not estimate."""
    return estimate_memory(model, **kw)["total_gb"] < budget_gb * 0.75


def build_optimizer(model, lr: float = 1e-4, weight_decay: float = 1e-5,
                    kind: str = "adamw"):
    """Optimiser over trainable parameters only.

    Passing frozen parameters to AdamW would allocate moments for all ~3B of them, which
    is the single easiest way to OOM a T4 while believing the backbone is free.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters — check the preset")

    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # never silently fall back to a 4x heavier optimiser
            raise ImportError(
                "adamw8bit needs bitsandbytes (pip install bitsandbytes); refusing to "
                "fall back to AdamW, which needs 4x the optimiser state and would OOM"
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    if kind in ("sgd", "sgd_momentum"):
        return torch.optim.SGD(params, lr=lr,
                               momentum=0.9 if kind == "sgd_momentum" else 0.0,
                               weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {kind!r}")


def make_loss_fn():
    """`loss_fn(model, batch) -> loss` for `common.trainer.ResumableTrainer`.

    The head already returns the masked flow-matching MSE, so this only unwraps it.
    Training goes through ResumableTrainer per CLAUDE.md -- never a bespoke loop -- so
    that a killed Kaggle session resumes at step level.
    """

    def loss_fn(model, batch):
        out = model(batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        return loss, {"loss": float(loss.detach())}

    return loss_fn

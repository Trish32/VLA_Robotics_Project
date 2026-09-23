"""Tests for the fine-tune presets and the memory floor estimator.

The estimator is the thing that decides whether a Kaggle session is worth starting, so a
wrong bytes-per-parameter is expensive in the "believed it fit, OOMed 40 minutes in"
direction. These pin the arithmetic against hand-computed values.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_finetune.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grootN1_Robotics.finetune import (  # noqa: E402
    OPTIMIZER_BYTES_PER_PARAM,
    PRESETS,
    build_optimizer,
    estimate_memory,
    fits,
    make_loss_fn,
    parameter_census,
)


class Toy(nn.Module):
    """1M trainable + 1M frozen, so the arithmetic is checkable by hand."""

    def __init__(self):
        super().__init__()
        self.train_part = nn.Linear(1000, 1000, bias=False)   # 1e6 params
        self.frozen_part = nn.Linear(1000, 1000, bias=False)  # 1e6 params
        self.frozen_part.requires_grad_(False)


def test_census_splits_trainable_from_frozen():
    c = parameter_census(Toy())
    assert c["trainable"] == 1_000_000
    assert c["frozen"] == 1_000_000
    assert c["total"] == 2_000_000


def test_memory_floor_matches_hand_calculation():
    """AdamW = 16 B/param trainable; frozen fp16 = 2 B/param."""
    est = estimate_memory(Toy(), optimizer="adamw", frozen_dtype=torch.float16)
    gb = 1024 ** 3
    assert est["trainable_gb"] == pytest.approx(1_000_000 * 16 / gb)
    assert est["frozen_gb"] == pytest.approx(1_000_000 * 2 / gb)
    assert est["total_gb"] == pytest.approx(est["trainable_gb"] + est["frozen_gb"])


def test_8bit_adam_saves_exactly_the_two_moments():
    """The whole point of 8-bit Adam: 8 bytes of fp32 moments become 2."""
    adamw = estimate_memory(Toy(), optimizer="adamw")
    adam8 = estimate_memory(Toy(), optimizer="adamw8bit")
    gb = 1024 ** 3
    assert adamw["trainable_gb"] - adam8["trainable_gb"] == pytest.approx(
        1_000_000 * 6 / gb
    )


def test_unknown_optimizer_raises_rather_than_defaulting():
    """Defaulting would silently turn 'does not fit' into 'fits'."""
    with pytest.raises(ValueError, match="unknown optimizer"):
        estimate_memory(Toy(), optimizer="lion")


def test_fits_discounts_for_unestimated_activations():
    model = Toy()
    total = estimate_memory(model)["total_gb"]
    assert fits(model, budget_gb=total / 0.75 + 0.1)
    assert not fits(model, budget_gb=total)  # exactly at the floor is not "fits"


def test_optimizer_covers_only_trainable_parameters():
    """Handing frozen params to AdamW allocates moments for all of them — the easiest
    way to OOM a T4 while believing the backbone is free."""
    opt = build_optimizer(Toy(), kind="adamw")
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == 1_000_000


def test_optimizer_refuses_an_empty_parameter_set():
    model = Toy()
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model)


def test_adamw8bit_refuses_to_fall_back_silently():
    """If bitsandbytes is missing, falling back to AdamW would quadruple optimiser state
    and OOM — loudly failing is the safer behaviour."""
    pytest.importorskip
    try:
        import bitsandbytes  # noqa: F401
        pytest.skip("bitsandbytes installed; the fallback path cannot be exercised")
    except ImportError:
        with pytest.raises(ImportError, match="refusing to fall back"):
            build_optimizer(Toy(), kind="adamw8bit")


def test_presets_are_ordered_by_decreasing_cost():
    assert PRESETS["released"].tune_diffusion_model
    assert PRESETS["released"].tune_top_llm_layers == 4
    assert PRESETS["action_head"].tune_top_llm_layers == 0
    assert not PRESETS["projectors"].tune_diffusion_model, (
        "the projectors preset exists precisely because the 1.09B DiT does not fit 16GB"
    )


def test_loss_fn_unwraps_the_heads_dict():
    loss_fn = make_loss_fn()

    class FakeModel(nn.Module):
        def forward(self, batch):
            return {"loss": torch.tensor(1.5)}

    loss, metrics = loss_fn(FakeModel(), {})
    assert float(loss) == 1.5
    assert metrics["loss"] == 1.5


def test_optimizer_byte_table_is_self_consistent():
    """fp32 master + fp32 grad = 8 bytes floor for every optimiser."""
    for name, total in OPTIMIZER_BYTES_PER_PARAM.items():
        assert total >= 8, f"{name} cannot cost less than master+grad"
    assert OPTIMIZER_BYTES_PER_PARAM["sgd"] == 8

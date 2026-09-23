"""Strict checkpoint loading.

The fidelity bar for every port here is: the official checkpoint loads with
0 missing and 0 unexpected keys. A port that quietly drops tensors will still run
and still produce plausible-looking outputs, which is exactly how the Sparse4D-v3
`fix_scale` bug survived for days. So loading is loud by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointMismatch(RuntimeError):
    pass


def load_state_dict(path: str | Path, key: str | None = None) -> dict[str, torch.Tensor]:
    """Read a state dict from .pth/.pt/.ckpt/.safetensors, unwrapping common nesting."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))

    try:
        # weights_only only exists from torch 1.13; the mmdet-2.x env pins 1.12.
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if key is not None:
        return obj[key]
    for candidate in ("state_dict", "model", "module", "ema_state_dict"):
        if isinstance(obj, dict) and candidate in obj and isinstance(obj[candidate], dict):
            return obj[candidate]
    return obj


def strict_load(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    allow_missing: tuple[str, ...] = (),
    allow_unexpected: tuple[str, ...] = (),
    report: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Load `state` into `model` and raise unless the mismatch is explicitly allowed.

    `allow_missing` / `allow_unexpected` are prefix whitelists, for the genuinely
    expected cases: a freshly initialised task head, or reference-only buffers.
    Everything else is a porting bug and should stop the run.

    Writes a full key/shape diff to `report` when the load fails, which is the
    artifact you actually want when chasing a structural mismatch.
    """
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith(allow_missing)]
    unexpected = [
        k for k in incompatible.unexpected_keys if not k.startswith(allow_unexpected)
    ]

    if missing or unexpected:
        if report is not None:
            _write_report(model, state, missing, unexpected, Path(report))
        raise CheckpointMismatch(
            f"{len(missing)} missing / {len(unexpected)} unexpected keys.\n"
            f"  missing[:10]:    {missing[:10]}\n"
            f"  unexpected[:10]: {unexpected[:10]}"
            + (f"\n  full diff written to {report}" if report else "")
        )

    shape_errors = [
        f"{k}: model {tuple(dict(model.named_parameters()).get(k, torch.empty(0)).shape)}"
        f" != ckpt {tuple(v.shape)}"
        for k, v in state.items()
        if k in dict(model.named_parameters())
        and dict(model.named_parameters())[k].shape != v.shape
    ]
    if shape_errors:
        raise CheckpointMismatch("shape mismatch:\n  " + "\n  ".join(shape_errors[:10]))

    return incompatible.missing_keys, incompatible.unexpected_keys


def _write_report(model, state, missing, unexpected, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing": {k: list(v.shape) for k, v in model.state_dict().items() if k in set(missing)},
        "unexpected": {k: list(v.shape) for k, v in state.items() if k in set(unexpected)},
        "model_keys": len(model.state_dict()),
        "ckpt_keys": len(state),
    }
    path.write_text(json.dumps(payload, indent=2))


def param_count(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts — the LoRA sanity check."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

"""Load the official `droid.pth` into DroidNet and report missing / unexpected keys.

The Fidelity Rule gate for week 4. The update operator is plain PyTorch — no lietorch, no
droid_backends — so this runs on a Mac once those extensions are import-guarded at their
use sites (see ../compat.py and patches/).

The published checkpoint was saved from a DataParallel model, so every key carries a
`module.` prefix that has to come off before comparison.

Usage:
    cd droidSLAM_monocular/upstream/droid_slam
    PYTHONPATH=. python ../../tools/load_checkpoint.py --checkpoint ../../checkpoints/droid.pth
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # weights_only only exists from torch 1.13
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def strip_dataparallel(sd: dict) -> tuple[dict, bool]:
    if all(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}, True
    return sd, False


# The released droid.pth was trained with THREE output channels on both prediction heads;
# the shipped network declares two. Upstream reconciles this in
# `droid_slam/droid.py::Droid.load_weights` by keeping the first two channels and
# discarding the third. Replicating it here rather than inventing our own fix — the slice
# is upstream's stated intent, not a workaround.
_SLICED_TO_2 = (
    "update.weight.2.weight",
    "update.weight.2.bias",
    "update.delta.2.weight",
    "update.delta.2.bias",
)


def apply_upstream_slice(sd: dict) -> tuple[dict, list[str]]:
    sd = dict(sd)
    applied = []
    for key in _SLICED_TO_2:
        if key in sd and sd[key].shape[0] == 3:
            sd[key] = sd[key][:2]
            applied.append(key)
    return sd, applied


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

    from droid_net import DroidNet

    raw = load_state_dict(Path(args.checkpoint))
    ckpt, stripped = strip_dataparallel(raw)
    ckpt, sliced = apply_upstream_slice(ckpt)
    print(f"[ckpt] {len(ckpt)} tensors, "
          f"{sum(v.numel() for v in ckpt.values()) / 1e6:.2f}M params"
          f"{'  (module. prefix stripped)' if stripped else ''}")
    if sliced:
        print(f"[ckpt] applied upstream's 3->2 channel slice to {len(sliced)} tensors "
              f"(droid.py::load_weights)")

    model = DroidNet()
    model_sd = model.state_dict()
    print(f"[model] {len(model_sd)} tensors, "
          f"{sum(v.numel() for v in model_sd.values()) / 1e6:.2f}M params")

    missing = sorted(set(model_sd) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model_sd))
    mismatched = sorted(k for k in set(ckpt) & set(model_sd)
                        if ckpt[k].shape != model_sd[k].shape)

    report("MISSING (model wants, checkpoint lacks)", missing)
    report("UNEXPECTED (checkpoint has, model lacks)", unexpected)
    print(f"\nSHAPE MISMATCH: {len(mismatched)}")
    for k in mismatched[:12]:
        print(f"    {k}: ckpt {tuple(ckpt[k].shape)} vs model {tuple(model_sd[k].shape)}")

    if not (missing or unexpected or mismatched):
        model.load_state_dict(ckpt, strict=True)  # prove it, do not just compare keys
        print("\n[load] strict=True succeeded")

    print("\n" + "=" * 64)
    if not (missing or unexpected or mismatched):
        print("0 missing / 0 unexpected / 0 shape mismatch  -- FIDELITY GATE PASSED")
    else:
        print(f"{len(missing)} missing / {len(unexpected)} unexpected / "
              f"{len(mismatched)} mismatched  -- GATE FAILED")
    print("=" * 64)


if __name__ == "__main__":
    main()

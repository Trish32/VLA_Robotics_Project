#!/usr/bin/env python
"""Fidelity gate, step 1: do the refiner and scorer load 0 missing / 0 unexpected?

Nothing else about FoundationPose is measurable until this passes, and it is reachable
on CPU: `learning/models/{refine,score}_network.py` are plain PyTorch, and `compat.py`
supplies raise-on-use placeholders for the CUDA-only renderer they import via `Utils`.

Two things this does NOT do, deliberately:

  * it does not use `PoseRefinePredictor` / `ScorePredictor`. Both hardcode `.cuda()` in
    their constructors and both build an H5 dataset object on the way, so neither can
    run here. The networks are constructed directly instead.
  * it does not relax `strict=True`. A shape mismatch or a stray key is the finding, not
    an inconvenience -- see droidSLAM's bug_log [2], where four "mismatched" heads turned
    out to be an intentional upstream slice, and the wrong conclusion came from reading
    the model definition without reading the loader.

The config defaults below are copied from the predictors verbatim. They are not
cosmetic: `c_in`, `use_normal` and `use_BN` change the first convolution and the norm
layers, so a missing default produces a genuinely different network and a load failure
that looks like a checkpoint problem.

    python foundationpose_6dof/tools/load_checkpoint.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT / "upstream"

REFINER_RUN = "2023-10-28-18-33-37"   # readme.md: "For the refiner, you will need ..."
SCORER_RUN = "2024-01-11-20-02-45"    # readme.md: "For scorer, you will need ..."

# Verbatim from predict_pose_refine.py / predict_score.py. Shared keys first.
COMMON_DEFAULTS = {
    "use_normal": False,
    "use_BN": False,
    "c_in": 4,
    "normalize_xyz": False,
    "crop_ratio": 1.2,
}
REFINER_DEFAULTS = {
    **COMMON_DEFAULTS,
    "use_mask": False,
    "n_view": 1,
    "trans_rep": "tracknet",
    "rot_rep": "axis_angle",
    "zfar": 3,
    "normal_uint8": False,
}
SCORER_DEFAULTS = {**COMMON_DEFAULTS, "zfar": float("inf")}


def _prepare_imports() -> dict[str, bool]:
    sys.path.insert(0, str(PROJECT))
    import compat

    status = compat.ensure_importable()

    # `from network_modules import *` is a flat import, so the models directory has to be
    # importable on its own, not just the upstream root.
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / "learning" / "models"))
    return status


def _config(run: str, defaults: dict):
    from omegaconf import OmegaConf

    path = _weights_dir(run) / "config.yml"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun scripts/fetch_checkpoints.sh first -- the config ships "
            "beside the weights and defines the architecture they were trained with."
        )
    cfg = OmegaConf.load(path)
    for key, value in defaults.items():
        if key not in cfg or (key == "crop_ratio" and cfg[key] is None):
            cfg[key] = value
    if isinstance(cfg["zfar"], str) and "inf" in cfg["zfar"].lower():
        cfg["zfar"] = float("inf")
    return cfg


def _weights_dir(run: str) -> Path:
    """Our checkpoints/ is the source of truth; upstream/weights/ may symlink to it."""
    ours = PROJECT / "checkpoints" / run
    return ours if ours.exists() else UPSTREAM / "weights" / run


def check(name: str, run: str, defaults: dict, build) -> bool:
    import torch

    ckpt_path = _weights_dir(run) / "model_best.pth"
    if not ckpt_path.exists():
        print(f"  {name}: MISSING {ckpt_path}")
        return False

    cfg = _config(run, defaults)
    model = build(cfg)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    # DataParallel-era checkpoints carry a `module.` prefix the bare network lacks.
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    own = model.state_dict()
    mismatched = [
        (k, tuple(v.shape), tuple(own[k].shape))
        for k, v in state.items()
        if k in own and tuple(v.shape) != tuple(own[k].shape)
    ]
    result = model.load_state_dict(state, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    ok = not result.missing_keys and not result.unexpected_keys and not mismatched
    print(
        f"  {name}: {len(state)} tensors, {n_params/1e6:.2f}M params | "
        f"missing {len(result.missing_keys)}, unexpected {len(result.unexpected_keys)}, "
        f"shape-mismatch {len(mismatched)}  -> {'PASS' if ok else 'FAIL'}"
    )
    for key in result.missing_keys[:5]:
        print(f"      missing:    {key}")
    for key in result.unexpected_keys[:5]:
        print(f"      unexpected: {key}")
    for key, got, want in mismatched[:5]:
        print(f"      mismatch:   {key}  ckpt {got} vs code {want}")
    return ok


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    status = _prepare_imports()
    import compat

    print(f"[env] {compat.describe(status)}")

    try:
        from refine_network import RefineNet
        from score_network import ScoreNetMultiPair
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the networks: {exc}\n"
            "These are plain PyTorch, but `Utils.py` pulls in trimesh, open3d, pandas, "
            "transformations and ruamel.yaml at module scope. Install those in this "
            "environment; do NOT stub them -- unlike the renderer they are cheap and "
            "real, and stubbing them would hide a genuine import error."
        ) from exc

    print("[load] strict comparison against the released weights")
    refiner = check(
        "refiner", REFINER_RUN, REFINER_DEFAULTS,
        lambda cfg: RefineNet(cfg=cfg, c_in=cfg["c_in"]),
    )
    scorer = check(
        "scorer", SCORER_RUN, SCORER_DEFAULTS,
        lambda cfg: ScoreNetMultiPair(cfg=cfg, c_in=cfg["c_in"]),
    )

    passed = refiner and scorer
    print(f"\n[gate] {'PASSED' if passed else 'NOT PASSED'} — 0 missing / 0 unexpected "
          "/ 0 shape mismatch on both networks")
    if passed:
        print("       Note: this is the LOAD gate only. The published ADD-S on "
              "YCB-Video needs the renderer and a CUDA box.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

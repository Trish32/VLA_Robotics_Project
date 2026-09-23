"""Thin adaptation layer over upstream's `LeRobotEpisodeLoader`.

Per the Adaptation Rule we use upstream's reader rather than reimplementing it. This
module adds only what upstream leaves dangerous or undiscoverable:

1. **Strict modality keys.** Upstream warns and drops the column when a state/action key
   is not found — `_extract_joint_groups` prints "Joint group 'x' not found" and moves
   on. The result is a DataFrame silently missing `state.*`, which trains a model on
   nothing and looks like a bad hyperparameter. We validate against `modality.json` and
   raise.

2. **A working video backend on macOS.** The default is `torchcodec`, which is not
   installed; `torchvision_av` shells out to `ffprobe`; `av` raises NotImplementedError.
   `opencv` works. We resolve to the first backend that actually imports rather than
   letting a cloud-only default fail locally.

Key naming is the subtle part and is not documented upstream:

    video / state / action  ->  BARE keys      ("front", "single_arm", "gripper")
    language                ->  PREFIXED key   ("annotation.human.task_description")
    output DataFrame columns -> ALWAYS PREFIXED ("state.single_arm", "video.front")

So the strings you pass in are not the strings you get back, except for language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Backends in preference order. torchcodec is upstream's default and is the fastest,
# but it is not installable on Apple Silicon; opencv is the local fallback.
# Ordered by preference, but ONLY backends that upstream's index-based reader actually
# dispatches on (`gr00t/utils/video_utils.py: get_frames_by_indices` handles torchcodec,
# decord, ffmpeg and opencv, and raises NotImplementedError on anything else).
#
# `torchvision_av` is deliberately NOT here. It reads fine by TIMESTAMP, so it looks like
# a working backend, and the probe used to return it whenever `ffprobe` was on PATH --
# which made installing ffmpeg for an unrelated reason silently break frame-index reads
# with a bare NotImplementedError several layers down. A backend this module offers must
# be one every reader upstream can dispatch.
_VIDEO_BACKENDS = ("torchcodec", "decord", "opencv")
_INDEX_READABLE = frozenset({"torchcodec", "decord", "ffmpeg", "opencv"})

# Modalities whose modality_keys are bare names looked up inside modality.json.
_BARE_KEY_MODALITIES = ("video", "state", "action")


def available_video_backend(preferred: str | None = None) -> str:
    """First video backend that actually works here.

    Probing beats guessing: `torchvision_av` imports fine and only fails later, when it
    cannot find the `ffprobe` binary.
    """
    import importlib
    import shutil

    candidates = (preferred,) if preferred else _VIDEO_BACKENDS
    if preferred and preferred not in _INDEX_READABLE:
        raise ValueError(
            f"{preferred!r} cannot serve frame-INDEX reads; upstream's "
            f"get_frames_by_indices dispatches only on {sorted(_INDEX_READABLE)} and "
            "raises NotImplementedError otherwise."
        )
    for backend in candidates:
        if backend == "torchcodec":
            try:
                importlib.import_module("torchcodec")
                return backend
            except ImportError:
                continue
        if backend == "decord":
            try:
                importlib.import_module("decord")
                return backend
            except ImportError:
                continue
        if backend == "opencv":
            try:
                importlib.import_module("cv2")
                return backend
            except ImportError:
                continue
    raise RuntimeError(
        "no usable video backend: install torchcodec, or ffmpeg (for torchvision_av), "
        "or opencv-python"
    )


def read_modality_meta(dataset_path: str | Path) -> dict[str, Any]:
    path = Path(dataset_path) / "meta" / "modality.json"
    if not path.exists():
        raise FileNotFoundError(f"not a LeRobot dataset (no meta/modality.json): {path}")
    return json.loads(path.read_text())


def validate_modality_configs(dataset_path: str | Path, modality_configs: dict) -> None:
    """Raise if any modality key is absent from modality.json.

    Upstream only warns for state/action, and the offending column then never appears in
    the returned DataFrame. A typo therefore produces a model trained without
    proprioception rather than a traceback.
    """
    meta = read_modality_meta(dataset_path)
    problems: list[str] = []

    for modality, cfg in modality_configs.items():
        if modality in _BARE_KEY_MODALITIES:
            known = set(meta.get(modality, {}))
            for key in cfg.modality_keys:
                if key in known:
                    continue
                hint = ""
                if "." in key and key.split(".", 1)[1] in known:
                    hint = f" (drop the prefix: use {key.split('.', 1)[1]!r})"
                problems.append(
                    f"{modality}.modality_keys: {key!r} not in modality.json"
                    f"{hint}; available: {sorted(known)}"
                )
        elif modality == "language":
            # Language keys ARE prefixed, and the prefix is the modality.json section.
            known = {f"annotation.{k}" for k in meta.get("annotation", {})}
            for key in cfg.modality_keys:
                if key not in known:
                    problems.append(
                        f"language.modality_keys: {key!r} not in modality.json; "
                        f"available: {sorted(known)}"
                    )

    if problems:
        raise ValueError(
            "modality config does not match this dataset:\n  " + "\n  ".join(problems)
        )


def build_loader(
    dataset_path: str | Path,
    modality_configs: dict,
    video_backend: str | None = None,
    **kwargs,
):
    """`LeRobotEpisodeLoader` with keys validated and a working video backend."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    validate_modality_configs(dataset_path, modality_configs)
    return LeRobotEpisodeLoader(
        dataset_path,
        modality_configs,
        video_backend=video_backend or available_video_backend(),
        **kwargs,
    )


def default_modality_configs(dataset_path: str | Path, action_horizon: int = 16) -> dict:
    """Every key the dataset declares, at the given action horizon.

    Useful for smoke tests and for inspecting an unfamiliar dataset; a real training run
    should name its keys explicitly so that a dataset change is a loud failure.
    """
    from gr00t.data.types import ModalityConfig

    meta = read_modality_meta(dataset_path)
    cfgs = {}
    for modality in _BARE_KEY_MODALITIES:
        if modality in meta:
            cfgs[modality] = ModalityConfig(
                delta_indices=list(range(action_horizon)) if modality == "action" else [0],
                modality_keys=list(meta[modality]),
            )
    if "annotation" in meta:
        keys = [f"annotation.{k}" for k in meta["annotation"]]
        # Upstream asserts exactly one language key and a single timestep.
        cfgs["language"] = ModalityConfig(delta_indices=[0], modality_keys=keys[:1])
    return cfgs

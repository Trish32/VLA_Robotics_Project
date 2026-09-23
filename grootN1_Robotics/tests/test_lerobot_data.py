"""Contract tests for LeRobot loading, against upstream's real demo_data.

These run on the actual parquet + mp4 files upstream ships, so they pin the real format
rather than a mock of it. They need the git-lfs objects fetched:

    cd grootN1_Robotics/upstream && git lfs install --local && git lfs pull

Without that, every data file is a ~130-byte pointer and parquet reads fail with
"magic bytes not found" — which is what a fresh clone hits.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_lerobot_data.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grootN1_Robotics.lerobot_data import (  # noqa: E402
    available_video_backend,
    build_loader,
    default_modality_configs,
    read_modality_meta,
    validate_modality_configs,
)

DATASET = Path(__file__).resolve().parents[1] / "upstream/demo_data/cube_to_bowl_5"


def lfs_fetched() -> bool:
    parquet = next(DATASET.glob("data/**/*.parquet"), None)
    return parquet is not None and parquet.stat().st_size > 1000


requires_data = pytest.mark.skipif(
    not DATASET.exists() or not lfs_fetched(),
    reason="demo_data git-lfs objects not fetched (see module docstring)",
)


def modality_configs():
    from gr00t.data.types import ModalityConfig

    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
        "action": ModalityConfig(delta_indices=list(range(16)),
                                 modality_keys=["single_arm", "gripper"]),
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


@requires_data
def test_modality_json_declares_the_expected_sections():
    meta = read_modality_meta(DATASET)
    assert set(meta) == {"state", "action", "video", "annotation"}
    assert set(meta["state"]) == {"single_arm", "gripper"}
    assert set(meta["video"]) == {"front", "wrist"}


@requires_data
def test_loads_an_episode_with_every_modality():
    loader = build_loader(DATASET, modality_configs())
    assert len(loader) == 5

    ep = loader[0]
    cols = set(ep.columns)

    # Output columns are PREFIXED even though the input keys were bare.
    assert {"state.single_arm", "state.gripper", "action.single_arm", "action.gripper",
            "video.front", "language.annotation.human.task_description"} == cols

    assert np.asarray(ep["state.single_arm"].iloc[0]).shape == (5,)
    assert np.asarray(ep["state.gripper"].iloc[0]).shape == (1,)
    assert np.asarray(ep["video.front"].iloc[0]).shape == (480, 640, 3)
    assert np.asarray(ep["video.front"].iloc[0]).dtype == np.uint8
    assert isinstance(ep["language.annotation.human.task_description"].iloc[0], str)


@requires_data
def test_getitem_returns_a_whole_episode_not_one_sample():
    """Easy to misread: this is an *episode* loader.

    `loader[0]` is a DataFrame of every frame in episode 0, so treating it as a training
    sample yields a batch of one entire trajectory.
    """
    loader = build_loader(DATASET, modality_configs())
    assert len(loader[0]) == loader.episode_lengths[0]
    assert list(loader.episode_lengths) == [568, 745, 454, 1797, 584]


@requires_data
def test_prefixed_state_key_raises_instead_of_silently_dropping_the_column():
    """The failure this module exists to prevent.

    Upstream's `_extract_joint_groups` prints "Joint group not found" and continues, so
    a prefixed key yields a DataFrame with no state at all — a model trained without
    proprioception, and no traceback anywhere.
    """
    from gr00t.data.types import ModalityConfig

    cfgs = modality_configs()
    cfgs["state"] = ModalityConfig(delta_indices=[0], modality_keys=["state.single_arm"])

    with pytest.raises(ValueError, match="drop the prefix"):
        build_loader(DATASET, cfgs)


@requires_data
def test_unknown_video_key_raises():
    from gr00t.data.types import ModalityConfig

    cfgs = modality_configs()
    cfgs["video"] = ModalityConfig(delta_indices=[0], modality_keys=["nonexistent_cam"])

    with pytest.raises(ValueError, match="not in modality.json"):
        validate_modality_configs(DATASET, cfgs)


@requires_data
def test_unknown_language_key_raises():
    from gr00t.data.types import ModalityConfig

    cfgs = modality_configs()
    cfgs["language"] = ModalityConfig(delta_indices=[0], modality_keys=["annotation.nope"])

    with pytest.raises(ValueError, match="not in modality.json"):
        validate_modality_configs(DATASET, cfgs)


@requires_data
def test_default_configs_round_trip():
    cfgs = default_modality_configs(DATASET)
    validate_modality_configs(DATASET, cfgs)  # must not raise
    assert set(cfgs) == {"state", "action", "video", "language"}
    assert len(cfgs["language"].modality_keys) == 1, "upstream asserts exactly one"
    assert cfgs["action"].delta_indices == list(range(16))


def test_video_backend_probe_returns_something_usable():
    """torchcodec is upstream's default and is absent on Apple Silicon; torchvision_av
    needs the ffprobe binary. The probe must not return a backend that only fails later."""
    assert available_video_backend() in {"torchcodec", "torchvision_av", "opencv"}


def test_missing_dataset_raises_clearly():
    with pytest.raises(FileNotFoundError, match="no meta/modality.json"):
        read_modality_meta("/nonexistent/dataset")

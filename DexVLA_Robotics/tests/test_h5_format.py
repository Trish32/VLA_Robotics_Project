"""Tests for the DexVLA HDF5 episode format validator.

Built against synthetic files rather than a real dataset, because no DexVLA dataset is
public — the format exists only as index expressions in `data_utils/utils.py`. Each test
constructs a file that is wrong in exactly one way, which is also how the format gets
documented by example.

Run: pytest DexVLA_Robotics/tests/test_h5_format.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DexVLA_Robotics.h5_format import (  # noqa: E402
    DEFAULT_CAMERAS,
    check_dimensional_consistency,
    inspect_episode,
    validate_dataset,
)

h5py = pytest.importorskip("h5py")

T, STATE, ACTION, H, W = 12, 14, 14, 8, 10


def write_episode(path, length=T, cameras=DEFAULT_CAMERAS, reasoning="substep",
                  action_len=None, camera_frames=None, camera_dtype=np.uint8,
                  language=True, substep_len=None):
    with h5py.File(path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((length, STATE), np.float32))
        obs.create_dataset("qvel", data=np.zeros((length, STATE), np.float32))
        images = obs.create_group("images")
        for cam in cameras:
            n = camera_frames if camera_frames is not None else length
            images.create_dataset(
                cam, data=np.zeros((n, H, W, 3), dtype=camera_dtype)
            )
        f.create_dataset("action", data=np.zeros((action_len or length, ACTION), np.float32))
        if language:
            f.create_dataset("language_raw", data=[b"pick up the cube"])
        if reasoning == "substep":
            n = substep_len if substep_len is not None else length
            f.create_dataset("substep_reasonings", data=[b"move left"] * n)
        elif reasoning == "episode":
            f.create_dataset("reasoning", data=[b"first grasp, then place"])
    return path


def test_valid_episode_reports_no_problems(tmp_path):
    info = inspect_episode(write_episode(tmp_path / "ep0.hdf5"))

    assert info.ok, info.problems
    assert info.length == T
    assert info.state_dim == STATE
    assert info.action_dim == ACTION
    assert set(info.cameras) == set(DEFAULT_CAMERAS)
    assert info.language == "pick up the cube"
    assert info.reasoning_mode == "substep"


def test_substep_reasonings_takes_precedence_over_reasoning(tmp_path):
    """load_from_h5 checks substep_reasonings FIRST and only falls back.

    A file with both is read per-timestep, which is easy to get wrong if you assume the
    episode-level string is used.
    """
    path = tmp_path / "both.hdf5"
    write_episode(path, reasoning="substep")
    with h5py.File(path, "a") as f:
        f.create_dataset("reasoning", data=[b"episode level"])

    assert inspect_episode(path).reasoning_mode == "substep"


def test_short_substep_reasonings_is_caught(tmp_path):
    """Indexed by timestep, so a short array raises mid-epoch — hours in, not at start."""
    info = inspect_episode(write_episode(tmp_path / "short.hdf5", substep_len=T - 3))

    assert not info.ok
    assert any("substep_reasonings" in p for p in info.problems)


def test_episode_level_reasoning_is_accepted(tmp_path):
    info = inspect_episode(write_episode(tmp_path / "ep.hdf5", reasoning="episode"))
    assert info.ok, info.problems
    assert info.reasoning_mode == "episode"


def test_missing_reasoning_is_flagged_when_required(tmp_path):
    """use_reasoning=True on a file with neither trains on a blank string — silently."""
    info = inspect_episode(write_episode(tmp_path / "none.hdf5", reasoning=None))

    assert not info.ok
    assert info.reasoning_mode == "none"

    relaxed = inspect_episode(tmp_path / "none.hdf5", require_reasoning=False)
    assert relaxed.ok, relaxed.problems


def test_action_length_mismatch_is_caught(tmp_path):
    info = inspect_episode(write_episode(tmp_path / "bad.hdf5", action_len=T + 5))
    assert any("action length" in p for p in info.problems)


def test_camera_frame_count_mismatch_is_caught(tmp_path):
    info = inspect_episode(write_episode(tmp_path / "bad.hdf5", camera_frames=T - 1))
    assert any("frames" in p for p in info.problems)


def test_renamed_camera_is_caught(tmp_path):
    """Camera names are dataset KEYS from aloha_scripts/constants.py, not metadata.

    A renamed camera is a KeyError deep inside the loader; here it is one clear message.
    """
    info = inspect_episode(
        write_episode(tmp_path / "renamed.hdf5", cameras=("cam_top", "cam_wrist"))
    )
    assert sum("missing camera" in p for p in info.problems) == len(DEFAULT_CAMERAS)


def test_float_images_are_caught(tmp_path):
    """Images must be uint8; float would silently blow up memory and skew normalisation."""
    info = inspect_episode(
        write_episode(tmp_path / "float.hdf5", camera_dtype=np.float32)
    )
    assert any("dtype" in p for p in info.problems)


def test_missing_required_dataset_short_circuits(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("language_raw", data=[b"hello"])

    info = inspect_episode(path)
    assert not info.ok
    assert any("/observations/qpos" in p for p in info.problems)


def test_unopenable_file_is_a_problem_not_a_skip(tmp_path):
    """The upstream failure this module exists to avoid.

    `check_data_integrity.py` logs the error and falls through, appending the PREVIOUS
    file's qpos — so a corrupt episode duplicates its predecessor into the normalisation
    statistics instead of failing.
    """
    (tmp_path / "corrupt.hdf5").write_bytes(b"not an hdf5 file at all")
    write_episode(tmp_path / "good.hdf5")

    episodes = validate_dataset(tmp_path)

    assert len(episodes) == 2
    corrupt = next(e for e in episodes if e.path.name == "corrupt.hdf5")
    assert not corrupt.ok
    assert any("could not open" in p for p in corrupt.problems)


def test_inconsistent_dims_across_episodes_are_caught(tmp_path):
    """state/action dims size the ScaleDP head; a mixed dataset must not train."""
    write_episode(tmp_path / "a.hdf5")
    path_b = tmp_path / "b.hdf5"
    with h5py.File(path_b, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((T, 7), np.float32))
        obs.create_dataset("qvel", data=np.zeros((T, 7), np.float32))
        images = obs.create_group("images")
        for cam in DEFAULT_CAMERAS:
            images.create_dataset(cam, data=np.zeros((T, H, W, 3), np.uint8))
        f.create_dataset("action", data=np.zeros((T, 7), np.float32))
        f.create_dataset("language_raw", data=[b"other task"])
        f.create_dataset("substep_reasonings", data=[b"x"] * T)

    problems = check_dimensional_consistency(validate_dataset(tmp_path))

    assert any("state_dim" in p for p in problems)
    assert any("action_dim" in p for p in problems)

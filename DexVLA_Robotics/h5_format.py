"""The DexVLA/DexVLA HDF5 episode format, documented and validated.

The layout is not written down anywhere upstream — it exists only as index expressions
scattered through `data_utils/utils.py::load_from_h5`. This module states it once and
checks a file against it, so a malformed dataset fails before a training run rather than
during one.

    /observations/qpos              (T, state_dim)   float
    /observations/qvel              (T, state_dim)   float
    /observations/images/<cam>      (T, H, W, C)     uint8, one dataset per camera
    /action                         (T, action_dim)  float
    language_raw                    (>=1,) bytes     [0] decoded utf-8
    reasoning                       (>=1,) bytes     [0] decoded utf-8   (per EPISODE)
    substep_reasonings              (T,)   bytes     [t] decoded utf-8   (per TIMESTEP)

Two things the index expressions imply but never state:

1. **`substep_reasonings` takes precedence over `reasoning`.** `load_from_h5` checks for
   it first and only falls back. So a file containing both is read per-timestep, and a
   file whose `substep_reasonings` is shorter than the episode raises mid-epoch.
2. **Camera names are dataset keys, not metadata.** They come from
   `aloha_scripts/constants.py` (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) and must
   match the file exactly — a renamed camera is a KeyError deep in the loader.

Why not reuse upstream's `data_utils/check_data_integrity.py`: it hardcodes an absolute
path at module scope, and on a file that fails to open it logs the error and then falls
through to `all_qpos_data.append(torch.from_numpy(qpos))` — appending the PREVIOUS file's
array. A corrupt episode therefore duplicates its predecessor into the normalisation
statistics instead of failing. See bug_log.txt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass
class EpisodeInfo:
    path: Path
    length: int
    state_dim: int
    action_dim: int
    cameras: dict[str, tuple]
    language: str
    reasoning_mode: str  # "substep" | "episode" | "none"
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def inspect_episode(path: str | Path, cameras: tuple[str, ...] = DEFAULT_CAMERAS,
                    require_reasoning: bool = True) -> EpisodeInfo:
    """Describe one episode file and collect every problem found.

    Collects rather than raising on the first, so one pass reports everything wrong with
    a file instead of one thing per run.
    """
    import h5py

    path = Path(path)
    problems: list[str] = []

    with h5py.File(path, "r") as root:
        for key in ("/observations/qpos", "/observations/qvel", "/action"):
            if key not in root:
                problems.append(f"missing required dataset {key}")
        if problems:
            return EpisodeInfo(path, 0, 0, 0, {}, "", "none", problems)

        qpos = root["/observations/qpos"]
        qvel = root["/observations/qvel"]
        action = root["/action"]
        length = qpos.shape[0]

        if qvel.shape[0] != length:
            problems.append(f"qvel length {qvel.shape[0]} != qpos length {length}")
        if action.shape[0] != length:
            problems.append(f"action length {action.shape[0]} != qpos length {length}")

        found_cameras: dict[str, tuple] = {}
        for cam in cameras:
            key = f"/observations/images/{cam}"
            if key not in root:
                problems.append(f"missing camera {cam!r} (names must match exactly)")
                continue
            ds = root[key]
            found_cameras[cam] = tuple(ds.shape)
            if ds.shape[0] != length:
                problems.append(f"camera {cam!r} has {ds.shape[0]} frames, expected {length}")
            if ds.dtype != "uint8":
                problems.append(f"camera {cam!r} dtype is {ds.dtype}, expected uint8")
            if ds.ndim != 4 or ds.shape[-1] != 3:
                problems.append(f"camera {cam!r} shape {tuple(ds.shape)}, expected (T,H,W,3)")

        language = ""
        if "language_raw" not in root:
            problems.append("missing 'language_raw'")
        else:
            language = _decode(root["language_raw"][0])

        # substep_reasonings WINS over reasoning -- load_from_h5 checks it first.
        if "substep_reasonings" in root:
            mode = "substep"
            n = root["substep_reasonings"].shape[0]
            if n != length:
                problems.append(
                    f"substep_reasonings has {n} entries but the episode is {length} "
                    "steps; load_from_h5 indexes it by timestep and will raise mid-epoch"
                )
        elif "reasoning" in root:
            mode = "episode"
        else:
            mode = "none"
            if require_reasoning:
                problems.append(
                    "no 'substep_reasonings' or 'reasoning'; use_reasoning=True will "
                    "train on a blank reasoning string"
                )

        return EpisodeInfo(
            path=path,
            length=length,
            state_dim=int(qpos.shape[1]),
            action_dim=int(action.shape[1]),
            cameras=found_cameras,
            language=language,
            reasoning_mode=mode,
            problems=problems,
        )


def validate_dataset(directory: str | Path, cameras: tuple[str, ...] = DEFAULT_CAMERAS,
                     require_reasoning: bool = True) -> list[EpisodeInfo]:
    """Inspect every .hdf5 under `directory`. A file that cannot be opened is a problem,
    never a skip — that is the upstream failure mode this exists to avoid."""
    directory = Path(directory)
    episodes = []
    for path in sorted(directory.rglob("*.hdf5")):
        try:
            episodes.append(inspect_episode(path, cameras, require_reasoning))
        except Exception as exc:  # noqa: BLE001 - report, never silently continue
            episodes.append(
                EpisodeInfo(path, 0, 0, 0, {}, "", "none",
                            [f"could not open: {type(exc).__name__}: {exc}"])
            )
    return episodes


def check_dimensional_consistency(episodes: list[EpisodeInfo]) -> list[str]:
    """State/action dims must agree across episodes: they size the ScaleDP head."""
    problems = []
    for name, dims in (("state_dim", {e.state_dim for e in episodes if e.ok}),
                       ("action_dim", {e.action_dim for e in episodes if e.ok})):
        if len(dims) > 1:
            problems.append(f"inconsistent {name} across episodes: {sorted(dims)}")
    return problems

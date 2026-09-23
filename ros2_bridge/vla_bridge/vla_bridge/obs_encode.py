"""Encoding ROS inputs into the observation layout a real policy server accepts.

Kept free of rclpy, like `wire.py` and `action_decode.py`, so it can be tested outside a
ROS container — and, more importantly, so it can be handed straight to a REAL policy's
`check_observation` on the Mac. That is the only test that means anything here.

The layout is NOT obvious and the node originally got it wrong in three ways at once
(see ../../bug_log.txt [3]). Upstream `Gr00tPolicy.check_observation` requires:

    observation["video"][key]     np.uint8   (B, T, H, W, C)   ndim 5, C == 3
    observation["state"][key]     np.float32 (B, T, D)         ndim 3
    observation["language"][key]  list[list[str]]              B items x T strings

Note what that means:

  * the dict is **nested**, not flat — `observation["video"]["ego_view"]`, never
    `observation["video.ego_view"]`. The flat form exists, but only inside
    `Gr00tSimPolicyWrapper`, which converts it to this one before calling the policy;
  * the keys inside each group are **bare** (`ego_view`), because the `video.` prefix
    names the *dataset column*, not the modality key;
  * there is a **batch axis as well as a time axis**. A single live frame is
    `(1, 1, H, W, C)`, not `(1, H, W, C)`.

DiffusionVLA's server (`DexVLA_Robotics/policy_server.py`) was written by mirroring the same
upstream code, so it expects the same nested layout. One encoder therefore serves both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIDEO_PREFIX = "video."
STATE_PREFIX = "state."
# What GR00T's GR1 configs call the instruction. Only used when the server declares no
# language modality of its own (DiVLA does not — it reads whatever key it is given).
DEFAULT_LANGUAGE_KEY = "annotation.human.task_description"


class ObservationEncodeError(Exception):
    """Raised instead of sending an observation the server will reject or misread."""


def strip_modality_prefix(key: str, prefix: str) -> str:
    """`video.ego_view` -> `ego_view`; `ego_view` -> `ego_view`.

    Accepting both spellings is safe because the mapping is one-way and unambiguous, and
    the result is checked against the server's declared keys at startup anyway.
    """
    return key[len(prefix):] if key.startswith(prefix) else key


@dataclass(frozen=True)
class ModalityLayout:
    """What the server says it wants. Horizons are `len(delta_indices)`.

    `state_dims` is the width of each entry in `state_keys`, in the same order, and is
    the one thing here the wire cannot tell us: `get_modality_config` carries key names
    and delta indices but no dimensions. Real embodiments split the state — GR1 is
    `left_arm 7, left_hand 6, right_arm 7, right_hand 6, waist 3` — so a single
    `/joint_states` vector has to be cut up, and only the operator knows where the cuts
    go relative to their `joint_names`. Empty means "one key, take the whole vector".
    """

    video_keys: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    state_dims: tuple[int, ...] = ()
    language_key: str | None = None
    video_horizon: int = 1
    state_horizon: int = 1
    language_horizon: int = 1


def _field(cfg, name):
    """ModalityConfig arrives as a dataclass locally and as a plain dict over msgpack."""
    value = getattr(cfg, name, None)
    if value is None and isinstance(cfg, dict):
        value = cfg.get(name)
    return value


def _horizon(cfg) -> int:
    deltas = _field(cfg, "delta_indices")
    return len(deltas) if deltas else 1


def parse_layout(modality_config) -> ModalityLayout | None:
    """Read the video/state/language layout out of `get_modality_config`.

    Returns None when the server declares none of them — DiVLA declares only `action` —
    in which case the caller falls back to its own parameters.
    """
    if not isinstance(modality_config, dict):
        return None

    fields: dict = {}
    for group, attr in (("video", "video_keys"), ("state", "state_keys")):
        cfg = modality_config.get(group)
        if cfg is None:
            continue
        keys = _field(cfg, "modality_keys") or []
        fields[attr] = tuple(keys)
        fields[f"{group}_horizon"] = _horizon(cfg)

    lang_cfg = modality_config.get("language")
    if lang_cfg is not None:
        keys = _field(lang_cfg, "modality_keys") or []
        if keys:
            # Upstream asserts len(language_keys) == 1; mirror that rather than guessing.
            if len(keys) != 1:
                raise ObservationEncodeError(
                    f"server declares {len(keys)} language keys ({list(keys)}); "
                    "upstream supports exactly one"
                )
            fields["language_key"] = keys[0]
            fields["language_horizon"] = _horizon(lang_cfg)

    return ModalityLayout(**fields) if fields else None


def reconcile(configured_video: list[str], configured_state: list[str],
              configured_state_dims: list[int],
              layout: ModalityLayout | None) -> ModalityLayout:
    """Merge the node's parameters with what the server declared, or fail loudly.

    A mismatch here is a misconfiguration that would otherwise surface as an assertion
    inside the model, one tick into a live episode.

    The configured ORDER is preserved, not sorted: `state_dims[i]` slices the joint
    vector for `state_keys[i]`, so reordering the keys would silently reassign joints to
    the wrong limb. (Contrast `action_decode.select_keys`, which sorts — there the order
    is a column layout the node itself defines, with nothing to disagree with.)
    """
    video = tuple(strip_modality_prefix(k, VIDEO_PREFIX) for k in configured_video)
    state = tuple(strip_modality_prefix(k, STATE_PREFIX) for k in configured_state)
    dims = tuple(int(d) for d in configured_state_dims)

    if len(set(state)) != len(state):
        raise ObservationEncodeError(f"duplicate state modality keys: {list(state)}")
    if dims and len(dims) != len(state):
        raise ObservationEncodeError(
            f"state_dims has {len(dims)} entries but there are {len(state)} state keys; "
            "they are paired positionally"
        )

    if layout is None:
        _require_dims(state, dims, declared=None)
        return ModalityLayout(video_keys=video, state_keys=state, state_dims=dims,
                              language_key=DEFAULT_LANGUAGE_KEY)

    if layout.video_keys and set(video) != set(layout.video_keys):
        raise ObservationEncodeError(
            f"camera_modality_keys {sorted(video)} do not match the keys the server "
            f"declares {sorted(layout.video_keys)}"
        )
    if layout.state_keys and set(state) != set(layout.state_keys):
        raise ObservationEncodeError(
            f"state_modality_keys {sorted(state)} do not match the server's declared "
            f"state keys {sorted(layout.state_keys)}"
        )
    if layout.video_horizon != 1 or layout.state_horizon != 1:
        raise ObservationEncodeError(
            f"server wants video horizon {layout.video_horizon} and state horizon "
            f"{layout.state_horizon}; this node holds only the latest frame. Padding by "
            "repeating it would feed the model a stationary history it never saw in "
            "training."
        )

    keys = state or layout.state_keys
    _require_dims(keys, dims, declared=layout.state_keys)
    return ModalityLayout(
        video_keys=video or layout.video_keys,
        state_keys=keys,
        state_dims=dims,
        language_key=layout.language_key or DEFAULT_LANGUAGE_KEY,
        video_horizon=layout.video_horizon,
        state_horizon=layout.state_horizon,
        language_horizon=layout.language_horizon,
    )


def _require_dims(state_keys, dims, declared) -> None:
    """A multi-key embodiment cannot be served without knowing where to cut.

    Guessing an even split would produce a perfectly well-formed observation with the
    wrong joints in every group — the model would run, and the arm would move wrongly.
    """
    if len(state_keys) > 1 and not dims:
        hint = f" (the server declares {sorted(declared)})" if declared else ""
        raise ObservationEncodeError(
            f"this embodiment splits state across {len(state_keys)} keys "
            f"{list(state_keys)}{hint}, so `state_dims` must give each one's width, in "
            "the same order, summing to the number of joints in `joint_names`. "
            "GR00T's GR1, for example, is left_arm 7, left_hand 6, right_arm 7, "
            "right_hand 6, waist 3."
        )


def build_observation(frames: dict[str, np.ndarray], state: np.ndarray, task: str,
                      layout: ModalityLayout) -> dict:
    """(latest frames, joint positions, instruction) -> the nested observation dict.

    `frames` is keyed by modality key (bare or `video.`-prefixed) and holds one
    `(H, W, C)` uint8 image each; `state` is a 1-D vector of joint positions.
    """
    frames = {strip_modality_prefix(k, VIDEO_PREFIX): v for k, v in frames.items()}

    missing = sorted(set(layout.video_keys) - set(frames))
    if missing:
        raise ObservationEncodeError(f"no frame for video key(s) {missing}")

    video = {}
    for key in layout.video_keys:
        arr = np.asarray(frames[key])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ObservationEncodeError(
                f"video {key!r} has shape {arr.shape}; expected (H, W, 3)"
            )
        if arr.dtype != np.uint8:
            # Upstream asserts on dtype. Converting a float image here would silently
            # wrap values; refuse instead.
            raise ObservationEncodeError(
                f"video {key!r} has dtype {arr.dtype}; the server requires uint8"
            )
        video[key] = arr[None, None]  # (B=1, T=1, H, W, C)

    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_arr.size == 0:
        raise ObservationEncodeError("state vector is empty")
    if not np.isfinite(state_arr).all():
        raise ObservationEncodeError("non-finite values in the state vector")

    state_dict = _split_state(state_arr, layout)

    language_key = layout.language_key or DEFAULT_LANGUAGE_KEY
    # list[list[str]]: B batch items, each holding `language_horizon` strings.
    language = {language_key: [[str(task)] * layout.language_horizon]}

    return {"video": video, "state": state_dict, "language": language}


def _split_state(state: np.ndarray, layout: ModalityLayout) -> dict[str, np.ndarray]:
    """Cut the joint vector into per-key slices, each `(B=1, T=1, D)`."""
    if not layout.state_keys:
        raise ObservationEncodeError("no state modality key configured")

    if not layout.state_dims:
        if len(layout.state_keys) != 1:
            raise ObservationEncodeError(
                f"{len(layout.state_keys)} state keys but no state_dims to split by"
            )
        return {layout.state_keys[0]: state[None, None, :]}

    total = sum(layout.state_dims)
    if total != state.size:
        raise ObservationEncodeError(
            f"state_dims sum to {total} but /joint_states supplied {state.size} values; "
            f"keys {list(layout.state_keys)} widths {list(layout.state_dims)}"
        )

    out, start = {}, 0
    for key, width in zip(layout.state_keys, layout.state_dims):
        out[key] = state[None, None, start:start + width]
        start += width
    return out

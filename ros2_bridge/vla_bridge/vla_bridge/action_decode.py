"""Decoding a policy server's action reply into a (horizon, dof) chunk.

Kept free of rclpy — like `wire.py` — so it is testable outside a ROS container. That
matters here specifically: this logic is where two servers disagree, and the disagreement
is silent.

    GR00T raw Gr00tPolicy      -> BARE keys      "left_arm", "right_arm", "waist"
    GR00T sim wrapper / DiVLA  -> PREFIXED keys  "action.x", "action.joints"

The node originally required the `action.` prefix, so every chunk from a raw GR00T server
was dropped with "no action.* keys in reply" — and the end-to-end test missed it because
its fake server was written to send `action.arm`.
"""

from __future__ import annotations

import numpy as np

ACTION_PREFIX = "action."


class ActionDecodeError(Exception):
    """Raised instead of returning a partial or guessed chunk."""


def declared_action_keys(modality_config) -> list[str] | None:
    """Pull the action key names out of a server's modality config.

    Handles both the dataclass-ish and plain-dict shapes that come back over msgpack.
    Returns None when the server declares nothing, which is a valid answer.
    """
    if not isinstance(modality_config, dict):
        return None
    action_cfg = modality_config.get("action")
    if action_cfg is None:
        return None
    keys = getattr(action_cfg, "modality_keys", None)
    if keys is None and isinstance(action_cfg, dict):
        keys = action_cfg.get("modality_keys")
    return sorted(keys) if keys else None


def select_keys(action: dict, declared: list[str] | None) -> list[str]:
    """Which reply keys hold the action, in the order their columns will appear.

    Order is sorted and deterministic because the columns map positionally onto
    `joint_names` — a reordering silently permutes the joints.

    Never falls back to "every value that looks like an array": DiVLA's reply carries a
    `reasoning` string and servers may add diagnostics, and widening the chunk by one
    column shifts every joint after it.
    """
    if declared:
        # Sorted HERE as well as in declared_action_keys, so column order does not depend
        # on how the caller happened to obtain the list. The columns map positionally
        # onto joint_names; if that mapping shifted with the server's declaration order,
        # a server-side reorder would silently permute the joints.
        wanted = sorted(declared)
        missing = sorted(set(wanted) - set(action))
        if missing:
            raise ActionDecodeError(
                f"server declared {wanted} but omitted {missing}; refusing to publish "
                "a partial chunk"
            )
        return wanted

    prefixed = sorted(k for k in action if k.startswith(ACTION_PREFIX))
    if prefixed:
        return prefixed

    raise ActionDecodeError(
        f"no usable action keys in reply: {sorted(action)[:6]}. The server neither "
        f"declared modality keys nor used the {ACTION_PREFIX!r} prefix."
    )


def decode_chunk(action: dict, declared: list[str] | None = None) -> np.ndarray:
    """Reply dict -> (horizon, dof) float32, or raise.

    A leading batch axis is dropped: servers return (1, H, D) but the node wants (H, D).
    """
    keys = select_keys(action, declared)

    parts = []
    for key in keys:
        arr = np.asarray(action[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ActionDecodeError(
                f"action {key!r} has shape {arr.shape}; expected (H,), (H, D) or (1, H, D)"
            )
        parts.append(arr)

    horizons = {p.shape[0] for p in parts}
    if len(horizons) != 1:
        raise ActionDecodeError(
            f"inconsistent horizons across action keys: "
            f"{ {k: p.shape[0] for k, p in zip(keys, parts)} }"
        )

    chunk = np.concatenate(parts, axis=1)
    if not np.isfinite(chunk).all():
        raise ActionDecodeError("non-finite values in the action chunk")
    return chunk

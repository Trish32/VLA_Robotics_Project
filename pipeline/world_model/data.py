"""(z_t, a_t, z_{t+1}) transitions from LeRobot episodes.

Two observable streams, both real and both action-conditioned:

  * **proprioception** — `observation.state` (44 dims on these datasets), which becomes
    the robot half of the latent;
  * **object motion** — the per-frame foreground masks shipped with
    `cube_to_bowl_5_with_mask`, reduced to a centroid and an area per camera, which
    become a slot.

The slot geometry from masks is in **image-plane units** (normalised pixels), not
metres. The architecture does not care — it is a latent either way — but the honest
consequence is that a model trained here has learned how a mask moves under an action,
not how a 3-D object does. Nothing in this file converts between the two, and nothing
should pretend the trained weights transfer to the metric scene graph without
re-fitting.

Standardisation is per-dimension and computed on the training split only. The 44 state
dims span joint angles, gripper widths and waist positions with wildly different scales;
unstandardised, the loss is dominated by whichever dims happen to be largest and the
model simply ignores the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Normaliser:
    """Per-dimension standardisation, fitted on train and applied everywhere."""

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, floor: float = 1e-3) -> "Normaliser":
        # A dimension that never varies has std 0; dividing by it produces inf and one
        # dead channel poisons the whole loss. The floor keeps such dims at ~0 instead.
        return Normaliser(x.mean(0), np.maximum(x.std(0), floor))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def invert(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def mask_observables(path: Path) -> np.ndarray | None:
    """(T, 3) normalised centroid-x, centroid-y and sqrt(area) per frame.

    sqrt(area) rather than area so the channel is a LENGTH, comparable in scale to the
    centroid channels. Feeding an area alongside two positions gives one channel a
    squared dynamic range and it dominates the loss.

    Frames where the mask is empty inherit the previous frame's value: the object did
    not teleport, it was occluded, and interpolating a disappearance as motion to the
    origin would teach the model exactly the wrong thing.
    """
    if not path.exists():
        return None
    m = np.load(path)["arr_0"]
    T, H, W = m.shape
    out = np.zeros((T, 3), np.float32)
    last = np.array([0.5, 0.5, 0.0], np.float32)
    for t in range(T):
        ys, xs = np.nonzero(m[t])
        if len(xs):
            last = np.array([xs.mean() / W, ys.mean() / H,
                             np.sqrt(len(xs) / (H * W))], np.float32)
        out[t] = last
    return out


def load_episodes(root: str | Path, *, with_masks: bool = True
                  ) -> list[dict[str, np.ndarray]]:
    """Every episode in a LeRobot dataset as arrays of state, action and slot."""
    import pandas as pd

    root = Path(root)
    episodes = []
    for pq in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        ep = {
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "action": np.stack(df["action"].to_numpy()).astype(np.float32),
        }
        if with_masks:
            slots = []
            for cam in sorted((root / "masks").rglob(f"{pq.stem}_masks.npz")):
                o = mask_observables(cam)
                if o is not None:
                    slots.append(o)
            if slots:
                n = min(len(ep["state"]), min(len(s) for s in slots))
                ep = {k: v[:n] for k, v in ep.items()}
                # (T, M, 3) — one slot per camera view.
                ep["slots"] = np.stack([s[:n] for s in slots], axis=1)
        episodes.append(ep)
    return episodes


class TransitionDataset:
    """Single-step transitions, kept inside episode boundaries.

    Crossing an episode boundary would pair the end of one demonstration with the start
    of the next and teach the model a discontinuity that never happens. The split is by
    EPISODE, not by frame, for the same reason a random frame split would leak: adjacent
    frames are nearly identical, so a frame-wise holdout measures interpolation and
    reports it as generalisation.
    """

    def __init__(self, episodes: list[dict[str, np.ndarray]], *,
                 state_norm: Normaliser | None = None,
                 action_norm: Normaliser | None = None,
                 slot_norm: Normaliser | None = None,
                 velocity: bool = True) -> None:
        self.eps = [e for e in episodes if len(e["state"]) >= 2]
        if not self.eps:
            raise ValueError("no episode has two frames; nothing to predict")
        self.has_slots = all("slots" in e for e in self.eps)
        cat = lambda k: np.concatenate([e[k] for e in self.eps])  # noqa: E731
        self.state_norm = state_norm or Normaliser.fit(cat("state"))
        self.action_norm = action_norm or Normaliser.fit(cat("action"))
        if self.has_slots:
            flat = cat("slots").reshape(-1, self.eps[0]["slots"].shape[-1])
            self.slot_norm = slot_norm or Normaliser.fit(flat)
        else:
            self.slot_norm = None
        self.velocity = velocity
        # Transitions start at t=1 when velocity is carried, because t=0 has no
        # predecessor to difference against. Faking it with a zero velocity would teach
        # the model that every episode begins at rest, which is a property of the
        # indexing rather than of the robot.
        first = 1 if velocity else 0
        self.index = [(i, t) for i, e in enumerate(self.eps)
                      for t in range(first, len(e["state"]) - 1)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        """One transition. With `velocity`, the latent is [position, delta].

        Measured on gr1.PickNPlace: without velocity in the latent the model cannot
        represent constant-velocity motion at all, and a constant-velocity baseline beat
        it more than two to one on held-out data. Carrying the previous delta is what
        lets the dynamics model start from that baseline and improve on it, rather than
        having to rediscover momentum from position alone.
        """
        e, t = self.index[i]
        ep = self.eps[e]
        s_t, s_n = self.state_norm(ep["state"][t]), self.state_norm(ep["state"][t + 1])
        out = {"action": self.action_norm(ep["action"][t])}
        if self.velocity:
            s_p = self.state_norm(ep["state"][t - 1])
            out["robot"] = np.concatenate([s_t, s_t - s_p])
            out["robot_next"] = np.concatenate([s_n, s_n - s_t])
        else:
            out["robot"], out["robot_next"] = s_t, s_n
        if self.has_slots:
            g_t = self.slot_norm(ep["slots"][t])
            g_n = self.slot_norm(ep["slots"][t + 1])
            if self.velocity:
                g_p = self.slot_norm(ep["slots"][t - 1])
                out["slots"] = np.concatenate([g_t, g_t - g_p], axis=-1)
                out["slots_next"] = np.concatenate([g_n, g_n - g_t], axis=-1)
            else:
                out["slots"], out["slots_next"] = g_t, g_n
        return out

    @property
    def robot_dim(self) -> int:
        d = int(self.state_norm.mean.shape[0])
        return 2 * d if self.velocity else d

    @property
    def slot_dim(self) -> int:
        d = self.eps[0]["slots"].shape[-1] if self.has_slots else 0
        return 2 * d if (self.has_slots and self.velocity) else d

    @property
    def n_slots(self) -> int:
        return self.eps[0]["slots"].shape[1] if self.has_slots else 0


def split_episodes(episodes: list, holdout: int = 1) -> tuple[list, list]:
    """Last `holdout` episodes are the test set. By episode, never by frame."""
    if len(episodes) <= holdout:
        raise ValueError(
            f"{len(episodes)} episodes cannot give a {holdout}-episode holdout")
    return episodes[:-holdout], episodes[-holdout:]

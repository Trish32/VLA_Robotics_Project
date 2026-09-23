"""The latent a world model predicts forward: object slots plus robot state.

`z` is deliberately OBJECT-CENTRIC rather than a single pixel-space vector. The whole
stack upstream produces objects — instances with poses, extents and labels — and a
latent that dissolves them back into one undifferentiated blob throws that away exactly
where it starts being useful. Keeping slots separate buys three things a flat latent
cannot give:

  * the scorer can ask object-level questions (does THIS object collide, is THIS one
    unsupported) instead of scoring an opaque vector;
  * the dynamics model can carry the prior that **most objects do not move**, which is
    the single strongest regularity in manipulation and is hard to express otherwise;
  * a prediction is inspectable — you can say which object the model thinks will move
    and by how much, which is what makes a wrong rollout debuggable.

Slot layout is fixed and public (`SLOT_*`) because the scorer reads geometry straight
out of it. A learned-but-opaque slot encoding would make collision and stability terms
impossible to write honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Slot channels. The first six are metric geometry in the anchor frame; the rest are a
# learned embedding carrying identity/appearance. Geometry stays raw and interpretable.
SLOT_CENTRE = slice(0, 3)
SLOT_EXTENT = slice(3, 6)
SLOT_GEOM = slice(0, 6)
SLOT_EMBED_START = 6


@dataclass
class SceneLatent:
    """One batch of scene latents.

    slots:   (B, M, D) per-object state; channels 0:6 are centre+extent in metres.
    robot:   (B, R) proprioceptive state.
    mask:    (B, M) bool, True where a slot holds a real object. Scenes have different
             object counts and padding must not be scored as if it were a real object
             sitting at the origin.
    """

    slots: torch.Tensor
    robot: torch.Tensor
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.slots.dim() != 3:
            raise ValueError(f"slots must be (B, M, D), got {tuple(self.slots.shape)}")
        if self.robot.dim() != 2:
            raise ValueError(f"robot must be (B, R), got {tuple(self.robot.shape)}")
        if self.slots.shape[0] != self.robot.shape[0]:
            raise ValueError(
                f"batch mismatch: {self.slots.shape[0]} slot rows vs "
                f"{self.robot.shape[0]} robot rows"
            )
        if self.mask is None:
            self.mask = torch.ones(self.slots.shape[:2], dtype=torch.bool,
                                   device=self.slots.device)
        elif tuple(self.mask.shape) != tuple(self.slots.shape[:2]):
            raise ValueError(
                f"mask {tuple(self.mask.shape)} does not match slots "
                f"{tuple(self.slots.shape[:2])}"
            )

    # ------------------------------------------------------------------ shapes
    @property
    def batch(self) -> int:
        return self.slots.shape[0]

    @property
    def n_slots(self) -> int:
        return self.slots.shape[1]

    @property
    def slot_dim(self) -> int:
        return self.slots.shape[2]

    @property
    def robot_dim(self) -> int:
        return self.robot.shape[1]

    # ------------------------------------------------------------------ geometry
    @property
    def centre(self) -> torch.Tensor:
        return self.slots[..., SLOT_CENTRE]

    @property
    def extent(self) -> torch.Tensor:
        """Non-negative by construction: a predicted negative side length is
        meaningless, and an unclamped one silently inverts every overlap test."""
        return self.slots[..., SLOT_EXTENT].abs()

    def aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.extent / 2.0
        return self.centre - half, self.centre + half

    # ------------------------------------------------------------------ plumbing
    def to(self, device) -> "SceneLatent":
        return SceneLatent(self.slots.to(device), self.robot.to(device),
                           self.mask.to(device))

    def detach(self) -> "SceneLatent":
        return SceneLatent(self.slots.detach(), self.robot.detach(), self.mask)

    def clone(self) -> "SceneLatent":
        return SceneLatent(self.slots.clone(), self.robot.clone(), self.mask.clone())

    def expand_batch(self, k: int) -> "SceneLatent":
        """Repeat each row `k` times — one copy per candidate action.

        Interleaved (`repeat_interleave`, not `repeat`) so row `b*k + i` is candidate `i`
        of scene `b`. The scorer and the planner both index on that convention.
        """
        return SceneLatent(self.slots.repeat_interleave(k, 0),
                           self.robot.repeat_interleave(k, 0),
                           self.mask.repeat_interleave(k, 0))

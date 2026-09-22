"""One time base, one frame-naming authority, one stable instance identity.

An audit of this repo found three representations of time (float seconds in 15 files,
integer nanoseconds in 12, ROS `sec`+`nanosec` in 5), TF frame names hardcoded in five
places and minted in four others, and instance IDs assigned by SCORE RANK. Each is
survivable alone. Together they make a unified timeline impossible and object identity
meaningless across runs.

TIME — nanoseconds are canonical, and float seconds cannot be
-------------------------------------------------------------
This is not a style preference. A Unix timestamp is ~1.3e9 seconds, which is ~1.3e18
nanoseconds. float64 carries 53 bits of mantissa, about 9.0e15 integers exactly — three
orders of magnitude short. **Nanosecond precision at epoch scale is not representable in
float64 seconds**, so every ns -> float seconds -> ns round trip silently quantises to
roughly 200 ns, and comparisons that should be exact start drifting.

TUM's own files are float seconds, so ingestion converts once, at the boundary, and
`Stamp.from_seconds` records that it is lossy. Everything internal stays integral.

FRAMES — one authority, because a typo is a silent disconnection
---------------------------------------------------------------
`tf2` does not validate frame names. A node publishing `camera_link` while another looks
up `camera_color_optical_frame` does not error; the lookup simply never resolves, and
the symptom is "no transform available" far from the typo. `Frames` mints every name
used in this stack so the strings exist in exactly one file.

It also encodes the optical-frame distinction that is easy to get wrong: SLAM poses are
in the OpenCV optical convention (x right, y down, z forward), while REP-103 body frames
are x forward, y left, z up. Publishing an optical-convention pose on a `_link` frame is
a 90-degree error that every individual transform still reports as well-formed.

IDENTITY — rank-ordered IDs are not identity
--------------------------------------------
`inst_0` currently means "the highest-scoring proposal in this run". Re-run segmentation
and the same physical object is `inst_2`, while `inst_0` is something else. Anything
holding a reference — a TF frame, a scene-graph node, a planner's target — is now
pointing at a different object with no error anywhere.

`InstanceRegistry` assigns identity by DATA ASSOCIATION instead: a new observation is
matched to an existing instance by spatial overlap and label agreement, and only mints a
new ID when nothing matches. IDs are then stable across runs and across time, which is
what makes `object/<id>` a meaningful TF frame and `upsert` a meaningful operation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

NS_PER_S = 1_000_000_000

# float64 holds integers exactly only up to 2**53. Nanoseconds since the epoch are far
# past that, which is why float seconds are a lossy carrier rather than a lossless one.
FLOAT64_EXACT_INT = 2 ** 53


class TimeError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Stamp:
    """An instant, as integer nanoseconds. The only time type this pipeline stores."""

    ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.ns, (int, np.integer)):
            raise TimeError(
                f"Stamp takes integer nanoseconds, got {type(self.ns).__name__}. "
                "Use Stamp.from_seconds() at an ingestion boundary; it is lossy and "
                "says so."
            )
        object.__setattr__(self, "ns", int(self.ns))

    # ---------------------------------------------------------------- ingestion

    @classmethod
    def from_seconds(cls, seconds: float) -> "Stamp":
        """Float seconds -> Stamp. LOSSY at epoch scale; use only at a boundary.

        A dataset that stores float seconds (TUM, and most ROS bags exported to CSV)
        cannot express nanoseconds at epoch scale — see the module docstring. Converting
        once on ingestion is fine; converting back and forth is not, so there is
        deliberately no `to_seconds` round-trip guarantee.
        """
        return cls(int(round(float(seconds) * NS_PER_S)))

    @classmethod
    def from_ros(cls, msg_stamp) -> "Stamp":
        """builtin_interfaces/Time -> Stamp. Exact: ROS already stores sec + nanosec."""
        return cls(int(msg_stamp.sec) * NS_PER_S + int(msg_stamp.nanosec))

    # ------------------------------------------------------------------- export

    def to_ros(self, message_type=None):
        """Stamp -> builtin_interfaces/Time. Exact."""
        if message_type is None:
            from builtin_interfaces.msg import Time as message_type  # noqa: N813
        return message_type(sec=self.ns // NS_PER_S, nanosec=self.ns % NS_PER_S)

    def to_seconds(self) -> float:
        """For libraries that demand float seconds. Lossy; never feed the result back."""
        return self.ns / NS_PER_S

    # --------------------------------------------------------------- arithmetic

    def age_s(self, now: "Stamp") -> float:
        return (now.ns - self.ns) / NS_PER_S

    def __sub__(self, other: "Stamp") -> float:
        return (self.ns - other.ns) / NS_PER_S


def assert_lossless(ns: int) -> int:
    """Raise if this value would not survive a trip through float seconds.

    Used at boundaries that are about to hand a timestamp to something float-based, so
    the precision loss is refused rather than absorbed.
    """
    if abs(ns) > FLOAT64_EXACT_INT:
        raise TimeError(
            f"{ns} ns exceeds float64's exact-integer range ({FLOAT64_EXACT_INT}); "
            "converting to float seconds would quantise it. Keep it integral, or "
            "subtract a session epoch first."
        )
    return ns


class Frames:
    """The only place TF frame names are constructed.

    tf2 never validates a frame name, so a mismatched string is a silent disconnection
    rather than an error. Centralising the spellings makes a typo a Python error.
    """

    WORLD = "map"
    ODOM = "odom"
    BASE = "base_link"

    # OpenCV optical convention: x right, y down, z forward. SLAM poses live HERE.
    CAMERA_OPTICAL = "camera_color_optical_frame"
    # REP-103 body convention: x forward, y left, z up. The camera driver publishes the
    # static transform between the two; a SLAM pose published directly on this frame is
    # rotated 90 degrees and every individual transform still looks well-formed.
    CAMERA_BODY = "camera_link"

    @staticmethod
    def keyframe(index: int) -> str:
        """Anchor frame for observations made from a given keyframe.

        Objects anchor here rather than to `map` so that when the SLAM backend's bundle
        adjustment moves a keyframe, TF recomposes and every object anchored to it
        follows the correction for free.
        """
        return f"keyframe/{int(index)}"

    @staticmethod
    def instance(instance_id: str) -> str:
        return f"object/{instance_id}"

    @staticmethod
    def pregrasp(instance_id: str) -> str:
        return f"object/{instance_id}/pregrasp"


@dataclass
class TrackedInstance:
    instance_id: str
    label: str
    centre: np.ndarray
    extent: np.ndarray
    stamp: Stamp
    observations: int = 1

    def iou(self, centre: np.ndarray, extent: np.ndarray) -> float:
        """Axis-aligned 3-D IoU against a candidate box."""
        a_lo, a_hi = self.centre - self.extent / 2, self.centre + self.extent / 2
        b_lo, b_hi = centre - extent / 2, centre + extent / 2
        overlap = np.maximum(0.0, np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo))
        inter = float(np.prod(overlap))
        union = float(np.prod(self.extent) + np.prod(extent)) - inter
        return inter / union if union > 1e-12 else 0.0


@dataclass
class InstanceRegistry:
    """Stable instance identity by data association.

    Two observations are the same object if their boxes overlap enough AND their labels
    agree. Both conditions matter: overlap alone merges a cup sitting on a book, and
    label alone merges every chair in the room.
    """

    iou_threshold: float = 0.3
    require_label_match: bool = True
    tracked: dict[str, TrackedInstance] = field(default_factory=dict)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def _mint(self, label: str) -> str:
        """`pear_1`, not `inst_0`: readable, and carries no rank information.

        The number is a monotonic counter, never a score position, so it cannot change
        meaning when segmentation is re-run.
        """
        safe = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
        return f"{safe or 'object'}_{next(self._counter)}"

    def associate(self, centre, extent, label: str, stamp: Stamp) -> tuple[str, bool]:
        """Return (stable id, is_new) for one observation."""
        centre = np.asarray(centre, dtype=float).reshape(3)
        extent = np.asarray(extent, dtype=float).reshape(3)

        best_id, best_iou = None, 0.0
        for instance_id, tracked in self.tracked.items():
            if self.require_label_match and tracked.label != label:
                continue
            score = tracked.iou(centre, extent)
            if score > best_iou:
                best_id, best_iou = instance_id, score

        if best_id is not None and best_iou >= self.iou_threshold:
            existing = self.tracked[best_id]
            if stamp.ns < existing.stamp.ns:
                # A late observation must not roll an instance backwards; stages run on
                # their own threads and replies arrive out of order.
                return best_id, False
            existing.centre, existing.extent, existing.stamp = centre, extent, stamp
            existing.observations += 1
            return best_id, False

        instance_id = self._mint(label)
        self.tracked[instance_id] = TrackedInstance(
            instance_id=instance_id, label=label, centre=centre, extent=extent,
            stamp=stamp,
        )
        return instance_id, True

    def associate_batch(self, observations) -> list[str]:
        """Associate a whole frame's worth, largest first.

        Order matters: a big instance claims its identity before a small one that
        overlaps it can, which keeps a cup from stealing the table's ID.
        """
        indexed = sorted(
            enumerate(observations),
            key=lambda item: -float(np.prod(np.asarray(item[1][1]))),
        )
        out: list[str | None] = [None] * len(observations)
        for index, (centre, extent, label, stamp) in indexed:
            out[index], _ = self.associate(centre, extent, label, stamp)
        return out  # type: ignore[return-value]

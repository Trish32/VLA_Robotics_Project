"""Stage outputs -> world-model observations. The camera-frame to world-frame step.

OpenMask3D and FoundationPose produce different halves of the same fact, in different
frames, at different rates:

    OpenMask3D      WHICH instance, WHAT it is, and roughly how big.
                    Masks index the fused cloud, so its geometry is ALREADY in the
                    anchor frame — no transform, just an extent and a label.
    FoundationPose  WHERE it is, precisely, as T_cam_obj — in the CAMERA frame, which
                    moves. This is the half that has to be transformed.

So the two compose rather than compete: OpenMask3D creates the node, FoundationPose
refines its pose. `refine_with_pose` keeps the id, label and extent and replaces only
what FoundationPose actually measured.

    T_anchor_obj = T_anchor_cam @ T_cam_obj

THE TIMESTAMP IS PART OF THE TRANSFORM
--------------------------------------
That composition is only valid if `T_anchor_cam` is the camera pose *at the moment the
detection was made*. Using a camera pose from a different instant displaces the object by
exactly how far the camera travelled in between — at 30 Hz and walking pace that is a
couple of centimetres per frame, which is enough to miss a grasp and small enough to look
like calibration error. Nothing downstream can detect it, so `refine_with_pose` refuses a
mismatch rather than silently composing across time.

Everything here is rclpy-free so it can be tested without a container, matching how
`obs_encode.py` is kept free of ROS for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NS_PER_MS = 1_000_000


class ObservationError(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    """One reported fact about one instance, ready for the world model."""

    node_id: str
    label: str
    pose: np.ndarray            # (4, 4) in anchor_frame
    extent: np.ndarray          # (3,) axis-aligned metres
    anchor_frame: str
    stamp_ns: int
    score: float = 1.0
    # Indices (into the cloud this observation was built from) of the points that
    # survived outlier rejection. Carried so every downstream consumer — hull, sample
    # points, the published box — describes the SAME set of points the extent came from.
    kept_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", np.asarray(self.pose, float).reshape(4, 4))
        object.__setattr__(self, "extent", np.asarray(self.extent, float).reshape(3))
        _assert_rigid(self.pose, self.node_id)


def _assert_rigid(T: np.ndarray, who: str) -> None:
    if not np.isfinite(T).all():
        raise ObservationError(f"{who}: pose contains non-finite values")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ObservationError(
            f"{who}: bottom row is {T[3]}, not [0,0,0,1] — not a rigid transform, or "
            "it is transposed"
        )
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ObservationError(
            f"{who}: rotation block is not orthonormal. A scaled or transposed rotation "
            "places the object plausibly and wrongly."
        )


def from_openmask3d(
    mask_point_indices,
    points: np.ndarray,
    labels: list[str],
    scores: list[float],
    *,
    anchor_frame: str,
    stamp_ns: int,
    registry=None,
    reject_outliers_: bool = True,
) -> list[Observation]:
    """Instance masks over the fused cloud -> observations, axis-aligned.

    No transform: masks index the cloud that `scene_export.fuse_tsdf` already built in
    the anchor frame. The pose here is position-only (identity rotation) because a
    segmentation mask carries no orientation — claiming one would invent precision
    FoundationPose is the stage that actually supplies.

    IDs come from `InstanceRegistry`, not from enumeration order. The previous
    `f"{prefix}_{index}"` numbered proposals by SCORE RANK, so re-running segmentation
    gave the same physical object a different name and every TF frame, graph node and
    planner target silently pointed elsewhere. Association also merges duplicate
    proposals of one object: on the TUM run it collapsed two desk proposals overlapping
    at IoU 0.498 into a single instance.
    """
    from pipeline.identity import InstanceRegistry, Stamp

    registry = registry if registry is not None else InstanceRegistry()
    stamp = Stamp(int(stamp_ns))
    points = np.asarray(points, float)
    if not (len(mask_point_indices) == len(labels) == len(scores)):
        raise ObservationError(
            f"{len(mask_point_indices)} masks, {len(labels)} labels, {len(scores)} "
            "scores — these are parallel arrays and must agree"
        )

    out: list[Observation] = []
    for index, (indices, label, score) in enumerate(zip(mask_point_indices, labels, scores)):
        indices = np.asarray(indices)
        if indices.size == 0:
            # An empty mask has no centroid. Emitting one would put an object at the
            # origin with a zero extent, which reads as a real detection at the robot's
            # own base.
            raise ObservationError(f"mask {index} ({label!r}) is empty")

        # Reject outliers BEFORE the extent is taken, because the extent is a maximum
        # and a single flier moves it. See `reject_outliers`.
        kept = indices[reject_outliers(points[indices])] if reject_outliers_ else indices
        xyz = points[kept]
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        centre, extent = (lo + hi) / 2.0, hi - lo
        pose = np.eye(4)
        pose[:3, 3] = centre
        instance_id, _ = registry.associate(centre, extent, label, stamp)
        out.append(Observation(
            node_id=instance_id, label=label, pose=pose, extent=extent,
            anchor_frame=anchor_frame, stamp_ns=stamp_ns, score=float(score),
            kept_indices=kept,
        ))
    return out


def refine_with_pose(
    observation: Observation,
    T_cam_obj: np.ndarray,
    T_anchor_cam: np.ndarray,
    *,
    stamp_ns: int,
    camera_stamp_ns: int,
    max_dt_ns: int = 20 * NS_PER_MS,
) -> Observation:
    """FoundationPose's camera-frame pose, placed in the world.

    Keeps the id, label and extent that OpenMask3D established — FoundationPose measures
    pose, not identity — and replaces the position-only pose with a full 6-DoF one.

    `camera_stamp_ns` is the stamp of the camera pose, NOT of the detection. They must
    agree to within `max_dt_ns`; see the module docstring for why composing across time
    is undetectable downstream.
    """
    T_cam_obj = np.asarray(T_cam_obj, float).reshape(4, 4)
    T_anchor_cam = np.asarray(T_anchor_cam, float).reshape(4, 4)
    _assert_rigid(T_cam_obj, f"{observation.node_id} T_cam_obj")
    _assert_rigid(T_anchor_cam, f"{observation.node_id} T_anchor_cam")

    dt = abs(stamp_ns - camera_stamp_ns)
    if dt > max_dt_ns:
        raise ObservationError(
            f"{observation.node_id}: detection and camera pose are {dt / NS_PER_MS:.1f} "
            f"ms apart (limit {max_dt_ns / NS_PER_MS:.1f} ms). Composing them would "
            "displace the object by however far the camera moved in between — a couple "
            "of centimetres at 30 Hz, enough to miss a grasp and small enough to look "
            "like calibration error."
        )

    return Observation(
        node_id=observation.node_id,
        label=observation.label,
        pose=T_anchor_cam @ T_cam_obj,
        extent=observation.extent,
        anchor_frame=observation.anchor_frame,
        stamp_ns=stamp_ns,
        score=observation.score,
    )


def reject_outliers(points: np.ndarray, *, k: int = 20, std_ratio: float = 2.0
                    ) -> np.ndarray:
    """Indices of the points whose local neighbourhood is not anomalously sparse.

    An axis-aligned extent is a MAXIMUM over points, so it is decided by the single
    furthest one in each direction — the statistic most sensitive to a stray. On the TUM
    run 13 stray points out of 1,142 added 40 cm to a chair's height (0.99 -> 0.59 m
    after rejection), and that extent is what sets the published box and the pre-grasp
    standoff. Rejecting 0.4-3.6% of points shrank instance volumes to 0.52-0.89x.

    Density-based rather than a percentile trim: a percentile discards a fixed fraction
    whether or not anything is wrong, which eats real geometry on a clean instance. This
    drops only points that are far from their own neighbours.

    Returns all indices unchanged when scipy is unavailable or the set is too small to
    have a meaningful neighbourhood.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) <= k:
        return np.arange(len(pts))
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.arange(len(pts))
    # k + 1 because the nearest neighbour of a point is itself.
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    cutoff = mean_d.mean() + std_ratio * mean_d.std()
    keep = np.flatnonzero(mean_d <= cutoff)
    # Never return an empty or near-empty set: a degenerate instance should keep its
    # (bad) extent rather than collapse to a point at some arbitrary location.
    return keep if len(keep) >= max(4, 0.5 * len(pts)) else np.arange(len(pts))


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """(hull vertices, outward half-space planes) for a point set.

    Returns (None, None) rather than raising when a hull is not defined — fewer than
    four points, or points that are coplanar/collinear, which happens for a thin or
    barely-observed instance. Callers treat a missing hull as "no evidence" and fall
    back to the bounding box, so a degenerate instance loses precision but never
    silently gains a wrong answer.
    """
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:          # scipy is optional; the box path still works
        return None, None
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 4:
        return None, None
    try:
        h = ConvexHull(pts)
    except (QhullError, ValueError):
        return None, None
    return pts[h.vertices], h.equations


def downsample(points: np.ndarray, cap: int = 200, seed: int = 0) -> np.ndarray:
    """An evenly-spread subset of at most `cap` points, for the `near` test.

    Voxel-first so the sample follows the shape rather than the density: a face seen
    from close up contributes thousands of points and would otherwise dominate a uniform
    random draw, pulling the apparent surface toward whatever the camera happened to
    dwell on.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) <= cap:
        return pts
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    # Aim for roughly `cap` occupied voxels: cap^(1/3) cells along each axis.
    n = max(1, int(round(cap ** (1 / 3))))
    key = np.floor((pts - lo) / span * n).astype(np.int64).clip(0, n - 1)
    flat = (key[:, 0] * n + key[:, 1]) * n + key[:, 2]
    _, first = np.unique(flat, return_index=True)
    out = pts[np.sort(first)]
    if len(out) > cap:
        out = out[np.random.default_rng(seed).choice(len(out), cap, replace=False)]
    return out


def point_sets_from(observations: list[Observation],
                    points: np.ndarray) -> dict[str, np.ndarray]:
    """node_id -> that instance's points, ready for `to_scene_nodes(point_sets=...)`.

    Uses each observation's `kept_indices`, so the hull and the sample points describe
    the same points the extent was taken from. Several observations can share a node_id
    — association merges duplicate proposals of one object — so their points are unioned
    rather than one silently winning.
    """
    pts = np.asarray(points, float)
    out: dict[str, np.ndarray] = {}
    for o in observations:
        if o.kept_indices is None:
            continue
        idx = np.asarray(o.kept_indices)
        prev = out.get(o.node_id)
        out[o.node_id] = pts[idx] if prev is None else np.vstack([prev, pts[idx]])
    return out


def to_scene_nodes(observations: list[Observation], *,
                   point_sets: dict[str, np.ndarray] | None = None,
                   surfaces=frozenset(
                       {"table", "floor", "shelf", "counter", "desk", "wall"})):
    """Observations -> SceneNode list, ready for `SceneGraph.upsert`.

    `point_sets` maps node_id -> that instance's points. Supplying it attaches a convex
    hull to each node, which is what lets `inside` be decided on the geometry instead of
    on nesting bounding boxes. Omitting it is supported and falls back to the box.
    """
    from pipeline.identity import Frames
    from pipeline.scene_graph import NodeKind, SceneNode

    nodes = []
    for o in observations:
        hv = hp = sp = None
        if point_sets is not None and o.node_id in point_sets:
            hv, hp = convex_hull(point_sets[o.node_id])
            sp = downsample(point_sets[o.node_id])
        nodes.append(SceneNode(
            node_id=o.node_id,
            kind=NodeKind.SURFACE if o.label in surfaces else NodeKind.OBJECT,
            label=o.label,
            tf_frame=Frames.instance(o.node_id),
            anchor_frame=o.anchor_frame,
            pose=o.pose,
            extent=o.extent,
            stamp_ns=o.stamp_ns,
            clip_similarity=o.score,
            hull_vertices=hv,
            hull_planes=hp,
            sample_points=sp,
        ))
    return nodes

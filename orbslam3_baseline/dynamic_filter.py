"""Reject dynamic feature points: semantic prior first, epipolar geometry deciding.

ORB-SLAM3 assumes a static world. A person walking through the frame contributes
correspondences that are perfectly good matches and completely wrong constraints, and
the pose optimiser has no way to tell — it just drags the camera along with them.

The fix here is the DS-SLAM / DynaSLAM pattern: a lightweight detector marks regions that
*could* move, and multi-view geometry decides which ones actually did. Neither half works
alone, and the reasons are asymmetric:

  * **Semantics alone over-rejects.** A parked car and a seated person are in "dynamic"
    classes and are excellent static structure. Deleting them throws away exactly the
    well-textured regions ORB likes, and in a street scene can remove most of the frame.
  * **Geometry alone under-rejects, and is fragile.** The epipolar test needs a
    fundamental matrix, and F is estimated from the same correspondences being tested.
    A large moving object biases F toward its own motion, after which the object looks
    consistent and the static background looks like the outlier. This is the failure
    that makes naive geometric filtering worse than none.

So the semantic prior's real job is not to label dynamic points. It is to **hold
suspicious correspondences out of the F estimate**, so the geometry is fitted on a set
that is static by assumption and can then judge the rest. That is what "semantic prior
guidance" buys, and it is why the order matters.

MEASURED, on the synthetic two-view scenes in tests/test_dynamic_filter.py:

    moving fraction   F fitted on ALL     F fitted on non-box points
                      (object rejected)   (object rejected)
        21%                  0%                  100%
        46%                  0%                  100%
        68%                  0%                  100%

Geometry alone is not weaker here, it is inert — and it stays inert even when the mover
is a fifth of the correspondences and the static set is a clear majority. The mechanism
is that a COMPACT moving cluster is cheap for a robust estimator to absorb: tilting F
enough to make those points inliers costs only a few percent of the spread-out
background, so MAGSAC takes that trade every time. The object then satisfies the very
constraint meant to expose it. Holding the box out of the fit is the whole difference.

The cost of the design is ~9% of good background rejected, because F comes from a subset
and borderline correspondences fall the wrong side of the threshold.

THE EPIPOLAR TEST HAS A BLIND DIRECTION, and depth closes it. The constraint only sees
motion that takes a point OFF its epipolar line; an object translating parallel to the
camera baseline slides ALONG that line and stays consistent — measured at 0.49 px versus
5.6 px for the same displacement perpendicular to the baseline. A person walking the way
the camera is panning is invisible to it.

So when depth is supplied (`prev_depth`, `curr_depth`, `K`) a second test runs: fit a
rigid transform to the trusted 3-D correspondences by Kabsch/RANSAC, then measure each
point's residual in metres. That has no blind direction, because a point that moved
genuinely is somewhere else whatever direction it went. On the degenerate scene above it
turns a 0.49 px non-signal into a ~0.22 m residual and takes rejection from under 20% to
over 95%.

The two tests are combined with OR, not AND: they are blind to different things, so
failing either means the point did not move with the scene. On RGB-D input there is no
reason to run only the 2-D one — the extra cost is a single Kabsch fit.

The decision is object-level as well as point-level. Geometry alone yields a scatter of
flagged points across a moving person; the detector's box is what turns that scatter into
"this object is moving, drop all of it" — including the points on it that happened to
match consistently, which are usually mismatches rather than genuinely static structure.

WHAT THIS MODULE IS NOT: it does not run a detector. Boxes come in as an argument, so
the geometry is testable with no weights, no ONNX runtime and no GPU. `yolo_detector.py`
is the thin layer that produces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO classes that can move under their own power or be carried. Everything else is
# treated as structure. This is a prior about the WORLD, not about the scene: it says
# "a chair could be moved", not "this chair is moving".
MOVABLE_COCO = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "chair", "laptop", "mouse",
    "cell phone", "book",
})


class DegenerateMotion(RuntimeError):
    """Raised when the epipolar constraint carries no information about this frame pair."""


@dataclass(frozen=True)
class Detection:
    """One detector box in pixel coordinates, xyxy."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float = 1.0

    def contains(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N,) bool."""
        return (
            (points[:, 0] >= self.x1) & (points[:, 0] <= self.x2)
            & (points[:, 1] >= self.y1) & (points[:, 1] <= self.y2)
        )


@dataclass(frozen=True)
class FilterConfig:
    # Sampson distance, in pixels, beyond which a correspondence is inconsistent with the
    # estimated epipolar geometry. ORB keypoints on a 640x480 frame localise to roughly a
    # pixel, so ~1.5 px separates real motion from matching noise without being brittle.
    epipolar_px: float = 1.5
    # RANSAC inlier threshold when fitting F. Looser than the test threshold on purpose:
    # F should be fitted tolerantly and judged strictly.
    ransac_px: float = 2.5
    # Below this many trusted correspondences, F is not identifiable and the frame is
    # reported degenerate rather than filtered on a meaningless matrix.
    min_trusted: int = 30
    # If this fraction of a box's correspondences are inconsistent, the whole object is
    # moving and every point on it goes, including the ones that matched consistently.
    box_cull_fraction: float = 0.35
    # Median displacement below which the camera has not moved enough to triangulate.
    # Under pure rotation or a static camera EVERY correspondence satisfies the epipolar
    # constraint, so the test silently passes everything and filters nothing.
    min_parallax_px: float = 1.0
    # 3-D rigid residual, metres, beyond which a correspondence did not move with the
    # scene. Used only when depth is supplied. Unlike the epipolar test this has no blind
    # direction: motion along the baseline still changes the 3-D position.
    depth_residual_m: float = 0.06
    depth_ransac_m: float = 0.10
    movable: frozenset = field(default=MOVABLE_COCO)


@dataclass
class FilterResult:
    keep: np.ndarray            # (N,) bool — feed these to the pose optimiser
    sampson: np.ndarray         # (N,) float — epipolar residual, px
    suspect: np.ndarray         # (N,) bool — inside a movable-class box
    culled_boxes: list[int]     # indices of detections judged to be moving
    fundamental: np.ndarray | None
    n_trusted: int
    rigid_residual: np.ndarray | None = None   # metres; NaN where depth was unusable

    @property
    def n_culled(self) -> int:
        return int((~self.keep).sum())


def sampson_distance(F: np.ndarray, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """First-order geometric error of x2^T F x1 = 0, in pixels.

    Sampson rather than the raw algebraic residual or a point-to-line distance in one
    image: the algebraic value scales with image coordinates and is not comparable across
    the frame, and a one-sided line distance ignores the uncertainty of the other view.
    Sampson normalises by the gradient, so a single pixel threshold means the same thing
    everywhere.
    """
    ones = np.ones((len(prev), 1))
    x1 = np.hstack([prev, ones])
    x2 = np.hstack([curr, ones])

    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    numerator = np.einsum("ij,ij->i", x2, Fx1) ** 2
    denominator = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    # A correspondence whose epipolar lines are degenerate carries no constraint; calling
    # it consistent (0) would silently keep a point the geometry cannot vouch for.
    safe = denominator > 1e-12
    out = np.full(len(prev), np.inf)
    out[safe] = np.sqrt(numerator[safe] / denominator[safe])
    return out


def unproject(points_px: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 2) pixels + (N,) metres -> (N, 3) camera-frame points."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = np.asarray(depth_m, dtype=np.float64)
    x = (points_px[:, 0] - cx) * z / fx
    y = (points_px[:, 1] - cy) * z / fy
    return np.column_stack([x, y, z])


def _kabsch(A: np.ndarray, B: np.ndarray):
    """Rigid transform taking A onto B, least squares. Returns (R, t)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    # Without this reflection guard SVD can return an improper rotation (det = -1),
    # which fits mirrored data beautifully and is not a motion any camera can make.
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def rigid_residuals(R: np.ndarray, t: np.ndarray, X1: np.ndarray, X2: np.ndarray):
    """||(R X1 + t) - X2|| per correspondence, in metres."""
    return np.linalg.norm(X1 @ R.T + t - X2, axis=1)


def estimate_rigid_ransac(X1: np.ndarray, X2: np.ndarray, threshold_m: float,
                          iters: int = 200, seed: int = 0):
    """RANSAC over 3-point samples. Returns (R, t, inliers) or (None, None, None)."""
    n = len(X1)
    if n < 3:
        return None, None, None
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        # Collinear samples give a rotation that is unconstrained about their axis.
        span = np.linalg.matrix_rank(X1[idx] - X1[idx].mean(axis=0), tol=1e-6)
        if span < 2:
            continue
        R, t = _kabsch(X1[idx], X2[idx])
        inliers = rigid_residuals(R, t, X1, X2) < threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_count < 3:
        return None, None, None
    R, t = _kabsch(X1[best_inliers], X2[best_inliers])   # refit on the consensus
    return R, t, best_inliers


def _estimate_fundamental(prev: np.ndarray, curr: np.ndarray, ransac_px: float):
    import cv2

    try:
        F, inliers = cv2.findFundamentalMat(
            prev.astype(np.float64), curr.astype(np.float64),
            method=cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_px,
            confidence=0.999, maxIters=5000,
        )
    except cv2.error:
        # MAGSAC RAISES on degenerate input rather than returning None — e.g. all
        # correspondences collinear, or a near-pure rotation. Found on TUM fr3/walking,
        # where it killed an 827-frame run at frame ~200. A live tracker must survive a
        # single bad frame pair, so this is reported as degenerate, not propagated.
        return None, None
    if F is None or F.shape != (3, 3):
        # findFundamentalMat can return a 9x3 stack of candidate solutions for the
        # 7-point case, or None. Either way there is no single usable F.
        return None, None
    return F, (inliers.ravel().astype(bool) if inliers is not None else None)


def filter_dynamic(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    detections: list[Detection],
    config: FilterConfig | None = None,
    prev_depth: np.ndarray | None = None,
    curr_depth: np.ndarray | None = None,
    K: np.ndarray | None = None,
) -> FilterResult:
    """Decide which correspondences the pose optimiser is allowed to use.

    `prev_pts`/`curr_pts` are (N, 2) matched pixel coordinates in the previous and
    current frame. `detections` are boxes in the CURRENT frame.

    Supplying `prev_depth`, `curr_depth` (one metre value per correspondence, 0 =
    invalid) and `K` adds the 3-D rigid-motion test alongside the epipolar one. On RGB-D
    input there is no reason not to: it costs one Kabsch fit and removes the epipolar
    test's blind direction.
    """
    config = config or FilterConfig()
    prev_pts = np.asarray(prev_pts, dtype=np.float64).reshape(-1, 2)
    curr_pts = np.asarray(curr_pts, dtype=np.float64).reshape(-1, 2)
    if len(prev_pts) != len(curr_pts):
        raise ValueError(
            f"{len(prev_pts)} previous points vs {len(curr_pts)} current — these must be "
            "matched pairs, not two independent keypoint sets"
        )
    n = len(prev_pts)

    # --- the semantic prior: mark suspects, do not judge them -------------------
    suspect = np.zeros(n, dtype=bool)
    box_members: list[np.ndarray] = []
    for det in detections:
        inside = det.contains(curr_pts)
        box_members.append(inside)
        if det.label in config.movable:
            suspect |= inside

    # --- is there enough motion for the constraint to mean anything? ------------
    parallax = float(np.median(np.linalg.norm(curr_pts - prev_pts, axis=1))) if n else 0.0
    if parallax < config.min_parallax_px:
        raise DegenerateMotion(
            f"median parallax {parallax:.2f}px < {config.min_parallax_px}px. Under pure "
            "rotation or a static camera every correspondence satisfies the epipolar "
            "constraint, so this test would pass everything and report a clean frame."
        )

    trusted = ~suspect
    if int(trusted.sum()) < config.min_trusted:
        raise DegenerateMotion(
            f"only {int(trusted.sum())} correspondences outside movable-class boxes "
            f"(need {config.min_trusted}). F would have to be fitted on the points it is "
            "meant to judge, which biases it toward their motion and inverts the result."
        )

    # --- geometry, fitted on the trusted set only -------------------------------
    F, _ = _estimate_fundamental(prev_pts[trusted], curr_pts[trusted], config.ransac_px)
    if F is None:
        raise DegenerateMotion("fundamental matrix estimation failed on the trusted set")

    sampson = sampson_distance(F, prev_pts, curr_pts)
    inconsistent = sampson > config.epipolar_px

    # --- depth consistency, where RGB-D makes it available ----------------------
    # The epipolar test cannot see motion parallel to the baseline: such a point slides
    # ALONG its epipolar line and stays consistent. A 3-D rigid residual has no such
    # blind direction, because the point genuinely is somewhere else.
    rigid_residual = np.full(n, np.nan)
    if prev_depth is not None and curr_depth is not None and K is not None:
        prev_depth = np.asarray(prev_depth, dtype=np.float64).ravel()
        curr_depth = np.asarray(curr_depth, dtype=np.float64).ravel()
        if len(prev_depth) != n or len(curr_depth) != n:
            raise ValueError("depth arrays must be one value per correspondence")

        # 0 is the no-return sentinel throughout this pipeline; unprojecting it would
        # place the point at the camera centre and drag the rigid fit toward the origin.
        valid = (prev_depth > 0) & (curr_depth > 0)
        X1 = unproject(prev_pts[valid], prev_depth[valid], np.asarray(K, dtype=np.float64))
        X2 = unproject(curr_pts[valid], curr_depth[valid], np.asarray(K, dtype=np.float64))

        fit_mask = trusted[valid]
        if int(fit_mask.sum()) >= 3:
            R, t, _ = estimate_rigid_ransac(
                X1[fit_mask], X2[fit_mask], config.depth_ransac_m, seed=0
            )
            if R is not None:
                residual = rigid_residuals(R, t, X1, X2)
                rigid_residual[valid] = residual
                moved = np.zeros(n, dtype=bool)
                moved[np.flatnonzero(valid)] = residual > config.depth_residual_m
                # OR, not AND: the two tests are blind to different things, so a point
                # that fails either one has failed to move with the scene.
                inconsistent |= moved

    # --- object-level escalation ------------------------------------------------
    # Point-level geometry gives a scatter across a moving person. The box is what turns
    # that into an object decision, and sweeps up the points on it that matched
    # consistently — usually mismatches onto background, not real static structure.
    keep = ~inconsistent
    culled_boxes: list[int] = []
    for index, (det, inside) in enumerate(zip(detections, box_members)):
        if det.label not in config.movable or not inside.any():
            continue
        fraction = float(inconsistent[inside].mean())
        if fraction >= config.box_cull_fraction:
            keep[inside] = False
            culled_boxes.append(index)

    return FilterResult(
        keep=keep, sampson=sampson, suspect=suspect, culled_boxes=culled_boxes,
        fundamental=F, n_trusted=int(trusted.sum()), rigid_residual=rigid_residual,
    )

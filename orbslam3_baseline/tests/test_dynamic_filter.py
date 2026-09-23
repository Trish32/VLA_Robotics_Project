"""Synthetic two-view scenes with a known-moving object, so ground truth is exact.

Everything here is projected from 3-D by hand: the static background genuinely satisfies
the epipolar constraint and the moving cluster genuinely violates it, so a pass means the
filter recovered the right answer rather than that a threshold happened to fit.

Two tests carry the design argument:

  * `test_semantic_guidance_is_what_makes_the_geometric_test_work_at_all` — geometry
    alone rejects 0% of the moving object at every contamination level from 21% to 68%.
    Holding the detector's box out of the F estimate takes that to 100%.
  * `test_depth_consistency_sees_the_motion_epipolar_geometry_cannot` — the 2-D test is
    structurally blind to motion along the camera baseline; the 3-D rigid residual is not.

The rest pin the refusal paths and the failure modes that would otherwise look like
clean frames.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orbslam3_baseline.dynamic_filter import (  # noqa: E402
    DegenerateMotion,
    Detection,
    FilterConfig,
    filter_dynamic,
    sampson_distance,
)

pytest.importorskip("cv2")

F_LEN, CX, CY = 500.0, 320.0, 240.0
K = np.array([[F_LEN, 0, CX], [0, F_LEN, CY], [0, 0, 1.0]])


def rot_y(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def project(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = points_3d @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def two_view_scene(n_static=220, n_dynamic=60, object_shift=(0.25, 0.0, 0.0),
                   cam_t=(0.05, 0.01, 0.35), seed=0, return_depth=False):
    """Static background plus a rigidly translating object, seen from two poses.

    The default is a camera moving FORWARD while the object moves LATERALLY, which is
    both the common handheld/robot case and a geometry the epipolar test can actually
    see. Object motion parallel to the camera baseline slides points along their own
    epipolar lines and is close to invisible — measured at 0.49 px versus 5.6 px here.
    `test_motion_along_the_baseline_is_invisible...` pins that limit deliberately.
    """
    rng = np.random.default_rng(seed)
    static = np.column_stack([
        rng.uniform(-2.2, 2.2, n_static),
        rng.uniform(-1.6, 1.6, n_static),
        rng.uniform(4.0, 9.0, n_static),
    ])
    # A compact cluster, like a person: small in the image, coherent in depth.
    centre = np.array([0.75, 0.0, 5.0])
    moving = centre + rng.uniform(-0.32, 0.32, (n_dynamic, 3))

    R, t = rot_y(4.0), np.asarray(cam_t, dtype=float)

    prev = np.vstack([project(static, np.eye(3), np.zeros(3)),
                      project(moving, np.eye(3), np.zeros(3))])
    curr = np.vstack([project(static, R, t),
                      project(moving + np.asarray(object_shift), R, t)])

    is_dynamic = np.zeros(len(prev), dtype=bool)
    is_dynamic[n_static:] = True

    box_pts = curr[n_static:]
    box = None if n_dynamic == 0 else Detection(
        box_pts[:, 0].min() - 6, box_pts[:, 1].min() - 6,
        box_pts[:, 0].max() + 6, box_pts[:, 1].max() + 6,
        label="person", score=0.94,
    )
    if not return_depth:
        return prev, curr, is_dynamic, box

    # Per-correspondence depth, exactly as an RGB-D sensor would report it: the z
    # coordinate in each camera's own frame.
    all_1 = np.vstack([static, moving])
    all_2 = np.vstack([static, moving + np.asarray(object_shift)])
    d_prev = all_1[:, 2].copy()
    d_curr = (all_2 @ R.T + t)[:, 2]
    return prev, curr, is_dynamic, box, d_prev, d_curr


# --------------------------------------------------------------- the geometry


def test_a_static_scene_is_epipolar_consistent_everywhere():
    """Sanity: with no moving object the constraint holds for every correspondence, so
    any rejection later is attributable to motion rather than to the projection code."""
    prev, curr, is_dynamic, _ = two_view_scene(n_dynamic=0)
    result = filter_dynamic(prev, curr, detections=[])

    assert result.keep.all(), f"rejected {result.n_culled} points from a rigid scene"
    assert np.median(result.sampson) < 0.05


def test_the_moving_cluster_violates_the_constraint_and_the_background_does_not():
    prev, curr, is_dynamic, box = two_view_scene()
    result = filter_dynamic(prev, curr, detections=[box])

    assert result.sampson[~is_dynamic].max() < 1.0
    assert np.median(result.sampson[is_dynamic]) > 3.0


def test_dynamic_points_are_culled_and_background_is_kept():
    prev, curr, is_dynamic, box = two_view_scene()
    result = filter_dynamic(prev, curr, detections=[box])

    recall = result.keep[~is_dynamic].mean()
    rejection = (~result.keep[is_dynamic]).mean()
    assert rejection > 0.95, f"kept {1 - rejection:.0%} of the moving object"
    # ~9% of good background is lost. That is the real price of the design: F is fitted
    # on the trusted subset rather than on everything, so it is a little less accurate,
    # and borderline correspondences fall the wrong side of a 1.5px threshold. Cheap
    # compared with feeding the optimiser a walking person, but not free — and worth
    # knowing before tightening `epipolar_px`.
    assert recall > 0.90, f"threw away {1 - recall:.0%} of good background"


# ------------------------------------------------- why the semantic prior exists


@pytest.mark.parametrize("n_static,n_dynamic", [(220, 60), (140, 120), (70, 150)])
def test_semantic_guidance_is_what_makes_the_geometric_test_work_at_all(n_static, n_dynamic):
    """The core claim of the design, as a controlled A/B across contamination levels.

    Measured here: a fundamental matrix fitted on ALL correspondences rejects **0%** of
    the moving object — at every moving fraction from 21% to 68%. Geometry alone is not
    merely weaker, it is inert.

    The mechanism is that a compact moving cluster is CHEAP for RANSAC to accommodate.
    Tilting F slightly to make the object's 60 points inliers costs only a few percent
    of the spread-out background, so the robust fit takes that trade every time. The
    object then satisfies the very constraint meant to expose it.

    Holding the detector's box out of the estimate — and nothing else — takes rejection
    from 0% to 100%.
    """
    prev, curr, is_dynamic, box = two_view_scene(n_static=n_static, n_dynamic=n_dynamic)

    naive = filter_dynamic(prev, curr, detections=[])          # geometry alone
    guided = filter_dynamic(prev, curr, detections=[box])      # semantic prior guiding

    naive_rejected = (~naive.keep[is_dynamic]).mean()
    guided_rejected = (~guided.keep[is_dynamic]).mean()

    assert naive_rejected < 0.10, (
        f"unguided fit rejected {naive_rejected:.0%} of the moving object — if this "
        "starts working, the argument for the semantic prior needs revisiting"
    )
    assert guided_rejected > 0.95, f"guided fit rejected only {guided_rejected:.0%}"
    assert guided.keep[~is_dynamic].mean() > 0.90


def test_motion_along_the_baseline_is_invisible_to_epipolar_geometry():
    """A documented blind spot, pinned so it cannot be mistaken for a clean frame.

    The epipolar constraint only sees the component of object motion that takes a point
    OFF its epipolar line. An object translating parallel to the camera baseline slides
    along that line instead, and the correspondence stays consistent.

    Measured against the ground-truth F on this scene: 0.49 px for baseline-parallel
    motion versus 5.6 px for the same displacement perpendicular to it — a factor of 11,
    straddling any sane threshold.

    The practical consequence: a person walking in the same direction the camera is
    panning is NOT rejected by geometry. Only the semantic box catches them, and only if
    the object-level escalation is reached. Depth-based consistency (which RGB-D gives us
    for free) would close this gap and is the obvious next improvement.
    """
    prev, curr, is_dynamic, box = two_view_scene(
        cam_t=(0.32, 0.01, 0.06),            # camera pans laterally
        object_shift=(0.22, 0.0, 0.0),       # object moves the same way
    )
    result = filter_dynamic(prev, curr, detections=[box])

    assert np.median(result.sampson[is_dynamic]) < 1.0, (
        "this scene is supposed to be the degenerate one; if geometry now detects it, "
        "the blind-spot documentation is stale"
    )
    assert result.keep[is_dynamic].mean() > 0.8, "unexpectedly rejected — recheck the claim"


def test_depth_consistency_sees_the_motion_epipolar_geometry_cannot():
    """Closes the blind spot above, using depth RGB-D already gives us.

    Same degenerate scene: camera pans laterally, object translates the same way, so
    every moving point stays on its epipolar line. The 2-D test is structurally unable
    to see it.

    A 3-D rigid residual has no blind direction — the point genuinely is somewhere else,
    whatever direction it went. One Kabsch fit on the trusted correspondences turns a
    0.49 px non-signal into a residual in metres.
    """
    prev, curr, is_dynamic, box, d_prev, d_curr = two_view_scene(
        cam_t=(0.32, 0.01, 0.06),
        object_shift=(0.22, 0.0, 0.0),
        return_depth=True,
    )

    epipolar_only = filter_dynamic(prev, curr, detections=[box])
    with_depth = filter_dynamic(prev, curr, detections=[box],
                                prev_depth=d_prev, curr_depth=d_curr, K=K)

    assert epipolar_only.keep[is_dynamic].mean() > 0.80, (
        "the epipolar test was supposed to miss this; the blind-spot claim is stale"
    )
    assert (~with_depth.keep[is_dynamic]).mean() > 0.95, (
        "depth consistency failed to catch baseline-parallel motion"
    )
    assert with_depth.keep[~is_dynamic].mean() > 0.90, "background lost to the depth test"

    moved = with_depth.rigid_residual[is_dynamic]
    still = with_depth.rigid_residual[~is_dynamic]
    assert np.nanmedian(moved) > 0.15, "moving object should be ~0.22 m out of place"
    assert np.nanmedian(still) < 0.02


def test_invalid_depth_falls_back_to_the_epipolar_test_rather_than_the_origin():
    """0 is the no-return sentinel. Unprojecting it puts the point at the camera centre,
    which would drag the rigid fit toward the origin and corrupt every residual."""
    prev, curr, is_dynamic, box, d_prev, d_curr = two_view_scene(return_depth=True)
    d_prev = d_prev.copy()
    d_prev[:40] = 0.0                       # sensor dropouts on part of the background

    result = filter_dynamic(prev, curr, detections=[box],
                            prev_depth=d_prev, curr_depth=d_curr, K=K)

    assert np.isnan(result.rigid_residual[:40]).all(), "invalid depth was used anyway"
    assert result.keep[:40].mean() > 0.90, "dropout points were wrongly culled"
    assert (~result.keep[is_dynamic]).mean() > 0.95


def test_a_parked_car_is_kept_because_geometry_overrules_the_label():
    """Semantics alone would delete a stationary vehicle — well-textured, rigid, and
    exactly the structure ORB depends on. The label marks it suspect; the geometry clears
    it."""
    prev, curr, is_dynamic, _ = two_view_scene(n_dynamic=60, object_shift=(0.0, 0.0, 0.0))
    box_pts = curr[-60:]
    parked = Detection(box_pts[:, 0].min() - 6, box_pts[:, 1].min() - 6,
                       box_pts[:, 0].max() + 6, box_pts[:, 1].max() + 6, label="car")

    result = filter_dynamic(prev, curr, detections=[parked])

    assert result.suspect[is_dynamic].all(), "the box should still mark these as suspect"
    assert result.keep[is_dynamic].mean() > 0.95, "a stationary car was culled anyway"
    assert not result.culled_boxes


def test_a_box_on_a_non_movable_class_does_not_protect_the_fit():
    """Only movable classes are held out. A 'tv' box must not become a way to exclude
    real structure from the estimate."""
    prev, curr, is_dynamic, box = two_view_scene()
    mislabelled = Detection(box.x1, box.y1, box.x2, box.y2, label="tv")

    result = filter_dynamic(prev, curr, detections=[mislabelled])
    assert not result.suspect.any()
    assert result.n_trusted == len(prev)


def test_the_whole_object_goes_once_enough_of_it_is_inconsistent():
    """Point-level geometry leaves a few points on a moving person looking fine — almost
    always mismatches onto background rather than static structure. The box escalates."""
    prev, curr, is_dynamic, box = two_view_scene()
    result = filter_dynamic(prev, curr, detections=[box])

    assert result.culled_boxes == [0]
    assert not result.keep[is_dynamic].any(), "object-level cull left survivors"


# --------------------------------------------------------- degenerate geometry


def test_a_stationary_camera_is_refused_rather_than_reported_clean():
    """Under no translation every correspondence satisfies the constraint, so the test
    passes everything. Reporting 'no dynamic points' there is a false negative dressed as
    a clean frame."""
    prev, _, _, _ = two_view_scene()
    with pytest.raises(DegenerateMotion, match="parallax"):
        filter_dynamic(prev, prev.copy(), detections=[])


def test_too_few_trusted_correspondences_is_refused():
    """If boxes cover nearly the frame, F would have to be fitted on the points it is
    meant to judge — which is the corruption this design exists to prevent."""
    prev, curr, _, _ = two_view_scene()
    everything = Detection(-1e4, -1e4, 1e4, 1e4, label="person")

    with pytest.raises(DegenerateMotion, match="outside movable-class boxes"):
        filter_dynamic(prev, curr, detections=[everything])


def test_mismatched_point_counts_are_refused():
    prev, curr, _, _ = two_view_scene()
    with pytest.raises(ValueError, match="matched pairs"):
        filter_dynamic(prev, curr[:-5], detections=[])


def test_a_degenerate_epipolar_line_scores_infinite_not_zero():
    """With F = 0 the numerator and denominator both vanish. Returning 0 would read as
    'perfectly consistent' and keep a point the geometry cannot vouch for at all."""
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    distances = sampson_distance(np.zeros((3, 3)), pts, pts)
    assert np.isinf(distances).all()


def test_the_config_thresholds_are_reachable_from_the_result():
    """Tuning needs the residuals, not just the verdict."""
    prev, curr, is_dynamic, box = two_view_scene()
    result = filter_dynamic(prev, curr, [box], FilterConfig(epipolar_px=0.2))

    assert result.sampson.shape == (len(prev),)
    assert result.fundamental.shape == (3, 3)
    assert result.n_trusted == int((~result.suspect).sum())


def test_a_magsac_exception_becomes_degenerate_motion(monkeypatch):
    """cv2.findFundamentalMat with USAC_MAGSAC RAISES on input it cannot model, rather
    than returning None. Found the hard way: an uncaught cv2.error killed an 827-frame
    TUM fr3/walking run at frame ~200. A tracker must survive one bad frame pair, so the
    exception is translated into the same refusal every other degenerate case uses.

    Forced rather than contrived: constructing input MAGSAC reliably chokes on is
    fragile, and the behaviour under test is the handler, not OpenCV's failure criteria.
    """
    import cv2

    def boom(*_args, **_kwargs):
        raise cv2.error("simulated USAC degeneracy")

    monkeypatch.setattr(cv2, "findFundamentalMat", boom)

    prev, curr, _, box = two_view_scene()
    with pytest.raises(DegenerateMotion, match="estimation failed"):
        filter_dynamic(prev, curr, detections=[box])

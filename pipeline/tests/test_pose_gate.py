"""The stage-4 pose gate: what it admits, what it refuses, and why.

The gate is the only thing standing between a wrong 6-DoF pose and a planner acting on
it, so its failure modes are worth pinning down. The numbers in `test_chair4_*` are the
real measured ones from the Kaggle run — they are here so a future change that would
have admitted that pose fails a test instead of shipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.tools.e2e_pose import check_pose, rotation_error_deg


def R_z(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def pose(t=(0, 0, 2.0), R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = t
    return T


def gate(T, *, origin=(0, 0, 2.0), R_ref=None, depth=None, span=1.0,
         max_t=10.0, max_r=30.0):
    return check_pose(T, frame=0, origin_cam=np.asarray(origin, float),
                      R_world_to_cam=np.eye(3) if R_ref is None else R_ref,
                      depth_median_m=depth, depth_span_m=span,
                      max_translation_cm=max_t, max_rotation_deg=max_r)


# --------------------------------------------------------------------- rotation metric

def test_rotation_error_is_zero_for_identical_rotations():
    R = R_z(37.0)
    assert rotation_error_deg(R, R) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("deg", [1.0, 30.0, 90.0, 179.0])
def test_rotation_error_recovers_the_angle(deg):
    assert rotation_error_deg(R_z(deg), np.eye(3)) == pytest.approx(deg, abs=1e-6)


def test_rotation_error_is_symmetric():
    A, B = R_z(20.0), R_z(95.0)
    assert rotation_error_deg(A, B) == pytest.approx(rotation_error_deg(B, A), abs=1e-9)


def test_rotation_error_survives_trace_just_outside_arccos_domain():
    """A perfect match can push the trace a float hair above 3; arccos must not NaN."""
    R = R_z(0.0) * (1 + 1e-12)
    assert not np.isnan(rotation_error_deg(R, np.eye(3)))


# ------------------------------------------------------------------------- the gate

def test_clean_pose_is_accepted():
    c = gate(pose())
    assert c.accepted and not c.reasons
    assert c.translation_cm == pytest.approx(0.0)


def test_translation_just_inside_and_just_outside_the_threshold():
    assert gate(pose(t=(0, 0, 2.09))).accepted
    assert not gate(pose(t=(0, 0, 2.11))).accepted


def test_rotation_alone_can_fail_an_otherwise_perfect_pose():
    """The case that mattered: translation fine, orientation nonsense."""
    c = gate(pose(R=R_z(99.5)))
    assert not c.accepted
    assert c.translation_cm == pytest.approx(0.0)
    assert "rotation" in c.reasons[0]


def test_centroid_behind_the_surface_is_allowed_within_the_observed_depth():
    # A surface spanning 0.5 m in depth legitimately puts the origin ~0.4 m behind the
    # median. Within span + margin, so allowed.
    c = gate(pose(t=(0, 0, 2.4)), origin=(0, 0, 2.4), depth=2.0, span=0.5)
    assert c.accepted, c.reasons


def test_origin_too_far_behind_the_surface_is_refused():
    c = gate(pose(t=(0, 0, 3.0)), origin=(0, 0, 3.0), depth=2.0, span=0.1)
    assert not c.accepted
    assert "behind the measured surface" in c.reasons[0]


def test_depth_check_uses_the_observed_slab_not_the_instance_extent():
    """The regression that matters: chair_4 spans 1.58 m in 3-D but the camera sees a
    13.4 cm slab. Bounding by half the extent gave a 79 cm tolerance and let a 60.8 cm
    error through; bounding by the observed span refuses it."""
    c = gate(pose(t=(0.5465, 0.4297, 2.7108)), origin=(0.3879, 0.5824, 2.05),
             depth=2.103, span=0.134)
    assert not c.accepted
    assert c.behind_surface_cm == pytest.approx(60.78, abs=0.05)
    assert any("behind the measured surface" in r for r in c.reasons)


def test_depth_check_is_skipped_when_depth_is_unavailable():
    c = gate(pose(t=(0, 0, 9.0)), origin=(0, 0, 9.0), depth=None, span=1.0)
    assert c.accepted and c.behind_surface_cm == 0.0


def test_all_three_failures_are_reported_together():
    c = gate(pose(t=(1.0, 0, 3.0), R=R_z(120.0)), origin=(0, 0, 2.0),
             depth=2.0, span=0.05)
    assert not c.accepted and len(c.reasons) == 3


# ------------------------------------------------------- the real chair_4 measurement

def test_chair4_register_pose_is_refused_on_translation():
    """v10's register() output against the centroid our map reports: 72.14 cm."""
    c = gate(pose(t=(0.5673, 0.4239, 2.7305)), origin=(0.3879, 0.5824, 2.05),
             depth=2.103, span=0.134)
    assert not c.accepted
    assert c.translation_cm == pytest.approx(72.14, abs=0.05)


def test_chair4_rotation_probe_angle():
    """Moving the mesh origin 52.9 cm moved FoundationPose's answer 53.5 cm, but 99.5
    deg away from where the map says it should have gone. Same offset vector, two
    rotations — so the angle between them is a lower bound on the rotation error."""
    d_fp = np.array([0.9911, 0.1942, 2.9620]) - np.array([0.5673, 0.4239, 2.7305])
    d_map = np.array([0.0707, 0.6246, 2.4716]) - np.array([0.3879, 0.5824, 2.05])
    assert np.linalg.norm(d_fp) == pytest.approx(np.linalg.norm(d_map), abs=0.01)
    cos = d_fp @ d_map / np.linalg.norm(d_fp) / np.linalg.norm(d_map)
    assert np.degrees(np.arccos(cos)) == pytest.approx(99.5, abs=0.5)


# ------------------------------------------- hypothesis agreement (map-free check)

from pipeline.tools.e2e_pose import check_agreement


def test_agreement_accepts_converged_hypotheses():
    """mustard0's real numbers: the top-16 are one orientation from different starts."""
    ok, detail = check_agreement(
        {"k": 16, "clustered_within_15deg": 16, "median_pairwise_deg": 0.29})
    assert ok and "16/16" in detail


def test_agreement_rejects_scattered_hypotheses():
    """chair_4's real numbers: the top-16 span 127 deg, so the winner is arbitrary."""
    ok, _ = check_agreement(
        {"k": 16, "clustered_within_15deg": 2, "median_pairwise_deg": 127.31})
    assert not ok


def test_agreement_rejects_on_pairwise_spread_even_when_the_count_looks_fine():
    """A majority can sit within 15 deg of the top while the set is still spread."""
    ok, _ = check_agreement(
        {"k": 16, "clustered_within_15deg": 9, "median_pairwise_deg": 40.0})
    assert not ok


def test_agreement_is_skipped_for_pre_r14_results():
    """Older pose_result.json has no agreement block; absence must not fail the gate."""
    ok, detail = check_agreement(None)
    assert ok and "skipped" in detail


def test_agreement_inverts_the_score_based_reading():
    """The withdrawn metric ranked mustard0 WORSE; this one must rank it better."""
    ours = {"k": 16, "clustered_within_15deg": 2, "median_pairwise_deg": 127.31,
            "within_1pct": 48}
    mustard = {"k": 16, "clustered_within_15deg": 16, "median_pairwise_deg": 0.29,
               "within_1pct": 91}
    assert mustard["within_1pct"] > ours["within_1pct"]      # flatter scores
    assert check_agreement(mustard)[0] and not check_agreement(ours)[0]

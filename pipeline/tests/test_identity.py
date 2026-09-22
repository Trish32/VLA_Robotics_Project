"""Time, frames and identity — the three things that must agree across the whole stack.

Each test here corresponds to a way the pipeline was previously inconsistent, found by
auditing the repo: three time representations, frame names hardcoded in five files, and
instance IDs assigned by score rank.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.identity import (  # noqa: E402
    FLOAT64_EXACT_INT,
    NS_PER_S,
    Frames,
    InstanceRegistry,
    Stamp,
    TimeError,
    assert_lossless,
)

EPOCH_NS = 1_305_031_102_175_304_000     # a real TUM timestamp, in ns


# ------------------------------------------------------------------------ time


def test_float_seconds_cannot_carry_nanoseconds_at_epoch_scale():
    """The measurement behind the whole design. float64 holds integers exactly only to
    2**53; epoch nanoseconds are ~1.3e18, three orders of magnitude past it. A ns ->
    float seconds -> ns round trip therefore quantises, silently."""
    original = Stamp(EPOCH_NS)
    seconds = original.to_seconds()
    round_tripped = Stamp.from_seconds(seconds)

    # The rigorous statement is the ULP, not one sample's rounding: at ~1.3e9 seconds
    # the gap between adjacent float64 values is already coarser than a nanosecond, so
    # most ns values in this range are simply not representable.
    ulp_ns = float(np.spacing(seconds)) * NS_PER_S
    assert ulp_ns > 1.0, (
        f"float64 resolution at epoch scale is {ulp_ns:.1f} ns; if this drops below "
        "1 ns the module docstring is stale"
    )
    assert round_tripped.ns != original.ns, "this timestamp happened to be representable"
    assert abs(round_tripped.ns - original.ns) <= ulp_ns
    assert EPOCH_NS > FLOAT64_EXACT_INT


def test_ros_conversion_is_exact_in_both_directions():
    """ROS already stores sec + nanosec as integers, so nothing is lost either way."""
    class FakeTime:
        def __init__(self, sec, nanosec):
            self.sec, self.nanosec = sec, nanosec

    stamp = Stamp(EPOCH_NS)
    msg = stamp.to_ros(FakeTime)
    assert Stamp.from_ros(msg).ns == EPOCH_NS
    assert msg.sec * NS_PER_S + msg.nanosec == EPOCH_NS


def test_a_float_is_refused_where_a_stamp_is_expected():
    """The failure this type exists to prevent: a float seconds value passed where ns
    were meant is off by a factor of 1e9 and looks like a plausible timestamp."""
    with pytest.raises(TimeError, match="integer nanoseconds"):
        Stamp(1305031102.175304)


def test_the_lossless_guard_refuses_values_float_cannot_hold():
    assert assert_lossless(1_000_000) == 1_000_000
    with pytest.raises(TimeError, match="exact-integer range"):
        assert_lossless(EPOCH_NS)


def test_stamps_order_and_subtract_as_time():
    early, late = Stamp(EPOCH_NS), Stamp(EPOCH_NS + 2 * NS_PER_S)
    assert early < late
    assert late - early == pytest.approx(2.0)
    assert early.age_s(late) == pytest.approx(2.0)


# ---------------------------------------------------------------------- frames


def test_frame_names_come_from_one_authority():
    assert Frames.instance("pear_1") == "object/pear_1"
    assert Frames.pregrasp("pear_1") == "object/pear_1/pregrasp"
    assert Frames.keyframe(7) == "keyframe/7"


def test_the_optical_and_body_camera_frames_are_distinct():
    """tf2 never validates a name, so publishing an optical-convention pose on a
    `_link` frame is a silent 90-degree error — every transform still well-formed."""
    assert Frames.CAMERA_OPTICAL != Frames.CAMERA_BODY
    assert Frames.CAMERA_OPTICAL.endswith("optical_frame")


def test_pregrasp_is_a_child_of_its_object_frame():
    """So TF composes it into any frame the caller works in, and it tracks the object."""
    assert Frames.pregrasp("mug_2").startswith(Frames.instance("mug_2"))


# -------------------------------------------------------------------- identity


def box(x, y, z, size=0.1):
    return np.array([x, y, z]), np.array([size, size, size])


def test_the_same_object_observed_twice_keeps_one_id():
    """The whole point. Rank-ordered IDs gave the same physical object a different name
    on every run, and anything holding a reference pointed somewhere else."""
    reg = InstanceRegistry()
    c, e = box(1.0, 0.0, 0.8)
    first, new1 = reg.associate(c, e, "pear", Stamp(1 * NS_PER_S))
    second, new2 = reg.associate(c + 0.01, e, "pear", Stamp(2 * NS_PER_S))

    assert first == second
    assert new1 and not new2
    assert reg.tracked[first].observations == 2


def test_a_different_object_gets_a_different_id():
    reg = InstanceRegistry()
    a, _ = reg.associate(*box(0.0, 0, 0.8), "pear", Stamp(NS_PER_S))
    b, _ = reg.associate(*box(2.0, 0, 0.8), "pear", Stamp(NS_PER_S))
    assert a != b and len(reg.tracked) == 2


def test_overlap_alone_does_not_merge_two_different_things():
    """A cup sitting on a book overlaps it substantially. Without the label test they
    become one object, and the planner is told to grasp a cup-book."""
    reg = InstanceRegistry()
    book, _ = reg.associate(*box(0, 0, 0.75, 0.2), "book", Stamp(NS_PER_S))
    cup, _ = reg.associate(*box(0, 0, 0.78, 0.18), "cup", Stamp(NS_PER_S))
    assert book != cup


def test_ids_carry_no_rank_information():
    """`inst_0` meant 'highest-scoring proposal this run'. The replacement must not."""
    reg = InstanceRegistry()
    first, _ = reg.associate(*box(0, 0, 0.8), "pear", Stamp(NS_PER_S))
    second, _ = reg.associate(*box(2, 0, 0.8), "mug", Stamp(NS_PER_S))

    assert first.startswith("pear_") and second.startswith("mug_")
    assert "inst" not in first
    # Minted in observation order, never re-sorted by score.
    assert first.endswith("_1") and second.endswith("_2")


def test_a_label_with_punctuation_still_yields_a_usable_frame_name():
    """Labels come from an open vocabulary, so they can contain anything. A TF frame
    name with a slash in it would silently create a nested frame."""
    reg = InstanceRegistry()
    ident, _ = reg.associate(*box(0, 0, 0.8), "a computer screen/monitor", Stamp(NS_PER_S))
    assert "/" not in ident
    assert Frames.instance(ident).count("/") == 1


def test_a_late_observation_does_not_roll_an_instance_backwards():
    """Stages run on their own threads; replies arrive out of order."""
    reg = InstanceRegistry()
    ident, _ = reg.associate(*box(1.0, 0, 0.8), "pear", Stamp(5 * NS_PER_S))
    reg.associate(*box(1.02, 0, 0.8), "pear", Stamp(1 * NS_PER_S))
    assert reg.tracked[ident].stamp.ns == 5 * NS_PER_S
    assert reg.tracked[ident].centre[0] == pytest.approx(1.0)


def test_batch_association_lets_the_larger_instance_claim_its_identity_first():
    """Processing smallest-first would let a cup on a table match the table's box and
    steal its ID."""
    reg = InstanceRegistry()
    table = (np.array([0, 0, 0.72]), np.array([1.2, 0.8, 0.05]), "table", Stamp(NS_PER_S))
    cup = (np.array([0, 0, 0.80]), np.array([0.08, 0.08, 0.1]), "cup", Stamp(NS_PER_S))

    ids = reg.associate_batch([cup, table])          # small one listed first
    assert ids[0].startswith("cup_") and ids[1].startswith("table_")


def test_reassociation_survives_a_rerun_with_reordered_proposals():
    """The regression that motivated this module: segmentation re-run returns the same
    objects in a different score order, and identity must not follow the order."""
    reg = InstanceRegistry()
    run_a = [(np.array([0, 0, 0.8]), np.array([0.1]*3), "pear", Stamp(NS_PER_S)),
             (np.array([1, 0, 0.8]), np.array([0.1]*3), "mug", Stamp(NS_PER_S))]
    ids_a = reg.associate_batch(run_a)

    ids_b = reg.associate_batch(list(reversed(run_a)))    # same scene, reversed order
    assert set(ids_a) == set(ids_b)
    assert ids_b == list(reversed(ids_a))
    assert len(reg.tracked) == 2, "a reorder must not create new instances"

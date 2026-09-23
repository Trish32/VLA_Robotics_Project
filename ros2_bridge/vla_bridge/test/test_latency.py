"""The latency tracker, and the clock discipline it enforces.

Pure Python — no ROS, no network. The thing worth testing here is not the arithmetic but
the refusal: a negative duration means two different clocks were compared, and silently
averaging it in produces a percentile describing something that never happened.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge.latency import STAGES, LatencyTracker, Throttle  # noqa: E402


def test_percentiles_come_from_the_window_not_a_running_mean():
    """A pipeline that is fine at p50 and terrible at p95 is one that drops frames, and
    an average is exactly the statistic that hides it."""
    tracker = LatencyTracker(window=100)
    for value in list(range(1, 100)) + [500.0]:
        tracker.record(request=float(value))

    stats = tracker.summary()["request"]
    assert stats["p50"] == pytest.approx(50, abs=2)
    assert stats["max"] == 500.0
    assert stats["p95"] < stats["max"], "p95 should not be dragged to the outlier"


def test_the_window_forgets_so_a_slow_start_does_not_haunt_the_numbers():
    tracker = LatencyTracker(window=10)
    for _ in range(10):
        tracker.record(request=900.0)
    for _ in range(10):
        tracker.record(request=5.0)

    assert tracker.summary()["request"]["max"] == 5.0


def test_a_negative_duration_is_refused_as_a_mixed_clock():
    """ROS time can step — sim time, a bag replay, an NTP correction. A duration
    measured across a backwards step goes negative, which reads as the reply arriving
    before the request was sent."""
    tracker = LatencyTracker()
    with pytest.raises(ValueError, match="ROS-clock step"):
        tracker.record(request=-3.0)


def test_an_unknown_stage_is_refused_rather_than_silently_collected():
    """A typo'd stage name would otherwise accumulate into a bucket nothing reports."""
    tracker = LatencyTracker()
    with pytest.raises(KeyError, match="unknown stage"):
        tracker.record(requset=1.0)


def test_the_summary_omits_stages_with_no_samples():
    tracker = LatencyTracker()
    tracker.record(request=1.0)
    assert set(tracker.summary()) == {"request"}


def test_format_reports_drops_and_failures_because_they_bias_the_rest():
    """Latency measured only over successful requests looks best exactly when the system
    is dropping the most."""
    tracker = LatencyTracker()
    tracker.record(age=1.0, request=2.0, publish=0.5, total=3.5)
    tracker.dropped = 7
    tracker.failed = 2

    text = tracker.format()
    assert "dropped=7" in text and "failed=2" in text
    for stage in STAGES:
        assert stage in text


def test_format_is_safe_before_any_sample_arrives():
    assert "no samples" in LatencyTracker().format()


def test_throttle_allows_one_pass_per_interval():
    throttle = Throttle(interval_s=3600.0)
    assert throttle.ready()
    assert not throttle.ready()


def test_a_zero_interval_throttle_is_how_logging_is_disabled():
    """`latency_log_s: 0` must mean off, not 'log every single frame'."""
    throttle = Throttle(interval_s=0.0)
    assert throttle.interval_ns == 0

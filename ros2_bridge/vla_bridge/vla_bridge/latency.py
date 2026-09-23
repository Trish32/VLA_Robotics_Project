"""Where the time actually goes between a camera frame and a published pose.

The bridge's latency is usually assumed to be transport. It is worth measuring before
optimising, because the plausible answers differ by two orders of magnitude: copying a
640x480 RGB frame is ~0.1 ms and CDR serialisation a few ms, while a VLA forward pass is
30-100 ms. An optimisation aimed at the wrong one of those is free to look successful and
change nothing.

Four stages, chosen so each maps to a different fix:

    age        header stamp -> just before we send. ROS transport, the driver's own
               latency, and any time the callback spent queued behind something else.
               This is the stage a blocked executor inflates.
    request    send -> reply received. Network plus whatever the server did. Shrinking
               this means a smaller payload or a faster model, not a faster transport.
    publish    reply -> published. Decode and publish; should be sub-millisecond, and if
               it is not, the decoder is doing something expensive.
    total      header stamp -> published. What a downstream consumer actually waits for.

TWO CLOCKS, DELIBERATELY
------------------------
`age` compares against a message header, so it must use the ROS clock — the only clock
those stamps are in. Every other stage is a duration and uses `time.monotonic_ns`.

Mixing them is not a style preference. ROS time can step: `use_sim_time`, a bag replay,
or an NTP correction mid-run. A duration measured across a step is wrong by the step,
and with a backwards step it goes negative — which reads as "the reply arrived before we
sent it" and quietly poisons a percentile. Durations therefore never touch the ROS clock,
and `age` is the only stage that can legitimately be affected by one.
"""

from __future__ import annotations

import time
from collections import deque

NS_PER_MS = 1_000_000.0

# The order stages are reported in. `total` last because it is the sum a reader cares
# about, and the stages above it are the explanation.
STAGES = ("age", "request", "publish", "total")


def monotonic_ns() -> int:
    """Durations only. Never compare this against a ROS message stamp."""
    return time.monotonic_ns()


class LatencyTracker:
    """Fixed-window percentiles, cheap enough to leave on in production.

    A ring buffer rather than a running mean: the mean of a latency distribution is the
    least informative statistic it has. A pipeline that is fine at p50 and 400 ms at p95
    is a pipeline that drops frames, and an average hides exactly that.
    """

    def __init__(self, window: int = 200) -> None:
        self._samples = {stage: deque(maxlen=window) for stage in STAGES}
        self.dropped = 0
        self.failed = 0

    def record(self, **stages_ms: float) -> None:
        for stage, value in stages_ms.items():
            if stage not in self._samples:
                raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
            # A negative duration means the clocks were mixed somewhere. Recording it
            # would drag a percentile toward a number that never happened.
            if value < 0:
                raise ValueError(
                    f"negative duration for {stage!r} ({value:.3f} ms) — a ROS-clock "
                    "step was measured as a duration; use monotonic_ns for durations"
                )
            self._samples[stage].append(value)

    @staticmethod
    def _percentile(ordered: list[float], fraction: float) -> float:
        if not ordered:
            return float("nan")
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, samples in self._samples.items():
            if not samples:
                continue
            ordered = sorted(samples)
            out[stage] = {
                "p50": self._percentile(ordered, 0.50),
                "p95": self._percentile(ordered, 0.95),
                "max": ordered[-1],
                "n": len(ordered),
            }
        return out

    def format(self) -> str:
        summary = self.summary()
        if not summary:
            return "latency: no samples yet"
        parts = [
            f"{stage} p50={s['p50']:.1f} p95={s['p95']:.1f} max={s['max']:.1f}"
            for stage in STAGES if (s := summary.get(stage))
        ]
        counts = f"n={next(iter(summary.values()))['n']}"
        if self.dropped:
            counts += f" dropped={self.dropped}"
        if self.failed:
            counts += f" failed={self.failed}"
        return "latency ms | " + " | ".join(parts) + f" | {counts}"


class Throttle:
    """True at most once per interval. For logging a summary without flooding."""

    def __init__(self, interval_s: float) -> None:
        self.interval_ns = int(interval_s * 1e9)
        self._last = 0

    def ready(self) -> bool:
        now = monotonic_ns()
        if now - self._last < self.interval_ns:
            return False
        self._last = now
        return True

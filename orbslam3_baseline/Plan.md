# ORB-SLAM3 — plan

The localization result reproduces upstream, so the gaps here are about *scope*, not
correctness.

| item | note |
|---|---|
| **Monocular tracking-loss reduction** | not measured, and currently ill-posed: RGB-D tracks 827/827 on both the baseline and filtered runs, so there are no losses to reduce. Needs the monocular variant |
| **Multi-run medians** | every number is a single run. ORB-SLAM3 is multithreaded and non-deterministic; upstream reports medians over several runs |
| **The throughput cost** | dynamic rejection costs 51.4% of FPS and lands at 18.5, below a 30 Hz loop. Three ways to buy it back are listed in the root [RESULTS.md](../RESULTS.md) §1.2 — **none implemented or measured** |

## Why the filtered path is slow, precisely

The cost is **not** the detector — YOLOv8n + ByteTrack run as a separate cached pass and
contribute nothing to the tracking loop's frame time. It is that **ORB features are
extracted and matched twice per frame**, once by our geometric test and once inside
ORB-SLAM3, because upstream's `ORBextractor` does not expose its correspondences
(`ORBextractor.h:56` documents its mask argument as ignored).

Reusing upstream's correspondences would close it, but needs a patch exposing them from
`Tracking` — which trades the current zero-patches property for throughput.

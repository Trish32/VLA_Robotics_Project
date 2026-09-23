# ORB-SLAM3 — results

Full detail in the root [RESULTS.md](../RESULTS.md) §1. Summarised here.

## Static sequence — TUM `fr1/xyz` · MEASURED

| metric | value |
|---|---|
| frames tracked | **798 / 798** (100%), 0 lost |
| throughput | **41.4 FPS**, CPU only |
| **ATE RMSE** | **1.03 cm** |
| mean / median / max | 0.90 / 0.80 / 2.76 cm |

Published ORB-SLAM3 for this sequence is ~1.0 cm, so this **reproduces upstream**.

Alignment is **SE(3), 6-DoF, scale fixed** — RGB-D is metric, so fitting a scale would
let the optimiser absorb real metric drift into a scale factor. `--sim3` exists for the
monocular case and is off by default.

## Dynamic-feature rejection — `fr3/walking_xyz` · MEASURED

| metric | baseline | + rejection | change |
|---|---|---|---|
| **ATE RMSE** | 80.92 cm | **18.70 cm** | **−76.9%** |
| mean / median | 67.91 / 58.07 | 11.25 / 7.60 cm | −83% / −87% |
| frames tracked | 827 / 827 | 827 / 827 | — |
| throughput | 38.1 FPS | 18.5 FPS | **−51.4%** |

**Three things this does not say.** There is **no tracking-loss improvement** — both runs
tracked 827/827 with zero losses, so that experiment needs the monocular variant.
**ByteTrack's own contribution is 0.4%** (4 false-positive boxes suppressed), not 76.9% —
the gain is from the geometric verification. And it is a **single run**; ORB-SLAM3 is
multithreaded and non-deterministic, while upstream reports medians.

*Sanity check that the gain is not an alignment artifact:* baseline trajectory length
17.03 m, filtered 12.81 m, ground truth 5.79 m. The excess **is** the drift.

## Why both geometric halves are needed · MEASURED (synthetic, exact GT)

| moving fraction | F on **all** points | F on **non-box** points |
|---|---|---|
| 21% / 46% / 68% | **0%** rejected | **100%** rejected |

Geometry alone is not weaker — it is **inert**. A compact moving cluster is cheap for a
robust estimator to absorb, so MAGSAC tilts F to fit it and the object then satisfies the
constraint meant to expose it. Holding the detector's box out of the fit is the entire
difference. Cost: ~9% of good background also rejected.

Epipolar geometry has a blind direction — motion parallel to the baseline slides points
*along* their epipolar lines (0.49 px, undetectable). A 3-D rigid residual turns that
into **0.22 m**. The two tests combine with OR, not AND: they are blind to different
things.

## Tests

18, plus 8 root-caused bugs in `bug_log.txt`.

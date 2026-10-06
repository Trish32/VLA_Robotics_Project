# FoundationPose — 6-DoF object pose

Upstream [FoundationPose](https://github.com/NVlabs/FoundationPose) pinned at `a1b694b`.
Model-based pose estimation and tracking: `register(K, rgb, depth, ob_mask, mesh)` then
`track_one(...)`.

**→ [RESULTS.md](RESULTS.md)** — checkpoint fidelity, and what runs on a T4
**→ [Plan.md](Plan.md)** — the one thing that has not happened yet

## The limit, up front

`register()` and `track_one()` **do run** on a Tesla T4 against a mesh cut from our own
TSDF. **No pose accuracy is claimed** — on the rebuilt input the rotation error is
**124.27°**, the top-16 hypotheses do not agree on an orientation, and the tracked object
wanders **33.17 cm** across views that actually differ.

**The pose is refused downstream** on all four gate checks. On the original input it was
**175.63°** off — essentially flipped.

**The port itself is validated.** Upstream's own `demo_data/mustard0`, through the
identical code path, returns a correct pose — its origin sits **2.06 cm** behind the
measured surface against a 9.6 cm half-depth. So the error belongs to our input, not to
the model: a 695 px mask against mustard0's 3,252 px over a CAD mesh. See
[RESULTS.md](RESULTS.md) and `bug_log.txt` [5].

**Why ours was 695 px was then misdiagnosed, and the correction is the live item.** It
was published as a one-sided Poisson shell needing a denser fusion. It was neither: the
bundle builder kept the first 8 frames in which ≥ 50 of 1,142 instance points survived
its depth test — a 4.4% floor — and so selected frames showing **11–17%** of the chair
where **73%** was available in the same sequence, from the start of the run where the
SLAM trajectory is least converged. Rebuilt on measured visibility, same fusion:
**6,908 px**, camera baseline **3.4 → 20.8 cm**. The old bundle's 2.08 cm tracking spread
was weak for the same reason — its 8 frames spanned **1.1°** of view angle. `bug_log.txt` [8].

**The re-run then answered the question the fix was for.** On the rebuilt input,
verified in-job by fingerprint, rotation moved 175.63° → 124.27° and the hypotheses
still scatter (1/16 within 15°) — while mustard0 in the same job converges 16/16. The
input defects were real and **not binding**. `bug_log.txt` [10].

**Then the mesh recipe was cleared.** mustard0 meshed the chair's way — its own depth fused
along the CAD track, our `reject_outliers` and `poisson_mesh` shipped verbatim, matched
resolution — registers within **2.95°** of its CAD pose, 16/16. **And so was coverage:**
limited to the chair's **31°** of views, it still lands within **4.79°**, 16/16.

**Then the failure was located.** Our mesh at our map's pose explains the measured depth to
**3.9 cm** on the same pixels; FoundationPose's scorer prefers its own pose (74–166 cm off)
and one refinement call (5 internal iterations) from the map pose moves it **77 cm**. That
locates the behaviour in the estimator; it does not clear the inputs, which were built from
the same depth.

**And a compact object is accepted.** A book on TUM `fr1/xyz` — held-out frames, SAM masks,
our trajectory and mesh recipe — passes the hardened gate: 6/8 frames, 15/16 hypotheses
agree, depth residual 0.72 cm. Registration lands within 3.4 cm on all five compact targets
tried. [RESULTS.md](RESULTS.md).

## Where the mesh comes from — no CAD models

`register()` needs a mesh. Upstream's model-free alternative is BundleSDF's NeRF, an
optional CUDA build upstream leaves commented out. This stack uses neither: **DROID-SLAM
or ORB-SLAM3 in RGB-D gives metric poses, those fuse into a TSDF, and OpenMask3D cuts a
per-instance mesh out of the fused scene**, with `ob_mask` the instance projected back
into the frame. See `pipeline/tools/build_pose_bundle.py`.

That only holds because the poses are **metric**. Monocular SLAM carries an arbitrary
scale, which would make the mesh the wrong *size* and the resulting pose confidently
wrong — the failure nothing downstream can catch.

## compat.py raises; it does not stub

`Utils.py` imports `nvdiffrast` and six `pytorch3d` symbols at module scope, and
nvdiffrast is a CUDA rasteriser with no CPU build. `compat.py` registers modules that
**raise on use** rather than returning plausible geometry: nvdiffrast rasterises the pose
hypotheses the refiner scores, so a stub returning zeros would yield a confidently wrong
pose, and a pose is the canonical value nothing downstream can sanity-check.

`tests/` — 6 tests, all about that guard behaving correctly.

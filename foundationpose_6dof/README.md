# FoundationPose — 6-DoF object pose

Upstream [FoundationPose](https://github.com/NVlabs/FoundationPose) pinned at `a1b694b`.
Model-based pose estimation and tracking: `register(K, rgb, depth, ob_mask, mesh)` then
`track_one(...)`.

**→ [RESULTS.md](RESULTS.md)** — checkpoint fidelity, and what runs on a T4
**→ [Plan.md](Plan.md)** — the one thing that has not happened yet

## The limit, up front

`register()` and `track_one()` **do run** on a Tesla T4 against a mesh cut from our own
TSDF, and the tracker is self-consistent — **2.08 cm** world-frame spread over 8 frames.
That bounds drift; it is not an accuracy result, because a tracker locked onto a wrong
pose holds it just as steadily.

**No pose accuracy is claimed, and the pose is refused downstream.** The returned pose
disagrees with our segmentation centroid by **72 cm**, and the disagreement is a
**rotation error of 175.63°** — the object is essentially flipped.

**The port itself is validated.** Upstream's own `demo_data/mustard0`, through the
identical code path, returns a correct pose — its origin sits **2.06 cm** behind the
measured surface against a 9.6 cm half-depth. So the error belongs to our input, not to
the model: a 695 px mask over a one-sided Poisson shell, against mustard0's 3,252 px over
a CAD mesh. See [RESULTS.md](RESULTS.md) and `bug_log.txt` [5].

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

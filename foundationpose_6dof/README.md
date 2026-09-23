# FoundationPose — 6-DoF object pose

Upstream [FoundationPose](https://github.com/NVlabs/FoundationPose) pinned at `a1b694b`.
Model-based pose estimation and tracking: `register(K, rgb, depth, ob_mask, mesh)` then
`track_one(...)`.

**→ [RESULTS.md](RESULTS.md)** — checkpoint fidelity, and what runs on a T4
**→ [Plan.md](Plan.md)** — the one thing that has not happened yet

## The limit, up front

**`register()` has never returned a pose.** Both official checkpoints load exactly and
the model constructs on a Tesla T4, but no pose result exists. Nothing in this directory
should be read as a pose-accuracy claim.

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

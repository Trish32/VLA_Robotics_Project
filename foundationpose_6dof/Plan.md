# FoundationPose — plan

**The binding gap: `register()` has never returned a pose.**

Everything upstream of it is done — nvdiffrast builds and rasterises on a T4, the import
chain resolves, both checkpoints load and construct on `cuda:0`, and the mesh + mask
bundle is built from our own TSDF (9,657 triangles from 1,142 instance points, 8 frames
with occlusion-tested masks).

## How it gets judged when it runs

Two checks, neither of which needs ground truth:

1. **Agreement** — `register()`'s translation against what our own segmentation believed
   the centroid was. These are independent estimates; agreement is evidence, not proof,
   and disagreement does not say which is wrong.
2. **World-frame spread** — the tracked pose composed with each camera pose. A static
   object fused into a world frame must not move. This is the check that actually
   discriminates, because a drifting tracker cannot satisfy it by luck.

## Known weakness going in

The target instance is 1.58 × 1.18 × 0.99 m — a coarse region, not a clean object — and
its masks are 655–954 px. **A large spread would more likely indict the mesh than the
tracker**, so a bad result needs the mesh ruled out before it says anything about
FoundationPose.

Instance density is the lever: a denser fusion (smaller `--stride` in
`pipeline/tools/e2e_tum.py`) at TSDF memory cost.

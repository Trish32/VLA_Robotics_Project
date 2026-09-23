# DROID-SLAM — inference reproduction

Reference: https://github.com/princeton-vl/DROID-SLAM. Checkpoint: `droid.pth`.
Scope decision: **inference reproduction only.** Training is 4x RTX-3090 for ~1 week on
TartanAir and is not reachable on a free tier, so we do not pretend to.

This is the largest genuine port in the repo. Everything else here is a model; this is a model
plus a solver.


## What the compiled dependencies are (kept for reference, not as a plan)

The official system is PyTorch plus three compiled dependencies:

**1. `lietorch`** — Lie group tensors with custom autograd in the tangent space.
- SE3 and Sim3: `exp`/`log`, `Adj`, group composition, action on points, inverse.
- The gradients are the hard part: backprop happens in the tangent space, not on the raw
  parameterisation. A naive quaternion-with-autograd version will run and will be subtly wrong.
- Watch the small-angle regimes — `sin(θ)/θ` style terms need Taylor branches or they produce
  NaNs at identity, which is exactly where a SLAM system spends a lot of its time.

**2. `droid_backends`** — the CUDA kernels:
- correlation volume construction and lookup,
- projective transform / reprojection,
- depth filtering,
- **`ba`: the dense bundle adjustment layer.** Gauss-Newton over the factor graph, solved with a
  block-sparse Schur complement (eliminate per-pixel inverse depths, solve the reduced camera
  system, back-substitute). This is the centrepiece and the thing most likely to eat a week.

**3. `torch_scatter`** — already solved once. Reuse the approach from `QCNet_vl/utils/pyg_compat.py`
(`index_add`/`scatter_reduce`), and reuse the lesson from that project's one root-cause bug:
`0 * inf = NaN` silently wiped every off-diagonal entry. Test each primitive in isolation.

## Local setup

Upstream pinned at `2dfd39f`. `DroidNet()` builds on Mac (4.00M params) with no CUDA. The
three compiled deps are handled per CLAUDE.md, and the split is deliberate —
[compat.py](compat.py):

- **`torch_scatter` is substituted.** `scatter_mean`/`scatter_sum` are index arithmetic,
  reproduced exactly by `index_add_`; same approach already validated in
  `QCNet_vl/utils/pyg_compat.py`.
- **`lietorch` and `droid_backends` are NOT.** They resolve to objects that import
  cleanly and **raise on first use**. SE3/Sim3 tangent-space autograd and the dense-BA
  Schur solve are the real algorithms; a stub returning plausible shapes would silently
  corrupt every pose.

**→ [RESULTS.md](RESULTS.md)** — what is verified
**→ [Plan.md](Plan.md)** — why tracking has never run

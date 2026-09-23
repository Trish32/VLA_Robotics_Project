# OpenMask3D — open-vocabulary 3D instance segmentation

Mask3D proposes **where** instances are; CLIP says **what** they are. Upstream
[OpenMask3D](https://github.com/OpenMask3D/openmask3d) pinned at `3bc3fc5`, Mask3D
backbone `Res16UNet34C`.

**This runs without MinkowskiEngine.** Upstream's sparse convolution is a CUDA extension
that does not build on Apple Silicon, and on a current CUDA-12.8 image it needs patches
to both its build system and a bundled header it does not maintain (measured — see
[RESULTS.md](RESULTS.md)). `minkowski_compat.py` replaces it in pure PyTorch, and
`me_shim.py` registers it under the name upstream imports so no vendored source changes.

**→ [RESULTS.md](RESULTS.md)** — checkpoint fidelity, the 1e-10 verification, acceleration
**→ [Plan.md](Plan.md)** — what is not measured, and why

## The limit, up front

**No mIoU or open-vocabulary recall has been measured.** The checkpoint loads exactly and
the convolution is verified two independent ways, but "does it segment correctly" needs
ground-truth instance labels, and ScanNet200 is a gated ~1 TB download. Everything here
is a *fidelity* result, not an accuracy result, and the two are not interchangeable.

## Layout

```
minkowski_compat.py   pure-PyTorch sparse conv: hashing, gather/scatter, kernel offsets
me_shim.py            registers it as `MinkowskiEngine` so upstream imports unchanged
fast_features.py      view-major SAM+CLIP feature extraction (see RESULTS §2.3)
tools/
  verify_kernel_order.py   compiles upstream's kernel_region.hpp and diffs 195 offsets
  eval_instances.py        Mask3D inference + scoring + DBSCAN
  load_checkpoint.py       strict load, reports missing/unexpected/shape-mismatch
```

`tests/` — 39 tests, CPU only.

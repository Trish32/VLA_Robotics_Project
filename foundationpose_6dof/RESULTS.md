# FoundationPose — results

## Checkpoint fidelity · MEASURED

| model | tensors | params | strict load |
|---|---|---|---|
| refiner `2023-10-28-18-33-37` | — | 16.83 M | **0 / 0 / 0** |
| scorer `2024-01-11-20-02-45` | 116 | 15.77 M | **0 / 0 / 0** |

## The model runs on a T4 · MEASURED (Kaggle T4)

Both predictors construct on `cuda:0` with the official weights loaded, and the full
import chain resolves — `nvdiffrast`, `pytorch3d`, `trimesh`, `open3d`, `Utils`,
`estimater`.

Six rounds got there, and **five were dependency resolution, not FoundationPose**. Worth
recording because the pattern repeats: NumPy was the fault line three separate times in
three different disguises.

| round | cause |
|---|---|
| 2 | nvdiffrast fails inside pip's build isolation — `setup.py` imports torch, isolation hides it |
| 3 | pinned `numpy<2` → trimesh ABI break (*"dtype size changed, Expected 96, got 88"*). Self-inflicted |
| 4 | `open3d` simply absent — fixed by enumerating deps with `ast` instead of one round per missing import |
| 5 | something moved numpy again → `'numpy.ufunc' has no attribute '__qualname__'`, the pure-Python half of the same mismatch |
| 6 | **guard instead of reason**: record the image's numpy, install, restore if anything moved it |

`pytorch3d` is installed **CPU-only** on purpose: the model-based path rasterises with
nvdiffrast and never calls pytorch3d's renderer, so a CUDA build would spend the session
compiling kernels nothing calls. That hypothesis held.

## register() + track_one() on our own mesh · MEASURED (Kaggle T4)

No CAD model: the mesh is cut from our TSDF and `ob_mask` is the instance projected back
into the frame (`pipeline/tools/build_pose_bundle.py`). Target `chair_4`, mesh 4,961
verts / 9,657 faces, extents 1.573 × 1.168 × 0.986 m.

| check | result |
|---|---|
| `register()` | **returns a pose** — t = [0.567, 0.424, 2.730] m |
| `track_one()` across 8 frames | all succeed |
| **world-frame spread** | **2.08 cm** |
| agreement with our segmentation centroid | **72.14 cm** apart |

**The spread is the result that means something.** Composing each tracked pose with its
camera pose puts the object in the world frame, and a static object must not move there.
Over 8 frames it moves **2.08 cm** — so the tracker is strongly self-consistent, and that
check needed no ground truth to be meaningful.

**The 72 cm disagreement is not explained, and is not evidence either way.** Two
independent estimates of the same object differ: `register()` places it at z = 2.73 m,
our segmentation centroid at z = 2.05 m — most of the gap is in depth. Neither is ground
truth, so this does not say which is wrong. A hypothesis worth testing rather than
asserting: the mask is **695 px**, while a 1.58 m object at 2 m under fx = 535 should
subtend roughly 410 px across — so the mask is a sparse partial view, and the two methods
may be centring on different subsets of the object. Until that is checked, **no pose
accuracy is claimed**.

## The bug that blocked this for three rounds

`AttributeError: module 'mycpp' has no attribute 'cluster_poses'`, raised ~100 lines past
the guard meant to catch it. Root cause, every link verified:

1. `cluster_poses` exists **only in compiled C++** (`pybind_api.cpp:76`, bound via pybind11).
2. `mycpp/` was never built.
3. `mycpp/` has **no `__init__.py`**, so it is an implicit namespace package: `import
   mycpp` returns an empty module with `__file__ = None` instead of raising.
4. Upstream's `try: import mycpp / except: mycpp = None` therefore **never fires** — the
   import succeeded.

Reproduced locally before fixing. A guard that looks like it handles a missing dependency,
defeated by PEP 420. The kernel now builds `mycpp` (Boost, Eigen3, pybind11, OpenMP) and
asserts `cluster_poses` is present, printing `mycpp.__file__`, so a namespace-package
resolution fails at the check rather than somewhere unrelated.

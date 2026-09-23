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
| **world-frame spread** (self-consistency) | **2.08 cm** (3.13 cm on the r13 rerun) |
| agreement with our segmentation centroid | **72.14 cm** apart (median 73.02 cm over 8 frames) |
| **rotation error** vs `R_world_to_cam` | **175.63°** — measured directly |
| scorer range over 252 rotation hypotheses | **1.75** (std 0.196); **48/252 within 1% of the top** |
| frames passing the stage-4 gate | **0 / 8** |

**The spread means less than it appears to.** Composing each tracked pose with its camera
pose puts the object in the world frame, where a static object must not move; over 8
frames it moves **2.08 cm**. That is a *consistency* check. A tracker locked onto a wrong
pose holds it exactly as steadily as one locked onto the right pose, so this bounds drift
and says nothing about accuracy.

**The disagreement is a rotation error, not a translation one.** The origin probe first
put a lower bound on it: re-centring the mesh moved its origin a known **52.9 cm**, and
FoundationPose's answer moved **53.5 cm** — right magnitude, so the pose convention is
understood — but **99.5° away** from the direction our map predicts. Measuring the
rotation directly against `R_world_to_cam` (the mesh is cut from the world cloud
unrotated, so its frame *is* the world frame up to translation) gives **175.63°**,
consistent with that bound and close enough to 180° to name the failure: the pose is
essentially flipped. The 72 cm translation residual is what that looks like measured at a
point offset from the rotation centre.

## The control: the port is correct, our bundle is not

Upstream's own `demo_data/mustard0`, through the identical code path, same weights, same
252 hypotheses — only the mesh and mask differ:

| | ours (`chair_4`) | mustard0 |
|---|---|---|
| mesh | 4,961 verts, Poisson over a one-sided shell | 10,983 verts, CAD |
| mask | **695 px** | **3,252 px** |
| origin vs measured surface | **60.8 cm** behind (surface is 13.4 cm deep) | **+2.06 cm** behind (half-depth 9.6 cm) |
| pose correct? | **no** — 175.63° | **yes** |

**That answers the Fidelity Rule question.** FoundationPose, both checkpoints, `mycpp`,
nvdiffrast and our adaptation layer all produce a correct pose on upstream's data. The
175.63° error belongs to *our input*, not to the port.

### A metric that did not survive its own control

An earlier revision of this page reported the scorer's spread — 1.75 points end to end,
48/252 within 1% of the top — and read the flat top as "the orientation is
unidentifiable". **The control refutes that reading.** mustard0, which returns a *correct*
pose, is flatter still:

| | ours | mustard0 |
|---|---|---|
| margin over top-2 | 0.50 | **0.0000** (exact tie) |
| within 1% of top | 48/252 | **91/252** |

Scores are computed *after* refinement, so hypotheses that converge onto the same pose
legitimately tie. A flat top means **agreement**, not ambiguity — the opposite of what was
claimed. The discriminating quantity is whether the refined **poses** cluster, not whether
their **scores** do; r14 measures pairwise geodesic angle among the top-16 instead.

*(An earlier revision of this page attributed the 72 cm to `reset_object` subtracting the
mesh's bbox centre. `estimater.py:233` undoes that subtraction before returning, so the
convention was never the cause — see `bug_log.txt` [4]. The failed fix is what produced
the rotation measurement, so it is recorded rather than removed.)*

**Why this is plausible but not yet attributed.** The mask is **695 px**, while a 1.58 m
object at 2 m under fx = 535 should subtend roughly 410 px across — a one-sided Poisson
shell seen through a sparse partial view carries little signal to fix an orientation. That
is a hypothesis about *our input*, and the test that separates it from a broken port is
upstream's own `demo_data/mustard0` (kernel r12). Until it runs, **no pose accuracy is
claimed**, and `pipeline/tools/e2e_pose.py` refuses to admit the pose into the world
model.

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

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

## Not a pose result

See [Plan.md](Plan.md). `register()` has not run.

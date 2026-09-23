#!/usr/bin/env python
"""Settle the one claim `minkowski_compat.py` cannot check locally: kernel offset ORDER.

Everything else about the replacement is already pinned on CPU — dense-grid equivalence
against nn.Conv3d at 1e-10, coordinate generation, the encoder/decoder map reuse, and a
0/0/0 strict load of the released Mask3D checkpoint. None of those can catch an ordering
error, because shapes are order-independent and the dense oracle builds its reference
weight from the same offsets it is testing.

So this runs the ONLY test that can: hand the SAME weight tensor to real MinkowskiEngine
and to ours, on the SAME coordinates, and compare outputs. If our enumeration disagrees
with `kernel_region.hpp` the kernel planes are permuted and the outputs diverge.

Three regimes, because the conventions differ across them:
  k=3 stride 1   odd kernel, centered offsets
  k=2 stride 2   EVEN kernel, offsets start at 0 rather than centering
  k=5 stride 1   the stem, and a wider spread to catch an ordering error that a 3^3
                 kernel might alias

Run on any CUDA box:
    python openmask3d_semantic/tools/verify_on_cuda.py

MinkowskiEngine install (its last release predates modern torch):
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \\
        --no-deps --install-option="--blas=openblas"
If it will not build, that is itself worth recording — it is the reason this module
exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOL = 1e-4          # fp32 accumulation over a 125-tap kernel; anything real is far larger


def _require():
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA. This script exists precisely because the comparison cannot be made "
            "without real MinkowskiEngine, which has no CPU build."
        )
    try:
        import MinkowskiEngine  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"MinkowskiEngine not importable: {exc}\n"
            "See the install line in this file's docstring. Do NOT let our shim satisfy "
            "this import — the whole point is to compare against the real thing."
        ) from exc


def random_sparse(n_points=4000, channels=8, span=48, seed=0, device="cuda"):
    rng = np.random.default_rng(seed)
    coords = rng.integers(-span, span, size=(n_points, 3))
    coords = np.unique(coords, axis=0)
    batched = np.column_stack([np.zeros(len(coords), dtype=np.int64), coords])
    feats = rng.standard_normal((len(coords), channels)).astype(np.float32)
    return (torch.as_tensor(batched, dtype=torch.int32, device=device),
            torch.as_tensor(feats, device=device))


def compare(kernel_size: int, stride: int, c_in=8, c_out=6) -> tuple[float, int]:
    """Max |ours - ME| over the shared coordinates, and how many coords were shared."""
    import MinkowskiEngine as ME

    from openmask3d_semantic import minkowski_compat as mc

    coords, feats = random_sparse(channels=c_in)

    me_tensor = ME.SparseTensor(features=feats, coordinates=coords)
    me_conv = ME.MinkowskiConvolution(
        c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False, dimension=3
    ).cuda()
    me_out = me_conv(me_tensor)

    # The SAME weights, not a fresh random draw: this compares the OPERATION.
    weight = me_conv.kernel.detach().clone()

    ours_in = mc.SparseTensor(feats.cpu().double(), coords.cpu(), (1, 1, 1))
    ours_out = mc.sparse_conv(
        ours_in, weight.cpu().double(), None, kernel_size, stride=stride
    )

    # ME does not guarantee row order, so align on coordinates before comparing.
    me_coords = me_out.C.cpu().numpy()
    me_feats = me_out.F.detach().cpu().numpy()
    our_coords = ours_out.coordinates.numpy()
    our_feats = ours_out.features.numpy()

    key = lambda a: [tuple(row) for row in a]  # noqa: E731
    lookup = {k: i for i, k in enumerate(key(our_coords))}
    shared = [(i, lookup[k]) for i, k in enumerate(key(me_coords)) if k in lookup]
    if not shared:
        return float("inf"), 0

    mine = np.stack([our_feats[j] for _, j in shared])
    theirs = np.stack([me_feats[i] for i, _ in shared])
    return float(np.abs(mine - theirs).max()), len(shared)


def main() -> int:
    _require()
    import MinkowskiEngine as ME

    print(f"torch {torch.__version__} | ME {ME.__version__} | "
          f"{torch.cuda.get_device_name(0)}\n")

    cases = [
        ("k=3 s=1  odd kernel, centered", 3, 1),
        ("k=2 s=2  EVEN kernel, not centered", 2, 2),
        ("k=5 s=1  stem width", 5, 1),
        ("k=3 s=2  odd kernel, strided", 3, 2),
    ]

    worst = 0.0
    print(f"{'case':<38}{'shared coords':>14}{'max |diff|':>14}  verdict")
    print("-" * 80)
    for label, k, s in cases:
        diff, shared = compare(k, s)
        worst = max(worst, diff)
        verdict = "MATCH" if diff < TOL else "MISMATCH"
        print(f"{label:<38}{shared:>14}{diff:>14.3e}  {verdict}")
    print("-" * 80)

    if worst < TOL:
        print(f"\nPASS — max deviation {worst:.3e} < {TOL:.0e}.")
        print("Kernel offset ORDER and the correlation convention "
              "`out[p] = sum_k x[p + o_k] @ W[k]` are confirmed against upstream.")
        print("bug_log.txt [2] can move from CLOUD-VERIFIED-PENDING to verified.")
        return 0

    print(f"\nFAIL — max deviation {worst:.3e}.")
    print("Most likely cause, in order: (1) offset enumeration order — ME increments the")
    print("FIRST axis fastest; (2) even-kernel centering; (3) the correlation sign, i.e.")
    print("x[p + o_k] versus x[p - o_k]. Diff one kernel plane at a time to localise:")
    print("a single-tap weight (all planes zero but one) isolates which offset is which.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

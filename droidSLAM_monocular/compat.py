"""Local-dev shims for DROID-SLAM's compiled dependencies.

CLAUDE.md draws a hard line here, and the two halves of this file sit on opposite sides
of it.

**Reimplemented, because they are exactly reproducible.** `torch_scatter`'s
`scatter_mean` / `scatter_sum` are index arithmetic, not CUDA algorithms; `index_add_`
gives bit-comparable results and is testable against the real package. This is the same
substitution already validated in `~/VLMProjects/QCNet_vl/utils/pyg_compat.py`, including
its one root-cause bug (`0 * inf = NaN` silently wiping every off-diagonal entry) — hence
the explicit non-finite test in tests/.

**NOT reimplemented, because faking them would poison every downstream result.**
`lietorch` (SE3/Sim3 with tangent-space autograd) and `droid_backends` (correlation
volumes, projective transforms, and the dense bundle-adjustment Schur solve) are the
system's actual algorithms. A stub that returned zeros would pass every shape test and
make the trajectory silently wrong. So `missing_extension()` builds an object that
imports fine and **raises on first use**, which is what CLAUDE.md requires: "If it cannot
run locally, it must raise locally."

This lets `droid_net.py` import on a Mac — enough to construct `DroidNet` and check the
checkpoint loads 0/0, since neither extension is touched by `__init__`.
"""

from __future__ import annotations

from typing import Any

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                dim_size: int | None = None) -> torch.Tensor:
    """torch_scatter.scatter_sum via index_add_.

    `index` is 1-D over `dim`, matching how DROID-SLAM calls it.
    """
    if index.dim() != 1:
        raise ValueError(f"expected a 1-D index, got shape {tuple(index.shape)}")
    dim = dim % src.dim()
    size = list(src.shape)
    size[dim] = int(index.max()) + 1 if dim_size is None else dim_size
    out = src.new_zeros(size)
    return out.index_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 dim_size: int | None = None) -> torch.Tensor:
    """torch_scatter.scatter_mean.

    Empty slots divide by a clamped count so they yield 0 rather than NaN — matching
    torch_scatter, and avoiding the QCNet failure mode where a NaN from an empty group
    propagated through the whole graph.
    """
    summed = scatter_sum(src, index, dim, dim_size)
    ones = src.new_ones(src.shape[dim % src.dim()])
    counts = scatter_sum(ones, index, 0, summed.shape[dim % src.dim()])
    shape = [1] * summed.dim()
    shape[dim % summed.dim()] = -1
    return summed / counts.clamp(min=1).reshape(shape)


class _MissingExtension:
    """Imports cleanly; raises with a useful message the moment it is touched."""

    def __init__(self, name: str, why: str) -> None:
        self._name = name
        self._why = why

    def _raise(self, attr: str = "") -> Any:
        raise NotImplementedError(
            f"{self._name}{'.' + attr if attr else ''} is unavailable here. {self._why}\n"
            "It is deliberately NOT stubbed: a fake would return plausible shapes and "
            "silently corrupt every pose downstream. Run this on a CUDA machine."
        )

    def __getattr__(self, attr: str) -> Any:
        def _fail(*_args: Any, **_kwargs: Any) -> Any:
            return self._raise(attr)

        return _fail

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._raise()


def missing_extension(name: str, why: str) -> _MissingExtension:
    return _MissingExtension(name, why)


LIETORCH_WHY = (
    "lietorch compiles CUDA kernels for SE3/Sim3 tangent-space autograd and has no "
    "CPU-only or Apple-Silicon build."
)
DROID_BACKENDS_WHY = (
    "droid_backends compiles the correlation volume, projective transform and dense "
    "bundle-adjustment CUDA kernels; there is no CPU build."
)


def try_import_lietorch():
    """Real lietorch if present, otherwise a raise-on-use placeholder."""
    try:
        import lietorch  # noqa: F401

        return lietorch, True
    except ImportError:
        return missing_extension("lietorch", LIETORCH_WHY), False


def try_import_droid_backends():
    try:
        import droid_backends  # noqa: F401

        return droid_backends, True
    except ImportError:
        return missing_extension("droid_backends", DROID_BACKENDS_WHY), False


def try_import_scatter():
    """Real torch_scatter if present, else our equivalents.

    Unlike the extensions above these ARE substituted, because they are reproducible
    exactly — see the module docstring.
    """
    try:
        from torch_scatter import scatter_mean as sm, scatter_sum as ss

        return sm, ss, True
    except ImportError:
        return scatter_mean, scatter_sum, False

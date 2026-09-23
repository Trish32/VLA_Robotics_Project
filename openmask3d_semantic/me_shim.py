"""Make `import MinkowskiEngine as ME` resolve to `minkowski_compat`.

Mask3D spreads its Minkowski imports across seven files in four spellings — `import
MinkowskiEngine as ME`, `from MinkowskiEngine import MinkowskiReLU`, `import
MinkowskiEngine.MinkowskiOps as me`, `from MinkowskiEngine.MinkowskiPooling import
MinkowskiAvgPooling`. Editing all of them would be a wide patch over vendored source for
no benefit, so this registers synthetic modules in `sys.modules` instead and upstream
stays byte-identical to the pinned commit. Same approach as
`foundationpose_6dof/compat.py`, and it keeps the Debug Rule's activation diff available.

Two adapters earn their keep:

`SparseTensor(...)` — ME's constructor is keyword-heavy (`features=`, `coordinates=`,
`device=`, `coordinate_manager=`, `coordinate_map_key=`) and Mask3D uses several of
those spellings. Ours is a dataclass, so the call is translated rather than matched.

`KernelGenerator(...)` — accepted and IGNORED, on purpose. Upstream builds one for every
convolution, but `modules/common.py` passes `axis_types=None` unconditionally (there is a
`# JONAS` comment where it was forced), so every kernel in this network is a plain
HYPER_CUBE. Storing the object would imply we honour region types we do not implement.
Anything other than HYPER_CUBE raises instead of being silently ignored.
"""

from __future__ import annotations

import enum
import sys
import types

import torch

from openmask3d_semantic import minkowski_compat as mc


class RegionType(enum.IntEnum):
    """Values match ME's enum; only HYPER_CUBE is implemented."""

    HYPER_CUBE = 0
    HYPER_CROSS = 1
    CUSTOM = 2


class KernelGenerator:
    """Constructible, inspectable, and deliberately not consulted by the convolutions."""

    def __init__(self, kernel_size=3, stride=1, dilation=1, region_type=RegionType.HYPER_CUBE,
                 axis_types=None, dimension=3, **_ignored) -> None:
        if region_type != RegionType.HYPER_CUBE:
            raise NotImplementedError(
                f"region_type {region_type!r} is not implemented. Res16UNet34C only ever "
                "builds HYPER_CUBE kernels (modules/common.py forces axis_types=None), "
                "so supporting anything else would be untested code pretending to work."
            )
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.dimension, self.region_type = dimension, region_type


def SparseTensor(features=None, coordinates=None, tensor_stride=1,  # noqa: N802
                 coordinate_manager=None, coordinate_map_key=None, device=None,
                 **_ignored) -> mc.SparseTensor:
    """ME's keyword-heavy constructor, translated onto our dataclass.

    Two construction paths, and the second is why this is not a one-liner:

      explicit coordinates  the normal case.
      map key + manager     `Mask3D.forward` builds a tensor of raw coordinates over the
                            coordinate map the BACKBONE already created, so the two are
                            guaranteed to align. Resolving the key against the manager
                            reuses those exact coordinates.

    A missing key is still refused. Guessing which map was meant would produce a tensor
    whose features and coordinates describe different points — well-formed, and wrong.
    """
    if features is None:
        raise ValueError("SparseTensor needs features")

    if coordinates is None:
        if coordinate_manager is None or coordinate_map_key is None:
            raise ValueError(
                "SparseTensor without coordinates needs BOTH a coordinate_manager and a "
                "coordinate_map_key. Guessing the map would pair these features with "
                "the wrong points."
            )
        resolved = coordinate_manager.get(tuple(coordinate_map_key))
        if resolved is None:
            raise ValueError(
                f"no coordinate map registered at stride {tuple(coordinate_map_key)}; "
                f"manager holds {sorted(coordinate_manager._maps)}"
            )
        if len(resolved) != len(features):
            raise ValueError(
                f"map at stride {tuple(coordinate_map_key)} has {len(resolved)} points "
                f"but {len(features)} features were supplied"
            )
        coordinates = resolved
        tensor_stride = tuple(coordinate_map_key)
    if device is not None:
        features = features.to(device)
        coordinates = coordinates.to(device)
    tensor = mc.SparseTensor(
        features, coordinates, tensor_stride,
        coordinate_manager or mc.CoordinateManager(),
    )
    return tensor


class MinkowskiNetwork(torch.nn.Module):
    """ME's base class: an nn.Module that remembers its spatial dimension."""

    def __init__(self, D: int) -> None:  # noqa: N803 -- ME's spelling
        super().__init__()
        self.D = D


class MinkowskiBroadcastMultiplication(torch.nn.Module):
    """Per-batch scaling, used by the SE blocks Res16UNet34C does not instantiate."""

    def forward(self, x: mc.SparseTensor, y: mc.SparseTensor) -> mc.SparseTensor:
        batch = x.coordinates[:, 0].long()
        lookup = {int(b): i for i, b in enumerate(y.coordinates[:, 0].tolist())}
        rows = torch.tensor([lookup[int(b)] for b in batch], device=x.features.device)
        return x.like(x.features * y.features[rows])


# `MinkowskiAvgUnpooling` and `MinkowskiPoolingTranspose` are ME's names for the same
# upsampling op; both appear in resunet.py, which Res16UNet34C does not use.
MinkowskiAvgUnpooling = mc.MinkowskiAvgPooling
MinkowskiPoolingTranspose = mc.MinkowskiAvgPooling

_EXPORTS = {
    "SparseTensor": SparseTensor,
    "RegionType": RegionType,
    "KernelGenerator": KernelGenerator,
    "MinkowskiNetwork": MinkowskiNetwork,
    "MinkowskiConvolution": mc.MinkowskiConvolution,
    "MinkowskiConvolutionTranspose": mc.MinkowskiConvolutionTranspose,
    "MinkowskiBatchNorm": mc.MinkowskiBatchNorm,
    "MinkowskiInstanceNorm": mc.MinkowskiInstanceNorm,
    "MinkowskiReLU": mc.MinkowskiReLU,
    "MinkowskiSigmoid": mc.MinkowskiSigmoid,
    "MinkowskiLinear": mc.MinkowskiLinear,
    "MinkowskiAvgPooling": mc.MinkowskiAvgPooling,
    "MinkowskiSumPooling": mc.MinkowskiSumPooling,
    "MinkowskiGlobalPooling": mc.MinkowskiGlobalPooling,
    "MinkowskiAvgUnpooling": MinkowskiAvgUnpooling,
    "MinkowskiPoolingTranspose": MinkowskiPoolingTranspose,
    "MinkowskiBroadcastMultiplication": MinkowskiBroadcastMultiplication,
    "cat": mc.cat,
}


def _module(name: str, contents: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(contents)
    return module


def install() -> types.ModuleType:
    """Register the synthetic package tree. Idempotent; returns the root module."""
    existing = sys.modules.get("MinkowskiEngine")
    if existing is not None and getattr(existing, "__vldrive_shim__", False):
        return existing
    if existing is not None:
        raise RuntimeError(
            "the real MinkowskiEngine is already imported. Refusing to shadow it — if "
            "it works in this environment, use it and diff our implementation against "
            "it instead (that diff is the cloud-verification bug_log.txt [2] is waiting "
            "on)."
        )

    root = _module("MinkowskiEngine", _EXPORTS)
    root.__vldrive_shim__ = True
    root.__path__ = []

    ops = _module("MinkowskiEngine.MinkowskiOps", {"cat": mc.cat, "SparseTensor": SparseTensor})
    pooling = _module("MinkowskiEngine.MinkowskiPooling", {
        "MinkowskiAvgPooling": mc.MinkowskiAvgPooling,
        "MinkowskiSumPooling": mc.MinkowskiSumPooling,
        "MinkowskiAvgUnpooling": MinkowskiAvgUnpooling,
        "MinkowskiPoolingTranspose": MinkowskiPoolingTranspose,
    })
    # `ME.utils.*` is dataloader-side quantisation. It is NOT implemented here: Mask3D
    # calls it from its ScanNet dataset pipeline, which this project does not use, and a
    # wrong sparse_quantize would silently change which points exist.
    utils = _module("MinkowskiEngine.utils", {
        "sparse_quantize": _unimplemented("utils.sparse_quantize"),
        "sparse_collate": _unimplemented("utils.sparse_collate"),
        "batched_coordinates": _unimplemented("utils.batched_coordinates"),
    })

    root.MinkowskiOps, root.MinkowskiPooling, root.utils = ops, pooling, utils
    sys.modules.update({
        "MinkowskiEngine": root,
        "MinkowskiEngine.MinkowskiOps": ops,
        "MinkowskiEngine.MinkowskiPooling": pooling,
        "MinkowskiEngine.utils": utils,
    })
    return root


# ------------------------------------------------------------------ torch_scatter
#
# `models/mask3d.py` imports scatter_mean / scatter_max / scatter_min to pool point
# features onto supervoxel segments. torch-scatter is a compiled extension pinned to a
# torch/CUDA pair, and these three are index arithmetic, so they are substituted for the
# same reason `droidSLAM_monocular/compat.py` substitutes the same package — and with
# the same warning attached, because that port's one root-caused bug was a scatter
# returning NaN for empty groups (`0 * inf`) which silently wiped every off-diagonal.


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcast a 1-D group index across the non-scatter dimensions of `src`."""
    if index.dim() != 1:
        raise ValueError(f"expected a 1-D index, got shape {tuple(index.shape)}")
    shape = [1] * src.dim()
    shape[dim] = -1
    return index.view(shape).expand_as(src)


def _out_size(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int | None):
    size = list(src.shape)
    size[dim] = int(index.max()) + 1 if dim_size is None else dim_size
    return size


def scatter_sum(src, index, dim: int = -1, dim_size: int | None = None):
    dim = dim % src.dim()
    out = src.new_zeros(_out_size(src, index, dim, dim_size))
    return out.scatter_add_(dim, _expand_index(index, src, dim), src)


def scatter_mean(src, index, dim: int = -1, dim_size: int | None = None):
    """Empty groups yield 0, not NaN — the QCNet failure, guarded by a clamped count."""
    dim = dim % src.dim()
    summed = scatter_sum(src, index, dim, dim_size)
    ones = src.new_ones(src.shape[dim])
    counts = scatter_sum(ones, index, 0, summed.shape[dim])
    shape = [1] * summed.dim()
    shape[dim] = -1
    return summed / counts.clamp(min=1).reshape(shape)


def _scatter_extreme(src, index, dim, dim_size, reduce: str):
    """Returns (values, argmax/argmin) like torch_scatter.

    Empty groups are filled with 0 and an index of -1, matching torch_scatter's
    behaviour for an unwritten slot. Ties resolve to the LOWEST source position; real
    torch_scatter does not document a tie rule, and Mask3D only consumes `[0]`, so this
    is stated rather than assumed.
    """
    dim = dim % src.dim()
    size = _out_size(src, index, dim, dim_size)
    idx = _expand_index(index, src, dim)
    fill = float("-inf") if reduce == "amax" else float("inf")

    values = src.new_full(size, fill)
    values.scatter_reduce_(dim, idx, src, reduce=reduce, include_self=True)
    empty = torch.isinf(values)
    values = values.masked_fill(empty, 0.0)

    gathered = values.gather(dim, idx)
    positions = torch.arange(src.shape[dim], device=src.device)
    positions = _expand_index(positions, src, dim).clone()
    positions = positions.masked_fill(src != gathered, src.shape[dim])
    arg = src.new_full(size, src.shape[dim], dtype=torch.long)
    arg.scatter_reduce_(dim, idx, positions, reduce="amin", include_self=True)
    arg = arg.masked_fill(arg == src.shape[dim], -1)
    return values, arg


def scatter_max(src, index, dim: int = -1, dim_size: int | None = None):
    return _scatter_extreme(src, index, dim, dim_size, "amax")


def scatter_min(src, index, dim: int = -1, dim_size: int | None = None):
    return _scatter_extreme(src, index, dim, dim_size, "amin")


def install_scatter() -> types.ModuleType:
    """Register a synthetic `torch_scatter`, unless the real package is importable."""
    existing = sys.modules.get("torch_scatter")
    if existing is not None:
        return existing
    try:
        import torch_scatter  # noqa: F401

        return sys.modules["torch_scatter"]
    except ImportError:
        pass

    module = _module("torch_scatter", {
        "scatter_sum": scatter_sum, "scatter_mean": scatter_mean,
        "scatter_max": scatter_max, "scatter_min": scatter_min,
    })
    module.__vldrive_shim__ = True
    sys.modules["torch_scatter"] = module
    return module


# -------------------------------------------------------------------- pointnet2
#
# `mask3d.py` imports `furthest_point_sample` from third_party/pointnet2, a compiled
# CUDA extension. Unlike the renderer in FoundationPose this one IS exactly
# reproducible: farthest-point sampling is deterministic greedy selection, so a PyTorch
# version is the same function rather than an approximation of it. Only this one symbol
# is provided — the rest of pointnet2 (ball query, grouping, interpolation) is not used
# by Mask3D and is left to raise.


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """(B, N, 3) -> (B, npoint) int32 indices, matching pointnet2's kernel.

    Two conventions are copied from the CUDA source rather than invented: the first
    selected index is always 0, and distances start at 1e10. Both are observable —
    Mask3D uses these indices to seed its non-parametric queries, so a different
    starting point produces a different (not wrong, but different) set of queries and
    would break an activation diff against upstream for reasons unrelated to the port.
    """
    if xyz.dim() != 3 or xyz.shape[2] != 3:
        raise ValueError(f"expected (B, N, 3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    if npoint > n:
        raise ValueError(f"cannot sample {npoint} points from {n}")

    indices = torch.zeros(b, npoint, dtype=torch.int32, device=xyz.device)
    closest = torch.full((b, n), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    rows = torch.arange(b, device=xyz.device)

    for step in range(npoint):
        indices[:, step] = farthest.to(torch.int32)
        centroid = xyz[rows, farthest].unsqueeze(1)
        closest = torch.minimum(closest, ((xyz - centroid) ** 2).sum(-1))
        farthest = closest.argmax(-1)
    return indices


def install_pointnet2() -> None:
    """Provide `third_party.pointnet2.pointnet2_utils.furthest_point_sample`.

    Registered under the module path Mask3D imports it from, so upstream's
    `from third_party.pointnet2.pointnet2_utils import furthest_point_sample` resolves
    without the vendored file's `import pointnet2._ext` ever running.
    """
    if sys.modules.get("third_party.pointnet2.pointnet2_utils") is not None:
        return

    utils = _module("third_party.pointnet2.pointnet2_utils", {
        "furthest_point_sample": furthest_point_sample,
        "gather_operation": _unimplemented("pointnet2_utils.gather_operation"),
        "ball_query": _unimplemented("pointnet2_utils.ball_query"),
        "three_nn": _unimplemented("pointnet2_utils.three_nn"),
        "three_interpolate": _unimplemented("pointnet2_utils.three_interpolate"),
    })
    utils.__vldrive_shim__ = True

    for name in ("third_party", "third_party.pointnet2"):
        if name not in sys.modules:
            package = _module(name, {})
            package.__path__ = []
            sys.modules[name] = package
    sys.modules["third_party.pointnet2"].pointnet2_utils = utils
    sys.modules["third_party.pointnet2.pointnet2_utils"] = utils


def install_all() -> None:
    """Everything Mask3D needs in order to import off-GPU, in one call."""
    install()
    install_scatter()
    install_pointnet2()


def _unimplemented(name: str):
    def _fail(*_args, **_kwargs):
        raise NotImplementedError(
            f"MinkowskiEngine.{name} is not implemented in the shim. It is dataloader-"
            "side voxelisation, used by Mask3D's ScanNet pipeline rather than by the "
            "network. Implementing it wrongly would change which points exist, which no "
            "downstream check would catch."
        )
    return _fail

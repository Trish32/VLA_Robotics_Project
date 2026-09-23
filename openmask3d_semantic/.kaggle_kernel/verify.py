"""Pure-PyTorch stand-in for the MinkowskiEngine ops Mask3D's Res16UNet34C needs.

CLAUDE.md says adapt upstream rather than rewrite it, and that rule holds for OpenMask3D
as a whole. MinkowskiEngine is the one exception, for a reason that is about packaging
rather than taste: it pins `torch==1.12.1+cu113`, needs `openblas-devel` and a matching
nvcc, and has no Apple-Silicon build at all. Installing it would fork this repo's torch
version and make the mask stage undevelopable on the Mac. The substitution is the one
already validated once in `~/VLMProjects/bevfusion_vl` against spconv: a coordinate hash
plus gather/matmul/scatter.

Scope is deliberately narrow — exactly what `Res16UNet34C` and `Mask3D` call, no more:
convolution and transposed convolution over `HYPER_CUBE` regions at D=3 with dilation 1,
batch norm, ReLU, average/sum pooling, and concatenation.

WHAT IS AND IS NOT VERIFIED HERE
--------------------------------
Locally verified (tests/test_minkowski_compat.py, CPU):
  * on a fully occupied grid the sparse convolution reduces EXACTLY to `nn.Conv3d`,
    which pins the hashing, the gather/scatter, the accumulation and the boundary
    behaviour against a real oracle;
  * coordinate generation, stride bookkeeping and the encoder/decoder coordinate reuse
    that `me.cat` depends on.

NOT verified locally — read out of upstream source, pinned as a literal, and needing a
numerical diff against real MinkowskiEngine on the GPU host before any metric from this
module is trusted:
  * the kernel OFFSET ORDER (see `kernel_offsets`). The dense-grid oracle above cannot
    catch an ordering error, because it builds its reference `Conv3d` weight from these
    same offsets. A wrong order permutes the checkpoint's kernel planes, loads 0 missing
    / 0 unexpected, and produces plausible garbage. This is the Swin `PatchMerging`
    failure from `bevfusion_vl` in a new costume.
  * the correlation convention: `out[p] = sum_k x[p + o_k] @ W[k]`.
Both are recorded in bug_log.txt as cloud-verified-pending.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def kernel_offsets(
    kernel_size: int | tuple[int, ...],
    dimension: int = 3,
    tensor_stride: tuple[int, ...] | int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """(K, D) integer offsets in ORIGINAL coordinate units, in MinkowskiEngine's order.

    Three conventions are load-bearing and all come from
    `MinkowskiEngine/src/kernel_region.hpp`:

    **Odd kernels are centered, even kernels are not.** `coordinate_at` computes
    `(idx - kernel_size / 2) * dilation * tensor_stride` for odd sizes but plain
    `idx * dilation * tensor_stride` for even ones. Res16UNet34C uses both — kernel 3
    inside the residual blocks, kernel 2 for every stride-2 down/upsample — so a
    uniformly centered implementation is wrong on exactly half the network.

    **The FIRST spatial dimension varies fastest.** The iterator increments `m_axis = 0`
    innermost and carries into higher axes. That is column-major, the opposite of what
    `torch.meshgrid(..., indexing="ij").reshape(-1, D)` produces. Getting this backwards
    transposes the kernel about its diagonal.

    **Offsets step by the INPUT tensor stride.** ME keeps coordinates in the original
    coordinate system scaled up by `tensor_stride`, so at stride 8 the neighbouring
    voxel is 8 away, not 1.
    """
    ks = (kernel_size,) * dimension if isinstance(kernel_size, int) else tuple(kernel_size)
    ts = (tensor_stride,) * dimension if isinstance(tensor_stride, int) else tuple(tensor_stride)
    if len(ks) != dimension or len(ts) != dimension:
        raise ValueError(f"kernel_size {ks} and tensor_stride {ts} must have {dimension} entries")

    per_axis = []
    for axis in range(dimension):
        k = ks[axis]
        # floor((k-1)/2) is ME's `center`; for even k the region starts at 0 instead.
        centre = k // 2 if k % 2 == 1 else 0
        per_axis.append([(i - centre) * dilation * ts[axis] for i in range(k)])

    # itertools.product varies its LAST argument fastest, so feed the axes reversed and
    # un-reverse each tuple to make axis 0 the fastest.
    rows = [tuple(reversed(combo)) for combo in itertools.product(*reversed(per_axis))]
    return torch.tensor(rows, dtype=torch.long)


# --------------------------------------------------------------------- coordinates


def _ravel(coords: torch.Tensor, origin: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    """(N, 1+D) int coordinates -> (N,) int64 keys, collision-free within `extent`."""
    shifted = (coords - origin).to(torch.int64)
    keys = shifted[:, 0]
    for axis in range(1, coords.shape[1]):
        keys = keys * extent[axis] + shifted[:, axis]
    return keys


class CoordinateManager:
    """Remembers the coordinate set at each tensor stride, as ME's does.

    This is not an optimisation — the UNet is incorrect without it. `convtr4p16s2`
    upsamples stride 16 -> 8 and the result is immediately `me.cat`-ed with the
    encoder's stride-8 tensor. That concatenation is only defined if the two coordinate
    sets are identical and identically ordered, which ME guarantees by having the
    transposed convolution reuse the existing stride-8 map rather than generate one.
    """

    def __init__(self) -> None:
        self._maps: dict[tuple[int, ...], torch.Tensor] = {}
        self._keys: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._origin: torch.Tensor | None = None
        self._extent: torch.Tensor | None = None

    def _bounds(self, coords: torch.Tensor) -> None:
        """Hash bounds are latched from the first (finest) coordinate set.

        Every later stride is a subset in the same coordinate system, and kernel offsets
        reach outside the occupied region by design, so the box is padded generously.
        """
        if self._origin is not None:
            return
        lo = coords.min(dim=0).values - 64
        hi = coords.max(dim=0).values + 64
        self._origin = lo
        self._extent = (hi - lo + 1).to(torch.int64)
        # math.prod over Python ints, NOT torch.prod: an int64 product of an oversized
        # box wraps silently, so the overflow guard would itself overflow and pass.
        if math.prod(self._extent.tolist()) > 2**62:
            raise ValueError(
                f"coordinate box {self._extent.tolist()} overflows an int64 hash key; "
                "the point cloud needs a coarser voxel_size"
            )

    def register(self, tensor_stride: tuple[int, ...], coords: torch.Tensor) -> torch.Tensor:
        key = tuple(tensor_stride)
        if key in self._maps:
            return self._maps[key]
        self._bounds(coords)
        self._maps[key] = coords
        raveled = _ravel(coords, self._origin, self._extent)
        order = torch.argsort(raveled)
        self._keys[key] = (raveled[order], order)
        return coords

    def get(self, tensor_stride: tuple[int, ...]) -> torch.Tensor | None:
        return self._maps.get(tuple(tensor_stride))

    def lookup(self, tensor_stride: tuple[int, ...], query: torch.Tensor):
        """(rows into the registered coords, hit mask) for `query` (M, 1+D)."""
        sorted_keys, order = self._keys[tuple(tensor_stride)]
        q = _ravel(query, self._origin, self._extent)
        if sorted_keys.numel() == 0:
            return torch.zeros_like(q), torch.zeros_like(q, dtype=torch.bool)
        idx = torch.searchsorted(sorted_keys, q).clamp(max=sorted_keys.numel() - 1)
        hit = sorted_keys[idx] == q
        return order[idx], hit


@dataclass
class SparseTensor:
    """Minimal stand-in for `ME.SparseTensor`: features, coordinates, tensor stride."""

    features: torch.Tensor                 # (N, C) float
    coordinates: torch.Tensor              # (N, 1+D) int, [batch, x, y, z]
    tensor_stride: tuple[int, ...] = (1, 1, 1)
    coordinate_manager: CoordinateManager = field(default_factory=CoordinateManager)

    def __post_init__(self) -> None:
        if isinstance(self.tensor_stride, int):
            self.tensor_stride = (self.tensor_stride,) * (self.coordinates.shape[1] - 1)
        self.tensor_stride = tuple(self.tensor_stride)
        self.coordinate_manager.register(self.tensor_stride, self.coordinates)

    # ME spells these `.F` and `.C`; upstream Mask3D uses both spellings.
    @property
    def F(self) -> torch.Tensor:  # noqa: N802
        return self.features

    @property
    def C(self) -> torch.Tensor:  # noqa: N802
        return self.coordinates

    @property
    def dimension(self) -> int:
        return self.coordinates.shape[1] - 1

    def like(self, features: torch.Tensor) -> "SparseTensor":
        """Same coordinates and stride, new features — the shape of every pointwise op."""
        return SparseTensor(
            features, self.coordinates, self.tensor_stride, self.coordinate_manager
        )

    def _assert_aligned(self, other: "SparseTensor", op: str) -> None:
        if self.tensor_stride != other.tensor_stride:
            raise ValueError(
                f"{op} needs matching tensor strides, got {self.tensor_stride} and "
                f"{other.tensor_stride}"
            )
        if self.coordinates.shape != other.coordinates.shape:
            raise ValueError(
                f"{op} needs identical coordinate sets; got {self.coordinates.shape[0]} "
                f"and {other.coordinates.shape[0]} points. In a UNet this means the "
                "decoder produced a different coordinate map than the encoder — check "
                "that the transposed convolution reused the manager's map."
            )

    def __add__(self, other: "SparseTensor") -> "SparseTensor":
        self._assert_aligned(other, "residual add")
        return self.like(self.features + other.features)


def cat(*tensors: SparseTensor) -> SparseTensor:
    """`MinkowskiOps.cat` — concatenate features on a shared coordinate set."""
    head = tensors[0]
    for other in tensors[1:]:
        head._assert_aligned(other, "cat")
    return head.like(torch.cat([t.features for t in tensors], dim=1))


# --------------------------------------------------------------------- convolution


def _strided_coords(coords: torch.Tensor, out_stride: tuple[int, ...]) -> torch.Tensor:
    """Quantise to the coarser stride and deduplicate, keeping ME's sorted order.

    Floor division, not truncation: coordinates are signed and truncation would fold
    the two sides of the origin onto each other.
    """
    stride = torch.tensor(out_stride, dtype=coords.dtype, device=coords.device)
    quantised = coords.clone()
    quantised[:, 1:] = torch.div(coords[:, 1:], stride, rounding_mode="floor") * stride
    return torch.unique(quantised, dim=0)


def sparse_conv(
    x: SparseTensor,
    weight: torch.Tensor,           # (K, C_in, C_out)
    bias: torch.Tensor | None,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] = 1,
    dilation: int = 1,
    transpose: bool = False,
) -> SparseTensor:
    """Gather / matmul / scatter over the kernel's offsets.

    `out[p] = sum_k x[p + o_k] @ W[k]` — a correlation, matching ME's kernel region,
    which is enumerated around the OUTPUT coordinate.
    """
    dim = x.dimension
    stride_t = (stride,) * dim if isinstance(stride, int) else tuple(stride)
    in_ts = x.tensor_stride
    manager = x.coordinate_manager

    # ME stores a volume-1 kernel as (C_in, C_out); restore the kernel axis so the rest
    # of this function is uniform. kernel_offsets(1) is exactly [[0, 0, 0]], so the
    # single plane pairs with the single zero offset.
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)

    if transpose:
        out_ts = tuple(t // s for t, s in zip(in_ts, stride_t))
        out_coords = manager.get(out_ts)
        if out_coords is None:
            raise ValueError(
                f"no coordinate map at tensor stride {out_ts}. A transposed convolution "
                "reuses the map the encoder created at that stride; generating a fresh "
                "one would give the decoder coordinates the skip connection cannot be "
                "concatenated with."
            )
    else:
        out_ts = tuple(t * s for t, s in zip(in_ts, stride_t))
        out_coords = (
            x.coordinates if stride_t == (1,) * dim else _strided_coords(x.coordinates, out_ts)
        )
        manager.register(out_ts, out_coords)

    # The kernel region lives in the coordinate space it gathers FROM: the input for a
    # forward convolution, the (finer) output for a transposed one.
    region_ts = out_ts if transpose else in_ts
    offsets = kernel_offsets(kernel_size, dim, region_ts, dilation).to(x.coordinates.device)

    out = x.features.new_zeros((out_coords.shape[0], weight.shape[2]))
    for k in range(offsets.shape[0]):
        if transpose:
            # Gathering from the coarse input: step to the neighbour, then quantise back
            # onto the input's stride, mirroring ME's transposed kernel map.
            probe = out_coords.clone()
            probe[:, 1:] = out_coords[:, 1:] - offsets[k]
            probe = _quantise(probe, in_ts)
        else:
            probe = out_coords.clone()
            probe[:, 1:] = out_coords[:, 1:] + offsets[k]

        rows, hit = manager.lookup(in_ts, probe)
        if not bool(hit.any()):
            continue
        gathered = x.features[rows[hit]]
        out[hit] += gathered @ weight[k]

    if bias is not None:
        out = out + bias
    return SparseTensor(out, out_coords, out_ts, manager)


def _quantise(coords: torch.Tensor, stride: tuple[int, ...]) -> torch.Tensor:
    s = torch.tensor(stride, dtype=coords.dtype, device=coords.device)
    out = coords.clone()
    out[:, 1:] = torch.div(coords[:, 1:], s, rounding_mode="floor") * s
    return out


# ------------------------------------------------------------------------- modules


class MinkowskiConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        dimension: int = 3,
        **_ignored,          # ME accepts kernel_generator=...; the region is always HYPER_CUBE here
    ) -> None:
        super().__init__()
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.dimension = dimension
        # Derived from the same generator the forward pass uses, so the parameter can
        # never disagree with the offsets it is indexed by.
        volume = kernel_offsets(kernel_size, dimension).shape[0]
        # ME stores kernels as (K, C_in, C_out) -- NOT torch's (C_out, C_in, *spatial).
        # The checkpoint is written in this layout, so it is kept verbatim.
        #
        # EXCEPT at volume 1: ME drops the kernel axis entirely and stores (C_in, C_out),
        # because a 1x1x1 convolution is a linear layer. Res16UNet34C's nine residual
        # `downsample` projections are exactly that, and allocating (1, C_in, C_out)
        # here makes them the only tensors in the checkpoint that fail to load.
        shape = (in_channels, out_channels) if volume == 1 else (volume, in_channels, out_channels)
        self.kernel = nn.Parameter(torch.empty(*shape))
        # ME's bias carries a leading singleton so it broadcasts over the point axis of
        # an (N, C) feature matrix. A bare (C,) broadcasts identically, but the released
        # tensor is (1, C) and strict loading compares shapes, not semantics.
        self.bias = nn.Parameter(torch.zeros(1, out_channels)) if bias else None
        nn.init.normal_(self.kernel, std=(2.0 / (volume * in_channels)) ** 0.5)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_conv(
            x, self.kernel, self.bias, self.kernel_size, self.stride, self.dilation,
            transpose=False,
        )


class MinkowskiConvolutionTranspose(MinkowskiConvolution):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_conv(
            x, self.kernel, self.bias, self.kernel_size, self.stride, self.dilation,
            transpose=True,
        )


class MinkowskiBatchNorm(nn.Module):
    """BatchNorm over the feature dimension of the occupied voxels only.

    Empty space is not a zero to be normalised over — that is the whole point of a
    sparse tensor, and averaging it in would shift every running statistic.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.like(self.bn(x.features))


class MinkowskiInstanceNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.like(self.norm(x.features[None].transpose(1, 2)).transpose(1, 2)[0])


class MinkowskiReLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.like(self.relu(x.features))


class MinkowskiSigmoid(nn.Module):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.like(torch.sigmoid(x.features))


class MinkowskiLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.like(self.linear(x.features))


class MinkowskiGlobalPooling(nn.Module):
    """One feature vector per batch element, on a coordinate set of one voxel each."""

    def forward(self, x: SparseTensor) -> SparseTensor:
        batch = x.coordinates[:, 0]
        uniq = torch.unique(batch)
        pooled = torch.stack([x.features[batch == b].mean(dim=0) for b in uniq])
        coords = torch.zeros(
            (uniq.shape[0], x.coordinates.shape[1]), dtype=x.coordinates.dtype,
            device=x.coordinates.device,
        )
        coords[:, 0] = uniq
        return SparseTensor(pooled, coords, x.tensor_stride, CoordinateManager())


def _pool(x: SparseTensor, kernel_size, stride, dilation, reduce: str) -> SparseTensor:
    dim = x.dimension
    stride_t = (stride,) * dim if isinstance(stride, int) else tuple(stride)
    in_ts = x.tensor_stride
    out_ts = tuple(t * s for t, s in zip(in_ts, stride_t))
    out_coords = (
        x.coordinates if stride_t == (1,) * dim else _strided_coords(x.coordinates, out_ts)
    )
    x.coordinate_manager.register(out_ts, out_coords)

    offsets = kernel_offsets(kernel_size, dim, in_ts, dilation).to(x.coordinates.device)
    total = x.features.new_zeros((out_coords.shape[0], x.features.shape[1]))
    count = x.features.new_zeros((out_coords.shape[0], 1))
    for k in range(offsets.shape[0]):
        probe = out_coords.clone()
        probe[:, 1:] = out_coords[:, 1:] + offsets[k]
        rows, hit = x.coordinate_manager.lookup(in_ts, probe)
        if not bool(hit.any()):
            continue
        total[hit] += x.features[rows[hit]]
        count[hit] += 1.0

    pooled = total / count.clamp(min=1.0) if reduce == "mean" else total
    return SparseTensor(pooled, out_coords, out_ts, x.coordinate_manager)


class MinkowskiAvgPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2, dilation=1, dimension=3, **_ignored) -> None:
        super().__init__()
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _pool(x, self.kernel_size, self.stride, self.dilation, "mean")


class MinkowskiSumPooling(MinkowskiAvgPooling):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return _pool(x, self.kernel_size, self.stride, self.dilation, "sum")


# ===========================================================================
# Comparison against real MinkowskiEngine. Everything above is inlined verbatim
# from openmask3d_semantic/minkowski_compat.py.
# ===========================================================================
import subprocess, sys, traceback

TOL = 1e-4   # fp32 accumulation over up to 125 taps; a real ordering error is ~1e0


def install_minkowski():
    """Build MinkowskiEngine with BLAS specified explicitly.

    ROOT CAUSE, found by running setup.py directly on a Kaggle T4 (rounds 1-3 all hid it
    under pip's <pip-setuptools-caller> boilerplate):

        File "/tmp/ME/setup.py", line 201
            import numpy.distutils.system_info as sysinfo
        ModuleNotFoundError: No module named 'numpy.distutils'

    `numpy.distutils` was REMOVED in NumPy 2.0, and Kaggle ships NumPy 2.x. It is not
    stdlib distutils, which is what rounds 2 and 3 were aimed at.

    The fix is not a patch: that import sits in the `else` branch taken only when BLAS
    was not specified. `--blas=openblas` takes the other branch and never imports it.
    Modern pip dropped --install-option, so setup.py is invoked directly.
    """
    import os
    env = dict(os.environ, MAX_JOBS="2")

    def run(label, cmd, **kw):
        print(f"[install] {label} ...", flush=True)
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, **kw)
        stream = ((r.stdout or "") + (r.stderr or "")).strip()
        print(f"[install] {label} rc={r.returncode}")
        for line in stream.splitlines()[-30:]:
            print("   ", line)
        return r.returncode == 0

    run("openblas", "apt-get -qq update >/dev/null 2>&1 && "
        "apt-get -qq install -y libopenblas-dev build-essential >/dev/null 2>&1; echo ok")
    run("clone", "rm -rf /tmp/ME && git clone -q --depth 1 "
        "https://github.com/NVIDIA/MinkowskiEngine.git /tmp/ME && echo cloned")
    # Capture the WHOLE build log then print the FIRST error with context. Rounds 1-4
    # used `| tail -30`, which shows the last error -- and compiler errors cascade, so
    # the tail is the consequence and the head is the cause.
    run("build --blas=openblas", r"""
cd /tmp/ME && python setup.py install --blas=openblas > /tmp/build.log 2>&1
echo "exit=$?  lines=$(wc -l < /tmp/build.log)"
echo '--- first 3 errors with context ---'
grep -n -m3 -B4 -A12 -E 'error:|Error [0-9]|fatal error' /tmp/build.log || echo '(no error: lines)'
echo '--- last 8 lines ---'
tail -8 /tmp/build.log
""", env=env)

    try:
        import MinkowskiEngine  # noqa: F401
        return True
    except Exception as exc:
        print(f"[install] still not importable: {exc}")
        return False


def random_sparse(n_points=4000, channels=8, span=48, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.unique(rng.integers(-span, span, size=(n_points, 3)), axis=0)
    batched = np.column_stack([np.zeros(len(coords), dtype=np.int64), coords])
    feats = rng.standard_normal((len(coords), channels)).astype(np.float32)
    return (torch.as_tensor(batched, dtype=torch.int32, device="cuda"),
            torch.as_tensor(feats, device="cuda"))


def compare(kernel_size, stride, c_in=8, c_out=6):
    import MinkowskiEngine as ME
    coords, feats = random_sparse(channels=c_in)

    me_conv = ME.MinkowskiConvolution(
        c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False, dimension=3
    ).cuda()
    me_out = me_conv(ME.SparseTensor(features=feats, coordinates=coords))

    # The SAME weight tensor for both: this compares the OPERATION, not an init.
    weight = me_conv.kernel.detach().clone()
    print(f"    ME kernel shape {tuple(weight.shape)}")

    ours_in = SparseTensor(feats.cpu().double(), coords.cpu(), (1, 1, 1))
    ours_out = sparse_conv(ours_in, weight.cpu().double(), None, kernel_size, stride=stride)

    me_coords = me_out.C.cpu().numpy(); me_feats = me_out.F.detach().cpu().numpy()
    our_coords = ours_out.coordinates.numpy(); our_feats = ours_out.features.numpy()
    lookup = {tuple(r): i for i, r in enumerate(our_coords)}
    shared = [(i, lookup[tuple(r)]) for i, r in enumerate(me_coords) if tuple(r) in lookup]
    if not shared:
        return float("inf"), 0, len(me_coords), len(our_coords)

    mine = np.stack([our_feats[j] for _, j in shared])
    theirs = np.stack([me_feats[i] for i, _ in shared])
    return (float(np.abs(mine - theirs).max()), len(shared),
            len(me_coords), len(our_coords))


print("=" * 78)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("=" * 78)

if not torch.cuda.is_available():
    print("RESULT: NO_GPU — kernel was not given an accelerator")
    raise SystemExit(0)

if not install_minkowski():
    print()
    print("RESULT: ME_UNBUILDABLE")
    print("MinkowskiEngine could not be built on this image. That is the very reason")
    print("minkowski_compat.py exists; record it in bug_log.txt [1] as supporting")
    print("evidence, and the kernel ORDER claim in [2] stays unverified.")
    raise SystemExit(0)

import MinkowskiEngine as ME
print("MinkowskiEngine", getattr(ME, "__version__", "?"))

cases = [
    ("k=3 s=1  odd, centered", 3, 1),
    ("k=2 s=2  EVEN, not centered", 2, 2),
    ("k=5 s=1  stem width", 5, 1),
    ("k=3 s=2  odd, strided", 3, 2),
]
worst = 0.0
print()
for label, k, s in cases:
    try:
        diff, shared, n_me, n_ours = compare(k, s)
    except Exception:
        traceback.print_exc()
        print(f"{label:<32} EXCEPTION")
        worst = float("inf")
        continue
    worst = max(worst, diff)
    verdict = "MATCH" if diff < TOL else "MISMATCH"
    print(f"{label:<32} coords ME={n_me} ours={n_ours} shared={shared} "
          f"max|diff|={diff:.3e}  {verdict}")

print()
if worst < TOL:
    print(f"RESULT: PASS  max deviation {worst:.3e} < {TOL:.0e}")
    print("Kernel offset ORDER and the correlation convention are confirmed.")
else:
    print(f"RESULT: FAIL  max deviation {worst:.3e}")
    print("Suspect, in order: (1) offset enumeration order (ME increments the FIRST")
    print("axis fastest), (2) even-kernel centering, (3) correlation sign.")

"""What can be proven about the MinkowskiEngine replacement without MinkowskiEngine.

The strong local oracle is dense equivalence: on a fully occupied grid a sparse
convolution has no sparsity left to exploit and must equal `nn.Conv3d` to floating-point
tolerance. That pins the hash, the gather/scatter, the accumulation and the boundary
behaviour against real, independently-implemented code.

What it CANNOT pin is the kernel offset ORDER, because the reference `Conv3d` weight is
built from the same offsets under test — any self-consistent order passes. The order is
therefore pinned separately as a literal read out of
`MinkowskiEngine/src/kernel_region.hpp`, and is marked cloud-verified-pending in
bug_log.txt until it is diffed against the real package on a CUDA box.
"""

import itertools
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmask3d_semantic.minkowski_compat import (  # noqa: E402
    CoordinateManager,
    MinkowskiAvgPooling,
    MinkowskiBatchNorm,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    SparseTensor,
    cat,
    kernel_offsets,
    sparse_conv,
)


def dense_grid(size: int, channels: int, batch: int = 1):
    """Every voxel of a `size`^3 cube occupied — the regime where sparse == dense."""
    coords = torch.tensor(
        [(b, x, y, z)
         for b in range(batch)
         for x, y, z in itertools.product(range(size), repeat=3)],
        dtype=torch.int32,
    )
    feats = torch.randn(coords.shape[0], channels, dtype=torch.float64)
    return SparseTensor(feats, coords)


# ------------------------------------------------------------------ offset order


def test_odd_kernel_offsets_are_centered_with_the_first_axis_fastest():
    """ME's iterator increments axis 0 innermost (kernel_region.hpp), which is the
    opposite of `meshgrid(..., indexing="ij")`. A transposed order permutes the
    checkpoint's kernel planes while loading 0 missing / 0 unexpected."""
    offsets = kernel_offsets(3, dimension=3).tolist()

    assert len(offsets) == 27
    assert offsets[:4] == [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [-1, 0, -1]], (
        "axis 0 must vary fastest"
    )
    assert offsets[13] == [0, 0, 0], "the centre tap must sit at the middle index"
    assert offsets[-1] == [1, 1, 1]


def test_even_kernel_offsets_start_at_zero_rather_than_centering():
    """`coordinate_at` centres only odd sizes. Res16UNet34C uses kernel 2 for every
    stride-2 down/upsample, so centering them would offset half the network by one voxel."""
    offsets = kernel_offsets(2, dimension=3).tolist()

    assert len(offsets) == 8
    assert offsets[0] == [0, 0, 0] and offsets[1] == [1, 0, 0]
    assert min(min(o) for o in offsets) == 0, "even kernels must not produce -1"


def test_offsets_step_by_the_tensor_stride():
    """ME keeps coordinates in the original system scaled by tensor_stride, so at
    stride 8 the adjacent voxel is 8 away. Using 1 would gather from empty space."""
    offsets = kernel_offsets(3, dimension=3, tensor_stride=8).tolist()

    assert [0, -8, -8] in offsets and [8, 8, 8] in offsets
    assert all(all(v % 8 == 0 for v in o) for o in offsets)


# -------------------------------------------------------------- the dense oracle


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_on_a_full_grid_sparse_conv_equals_conv3d(kernel_size):
    """The real test in this file. Independent implementation, exact agreement."""
    torch.manual_seed(0)
    size, c_in, c_out = 6, 3, 4
    x = dense_grid(size, c_in)

    offsets = kernel_offsets(kernel_size, dimension=3)
    weight = torch.randn(offsets.shape[0], c_in, c_out, dtype=torch.float64)

    got = sparse_conv(x, weight, None, kernel_size, stride=1)

    # Reference: pack the same features into a dense volume and convolve.
    dense = torch.zeros(1, c_in, size, size, size, dtype=torch.float64)
    for row, (_, cx, cy, cz) in enumerate(x.coordinates.tolist()):
        dense[0, :, cx, cy, cz] = x.features[row]

    pad = kernel_size // 2
    w3d = torch.zeros(c_out, c_in, kernel_size, kernel_size, kernel_size, dtype=torch.float64)
    for k, (ox, oy, oz) in enumerate(offsets.tolist()):
        w3d[:, :, ox + pad, oy + pad, oz + pad] = weight[k].T
    expected = torch.nn.functional.conv3d(dense, w3d, padding=pad)

    for row, (_, cx, cy, cz) in enumerate(got.coordinates.tolist()):
        assert torch.allclose(got.features[row], expected[0, :, cx, cy, cz], atol=1e-10), (
            f"mismatch at voxel {(cx, cy, cz)}"
        )


def test_the_dense_oracle_covers_the_boundary():
    """Voxels on the grid edge have no neighbour in some directions; those taps must
    contribute nothing, which is what Conv3d's zero padding encodes."""
    torch.manual_seed(1)
    x = dense_grid(3, 2)
    weight = torch.randn(27, 2, 2, dtype=torch.float64)

    got = sparse_conv(x, weight, None, 3, stride=1)
    corner = got.features[0]          # coordinate (0, 0, 0): 19 of 27 taps are missing

    interior_only = x.features[0] @ weight[13]
    assert not torch.allclose(corner, interior_only), "the corner ignored its neighbours"
    assert torch.isfinite(corner).all()


# ------------------------------------------------------------ coordinate handling


def test_stride_two_quantises_the_coordinate_grid():
    x = dense_grid(4, 2)
    weight = torch.randn(8, 2, 3, dtype=torch.float64)

    got = sparse_conv(x, weight, None, kernel_size=2, stride=2)

    assert got.tensor_stride == (2, 2, 2)
    assert got.coordinates.shape[0] == 2**3, "a 4^3 grid should halve to 2^3"
    assert (got.coordinates[:, 1:] % 2 == 0).all(), "coords must be multiples of the stride"


def test_transpose_reuses_the_encoder_map_so_skip_connections_concatenate():
    """The property the UNet actually depends on: after down-then-up, the coordinates
    are the encoder's, in the encoder's order, so `me.cat` is defined."""
    x = dense_grid(4, 2)
    down = sparse_conv(x, torch.randn(8, 2, 3, dtype=torch.float64), None, 2, stride=2)
    up = sparse_conv(down, torch.randn(8, 3, 2, dtype=torch.float64), None, 2,
                     stride=2, transpose=True)

    assert up.tensor_stride == x.tensor_stride
    assert torch.equal(up.coordinates, x.coordinates)
    assert cat(x, up).features.shape == (x.features.shape[0], 4)


def test_transpose_without_a_registered_map_refuses():
    """Generating fresh coordinates would silently produce a decoder tensor the skip
    connection cannot be concatenated with."""
    coords = torch.tensor([[0, 0, 0, 0], [0, 2, 0, 0]], dtype=torch.int32)
    lonely = SparseTensor(torch.randn(2, 3, dtype=torch.float64), coords, tensor_stride=(2, 2, 2))

    with pytest.raises(ValueError, match="no coordinate map at tensor stride"):
        sparse_conv(lonely, torch.randn(8, 3, 2, dtype=torch.float64), None, 2,
                    stride=2, transpose=True)


def test_cat_refuses_mismatched_coordinate_sets():
    a = dense_grid(3, 2)
    b = dense_grid(2, 2)
    with pytest.raises(ValueError, match="identical coordinate sets"):
        cat(a, b)


def test_residual_add_keeps_coordinates():
    x = dense_grid(3, 4)
    out = x + x.like(x.features * 2)
    assert torch.allclose(out.features, x.features * 3)
    assert torch.equal(out.coordinates, x.coordinates)


def test_negative_coordinates_are_floor_divided_not_truncated():
    """Truncation folds the two sides of the origin together, merging voxels that are
    two strides apart. ScanNet clouds are centred, so half of every scene is negative."""
    coords = torch.tensor([[0, -3, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    x = SparseTensor(torch.randn(3, 2, dtype=torch.float64), coords)

    got = sparse_conv(x, torch.randn(8, 2, 2, dtype=torch.float64), None, 2, stride=2)

    xs = sorted(got.coordinates[:, 1].tolist())
    assert xs == [-4, -2, 0], f"expected floor-divided bins, got {xs}"


# ---------------------------------------------------------------- other modules


def test_modules_preserve_the_coordinate_set():
    x = dense_grid(3, 4)
    for module in (MinkowskiBatchNorm(4).double(), MinkowskiAvgPooling(1, 1)):
        out = module(x)
        assert torch.equal(out.coordinates, x.coordinates)


def test_conv_module_kernel_is_me_layout():
    """(K, C_in, C_out), not torch's (C_out, C_in, *spatial). The checkpoint is written
    in ME's layout and a reshape here would load cleanly and scramble the kernel."""
    conv = MinkowskiConvolution(8, 16, kernel_size=3, dimension=3)
    assert tuple(conv.kernel.shape) == (27, 8, 16)

    tr = MinkowskiConvolutionTranspose(16, 8, kernel_size=2, stride=2, dimension=3)
    assert tuple(tr.kernel.shape) == (8, 16, 8)


def test_a_forward_backward_pass_produces_gradients():
    """CLAUDE.md's local bar: one end-to-end forward+backward on a toy batch."""
    x = dense_grid(4, 3)
    conv = MinkowskiConvolution(3, 5, kernel_size=3, dimension=3).double()

    out = conv(x)
    out.features.sum().backward()

    assert conv.kernel.grad is not None
    assert torch.isfinite(conv.kernel.grad).all()
    assert conv.kernel.grad.abs().sum() > 0


# ------------------------------------------------- against the released checkpoint


CKPT = ROOT / "openmask3d_semantic/checkpoints/scannet200_model.ckpt"


@pytest.mark.skipif(not CKPT.exists(), reason="run scripts/fetch_checkpoints.sh first")
def test_released_kernels_are_in_minkowski_layout_with_the_volumes_we_implement():
    """Ties this module to the real weights rather than to my reading of the docs.

    Two things are established here. First, kernels are stored `(K, C_in, C_out)` and
    there is not a single torch-style `(C_out, C_in, k, k, k)` tensor — so a reshape in
    our Parameter would load cleanly and scramble every kernel plane. Second, the volume
    histogram shows which kernel sizes actually occur, and k=2 (even, NOT centered)
    appears eight times: the odd/even asymmetry in `kernel_offsets` is load-bearing for
    real layers, not a hypothetical.
    """
    import collections

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)

    kernels = {k: tuple(v.shape) for k, v in state.items()
               if k.endswith(".kernel") and v.dim() == 3}
    assert kernels, "no ME-layout kernels found; the checkpoint format changed"
    assert not [k for k, v in state.items() if v.dim() == 5], (
        "found a torch-style 5-D conv weight; ME layout is no longer universal here"
    )

    assert kernels["model.backbone.conv0p1s1.kernel"] == (125, 3, 32), (
        "stem should be a 5^3 kernel over RGB -> 32"
    )

    volumes = collections.Counter(shape[0] for shape in kernels.values())
    assert volumes[125] == 1 and volumes[27] == 46 and volumes[8] == 8

    for volume in (8, 27, 125):
        size = round(volume ** (1 / 3))
        assert kernel_offsets(size, dimension=3).shape[0] == volume, (
            f"our generator disagrees with the checkpoint at kernel_size {size}"
        )


@pytest.mark.skipif(not CKPT.exists(), reason="run scripts/fetch_checkpoints.sh first")
def test_our_conv_allocates_the_checkpoints_exact_kernel_shape():
    """Constructing the stem from our module must produce the released shape verbatim."""
    conv = MinkowskiConvolution(3, 32, kernel_size=5, dimension=3)
    assert tuple(conv.kernel.shape) == (125, 3, 32)

    down = MinkowskiConvolution(32, 32, kernel_size=2, stride=2, dimension=3)
    assert tuple(down.kernel.shape) == (8, 32, 32)


def test_manager_rejects_a_coordinate_box_that_overflows_the_hash():
    """A wrapped key silently aliases two distinct voxels onto one row, so the
    convolution gathers the wrong features and nothing raises. Refuse instead."""
    mgr = CoordinateManager()
    span = 2**21
    huge = torch.tensor(
        [[0, -span, -span, -span], [0, span, span, span]], dtype=torch.int64
    )
    with pytest.raises(ValueError, match="overflows an int64 hash key"):
        mgr.register((1, 1, 1), huge)

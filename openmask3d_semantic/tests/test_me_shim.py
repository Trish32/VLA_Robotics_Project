"""The registry shim: does upstream's import surface resolve to our implementation?

Two substitutions are checked here. The MinkowskiEngine module tree is mostly plumbing —
what matters is that it refuses the cases it does not implement instead of quietly
accepting them. The torch_scatter half is arithmetic, and is checked against
hand-computed values, because this exact substitution is where QCNet's one root-caused
bug lived (`0 * inf = NaN` from an empty group, which wiped every off-diagonal entry
without raising).
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmask3d_semantic import me_shim  # noqa: E402


@pytest.fixture(autouse=True)
def clean_modules():
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in {"MinkowskiEngine", "torch_scatter"}}
    for key in saved:
        del sys.modules[key]
    yield
    for key in [k for k in sys.modules
                if k.split(".")[0] in {"MinkowskiEngine", "torch_scatter"}]:
        del sys.modules[key]
    sys.modules.update(saved)


# ------------------------------------------------------------------ module tree


def test_every_spelling_upstream_uses_resolves():
    """Mask3D imports Minkowski four different ways across seven files."""
    me_shim.install()

    import MinkowskiEngine as ME  # noqa: N817
    import MinkowskiEngine.MinkowskiOps as me
    from MinkowskiEngine import MinkowskiNetwork, MinkowskiReLU  # noqa: F401
    from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling  # noqa: F401

    assert ME.RegionType.HYPER_CUBE == 0
    assert callable(me.cat) and callable(ME.SparseTensor)


def test_install_is_idempotent():
    first = me_shim.install()
    assert me_shim.install() is first


def test_a_real_minkowskiengine_is_never_shadowed():
    """If the real package works here, it is the oracle our implementation owes a diff
    to — silently replacing it would destroy the only way to verify the kernel order."""
    sys.modules["MinkowskiEngine"] = type(sys)("MinkowskiEngine")  # pretend it is real
    with pytest.raises(RuntimeError, match="already imported"):
        me_shim.install()


def test_non_hypercube_kernels_raise_rather_than_being_ignored():
    """Res16UNet34C only builds HYPER_CUBE, but accepting HYPER_CROSS silently would
    produce a differently-shaped receptive field with no error."""
    me_shim.install()
    import MinkowskiEngine as ME  # noqa: N817

    ME.KernelGenerator(3, 1, 1, region_type=ME.RegionType.HYPER_CUBE, dimension=3)
    with pytest.raises(NotImplementedError, match="region_type"):
        ME.KernelGenerator(3, 1, 1, region_type=ME.RegionType.HYPER_CROSS, dimension=3)


def test_dataloader_side_utils_refuse_instead_of_guessing():
    """`sparse_quantize` decides which points exist; a wrong one changes the input
    cloud, and no downstream check would notice."""
    me_shim.install()
    import MinkowskiEngine as ME  # noqa: N817

    with pytest.raises(NotImplementedError, match="dataloader-"):
        ME.utils.sparse_quantize(coordinates=None)


def test_sparse_tensor_without_coordinates_refuses():
    me_shim.install()
    import MinkowskiEngine as ME  # noqa: N817

    with pytest.raises(ValueError, match="coordinate_map_key"):
        ME.SparseTensor(features=torch.zeros(4, 3))


# ----------------------------------------------------------------- torch_scatter


def test_scatter_mean_against_hand_computed_values():
    me_shim.install_scatter()
    from torch_scatter import scatter_mean

    src = torch.tensor([[1.0], [3.0], [10.0]])
    index = torch.tensor([0, 0, 1])

    got = scatter_mean(src, index, dim=0)
    assert torch.allclose(got, torch.tensor([[2.0], [10.0]]))


def test_an_empty_group_yields_zero_not_nan():
    """The QCNet bug, guarded. Group 1 receives nothing; 0/0 would be NaN and would
    propagate through every downstream segment feature."""
    me_shim.install_scatter()
    from torch_scatter import scatter_mean

    src = torch.tensor([[4.0], [6.0]])
    index = torch.tensor([0, 2])

    got = scatter_mean(src, index, dim=0, dim_size=3)
    assert torch.isfinite(got).all(), f"non-finite entry: {got}"
    assert got[1].item() == 0.0


def test_scatter_max_returns_values_and_argmax():
    me_shim.install_scatter()
    from torch_scatter import scatter_max

    src = torch.tensor([[1.0], [7.0], [3.0], [2.0]])
    index = torch.tensor([0, 0, 1, 1])

    values, arg = scatter_max(src, index, dim=0)
    assert torch.allclose(values, torch.tensor([[7.0], [3.0]]))
    assert arg.flatten().tolist() == [1, 2]


def test_scatter_max_on_an_empty_group_is_zero_with_index_minus_one():
    """-inf would survive into the attention mask as a real value."""
    me_shim.install_scatter()
    from torch_scatter import scatter_max

    src = torch.tensor([[5.0]])
    values, arg = scatter_max(src, torch.tensor([0]), dim=0, dim_size=2)

    assert values.flatten().tolist() == [5.0, 0.0]
    assert arg.flatten().tolist() == [0, -1]
    assert torch.isfinite(values).all()


def test_scatter_min_mirrors_scatter_max():
    me_shim.install_scatter()
    from torch_scatter import scatter_min

    src = torch.tensor([[1.0], [7.0], [3.0], [2.0]])
    values, _ = scatter_min(src, torch.tensor([0, 0, 1, 1]), dim=0)
    assert torch.allclose(values, torch.tensor([[1.0], [2.0]]))


def test_negative_values_are_not_confused_with_empty_slots():
    """The fill sentinel is +-inf precisely so that a legitimately negative maximum is
    not mistaken for an unwritten slot and zeroed."""
    me_shim.install_scatter()
    from torch_scatter import scatter_max

    src = torch.tensor([[-5.0], [-2.0]])
    values, _ = scatter_max(src, torch.tensor([0, 0]), dim=0)
    assert values.flatten().tolist() == [-2.0]


def test_the_real_torch_scatter_is_preferred_when_present():
    sentinel = type(sys)("torch_scatter")
    sys.modules["torch_scatter"] = sentinel
    assert me_shim.install_scatter() is sentinel

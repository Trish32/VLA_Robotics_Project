"""The import guard, and the honesty of what it reports.

FoundationPose's networks are plain PyTorch; only its renderer is CUDA-only. These tests
pin that the guard rescues the former without pretending to supply the latter.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "foundationpose_6dof"))

import compat  # noqa: E402


@pytest.fixture(autouse=True)
def clean_modules():
    """Each test starts without leftovers from the last one's sys.modules edits."""
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in {"nvdiffrast", "pytorch3d"}}
    for key in saved:
        del sys.modules[key]
    yield
    for key in [k for k in sys.modules
                if k.split(".")[0] in {"nvdiffrast", "pytorch3d"}]:
        del sys.modules[key]
    sys.modules.update(saved)


def test_the_guard_makes_the_cuda_only_imports_resolve():
    compat.ensure_importable()

    import nvdiffrast.torch  # noqa: F401
    from pytorch3d.transforms import so3_log_map  # noqa: F401
    from pytorch3d.structures import Meshes  # noqa: F401


def test_touching_a_stub_raises_instead_of_returning_geometry():
    """A fake rasteriser is the worst possible stub here: FoundationPose scores rendered
    pose hypotheses against the observed crop, so plausible-but-wrong pixels produce a
    confident, wrong pose with nothing downstream able to detect it."""
    compat.ensure_importable()
    import nvdiffrast.torch as dr

    with pytest.raises(NotImplementedError, match="CUDA rasteriser"):
        dr.RasterizeCudaContext()


def test_the_error_says_it_is_deliberate_and_where_to_run_it():
    compat.ensure_importable()
    import nvdiffrast.torch as dr

    with pytest.raises(NotImplementedError, match="GPU host"):
        dr.rasterize()


def test_calling_the_guard_twice_does_not_claim_the_stubs_are_real():
    """A stub satisfies `__import__`, so a naive availability probe flips to True on the
    second call and `describe` announces a renderer that does not exist."""
    first = compat.ensure_importable()
    second = compat.ensure_importable()

    assert first == second, f"status changed between calls: {first} -> {second}"
    if not first["nvdiffrast"]:
        assert "stubbed" in compat.describe(second)
        assert "can actually run" not in compat.describe(second)


def test_a_real_package_is_never_shadowed_by_a_stub():
    """torch is certainly installed; the guard must leave real packages alone."""
    import torch

    compat.ensure_importable()
    assert not isinstance(torch, compat.MissingModule)
    assert compat._available("torch")


def test_dunder_lookups_do_not_masquerade_as_callables():
    """`__path__`, `__all__` and friends are probed by the import machinery itself; a
    stub that answered them with a raising function would break `from x import *`."""
    compat.ensure_importable()
    import pytorch3d

    with pytest.raises(AttributeError):
        pytorch3d.__does_not_exist__

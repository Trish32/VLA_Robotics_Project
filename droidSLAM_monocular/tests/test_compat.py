"""Tests for the DROID-SLAM compiled-dependency shims.

Two categories, matching the split in ../compat.py:

**Substitutions that must be exact.** `scatter_mean`/`scatter_sum` replace torch_scatter.
They are index arithmetic, so they are checkable against hand-computed values here, and
against the real package on any machine that has it (`test_matches_real_torch_scatter`).

**Extensions that must RAISE.** lietorch and droid_backends are the system's actual
algorithms. CLAUDE.md: "Never fake a CUDA op with a silently-wrong CPU stub... If it
cannot run locally, it must raise locally." These tests are the guard that nobody later
'helpfully' turns them into zeros-returning stubs.

Run: PYTHONPATH=droidSLAM_monocular/upstream/droid_slam pytest droidSLAM_monocular/tests -q
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compat import (  # noqa: E402
    missing_extension,
    scatter_mean,
    scatter_sum,
    try_import_droid_backends,
    try_import_lietorch,
    try_import_scatter,
)


# --------------------------------------------------------------- scatter_sum


def test_scatter_sum_matches_hand_calculation():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])

    got = scatter_sum(src, index, dim=0)

    torch.testing.assert_close(got, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))


def test_scatter_sum_respects_dim_size():
    """Trailing empty groups must be preserved, not truncated.

    GraphAgg indexes by keyframe id; a silently shorter output would misalign every
    subsequent frame rather than raise.
    """
    src = torch.ones(3, 2)
    out = scatter_sum(src, torch.tensor([0, 0, 1]), dim=0, dim_size=5)
    assert out.shape == (5, 2)
    torch.testing.assert_close(out[2:], torch.zeros(3, 2))


def test_scatter_sum_rejects_multidimensional_index():
    with pytest.raises(ValueError, match="1-D index"):
        scatter_sum(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.long), dim=0)


# -------------------------------------------------------------- scatter_mean


def test_scatter_mean_matches_hand_calculation():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = scatter_mean(src, torch.tensor([0, 0, 1]), dim=0)
    torch.testing.assert_close(got, torch.tensor([[2.0, 3.0], [5.0, 6.0]]))


def test_scatter_mean_of_empty_group_is_zero_not_nan():
    """The QCNet bug, guarded.

    An empty group divides by a count of zero. torch_scatter yields 0; a naive
    implementation yields NaN, which then propagates through the whole factor graph. That
    exact failure (`0 * inf = NaN` wiping every off-diagonal) cost a debug cycle in
    QCNet_vl — see its bug_log.
    """
    out = scatter_mean(torch.ones(2, 3), torch.tensor([0, 0]), dim=0, dim_size=4)

    assert torch.isfinite(out).all(), "empty group produced a non-finite value"
    torch.testing.assert_close(out[1:], torch.zeros(3, 3))


def test_scatter_mean_on_the_dim_GraphAgg_uses():
    """GraphAgg calls scatter_mean(net, ix, dim=1) on (batch, num, ch, ht, wd)."""
    net = torch.randn(2, 4, 8, 3, 3)
    ix = torch.tensor([0, 0, 1, 1])

    out = scatter_mean(net, ix, dim=1)

    assert out.shape == (2, 2, 8, 3, 3)
    torch.testing.assert_close(out[:, 0], net[:, :2].mean(dim=1))
    torch.testing.assert_close(out[:, 1], net[:, 2:].mean(dim=1))


def test_matches_real_torch_scatter():
    """Oracle against the real package, wherever it happens to be installed."""
    torch_scatter = pytest.importorskip("torch_scatter")

    torch.manual_seed(0)
    src = torch.randn(6, 5)
    index = torch.tensor([0, 2, 0, 1, 2, 2])

    torch.testing.assert_close(
        scatter_sum(src, index, dim=0), torch_scatter.scatter_sum(src, index, dim=0)
    )
    torch.testing.assert_close(
        scatter_mean(src, index, dim=0), torch_scatter.scatter_mean(src, index, dim=0)
    )


# ------------------------------------------------- extensions that must raise


def test_missing_extension_imports_but_raises_on_call():
    ext = missing_extension("fake_ext", "because it is a test")

    with pytest.raises(NotImplementedError, match="fake_ext"):
        ext.some_kernel(torch.ones(3))


def test_missing_extension_message_says_it_is_deliberate():
    """The message has to stop the next person from stubbing it.

    A bare NotImplementedError invites someone to 'fix' it by returning zeros, which is
    exactly the silently-wrong outcome the design is avoiding.
    """
    ext = missing_extension("fake_ext", "reason here")
    with pytest.raises(NotImplementedError, match="deliberately NOT stubbed"):
        ext.anything()


@pytest.mark.parametrize("importer", [try_import_lietorch, try_import_droid_backends])
def test_extension_placeholders_raise_when_unavailable(importer):
    module, available = importer()
    if available:
        pytest.skip("real extension installed; the placeholder path cannot be exercised")

    with pytest.raises(NotImplementedError):
        module.any_function(torch.ones(2))


def test_scatter_substitution_is_reported_honestly():
    """try_import_scatter must say whether it returned the real package or ours."""
    _, _, is_real = try_import_scatter()
    try:
        import torch_scatter  # noqa: F401

        assert is_real, "torch_scatter is installed but the shim was used"
    except ImportError:
        assert not is_real, "torch_scatter is absent but was reported as real"


# ------------------------------------------------------------- CUDA-only oracle


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-only oracle; cloud-verified, not local"
)


@requires_cuda
def test_lietorch_se3_roundtrip_on_cuda():
    """exp(log(g)) == g for SE3, including near identity.

    Small-angle terms (sin(θ)/θ) need Taylor branches; a naive implementation returns NaN
    exactly at identity, which is where a SLAM system spends much of its time. This runs
    only on CUDA and is recorded in bug_log.txt as cloud-verified.
    """
    lietorch, available = try_import_lietorch()
    if not available:
        pytest.skip("lietorch not installed on this CUDA machine")

    for scale in (1.0, 1e-4, 1e-8):  # 1e-8 is the small-angle branch
        xi = torch.randn(8, 6, device="cuda") * scale
        g = lietorch.SE3.exp(xi)
        torch.testing.assert_close(g.log(), xi, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_droid_backends_is_importable_on_cuda():
    """If this fails on the GPU box, the extension was never built."""
    _, available = try_import_droid_backends()
    assert available, "droid_backends missing: run `pip install .` in upstream/"

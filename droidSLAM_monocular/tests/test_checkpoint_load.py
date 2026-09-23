"""Regression test for the DROID-SLAM fidelity gate.

`droid.pth` must keep loading into `DroidNet` at 0 missing / 0 unexpected on CPU, with
neither lietorch nor droid_backends installed. If a re-pull of upstream restores the
module-scope imports, this fails at collection rather than silently later.

Run: PYTHONPATH=droidSLAM_monocular/upstream/droid_slam pytest droidSLAM_monocular/tests/test_checkpoint_load.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/droid_slam"))
sys.path.insert(0, str(ROOT / "tools"))

CHECKPOINT = ROOT / "checkpoints/droid.pth"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="droid.pth not downloaded"
)


@pytest.fixture(scope="module")
def loaded():
    from load_checkpoint import apply_upstream_slice, load_state_dict, strip_dataparallel

    sd, _ = strip_dataparallel(load_state_dict(CHECKPOINT))
    sd, sliced = apply_upstream_slice(sd)
    return sd, sliced


def test_droidnet_builds_without_cuda_extensions():
    """The import guards are what make this possible; a re-pull would break it."""
    from droid_net import DroidNet

    net = DroidNet()
    assert sum(p.numel() for p in net.parameters()) == pytest.approx(4.0e6, rel=0.02)
    assert {n for n, _ in net.named_children()} == {"fnet", "cnet", "update"}


def test_checkpoint_loads_strictly(loaded):
    from droid_net import DroidNet

    sd, _ = loaded
    DroidNet().load_state_dict(sd, strict=True)  # raises on any mismatch


def test_upstream_slice_is_applied_to_exactly_four_tensors(loaded):
    """The released heads are 3-channel; upstream keeps the first two.

    Pinned because it looks like a bug (and I first recorded it as one — see
    bug_log.txt [2]). If a future checkpoint ships 2-channel heads this test fails and
    the slice should be dropped, rather than silently truncating something else.
    """
    _, sliced = loaded
    assert set(sliced) == {
        "update.weight.2.weight", "update.weight.2.bias",
        "update.delta.2.weight", "update.delta.2.bias",
    }


def test_sliced_heads_have_two_output_channels(loaded):
    sd, _ = loaded
    for key in ("update.delta.2.weight", "update.weight.2.weight"):
        assert sd[key].shape[0] == 2, f"{key} should be sliced to 2 output channels"


def test_lietorch_is_not_installed_here():
    """Documents WHY the guards exist. If lietorch ever becomes installable on this
    machine, the local bar rises and the CUDA-only tests stop being cloud-only."""
    from compat import try_import_lietorch  # noqa: E402

    _, available = try_import_lietorch()
    if available:
        pytest.skip("lietorch available — revisit which tests are CUDA-gated")
    assert not available

"""Cross-validate the torch-free wire format against upstream's MsgSerializer.

The bridge restates GR00T's serialization so the ROS2 container does not need torch.
That duplication is only safe if it is checked, so this test runs on the Mac (where
gr00t IS installed) and asserts byte-level agreement in both directions.

If upstream changes the encoding, this fails here rather than on a robot.

Run: conda run -n groot_vl python -m pytest ros2_bridge/vla_bridge/test -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge import wire  # noqa: E402

# NOT `pytest.importorskip`. That raises Skipped during collection, and in the ROS
# container — where gr00t is absent by design — it aborted the WHOLE directory: running
# `pytest test` reported "1 skipped" and a green exit while silently running none of the
# other 55 tests. A suite that passes by not running is worse than one that fails.
# `skipif` decides the same thing without raising mid-collection.
_HAS_GR00T = importlib.util.find_spec("gr00t") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_GR00T, reason="gr00t not installed; run in the grootN1_Robotics env"
)

if _HAS_GR00T:
    from gr00t.policy.server_client import MsgSerializer
else:  # pragma: no cover - every test in the module is skipped
    MsgSerializer = None


def observations():
    """Payloads shaped like a real GR00T observation."""
    rng = np.random.default_rng(0)
    return [
        {"video.ego_view": rng.integers(0, 255, (1, 8, 8, 3), dtype=np.uint8)},
        {"state.single_arm": rng.standard_normal((1, 7)).astype(np.float32)},
        {"state.gripper": np.array([[0.0]], dtype=np.float64)},
        {"annotation.human.task_description": ["pick up the red block"]},
        {
            "video.ego_view": rng.integers(0, 255, (1, 4, 4, 3), dtype=np.uint8),
            "state.single_arm": rng.standard_normal((1, 7)).astype(np.float32),
            "annotation.human.task_description": ["stack the cubes"],
        },
        {"empty": np.zeros((0,), dtype=np.float32)},
        {"scalar": np.array(3.14, dtype=np.float32)},
    ]


@pytest.mark.parametrize("payload", observations())
def test_our_bytes_match_upstream_bytes(payload):
    assert wire.to_bytes(payload) == MsgSerializer.to_bytes(payload)


@pytest.mark.parametrize("payload", observations())
def test_upstream_can_read_our_bytes(payload):
    decoded = MsgSerializer.from_bytes(wire.to_bytes(payload))
    _assert_same(decoded, payload)


@pytest.mark.parametrize("payload", observations())
def test_we_can_read_upstream_bytes(payload):
    decoded = wire.from_bytes(MsgSerializer.to_bytes(payload))
    _assert_same(decoded, payload)


def test_dtype_and_shape_survive_the_round_trip():
    """Silent dtype drift would corrupt actions without raising anything."""
    for dtype in (np.uint8, np.int32, np.int64, np.float32, np.float64):
        arr = np.arange(6, dtype=dtype).reshape(2, 3)
        out = wire.from_bytes(MsgSerializer.to_bytes({"a": arr}))["a"]
        assert out.dtype == arr.dtype, f"dtype changed for {dtype}"
        assert out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)


def test_request_envelope_matches():
    """The full request envelope, not just the payload."""
    req = {"endpoint": "get_action", "data": {"state.x": np.ones((1, 3), np.float32)}}
    assert wire.to_bytes(req) == MsgSerializer.to_bytes(req)


def _assert_same(got, want):
    assert type(got) is type(want)
    if isinstance(want, dict):
        assert got.keys() == want.keys()
        for k in want:
            _assert_same(got[k], want[k])
    elif isinstance(want, np.ndarray):
        assert got.dtype == want.dtype and got.shape == want.shape
        np.testing.assert_array_equal(got, want)
    elif isinstance(want, list):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            _assert_same(g, w)
    else:
        assert got == want

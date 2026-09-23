"""What can honestly be tested about the SLAM server without a GPU.

`lietorch` and `droid_backends` are CUDA-compile-only, so DROID itself cannot run here
and no amount of test-writing changes that. What CAN be pinned locally is everything
around it — and those parts are where the silent errors live:

  * the image/intrinsics preprocessing, where a rescale without a matching intrinsics
    rescale yields a trajectory that is quietly wrong by the scale factor;
  * the CUDA guard, which must RAISE rather than fall back to a stub;
  * the endpoint wiring, so the ROS node's `track` call reaches a handler at all.

The tracking itself is cloud-verified, not locally verified. Recorded in bug_log.txt.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from droidSLAM_monocular.slam_server import (  # noqa: E402
    CudaRequired,
    SlamService,
    build_slam,
    preprocess,
    preprocess_depth,
)

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")


def service() -> SlamService:
    from types import SimpleNamespace

    return SlamService(SimpleNamespace(image_size=[384, 512]))


# ------------------------------------------------------------------ preprocessing


def test_image_is_resized_toward_droids_384x512_budget():
    image = np.zeros((720, 1280, 3), np.uint8)
    tensor, _ = preprocess(image, np.array([600.0, 600.0, 640.0, 360.0]))

    assert tensor.shape[:2] == (1, 3), "DROID wants (B, 3, H, W)"
    h, w = tensor.shape[2], tensor.shape[3]
    assert h % 8 == 0 and w % 8 == 0, "feature pyramid needs a multiple of 8"
    assert 0.5 < (h * w) / (384 * 512) < 1.5


def test_intrinsics_are_rescaled_with_the_image():
    """The failure this test exists for is silent: DROID still tracks, and the whole
    trajectory is wrong by the scale factor."""
    h0, w0 = 720, 1280
    fx, fy, cx, cy = 600.0, 600.0, 640.0, 360.0
    tensor, scaled = preprocess(np.zeros((h0, w0, 3), np.uint8),
                                np.array([fx, fy, cx, cy]))

    h1, w1 = tensor.shape[2], tensor.shape[3]
    # The crop to a multiple of 8 happens after the resize, so compare against the
    # pre-crop scale the same way demo.py does.
    scale = np.sqrt((384 * 512) / (h0 * w0))
    assert scaled[0].item() == pytest.approx(fx * int(w0 * scale) / w0, rel=1e-5)
    assert scaled[2].item() == pytest.approx(cx * int(w0 * scale) / w0, rel=1e-5)
    assert scaled[1].item() == pytest.approx(fy * int(h0 * scale) / h0, rel=1e-5)
    assert scaled[3].item() == pytest.approx(cy * int(h0 * scale) / h0, rel=1e-5)
    assert h1 <= int(h0 * scale) and w1 <= int(w0 * scale)


def test_intrinsics_are_not_mutated_in_place():
    """The node reuses one intrinsics array for every frame; scaling it in place would
    shrink it a little further on each call until tracking collapsed."""
    original = np.array([600.0, 600.0, 640.0, 360.0])
    preprocess(np.zeros((720, 1280, 3), np.uint8), original)

    assert np.allclose(original, [600.0, 600.0, 640.0, 360.0])


# ----------------------------------------------------------------- the CUDA guard


@pytest.mark.skipif(torch.cuda.is_available(), reason="the guard only fires without CUDA")
def test_build_slam_raises_rather_than_stubbing():
    """A CPU stub would return well-shaped poses that are silently wrong — the worst
    possible outcome for a value nothing downstream can sanity-check."""
    with pytest.raises(CudaRequired, match="lietorch and droid_backends"):
        build_slam(Path("/nonexistent/droid.pth"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="the guard only fires without CUDA")
def test_the_error_says_where_to_run_it():
    with pytest.raises(CudaRequired, match="GPU host"):
        build_slam(Path("/nonexistent/droid.pth"))


# ------------------------------------------------------------------ endpoint shape


def test_track_rejects_a_malformed_image_before_touching_cuda():
    with pytest.raises(ValueError, match=r"expected \(H, W, 3\)"):
        service().track(np.zeros((8, 8), np.uint8), np.array([1.0, 1.0, 4.0, 4.0]))


def test_reset_clears_the_keyframe_graph():
    """A new episode inheriting the previous map is worse than a cold start."""
    svc = service()
    svc.droid, svc.frame_index = object(), 42

    assert svc.reset() == {"ok": True}
    assert svc.droid is None and svc.frame_index == 0


def test_get_action_refuses_instead_of_answering_emptily():
    """PolicyServer wires get_action unconditionally, so a VLA node CAN reach this
    server by pointing at the wrong port. Say so rather than return nothing."""
    with pytest.raises(NotImplementedError, match="track"):
        service().get_action({})


def test_the_reply_field_names_match_what_the_bridge_decodes():
    """Pinned against the node's decoder so the two cannot drift apart.

    `slam_decode` accepts a bare 7-vector too, but this server names the fields, so the
    scalar-last convention is stated on the wire instead of inferred from position.
    """
    sys.path.insert(0, str(ROOT / "ros2_bridge/vla_bridge"))
    from vla_bridge.slam_decode import decode_pose, is_tracking_lost

    reply = {"translation": [1.0, 2.0, 3.0], "quaternion": [0.0, 0.0, 0.0, 1.0],
             "n_keyframes": 12, "tracking_lost": False, "stamp_ns": 5}

    translation, quaternion = decode_pose(reply)
    assert np.allclose(translation, [1.0, 2.0, 3.0])
    assert np.allclose(quaternion, [0.0, 0.0, 0.0, 1.0])
    assert not is_tracking_lost(reply)


def test_a_pre_keyframe_reply_reads_as_tracking_lost():
    """Before the motion filter accepts a first keyframe there is no pose, and the
    identity would be a lie. The node must publish nothing."""
    sys.path.insert(0, str(ROOT / "ros2_bridge/vla_bridge"))
    from vla_bridge.slam_decode import is_tracking_lost

    assert is_tracking_lost({"n_keyframes": 0, "tracking_lost": True})


# ------------------------------------------------------------------------- RGB-D
#
# Depth enters DROID as a per-pixel inverse-depth prior consumed inside the BA kernel,
# so every error here is a wrong constraint rather than a crash. None of these tests can
# check tracking; they check that what we hand the tracker is what upstream's own RGB-D
# entry point would have handed it.


def test_depth_and_image_come_out_the_same_size():
    """They index each other pixel-for-pixel inside `video.append`."""
    image = np.zeros((480, 640, 3), np.uint8)
    tensor, _ = preprocess(image, np.array([600.0, 600.0, 320.0, 240.0]))
    depth = preprocess_depth(np.ones((480, 640), np.float32), (480, 640))

    assert tuple(depth.shape) == (tensor.shape[2], tensor.shape[3])


def test_depth_is_resized_nearest_not_bilinear():
    """Bilinear across a depth edge invents a surface at the midpoint of foreground and
    background. That value is not smoothed away downstream — it goes into the dense BA
    as a confident per-pixel prior at a depth where nothing exists.

    Upstream relies on `F.interpolate`'s nearest default (test_eth3d.py:46); this pins it.
    """
    depth = np.full((480, 640), 1.0, np.float32)
    depth[:, 320:] = 5.0

    out = preprocess_depth(depth, (480, 640)).numpy()

    assert set(np.unique(out).tolist()) <= {1.0, 5.0}, (
        "resize produced depths that were in neither plateau — that is interpolation "
        "across the discontinuity"
    )


def test_invalid_pixels_stay_exactly_zero():
    """`depth_video.append` computes where(depth > 0, 1/depth, depth), so 0 means 'no
    prior'. Any fringe of small non-zero values around a hole becomes a prior at a huge
    inverse depth."""
    depth = np.full((480, 640), 2.0, np.float32)
    depth[100:200, 100:200] = 0.0

    out = preprocess_depth(depth, (480, 640)).numpy()

    assert set(np.unique(out).tolist()) <= {0.0, 2.0}
    assert (out == 0.0).any(), "the hole disappeared entirely"


def test_depth_is_float32_for_the_disps_sens_buffer():
    out = preprocess_depth(np.ones((480, 640), np.float64), (480, 640))
    assert out.dtype == torch.float32


def test_unaligned_depth_is_refused_rather_than_resized():
    """A depth map at a different resolution is usually a different VIEWPOINT — the
    unaligned depth stream, not a scaling problem. Resizing it for the caller would
    produce a plausible map that is wrong at every pixel."""
    with pytest.raises(ValueError, match="not aligned"):
        SlamService._validate_depth(np.ones((240, 320), np.float32), (480, 640))


def test_non_finite_and_negative_depth_are_refused():
    bad = np.ones((4, 4), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        SlamService._validate_depth(bad, (4, 4))

    with pytest.raises(ValueError, match="negative"):
        SlamService._validate_depth(-np.ones((4, 4), np.float32), (4, 4))


def test_the_stream_cannot_switch_between_monocular_and_rgbd():
    """`disps_sens` is a persistent buffer and `droid_backend.py:30` decides scale
    normalisation from whether ANY of it is non-zero. A stream that supplies depth for
    only part of its frames gets a gauge that is neither fixed nor normalised, and
    nothing raises."""
    image = np.zeros((16, 16, 3), np.uint8)
    intrinsics = np.array([8.0, 8.0, 8.0, 8.0])

    svc = service()
    svc.rgbd = False
    with pytest.raises(ValueError, match="monocular"):
        svc.track(image, intrinsics, depth=np.ones((16, 16), np.float32))

    svc = service()
    svc.rgbd = True
    with pytest.raises(ValueError, match="omits"):
        svc.track(image, intrinsics)


def test_reset_clears_the_rgbd_latch():
    """Otherwise a monocular episode could never follow an RGB-D one."""
    svc = service()
    svc.rgbd = True
    svc.reset()
    assert svc.rgbd is None


def test_upstream_still_gates_scale_normalisation_on_sensor_depth():
    """Our `metric` flag claims RGB-D poses are in metres. That is only true because
    upstream skips `video.normalize()` when disps_sens is populated. If this line moves,
    the flag becomes a lie and every downstream metre — the TSDF, the object frame, the
    pre-grasp offset — is silently scaled."""
    source = (ROOT / "droidSLAM_monocular/upstream/droid_slam/droid_backend.py").read_text()
    assert "not torch.any(self.video.disps_sens)" in source, (
        "upstream changed how RGB-D disables scale normalisation; re-derive the `metric` "
        "field in slam_server.py before trusting it"
    )


def test_lietorch_identity_pose_is_scalar_last():
    """Not an assumption: `depth_video.py` initialises every pose to this literal, and
    it is the identity only if w is the LAST element.

    A scalar-first reading of the same seven numbers is a 180-degree rotation, still unit
    norm, with nothing to raise on.
    """
    source = (ROOT / "droidSLAM_monocular/upstream/droid_slam/depth_video.py").read_text()
    assert "[0, 0, 0, 0, 0, 0, 1]" in source, (
        "upstream's identity pose literal changed; re-check the quaternion convention "
        "in slam_server.py and vla_bridge/slam_decode.py before trusting either"
    )

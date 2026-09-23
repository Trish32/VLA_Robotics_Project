"""Serve DROID-SLAM behind the same ZMQ contract the ROS bridge already speaks.

The bridge node is transport-agnostic: it talks to whatever answers `ping`, `reset` and
`track` on a ZMQ REP socket. GR00T supplies `PolicyServer`; this reuses it, registering a
`track` endpoint instead of `get_action`, so the ROS container keeps exactly one wire
implementation for all three models.

**This file cannot run on the Mac and does not pretend to.** `lietorch` and
`droid_backends` are CUDA-compile-only extensions, so `build_slam` raises immediately on
a non-CUDA box rather than importing a stub. Per CLAUDE.md: never fake a CUDA op with a
silently-wrong CPU path — a stub that returns zeros passes every shape test and poisons
everything downstream. A pose is exactly the kind of value where "plausible but wrong" is
indistinguishable from correct until the map is ruined.

Two things worth stating because getting them wrong is silent:

**Pose direction.** `video.poses` holds WORLD-TO-CAMERA transforms; `droid.terminate()`
returns `camera_trajectory.inv()`, i.e. camera-to-world. ROS odometry wants the latter —
where the camera IS in the map frame — so this server inverts before replying.

**Quaternion order.** lietorch SE3 is `[tx, ty, tz, qx, qy, qz, qw]`, scalar-LAST. That
is not inferred: `depth_video.py` initialises every pose to `[0,0,0, 0,0,0,1]`, which is
the identity only if w is last. ROS `geometry_msgs/Quaternion` is also scalar-last, so
the four numbers carry across untouched — but Eigen, MuJoCo and older scipy put w first,
and a swapped layout is still a unit quaternion describing the wrong rotation. The reply
therefore names the fields explicitly rather than sending a bare 7-vector.

**Online, not batch.** `droid.track()` returns nothing and `droid.terminate()` needs the
whole stream replayed, so it is unusable for live odometry. The latest keyframe pose is
read out of `video.poses[counter - 1]` instead. That is a keyframe pose: DROID's motion
filter decides what becomes a keyframe, so the pose does not advance on every frame, and
`n_keyframes` in the reply is how the caller can tell.

**RGB-D and the scale gauge.** Depth is optional and changes what the poses MEAN. DROID
takes sensor depth as a per-pixel inverse-depth prior — `video.append` stores
`where(depth > 0, 1/depth, depth)` into `disps_sens`, the frontend initialises `disps`
from it, and `droid_backends.ba` consumes it inside the Schur solve. Crucially,
`droid_backend.py:30` skips `video.normalize()` when `disps_sens` is non-zero, because
metric depth already fixes the gauge. So monocular poses carry an arbitrary scale and
RGB-D poses are in metres, with nothing in the reply to distinguish them unless we say
so — hence `metric` in the response and the latched mode in `track`.

Run (GPU host):
    python droidSLAM_monocular/slam_server.py --weights droidSLAM_monocular/checkpoints/droid.pth --port 5557
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# DROID's own target: ~384x512, cropped to a multiple of 8 for the feature pyramid.
TARGET_PIXELS = 384 * 512


class CudaRequired(RuntimeError):
    """Raised instead of importing a stub that would return plausible nonsense."""


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise CudaRequired(
            "DROID-SLAM needs CUDA: lietorch and droid_backends are compiled extensions "
            "with no CPU build. There is no correct way to run this on the Mac, and a "
            "stub would return well-shaped poses that are silently wrong. Run it on the "
            "GPU host and point the ROS node at it with `host:=<gpu-host>`."
        )


def _target_size(h0: int, w0: int) -> tuple[int, int]:
    """DROID's pre-crop resize target. Shared so image and depth cannot drift apart."""
    scale = np.sqrt(TARGET_PIXELS / (h0 * w0))
    return int(h0 * scale), int(w0 * scale)


def preprocess(image: np.ndarray, intrinsics: np.ndarray):
    """(H, W, 3) uint8 + [fx, fy, cx, cy] -> ((1, 3, h, w) tensor, scaled intrinsics).

    Mirrors `demo.py::image_stream` exactly, including the crop to a multiple of 8.
    Rescaling the image without rescaling the intrinsics is a classic silent failure: the
    tracker still runs and the trajectory is quietly wrong by the scale factor.
    """
    import cv2
    import torch

    h0, w0 = image.shape[:2]
    h1, w1 = _target_size(h0, w0)

    resized = cv2.resize(image, (w1, h1))
    resized = resized[: h1 - h1 % 8, : w1 - w1 % 8]
    tensor = torch.as_tensor(resized).permute(2, 0, 1)[None]

    scaled = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32)).clone()
    scaled[0::2] *= w1 / w0
    scaled[1::2] *= h1 / h0
    return tensor, scaled


def preprocess_depth(depth_m: np.ndarray, image_hw: tuple[int, int]):
    """(H, W) float metres -> (h, w) float32 tensor aligned with `preprocess`'s output.

    Mirrors `evaluation_scripts/test_eth3d.py::image_stream`, which is upstream's only
    RGB-D entry point: interpolate to the PRE-crop size, then crop. Resizing straight to
    the cropped size would shift the nearest-neighbour sampling grid.

    Three properties here are load-bearing, and all three fail silently:

    **Nearest, never bilinear.** `F.interpolate` defaults to nearest and upstream relies
    on that default. Averaging across a depth discontinuity invents a surface at the
    midpoint of foreground and background, and this number goes straight into the dense
    BA as a per-pixel prior — it is not smoothed away downstream.

    **Zeros stay zeros.** `depth_video.append` computes `where(depth > 0, 1/depth, depth)`,
    so 0 is the "no prior" sentinel. Hole-filling with a plausible default converts
    missing data into a confident wrong constraint.

    **Metres.** ROS `16UC1` depth is millimetres, ETH3D PNGs are /5000, and neither
    raises if you pick the wrong one — it rescales the entire map.
    """
    import torch
    import torch.nn.functional as F

    h0, w0 = image_hw
    h1, w1 = _target_size(h0, w0)

    depth = torch.as_tensor(np.asarray(depth_m, dtype=np.float32))
    depth = F.interpolate(depth[None, None], (h1, w1)).squeeze(0).squeeze(0)
    return depth[: h1 - h1 % 8, : w1 - w1 % 8].contiguous()


class SlamService:
    """Duck-typed to what `PolicyServer` calls, plus a `track` endpoint of its own."""

    def __init__(self, args) -> None:
        self.args = args
        self.droid = None       # built lazily: DROID needs the image size up front
        self.frame_index = 0
        self.rgbd: bool | None = None   # latched by the first frame; see `track`

    # ---------------------------------------------------------------- endpoints

    def reset(self, options: dict | None = None) -> dict:
        """Drop the keyframe graph. A new episode must not inherit the old map."""
        self.droid = None
        self.frame_index = 0
        self.rgbd = None
        return {"ok": True}

    def get_action(self, observation: dict, options: dict | None = None):
        """`PolicyServer.__init__` wires this endpoint unconditionally.

        SLAM produces poses, not actions. Raising names the mistake; returning an empty
        dict would let a VLA node sit there receiving nothing and logging nothing useful.
        """
        raise NotImplementedError(
            "this is a DROID-SLAM server, not a policy — call the 'track' endpoint. "
            "The VLA node belongs on ports 5555 (GR00T) or 5556 (DiVLA)."
        )

    def track(self, image, intrinsics, depth=None, stamp_ns: int | None = None) -> dict:
        # Validate BEFORE importing the CUDA extensions, so a malformed request gets the
        # error it deserves rather than a ModuleNotFoundError about lietorch.
        frame = np.asarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"image has shape {frame.shape}; expected (H, W, 3) uint8")
        intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(-1)
        if intrinsics.size != 4:
            raise ValueError(
                f"intrinsics has {intrinsics.size} values; expected [fx, fy, cx, cy]"
            )

        depth_m = self._validate_depth(depth, frame.shape[:2])

        # `disps_sens` is a persistent buffer and the scale gauge is decided by whether
        # ANY of it is non-zero (droid_backend.py:30). Supplying depth for part of a
        # stream yields a map that is neither metric nor consistently normalised, and
        # nothing raises — so the mode is latched on the first frame.
        if self.rgbd is None:
            self.rgbd = depth_m is not None
        elif self.rgbd != (depth_m is not None):
            raise ValueError(
                f"stream started in {'RGB-D' if self.rgbd else 'monocular'} mode but "
                f"frame {self.frame_index} "
                f"{'omits' if self.rgbd else 'supplies'} depth. DROID decides scale "
                "normalisation from whether disps_sens is non-zero, so a mixed stream "
                "silently produces a half-constrained gauge. Call reset() to switch."
            )

        import lietorch
        import torch

        tensor, scaled = preprocess(frame, intrinsics)
        depth_t = (
            None if depth_m is None else preprocess_depth(depth_m, frame.shape[:2])
        )

        if self.droid is None:
            self.args.image_size = [tensor.shape[2], tensor.shape[3]]
            from droid import Droid

            self.droid = Droid(self.args)

        with torch.no_grad():
            self.droid.track(self.frame_index, tensor, depth_t, intrinsics=scaled)
        self.frame_index += 1

        n_keyframes = int(self.droid.video.counter.value)
        if n_keyframes == 0:
            # No keyframe yet: the motion filter has not seen enough parallax. There is
            # no pose to report and the identity would be a lie.
            return {"n_keyframes": 0, "tracking_lost": True, "metric": self.rgbd}

        # video.poses is world-to-camera; ROS odometry wants camera-to-world.
        world_to_cam = self.droid.video.poses[n_keyframes - 1][None]
        cam_to_world = lietorch.SE3(world_to_cam).inv().data[0].cpu().numpy()

        return {
            "translation": cam_to_world[:3].tolist(),
            "quaternion": cam_to_world[3:].tolist(),   # scalar-LAST, see module docstring
            "n_keyframes": n_keyframes,
            "tracking_lost": not np.isfinite(cam_to_world).all(),
            # Monocular DROID normalises disparity (depth_video.normalize), so its
            # translations carry an arbitrary scale. With depth the gauge is fixed and
            # the units are metres. Everything downstream — TSDF fusion, the object
            # frame, the pre-grasp offset — is wrong in a way that still looks plausible
            # if this is assumed rather than read.
            "metric": bool(self.rgbd),
            "stamp_ns": stamp_ns,
        }

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _validate_depth(depth, image_hw: tuple[int, int]):
        """None, or a (H, W) float32 array of metres aligned to the colour frame."""
        if depth is None:
            return None

        depth_m = np.asarray(depth, dtype=np.float32)
        if depth_m.ndim != 2:
            raise ValueError(
                f"depth has shape {depth_m.shape}; expected (H, W) float metres"
            )
        if depth_m.shape != tuple(image_hw):
            # Not a resize we can do for the caller: an unaligned depth map is a
            # different viewpoint, not a different resolution. Publishers must use
            # aligned_depth_to_color.
            raise ValueError(
                f"depth {depth_m.shape} is not aligned to colour {tuple(image_hw)}. "
                "DROID indexes depth by colour pixel, so a mismatch is a wrong prior "
                "at every pixel, not a scaling problem."
            )
        if not np.isfinite(depth_m).all():
            raise ValueError(
                "depth contains NaN/inf; invalid pixels must be 0, which DROID reads "
                "as 'no prior' (depth_video.append: where(depth > 0, 1/depth, depth))"
            )
        if (depth_m < 0).any():
            raise ValueError("depth contains negative values; expected metres, 0 = invalid")
        return depth_m


def build_slam(weights: Path, **overrides) -> SlamService:
    _require_cuda()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream/droid_slam"))

    # `Droid` reads a flat argparse namespace, so this mirrors demo.py's defaults rather
    # than inventing a config object upstream would not recognise.
    args = SimpleNamespace(
        weights=str(weights), buffer=512, image_size=[384, 512], stereo=False,
        disable_vis=True, upsample=False, beta=0.3, filter_thresh=2.4, warmup=8,
        keyframe_thresh=4.0, frontend_thresh=16.0, frontend_window=25,
        frontend_radius=2, frontend_nms=1, backend_thresh=22.0, backend_radius=2,
        backend_nms=3, frontend_device="cuda", backend_device="cuda",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return SlamService(args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="droid.pth")
    ap.add_argument("--port", type=int, default=5557)  # 5555 GR00T, 5556 DiVLA
    ap.add_argument("--host", default="*")
    ap.add_argument("--buffer", type=int, default=512, help="keyframe buffer size")
    args = ap.parse_args()

    from gr00t.policy.server_client import PolicyServer  # wire-compatible server

    service = build_slam(Path(args.weights), buffer=args.buffer)

    # PolicyServer registers get_action/reset/get_modality_config against `policy`.
    # SLAM has no actions, so `track` is registered explicitly and the policy endpoints
    # are simply never called by the SLAM node.
    server = PolicyServer(service, host=args.host, port=args.port)
    server.register_endpoint("track", service.track)

    print(f"[serve] DROID-SLAM on tcp://{args.host}:{args.port}", flush=True)
    print("[serve] endpoints: ping, reset, track", flush=True)
    server.run()


if __name__ == "__main__":
    main()

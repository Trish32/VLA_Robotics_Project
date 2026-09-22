#!/usr/bin/env python
"""Run the stages that can actually execute locally, and render what they produce.

This is deliberately not a mock. A synthetic room is raycast into real RGB-D frames,
those frames go through the real `SceneWriter` and the real `fuse_tsdf`, and what gets
plotted is the actual fused geometry with the actual on-disk scene layout beside it.

What that covers, and what it cannot:

  stage 1  RGB-D -> poses        SIMULATED. DROID's tracker needs lietorch and
                                 droid_backends, which are CUDA-only. Ground-truth poses
                                 stand in, so everything downstream is exercised with a
                                 metric trajectory — which is the property that matters
                                 to stages 2 and 3.
  stage 2  poses -> scene        REAL. SceneWriter + fuse_tsdf, unmodified.
  stage 3  scene -> instances    NOT RUN. Mask3D loads 0/0/0 but needs SAM + CLIP over a
                                 real capture to mean anything.
  stage 4  instance -> 6D pose   NOT RUN. nvdiffrast is CUDA-only and raises here.
  stage 5  pose -> ROS2          NOT RUN here; covered by the bridge's own suite.

    conda run -n foundationpose_vl python pipeline/tools/demo_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.scene_export import PosedFrame, SceneWriter, fuse_tsdf  # noqa: E402

OUT = ROOT / "pipeline/assets"
W, H = 320, 240
FX = FY = 250.0
CX, CY = W / 2.0, H / 2.0
N_VIEWS = 12

# (name, centre, half-extent, RGB) -- a room with one graspable object in it.
PARTS = [
    ("floor",     (0.0, 0.85, 2.2), (2.0, 0.02, 1.6), (150, 140, 130)),
    ("back wall", (0.0, 0.0, 3.6),  (2.0, 1.2, 0.02), (180, 178, 172)),
    ("left wall", (-1.6, 0.0, 2.2), (0.02, 1.2, 1.6), (165, 163, 158)),
    ("table",     (0.0, 0.35, 2.2), (0.55, 0.03, 0.40), (120, 85, 60)),
    ("pear",      (-0.12, 0.24, 2.10), (0.07, 0.08, 0.07), (140, 170, 60)),
    ("mug",       (0.22, 0.26, 2.25), (0.06, 0.07, 0.06), (200, 200, 205)),
]


def build_scene():
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    colors = []
    for _, centre, half, rgb in PARTS:
        box = o3d.geometry.TriangleMesh.create_box(*(2 * np.array(half)))
        box.translate(np.array(centre) - np.array(half))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(box))
        colors.append(rgb)
    return scene, np.array(colors, np.uint8)


def look_at(eye, target, world_down=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world for an OpenCV-convention camera (+x right, +y down, +z forward).

    The basis is built from WORLD-DOWN, not world-up, because this scene uses the same
    y-down convention the cameras do. Building it from an up vector flips the handedness:
    cross((0,-1,0), (0,0,1)) is -x, so `right` comes out pointing left and every rendered
    frame is mirrored vertically — which looks like a plausible view of the room from
    underneath the floor.
    """
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(world_down, float), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, down, forward
    pose[:3, 3] = eye
    return pose


def render(scene, colors, cam_to_world):
    """Raycast one RGB-D frame. Returns (rgb uint8, depth float32 metres)."""
    import open3d as o3d

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dirs_cam = np.stack([(us - CX) / FX, (vs - CY) / FY, np.ones_like(us)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    dirs_world = dirs_cam @ cam_to_world[:3, :3].T
    origins = np.broadcast_to(cam_to_world[:3, 3], dirs_world.shape)
    rays = o3d.core.Tensor(
        np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    )
    hit = scene.cast_rays(rays)

    t_hit = hit["t_hit"].numpy()
    geom = hit["geometry_ids"].numpy()
    valid = np.isfinite(t_hit)

    # t_hit is distance along a unit ray; DROID and Open3D both want z-depth.
    depth = np.where(valid, t_hit * dirs_cam[..., 2], 0.0).astype(np.float32)

    rgb = np.zeros((H, W, 3), np.uint8)
    for index, colour in enumerate(colors):
        mask = valid & (geom == index)
        # Cheap shading so the render reads as geometry rather than flat labels.
        shade = np.clip(1.15 - depth / 4.0, 0.45, 1.0)[mask][:, None]
        rgb[mask] = np.clip(colour[None, :] * shade, 0, 255).astype(np.uint8)
    return rgb, depth


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    scene, colors = build_scene()

    intrinsics = np.array([FX, FY, CX, CY])
    writer = SceneWriter(OUT, "demo_scene", intrinsics, (H, W), metric=True)

    frames = []
    target = np.array([0.0, 0.25, 2.15])
    for i in range(N_VIEWS):
        angle = np.deg2rad(-38 + 76 * i / (N_VIEWS - 1))
        eye = target + np.array([1.5 * np.sin(angle), -0.55, -1.5 * np.cos(angle)])
        pose = look_at(eye, target)
        rgb, depth = render(scene, colors, pose)
        frame = PosedFrame(color=rgb, depth_m=depth, cam_to_world=pose, stamp_ns=i * 33_000_000)
        writer.add(frame)
        frames.append(frame)

    cloud = fuse_tsdf(frames, _intrinsic_matrix(), (H, W),
                      voxel_length=0.015, sdf_trunc=0.05, depth_trunc=5.0)
    points = np.asarray(cloud.points)
    point_colors = np.asarray(cloud.colors)

    import open3d as o3d
    o3d.io.write_point_cloud(str(writer.ply_path()), cloud)

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), dpi=170)
    axes[0].imshow(frames[N_VIEWS // 2].color)
    axes[0].set_title("stage 1 in — colour (320x240)")
    d = frames[N_VIEWS // 2].depth_m
    im = axes[1].imshow(np.where(d > 0, d, np.nan), cmap="turbo")
    axes[1].set_title("stage 1 in — depth (metres)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    traj = np.stack([f.cam_to_world[:3, 3] for f in frames])
    axes[2].plot(traj[:, 0], traj[:, 2], "o-", ms=3, lw=1, color="#c2410c")
    axes[2].scatter([target[0]], [target[2]], marker="*", s=90, color="#0369a1", zorder=5)
    axes[2].set_aspect("equal")
    axes[2].set_title("stage 1 out — camera track (x-z, metres)")
    axes[2].grid(alpha=0.3)
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "stage1_inputs.png", bbox_inches="tight")
    plt.close(fig)

    # Brighten for display only: the TSDF stores the shaded render, which is dim by
    # design. The .ply keeps the true colours.
    shown = np.clip(point_colors * 1.55, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=170)
    order = np.argsort(-points[:, 2])          # painter's algorithm: far points first
    axes[0].scatter(points[order, 0], -points[order, 1], c=shown[order],
                    s=4.0, marker=".", linewidths=0)
    axes[0].set_aspect("equal")
    axes[0].set_title("stage 2 out — fused cloud (front: x vs -y)")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("height (m)")

    axes[1].scatter(points[:, 0], points[:, 2], c=shown, s=4.0, marker=".", linewidths=0)
    axes[1].plot(traj[:, 0], traj[:, 2], "-", lw=1.4, color="#f97316",
                 label="camera track")
    axes[1].scatter(traj[:, 0], traj[:, 2], s=14, color="#f97316", zorder=5)
    axes[1].set_aspect("equal")
    axes[1].set_title("stage 2 out — fused cloud + track (top-down: x vs z)")
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("z (m)")
    axes[1].legend(loc="lower right", framealpha=0.9)

    for ax in axes:
        ax.grid(alpha=0.2, color="#94a3b8")
        ax.set_facecolor("#f1f5f9")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_fused.png", bbox_inches="tight")
    plt.close(fig)

    stats = {
        "views": N_VIEWS,
        "image_hw": [H, W],
        "intrinsics_fx_fy_cx_cy": [FX, FY, CX, CY],
        "fused_points": int(len(points)),
        "cloud_extent_m": (points.max(axis=0) - points.min(axis=0)).round(3).tolist(),
        "trajectory_length_m": round(float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()), 3),
        "scene_dir": str(writer.scene_dir().relative_to(ROOT)),
        "files_written": sorted(p.name for p in (writer.scene_dir() / "color").iterdir())[:3],
        "ply": str(writer.ply_path().relative_to(ROOT)),
    }
    (OUT / "demo_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


def _intrinsic_matrix() -> np.ndarray:
    k = np.eye(4)
    k[0, 0], k[1, 1], k[0, 2], k[1, 2] = FX, FY, CX, CY
    return k


if __name__ == "__main__":
    raise SystemExit(main())

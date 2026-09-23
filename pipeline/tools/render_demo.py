#!/usr/bin/env python
"""Render the end-to-end run as README media: GIF + full-res MP4 + PNG still.

CLAUDE.md's media convention, and the reason for it: GitHub renders a committed `.mp4`
as a download LINK, not a player, so anything meant to play inline in a README has to be
a GIF. The mp4 and png ship alongside for the resolution a GIF cannot hold — 128 colours
and no dithering is what keeps the GIF under the ~8 MB that renders reliably.

Rendered from the run's own artifacts rather than screen-captured from the Rerun viewer.
Screen capture would need a GUI session, would not be reproducible, and would bake in
whatever panel layout happened to be open. This is a command anyone can re-run.

A DASHBOARD, not a slideshow. An earlier version showed six flat 2-D acts in sequence,
which read as six separate pictures of six separate programs. This stack's whole claim is
that the stages feed each other, so every panel is on screen the whole time and the
*state* moves: the cloud accumulates while the camera flies, instance boxes resolve out
of it in place, the graph fills in beside them, and the chunk executes underneath. The
3-D view orbits continuously, which is also what makes a point cloud readable as
geometry rather than as a smear.

    conda run -n foundationpose_vl python pipeline/tools/render_demo.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

E2E = ROOT / "pipeline/assets/e2e"
DATA = ROOT / "orbslam3_baseline/data"
ASSETS = ROOT / "pipeline/assets"

FX = FY = CX = CY = 0.0     # filled from fuse.json's camera
SEQ = None

PALETTE = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f"]
INK, MUTED, PANEL, GRID = "#10151c", "#5b6874", "#f2f5f8", "#dde4ea"
LIVE, TRACK = "#94a3b2", "#be123c"
# Stage 4 draws a REFUSED result, so it gets its own colour rather than borrowing
# the target's -- a reader should not have to read the caption to tell them apart.
BAD = "#c2410c"


def unproject(depth, T, stride=7, far=4.0):
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = (z > 0.1) & (z < far)
    if not ok.any():
        return np.empty((0, 3))
    z = z[ok]
    cam = np.column_stack([(us[ok] - CX) * z / FX, (vs[ok] - CY) * z / FY, z])
    return cam @ T[:3, :3].T + T[:3, 3]


def frustum(T, scale=0.28):
    """Camera body as 4 rays to the image corners — shows where it is AND facing."""
    corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], float)
    rays = np.column_stack([(corners[:, 0] - CX) / FX, (corners[:, 1] - CY) / FY,
                            np.ones(4)]) * scale
    world = rays @ T[:3, :3].T + T[:3, 3]
    origin = T[:3, 3]
    segs = [np.vstack([origin, w]) for w in world]
    segs.append(np.vstack([world, world[:1]]))
    return segs


def box_edges(lo, hi):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]],
                  [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [c[[a, b]] for a, b in pairs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gif-width", type=int, default=800)
    ap.add_argument("--stride", type=int, default=12, help="every Nth camera frame")
    ap.add_argument("--cloud-points", type=int, default=9000,
                    help="subsample for the 3-D view; matplotlib is the bottleneck")
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    from pipeline.identity import Frames, InstanceRegistry, Stamp
    from pipeline.observations import from_openmask3d, to_scene_nodes
    from pipeline.scene_graph import NodeKind, SceneGraph
    from pipeline.transforms import quaternion_to_matrix

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; brew install ffmpeg")

    # ----------------------------------------------------------------- inputs
    fuse = json.load(open(E2E / "fuse.json"))
    T_level = (np.asarray(fuse["T_level_slam"], float)
               if fuse.get("levelled") else np.eye(4))

    # Sequence, poses and intrinsics all follow fuse.json — the scene is a flag now.
    from pipeline.tools.e2e_tum import INTRINSICS
    global SEQ, FX, FY, CX, CY
    SEQ = DATA / fuse["sequence"]
    FX, FY, CX, CY = INTRINSICS[fuse["camera"]]
    traj_file = DATA / fuse["trajectory"]
    scene_name = fuse["sequence"].replace("rgbd_dataset_", "")

    traj = []
    for line in traj_file.read_text().splitlines():
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_matrix(v[4], v[5], v[6], v[7])
        T[:3, 3] = v[1:4]
        traj.append((v[0], T_level @ T))        # levelled, like the cloud
    traj_t = np.array([t for t, _ in traj])

    rgb = [(float(l.split()[0]), l.split()[1])
           for l in (SEQ / "rgb.txt").read_text().splitlines() if not l.startswith("#")]
    dep = [(float(l.split()[0]), l.split()[1])
           for l in (SEQ / "depth.txt").read_text().splitlines() if not l.startswith("#")]
    dep_t = np.array([t for t, _ in dep])

    cloud = o3d.io.read_point_cloud(fuse["ply"])
    points = np.asarray(cloud.points)
    colours = np.asarray(cloud.colors)
    masks = np.load(E2E / "instance_masks.npz")["masks"]
    if masks.shape[1] != len(points):
        raise SystemExit(
            f"masks index {masks.shape[1]} points, cloud has {len(points)} — these are "
            "from different runs. Re-run e2e_segment.py against the current fuse.json."
        )
    labelled = json.load(open(E2E / "labelled.json"))["instances"]
    ground = json.load(open(E2E / "ground.json"))
    policy = json.load(open(E2E / "policy.json"))
    # Stage 4 is optional: the chain still runs without a pose result, and the demo says
    # so rather than silently dropping a stage.
    pose_path = E2E / "pose.json"
    pose = json.load(open(pose_path)) if pose_path.exists() else None
    chunk = {k: np.asarray(v) for k, v in policy.get("values", {}).items()}

    NS = 1_000_000_000
    keep = [i for i, r in enumerate(labelled) if r.get("label")]
    registry = InstanceRegistry()
    grouped: dict[str, list] = {}
    for i in keep:
        ident, _ = registry.associate(labelled[i]["centre"], labelled[i]["extent"],
                                      labelled[i]["label"], Stamp(NS))
        grouped.setdefault(ident, []).append(i)

    observations = from_openmask3d(
        [np.flatnonzero(masks[i]) for i in keep], points,
        [labelled[i]["label"] for i in keep], [labelled[i]["similarity"] for i in keep],
        anchor_frame=Frames.keyframe(0), stamp_ns=NS, registry=InstanceRegistry())
    nodes = {n.node_id: n for n in to_scene_nodes(observations)}
    graph = SceneGraph()
    for node in nodes.values():
        graph.upsert(node)
    relations = graph.infer_relations(NS)
    # The target is whatever stage 5 actually grounded, not a hardcoded phrase — the
    # scene is a flag now, and "the monitor" does not exist in every scene.
    target = ground["target"]
    if target not in nodes:
        raise SystemExit(
            f"ground.json targets {target!r} but the graph holds {sorted(nodes)}. "
            "Re-run e2e_ground.py against the current segment/label output."
        )
    order = list(nodes)
    colour_of = {ident: PALETTE[n % len(PALETTE)] for n, ident in enumerate(order)}

    lo, hi = points.min(0), points.max(0)
    extent = hi - lo
    rs = np.random.default_rng(0)
    pick = rs.choice(len(points), min(args.cloud_points, len(points)), replace=False)
    cloud_pts, cloud_col = points[pick], colours[pick]

    # Which instance owns each sampled point. Colouring the CLOUD by instance shows the
    # segmentation itself; six nested wireframe boxes mostly show how much the proposals
    # overlap, which on this scene is a lot and reads as noise.
    member_of = np.full(len(points), -1, np.int16)
    for n, ident in enumerate(order):
        member_of[np.any([masks[i] for i in grouped[ident]], axis=0)] = n
    pick_member = member_of[pick]

    tmp = Path(tempfile.mkdtemp(prefix="e2e_demo_"))
    frame_no = 0

    # --------------------------------------------------------------- timeline
    chosen = []
    for k, (t, rel) in enumerate(rgb):
        if k % args.stride:
            continue
        j = int(np.argmin(np.abs(traj_t - t)))
        if abs(traj_t[j] - t) <= 0.02:
            chosen.append((t, rel, traj[j][1]))
    n_fly = len(chosen)
    n_seg = 5 * len(order) + 6           # instances resolving, then a hold
    n_pose = 26 if pose else 0           # stage 4: the pose, and the gate's verdict
    n_gnd = 22                           # grounding
    n_act = 3 * (len(next(iter(chunk.values()))) if chunk else 0) + 6
    total = n_fly + n_seg + n_pose + n_gnd + n_act
    horizon = len(next(iter(chunk.values()))) if chunk else 0
    print(f"[render]  {total} frames = {n_fly} fly + {n_seg} segment + {n_pose} pose "
          f"+ {n_gnd} ground + {n_act} act", flush=True)

    accumulated, track, metrics = [], [], []
    rgb_img = depth_img = None

    def save(fig):
        nonlocal frame_no
        fig.savefig(tmp / f"{frame_no:05d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_no += 1

    for f in range(total):
        phase = ("fly" if f < n_fly else
                 "segment" if f < n_fly + n_seg else
                 "pose" if f < n_fly + n_seg + n_pose else
                 "ground" if f < n_fly + n_seg + n_pose + n_gnd else "act")

        if phase == "fly":
            t, rel, T = chosen[f]
            depth = cv2.imread(str(SEQ / dep[int(np.argmin(np.abs(dep_t - t)))][1]),
                               cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
            rgb_img = cv2.cvtColor(cv2.imread(str(SEQ / rel)), cv2.COLOR_BGR2RGB)
            depth_img = np.where(depth > 0, depth, np.nan)
            accumulated.append(unproject(depth, T))
            track.append(T[:3, 3])
            metrics.append(sum(len(a) for a in accumulated))
            live = np.vstack(accumulated)
            cam_T = T
        else:
            live, cam_T = None, chosen[-1][2]

        shown = (order[: min(len(order), (f - n_fly) // 5 + 1)]
                 if phase == "segment" else order if phase != "fly" else [])
        step = (f - n_fly - n_seg - n_pose - n_gnd) // 3 if phase == "act" else 0
        # How far through the pose stage we are, so the checks appear one at a time
        # instead of all at once -- each one is a separate claim.
        pose_k = (f - n_fly - n_seg) if phase == "pose" else 0

        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        fig.patch.set_facecolor("#ffffff")

        # ------------------------------------------------------------ header
        fig.text(0.012, 0.963, "OBJECT-CENTRIC RGB-D PERCEPTION", fontsize=12,
                 weight="bold", color=INK, family="monospace")
        fig.text(0.012, 0.932, f"TUM {scene_name}  ·  dynamic scene",
                 fontsize=9, color=MUTED, family="monospace")
        stage_of = {"fly": "1–2  localize + fuse", "segment": "3  segment + label",
                    "pose": "4  6-DoF pose", "ground": "5  scene graph",
                    "act": "6  action chunk"}
        fig.text(0.70, 0.963, f"STAGE {stage_of[phase]}", fontsize=11, weight="bold",
                 color=PALETTE[0], family="monospace")
        fig.text(0.70, 0.932, "ORB-SLAM3 · Mask3D · CLIP · graph · GR00T",
                 fontsize=8.5, color=MUTED, family="monospace")

        # --------------------------------------------------- left: sensor feeds
        for row, (img, title, kw) in enumerate([
                (rgb_img, "RGB", {}),
                (depth_img, "depth (m)", dict(cmap="viridis", vmin=0.4, vmax=4.0))]):
            ax = fig.add_axes([0.012, 0.605 - row * 0.29, 0.185, 0.265])
            if img is not None:
                ax.imshow(img, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=8.5, color=MUTED)
            for sp in ax.spines.values():
                sp.set_color(GRID)

        # points-fused sparkline
        ax = fig.add_axes([0.012, 0.135, 0.185, 0.165])
        if metrics:
            ax.plot(metrics, lw=1.6, color=PALETTE[0])
            ax.fill_between(range(len(metrics)), metrics, color=PALETTE[0], alpha=0.14)
            ax.scatter([len(metrics) - 1], [metrics[-1]], s=18, color=PALETTE[0], zorder=5)
        ax.set_xlim(0, max(1, n_fly)); ax.set_ylim(0, max(metrics or [1]) * 1.15)
        ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.25, color=GRID)
        # "unprojected", not "fused": this counts raw points pushed out of each depth
        # image, which is what the animation is actually showing accumulate. The TSDF
        # cloud those condense into is smaller (133,928 here), and labelling this
        # "fused" quietly claimed the wrong number.
        ax.set_title(f"points unprojected   {metrics[-1] if metrics else 0:,}",
                     fontsize=8.5, color=MUTED)

        # ------------------------------------------------- centre: 3-D scene
        ax = fig.add_axes([0.205, 0.10, 0.47, 0.80], projection="3d")
        ax.set_facecolor("#ffffff")
        ax.view_init(elev=26 + 7 * np.sin(f / 44), azim=-72 + f * 0.55)
        if live is not None and len(live):
            ax.scatter(live[::2, 0], live[::2, 1], live[::2, 2], s=1.3, c=LIVE,
                       marker=".", linewidths=0, depthshade=False, alpha=0.8)
        else:
            shown_idx = [order.index(i) for i in shown]
            rest = ~np.isin(pick_member, shown_idx)
            ax.scatter(cloud_pts[rest, 0], cloud_pts[rest, 1], cloud_pts[rest, 2],
                       s=1.6, c=cloud_col[rest], marker=".", linewidths=0,
                       depthshade=False, alpha=0.55)
            for ident in shown:
                sel = pick_member == order.index(ident)
                if sel.any():
                    ax.scatter(cloud_pts[sel, 0], cloud_pts[sel, 1], cloud_pts[sel, 2],
                               s=3.2, c=colour_of[ident], marker=".", linewidths=0,
                               depthshade=False)
        if track:
            tr = np.array(track)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=1.5, color=TRACK)
        for seg in frustum(cam_T):
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=1.0, color=TRACK, alpha=0.85)

        # Stage 4: draw FoundationPose's answer next to where our map puts the object.
        # The disagreement is the result, so it should be visible rather than asserted
        # in a caption -- a reader can see 57 cm and a flipped frame at a glance.
        if phase == "pose" and pose and pose.get("rejected_pose_world"):
            Tp = np.asarray(pose["rejected_pose_world"], float)
            fp, mp_ = Tp[:3, 3], np.asarray(nodes[target].centre, float)
            ax.plot([fp[0], mp_[0]], [fp[1], mp_[1]], [fp[2], mp_[2]],
                    lw=1.4, color=BAD, ls="--", alpha=0.95)
            ax.scatter(*fp[:, None], s=70, color=BAD, marker="X", depthshade=False,
                       zorder=6)
            # The estimated frame's axes, which is where the 175 deg shows up: a flipped
            # pose draws its axes pointing the wrong way, and no number is needed.
            for d, col in enumerate((BAD, "#d98d3a", "#9a6fb0")):
                a = Tp[:3, d] * 0.30
                ax.plot([fp[0], fp[0] + a[0]], [fp[1], fp[1] + a[1]],
                        [fp[2], fp[2] + a[2]], lw=1.8, color=col, alpha=0.95)
            # No 3-D text here: matplotlib does not collide-check it, and at this
            # camera distance "FoundationPose" landed on top of the target's own label.
            # The marker is keyed in the legend instead, where it can be read.

        # Only the TARGET gets a wireframe. It is the one box a planner acts on.
        if phase in ("pose", "ground", "act") or target in shown:
            tnode = nodes[target]
            tlo, thi = tnode.aabb()
            for seg in box_edges(tlo, thi):
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=1.5, color=TRACK, alpha=0.9)
            ax.text(tnode.centre[0], tnode.centre[1], thi[2] + 0.10, target,
                    fontsize=8, color=TRACK, ha="center", weight="bold")

        # True proportions. A cube aspect squashes a 4.9 x 3.7 x 1.6 m room into a slab
        # and makes the floor read as a wall.
        pad = 0.12
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect(tuple(extent))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0, colors=GRID)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("#fbfcfd")
            pane.pane.set_edgecolor(GRID)
            pane.line.set_color(GRID)
            pane._axinfo["grid"]["color"] = GRID
        ax.set_title(
            f"world model — {len(track)} poses, "
            f"{(metrics[-1] if metrics else 0):,} pts, {len(shown)} instances",
            fontsize=9, color=MUTED, y=0.97)

        # ------------------------------------------------ right: compact legend
        # Deliberately terse. The previous version printed the node list, every
        # relation, the resolve line and the wrapped prompt paragraph -- so the panel
        # that was meant to support the 3-D view competed with it for attention. What
        # survives is what the picture cannot show by itself.
        y = 0.855
        fig.text(0.695, y, "INSTANCES", fontsize=9, weight="bold", color=INK,
                 family="monospace")
        y -= 0.040
        for ident in shown:
            mark = "→" if ident == target and phase in ("pose", "ground", "act") else "■"
            fig.text(0.70, y, f"{mark} {ident}", fontsize=9, family="monospace",
                     color=colour_of[ident])
            y -= 0.032
        if not shown:
            fig.text(0.70, y, "…", fontsize=9, color=MUTED, family="monospace")

        if phase in ("ground", "act") and relations:
            fig.text(0.695, y - 0.045, relations[0].as_text(nodes), fontsize=9.5,
                     family="monospace", color=INK)
            if len(relations) > 1:
                fig.text(0.695, y - 0.077, f"+{len(relations) - 1} more relation"
                         f"{'s' if len(relations) > 2 else ''}", fontsize=8.5,
                         color=MUTED, family="monospace")

        # ------------------------------------------------- stage 4: the gate
        # Revealed one line at a time. Each is an independent reason, and showing them
        # together would read as one verdict instead of four agreeing ones.
        if phase == "pose" and pose:
            rows = [
                (f"translation  {pose['median_translation_cm']:6.1f} cm",
                 pose["median_translation_cm"] <= pose["thresholds"]["translation_cm"]),
                (f"rotation     {pose['median_rotation_deg']:6.1f}°",
                 pose["median_rotation_deg"] <= pose["thresholds"]["rotation_deg"]),
                (f"depth        {abs(pose['per_frame'][0]['behind_surface_cm']):6.1f} cm"
                 " behind", not any("behind" in r for r in pose["per_frame"][0]["reasons"])),
                (f"convergence  {pose.get('agreement', {}).get('clustered_within_15deg', 0)}"
                 f"/{pose.get('agreement', {}).get('k', 0)} agree",
                 bool(pose.get("agreement_ok"))),
            ]
            fig.text(0.695, y - 0.055, "STAGE 4 GATE", fontsize=9, weight="bold",
                     color=INK, family="monospace")
            if pose.get("rejected_pose_world"):
                gap = np.linalg.norm(np.asarray(pose["rejected_pose_world"], float)[:3, 3]
                                     - np.asarray(nodes[target].centre, float)) * 100
                fig.text(0.70, y - 0.088, f"✕ FoundationPose — {gap:.0f} cm off",
                         fontsize=8.5, family="monospace", color=BAD)
            yy = y - 0.126
            for n, (text, ok) in enumerate(rows):
                if pose_k < 4 + n * 4:
                    break
                fig.text(0.70, yy, f"{'PASS' if ok else 'FAIL'}  {text}", fontsize=8.5,
                         family="monospace", color=(PALETTE[0] if ok else BAD))
                yy -= 0.030
            if pose_k >= 19:
                verdict = "POSE ADMITTED" if pose["accepted"] else "POSE REFUSED"
                fig.text(0.695, yy - 0.020, verdict, fontsize=10.5, weight="bold",
                         family="monospace", color=(PALETTE[0] if pose["accepted"] else BAD))
                fig.text(0.695, yy - 0.050,
                         "position-only pose stands" if not pose["accepted"] else
                         "6-DoF pose enters the graph",
                         fontsize=8.5, family="monospace", color=MUTED)

        # ------------------------------------------------ bottom: action chunk
        for n, (group, arr) in enumerate(chunk.items()):
            ax = fig.add_axes([0.695 + n * 0.0615, 0.10, 0.052, 0.15])
            for d in range(arr.shape[1]):
                ax.plot(arr[:, d], lw=0.8, color="#c9d3dc")
                if phase == "act" and step:
                    ax.plot(range(min(step, horizon)), arr[:min(step, horizon), d],
                            lw=1.3, color=PALETTE[n % len(PALETTE)])
            if phase == "act":
                ax.axvline(min(step, horizon - 1), color=TRACK, lw=1.0, ls="--")
            ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25, color=GRID)
            ax.set_title(group.replace("_", " "), fontsize=6.5, color=MUTED, pad=2)
        if chunk:
            label = (f"GR00T  {min(step, horizon)}/{horizon}"
                     if phase == "act" else
                     f"GR00T  {horizon} × "
                     f"{sum(a.shape[1] for a in chunk.values())} DoF")
            fig.text(0.695, 0.268, label, fontsize=8.5, color=MUTED, family="monospace")

        # The footnote used to say 6-DoF refinement "was not run here". It has been run
        # since, and refused -- which is a different and more useful statement.
        if pose and not pose["accepted"]:
            note = (f"6-DoF pose measured and REFUSED "
                    f"({pose['median_rotation_deg']:.0f}° off) — box is the "
                    f"axis-aligned extent, rotation deliberately unclaimed")
        elif pose:
            note = "6-DoF pose admitted by the stage-4 gate"
        else:
            note = "6-DoF refinement not run here — box is an axis-aligned extent"
        fig.text(0.012, 0.035, note, fontsize=8, color=MUTED, family="monospace")
        save(fig)
        if frame_no % 25 == 0:
            print(f"[render]  {frame_no}/{total}", flush=True)

    # ------------------------------------------------------------- encode
    mp4, gif, still = (ASSETS / "e2e_demo.mp4", ASSETS / "e2e_demo.gif",
                       ASSETS / "e2e_demo.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", str(mp4)], check=True)

    # stats_mode=full, NOT diff: diff weights the palette toward what CHANGES, which is
    # the flying phase's greys, and quantised the instance colours to near-black.
    palette = tmp / "palette.png"
    vf = f"fps={args.fps},scale={args.gif_width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"{vf},palettegen=max_colors=128:stats_mode=full", str(palette)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i",
                    str(palette), "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=none",
                    str(gif)], check=True)

    still_at = (n_fly + n_seg + n_pose - 2 if n_pose
                else n_fly + n_seg + n_gnd - 3)
    shutil.copy(tmp / f"{still_at:05d}.png", still)
    shutil.rmtree(tmp, ignore_errors=True)

    for path in (gif, mp4, still):
        print(f"[render]  {path.name:16} {path.stat().st_size / 1e6:6.2f} MB")
    if gif.stat().st_size > 8e6:
        print("[render]  WARNING: gif over ~8 MB; raise --stride or drop --gif-width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

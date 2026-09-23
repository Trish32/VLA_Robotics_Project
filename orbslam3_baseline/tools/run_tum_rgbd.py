#!/usr/bin/env python
"""Run the ORB-SLAM3 binding over a TUM-RGBD sequence and write a TUM-format trajectory.

The first end-to-end execution of a tracker in this repo. DROID-SLAM's frontend is
CUDA-only and has never run here; ORB-SLAM3 is pure CPU, which is the entire reason it
was built as the baseline.

Two conversions are done here rather than in the binding, and both are the kind that
fail silently:

**Depth to metres.** TUM stores depth as uint16 with 5000 counts per metre. The binding's
contract is float32 METRES (matching `depth_to_metres` on the ROS side), so the generated
settings file carries `RGBD.DepthMapFactor: 1.0`. Leaving the stock 5000.0 while also
dividing here would scale the map by 5000 — ORB-SLAM3 would still track, and every
distance would be wrong.

**RGB/depth association.** TUM's two streams are not synchronised and have separate
timestamp files; the dataset ships `associate.py` for this. Pairing by index instead of
by timestamp silently offsets colour from depth by a frame or two, which reads as a
badly calibrated sensor rather than as a bug.

    python orbslam3_baseline/tools/run_tum_rgbd.py --sequence <dir> [--max-frames N]
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
MAX_DT = 0.02          # TUM's own associate.py default


def read_index(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        stamp, rel = line.split()
        rows.append((float(stamp), rel))
    return rows


def associate(rgb, depth, max_dt=MAX_DT):
    """Pair each colour frame with the nearest depth frame in time."""
    depth_stamps = np.array([s for s, _ in depth])
    pairs = []
    for stamp, rgb_rel in rgb:
        j = int(np.argmin(np.abs(depth_stamps - stamp)))
        if abs(depth_stamps[j] - stamp) <= max_dt:
            pairs.append((stamp, rgb_rel, depth[j][1]))
    return pairs


def write_settings(source: Path, target: Path) -> Path:
    """Copy the stock TUM settings, forcing DepthMapFactor to 1.0.

    We hand the tracker metres; the stock 5000.0 would divide them again.
    """
    out = []
    for line in source.read_text().splitlines():
        if line.strip().startswith("RGBD.DepthMapFactor"):
            out.append("RGBD.DepthMapFactor: 1.0  # we pass float32 metres")
        else:
            out.append(line)
    target.write_text("\n".join(out) + "\n")
    return target


def confirm_moving(prev_gray, gray, prev_depth, depth, boxes, K, orb, matcher):
    """Which detected boxes does GEOMETRY agree are moving?

    The semantic prior alone would delete a seated person and a parked bicycle — good
    static structure, and exactly the well-textured regions ORB depends on. So the boxes
    only mark suspects; ORB correspondences plus the epipolar and depth tests decide.

    Returns (moving boxes, whether the frame pair was geometrically degenerate).
    """
    import cv2
    import numpy as np

    from orbslam3_baseline.dynamic_filter import (
        DegenerateMotion, Detection, FilterConfig, filter_dynamic,
    )

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    if des1 is None or des2 is None or len(kp1) < 40 or len(kp2) < 40:
        return [], True

    matches = matcher.match(des1, des2)
    if len(matches) < 40:
        return [], True

    prev_pts = np.array([kp1[m.queryIdx].pt for m in matches])
    curr_pts = np.array([kp2[m.trainIdx].pt for m in matches])
    d_prev = np.array([prev_depth[min(int(p[1]), prev_depth.shape[0] - 1),
                                  min(int(p[0]), prev_depth.shape[1] - 1)]
                       for p in prev_pts])
    d_curr = np.array([depth[min(int(p[1]), depth.shape[0] - 1),
                             min(int(p[0]), depth.shape[1] - 1)]
                       for p in curr_pts])

    dets = [Detection(*b["xyxy"], label=b["label"], score=b["conf"]) for b in boxes]
    try:
        result = filter_dynamic(prev_pts, curr_pts, dets, FilterConfig(),
                                prev_depth=d_prev, curr_depth=d_curr, K=K)
    except DegenerateMotion:
        # Too little parallax, or the boxes cover so much of the frame that F would have
        # to be fitted on the points it is meant to judge. Masking on a meaningless test
        # is worse than not masking.
        return [], True

    return [boxes[i] for i in result.culled_boxes], False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path,
                    default=HERE / "data/rgbd_dataset_freiburg1_xyz")
    ap.add_argument("--settings", type=Path,
                    default=HERE / "upstream/Examples/RGB-D/TUM1.yaml")
    ap.add_argument("--vocabulary", type=Path,
                    default=HERE / "upstream/Vocabulary/ORBvoc.txt")
    ap.add_argument("--out", type=Path, default=HERE / "data/trajectory_orbslam3.txt")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--detections", type=Path, default=None,
                    help="dynamic_detections.json from detect_dynamic.py")
    ap.add_argument("--reject-dynamic", action="store_true",
                    help="mask regions geometry CONFIRMS are moving")
    args = ap.parse_args()

    faulthandler.enable()          # print the Python frame on SIGBUS/SIGSEGV
    sys.path.insert(0, str(HERE / "bindings/build"))
    import cv2
    import orbslam3_py

    from pipeline.transforms import matrix_to_quaternion

    # fr1 and fr3 have different intrinsics; read them from the settings file in use.
    import re as _re
    cfg = args.settings.read_text()
    fx, fy, cx, cy = (float(_re.search(rf"Camera1?\.{k}:\s*([0-9.]+)", cfg).group(1))
                      for k in ("fx", "fy", "cx", "cy"))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    pairs = associate(read_index(args.sequence / "rgb.txt"),
                      read_index(args.sequence / "depth.txt"))
    if args.max_frames:
        pairs = pairs[: args.max_frames]
    print(f"[data] {len(pairs)} associated RGB-D pairs from {args.sequence.name}")

    settings = write_settings(args.settings, HERE / "data/_tum_metres.yaml")
    tracker = orbslam3_py.Tracker(str(args.vocabulary), str(settings), False)

    detections = {}
    if args.detections:
        import json
        blob = json.loads(args.detections.read_text())
        detections = {f["file"]: f["boxes"] for f in blob["frames"]}
        print(f"[detect] {len(detections)} frames of cached YOLO+ByteTrack boxes")

    # ORB-SLAM3's LocalMapping and LoopClosing threads keep running after TrackRGBD
    # returns (the binding releases the GIL), so OpenCV is live on their threads while
    # Python calls into it here. Sharing OpenCV's internal parallel_for pool across both
    # is the most likely source of the SIGBUS at ~frame 700, and costs nothing to rule
    # out: this side of the work is single-frame and does not need the pool.
    cv2.setNumThreads(0)
    orb = cv2.ORB_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray = prev_depth = None
    masked_frames = masked_boxes = degenerate = 0

    poses, lost, started = [], 0, time.monotonic()
    for index, (stamp, rgb_rel, depth_rel) in enumerate(pairs):
        colour = cv2.imread(str(args.sequence / rgb_rel), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(args.sequence / depth_rel), cv2.IMREAD_ANYDEPTH)
        if colour is None or raw is None:
            raise SystemExit(f"could not read frame {index}")
        # uint16 counts -> metres. 0 stays 0: TUM's no-return sentinel, same as ours.
        depth = (raw.astype(np.float32) / 5000.0)

        if args.reject_dynamic and detections:
            gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
            boxes = detections.get(rgb_rel.split("/")[-1], [])
            if prev_gray is not None and boxes:
                moving, was_degenerate = confirm_moving(
                    prev_gray, gray, prev_depth, depth, boxes, K, orb, matcher
                )
                degenerate += int(was_degenerate)
                for box in moving:
                    x1, y1, x2, y2 = (int(round(v)) for v in box["xyxy"])
                    # Zeroing leaves a hard rectangle edge that ORB will happily corner-
                    # detect. Cheaper than inpainting and the ablation measures whether
                    # the trade is worth it; the alternative is patching ORBextractor,
                    # whose mask argument upstream documents as ignored.
                    colour[max(0, y1):y2, max(0, x1):x2] = 0
                    depth[max(0, y1):y2, max(0, x1):x2] = 0.0
                masked_boxes += len(moving)
                masked_frames += int(bool(moving))
            prev_gray, prev_depth = gray, depth.copy()

        reply = tracker.track_rgbd(np.ascontiguousarray(colour), depth, stamp)
        if reply["tracking_lost"]:
            lost += 1
            continue
        poses.append((stamp, np.asarray(reply["camera_to_world"])))

        if index % 100 == 0:
            print(f"  frame {index:4d}/{len(pairs)}  state={reply['tracking_state']}  "
                  f"tracked={len(poses)}", flush=True)

    elapsed = time.monotonic() - started
    tracker.shutdown()

    with args.out.open("w") as handle:
        for stamp, T in poses:
            t = T[:3, 3]
            qx, qy, qz, qw = matrix_to_quaternion(T[:3, :3])
            handle.write(f"{stamp:.6f} {t[0]:.7f} {t[1]:.7f} {t[2]:.7f} "
                         f"{qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")

    print(f"\n[done] tracked {len(poses)}/{len(pairs)} frames "
          f"({100 * len(poses) / max(1, len(pairs)):.1f}%), {lost} lost")
    print(f"[perf] {elapsed:.1f} s -> {len(pairs) / elapsed:.1f} FPS on CPU")
    if args.reject_dynamic:
        print(f"[reject] masked {masked_boxes} confirmed-moving boxes across "
              f"{masked_frames} frames; {degenerate} frames had too little parallax "
              f"for the geometric test")
    print(f"[out]  {args.out}")
    if poses:
        track = np.stack([T[:3, 3] for _, T in poses])
        length = float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
        print(f"[traj] {length:.2f} m travelled, extent "
              f"{(track.max(0) - track.min(0)).round(3).tolist()} m")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    raise SystemExit(main())

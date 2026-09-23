#!/usr/bin/env python
"""YOLO + ByteTrack over a TUM sequence, cached to JSON for the SLAM run to consume.

Run as a SEPARATE PASS, in its own environment, on purpose. Two reasons:

**Environments.** conda-forge OpenCV (which ORB-SLAM3 links) and pip torch (which
ultralytics needs) both ship libomp, and loading both in one process aborts with
`OMP: Error #15`. Caching detections decouples them entirely.

**Repeatability.** Detections are a deterministic function of the sequence, so caching
means the SLAM ablation reruns against byte-identical input. A baseline and a filtered
run that saw different detections are not an ablation.

WHAT THE TRACKER ADDS OVER RAW DETECTION
----------------------------------------
Per-frame YOLO flickers: a chair edge reads as a person for one frame, a real person
drops out for one. Either costs something — a spurious box removes good static features,
a dropout lets a moving person back into the pose estimate.

ByteTrack gives each detection an identity across frames, which makes a cheap and
effective consistency test available: require a track to have been seen `min_track_len`
times before trusting it. A one-frame false positive never accumulates a track; a real
person does within a few frames. That is the "temporal consistency" contribution, and it
is measured in the ablation rather than assumed.

    python orbslam3_baseline/tools/detect_dynamic.py --sequence <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# COCO classes that move under their own power. Kept narrow: the semantic prior's job is
# to mark SUSPECTS for the geometry to judge, not to decide what is moving.
MOVABLE = {"person", "cat", "dog", "bicycle", "car", "motorcycle", "bird", "horse"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-track-len", type=int, default=3,
                    help="frames a track must appear in before it is trusted")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    rgb_dir = args.sequence / "rgb"
    frames = sorted(rgb_dir.glob("*.png"), key=lambda p: float(p.stem))
    if not frames:
        raise SystemExit(f"no PNGs under {rgb_dir}")

    model = YOLO(args.model)
    print(f"[detect] {len(frames)} frames, model={args.model}, "
          f"tracker=bytetrack, conf={args.conf}")

    raw: list[dict] = []
    track_counts: dict[int, int] = defaultdict(int)

    for index, path in enumerate(frames):
        results = model.track(str(path), persist=True, tracker="bytetrack.yaml",
                              conf=args.conf, verbose=False)
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label not in MOVABLE:
                    continue
                track_id = int(box.id) if box.id is not None else -1
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                boxes.append({"track_id": track_id, "label": label,
                              "conf": float(box.conf), "xyxy": [x1, y1, x2, y2]})
                if track_id >= 0:
                    track_counts[track_id] += 1
        raw.append({"stamp": float(path.stem), "file": path.name, "boxes": boxes})
        if index % 100 == 0:
            print(f"  {index:4d}/{len(frames)}  boxes={len(boxes)}", flush=True)

    # Temporal-consistency filter: drop tracks too short to be real. This is the step
    # that raw per-frame detection cannot do, and the reason the tracker is here.
    trusted = {tid for tid, n in track_counts.items() if n >= args.min_track_len}
    kept = dropped = 0
    for frame in raw:
        before = len(frame["boxes"])
        frame["boxes"] = [b for b in frame["boxes"]
                          if b["track_id"] in trusted or b["track_id"] < 0]
        kept += len(frame["boxes"])
        dropped += before - len(frame["boxes"])

    out = args.out or (args.sequence / "dynamic_detections.json")
    out.write_text(json.dumps({
        "sequence": args.sequence.name,
        "model": args.model,
        "min_track_len": args.min_track_len,
        "tracks_seen": len(track_counts),
        "tracks_trusted": len(trusted),
        "boxes_kept": kept,
        "boxes_dropped_by_track_length": dropped,
        "frames": raw,
    }, indent=1))

    print(f"\n[tracks]  {len(track_counts)} seen, {len(trusted)} survived "
          f">= {args.min_track_len} frames")
    print(f"[boxes]   {kept} kept, {dropped} dropped as short-lived false positives "
          f"({100 * dropped / max(1, kept + dropped):.1f}%)")
    print(f"[out]     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

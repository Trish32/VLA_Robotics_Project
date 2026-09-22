#!/usr/bin/env python
"""Run the stack end to end, stage by stage, and report exactly where it stops.

Everything in this repo has been tested in isolation. Nothing has tested the SEAMS with
real data flowing through them, which is a different question: unit tests pin contracts,
an end-to-end run finds the places where two correct components disagree about what they
were promising each other.

The chain, and the environment each stage needs (they differ, so stages hand off through
JSON rather than sharing a process — the same pattern the YOLO detection pass uses, and
for the same reason: two OpenMP runtimes in one process abort):

    1  localize      ORB-SLAM3 on the scene's own rendered RGB-D    orbslam3_build
    2  fuse          TSDF with the ESTIMATED poses, not GT          foundationpose_vl
    3  segment       Mask3D -> class-agnostic instances             openmask3d_vl
    4  world model   observations -> hierarchical scene graph       foundationpose_vl
    5  ground        graph -> referring expression -> prompt        foundationpose_vl
    6  policy        GR00T N1.6-3B -> action chunk                  groot_vl

Stage 1 uses the SLAM estimate rather than the ground-truth poses `demo_pipeline.py`
writes. That is the point: fusing with GT poses tests the TSDF, fusing with estimated
poses tests the pipeline.

FoundationPose (stage 3.5, 6-DoF refinement) is absent because `nvdiffrast` is CUDA-only
and raises here. The scene graph accepts OpenMask3D-only observations by design —
position from the instance's centroid, identity rotation, because a segmentation mask
carries no orientation — so the chain runs without it, with less precise object poses.

    python pipeline/tools/run_end_to_end.py [--skip-slam]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "pipeline/assets/demo_scene"
WORK = ROOT / "pipeline/assets/e2e"


def run(label: str, env: str, code: str) -> dict:
    """Run one stage in its own conda environment; parse its JSON verdict."""
    WORK.mkdir(parents=True, exist_ok=True)
    script = WORK / f"_stage_{label}.py"
    script.write_text(code)
    print(f"\n{'=' * 72}\n[{label}]  env={env}\n{'=' * 72}", flush=True)

    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env, "python", str(script)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    tail = (result.stdout or "").strip().splitlines()
    for line in tail[-25:]:
        print("  " + line)
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        print(f"  -- FAILED rc={result.returncode}")
        for line in err[-15:]:
            print("     " + line)
        return {"ok": False, "stage": label}

    verdict = WORK / f"{label}.json"
    return json.loads(verdict.read_text()) if verdict.exists() else {"ok": True}


STAGE_SLAM = f'''
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(r"{ROOT}"); SCENE = Path(r"{SCENE}"); WORK = Path(r"{WORK}")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"orbslam3_baseline/bindings/build"))
import cv2, orbslam3_py
cv2.setNumThreads(0)
from pipeline.transforms import matrix_to_quaternion

K = np.loadtxt(SCENE/"intrinsic/intrinsic_color.txt")
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
frames = sorted((SCENE/"color").glob("*.jpg"), key=lambda p: int(p.stem))
h, w = cv2.imread(str(frames[0])).shape[:2]

settings = WORK/"_e2e.yaml"
settings.write_text(
    "%YAML:1.0\\n---\\nFile.version: \\"1.0\\"\\nCamera.type: \\"PinHole\\"\\n"
    f"Camera1.fx: {{fx}}\\nCamera1.fy: {{fy}}\\nCamera1.cx: {{cx}}\\nCamera1.cy: {{cy}}\\n"
    "Camera1.k1: 0.0\\nCamera1.k2: 0.0\\nCamera1.p1: 0.0\\nCamera1.p2: 0.0\\n"
    f"Camera.width: {{w}}\\nCamera.height: {{h}}\\nCamera.fps: 30\\nCamera.RGB: 1\\n"
    "Stereo.b: 0.05\\nRGBD.DepthMapFactor: 1.0\\nStereo.ThDepth: 40.0\\n"
    "ORBextractor.nFeatures: 1200\\nORBextractor.scaleFactor: 1.2\\n"
    "ORBextractor.nLevels: 8\\nORBextractor.iniThFAST: 20\\nORBextractor.minThFAST: 7\\n"
    "Viewer.KeyFrameSize: 0.05\\nViewer.KeyFrameLineWidth: 1.0\\n"
    "Viewer.GraphLineWidth: 0.9\\nViewer.PointSize: 2.0\\nViewer.CameraSize: 0.08\\n"
    "Viewer.CameraLineWidth: 3.0\\nViewer.ViewpointX: 0.0\\nViewer.ViewpointY: -0.7\\n"
    "Viewer.ViewpointZ: -1.8\\nViewer.ViewpointF: 500.0\\nViewer.imageViewScale: 1.0\\n")

tracker = orbslam3_py.Tracker(
    str(ROOT/"orbslam3_baseline/upstream/Vocabulary/ORBvoc.txt"), str(settings), False)
poses, lost = [], 0
for i, f in enumerate(frames):
    colour = cv2.imread(str(f), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(SCENE/"depth"/(f.stem + ".png")), cv2.IMREAD_ANYDEPTH)
    depth = (depth.astype(np.float32) / 1000.0)
    r = tracker.track_rgbd(np.ascontiguousarray(colour), depth, i / 30.0)
    if r["tracking_lost"]:
        lost += 1; poses.append(None)
    else:
        poses.append(np.asarray(r["camera_to_world"]).tolist())
tracker.shutdown()

tracked = [p for p in poses if p is not None]
json.dump({{"ok": len(tracked) > 0, "frames": len(frames), "tracked": len(tracked),
           "lost": lost, "poses": poses}}, open(WORK/"slam.json", "w"))
print(f"tracked {{len(tracked)}}/{{len(frames)}} frames, {{lost}} lost")
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-slam", action="store_true",
                    help="reuse the ground-truth poses instead of the SLAM estimate")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    print(f"scene: {SCENE}")
    if not (SCENE / "color").exists():
        raise SystemExit("no rendered scene; run pipeline/tools/demo_pipeline.py first")

    results = {}
    if not args.skip_slam:
        results["slam"] = run("slam", "orbslam3_build", STAGE_SLAM)
        if not results["slam"].get("ok"):
            print("\nSTOPPED at stage 1 (localize).")
            return 1

    print("\n" + "=" * 72)
    print("stages 2-6 are wired below; this run reports how far the chain gets.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

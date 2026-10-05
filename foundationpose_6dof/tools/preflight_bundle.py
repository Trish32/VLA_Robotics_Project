#!/usr/bin/env python
"""Dry-run the Kaggle kernel's bundle handling locally, before a T4 session is spent.

`fp.py` runs on Kaggle and nowhere else, so every mistake in how it READS the bundle
costs a queue wait to discover — and the bundle has just been rebuilt, which is exactly
when a field goes missing. Everything here is the kernel's own code path up to the first
CUDA call: the same files, the same keys, the same depth clamp.

What it cannot check is the only thing that needs a GPU — `register()` itself.

    conda run -n foundationpose_vl python foundationpose_6dof/tools/preflight_bundle.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e")) / "pose_bundle"

#: Every key `fp.py` reads out of bundle.json, with the line it reads it on.
REQUIRED = {
    "K": "fp.py:190", "depth_scale": "fp.py:249", "target": "fp.py:192",
    "label": "fp.py:192", "sequence": "fp.py:310", "frames": "fp.py:286",
    "fingerprint": "fp.py:309 (staleness check in e2e_pose.py)",
}
REQUIRED_PER_FRAME = {
    "mesh_origin_cam": "fp.py:265", "R_world_to_cam": "fp.py:276",
    "cam_to_world": "fp.py:290",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    a = ap.parse_args(argv)

    import cv2

    B = a.bundle
    fail = []
    meta = json.load(open(B / "bundle.json"))
    print(f"[preflight]  {B.relative_to(ROOT)}  fingerprint "
          f"{meta.get('fingerprint', 'MISSING')}")

    for key, where in REQUIRED.items():
        if key not in meta:
            fail.append(f"bundle.json has no {key!r} — read by {where}")
    for key, where in REQUIRED_PER_FRAME.items():
        missing = [f["frame"] for f in meta.get("frames", []) if key not in f]
        if missing:
            fail.append(f"frames {missing} have no {key!r} — read by {where}")

    # trimesh is what the kernel uses, and it is stricter than open3d about OBJ files.
    try:
        import trimesh
        mesh = trimesh.load(str(B / "mesh.obj"), process=False)
        print(f"[preflight]  mesh {len(mesh.vertices)} verts / {len(mesh.faces)} faces, "
              f"extents {np.round(mesh.extents, 3).tolist()} m")
        if not len(mesh.faces):
            fail.append("mesh.obj loaded with zero faces")
        if np.asarray(mesh.vertex_normals).shape != np.asarray(mesh.vertices).shape:
            fail.append("mesh has no usable vertex normals; FoundationPose needs them")
    except ImportError:
        print("[preflight]  trimesh absent — mesh load NOT checked (the kernel uses it)")

    # The kernel's own loader, including the clamp that silently deletes depth.
    print(f"\n{'fr':>3}{'mask px':>9}{'depth>0':>10}{'mask w/ depth':>15}"
          f"{'kept by clamp':>15}{'median z':>10}")
    for rec in meta["frames"]:
        i = rec["frame"]
        for name in (f"rgb_{i:03d}.png", f"depth_{i:03d}.png", f"mask_{i:03d}.png"):
            if not (B / name).exists():
                fail.append(f"{name} missing")
        raw = cv2.imread(str(B / f"depth_{i:03d}.png"), cv2.IMREAD_ANYDEPTH)
        depth = raw.astype(np.float32) / meta["depth_scale"]
        mask = cv2.imread(str(B / f"mask_{i:03d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        before = int((mask & (depth > 0)).sum())
        depth[(depth < 0.1) | (depth > 4.0)] = 0          # the kernel's clamp
        after = int((mask & (depth > 0)).sum())
        z = depth[mask & (depth > 0)]
        print(f"{i:3d}{int(mask.sum()):9d}{int((depth>0).sum()):10d}{before:15d}"
              f"{after:15d}{float(np.median(z)):9.3f}m")
        if after < 500:
            fail.append(f"frame {i}: only {after} mask pixels survive the depth clamp")
        if before and after / before < 0.9:
            fail.append(f"frame {i}: the 0.1-4.0 m clamp deletes "
                        f"{100 * (1 - after / before):.0f}% of the mask")

    # register() sees frame 0 and nothing else, so it carries the whole run.
    anchor = meta["frames"][0]
    print(f"\n[preflight]  register() will see frame {anchor.get('source_index', '?')}: "
          f"{anchor['mask_pixels']} px, "
          f"{anchor.get('visible_fraction', float('nan')):.0%} of the instance")
    if anchor["mask_pixels"] < 1500:
        fail.append(f"anchor mask is {anchor['mask_pixels']} px; mustard0, the control "
                    f"that registers correctly, has 3,252")

    if fail:
        print("\n[preflight]  REFUSED — do not spend a session on this bundle:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\n[preflight]  OK — the kernel's bundle path is satisfied. "
          "register() itself still needs the T4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

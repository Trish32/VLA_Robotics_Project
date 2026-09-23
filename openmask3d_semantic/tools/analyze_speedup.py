#!/usr/bin/env python
"""Quantify each OpenMask3D speedup separately, so the credit lands where it is due.

Two kinds of number appear below and they are not the same kind of claim:

  MEASURED   call counts. Exact, derived by running the real selection logic over
             simulated scene layouts. These do not depend on hardware.
  MODELLED   time. Derived from published forward-pass costs, because SAM ViT-H on a
             T4/L4 is not runnable here. Labelled as a model everywhere it appears.

The scene model: views on a trajectory through the room, proposals scattered in it, and
each proposal's top-k views taken as its k nearest cameras. That reproduces the property
that actually drives the result — neighbouring proposals pick overlapping view sets —
without pretending to be a specific ScanNet scene.

    conda run -n openmask3d_vl python openmask3d_semantic/tools/analyze_speedup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openmask3d_semantic.fast_features import graspable_candidates  # noqa: E402

# Relative forward-pass cost, normalised to one SAM ViT-H image-encoder pass.
# ViT-H/16 at 1024x1024 dominates everything else in this pipeline by two orders of
# magnitude; the decoder is ~4M params against the encoder's 636M.
COST = {
    "sam_vit_h_encoder": 1.0,
    "sam_mobile_encoder": 0.014,   # TinyViT-5M, ~1/70th the encoder FLOPs
    "sam_decoder_call": 0.0015,    # ~4M params, tiny prompt set
    "clip_vit_l_image": 0.028,     # ViT-L/14 at 224^2 vs ViT-H at 1024^2
}


def simulate(num_views=180, num_proposals=110, topk=5, seed=0):
    """Return top-k view indices per proposal, plus proposal geometry for gating."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, num_views)
    views = np.column_stack([
        4.0 * np.sin(2 * np.pi * t), 3.0 * np.cos(2 * np.pi * t) , np.full(num_views, 1.4),
    ])

    # A realistic proposal mix: mostly structure, a minority graspable.
    kinds = rng.choice(["structure", "furniture", "object"], size=num_proposals,
                       p=[0.45, 0.30, 0.25])
    centres, groups, points = [], [], []
    for kind in kinds:
        centre = np.array([rng.uniform(-3.5, 3.5), rng.uniform(-2.5, 2.5), 0.0])
        if kind == "structure":       # floor / wall slabs
            extent, n, centre[2] = np.array([4.0, 3.0, 0.05]), 3000, 0.02
        elif kind == "furniture":
            extent, n, centre[2] = np.array([0.8, 0.7, 0.75]), 900, 0.4
        else:                          # graspable object on a surface
            extent, n, centre[2] = np.array([0.09, 0.08, 0.11]), 260, 0.80
        start = sum(len(g) for g in points)
        cloud = centre + rng.uniform(-extent / 2, extent / 2, (n, 3))
        points.append(cloud)
        groups.append(np.arange(start, start + n))
        centres.append(centre)

    centres = np.stack(centres)
    distance = np.linalg.norm(centres[:, None, :] - views[None, :, :], axis=2)
    topk_indices = np.argsort(distance, axis=1)[:, :topk]
    return topk_indices, groups, np.vstack(points), kinds


def costs(topk_indices, candidate_mask, *, encoder_cost, rounds, num_levels, batched):
    """Exact call counts, plus a modelled total cost."""
    rows = np.flatnonzero(candidate_mask)
    selected = topk_indices[rows]
    pairs = selected.size
    encoder_calls_mask_major = pairs
    encoder_calls_view_major = len(np.unique(selected))
    decoder_calls = pairs if batched else pairs * rounds
    clip_images = pairs * num_levels
    return {
        "proposals": len(rows),
        "pairs": pairs,
        "encoder_mask_major": encoder_calls_mask_major,
        "encoder_view_major": encoder_calls_view_major,
        "decoder_calls": decoder_calls,
        "clip_images": clip_images,
        "cost_mask_major": encoder_calls_mask_major * encoder_cost
                           + pairs * rounds * COST["sam_decoder_call"]
                           + clip_images * COST["clip_vit_l_image"],
        "cost_view_major": encoder_calls_view_major * encoder_cost
                           + decoder_calls * COST["sam_decoder_call"]
                           + clip_images * COST["clip_vit_l_image"],
    }


def main() -> int:
    topk_indices, groups, points, kinds = simulate()
    all_masks = np.ones(len(topk_indices), dtype=bool)
    keep, reasons = graspable_candidates(groups, points)

    print("scene model: 180 views, 110 proposals, topk=5")
    print(f"  proposal mix: {dict(zip(*np.unique(kinds, return_counts=True)))}\n")

    baseline = costs(topk_indices, all_masks, encoder_cost=COST["sam_vit_h_encoder"],
                     rounds=10, num_levels=3, batched=False)

    print("=" * 74)
    print(f"{'step':<34}{'SAM encoder':>13}{'decoder':>10}{'modelled':>10}{'speedup':>7}")
    print("=" * 74)
    print(f"{'0. upstream (mask-major, ViT-H)':<34}"
          f"{baseline['encoder_mask_major']:>13}{baseline['pairs']*10:>10}"
          f"{baseline['cost_mask_major']:>10.1f}{1.0:>7.1f}x")

    steps = []
    # 1. loop inversion only
    s1 = costs(topk_indices, all_masks, encoder_cost=COST["sam_vit_h_encoder"],
               rounds=10, num_levels=3, batched=False)
    steps.append(("1. + loop inversion", s1["encoder_view_major"], s1["pairs"] * 10,
                  s1["encoder_view_major"] * COST["sam_vit_h_encoder"]
                  + s1["pairs"] * 10 * COST["sam_decoder_call"]
                  + s1["clip_images"] * COST["clip_vit_l_image"]))
    # 2. + batched prompt rounds
    s2 = costs(topk_indices, all_masks, encoder_cost=COST["sam_vit_h_encoder"],
               rounds=10, num_levels=3, batched=True)
    steps.append(("2. + batched prompt rounds", s2["encoder_view_major"],
                  s2["decoder_calls"], s2["cost_view_major"]))
    # 3. + fp16 (encoder halves; decoder/CLIP assumed to follow)
    steps.append(("3. + fp16 autocast", s2["encoder_view_major"], s2["decoder_calls"],
                  s2["cost_view_major"] / 2.0))
    # 4. + candidate gating
    s4 = costs(topk_indices, keep, encoder_cost=COST["sam_vit_h_encoder"],
               rounds=10, num_levels=3, batched=True)
    steps.append(("4. + candidate gating", s4["encoder_view_major"],
                  s4["decoder_calls"], s4["cost_view_major"] / 2.0))
    # 5. + MobileSAM
    s5 = costs(topk_indices, keep, encoder_cost=COST["sam_mobile_encoder"],
               rounds=10, num_levels=3, batched=True)
    steps.append(("5. + MobileSAM encoder", s5["encoder_view_major"],
                  s5["decoder_calls"], s5["cost_view_major"] / 2.0))

    for name, enc, dec, cost in steps:
        print(f"{name:<34}{enc:>13}{dec:>10}{cost:>10.1f}"
              f"{baseline['cost_mask_major'] / cost:>7.1f}x")
    print("=" * 74)

    dropped = int((~keep).sum())
    print(f"\ngating: kept {int(keep.sum())}/{len(keep)} proposals, dropped {dropped}")
    from collections import Counter
    tally = Counter(r.split("—")[-1].strip() for r in reasons if r)
    for reason, count in tally.most_common():
        print(f"   {count:>3}  {reason}")

    print("\nEXACT (call counts, hardware-independent):")
    print(f"   SAM encoder passes {baseline['encoder_mask_major']} -> "
          f"{s4['encoder_view_major']}  "
          f"({baseline['encoder_mask_major'] / s4['encoder_view_major']:.1f}x fewer)")
    print(f"   SAM decoder calls  {baseline['pairs']*10} -> {s4['decoder_calls']}  "
          f"({baseline['pairs']*10 / s4['decoder_calls']:.1f}x fewer)")
    print(f"   CLIP images        {baseline['clip_images']} -> {s4['clip_images']}  "
          f"({baseline['clip_images'] / s4['clip_images']:.1f}x fewer)")
    print("\nMODELLED (relative cost; wall-clock needs a CUDA box and real weights).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

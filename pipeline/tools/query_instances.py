#!/usr/bin/env python
"""Query the saved per-instance CLIP features with arbitrary text. Instant.

This is what "open-vocabulary" is supposed to mean and what OpenMask3D is actually for:
the expensive pass computes ONE feature vector per instance, and any question you later
want to ask is a dot product against it. Until now stage 4 computed those features,
took an argmax against a hardcoded 16-word list, wrote the winning string and threw the
features away — so asking a different question meant re-running SAM + CLIP over every
view, about ten minutes on CPU, to answer something that takes a millisecond.

Both sides are unit-normalised here, so the numbers printed really are cosine
similarities. `extract_mask_features` normalises each crop and then AVERAGES, and the
mean of unit vectors is not a unit vector — its norm drops as the crops disagree across
views — so an un-normalised dot product returns |f|*cos(theta) and is not comparable
between instances.

    conda run -n openmask3d_vl python pipeline/tools/query_instances.py \\
        --query "a monitor" "a keyboard" "a cluttered desk" "a coffee mug"
    conda run -n openmask3d_vl python pipeline/tools/query_instances.py --rank "a monitor"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "pipeline/assets/e2e"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", nargs="+", help="words to score every instance against")
    ap.add_argument("--rank", help="one phrase; rank instances by it, best first")
    ap.add_argument("--bare", action="store_true",
                    help="skip the CLIP prompt-template ensemble, score the raw string")
    args = ap.parse_args()
    if not args.query and not args.rank:
        raise SystemExit("give --query WORD... or --rank PHRASE")

    import clip
    import torch

    from pipeline.tools.e2e_label import PROMPT_TEMPLATES, cosine_similarity

    path = OUT / "instance_features.npz"
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found — it is written by e2e_label.py. Re-run stage 4:\n"
            "  conda run -n openmask3d_vl python pipeline/tools/e2e_label.py"
        )
    blob = np.load(path, allow_pickle=True)
    features, ids = blob["features"], [str(i) for i in blob["ids"]]

    alive = [i for i in range(len(features)) if np.abs(features[i]).sum() > 0]
    print(f"[query ]  {len(alive)}/{len(features)} instances carry a feature "
          f"(the rest got no crops)")

    model, _ = clip.load("ViT-L/14@336px", device="cpu")

    def score(words):
        if args.bare:
            with torch.no_grad():
                text = model.encode_text(clip.tokenize(list(words))).float()
                text /= text.norm(dim=-1, keepdim=True)
            image = torch.from_numpy(features).float()
            image = image / image.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return image @ text.T
        return cosine_similarity(features, model, list(words))

    if args.rank:
        sims = score([args.rank])[:, 0]
        print(f"\n  instances ranked by {args.rank!r}"
              f"{'  (bare string)' if args.bare else '  (template ensemble)'}\n")
        for i in sorted(alive, key=lambda i: -float(sims[i])):
            bar = "█" * int(max(0.0, float(sims[i])) * 120)
            print(f"  {ids[i]:<10}{float(sims[i]):>7.3f}  {bar}")
        return 0

    sims = score(args.query)
    head = "".join(f"{w[:13]:>15}" for w in args.query)
    print(f"\n  {'instance':<10}{head}     best")
    print("  " + "-" * (12 + 15 * len(args.query) + 18))
    for i in alive:
        row = "".join(f"{float(sims[i, j]):>15.3f}" for j in range(len(args.query)))
        best = int(torch.argmax(sims[i]))
        margin = float(sims[i, best]) - float(sorted(sims[i].tolist())[-2]) \
            if len(args.query) > 1 else 0.0
        print(f"  {ids[i]:<10}{row}     {args.query[best]}"
              f"{f'  (+{margin:.3f})' if len(args.query) > 1 else ''}")

    # A margin this thin is the thing to report. Picking a winner from a 16-way argmax
    # where the top two differ by 0.016 is not recognition, and printing only the winner
    # hides that.
    if len(args.query) > 1:
        margins = []
        for i in alive:
            ordered = sorted(sims[i].tolist(), reverse=True)
            margins.append(ordered[0] - ordered[1])
        print(f"\n  median top-1 margin {np.median(margins):.3f} over {len(alive)} "
              f"instances — below ~0.01 the argmax is close to arbitrary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One real observation -> one action chunk, on CPU. The end-to-end plumbing check.

Everything before this validated pieces: the checkpoint loads 0/0, the action head's
flow matching is right on tiny inputs, the packing mask is right, the LeRobot reader
returns real frames. This is the first time all of it runs as one system on genuine data
from upstream's own `demo_data`.

What it does and does not prove
-------------------------------
DOES: that observations flow through the processor, Eagle, and the flow-matching head to
a correctly-shaped, finite action chunk for the declared embodiment -- i.e. that a cloud
job will fail on data or hyperparameters rather than on a shape bug.

DOES NOT: produce a trustworthy action. The attention kernel and dtype both differ from
the reference (bug_log [1], [3]), and no published metric is being reproduced here.

Usage:
    PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/run_policy_cpu.py \
        --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B \
        --dataset grootN1_Robotics/upstream/demo_data/gr1.PickNPlace
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_observation(dataset: Path, policy, frame: int = 0, task_override: str | None = None) -> dict:
    """Assemble one observation in the (B, T, ...) layout check_observation expects."""
    from grootN1_Robotics.lerobot_data import build_loader

    cfgs = policy.modality_configs
    loader = build_loader(dataset, {k: v for k, v in cfgs.items() if k != "language"})
    episode = loader[0]

    video, state = {}, {}
    for key in cfgs["video"].modality_keys:
        arr = np.asarray(episode[f"video.{key}"].iloc[frame], dtype=np.uint8)
        video[key] = arr[None, None]                      # (B=1, T=1, H, W, C)
    for key in cfgs["state"].modality_keys:
        arr = np.asarray(episode[f"state.{key}"].iloc[frame], dtype=np.float32)
        state[key] = arr[None, None]                      # (B=1, T=1, D)

    # The processor names its language key ("task") independently of the dataset's
    # annotation key, so the instruction is read from the dataset separately.
    from grootN1_Robotics.lerobot_data import read_modality_meta

    task = "pick up the object"
    annotations = read_modality_meta(dataset).get("annotation", {})
    if task_override:
        # The scene graph's serialised prompt, when this runs as pipeline stage 6. It
        # replaces the dataset's own annotation rather than being appended to it: two
        # instructions in one field is not a register the model was trained on.
        return {"video": video, "state": state,
                "language": {policy.language_key: [[task_override]]}}
    for name in ("human.action.task_description", "human.task_description"):
        if name in annotations:
            from gr00t.data.types import ModalityConfig

            lang_loader = build_loader(
                dataset,
                {"language": ModalityConfig(delta_indices=[0],
                                            modality_keys=[f"annotation.{name}"])},
            )
            task = str(lang_loader[0][f"language.annotation.{name}"].iloc[frame])
            break

    return {"video": video, "state": state, "language": {policy.language_key: [[task]]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embodiment", default="gr1")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--task", default=None,
                    help="override the dataset annotation; pipeline stage 6 passes the "
                         "scene graph's serialised prompt here")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="write the action chunk for the next pipeline stage")
    args = ap.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from grootN1_Robotics.policy import LocalGr00tPolicy

    print(f"[build] {args.checkpoint} on cpu/fp32")
    t0 = time.time()
    policy = LocalGr00tPolicy(EmbodimentTag(args.embodiment), args.checkpoint)
    print(f"[build] {time.time() - t0:.1f}s")

    obs = build_observation(Path(args.dataset), policy, args.frame,
                            task_override=args.task)
    print("[obs] " + ", ".join(
        f"{m}.{k}={np.asarray(v).shape}"
        for m in ("video", "state") for k, v in obs[m].items()
    ))
    print(f"[obs] task = {obs['language'][policy.language_key][0][0]!r}")

    print("[run] forward (CPU, several minutes at fp32)...")
    t0 = time.time()
    action, info = policy.get_action(obs)
    print(f"[run] {time.time() - t0:.1f}s\n")

    print(f"{'action key':<28}{'shape':>16}{'min':>12}{'max':>12}")
    print("-" * 68)
    ok = True
    for key, value in sorted(action.items()):
        arr = np.asarray(value)
        finite = np.isfinite(arr).all()
        ok &= bool(finite)
        print(f"{key:<28}{str(arr.shape):>16}{arr.min():>12.4f}{arr.max():>12.4f}"
              f"{'' if finite else '   NON-FINITE'}")
    print("-" * 68)

    if args.save_json:
        import json
        chunk = {k.removeprefix("action."): np.asarray(v)[0].tolist()
                 for k, v in sorted(action.items())}
        first = next(iter(chunk.values()))
        args.save_json.write_text(json.dumps({
            "ok": ok,
            "task": obs["language"][policy.language_key][0][0],
            "horizon": len(first),
            "width": sum(len(v[0]) for v in chunk.values()),
            "seconds": round(time.time() - t0, 1),
            "keys": {k: list(np.asarray(v).shape) for k, v in sorted(action.items())},
            "values": chunk,
        }, indent=1))
        print(f"[save] {args.save_json}")

    print("\nEnd-to-end plumbing OK." if ok else "\nNON-FINITE OUTPUT — investigate.")
    print("Not a metric: sdpa + fp32 differ from the shipped flash + bf16 reference.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

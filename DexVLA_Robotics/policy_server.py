"""Serve DexVLA behind the same ZMQ contract the ROS bridge already speaks.

The bridge node is policy-agnostic: it talks to whatever answers `ping`, `get_action` and
`get_modality_config` on a ZMQ REP socket. GR00T gets that for free from upstream's
`PolicyServer`. DexVLA does not — its model exposes `evaluate()`, which returns
`(action, reasoning_text)` and expects tokenised VLM inputs, not an observation dict.

This adapts DexVLA to that contract, and nothing more. Two things it deliberately does NOT
do:

  * It does not invent a modality config. DexVLA has no per-embodiment config the way
    GR00T does, so `get_modality_config` reports the single action key this server emits,
    which is what the node needs to size its chunk.
  * It does not fabricate reasoning. If the caller supplies no instruction the model is
    asked to produce one, exactly as at training time — a blank string would train-test
    mismatch the FiLM conditioning (see ../DexVLA_Robotics/bug_log.txt).

**`state_relative` must be FALSE for DexVLA.** N1.6 emits state-relative chunks; DexVLA
emits absolute joint targets. The bridge parameter defaults to True for GR00T, so a DexVLA
deployment MUST override it. Wrong value sends the arm toward the origin.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ACTION_KEY = "action.joints"  # single flat chunk; the node concatenates by key name


class DivlaPolicy:
    """Duck-typed to what `PolicyServer` calls: get_action / reset / get_modality_config."""

    def __init__(self, model, processor, tokenizer, action_dim: int, horizon: int,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.dtype = dtype
        self._last_reasoning = ""

    def get_modality_config(self) -> dict:
        """The node reads `action.modality_keys` from this to size its chunk."""
        return {"action": {"modality_keys": [ACTION_KEY],
                           "delta_indices": list(range(self.horizon))}}

    def reset(self, options: dict | None = None) -> dict:
        self._last_reasoning = ""
        return {"ok": True}

    def get_action(self, observation: dict, options: dict | None = None):
        """observation -> ({ACTION_KEY: (1, horizon, action_dim)}, info).

        Returns the (action, info) PAIR that upstream's client unwraps — matching
        `BasePolicy.get_action`, because the bridge's wire client expects that shape.
        """
        images = observation.get("video", {})
        state = observation.get("state", {})
        language = observation.get("language", {})

        if not images:
            raise ValueError("observation has no 'video' entries")
        if not state:
            raise ValueError("observation has no 'state' entries")

        instruction = ""
        for value in language.values():
            instruction = value[0][0] if isinstance(value, (list, tuple)) else str(value)
            break

        # (B, T, D) -> (B, D); the head consumes the current state only.
        qpos = np.concatenate(
            [np.asarray(v, dtype=np.float32).reshape(1, -1) for _, v in sorted(state.items())],
            axis=1,
        )
        frames = [np.asarray(v, dtype=np.uint8).reshape(-1, *np.asarray(v).shape[-3:])[0]
                  for _, v in sorted(images.items())]

        action, reasoning = self._infer(frames, qpos, instruction)
        self._last_reasoning = reasoning
        return ({ACTION_KEY: action}, {"reasoning": reasoning})

    def _messages(self, instruction: str, n_images: int) -> list:
        """Mirrors `smart_eval_agilex.py::datastruct_droid2qwen2vla`.

        One `{"type": "image"}` entry per camera followed by the text. The image count
        must match the number of frames passed to the processor, or the chat template
        emits the wrong number of image placeholders and every subsequent token index —
        including the reasoning boundary film_forward computes — shifts.
        """
        content = [{"type": "image", "image": None} for _ in range(n_images)]
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def _infer(self, frames, qpos, instruction):
        """Tokenise, generate reasoning, denoise an action chunk.

        Mirrors upstream's INFERENCE preprocessing (`process_batch_to_qwen2_vla`), not
        `Qwen2VLAProcess.forward_process` — the latter is training-only: it appends the
        ground-truth reasoning string and builds a label mask. At inference the model
        generates the reasoning itself and `evaluate()` fabricates the labels, so using
        the training path here would condition on an answer we do not have.
        """
        from PIL import Image

        messages = self._messages(instruction, len(frames))
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]

        model_inputs = self.processor(
            text=text, images=images, videos=None, padding=True, return_tensors="pt"
        )
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in model_inputs.items()}

        with torch.no_grad():
            action, reasoning = self.model.evaluate(
                **batch,
                states=torch.as_tensor(qpos, dtype=self.dtype, device=self.device),
                is_eval=True,
                tokenizer=self.tokenizer,
            )
        return np.asarray(action, dtype=np.float32), str(reasoning)


def build_policy(backbone: Path, head_size: str, action_dim: int, state_dim: int,
                 horizon: int, device: str) -> DivlaPolicy:
    here = Path(__file__).resolve().parent
    # BOTH are needed and for different reasons: `DexVLA_Robotics/` makes `tools.build_vla`
    # importable, while `policy_heads` and `qwen2_vla` are top-level packages inside the
    # vendored upstream tree — which is why upstream's own scripts are run as
    # `cd DexVLA_Robotics/upstream && PYTHONPATH=. python ...`. Adding only the first gets you
    # `ModuleNotFoundError: No module named 'policy_heads'` from inside build_config.
    for path in (here, here / "upstream"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from transformers import AutoProcessor, AutoTokenizer

    from tools.build_vla import build_config

    cfg = build_config(backbone, head_size, action_dim, state_dim, horizon, True)
    from qwen2_vla.models.modeling_qwen2_vla import (
        Qwen2VLForConditionalGenerationForVLA,
    )

    model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(
        backbone, config=cfg, torch_dtype=torch.float32
    ).eval().to(device)

    return DivlaPolicy(
        model=model,
        processor=AutoProcessor.from_pretrained(backbone),
        tokenizer=AutoTokenizer.from_pretrained(backbone, use_fast=True),
        action_dim=action_dim,
        horizon=horizon,
        device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--policy-head", default="ScaleDP_H")
    ap.add_argument("--action-dim", type=int, default=14)
    ap.add_argument("--state-dim", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=5556)  # 5555 is GR00T's default
    ap.add_argument("--host", default="*")
    args = ap.parse_args()

    from gr00t.policy.server_client import PolicyServer  # wire-compatible server

    policy = build_policy(Path(args.backbone), args.policy_head, args.action_dim,
                          args.state_dim, args.horizon, args.device)
    print(f"[serve] DexVLA on tcp://{args.host}:{args.port}", flush=True)
    print(f"[serve] action key {ACTION_KEY!r}, horizon {args.horizon}", flush=True)
    print("[serve] REMEMBER: set the bridge's state_relative:=false for DexVLA", flush=True)
    PolicyServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()

# DexVLA

**DexVLA: Vision-Language Model with Plug-In Diffusion Expert for Visuomotor Policy Learning** —
[arXiv 2502.05855](https://arxiv.org/abs/2502.05855), [project page](https://dex-vla.github.io/),
code [juruobenruo/DexVLA](https://github.com/juruobenruo/DexVLA) pinned at `fc21a82`.

**Retargeted from DiffusionVLA to DexVLA.** DiVLA (Wen et al., ICML 2025,
[project page](https://diffusion-vla.github.io)) is the predecessor, and its authors point at this
same repo — it trains DiVLA via `scripts/train_divla.sh`. Since DexVLA is the successor, is what
the repo is actually built around, and is the only one of the two with *any* released weights,
targeting DiVLA meant aiming at the strictly weaker of two things we already had checked out.
Everything already validated below carries over unchanged; only the target and the training path
do.

> **Not a reproduction, and it says so.** Only Stage 1 is public and upstream's only
> eval entry point drives a real AgileX robot, so the 0/0 + reproduce-the-metric bar is
> unreachable here no matter how much is ported. What is achievable, and why, is in
> **[Plan.md](Plan.md)**.

**→ [RESULTS.md](RESULTS.md)** — what is verified, and how
**→ [Plan.md](Plan.md)** — the limits, the free-GPU budget, and what is left

## Architecture

- **Backbone**: Qwen2-VL-2B, used as-is (the repo swaps in a custom config file for VLA use).
- **Action head**: **ScaleDP**, a transformer diffusion policy
  (`policy_heads/models/transformer_diffusion/`). This is DexVLA's head and the one with released
  weights. *DiVLA used a UNet head* — that difference is now a note about the predecessor rather
  than a deviation we have to excuse.
- **Reasoning**: autoregressive — the model emits a reasoning string, which is then injected as
  conditioning for the action head. Reasoning and action in one model rather than a diffusion
  policy bolted onto a frozen VLM.
- Reasoning injection ("reasoning following") is the subtle part — how the emitted reasoning tokens
  condition the denoiser. Diff this carefully against the reference.

## build GROOT first

Roughly 60% is infrastructure GR00T already built: the LeRobot/H5PY data path, the diffusion
action-head training loop, the LoRA + frozen-backbone free-GPU recipe, and the sim eval harness.
Only the backbone (Qwen2-VL vs Eagle-2.5), the head (ScaleDP vs flow-matching DiT), and the
reasoning channel are new. Doing it before GR00T would have meant building all of that twice —
that dependency is the whole reason for the ordering.

Qwen2-VL-2B loads via `transformers`, same recorded exception as GR00T's backbone.

## Local setup

Upstream pinned at `fc21a82`. Env `divla_vl` (py3.10, torch 2.4.1, transformers 4.45.2,
timm 0.9.10) — **separate from `groot_vl`, whose pins conflict**. The package imports on
Mac with no CUDA and no flash-attn. Five blockers cleared, full detail in `bug_log.txt`:

1. `requirements.txt` pins `torch==2.4.1` with `torchvision==0.15.2` — an impossible pair
   (0.15.2 belongs to torch 2.0). Used 0.19.1.
2. unpinned diffusers × torch 2.4.1 → `infer_schema` cannot parse PEP-604 string
   annotations in a custom op. Pin `diffusers==0.11.1` as requirements.txt actually says.
3. `cached_download` gone from `huggingface_hub` ≥ 0.26 → pin 0.25.2.
   **Identical to the DiffusionDrive blocker** — any repo pinning diffusers < 0.27
   needs hub < 0.26.
4. `policy_heads/__init__.py` imports a **top-level** package named `models`, which only
   resolves if `policy_heads/` is on sys.path and claims a global name. `pip install -e`
   does not help: `policy_heads/models/` has no `__init__.py`. → relative imports,
   `patches/0001`.
5. `MODEL_STRUCTURE` is keyed `ScaleDP_H` but `ScaleDP_models` is keyed `ScaleDP-H`.

```bash
cd DexVLA_Robotics/upstream && PYTHONPATH=. conda run -n divla_vl python \
  ../tools/load_scaledp.py --checkpoint ../checkpoints/scale_dp_h/open_scale_dp_h_backbone.ckpt \
  --size ScaleDP_H --prediction-horizon 50
```


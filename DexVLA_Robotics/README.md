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

## Read this before starting: what is and is not achievable

Two separate limits, and the second is the binding one.

**1. Only Stage 1 is released.** DexVLA trains in three stages; the public artifacts are the
Stage-1 action heads and nothing else:

| Released | What it is |
|---|---|
| `lesjie/scale_dp_h` (0.950 B) | ScaleDP-H diffusion action head, **Stage-1 weights** |
| `lesjie/scale_dp_l` (~0.41 B) | ScaleDP-L, same |
| `Qwen/Qwen2-VL-2B` | the backbone, off-the-shelf, no VLA post-training |

There is no Stage-2 or Stage-3 DexVLA checkpoint. Verified three ways: the repo links only the
two heads; upstream `HEAD` is still `fc21a82` with no commits since our pin; and the author's
HF profile publishes five models, none of them a full VLA.

**2. The evaluation is real-robot only.** The repo's sole eval entry point is
`evaluate/smart_eval_agilex.py` — AgileX, a bimanual robot. There is no sim harness: no LIBERO,
no SimplerEnv, no MuJoCo. So even if a Stage-2 checkpoint appeared tomorrow, reproducing a
published DexVLA number would need the hardware.

**Therefore the house fidelity rule is unreachable here, and no amount of porting changes that.**
The deliverable is a **correct architecture, fine-tuned from the released Stage-1 head, plus a
controlled comparison against our own GR00T baseline** on the same task suite and the same
free-GPU budget. Do not let it masquerade as a reproduction.

What retargeting *does* buy: the head we start from is now DexVLA's own Stage-1 artifact rather
than a substitute for a UNet we could not obtain, and `train_dexvla_stage2.sh` / `stage3.sh` are
the documented, supported continuation — "download the weights and directly finetuning your data
on Stage 2."

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

## Training path and the free-GPU budget

Stage 1 is released, so we enter at **Stage 2**. The binding constraint is the same one GR00T hit:
freezing the VLM is not enough, because the *head* is ~1 B on its own.

Steady-state floors with the VLM frozen (Qwen2-VL-2B at fp16 = 4.4 GB). Activations excluded, so
treat a 16 GB T4 as ~12 GB of usable budget — same method as `grootN1_Robotics/tools/plan_finetune_memory.py`:

| trainable | optimizer | total | fits |
|---|---|---|---|
| full VLA 3.18 B | AdamW | ~50 GB | nothing we have |
| ScaleDP-H 0.950 B | AdamW | 19.6 GB | L4 / A100 |
| ScaleDP-H 0.950 B | 8-bit | 13.9 GB | 2×T4 / L4 — **not** a single T4 |
| ScaleDP-H + LoRA | 8-bit | ~7 GB | **single T4** |
| ScaleDP-L 0.41 B | 8-bit | 8.5 GB | **single T4** |

So the free-tier path is **ScaleDP-L, or ScaleDP-H with LoRA**. Full ScaleDP-H head training needs
2×T4 or an L4. These are estimates from parameter counts, not measurements — measure before
trusting them, the way GR00T's budget was measured.

## Why this is cheap to finish

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

## Checklist
- [x] **Qwen2-VL-2B loading + the VLA config delta** → [tools/build_vla.py](tools/build_vla.py).
      The delta is a **config.json swap**, not a code path: `model_type qwen2_vl → qwen2_vla`
      plus `policy_head_type: scale_dp_policy`, and nothing else (README line 86). Original
      kept as `config.json.official`. After the swap you get
      *"Transformers does not recognize this architecture"*, which looks like a version
      problem but is a **missing import** — `AutoConfig.register("qwen2_vla", ...)` sits at
      the bottom of `configuration_qwen2_vla.py`.
      **VLA assembles: 3.179B** = visual 0.665B + language 1.544B + ScaleDP-H 0.956B +
      3 × 4.72M fusion modules. The "newly initialized" list is *only* `policy_head.*` and
      the fusion modules, so **every stock Qwen2-VL weight landed**.
      Note `using_film` / `policy_head_config` / `policy_head_size` are read by the model
      but **not declared on the config class** — `train_vla.py` injects them after
      construction, so reading the config class tells you nothing about them.
- [x] **ScaleDP head: `lesjie/scale_dp_h` loads 0 unexpected / 0 shape mismatch** against
      upstream's own class — 327/339 tensors, 0.950B →
      [tools/load_scaledp.py](tools/load_scaledp.py). Every checkpoint tensor maps onto
      the model, confirming the architecture reconstruction.
      The 12 "missing" are exactly the embodiment-specific adapters a `*_backbone.ckpt`
      should omit (`combine.*`, `cond_obs_emb.*`, `final_layer.linear.*`, +2).
      **The released head's `prediction_horizon` is 50, not the config default 16** —
      recovered from `pos_embed (1, 50, 1280)`; nothing in the release states it.
      Layout: raw torch save, no config.json, tensors at `ckpt["nets"]["nets"]` under a
      `noise_pred_net.` prefix. ScaleDP-H = 32 blocks × 1280 × 16 heads, DiT adaLN.
- [x] **Reasoning injection path exercised** →
      [tests/test_reasoning_injection.py](tests/test_reasoning_injection.py), 7 tests
      driving the real `film_forward` against a stub holding only the three modules it
      touches — no 2B backbone, ~2s. Pins the boundary logic (XOR over the `-100` label
      mask), left-padding exclusion via the Qwen2-VL pad id, the `(B, 1, D)` pooled shape,
      and that the residual is exactly `mean(hidden[start:end])`.
      **The hazard, now pinned as a property: `reasoning_film` is zero-initialised**, so
      at init the reasoning channel is *exactly* the identity and contributes nothing.
      Any "reasoning works" claim from an untrained model is vacuous; the complementary
      test shows reasoning does move the output once FiLM is non-zero.
- [x] **Reasoning generation at inference (the fabricated-labels path)** — at inference
      there are no labels, so `evaluate()` generates first and then *fabricates* them
      (`-100` over the prompt span, `1` over the generated span). Verified the fabricated
      mask recovers the **same boundary** the training mask does; a one-token disagreement
      would condition the model differently at inference than in training, silently.
      Also found a latent issue: `start` **counts occurrences** of id 151643 anywhere
      rather than measuring the leading padding run — and `bos_token_id` is *also* 151643.
      Not currently triggered (Qwen2's template prepends no BOS) but pinned; `bug_log.txt` [8].
- [x] **H5PY data format documented and validated** → [h5_format.py](h5_format.py),
      12 tests. The layout exists nowhere upstream except as index expressions in
      `data_utils/utils.py::load_from_h5`. Two rules it implies but never states:
      `substep_reasonings` **takes precedence** over `reasoning` (checked first, only
      falls back — so a short one raises mid-epoch), and camera names are dataset **keys**
      from `aloha_scripts/constants.py`, so a rename is a KeyError deep in the loader.
      We do not reuse upstream's `check_data_integrity.py`: on a file that fails to open
      it logs and falls through, appending the *previous* file's `qpos` into the
      normalisation statistics. `bug_log.txt` [7].
- [x] **Runs end to end on CPU, and over ZMQ from the ROS bridge** →
      [policy_server.py](policy_server.py),
      [tests/test_ros_bridge_against_real_divla.py](tests/test_ros_bridge_against_real_divla.py),
      6 tests. Build 11s, inference 3.3s, `action.joints (1, 8, 14)`, reasoning generated.
      Getting there needed `patches/0002` and `0003`: upstream hardcodes `'cuda'` in six
      places and assigns `torch.bfloat16` unconditionally, so DiVLA could not run on CPU
      **or on a T4** — Turing (sm_75) has no bf16, and half this project's compute is T4.
      The patched values come from `self.device` and `self.dtype`, which is a no-op on
      upstream's bf16-on-CUDA path; that no-op claim is read from the code, not measured,
      and is flagged in `bug_log.txt` for an activation diff on the first GPU run.
      The test sends the observation the ROS node itself builds, which is what caught the
      bridge's flat-vs-nested layout bug (`ros2_bridge/bug_log.txt` [3]).
- [ ] RLDS conversion scripts (untouched).
- [ ] Train on the same task suite used for GR00T; report both under identical budget.

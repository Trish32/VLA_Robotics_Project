# DexVLA — plan

What is not done, and the one limit that no amount of porting can move.

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

---

## Remaining

- [ ] RLDS conversion scripts (untouched).
- [ ] Train on the same task suite used for GR00T; report both under an identical budget.

The budget table above is **estimated from parameter counts, not measured**. Measure
before trusting it, the way GR00T's budget was measured
(`grootN1_Robotics/tools/plan_finetune_memory.py`).

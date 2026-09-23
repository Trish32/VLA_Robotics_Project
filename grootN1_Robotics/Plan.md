# GR00T N1.6 — plan

**The gap: no published metric reproduced.** The model runs; nothing has been scored.
LIBERO / SimplerEnv success rate is the target.

## Free-GPU fine-tune — **freezing the VLM is not enough**

Measured, not assumed ([tools/plan_finetune_memory.py](tools/plan_finetune_memory.py)):

```
total 2.976B    backbone 1.557B (vision 428M + Qwen3 1.116B)
                action head 1.419B (DiT 1.092B + projectors 327M)
trainable with the SHIPPED config: 1.620B   (whole action head + LLM layers 12-15)
```

Steady-state floor, activations excluded, "fits" discounted to 75% of VRAM:

| preset | optimiser | trainable | floor | fits |
|---|---|---|---|---|
| released    | AdamW      | 1.620B | 26.7 GB | A100 only |
| released    | AdamW-8bit | 1.620B | 17.6 GB | 2×T4, L4, A100 |
| action_head | AdamW      | 1.419B | 24.0 GB | A100 only |
| action_head | AdamW-8bit | 1.419B | 16.1 GB | 2×T4, L4, A100 |
| **projectors** | **AdamW** | **0.327B** | **9.8 GB** | **single T4** ✓ |

**An earlier version of this file said "freeze the backbone, train the action head" was the
free-tier plan. That is wrong** — the DiT alone is 1.09B parameters, so AdamW wants ~24GB
with the VLM already frozen. Freezing the VLM is necessary, not sufficient.

What actually runs where:
- **single T4 (Kaggle/Colab free)** → `projectors` preset only: state/action encoders,
  action decoder, vlln. 327M trainable.
- **2×T4 FSDP (Kaggle)** or **L4** → `action_head` with 8-bit Adam.
- the released recipe is an A100 job.

Also: T4 is sm_75 — no bf16 and no flash-attn v2 — so the SDPA/fp16 path is not a
convenience here, it is the only path that runs at all. Resolve dtype through
`common.device`, never hardcode bf16.

Training goes through `common.trainer.ResumableTrainer` (CLAUDE.md: never a bespoke loop)
so a killed session resumes at step level. Recipe and estimator live in
[finetune.py](finetune.py); `build_optimizer` covers only trainable parameters, since
handing AdamW the frozen 2.6B is the easiest way to OOM while believing the backbone is free.


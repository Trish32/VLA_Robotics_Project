# GR00T N1.6 — plan

**The gap: no published metric reproduced.** An experimental GR1 task baseline is now scored.
LIBERO / SimplerEnv success rate is the target.

The active strategy track is [STRATEGY.md](STRATEGY.md): an actual GR00T
no-planning baseline on the official GR1 task, with a locally tested driver.
As of 2026-10-04 the real task two-step smoke passes, including same-seed
observation/action reproducibility. The separate private Kaggle CUDA smoke and
ten-episode diagnostic baseline are complete: **3/10 successful (30%)**, seven
budget stops, on T4/fp16. SDPA-vs-official-flash numerical fidelity remains unverified;
this is an experimental task baseline, not a published-metric reproduction.

The paired [budget scan](BUDGET_SCAN.md) is complete: **3/10 at 720 steps, 5/10 at
1,080, and 6/10 at 1,440**, with all 787 old decisions reproduced exactly. Seeds
1/4/9 recover through late drawer closure. Remaining cases are missed placement
(2/7/8) and incomplete closure (0), checked against video/task signals.
The paired [execution contrast](EXECUTION_SCAN.md) is complete as of 2026-10-05:
**execute 4 scores 0/10 at 720 and 4/10 at 1,440**, below execute 8 at both
deadlines. Two long-budget gains are outweighed by four regressions; policy calls
increase 2.65×. **Retain execute 8.** All first predictions match; independent
analysis verifies the complete diagnostics, and the source-bundle/cloud suite
passes 61 tests (2 skipped).

[Lossless failure-stage capture](DECISION_CAPTURE.md) is complete: **46/46** CUDA
noise/action/post-call-RNG replays, **184** saved alternative predictions, unchanged
792 reference decisions on four failures and two successful controls. Independent
local analysis verifies every artifact/label and 6,333 video frames. Cloud tests
pass 76 (2 skipped); the source bundle passes 75 locally (3 skipped).

Next validate faithful simulator branch replay and evaluate candidate outcomes
from the same environment state before selecting a policy rule. Prediction variation
alone does not establish selection headroom. The same-input CUDA numerical oracle
on sm_80+ remains an independent fidelity gate.

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

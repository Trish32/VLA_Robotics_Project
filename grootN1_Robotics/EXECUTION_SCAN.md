# GR1 execution-horizon contrast — completed 2026-10-05

**Retain execute 8.** On the same ten seeds, execute 4 scores **0% by 720 steps /
40% by 1,440**, versus **30% / 60%** for execute 8. More frequent observation does
not improve this diagnostic set and costs more policy calls. Policy parameters,
candidate selection and task rules stay fixed. This experiment measures the
adapted SDPA/fp16 policy; official numerical fidelity and published-metric
reproduction remain unverified. The design and decision rule below were recorded
before submission.

## Fixed comparison

- Same official GR1 can-to-drawer-close task, checkpoint and patches, seeds **0–9**,
  T4/fp16, 20 Hz simulator, package versions and asset manifests.
- Reference: private `trishli/gr00t-gr1-budget-scan`, version 1, report SHA256
  `01925e5f107b7d85b3655513bf0e4ae1c27f52d51cab02f58c74cedd66e89bc7`.
- Treatment: execute **4 of 16** actions, observe again every **0.2 s** instead of
  0.4 s. Stop on official success or at **1,440 steps**; report the same
  360/720/1,080/1,440-step deadlines (18/36/54/72 s of simulation time).
- Mount and hash-check the reference before setup. Require matching checkpoint,
  policy/patch sources, simulator, assets, dependencies and accelerator.
- For every seed, require the **same initial observation hash and complete first
  state/action prediction**. The first four executed actions therefore agree.
  Later decisions and success steps may differ, including regressions. This uses
  an explicit initial-only reference gate; the old strict prefix gate stays the
  default for repeated-baseline/budget experiments.
- Reuse the recorded execute-8 control rather than spend quota rerunning it.
  Task and model RNG start from the same episode seed. Additional policy calls
  consume more noise draws: this is the practical execution-horizon contrast,
  not an isolation of feedback timing from all stochastic effects.
- Record both existing camera streams and the same read-only task predicates,
  with aligned videos and per-step JSONL. They do not enter the policy.

## Measurements and decision

Primary outcome: **success by 720 steps**, on all ten paired seeds. Secondary:
success by 1,440, both-budget improved/regressed seed sets, failure-stage changes,
per-seed completion times, policy calls and inference cost. Report all outcomes;
do not restrict the contrast to the four previously unsuccessful seeds. Compare
completion speed within seeds that succeed under both settings, avoiding a speed
average that changes its successful-episode composition.

Prefer execute 4 provisionally only if the 720-step success rate improves and the
1,440-step rate does not decrease, with all pairing/runtime checks passing.
Otherwise retain execute 8 as the current setting and use the observed stages to
choose the next diagnostic. Report individual regressions even when aggregates
improve. Ten reused scenes do not establish generalisation; a positive result
still needs held-out scenes before a broad performance claim.

## Measured paired result

Private kernel `trishli/gr00t-gr1-execution-scan`, **version 1** (kernel 137131725),
completed on T4/fp16. All ten initial observation hashes and complete first
state/action predictions match the execute-8 control exactly. Checkpoint,
policy/patch hashes, dependencies, assets and accelerator match. Later decisions
are intentionally allowed to differ. Independent local analysis verifies both
success curves, decision cadence and all **13,466** video frames/task-signal rows.

| Deadline, simulator steps | Simulation seconds | Execute 8 | Execute 4 |
|---:|---:|---:|---:|
| 360 | 18 | 2/10 (20%) | 0/10 (0%) |
| 720 | 36 | 3/10 (30%) | 0/10 (0%) |
| 1,080 | 54 | 5/10 (50%) | 0/10 (0%) |
| 1,440 | 72 | 6/10 (60%) | 4/10 (40%) |

At 720 steps, seeds **3/5/6** regress and none improve. At 1,440, seeds **0/2**
improve, **3/4/6/9** regress, **1/5** remain successful and **7/8** remain unsuccessful.
The final execute-4 Wilson 95% interval is **16.8%–68.7%**. This small reused scene
set supports retaining the current setting, not a general claim about shorter
execution horizons.

| Seed | Execute-8 success step | Execute-4 success step | Execute-4 observation |
|---:|---:|---:|---|
| 0 | — | 1,093 | Inside from 242; late closure |
| 1 | 1,348 | 1,159 | Inside from 199; late closure |
| 2 | — | 1,391 | Inside from 258; late closure |
| 3 | 284 | — | No placement; drawer closes empty |
| 4 | 778 | — | Inside from 265; drawer remains open |
| 5 | 283 | 1,173 | Inside from 259; much later closure |
| 6 | 676 | — | Inside from 201; drawer remains open |
| 7 | — | — | Placement improves, inside from 221; drawer remains open |
| 8 | — | — | Contact without placement |
| 9 | 1,020 | — | Contact without placement |

`—` means no success by 1,440 steps. Camera frames and task signals confirm the
placement/closure split. Contact alone does not establish a stable grasp. The
four new successes all place by step 259, then require another **851–1,133 steps**
to finish. Among seeds successful under both settings, seed 1 finishes 189 steps
earlier (9.45 simulation seconds), while seed 5 finishes 890 later (44.5 seconds).
Seed 7 improves placement without completing the task. These are observed
behaviours; they do not root-cause an adapter bug or isolate feedback timing from
the changed consumption of policy noise.

| Episode execution measurement | Execute 8 | Execute 4 |
|---|---:|---:|
| Simulator steps / policy calls | 10,149 / 1,272 | 13,456 / 3,366 |
| Total episode time | 80.7 min | 109.1 min |
| Total inference time | 287.1 s (5.9%) | 742.7 s (11.3%) |
| Inference latency, median / p95 | 0.233 / 0.249 s | 0.230 / 0.252 s |

The completed ten-episode workload takes **35.2% more wall time** and **2.65×**
as many policy calls; these totals include changed trajectories and stopping
times. There are no runtime errors or non-finite actions. The predeclared adoption
criterion fails at both deadlines, so the default remains **execute 8**.

## Follow-up diagnostic

The follow-up captures **lossless policy observations and noise/RNG state at failure
stages** and compares predictions on identical inputs. Candidate quality still
requires faithful simulator branching. This execution contrast saved only initial
NPZ inputs and later compressed videos/task signals; those recordings cannot
reconstruct exact later policy inputs. Prioritise failed placement and delayed/incomplete drawer closure before
designing a stage-aware selector or fine-tuning. Task predicates remain diagnostic
labels rather than deployed policy inputs. The sm_80+ numerical oracle is a
separate fidelity gate.

Follow-up [lossless capture](DECISION_CAPTURE.md) is complete: 46 exact CUDA
noise/action/RNG replays and 184 alternative predictions, with all reference
trajectories unchanged. Next validate simulator branching before evaluating
candidate quality; variation alone does not establish selection headroom.

## Reviewable implementation

- `baseline.py` checks the first full decision in initial-only mode, then permits
  re-observation at the changed horizon; strict-prefix behaviour remains tested.
- `tools/eval_baseline.py` allows the execution difference only in this explicit
  reference mode. Other provenance checks remain mandatory.
- `tools/prepare_kaggle_baseline.py --execution-scan-reference ...` creates the
  dedicated private `trishli/gr00t-gr1-execution-scan` job; the source allowlist
  contains no weights, upstream clones, datasets or credentials.
- `tools/analyze_execution_scan.py` independently checks pairing, first predictions,
  decision cadence, complete episode budgets, both curves and video/stage alignment.

## Commands

```bash
python grootN1_Robotics/tools/prepare_kaggle_baseline.py \
  --checkpoint-revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 \
  --smoke-report grootN1_Robotics/data/gr1_execute4_smoke.json \
  --execution-scan-reference grootN1_Robotics/data/kaggle_budget/v1/gr1_baseline.json \
  --out grootN1_Robotics/.kaggle_execution
# Use an environment with the authenticated Kaggle client.
python grootN1_Robotics/tools/kaggle_baseline.py status \
  --stage grootN1_Robotics/.kaggle_execution
python grootN1_Robotics/tools/kaggle_baseline.py pull \
  --stage grootN1_Robotics/.kaggle_execution \
  --out grootN1_Robotics/data/kaggle_execution/v1
python grootN1_Robotics/tools/analyze_execution_scan.py \
  --reference grootN1_Robotics/data/kaggle_budget/v1/gr1_baseline.json \
  --scan grootN1_Robotics/data/kaggle_execution/v1/gr1_baseline.json \
  --diagnostics grootN1_Robotics/data/kaggle_execution/v1/diagnostics \
  --out grootN1_Robotics/data/kaggle_execution/v1/analysis
```

The real two-step CPU smoke matches the reference initial observation and first
action prediction. The actual extracted source bundle and cloud suite each pass
**61 tests, 2 skipped**; baseline/telemetry tests pass 22, and the two independent
analysis suites pass 22 (13 execution-contrast and 9 budget-analysis checks).

Local ignored artifacts: `data/kaggle_execution/v1/gr1_baseline.json`,
`diagnostics/`, and `analysis/summary.json` plus per-seed contact sheets. SHA256:

- Completed report: `95af118ca3b15c8e03add4219ad6ed1f9c05fad87e8d41ec1b6306e3e22082a5`.
- Submitted script: `0db34f59dc92e412c0254813b37e79feb9fb44dfe97b4a9ab20800d316b1a56c`.
- Source bundle: `2277ff09959319d118de0e5e1c97e0711a3d9093f61adc99667b45ab465a9a7b`.

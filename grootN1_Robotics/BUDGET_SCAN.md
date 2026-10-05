# GR1 budget sensitivity diagnostic

**Completed, 2026-10-04:** the same adapted policy succeeds in **3/10 episodes
within 720 steps**, and **6/10 within 1,440 steps**. All ten initial observations
and all **787** old state/action decisions reproduce exactly. Three old budget
stops recover; four remain unsuccessful at the longer deadline. This establishes
budget sensitivity, not an improvement to the policy. The experiment specification
below was recorded before submission.

This extends the existing no-selection policy baseline. No policy parameters,
candidate selection, task sampling, camera processing, action decoding or execution
horizon are changed. It tests budget sensitivity, not planning effectiveness or
published-metric fidelity. See [STRATEGY.md](STRATEGY.md).

## Fixed experiment

- Official `PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands` task, seeds **0–9**.
- Same checkpoint, patches, T4/fp16, official simulator and asset manifests as
  private baseline kernel `trishli/gr00t-gr1-strategy-baseline`, version 3.
- Execute **8 of 16** predicted actions per call; simulator frequency **20 Hz**.
- One continuous rollout per seed, stopping immediately at official task success
  or at **1,440 steps**. Summarise success by 360/720/1,080/1,440 steps, corresponding
  to 18/36/54/72 seconds of **simulation time**, not wall time.
- Mount the previous private kernel's report and require its exact SHA256 before
  downloading dependencies. Require matching policy/patch hashes, checkpoints,
  packages, simulator and assets; compare initial observation hashes and every
  recorded state/action decision against the existing prefix. A mismatch aborts
  evaluation and is not counted as an unsuccessful task.
  Reference report SHA256:
  `8dd98959210fa143e560b7158834a6f9e92cb01a9f7182f66ce9dda4cb4aed0d`.
- Previously successful seeds must reproduce their success at the same step.
  The 720-step success set must remain **{3,5,6}**, or the paired comparison is invalid.

## Primary measurements

Report successful seeds and success rates at every budget; report newly successful
seeds in (720,1,440], time-to-success, and remaining censored episodes. A later
success never enters an earlier deadline's rate. All rates retain denominator 10.
No result at this finite horizon establishes that a remaining episode could never
succeed. Ten reused seeds are a diagnostic, not a population-level estimate.

## Read-only failure observations

Record both already-rendered ego-view camera observations at 20 frames/s into MP4;
record a JSONL row at reset and every simulator step, with matching step/time:

- left/right grasp-contact predicates from upstream `_check_grasp`;
- target object position;
- object-inside-drawer predicate with the official `partial_check=True`;
- drawer's normalised open fraction and official close threshold `<=0.005`;
- official success flag, checked against the same task predicate.

Camera panel order and video frame counts are saved separately. No additional
render, RNG sample, task-state write or policy-input change is introduced.
Contact is not proof of a stable grasp. The official inside predicate checks the
object centre, not complete bounding-box containment. Stage summaries distinguish
no detected grasp contact, contact without placement, lost placement, inside-drawer
without task success, and success; these observations require visual interpretation
before asserting a root cause. Reward is the source of the upstream success flag,
so their agreement is not independent confirmation.

## Next decision

The subsequent [execute-4 versus execute-8 contrast](EXECUTION_SCAN.md) completed
on 2026-10-05 with the checkpoint, seeds and deadlines fixed. Execute 4 scores
**0% / 40% at 720 / 1,440**, versus **30% / 60%** for execute 8. Retain execute 8;
more frequent observation did not improve this diagnostic set.

Lossless failure-stage policy observations and noise/RNG state are now captured
and exactly replayed, as recorded below. Next validate faithful simulator branching
and measure candidate outcomes before designing a stage-aware selector.
Compressed videos alone cannot reconstruct exact policy inputs.
Placement must be verified before
treating drawer closure as task completion. Simulator predicates used here are
diagnostic observations, not deployed policy inputs. Fine-tuning follows evidence
that execution or selection alone is insufficient. The sm_80+ flash-vs-SDPA
numerical oracle remains an independent validation gate before fidelity claims.

The [lossless capture follow-up](DECISION_CAPTURE.md) is now complete: 46 exact
CUDA replays and 184 saved alternatives, with unchanged reference trajectories.
Next validate faithful simulator branching to measure candidate outcomes; sampling
variation alone is not evidence for a selector.

## Measured paired result

Private kernel `trishli/gr00t-gr1-budget-scan`, **version 1**, completed on T4/fp16.
The mounted reference report matches the specified SHA256. The independent local
analyser checks policy/patch hashes, provenance, every old decision, success steps,
all diagnostic step/time rows, MP4 frame counts/FPS, and the recomputed budget curve.

| Deadline, simulator steps | Simulation seconds | Successful seeds | Success |
|---:|---:|---|---:|
| 360 | 18 | 3, 5 | 2/10 = 20% |
| 720 | 36 | 3, 5, 6 | 3/10 = 30% |
| 1,080 | 54 | 3, 4, 5, 6, 9 | 5/10 = 50% |
| 1,440 | 72 | 1, 3, 4, 5, 6, 9 | 6/10 = 60% |

The additional successes are seed **4 at step 778**, **9 at 1,020**, and **1 at
1,348**. The three prior successes remain at steps 284/283/676. The longer-budget
Wilson 95% interval is **31.3%–83.2%**. These ten reused scenes are a small diagnostic.
The additional 30 percentage points come from the deadline change, with unchanged
policy parameters and execution horizon.

| Seed | First grasp contact | First inside drawer | Success step / stop | Observed outcome |
|---:|---:|---:|---|---|
| 0 | 98 | 240 | budget 1,440 | object inside; drawer not closed |
| 1 | 102 | 211 | success 1,348 | late closure |
| 2 | 105 | — | budget 1,440 | lifted target, missed drawer; target beside fixture |
| 3 | 102 | 198 | success 284 | original success |
| 4 | 103 | 232 | success 778 | late closure |
| 5 | 102 | 208 | success 283 | original success |
| 6 | 104 | 197 | success 676 | original success |
| 7 | 100 | — | budget 1,440 | target leaves hand during transport and falls to floor |
| 8 | 276 | — | budget 1,440 | brief contact, no sustained lift into drawer |
| 9 | 114 | 208 | success 1,020 | late closure |

These stage observations were checked against the recorded camera frames and
object positions; they identify failure behaviour, not a root-caused adapter bug.
Seed 7's target is at z=0.033 m at the end; seed 8 has only seven contact-positive
steps and a maximum centre-height rise of 0.0113 m. Seed 2 lifts the target about
0.334 m, but it lands outside the drawer. None of these three ever satisfies the
official inside predicate.

All three late successes already place the object by step 232 (11.6 s); their
post-placement time to success is 27.3–56.85 s. Seed 0 remains inside from step 240
through the end, while the drawer repeatedly opens and closes without reaching the
success threshold. Its closest post-placement open fraction is 0.081 at step
1,210, versus the required <=0.005. The upstream ratio is **not clamped to [0,1]**;
reported values above 1 are retained. Arm motion is therefore not a reliable
task-progress measure. Closing an empty drawer also fails: seed 7 has a closed
drawer at step 720 but no object inside.

The scan executes **10,149 simulator steps / 1,272 policy calls**, taking **4,842 s
(80.7 min)** across episodes. Inference accounts for 287.1 s (5.9%); median/p95
latency is **0.233/0.249 s**. All **10,159** recorded frames align with reset plus
simulator steps. No runtime errors or non-finite actions occur.

## Reproduce and review

Raw reports, videos, JSONL signals, source manifest and saved first inputs are
ignored working data in `data/kaggle_budget/v1/`. The original report remains in
`data/kaggle_baseline/v3/`. The independently verified summaries and readable
contact sheets are in `analysis_v2/`; additional failure-stage frames are in
`phase_review/`. They are not GitHub assets.

```bash
python grootN1_Robotics/tools/prepare_kaggle_baseline.py \
  --checkpoint-revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 \
  --smoke-report grootN1_Robotics/data/gr1_smoke_captured.json \
  --budget-scan-reference grootN1_Robotics/data/kaggle_baseline/v3/gr1_baseline.json \
  --out grootN1_Robotics/.kaggle_budget
# Use an environment with the authenticated Kaggle client.
python grootN1_Robotics/tools/kaggle_baseline.py status \
  --stage grootN1_Robotics/.kaggle_budget
python grootN1_Robotics/tools/kaggle_baseline.py pull \
  --stage grootN1_Robotics/.kaggle_budget \
  --out grootN1_Robotics/data/kaggle_budget/v1
# Use a NEW output directory for another analysis run.
python grootN1_Robotics/tools/analyze_budget_scan.py \
  --reference grootN1_Robotics/data/kaggle_baseline/v3/gr1_baseline.json \
  --scan grootN1_Robotics/data/kaggle_budget/v1/gr1_baseline.json \
  --diagnostics grootN1_Robotics/data/kaggle_budget/v1/diagnostics \
  --out grootN1_Robotics/data/kaggle_budget/v1/analysis_review
```

Diagnostic hooks pass 16 targeted tests; the extracted/cloud upload bundle passes
55 tests (2 skipped), and independent paired-report/video checks pass 9 tests.
The real two-step local smoke reproduces its original input/action prefix with
telemetry enabled. Official numerical fidelity and published-metric reproduction
remain unverified.

# GR1 lossless decision capture and replay

**Completed 2026-10-05:** all **46/46** saved decisions replay exactly on CUDA;
**184** alternative action chunks are saved. All **792** reference decisions and
stopping outcomes remain unchanged. Independent local verification checks every
capture, alternative and **6,333** aligned video frames. Candidate quality has
not been evaluated; next validate faithful simulator branching.

This diagnostic supplies exact later-stage policy inputs for activation comparisons
and candidate experiments. **Execute 8 remains the current setting.** It does not
change selection, weights, task rules or the success deadline. The previous
[execution-horizon contrast](EXECUTION_SCAN.md) did not support adopting execute 4.
Official flash-vs-SDPA numerical fidelity and published-metric reproduction remain
unverified.

## Fixed cloud design

Record these choices before submission; report every captured point, not just
points with large candidate differences:

- Reference: private `trishli/gr00t-gr1-budget-scan`, v1; SHA256
  `01925e5f107b7d85b3655513bf0e4ae1c27f52d51cab02f58c74cedd66e89bc7`.
- Same checkpoint, pinned simulator, dependencies, patches, assets, T4/fp16,
  20 Hz, execute 8, 1,440-step deadline.
- Four remaining reference failures: seeds **0/2/7/8**; successful controls **3/5**.
  This is a diagnostic subset, not a new success-rate benchmark.
- Capture before decisions at steps **0/96/160/224/256/360/720/1,080/1,432**,
  whenever the original episode has not already ended. These cover early approach,
  grasp/transport, placement and later drawer interaction. Expect **46** records
  (nine per failure, five per successful control).
- Require every initial input and **every full reference state/action decision**
  and stopping outcome to remain exactly unchanged. Abort on divergence.
- Save lossless batched video/state/instruction inputs, including original container
  types, decoded full action chunks, Python/NumPy/Torch CPU and policy-device RNG
  before/after inference, and the actual initial flow-matching noise tensor.
- Read noise via a temporary pre-hook on the pinned upstream action encoder's
  first call, with timestep zero. No noise replacement or additional sampling.
- Save read-only drawer/grasp labels separately. They never enter policy inputs.
  Existing camera diagnostics continue without extra renders.
- Reconstruct the policy in a separate process on the same backend/dtype. Require
  **exact noise, full actions and post-call RNG** at every saved decision. A mismatch
  fails the job; no tolerant fallback or cherry-picking.
- After replay, sample **four** independent candidates per observation using
  fixed hash-derived seeds. Save all **184** alternative chunks and per-joint-group
  RMS/maximum action differences. Isolate their RNG from the caller and other GPUs.

## What this can establish

Exact replay makes later inputs usable for stage/operator activation differences
against pinned upstream, and supplies reproducible candidate predictions. Candidate
variation alone does **not** establish better actions or selection headroom. No
simulator state/controller/wrapper snapshot or branch rollout is implemented in
this step. Next validate faithful simulator branch replay, then evaluate candidates
from the same environment state before choosing a selector or fine-tuning.

The passive-recording gate measures reproducibility of our adapted policy, not
agreement with official flash attention. Compressed videos alone cannot supply
these exact observations. No training loop or upstream source is changed.

## Implementation and local gates

- `decision_capture.py`: lossless snapshots, temporary noise hook, safe tensor/primitive
  RNG loading (`weights_only=True`), file hashes, exact replay and isolated candidate
  sampling. Incomplete/changed artifacts fail; existing records are never overwritten.
- `baseline.py` / `tools/eval_baseline.py`: optional selected-decision hook and
  explicit diagnostic seeds; strict prefix checks stay enabled. Recording I/O is
  excluded from inference latency, while episode time includes its overhead.
- `tools/replay_decisions.py`: check complete rollout inventory, source/checkpoint/
  accelerator/package provenance and recorded state/actions before loading policy.
- `tools/analyze_decision_captures.py`: independently verify full reference trajectories,
  every expected capture/replay, diagnostic labels, video alignment and saved
  candidate action hashes/differences.
- `tools/prepare_kaggle_baseline.py --decision-capture-reference ...`: dedicated
  private `trishli/gr00t-gr1-decision-capture`; source-only allowlist, hash-mounted
  reference, CUDA smoke plus replay before the full diagnostic. Local smoke must
  match current source hashes before staging.

The real local two-step CPU smoke records the same scene and first full prediction
as the previous smoke. Exact replay validates noise/actions/post-call RNG and one
alternative sample. Unit tests cover passive action/RNG equivalence, corruption,
language-container preservation, hook cleanup, caller RNG restoration, saved
alternative actions and changed/missing trajectories. A CUDA-only tiny numerical
test covers CUDA noise/RNG and preservation of a second GPU's generator; it is
skipped locally and must pass in the cloud.

## Completed measurements

Private `trishli/gr00t-gr1-decision-capture` **v2** (kernel 137196659) completes
the fixed six-seed diagnostic on T4/fp16. Every full initial/continuing reference
decision matches the recorded execute-8 control. The independent process verifies
each saved noise tensor, entire decoded action chunk and post-call RNG exactly.
No tolerance, selection or modified stopping condition is used.

| Seed | Original stopping outcome, unchanged | Saved / exact-replayed decisions |
|---:|---|---:|
| 0 | Budget stop at 1,440 | 9 / 9 |
| 2 | Budget stop at 1,440 | 9 / 9 |
| 3 | Success at 284 | 5 / 5 |
| 5 | Success at 283 | 5 / 5 |
| 7 | Budget stop at 1,440 | 9 / 9 |
| 8 | Budget stop at 1,440 | 9 / 9 |

The independently checked capture labels distinguish **15** snapshots before any
detected contact, **5** with contact outside the drawer, **9** inside without
closure, and **17** outside the drawer after earlier contact, with no current
contact. The last category does not imply an earlier stable grasp or placement.
The fixed grid does not capture every short contact event; seed 8's brief contact
is present in full per-step diagnostics but absent at these selected decision points.

All 184 candidate files pass shape/dtype/finiteness and hash checks, and their
reported RMS/maximum differences recompute exactly. Per-group RMS ranges against
the recorded prediction, in that group's decoded action coordinates:

| Joint group | Minimum / maximum RMS difference, all 184 alternatives |
|---|---:|
| Left arm | 0.00989 / 0.42002 |
| Right arm | 0.01227 / 0.05208 |
| Left hand | 0.02441 / 1.61679 |
| Right hand | 0.02155 / 1.49296 |
| Waist | 0.00479 / 0.01166 |

These measure sampling variation, **not correctness or task improvement**. Groups
have different coordinates; this table is not a quality ranking. Original stopping
outcomes stay unchanged by design, so the selected two successful controls do not
provide a new benchmark score. No runtime/nonfinite-action error occurs in v2.

Episode execution totals **6,327 steps / 792 policy calls / 2,897.6 s (48.3 min)**,
including recording overhead; inference totals 175.1 s. Cloud unit tests pass
**76, 2 skipped** including the CUDA oracle; the extracted local bundle passes
**75, 3 skipped**. Twelve independent capture-analysis tests pass. Independent
artifact verification checks complete reference trajectories, all 46 inputs/RNG/
action records and labels, all 184 candidate hashes/contracts/differences, and
6,333 video frames. No simulator branch/candidate-quality result is claimed.

Ignored local artifacts: `data/kaggle_capture/v2/decisions/`,
`candidate_predictions/`, `diagnostics/`, `decision_replay.json` and
`analysis/summary_v2.json` (the earlier summary used a less precise category name).
Completed SHA256:

- Rollout report: `36d289982ecc9dfb5c02cc3f87038c754f65154e72d8d2d0719e61ec178e5eff`.
- Replay report: `b64769ab589fc8c72cb91256c56f96186fc823238fa0f409b0ab5e2382158812`.

## Commands

```bash
python grootN1_Robotics/tools/prepare_kaggle_baseline.py \
  --checkpoint-revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 \
  --smoke-report grootN1_Robotics/data/decision_capture_smoke/v4/rollout.json \
  --replay-smoke-report grootN1_Robotics/data/decision_capture_smoke/v4/replay.json \
  --decision-capture-reference grootN1_Robotics/data/kaggle_budget/v1/gr1_baseline.json \
  --out grootN1_Robotics/.kaggle_capture
python grootN1_Robotics/tools/kaggle_baseline.py status \
  --stage grootN1_Robotics/.kaggle_capture
python grootN1_Robotics/tools/kaggle_baseline.py pull \
  --stage grootN1_Robotics/.kaggle_capture \
  --out grootN1_Robotics/data/kaggle_capture/v2
python grootN1_Robotics/tools/analyze_decision_captures.py \
  --reference grootN1_Robotics/data/kaggle_budget/v1/gr1_baseline.json \
  --artifacts grootN1_Robotics/data/kaggle_capture/v2 \
  --out grootN1_Robotics/data/kaggle_capture/v2/analysis/summary_new.json
```

Status: private kernel **version 2** (kernel 137196659) completed after real CPU
save/replay and **75 source-bundle tests pass, 3 skipped**. The server confirms
`is_private=true`. Version 1 stopped at the CUDA tiny test before collecting any
policy results: accessing `torch.cuda.default_generators` before lazy initialization
raised `IndexError`. Version 2 initializes CUDA explicitly and adds a CPU cold-start
ordering regression. V2 cloud tests pass **76, 2 skipped**, including the CUDA noise/
RNG oracle and second-GPU generator preservation. Full later-stage CUDA replay
and candidate variation measurements pass independent local artifact analysis.

Submitted v2 SHA256:

- Script: `9480ab5a24e09484d3c41f8ecfbf777443c84dba675d9302350d8962c9ece430`.
- Source bundle: `89239a0ecf8c917215c8a49522f07f726484d7c6d95d5b23e3edbc2d08e147fb`.

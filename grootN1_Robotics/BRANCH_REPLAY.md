# GR1 cold simulator branch replay

Registered before the cloud run, 2026-10-05. **Full cloud gate passed and independently
reverified locally on 2026-10-06: six seeds, 46 captured inputs and 6,333 exact
control frames.** Candidate quality and policy success are not improved by this
diagnostic. Execute 8 remains the setting.

## Question and decision rule

Can a fresh official environment reconstruct a captured decision and its continuation
by replaying the recorded action history? Only then may a counterfactual action be
evaluated from that history. Restoring qpos/qvel alone is insufficient: controller
goals, solver warmstart, task bookkeeping and observation windows also matter.

This implementation deliberately uses the official constructor/reset/step paths,
not a partial simulator snapshot. It is a slower correctness oracle. A future fast
restore must match this oracle before it replaces history replay.

Fixed reference: completed six-seed decision capture v2, rollout SHA256
`36d289982ecc9dfb5c02cc3f87038c754f65154e72d8d2d0719e61ec178e5eff`.
Seeds **0/2/7/8** are failures; **3/5** are successful controls. All **46** captured
policy observations must match, with the same official simulator, assets, package
versions, task, execute 8 and stopping conditions. No policy weights are loaded or
new predictions sampled; the recorded decoded actions are replayed exactly.

For each seed:

1. Construct a fresh seeded environment. Replay the full recorded episode continuously.
   Require the original initial observation hash, every decision's state and every
   step's original drawer/contact/object-position signals to match exactly. At all
   saved decisions, require full batched video/state/language equality, including
   dtype and shape. Require the original stopping outcome and step.
2. Construct a second fresh seeded environment. Advance to and pause at each captured
   decision, then continue. Require **every step**, including frame zero, to match the
   continuous control exactly: full observation hash, native MuJoCo
   `mjSTATE_INTEGRATION` hash (including control and solver warmstart), task signals,
   per-step reward and termination/truncation/success flags.
3. Save both complete traces and hashes. Refuse on any mismatch, missing capture,
   changed provenance, non-finite data, premature stop or incomplete episode.

**Pass:** all six episodes and 46 inputs satisfy every exact gate. Then candidate
outcome experiments are admissible, although their cost still needs budgeting.
**Fail:** no candidate-quality claim; root-cause the first mismatch with the same
recorded inputs before introducing a fast snapshot or tolerance.

The original run did not save native integration state. Therefore native state is
compared between the two new replays; only the observations, decision states, task
signals and stopping outcome are compared directly with the original run. This
distinction is retained even on a pass.

## Local verification

The real official GR1 **two-step** smoke passes on CPU: continuous replay and fresh
history replay paused after step 1 produce identical initial inputs and **three**
step traces, each containing **1,209** native integration-state entries. Full RGB
observations, task signals, rewards and stop flags match exactly. Independent
analysis checks both saved trace hashes and contents. This is an early-state smoke,
not evidence about later grasp or drawer-contact branches.

Eleven tiny tests cover in-chunk offsets, decision pauses, changed pixels/dtypes,
hidden integration state, per-step contact labels, incomplete history, wrong initial
state, premature stops, altered outcomes and non-finite actions/observations.
Seven independent-analysis tests also reject changed continuation traces, contact
labels, camera pixels, missing frames, non-finite rewards and premature stops.
The selected regression suite passes **62, 1 skipped**. The exact source-only cloud
bundle passes **93, 3 skipped** locally; the CUDA RNG oracle remains cloud-only.

Local report and independent analysis: ignored `data/branch_replay_smoke/v4/report.json`
and `analysis.json`. Its source hashes
must match before cloud staging. Mac rendering needs access to macOS graphics
services: the sandbox-only attempt stalled during graphics initialization and was
terminated; the actual short smoke passes outside that sandbox.

## Cloud execution

Dedicated private job: `trishli/gr00t-gr1-cold-branch-replay`. It mounts the existing
private decision-capture output by exact report hash. Only **26** allowlisted
adaptation/patch/test files are embedded, not captured inputs, datasets, credentials
or checkpoints. It reuses the baseline worker's pinned official simulator setup,
pins the reference package versions, verifies assets and runs tiny tests before the
full gate. This job does not download or instantiate GR00T weights.

The first submission attempt was rejected before execution for insufficient
destination-specific source-egress authorization. The user then explicitly
authorized starting the reviewed task. Version **1**, kernel **137266247**, is now
accepted and has **COMPLETED**. Server metadata confirms `is_private=true`, GPU enabled,
the expected private capture mount, and an exact match to the reviewed script hash:
`90273b7c4d164c9801d552c92db3e00f16f8f11a7cc4ff26218ab5ffa48b6a67`.
All six seeds pass both full replays: 0/2/7/8 stop at 1,440 steps; successful controls
3/5 stop at 284/283. All 46 captured inputs match, and all 6,333 control frames match
between the continuous replay and the fresh replay paused at captured decisions.
Cloud tests pass **94, 2 skipped**. Cloud independent analysis and a separate local
analysis of the downloaded traces both verify the report, original captures and
task signals. Report SHA256:
`17989233e5d2353535b68893523e3a8a94fe829c2070b4aefee7db40ed34170e`.
Saved results are `data/kaggle_branch/v1/branch_replay.json`,
`branch_analysis.json` and `branch_analysis_local.json`. Candidate outcomes remain
unmeasured; this gate establishes exact reconstruction, not selection headroom.

Kaggle generated the new task slug from the title, adding `cold` to the originally
requested `gr00t-gr1-branch-replay` ID. Local lookup and future staging now use the
actual returned ID; this did not create a second job. Submission and remote checks
are saved under ignored `data/kaggle_branch/v1/`.

```bash
PYTHONPATH=grootN1_Robotics/upstream grootN1_Robotics/.venv-baseline/bin/python \
  grootN1_Robotics/tools/verify_branch_replay.py \
  --rollout-report grootN1_Robotics/data/decision_capture_smoke/v4/rollout.json \
  --capture-dir grootN1_Robotics/data/decision_capture_smoke/v4/decisions \
  --out grootN1_Robotics/data/branch_replay_smoke/new/report.json --smoke

PYTHONPATH=grootN1_Robotics/upstream grootN1_Robotics/.venv-baseline/bin/python \
  grootN1_Robotics/tools/prepare_branch_replay.py \
  --smoke-report grootN1_Robotics/data/branch_replay_smoke/v4/report.json \
  --capture-report grootN1_Robotics/data/kaggle_capture/v2/gr1_baseline.json
```

The reusable `RecordedReplay.advance(step)` returns the reconstructed temporal
observation at that step and retains the actual official environment for later
counterfactual execution. Candidate sampling remains a separate operation. Frozen
reference actions after a counterfactual are not a closed-loop policy rollout and
must not be reported as one.

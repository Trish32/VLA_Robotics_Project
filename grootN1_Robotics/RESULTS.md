# GR00T N1.6 — results

What is verified. The checkpoint loads **0 missing / 0 unexpected / 0 shape-mismatch**
(3.29 B params) and one observation runs end to end to a finite action chunk on CPU.

**Not a reproduction:** no published metric has been reproduced — that needs LIBERO or
SimplerEnv. See [Plan.md](Plan.md).

## Candidate outcomes — local gate passed, cloud running, 2026-10-07

A real official two-step CPU smoke executes the captured reference and one saved
alternative in separate fresh environments. The reference reproduces the existing
oracle exactly; independent analysis verifies both interventions, signals and
six video frames. This is execution correctness, not evidence of candidate quality.
Reactive replanning, absolute budgets, mid-chunk stopping, paired isolated policy
RNG and corruption rejection are covered by 18 new tests. The selected branch and
candidate suites pass 36 tests; the actual source-only cloud bundle passes
**111, 3 skipped**, including tiny network forward/backward regressions.

The fixed pilot uses six decisions and 30 arms, each with a single saved action
chunk followed by real GR00T closed-loop continuation. Its worst case is 40,887
simulator steps, including repeated prefixes. After explicit user authorization,
private Kaggle job `trishli/gr00t-gr1-candidate-outcomes` version 1 (kernel 137527779)
was submitted and is RUNNING. Server privacy, GPU setting, both expected private
input mounts and the reviewed script hash match. **No candidate task-success or
selection gain is measured yet.** See
[CANDIDATE_OUTCOMES.md](CANDIDATE_OUTCOMES.md).

## Cold simulator branch oracle — full gate verified, 2026-10-06

The official GR1 two-step CPU control is replayed continuously and in a fresh
environment paused after step 1. All three step traces match exactly: full RGB/state/
language observations, 1,209 native MuJoCo integration-state entries, original task
signals, rewards and stop flags. Independent saved-artifact analysis verifies the
trace hashes and original captured input. Selected regressions pass **62, 1 skipped**;
the source-only cloud bundle passes **93, 3 skipped**.

The full cloud gate has completed as private `trishli/gr00t-gr1-cold-branch-replay`
version 1 (kernel 137266247). All **six seeds / 46 inputs / 6,333 control frames**
pass. Each seed is replayed continuously and in a second fresh environment paused
at captured decisions; full observations, native integration hashes, task signals,
rewards and stopping flags match exactly between those replays. Original captures,
task signals and stopping outcomes also match. Original native state was not saved
and is compared only between the two new replays. Cloud tests pass **94, 2 skipped**.
Cloud independent analysis and a separate local audit of the downloaded artifacts
both pass. Server privacy, capture mount and reviewed script hash match.
No candidate-quality or task-success gain is claimed. Registered rule, implementation
and limitations: [BRANCH_REPLAY.md](BRANCH_REPLAY.md).

## Active strategy baseline — 2026-10-04

The actual released checkpoint drives the pinned official GR1
`PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands` environment through two real
simulator steps on CPU. Two separate seed-0 runs produce identical initial
observation hashes and state/action traces (maximum decoded action difference 0).
No task-success rate is inferred from this smoke. Compatible legacy fixtures are
now present: **0 missing of 23,707** audited mesh/texture references.

The selected six-file local suite passes **49 tests, 1 skipped**. The actual
source-only upload bundle passes **48 tests, 2 skipped** in an isolated directory.
The language-conditioning test also holds initial flow noise fixed; four existing
real-checkpoint end-to-end tests pass with this correction.

Private Kaggle strategy job version 3 **completed** CUDA smoke and ten episodes,
seeds 0–9, execute 8 of 16 actions, maximum 720 simulator steps, on **Tesla T4 / fp16**.
The measured success rate is **3/10 = 30%**, Wilson 95% interval **10.8%–60.3%**.
Successful seeds are 3, 5 and 6 at simulator steps 284, 283 and 676. The other seven
episodes reach the 720-step budget. There are no runtime errors or non-finite actions.

| Measurement | Result |
|---|---:|
| Policy inference latency, median / p95 | 0.240 / 0.253 s |
| Simulator control frequency | 20 Hz |
| Simulator steps / policy calls | 6,283 / 787 |
| Episode execution time, total | 3,026 s (50.4 min) |
| Policy inference time, total | 181.6 s (6.0% of episode time) |
| Local/cloud checkpoint hash differences | 0 |

The worker also passes 48 tiny tests (2 skipped), resolves all 23,707 asset references,
and reports 0 missing / 0 unexpected / 0 shape mismatch. Its separate smoke and
baseline seed-0 initial observations and first decoded actions match exactly.
All ten episode initial observation hashes differ, confirming distinct sampled scenes.
Raw JSON, saved NPZ inputs, source bundle and logs are ignored working files in
`data/kaggle_baseline/v3/`; they are not uploaded to GitHub.

This result measures the adapted SDPA/fp16 policy on one task. It does not establish
official flash/bf16 numerical agreement or a published benchmark reproduction. The
seven budget stops do not by themselves distinguish grasp, transport or drawer failure.
Previous versions stopped during setup/collection and produced no policy score. Complete
provenance, commands and limitations are in [STRATEGY.md](STRATEGY.md).

## Paired budget scan — 2026-10-04

Private budget-scan kernel version 1 completes the same ten seeds on T4/fp16,
extending the deadline to 1,440 steps with unchanged policy and execution horizon.
All **ten initial observations / 787 old state-action decisions** match the original
baseline exactly, including the original success steps. Independent local checks
also verify all **10,159** video frames and per-step task signals.

| Deadline | Simulation time | Successful episodes |
|---|---:|---:|
| 360 steps | 18 s | 2/10 (20%) |
| 720 steps | 36 s | 3/10 (30%) |
| 1,080 steps | 54 s | 5/10 (50%) |
| 1,440 steps | 72 s | 6/10 (60%) |

New successes are seeds 4/9/1 at steps **778/1,020/1,348**. They all place the object
by step 232, then take longer to close the drawer. The final Wilson 95% interval
is **31.3%–83.2%**. This is sensitivity to the deadline, not a policy improvement.

Video and task signals distinguish the four remaining cases: seed **0** keeps the
object inside but fails to close the drawer; **2** lifts the target but misses the
drawer; **7** loses the target during transport and it falls to the floor; **8**
has only brief contact without a sustained lift into the drawer. These identify
observed failure behaviour, not a root-caused adapter bug or permanent inability.
The subsequent execution-horizon contrast below retains execute 8. Numerical
fidelity remains a separate gate.

The scan takes **80.7 min** across episodes, executing 10,149 steps / 1,272 policy
calls. Inference uses 287.1 s (5.9%), with median/p95 **0.233/0.249 s**. No runtime
errors or non-finite actions. New diagnostics pass 16 targeted tests; the actual
upload bundle/cloud suite passes 55 (2 skipped), and independent analysis checks
pass 9 tests. Raw data remains ignored in `data/kaggle_budget/v1/`.
Specification, per-seed measurements and review commands: [BUDGET_SCAN.md](BUDGET_SCAN.md).

## Paired execution-horizon contrast — 2026-10-05

Private execution-scan version 1 completes the same seeds with **execute 4**,
matching every initial observation and full first state/action prediction to the
execute-8 control. Checkpoint, policy/patch sources, assets and dependencies match.
Independent local analysis verifies both curves, decision cadence and **13,466**
video frames with aligned task signals.

| Deadline | Execute 8 | Execute 4 |
|---|---:|---:|
| 360 steps | 20% | 0% |
| 720 steps | 30% | 0% |
| 1,080 steps | 50% | 0% |
| 1,440 steps | 60% | 40% |

Execute-4 successes are seeds **0/1/2/5** at steps **1,093/1,159/1,391/1,173**.
At 1,440, seeds 0/2 improve but 3/4/6/9 regress. Seed 5 remains successful but
finishes 890 steps later; seed 1 finishes 189 earlier. Failed cases split into
no placement (3/8/9) and placement without closure (4/6/7). All four successes
place by step 259 but finish late. These stage observations do not identify an
adapter root cause.

**Retain execute 8:** execute 4 fails the predeclared adoption rule at both
deadlines. Its episodes take **109.1 min**, with **13,456 steps / 3,366 policy
calls**, 35.2% more wall time and 2.65× more calls than the control. Inference takes
742.7 s (11.3%), median/p95 **0.230/0.252 s**; no runtime errors or non-finite
actions. Source-bundle/cloud tests pass 61 (2 skipped); independent execution
analysis tests pass 13. Additional policy calls consume additional noise draws,
so this practical contrast does not isolate feedback timing from stochasticity.
Ten reused scenes do not establish generalisation.

The lossless failure-stage capture below supplies exact later inputs; compressed
videos alone cannot recover them. Design, per-seed results and artifact hashes:
[EXECUTION_SCAN.md](EXECUTION_SCAN.md). Raw outputs remain ignored in
`data/kaggle_execution/v1/`.

## Lossless decision capture and replay — 2026-10-05

Private decision-capture v2 completes four failed seeds **0/2/7/8** and successful
controls **3/5**, retaining execute 8 and the 1,440-step cap. All **792** original
state/action decisions and stopping results match exactly, including successful
steps 284/283. This diagnostic subset does not supply a new benchmark score.

All **46/46** saved decisions reproduce the exact initial flow noise, full action
chunk and post-call RNG in a separate CUDA process. Four fixed independent seeds
per observation generate **184** losslessly saved alternative chunks. Independent
local checks verify their hashes, shapes, finite values and reported action
differences, every input/label and **6,333** aligned video frames. Stage samples
include 15 before contact, 5 with contact outside the drawer, 9 inside without
closure and 17 outside after previous contact with no current contact. Brief events
between the selected points remain visible in full per-step diagnostics.

Episode execution takes **48.3 min**, 6,327 steps / 792 calls; inference totals
175.1 s. Cloud tests pass **76, 2 skipped**, including CUDA RNG/noise and second-GPU
preservation. The source bundle passes 75 locally (3 skipped); independent capture
analysis tests pass 12. V1 stopped at a CUDA generator initialization bug before
policy collection; the fix and cloud verification are recorded in bug_log [16].

**Candidate quality remains unevaluated.** Sampling variation alone does not
establish selection headroom. Next validate faithful simulator branch replay and
compare candidate outcomes from the same environment state before selecting a
rule or fine-tuning. Execute 8 remains the setting. Measurement/provenance details:
[DECISION_CAPTURE.md](DECISION_CAPTURE.md). Raw outputs remain ignored in
`data/kaggle_capture/v2/`.

## Port checklist

- [x] Upstream pinned, adaptation imports clean in `simple_bev_vldrive`, tiny-input tests green.
- [x] Backbone constructible without flash-attn/bf16 → `patches/0001`.
- [x] **`nvidia/GR00T-N1.6-3B` loads 0 missing / 0 unexpected / 0 shape mismatch** —
      1106 tensors, 3.29B params → [tools/load_checkpoint.py](tools/load_checkpoint.py).
      **Checkpoint key/shape gate passed; full fidelity requires a reproduced metric.**
- [x] **Found and fixed: SDPA silently ignored Eagle's image packing** →
      `patches/0002`, verified by [tools/check_attention_packing.py](tools/check_attention_packing.py).
      Cross-image attention leak 2.187e-01 → 0.0; dense paths now match an independent
      block-diagonal reference (eager exactly, sdpa to 6e-08). 8 regression tests.
      **This invalidated the "same math, different kernel" justification** — see bug_log [3].
- [ ] **flash-vs-SDPA on GCP L4** → [tools/compare_flash_vs_sdpa.py](tools/compare_flash_vs_sdpa.py).
      Cannot run on the free tier: FlashAttention-2 needs sm_80+, and Kaggle/Colab give
      T4 (sm_75) or P100 (sm_60). L4 (sm_89) is the cheapest box that can run the
      reference half. 
- [x] **End-to-end: real observation → action chunk, on CPU** →
      [tools/run_policy_cpu.py](tools/run_policy_cpu.py), 4 tests. A genuine frame from
      `demo_data/gr1.PickNPlace` with its real instruction ("pick the pear from the
      counter and place it in the plate") through Eagle and the flow-matching head to a
      finite `(1, 16, D)` chunk per joint group, in ~3s. Runs via
      [policy.py](policy.py), which composes and delegates to upstream's CUDA-only
      `Gr00tPolicy`, overriding construction only.
      **Not a metric** — sdpa + fp32 differ from the shipped flash + bf16 reference.
- [x] **LeRobot reader working on upstream's real `demo_data`** (5 episodes, 6-DOF state,
      480×640 video, language) → [lerobot_data.py](lerobot_data.py), 9 tests.
      Upstream's loader needs no `lerobot` install, so we use it directly; our layer adds
      only strict key validation and a video-backend probe. Four traps in bug_log [5] —
      including a **silent** one: a wrong state/action key warns and drops the column,
      training a model with no proprioception and no traceback.
      Requires `git lfs pull` in `upstream/` — demo_data ships as pointers.
- [ ] Flow-matching loss verified against the reference on identical noise draws.
- [x] **LIBERO simulator runs on macOS** — reset, step, and 256×256 offscreen render on
      Apple Silicon. Upstream's `setup_libero.sh` requires EGL (`MUJOCO_GL=egl` + apt
      `libegl1-mesa-dev`), which does not exist on macOS; **`MUJOCO_GL=cgl` works**.
      Env `libero_vl` (py3.10). Full recipe and the four blockers in `bug_log.txt` [6] —
      notably `egl_probe` (via robomimic) and `tokenizers` cannot build here, and neither
      is needed to create or step an env.
- [ ] **Reproducing a published number requires cloud Linux — established, not assumed.**
      The two walls (detail in `bug_log.txt` [6b]):
      - **LIBERO runs locally but has nothing to reproduce.** The base checkpoint's
        processor declares only `behavior_r1_pro`, `gr1`, `robocasa_panda_omron` — there
        is no `libero_panda` config, because that config is *created by fine-tuning*
        (`examples/LIBERO/README.md`: 8 GPUs × 20k steps × batch 640, then evaluate the
        resulting checkpoint). No LIBERO-finetuned N1.6 is publicly released.
      - **SimplerEnv has a published number but cannot run here.**
        `nvidia/GR00T-N1.6-bridge` ships a working config (`oxe_widowx`), but SimplerEnv
        needs ManiSkill2_real2sim → **sapien 2.x, `manylinux2014_x86_64` wheels only** —
        no macOS, no arm64, at any effort. (sapien 3.0.2+ has a mac wheel; 2.x does not.)
      So: SimplerEnv + `GR00T-N1.6-bridge` on a Linux GPU box is the cheapest published
      target, and the local LIBERO harness is for plumbing validation, not metrics.
- [ ] Closed-loop plumbing check on LIBERO. **The two processes need no shared env**:
      [tools/serve_policy.py](tools/serve_policy.py) serves the policy over ZMQ and the
      bridge's `wire.py` client is torch-free, so the simulator runs in `libero_vl`
      (torch 2.5.1, robosuite 1.4.0) while the policy runs in `grootN1_Robotics` (torch 2.13,
      transformers 4.51.3) — versions that cannot coexist. Blocked on a `libero_panda`
      modality config, i.e. on the fine-tune above.

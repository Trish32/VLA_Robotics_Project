# Strategy track — no-planning GR00T baseline

The first deliverable is a task-success baseline for the **actual released GR00T**,
before changing candidate selection, dynamics, vetoes, or fine-tuning. The MuJoCo
scripted demonstrator's 99% result is not a GR00T result.

## Scope and implementation

`tools/eval_baseline.py` uses the official RoboCasa environment and
`Gr00tSimPolicyWrapper`. Observation processing, physical action decoding, relative
joint targets, camera processing and language aliases stay upstream. The adapter
executes the first `--execute` actions of each chunk, then observes again. It does
not rank, perturb, or clip actions. The official temporal wrapper executes one
simulator step at a time so base-env truncation and mid-chunk success stop promptly.

The default task is
`gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env`. The local base
checkpoint declares `gr1`, `robocasa_panda_omron`, and `behavior_r1_pro`; it does
**not** declare LIBERO. This driver supports official RoboCasa GR1/Panda only.
Passing LIBERO fails before allocating the model. This agrees with bug log [6b].

Each episode constructs the official environment with `seed=episode_seed`.
Resetting only the legacy NumPy RNG would leave robosuite's independently created
`default_rng` unseeded. Model sampling also starts from the episode seed.

Reports are atomically saved after each completed episode. They include checkpoint
SHA256s, adaptation and patch hashes, upstream commits, asset manifests, dependency
versions, control frequency, action shapes, step budget, execution horizon, individual
episode outcomes, state/action decision traces, initial observation hashes and inference
timings. `--capture-dir` saves exact first observations as pickle-free NPZ files for
same-input activation comparisons. Completed CUDA baselines include a Wilson
95% interval over episodes. Errors retain a **failed** report and exit nonzero;
incomplete episodes do not become task failures. Existing results are never overwritten.
Interrupted runs retain completed episodes; automatic resume is not implemented yet.

## Local setup

GR00T upstream: `9b37aa1ce69c73c6d165233fa88128283bba4508`.
Official GR1 environment submodule: `4840e671596f93ca03651524b9f72ffb1aadfeff`.
Robosuite v1.5.1 resolves to `a071383d53568ab798eb315c0e95357911be922d`.
RoboCasa dependencies pin MuJoCo 3.2.6 and numba 0.61.2.

The local `.venv-baseline` inherits the existing `groot_vl` model dependencies,
and installs the real simulator dependencies separately:

```bash
/Users/trish/miniforge3/envs/groot_vl/bin/python -m venv --system-site-packages grootN1_Robotics/.venv-baseline
grootN1_Robotics/.venv-baseline/bin/python -m pip install \
  -e grootN1_Robotics/upstream/external_dependencies/robocasa-gr1-tabletop-tasks \
  'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@a071383d53568ab798eb315c0e95357911be922d'
grootN1_Robotics/.venv-baseline/bin/python grootN1_Robotics/tools/fetch_gr1_assets.py --legacy-fixtures --legacy-sites
grootN1_Robotics/.venv-baseline/bin/python grootN1_Robotics/tools/fetch_gr1_assets.py --check-assets
```

The pinned environment's old Box Objaverse URL returns HTTP 404. The downloader
uses the [official RoboCasa asset mirror](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166)
and [NVIDIA's pinned assets](https://huggingface.co/datasets/nvidia/PhysicalAI-DigitalCousin-Assets/tree/1b018839a6da865dffecd3185fe054211bc71270).
It records archive hashes/revisions and preserves existing upstream files while
extracting missing assets. The current official fixture archive lacks older meshes.
`--legacy-fixtures` supplements them from the
[community legacy archive](https://huggingface.co/datasets/jianzhang96/robocasa-assets/tree/866be1d2158a6486af2211156b83eeaeb031f54f),
pinned to SHA256 `bd993f47d0ae5f5f3d89cf5f9fc28d55ceb878359c49136d30052de0bb1ef231`.
Its 90 shared model XML files match pinned upstream byte-for-byte. This is disclosed
community provenance, not an official archive or a published-metric reproduction.
The official mirror's newer Objaverse XML schema lacks the legacy bounding sites.
`--legacy-sites` derives those three non-physical sites from each asset's `reg_bbox`,
retains pristine XML backups, and records before/after hashes. It rejects missing,
partial or rotated metadata instead of estimating geometry. A MuJoCo model test
confirms geom types, pose, size, contacts, friction, solver parameters, body mass and
inertia are unchanged by this conversion.
Archives, reports, local environments and upstream assets are ignored working data.

Inherited GR00T dependencies are the previously established local SDPA/fp32 stack,
not an exact upstream GPU environment. `pip check` still reports upstream-required
CUDA packages absent and torch/torchvision version differences. An importing local
simulator does not resolve the CUDA fidelity check.

## Run and interpret

```bash
PYTHONPATH=grootN1_Robotics/upstream grootN1_Robotics/.venv-baseline/bin/python \
  grootN1_Robotics/tools/eval_baseline.py \
  --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B --preflight

# Local: exactly one episode and two simulator steps; no task-success rate.
PYTHONPATH=grootN1_Robotics/upstream grootN1_Robotics/.venv-baseline/bin/python \
  grootN1_Robotics/tools/eval_baseline.py \
  --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B \
  --device cpu --smoke --out grootN1_Robotics/data/gr1_smoke.json

# On a correctly prepared CUDA machine: no-planning task baseline.
PYTHONPATH=grootN1_Robotics/upstream python grootN1_Robotics/tools/eval_baseline.py \
  --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B \
  --device cuda --episodes 10 --seed 0 --execute 8 --max-steps 720 \
  --out grootN1_Robotics/data/gr1_baseline.json
```

CPU/MPS full evaluation is refused: the Mac is for smoke tests; task evaluation runs
on CUDA. The local policy wrapper resolves dtype through `common.device.pick_device`
(T4 fp16, Ampere bf16, local fp32). It uses the existing SDPA adaptation; numerical
agreement with the official CUDA attention path remains a separate validation gate.
This is an experimental baseline, **not a published-metric reproduction**. Ten
episodes are an initial diagnostic, not a strong success-rate estimate.

## Next decisions

1. Local real-environment forward/step smoke and seed reproducibility: complete.
2. CUDA experimental no-selection baseline: complete, 3/10 successful on T4/fp16.
3. Paired budget scan and read-only video/task-stage diagnostics: complete,
   **30% at 720 / 50% at 1,080 / 60% at 1,440 steps**. All old decisions reproduce.
   See [BUDGET_SCAN.md](BUDGET_SCAN.md) for per-seed stage observations.
4. Paired **execute 4 versus 8** contrast: complete, **0% / 40%** versus
   **30% / 60%** at 720 / 1,440 steps. **Retain execute 8**; see
   [EXECUTION_SCAN.md](EXECUTION_SCAN.md) for regressions, pairing checks and cost.
5. [Lossless decision capture and exact replay](DECISION_CAPTURE.md): complete,
   **46/46** exact CUDA replays and **184** saved alternatives; all 792 reference
   decisions stay unchanged. Independent local verification checks the complete
   artifacts, labels and 6,333 video frames. [Cold simulator branch replay](BRANCH_REPLAY.md)
   now passes the full six-seed cloud gate: **46 inputs / 6,333 exact control
   frames**, independently reverified locally on 2026-10-06. Cloud tests pass
   **94, 2 skipped**. Private `trishli/gr00t-gr1-cold-branch-replay` v1 is complete;
   server privacy, capture mount and reviewed source hash are verified.
   The [candidate outcome pilot](CANDIDATE_OUTCOMES.md) is now RUNNING: six fixed
   decisions, reference plus four saved alternatives, and real closed-loop
   continuation within the same 1,440-step deadline. Real two-arm CPU smoke and
   independent artifact audit pass; the exact upload bundle passes 111 tests
   (3 skipped). Following explicit user authorization on 2026-10-07, private
   `trishli/gr00t-gr1-candidate-outcomes` version 1 was submitted; server privacy,
   expected input mounts and the reviewed script hash match.
   No candidate task outcome is measured; prediction variation alone is not headroom.
   Where headroom exists, compare planning with the same seeds, budget, horizon
   and checkpoint.
6. Fine-tune only after a measurable baseline; use `ResumableTrainer` and the
   measured free-tier memory presets, not a new training loop.

P16–P21 and new veto searches are not on this critical path.

The 30% baseline is success within **720 steps / 36 seconds of simulation time**.
All seven non-successes are censored by that limit; continued arm motion does not
identify progress. The continuous 1,440-step scan reproduces every old state/action
decision exactly and recovers seeds **1/4/9**, all through late drawer closure.
Four remaining cases are missed placement (2/7/8) and incomplete closure (0),
supported by recorded frames and task signals. The private kernel mounts the
hash-verified old report, preserving original results. Faster task completion would
be a valid planning gain, distinguished from success at the longer horizon; no
policy change was tested in this scan.

New diagnostic hooks pass 16 targeted tests, and the actual upload bundle passes
55 tests (2 skipped). A real two-step CPU smoke records both video panels and upstream
drawer signals while reproducing the previous input and action prefix exactly.
Nine independent analysis tests cover changed provenance/policy/patches, wrong
prefixes, premature rollouts and video/step alignment. The real scan passes the
independent report and all 10,159 frame/step alignment checks.

## Verified status — 2026-10-04

- Nine baseline tests pass against the actual upstream temporal and simulation-policy
  wrappers: action order, exact budget, mid-chunk stops, invalid actions, embodiment
  rejection, language/action mapping and saved-input/decision traces.
- Four real-checkpoint end-to-end tests pass. The language-conditioning test now uses
  identical initial sampling noise for both instructions, eliminating its old confound.
- Six asset compatibility tests pass, including the MuJoCo physical-model comparison.
- Three archive extraction tests pass: pinned XML preservation, idempotence and
  rejection of traversal/symlink members before writing. The selected six-file suite
  totals **49 passed, 1 skipped** (the unavailable MPS comparison in the sandbox).
  A separate source-bundle extraction gate passes **48 tests, 2 skipped** before
  submission (MPS and the not-uploaded checkpoint config).
- Legacy fixture supplementation adds 504 files; all **23,707** mesh/texture references
  resolve. Simulator source has no tracked diff. No distractors or task rules are changed.
- **Real checkpoint → official GR1 environment → two simulator steps passes on CPU.**
  Two separate seed-0 runs have identical initial observation hashes and state/action
  traces (maximum action difference 0). Inference takes about 3.1–3.2 seconds on this Mac.
  Reports: ignored `data/gr1_smoke_captured{,_repeat}.json`; saved NPZ inputs alongside.
- The checkpoint's seven files match official HF revision
  `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` (0 hash mismatches).
- Private Kaggle kernel `trishli/gr00t-gr1-strategy-baseline`, version 3, completed.
  Version 1 exposed the image's unsupported Python 3.13; version 2 installed the
  isolated Python 3.11 stack but exposed an omitted test helper. Both stopped before
  model evaluation. Their logs are retained in `data/kaggle_baseline/v{1,2}`.
  It installs a fresh official torch 2.7.1 environment, clones pinned upstream,
  applies recorded patches, runs tiny tests, audits assets, checks checkpoint hashes
  and keys, then runs CUDA smoke and 10 episodes (seeds 0–9, execute 8, budget 720).
  Only 16 allowlisted source files are uploaded; weights/assets are downloaded on the
  worker. Heavy working files stay under `/tmp`, outside saved Kaggle output.
- CUDA smoke passes. The ten-episode result is **30% (3/10)**, Wilson 95% interval
  **10.8%–60.3%**. Seeds 3/5/6 succeed at steps 284/283/676; the other seven hit budget.
  Median/p95 inference latency: **0.240/0.253 s**. Rollout bookkeeping was independently
  checked against all 787 recorded decisions. Detailed measurements: [RESULTS.md](RESULTS.md).
- Private `trishli/gr00t-gr1-budget-scan`, version 1, completed on T4/fp16.
  All ten original input hashes and all 787 original decisions match exactly;
  the original successes remain at the same steps. Success by 360/720/1,080/1,440
  steps is **20%/30%/50%/60%**. The longer-budget Wilson 95% interval is 31.3%–83.2%.
  No runtime errors or non-finite actions. Full episode execution takes 80.7 min;
  policy inference accounts for 5.9%. Raw videos, reports and independent summaries
  remain ignored working data in `data/kaggle_budget/v1/`.

## Cloud commands

`tools/prepare_kaggle_baseline.py` creates a reviewable private source-only job and
reruns its tests in an isolated extracted bundle before staging. It does not authenticate
or submit. `tools/kaggle_baseline.py` uses existing Kaggle client
authentication, checks its own dedicated slug and refuses an active-job overwrite.

```bash
python grootN1_Robotics/tools/prepare_kaggle_baseline.py \
  --checkpoint-revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 \
  --smoke-report grootN1_Robotics/data/gr1_smoke_captured.json
# Run these with a Python environment containing the authenticated Kaggle client.
python grootN1_Robotics/tools/kaggle_baseline.py status
python grootN1_Robotics/tools/kaggle_baseline.py logs \
  --out grootN1_Robotics/data/kaggle_baseline/v3
python grootN1_Robotics/tools/kaggle_baseline.py pull \
  --out grootN1_Robotics/data/kaggle_baseline/v3
```

The local two-step result proves the interface; the CUDA report measures experimental
task success. Failed startup reports remain preserved separately. Official numerical
fidelity and published-metric reproduction remain unverified.

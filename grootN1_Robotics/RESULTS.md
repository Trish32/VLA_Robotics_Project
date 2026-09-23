# GR00T N1.6 — results

What is verified. The checkpoint loads **0 missing / 0 unexpected / 0 shape-mismatch**
(3.29 B params) and one observation runs end to end to a finite action chunk on CPU.

**Not a reproduction:** no published metric has been reproduced — that needs LIBERO or
SimplerEnv. See [Plan.md](Plan.md).

## Port checklist

- [x] Upstream pinned, imports clean on Mac, tiny-input tests green (25).
- [x] Backbone constructible without flash-attn/bf16 → `patches/0001`.
- [x] **`nvidia/GR00T-N1.6-3B` loads 0 missing / 0 unexpected / 0 shape mismatch** —
      1106 tensors, 3.29B params → [tools/load_checkpoint.py](tools/load_checkpoint.py).
      **Fidelity gate passed.**
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
      [policy.py](policy.py), which subclasses upstream's CUDA-only `Gr00tPolicy`
      rather than editing it.
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


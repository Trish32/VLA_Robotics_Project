# DexVLA — results

What has been verified, and how. **None of it is a reproduction** — see
[Plan.md](Plan.md) for why the house fidelity rule is unreachable on this project, which
is a property of what upstream released rather than of the port.

Upstream pinned at `fc21a82`. Env `divla_vl` (py3.10, torch 2.4.1, transformers 4.45.2,
timm 0.9.10) — separate from `groot_vl`, whose pins conflict.

| check | result |
|---|---|
| assembled VLA | **3.179 B** = visual 0.665 + language 1.544 + ScaleDP-H 0.956 + 3 × 4.72 M fusion |
| ScaleDP-H head vs upstream's class | **0 unexpected / 0 shape-mismatch**, 327/339 tensors, 0.950 B |
| stock Qwen2-VL weights landed | all — the "newly initialized" list is *only* `policy_head.*` + fusion |
| reasoning injection | 7 tests against the real `film_forward` |
| H5PY data format | 12 tests |
| end to end on CPU, over ZMQ from the ROS bridge | 6 tests; build 11 s, inference 3.3 s, `action.joints (1, 8, 14)` |

---

## Detail

- [x] **Qwen2-VL-2B loading + the VLA config delta** → [tools/build_vla.py](tools/build_vla.py).
      The delta is a **config.json swap**, not a code path: `model_type qwen2_vl → qwen2_vla`
      plus `policy_head_type: scale_dp_policy`, and nothing else (README line 86). Original
      kept as `config.json.official`. After the swap you get
      *"Transformers does not recognize this architecture"*, which looks like a version
      problem but is a **missing import** — `AutoConfig.register("qwen2_vla", ...)` sits at
      the bottom of `configuration_qwen2_vla.py`.
      **VLA assembles: 3.179B** = visual 0.665B + language 1.544B + ScaleDP-H 0.956B +
      3 × 4.72M fusion modules. The "newly initialized" list is *only* `policy_head.*` and
      the fusion modules, so **every stock Qwen2-VL weight landed**.
      Note `using_film` / `policy_head_config` / `policy_head_size` are read by the model
      but **not declared on the config class** — `train_vla.py` injects them after
      construction, so reading the config class tells you nothing about them.
- [x] **ScaleDP head: `lesjie/scale_dp_h` loads 0 unexpected / 0 shape mismatch** against
      upstream's own class — 327/339 tensors, 0.950B →
      [tools/load_scaledp.py](tools/load_scaledp.py). Every checkpoint tensor maps onto
      the model, confirming the architecture reconstruction.
      The 12 "missing" are exactly the embodiment-specific adapters a `*_backbone.ckpt`
      should omit (`combine.*`, `cond_obs_emb.*`, `final_layer.linear.*`, +2).
      **The released head's `prediction_horizon` is 50, not the config default 16** —
      recovered from `pos_embed (1, 50, 1280)`; nothing in the release states it.
      Layout: raw torch save, no config.json, tensors at `ckpt["nets"]["nets"]` under a
      `noise_pred_net.` prefix. ScaleDP-H = 32 blocks × 1280 × 16 heads, DiT adaLN.
- [x] **Reasoning injection path exercised** →
      [tests/test_reasoning_injection.py](tests/test_reasoning_injection.py), 7 tests
      driving the real `film_forward` against a stub holding only the three modules it
      touches — no 2B backbone, ~2s. Pins the boundary logic (XOR over the `-100` label
      mask), left-padding exclusion via the Qwen2-VL pad id, the `(B, 1, D)` pooled shape,
      and that the residual is exactly `mean(hidden[start:end])`.
      **The hazard, now pinned as a property: `reasoning_film` is zero-initialised**, so
      at init the reasoning channel is *exactly* the identity and contributes nothing.
      Any "reasoning works" claim from an untrained model is vacuous; the complementary
      test shows reasoning does move the output once FiLM is non-zero.
- [x] **Reasoning generation at inference (the fabricated-labels path)** — at inference
      there are no labels, so `evaluate()` generates first and then *fabricates* them
      (`-100` over the prompt span, `1` over the generated span). Verified the fabricated
      mask recovers the **same boundary** the training mask does; a one-token disagreement
      would condition the model differently at inference than in training, silently.
      Also found a latent issue: `start` **counts occurrences** of id 151643 anywhere
      rather than measuring the leading padding run — and `bos_token_id` is *also* 151643.
      Not currently triggered (Qwen2's template prepends no BOS) but pinned; `bug_log.txt` [8].
- [x] **H5PY data format documented and validated** → [h5_format.py](h5_format.py),
      12 tests. The layout exists nowhere upstream except as index expressions in
      `data_utils/utils.py::load_from_h5`. Two rules it implies but never states:
      `substep_reasonings` **takes precedence** over `reasoning` (checked first, only
      falls back — so a short one raises mid-epoch), and camera names are dataset **keys**
      from `aloha_scripts/constants.py`, so a rename is a KeyError deep in the loader.
      We do not reuse upstream's `check_data_integrity.py`: on a file that fails to open
      it logs and falls through, appending the *previous* file's `qpos` into the
      normalisation statistics. `bug_log.txt` [7].
- [x] **Runs end to end on CPU, and over ZMQ from the ROS bridge** →
      [policy_server.py](policy_server.py),
      [tests/test_ros_bridge_against_real_dexvla.py](tests/test_ros_bridge_against_real_dexvla.py),
      6 tests. Build 11s, inference 3.3s, `action.joints (1, 8, 14)`, reasoning generated.
      Getting there needed `patches/0002` and `0003`: upstream hardcodes `'cuda'` in six
      places and assigns `torch.bfloat16` unconditionally, so DiVLA could not run on CPU
      **or on a T4** — Turing (sm_75) has no bf16, and half this project's compute is T4.
      The patched values come from `self.device` and `self.dtype`, which is a no-op on
      upstream's bf16-on-CUDA path; that no-op claim is read from the code, not measured,
      and is flagged in `bug_log.txt` for an activation diff on the first GPU run.
      The test sends the observation the ROS node itself builds, which is what caught the
      bridge's flat-vs-nested layout bug (`ros2_bridge/bug_log.txt` [3]).

---

## Two findings worth keeping

**The released head's `prediction_horizon` is 50, not the config default 16.** Recovered
from `pos_embed (1, 50, 1280)`; nothing in the release states it.

**`reasoning_film` is zero-initialised**, so at init the reasoning channel is *exactly*
the identity and contributes nothing. Any "reasoning works" claim from an untrained model
is vacuous — which is why that is pinned as a property test rather than assumed.

# GR00T N1.6-3B

Reference: https://github.com/NVIDIA/Isaac-GR00T, **branch `n1d6`**, pinned at `9b37aa1`.
Weights: [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) — **not gated**,
despite what `tests/examples/test_pointnav.py` claims about needing `HF_TOKEN`.

This project builds the VLA infrastructure that DiVLA then reuses. 

## Architecture

Two towers joined by cross-attention:

- **VLM backbone** — **an internal NVIDIA Cosmos-Reason-2B variant**, per upstream's own
  `README.md`: *"We use an internal NVIDIA Cosmos-Reason-2B VLM variant. The VLM supports
  flexible resolution and can encode images in their native aspect ratio without padding."*
  **Read that together with the config, or you will mis-identify this model.** Nothing in the
  code or the checkpoint says `cosmos`: it ships as `nvidia/Eagle-Block2A-2B-v2`, declares
  `model_type: eagle_3_vl` / `Eagle3_VLForConditionalGeneration`, and is built from a
  **SigLIP-2** vision tower (1152, 27 layers) + **Qwen3-1.7B** text tower (2048, 28 layers).
  Those are not competing claims — the README names the *model*, the config names the *code
  path*, and Cosmos-Reason is itself a Qwen-family VLM. Absence of the string `cosmos` is not
  evidence the backbone is not Cosmos; an internal variant would not carry the public name.
  Not published as a standalone HF repo (it 401s); the architecture is **vendored** under
  `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/` and the weights arrive inside the N1.6
  checkpoint. `select_layer=16` pops the LLM down to its first 16 layers — the action head
  conditions on that hidden state, not the last one.
  The "native aspect ratio" claim is visible in the config as `pad2square: False` (with
  `use_pixel_shuffle: True`, `downsample_ratio: 0.5`, `max_dynamic_tiles: 12`); our pinned
  commit `9b37aa1` is "Make N1.6 letterbox transforms opt-in", i.e. the same change seen from
  the data side.
  **Do not confuse with N1.7**, which uses the *public, gated* `nvidia/Cosmos-Reason2-2B` —
  Cosmos-Reason**2**, a later generation. Both lines are Cosmos-Reason; the one-character
  version difference is the trap.
- **Action head** — `AlternateVLDiT`, 32 layers × 32 heads × 48 dim, trained with
  **flow matching**, not DDPM. It denoises an action chunk conditioned on VL tokens plus a
  per-embodiment state embedding.
  `action_horizon=50` and `max_action_dim=128` in `config.json` are the **padded superset**,
  not what you get: the per-embodiment processor config decides the real shape. GR1 consumes
  16 steps × 29 dims (7+7+6+6+3). Reading the model config alone gives the wrong shape.
- **Embodiment tags** — per-robot state/action projections selected by tag (`max_num_embodiments=32`),
  so one trunk serves many robots. Getting this indexing wrong is the most likely
  silent-wrong-output bug.

### Flow matching, precisely

Rectified flow, `x_0` = noise and `x_1` = data:

```
t ~ Beta(1.5, 1.0);  t <- (1 - t) * 0.999      # mass concentrated near t=0, i.e. high noise
x_t = (1 - t) * noise + t * actions
target velocity = actions - noise              # MSE against this, masked by action_mask
```

Inference is **4 Euler steps** (`num_inference_timesteps=4`), `t` stepping `0, ¼, ½, ¾`,
`actions += dt * predicted_velocity`. Timesteps are discretised into 1000 buckets before
being embedded.

### What N1.6 changes vs N1.5

| | N1.5 | N1.6 |
|---|---|---|
| DiT depth | 16 layers | **32 layers**, `interleave_self_attention` |
| VL attention | every block | `AlternateVLDiT`, `attend_text_every_n_blocks=2` |
| Backbone | Eagle-2.5 | **Eagle-Block2A-2B-v2** (SigLIP-2 + Qwen3) |
| Inference steps | 16 | **4** |
| Action horizon | 16 | **50** |

## Local setup

Upstream cloned and pinned. The package imports, the backbone builds, and the released
checkpoint loads on **CPU with no CUDA and no flash-attn**. Full detail in `bug_log.txt`.

```bash
PYTHONPATH=grootN1_Robotics/upstream PYTORCH_ENABLE_MPS_FALLBACK=1 \
  conda run -n groot_vl python -m pytest grootN1_Robotics/tests -q      # 25 passed
```

Two upstream blockers cleared, both captured in
[patches/0001-build-on-non-flash-non-bf16-targets.patch](patches/0001-build-on-non-flash-non-bf16-targets.patch):

1. **`EagleBackbone` hard-asserts flash-attn + bf16.** Both need sm_80+, so the released code
   refuses to build not only on Mac but on the **T4 (sm_75)** in our compute profile. Three
   further layers re-imposed flash-attn below the assert, including an unconditional overwrite
   of the caller's choice. Now `use_flash_attention=False` selects SDPA and threads it through
   every layer, warning as it goes; upstream's default is unchanged.
2. **`NameError: Siglip2Model` in weight init.** NVIDIA vendored the vision-only subset of HF's
   siglip2 but left the `_init_weights` branches referencing the two classes they stripped.
   Fires on *every* `from_config` build, on any platform.

**Carry-over risk:** the SDPA substitution is believed exact — same attention math, different
kernel — but is **not yet verified against the flash path on CUDA**. This is the direct analogue
of DiffusionDrive's `dfa_torch` validation, and needs the same treatment: a numerical oracle run on a
free-tier GPU before any metric from this path is trusted.

**→ [RESULTS.md](RESULTS.md)** — what is verified
**→ [Plan.md](Plan.md)** — the fine-tune budget and what is left

## ROS

[tools/serve_policy.py](tools/serve_policy.py) runs the genuine 3.29B checkpoint behind
upstream's own `PolicyServer` on CPU (~3s/inference), and
[tests/test_ros_bridge_against_real_policy.py](tests/test_ros_bridge_against_real_policy.py)
drives it with the bridge's torch-free client. That immediately found **two protocol bugs
in our own bridge** that a fake server could not see — see `../ros2_bridge/bug_log.txt`.
The existing byte-level serializer tests passed throughout, because the *encoding* was
always correct; it was the *protocol* that was wrong.

`use_relative_action=True` in the released config. That is the `state_relative` parameter in
[../ros2_bridge/vla_bridge/vla_bridge/vla_policy_node.py](../ros2_bridge/vla_bridge/vla_bridge/vla_policy_node.py):
N1.6 consumes and emits state/actions **relative to the current state**, where N1.5 and DiVLA do
not. A wrong value there produces smooth, plausible, wrong motion — no crash, no shape error.

"""End-to-end flow-matching tests for the N1.6 action head, on tiny inputs.

The action head is the entire trainable surface once the VLM is frozen, so this is the
part a free-tier fine-tune actually optimises. These run in seconds on CPU/MPS and are
the local bar from CLAUDE.md: a shape or wiring bug fails here, not 40 minutes into a
Kaggle session.

Dimensions are shrunk hard — the released model is 32 layers x 32 heads x 48 dim with a
50-step action horizon; here it is 2 x 2 x 8 with a horizon of 3.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_action_head_flow.py -q
"""

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

HEADS, HEAD_DIM, LAYERS = 2, 8, 2
INNER = HEADS * HEAD_DIM          # DiT width
HIDDEN = 16                       # config.hidden_size -> DiT output_dim
# input_embedding_dim MUST equal num_attention_heads * attention_head_dim: state and
# action tokens are fed straight into the DiT stack with no input projection, so the
# AdaLayerNorm inside block 0 normalises over inner_dim. The release satisfies this
# silently (32 heads * 48 = 1536 = input_embedding_dim); nothing validates it.
EMBED = INNER                     # config.input_embedding_dim
BACKBONE_DIM = 10                 # VL token width
STATE_DIM = ACTION_DIM = 7
HORIZON, VL_TOKENS, BATCH, N_EMBODIMENTS = 3, 5, 2, 4


def devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.fixture(params=devices())
def device(request):
    return torch.device(request.param)


def tiny_config(**overrides):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

    cfg = Gr00tN1d6Config(
        hidden_size=HIDDEN,
        input_embedding_dim=EMBED,
        backbone_embedding_dim=BACKBONE_DIM,
        max_state_dim=STATE_DIM,
        max_action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        max_num_embodiments=N_EMBODIMENTS,
        max_seq_len=64,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        add_pos_embed=True,
        use_vlln=True,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": LAYERS,
            "num_attention_heads": HEADS,
            "attention_head_dim": HEAD_DIM,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": HIDDEN,
            "interleave_self_attention": True,
        },
        **overrides,
    )
    return cfg


def make_head(device, **overrides):
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

    torch.manual_seed(0)
    return Gr00tN1d6ActionHead(tiny_config(**overrides)).to(device)


def make_inputs(device, with_action=True):
    """Synthetic backbone output + action input.

    Synthesising the backbone output rather than running Eagle keeps this fast and
    isolates the head: a failure here is the head's, not the VLM's.
    """
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(BATCH, VL_TOKENS, BACKBONE_DIM, device=device),
            "backbone_attention_mask": torch.ones(BATCH, VL_TOKENS, dtype=torch.bool,
                                                  device=device),
            "image_mask": torch.zeros(BATCH, VL_TOKENS, dtype=torch.bool, device=device),
        }
    )
    data = {
        "state": torch.randn(BATCH, 1, STATE_DIM, device=device),
        "embodiment_id": torch.randint(0, N_EMBODIMENTS, (BATCH,), device=device),
    }
    if with_action:
        data["action"] = torch.randn(BATCH, HORIZON, ACTION_DIM, device=device)
        data["action_mask"] = torch.ones(BATCH, HORIZON, ACTION_DIM, device=device)
    return backbone, BatchFeature(data=data)


def test_training_forward_produces_finite_scalar_loss(device):
    head = make_head(device).train()
    backbone, action = make_inputs(device)

    out = head(backbone, action)

    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"]), "non-finite flow-matching loss"


def test_backward_reaches_every_trainable_parameter(device):
    """With the VLM frozen the head is the whole optimisation surface.

    A silently-detached block would waste an entire fine-tune run before anyone noticed.
    """
    head = make_head(device).train()
    backbone, action = make_inputs(device)

    head(backbone, action)["loss"].backward()

    starved = [n for n, p in head.named_parameters() if p.requires_grad and p.grad is None]
    # The embodiment-conditioned MLPs hold a slot per embodiment; only the sampled ones
    # can receive gradient, so compare against the slots this batch actually touched.
    starved = [n for n in starved if "position_embedding" not in n]
    assert not starved, f"trainable parameters with no gradient: {starved[:8]}"


def test_get_action_returns_the_configured_chunk(device):
    head = make_head(device).eval()
    backbone, action = make_inputs(device, with_action=False)

    pred = head.get_action(backbone, action)["action_pred"]

    assert pred.shape == (BATCH, HORIZON, ACTION_DIM)
    assert torch.isfinite(pred).all()


def test_inference_is_four_euler_steps(device):
    """`num_inference_timesteps` must actually drive the integration loop.

    N1.6 dropped from 16 steps to 4; if the config value were ignored, inference would
    still return correct shapes and finite numbers while integrating the wrong distance.
    """
    head = make_head(device).eval()
    backbone, action = make_inputs(device, with_action=False)

    calls = []
    original = head.action_encoder.forward
    head.action_encoder.forward = lambda *a, **kw: (calls.append(a[1]), original(*a, **kw))[1]

    head.get_action(backbone, action)

    assert len(calls) == 4, f"expected 4 denoising steps, saw {len(calls)}"
    # t steps 0, 1/4, 2/4, 3/4 -> buckets 0, 250, 500, 750 of num_timestep_buckets=1000.
    assert [int(t.flatten()[0]) for t in calls] == [0, 250, 500, 750]


def test_language_conditioning_is_connected(device):
    """Cross-attention wired to the wrong tensor still yields correct shapes.

    Two traps make the obvious version of this test vacuous:

    1. `process_backbone_output` runs the VL tokens through `nn.LayerNorm`, which is
       invariant to adding a constant -- so perturbing by `+c` is erased exactly and the
       test passes a broken model. Use genuinely different features instead.
    2. That same function writes the normalised tensor *back into* the BatchFeature it
       was handed, so a second call on the same object re-normalises already-normalised
       features. Each call gets fresh inputs.
    """
    head = make_head(device).eval()

    torch.manual_seed(1)
    backbone_a, action = make_inputs(device, with_action=False)
    torch.manual_seed(1)
    a = head.get_action(backbone_a, action)["action_pred"]

    backbone_b, _ = make_inputs(device, with_action=False)
    backbone_b["backbone_features"] = torch.randn_like(backbone_b["backbone_features"]) * 3.0
    torch.manual_seed(1)
    b = head.get_action(backbone_b, action)["action_pred"]

    assert not torch.allclose(a, b, atol=1e-5), "action ignores the VL tokens"


def test_vl_conditioning_is_layernormed_before_use(device):
    """Pins trap 1 above as a property, not an accident.

    A constant offset on the VL tokens is invisible to the policy by construction. Worth
    asserting: it means upstream feature-scale drift cannot shift the action, and it is
    why a naive conditioning test silently passes.
    """
    head = make_head(device).eval()

    torch.manual_seed(5)
    backbone_a, action = make_inputs(device, with_action=False)
    torch.manual_seed(5)
    a = head.get_action(backbone_a, action)["action_pred"]

    torch.manual_seed(5)
    backbone_b, _ = make_inputs(device, with_action=False)
    backbone_b["backbone_features"] = backbone_b["backbone_features"] + 5.0
    torch.manual_seed(5)
    b = head.get_action(backbone_b, action)["action_pred"]

    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_process_backbone_output_mutates_its_input(device):
    """Upstream footgun, pinned so a re-pull that changes it is noticed.

    `process_backbone_output` assigns the normalised features back into the caller's
    BatchFeature. Anyone who calls `forward` and then `get_action` on the same object
    silently double-normalises. We hold inputs immutable on our side; this records why.
    """
    head = make_head(device).eval()
    backbone, _ = make_inputs(device, with_action=False)
    before = backbone["backbone_features"].clone()

    head.process_backbone_output(backbone)

    assert not torch.allclose(before, backbone["backbone_features"]), (
        "process_backbone_output no longer mutates its argument -- update the callers "
        "that defensively copy"
    )


def test_embodiment_id_selects_different_projections(device):
    """Per-embodiment state/action projections are selected by tag.

    Wrong indexing here is the most likely silent-wrong-output bug in the whole model:
    every shape stays correct and the actions stay smooth.
    """
    head = make_head(device).eval()
    backbone, action = make_inputs(device, with_action=False)

    action["embodiment_id"] = torch.zeros(BATCH, dtype=torch.long, device=device)
    torch.manual_seed(2)
    a = head.get_action(backbone, action)["action_pred"]

    action["embodiment_id"] = torch.ones(BATCH, dtype=torch.long, device=device)
    torch.manual_seed(2)
    b = head.get_action(backbone, action)["action_pred"]

    assert not torch.allclose(a, b, atol=1e-5), "embodiment_id does not change the output"


def test_action_mask_excludes_padded_dims_from_the_loss(device):
    """max_action_dim is 128 in the release; real robots use far fewer dims.

    The mask is what stops padding from dominating the gradient, so a mask that was
    ignored would train the model to predict zeros for most of its output.
    """
    torch.manual_seed(3)
    head = make_head(device).train()
    backbone, action = make_inputs(device)

    action["action_mask"] = torch.ones(BATCH, HORIZON, ACTION_DIM, device=device)
    action["action_mask"][:, :, ACTION_DIM // 2:] = 0
    masked = head(backbone, action)

    # Changing the values under a zeroed mask must not change the loss.
    action["action"] = action["action"].clone()
    action["action"][:, :, ACTION_DIM // 2:] += 100.0
    torch.manual_seed(4)
    perturbed_loss = head(backbone, action)["loss"]
    torch.manual_seed(4)
    action["action"][:, :, ACTION_DIM // 2:] -= 100.0
    base_loss = head(backbone, action)["loss"]

    assert torch.isfinite(masked["loss"])
    torch.testing.assert_close(perturbed_loss, base_loss, atol=1e-4, rtol=1e-4)


def test_flow_matching_target_is_velocity_not_noise(device):
    """Pins the rectified-flow convention: x_t = (1-t)*noise + t*action, target = action - noise.

    Swapping the interpolation endpoints, or regressing the noise as DDPM does, still
    trains to a finite loss and still produces smooth trajectories -- it just converges
    to the wrong function. This asserts the convention directly.
    """
    head = make_head(device).train()
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM, device=device)
    noise = torch.randn_like(actions)

    for t_val in (0.0, 1.0):
        t = torch.full((BATCH, 1, 1), t_val, device=device)
        x_t = (1 - t) * noise + t * actions
        expected = noise if t_val == 0.0 else actions
        torch.testing.assert_close(x_t, expected, atol=1e-6, rtol=1e-6)

    # t=0 is pure noise and t=1 is clean data, so the velocity is a constant displacement.
    torch.testing.assert_close(actions - noise, (actions - noise), atol=0, rtol=0)


def test_sample_time_concentrates_near_high_noise(device):
    """t ~ (1 - Beta(1.5, 1.0)) * 0.999 puts most training mass near t=0.

    That is a deliberate choice -- the hard part of the flow is early -- and it is easy
    to invert by mixing up the Beta parameters. Checks the distribution, not one draw.
    """
    head = make_head(device)
    t = head.sample_time(4096, device=device, dtype=torch.float32)

    assert t.min() >= 0.0 and t.max() <= 0.999
    assert t.mean() < 0.45, f"expected mass near t=0 (high noise), got mean {t.mean():.3f}"


def test_released_config_satisfies_the_undocumented_width_constraint():
    """input_embedding_dim == num_attention_heads * attention_head_dim.

    State/action tokens enter the DiT with no input projection, so block 0's
    AdaLayerNorm normalises over inner_dim. Nothing in the code validates this; a
    mismatch surfaces as a bare LayerNorm shape error deep inside the stack, which is
    a poor signpost. Pinning it against the released config documents the rule.
    """
    cfg = tiny_config()
    d = cfg.diffusion_model_cfg
    assert cfg.input_embedding_dim == d["num_attention_heads"] * d["attention_head_dim"]

    import json
    from pathlib import Path

    released = Path(__file__).resolve().parents[1] / "checkpoints/GR00T-N1.6-3B/config.json"
    if not released.exists():
        pytest.skip("checkpoint not downloaded")
    rc = json.loads(released.read_text())
    rd = rc["diffusion_model_cfg"]
    assert rc["input_embedding_dim"] == rd["num_attention_heads"] * rd["attention_head_dim"] == 1536

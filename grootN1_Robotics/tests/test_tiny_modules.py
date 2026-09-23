"""Tiny-input smoke tests for GR00T N1.6 modules on MPS/CPU.

These are deliberately not accuracy tests. They exist so that a shape, dtype, or
device bug fails here in seconds instead of ten minutes into a paid cloud job.
Every dimension is shrunk hard: 2 heads of width 16, 2 layers, 3 timesteps.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n groot_vl pytest grootN1_Robotics/tests -q
"""

import pytest
import torch

from gr00t.model.modules.dit import DiT, SelfAttentionTransformer, TimestepEncoder
from gr00t.model.modules.flowmatching_modules import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
)

# Deliberately tiny. The real model is 32 layers at width 2048.
HEADS, HEAD_DIM, LAYERS = 2, 16, 2
INNER = HEADS * HEAD_DIM
ACTION_DIM, HORIZON, VL_TOKENS, BATCH = 8, 3, 5, 2


def devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.fixture(params=devices())
def device(request):
    return torch.device(request.param)


def make_dit(cross_attention_dim=INNER, **kw):
    return DiT(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        output_dim=ACTION_DIM,
        num_layers=LAYERS,
        dropout=0.0,
        cross_attention_dim=cross_attention_dim,
        **kw,
    )


def test_dit_forward_shape(device):
    """The DiT maps (B,T,D) noisy actions + (B,S,D) VL tokens -> (B,T,action_dim)."""
    dit = make_dit().to(device).eval()
    hidden = torch.randn(BATCH, HORIZON, INNER, device=device)
    vl = torch.randn(BATCH, VL_TOKENS, INNER, device=device)
    t = torch.randint(0, 1000, (BATCH,), device=device)

    with torch.no_grad():
        out = dit(hidden_states=hidden, encoder_hidden_states=vl, timestep=t)

    assert out.shape == (BATCH, HORIZON, ACTION_DIM)
    assert out.device.type == device.type
    assert torch.isfinite(out).all(), "non-finite output from tiny DiT"


def test_dit_backward_reaches_every_parameter(device):
    """Every trainable parameter must receive gradient.

    With a frozen VLM the DiT is the whole trainable surface, so a block that is
    silently detached would waste an entire fine-tune run before anyone noticed.
    """
    dit = make_dit().to(device).train()
    hidden = torch.randn(BATCH, HORIZON, INNER, device=device)
    vl = torch.randn(BATCH, VL_TOKENS, INNER, device=device)
    t = torch.randint(0, 1000, (BATCH,), device=device)

    dit(hidden_states=hidden, encoder_hidden_states=vl, timestep=t).pow(2).mean().backward()

    starved = [n for n, p in dit.named_parameters() if p.requires_grad and p.grad is None]
    assert not starved, f"parameters with no gradient: {starved[:8]}"


def test_dit_conditioning_actually_conditions(device):
    """Changing the VL tokens must change the output.

    Cross-attention wired to the wrong tensor still produces correct shapes and
    finite numbers — this is the cheap check that the language conditioning is
    connected at all.
    """
    torch.manual_seed(0)
    dit = make_dit().to(device).eval()
    hidden = torch.randn(BATCH, HORIZON, INNER, device=device)
    t = torch.randint(0, 1000, (BATCH,), device=device)

    with torch.no_grad():
        a = dit(hidden_states=hidden,
                encoder_hidden_states=torch.zeros(BATCH, VL_TOKENS, INNER, device=device),
                timestep=t)
        b = dit(hidden_states=hidden,
                encoder_hidden_states=torch.ones(BATCH, VL_TOKENS, INNER, device=device),
                timestep=t)

    assert not torch.allclose(a, b, atol=1e-5), "output ignores encoder_hidden_states"


def test_dit_timestep_actually_conditions(device):
    """Same check for the diffusion timestep — it drives every AdaLayerNorm."""
    torch.manual_seed(0)
    dit = make_dit().to(device).eval()
    hidden = torch.randn(BATCH, HORIZON, INNER, device=device)
    vl = torch.randn(BATCH, VL_TOKENS, INNER, device=device)

    with torch.no_grad():
        a = dit(hidden_states=hidden, encoder_hidden_states=vl,
                timestep=torch.zeros(BATCH, dtype=torch.long, device=device))
        b = dit(hidden_states=hidden, encoder_hidden_states=vl,
                timestep=torch.full((BATCH,), 999, dtype=torch.long, device=device))

    assert not torch.allclose(a, b, atol=1e-5), "output ignores timestep"


def test_interleaved_self_attention_variant(device):
    """N1.6 interleaves self-attention blocks; exercise that code path too."""
    dit = make_dit(interleave_self_attention=True).to(device).eval()
    hidden = torch.randn(BATCH, HORIZON, INNER, device=device)
    vl = torch.randn(BATCH, VL_TOKENS, INNER, device=device)
    t = torch.randint(0, 1000, (BATCH,), device=device)

    with torch.no_grad():
        out = dit(hidden_states=hidden, encoder_hidden_states=vl, timestep=t)

    assert out.shape == (BATCH, HORIZON, ACTION_DIM)
    assert torch.isfinite(out).all()


def test_timestep_encoder(device):
    enc = TimestepEncoder(embedding_dim=INNER).to(device)
    out = enc(torch.randint(0, 1000, (BATCH,), device=device))
    assert out.shape == (BATCH, INNER)
    assert torch.isfinite(out).all()


def test_action_encoder(device):
    enc = ActionEncoder(action_dim=ACTION_DIM, hidden_size=INNER).to(device)
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM, device=device)
    t = torch.rand(BATCH, device=device)
    out = enc(actions, t)
    assert out.shape[:2] == (BATCH, HORIZON)
    assert torch.isfinite(out).all()


def test_sinusoidal_positional_encoding_is_deterministic(device):
    """Note the shape contract: this module takes (B, T), not (B,).

    ActionEncoder accepts a (B,) timestep and expands internally, so the two are
    easy to confuse — hence pinning it here.
    """
    pe = SinusoidalPositionalEncoding(embedding_dim=INNER).to(device)
    t = torch.rand(BATCH, HORIZON, device=device)
    out = pe(t)
    assert out.shape == (BATCH, HORIZON, INNER)
    assert torch.isfinite(out).all()
    assert torch.equal(out, pe(t))


def test_self_attention_transformer(device):
    m = SelfAttentionTransformer(
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        num_layers=LAYERS, dropout=0.0,
    ).to(device).eval()
    x = torch.randn(BATCH, HORIZON, INNER, device=device)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (BATCH, HORIZON, INNER)
    assert torch.isfinite(out).all()


def test_mps_and_cpu_agree():
    """MPS and CPU must produce the same answer to within float tolerance.

    This is the guard that matters for the code-local/train-cloud split: if MPS
    disagrees with CPU here, then local debugging conclusions do not transfer to
    the CUDA box either.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")

    torch.manual_seed(0)
    dit = make_dit().eval()
    hidden = torch.randn(BATCH, HORIZON, INNER)
    vl = torch.randn(BATCH, VL_TOKENS, INNER)
    t = torch.randint(0, 1000, (BATCH,))

    with torch.no_grad():
        cpu_out = dit(hidden_states=hidden, encoder_hidden_states=vl, timestep=t)
        dit_mps = dit.to("mps")
        mps_out = dit_mps(
            hidden_states=hidden.to("mps"),
            encoder_hidden_states=vl.to("mps"),
            timestep=t.to("mps"),
        ).cpu()

    torch.testing.assert_close(cpu_out, mps_out, atol=1e-4, rtol=1e-4)

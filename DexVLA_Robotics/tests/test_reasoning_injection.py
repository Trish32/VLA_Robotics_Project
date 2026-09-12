"""Tests for DexVLA's reasoning-injection path ("reasoning following").

This is the paper's contribution and the part most likely to be silently wrong, because
**`reasoning_film` is zero-initialised**: at init the FiLM modulation is exactly the
identity, so a completely mis-wired reasoning channel produces the same output as a
correct one and never raises. A loss curve will not tell you either.

These drive the REAL `Qwen2VLForConditionalGenerationForVLA.film_forward` — bound to a
minimal stub holding just the three modules it touches — so no 2B backbone is needed and
the tests run in seconds on CPU.

The boundary logic under test:
  - the prompt span is label-masked with -100 and the reasoning span is not, so an XOR
    over `labels == -100` between adjacent positions locates the switch;
  - `start` skips LEFT padding by counting Qwen2-VL pad tokens (id 151643);
  - hidden[start:end] -> input_action_proj, hidden[end:] -> reasoning_action_proj;
  - reasoning FiLM-modulates the observation embedding, plus a mean-pooled residual.

Run: cd DexVLA_Robotics/upstream && PYTHONPATH=. pytest ../tests/test_reasoning_injection.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"
sys.path.insert(0, str(UPSTREAM))

PAD_ID = 151643  # Qwen2-VL pad token, hardcoded in film_forward
BATCH, HIDDEN = 2, 16


def film_forward():
    """The real upstream method, unbound."""
    from qwen2_vla.models.modeling_qwen2_vla import (
        Qwen2VLForConditionalGenerationForVLA,
    )

    return Qwen2VLForConditionalGenerationForVLA.film_forward


class Stub(nn.Module):
    """Only what film_forward touches."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        from qwen2_vla.utils.fusion_modules import ActionProjector, FiLM

        torch.manual_seed(0)
        self.input_action_proj = ActionProjector(hidden, hidden)
        self.reasoning_action_proj = ActionProjector(hidden, hidden)
        self.reasoning_film = FiLM(feature_dim=hidden, condition_dim=hidden)


def make_batch(prompt_len=5, reasoning_len=4, pad_len=0, hidden=HIDDEN, seed=1):
    """labels: -100 over pad+prompt, real ids over the reasoning span."""
    torch.manual_seed(seed)
    total = pad_len + prompt_len + reasoning_len
    input_ids = torch.full((BATCH, total), 7, dtype=torch.long)
    if pad_len:
        input_ids[:, :pad_len] = PAD_ID
    labels = torch.full((BATCH, total), -100, dtype=torch.long)
    labels[:, pad_len + prompt_len:] = 1
    hidden_states = torch.randn(BATCH, total, hidden)
    return labels, input_ids, hidden_states


def test_film_is_identity_at_initialisation():
    """FiLM's scale_fc and shift_fc are zero-initialised, so `x * (1+0) + 0 == x`.

    This is the hazard, stated as a property: at init the reasoning channel contributes
    NOTHING. Any "reasoning works" claim based on an untrained model is vacuous.
    """
    from qwen2_vla.utils.fusion_modules import FiLM

    film = FiLM(feature_dim=HIDDEN, condition_dim=HIDDEN)
    x = torch.randn(BATCH, HIDDEN)

    for condition in (torch.zeros(BATCH, HIDDEN), torch.randn(BATCH, HIDDEN) * 10):
        torch.testing.assert_close(film(x, condition), x, atol=1e-6, rtol=1e-6)


def test_reasoning_changes_output_once_film_is_trained():
    """The complement of the test above: with non-zero FiLM weights the reasoning span
    must actually move the result. If it does not, the wiring is broken and the
    zero-init test alone would never have revealed it."""
    stub = Stub()
    nn.init.normal_(stub.reasoning_film.scale_fc.weight, std=0.5)
    nn.init.normal_(stub.reasoning_film.shift_fc.weight, std=0.5)

    labels, input_ids, hidden = make_batch()
    a = film_forward()(stub, labels, input_ids, hidden)

    hidden2 = hidden.clone()
    hidden2[:, 5:] = torch.randn_like(hidden2[:, 5:])  # reasoning span only
    b = film_forward()(stub, labels, input_ids, hidden2)

    assert not torch.allclose(a, b, atol=1e-5), "output ignores the reasoning span"


def test_output_shape_is_one_token_per_batch_element():
    """ActionProjector pools over the sequence, so each element collapses to (1, D).

    That is what makes the `cat(dim=0)` inside film_forward safe for variable-length
    spans — worth pinning, because a projector that did not pool would silently
    concatenate batch elements together.
    """
    stub = Stub()
    labels, input_ids, hidden = make_batch()

    out = film_forward()(stub, labels, input_ids, hidden)

    assert out.shape == (BATCH, 1, HIDDEN)


def test_left_padding_is_excluded_from_the_observation_span():
    """`start` counts pad tokens (151643). Padding must not enter the pooled observation.

    Getting this wrong dilutes the observation embedding with padding hidden states —
    no crash, just a quietly worse policy.
    """
    stub = Stub()

    labels, input_ids, hidden = make_batch(pad_len=0, seed=2)
    without_pad = film_forward()(stub, labels, input_ids, hidden)

    pad_labels, pad_ids, pad_hidden = make_batch(pad_len=3, seed=2)
    # Same real content, with padding prepended; pad hidden states are garbage.
    pad_hidden[:, 3:] = hidden
    pad_hidden[:, :3] = 999.0
    with_pad = film_forward()(stub, pad_labels, pad_ids, pad_hidden)

    torch.testing.assert_close(without_pad, with_pad, atol=1e-4, rtol=1e-4)


def test_boundary_moves_with_the_label_mask():
    """The reasoning span is defined solely by where labels stop being -100.

    Shifting the boundary must change which hidden states are treated as reasoning.
    """
    stub = Stub()
    nn.init.normal_(stub.reasoning_film.scale_fc.weight, std=0.5)

    torch.manual_seed(3)
    hidden = torch.randn(BATCH, 9, HIDDEN)
    input_ids = torch.full((BATCH, 9), 7, dtype=torch.long)

    outs = []
    for prompt_len in (3, 6):
        labels = torch.full((BATCH, 9), -100, dtype=torch.long)
        labels[:, prompt_len:] = 1
        outs.append(film_forward()(stub, labels, input_ids, hidden))

    assert not torch.allclose(outs[0], outs[1], atol=1e-5), (
        "the reasoning boundary is not driven by the label mask"
    )


def test_identity_residual_is_the_mean_of_the_observation_span():
    """film_forward adds a mean-pooled residual over hidden[start:end].

    At init FiLM is the identity, so the output reduces to
    input_action_proj(obs) + mean(obs) — which pins the residual exactly.
    """
    stub = Stub()
    labels, input_ids, hidden = make_batch(prompt_len=5, reasoning_len=4)

    out = film_forward()(stub, labels, input_ids, hidden)

    obs = hidden[:, :5, :]
    expected = torch.stack([stub.input_action_proj(obs[i])[0] for i in range(BATCH)])
    expected = expected + obs.mean(dim=1)

    torch.testing.assert_close(out.squeeze(1), expected, atol=1e-5, rtol=1e-5)


def test_action_projector_pools_over_the_sequence():
    """Directly pin ActionProjector's contract: (T, D) -> (1, D), independent of T."""
    from qwen2_vla.utils.fusion_modules import ActionProjector

    proj = ActionProjector(HIDDEN, HIDDEN)
    for length in (1, 7, 33):
        assert proj(torch.randn(length, HIDDEN)).shape == (1, HIDDEN)


# ----------------------------------------------------- inference-time label fabrication


def fabricate_labels(input_token_len: int, total_len: int) -> torch.Tensor:
    """Exactly what `evaluate()` does after generation.

    There are no labels at inference, so the boundary is reconstructed by marking the
    prompt span -100 and the generated span 1. Copied from modeling_qwen2_vla.py so the
    equivalence below is tested against upstream's actual construction.
    """
    labels_input = torch.ones((1, input_token_len)) * -100
    labels_output = torch.ones((1, total_len - input_token_len))
    return torch.cat([labels_input, labels_output], dim=1)


def test_fabricated_labels_recover_the_training_boundary():
    """Train and inference must agree on where reasoning starts.

    Training reads a real label mask from the dataset; inference invents one. If the two
    disagreed by even one token, the model would be conditioned on a different split at
    inference than it was trained on — with no error anywhere.
    """
    prompt_len, reasoning_len = 5, 4
    total = prompt_len + reasoning_len

    trained = torch.full((1, total), -100, dtype=torch.long)
    trained[:, prompt_len:] = 1
    fabricated = fabricate_labels(prompt_len, total)

    def boundary(labels):
        idx = (labels[:, :] == -100).int()
        xor = torch.bitwise_xor(idx[:, :-1], idx[:, 1:])
        return torch.argmax((xor != 0).float(), dim=1)

    assert torch.equal(boundary(trained), boundary(fabricated))


def test_fabricated_labels_are_float_not_long():
    """`torch.ones(...) * -100` yields float; training labels are long token ids.

    Harmless for the `== -100` comparison that drives the boundary, but pinned because it
    is a real type difference between the two paths and would matter to anyone indexing
    with them.
    """
    assert fabricate_labels(3, 7).dtype == torch.float32


def test_pad_count_is_occurrence_based_not_leading_run():
    """LATENT FRAGILITY, pinned deliberately.

    `start = sum(input_ids[i] == 151643)` COUNTS every occurrence of that id anywhere in
    the sequence — it does not measure the leading padding run. And 151643 is Qwen2-VL's
    `bos_token_id` as well as its pad id (see docs/config.json), so any occurrence outside
    the left padding silently shifts the observation span.

    Not currently triggered: Qwen2's chat template does not prepend BOS. But the code is
    one tokenizer change away from mis-slicing, and it would fail silently.
    """
    stub = Stub()
    prompt_len, reasoning_len = 5, 4
    total = prompt_len + reasoning_len

    labels = torch.full((BATCH, total), -100, dtype=torch.long)
    labels[:, prompt_len:] = 1
    torch.manual_seed(9)
    hidden = torch.randn(BATCH, total, HIDDEN)

    clean_ids = torch.full((BATCH, total), 7, dtype=torch.long)
    clean = film_forward()(stub, labels, clean_ids, hidden)

    # One PAD/BOS id in the middle of the prompt — not padding, but counted as such.
    contaminated_ids = clean_ids.clone()
    contaminated_ids[:, 2] = PAD_ID
    contaminated = film_forward()(stub, labels, contaminated_ids, hidden)

    assert not torch.allclose(clean, contaminated, atol=1e-6), (
        "expected the occurrence-count to shift the observation span; if this now passes, "
        "upstream has switched to measuring the leading run and this note can be dropped"
    )

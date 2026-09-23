"""The Eagle backbone must be constructible without flash-attn or bf16.

Upstream hard-asserts both. They are sm_80+ features, so the released code refuses to
build on the T4 (sm_75) this repo trains on, and on CPU/MPS entirely. patches/0001
relaxes that; these tests are what stops a re-pull from silently reverting it.

Deliberately tiny: `select_layer=1` keeps a single LLM layer, so this builds in seconds
instead of instantiating 3B parameters.

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_backbone_build.py -q
"""

import warnings

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backbone():
    from gr00t.model.modules.eagle_backbone import EagleBackbone

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EagleBackbone(select_layer=1, use_flash_attention=False, load_bf16=False)


def test_builds_without_flash_attn_or_bf16(backbone):
    """Bug [1] + [2]: this raised AssertionError, then NameError, before the patch."""
    assert sum(p.numel() for p in backbone.parameters()) > 0


def test_sdpa_is_actually_selected(backbone):
    """The choice must reach both towers.

    Upstream overwrote vision_config._attn_implementation unconditionally, so asserting
    on the constructor argument alone would pass while the model still ran flash-attn.
    """
    assert backbone.model.vision_model.config._attn_implementation == "sdpa"
    assert backbone.model.language_model.config._attn_implementation == "sdpa"


def test_non_flash_build_warns(backbone):
    """A non-default numerical path must announce itself.

    The kernel substitution is believed exact but is not yet verified against flash on
    CUDA, so a silent build is the failure mode to avoid.
    """
    from gr00t.model.modules.eagle_backbone import EagleBackbone

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EagleBackbone(select_layer=1, use_flash_attention=False, load_bf16=False)

    assert any("flash" in str(w.message).lower() for w in caught), (
        "building off the flash-attn path must warn"
    )


def test_select_layer_truncates_the_llm():
    """`select_layer` pops LLM layers off the top; getting it wrong changes which
    hidden state conditions the action head, silently."""
    from gr00t.model.modules.eagle_backbone import EagleBackbone

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bb = EagleBackbone(select_layer=2, use_flash_attention=False, load_bf16=False)

    assert len(bb.model.language_model.model.layers) == 2


def test_frozen_by_default(backbone):
    """tune_llm / tune_visual default False. If these ever flip, a 16GB fine-tune OOMs
    on the cloud rather than here."""
    assert not any(p.requires_grad for p in backbone.model.vision_model.parameters())
    assert not any(p.requires_grad for p in backbone.model.mlp1.parameters())


def test_released_config_overrides_are_what_we_think(tmp_path):
    """Bug [3]: the dataclass defaults are NOT the shipped model.

    Pinned because `use_relative_action` drives the ROS bridge's `state_relative`
    parameter, and a wrong value there yields smooth, plausible, wrong actions.
    """
    import json
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "checkpoints/GR00T-N1.6-3B/config.json"
    if not cfg_path.exists():
        pytest.skip("checkpoint not downloaded")

    cfg = json.loads(cfg_path.read_text())
    assert cfg["action_horizon"] == 50
    assert cfg["max_state_dim"] == 128
    assert cfg["max_action_dim"] == 128
    assert cfg["use_relative_action"] is True
    assert cfg["apply_sincos_state_encoding"] is True
    assert cfg["select_layer"] == 16

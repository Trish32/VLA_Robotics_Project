"""End-to-end: real observation -> action chunk, with the real 3.29B checkpoint.

Slow (~20s and ~13GB RAM at fp32) and skipped unless the checkpoint and the git-lfs
demo_data are both present. This is the local gate CLAUDE.md asks for — one forward on a
toy batch so a cloud job fails on data or hyperparameters rather than a shape bug.

It is NOT a metric: sdpa + fp32 differ from the shipped flash + bf16 reference
(bug_log [1], [3]).

Run: PYTHONPATH=grootN1_Robotics/upstream pytest grootN1_Robotics/tests/test_policy_endtoend.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grootN1_Robotics/tools"))

CHECKPOINT = ROOT / "grootN1_Robotics/checkpoints/GR00T-N1.6-3B"
DATASET = ROOT / "grootN1_Robotics/upstream/demo_data/gr1.PickNPlace"


def lfs_fetched() -> bool:
    parquet = next(DATASET.glob("data/**/*.parquet"), None)
    return parquet is not None and parquet.stat().st_size > 1000


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists() or not DATASET.exists() or not lfs_fetched(),
    reason="needs the N1.6 checkpoint and fetched demo_data git-lfs objects",
)


@pytest.fixture(scope="module")
def policy():
    from gr00t.data.embodiment_tags import EmbodimentTag

    from grootN1_Robotics.policy import LocalGr00tPolicy

    return LocalGr00tPolicy(EmbodimentTag.GR1, CHECKPOINT)


@pytest.fixture(scope="module")
def action(policy):
    from run_policy_cpu import build_observation

    obs = build_observation(DATASET, policy)
    act, _ = policy.get_action(obs)
    return act


def test_action_chunk_has_one_entry_per_declared_joint_group(policy, action):
    assert set(action) == set(policy.modality_configs["action"].modality_keys)


def test_action_chunk_is_finite(action):
    for key, value in action.items():
        assert np.isfinite(np.asarray(value)).all(), f"non-finite action for {key}"


def test_horizon_comes_from_the_processor_not_the_model_config(policy, action):
    """`action_horizon=50` / `max_action_dim=128` in config.json are the PADDED superset.

    The per-embodiment processor config decides what is actually consumed — 16 steps for
    GR1 — and the joint dims are the embodiment's real widths, not 128. Reading the model
    config alone gives the wrong shape, which is exactly the kind of mismatch that only
    shows up once a robot is attached.
    """
    horizon = len(policy.modality_configs["action"].delta_indices)
    assert horizon == 16
    assert policy.model.config.action_horizon == 50
    assert policy.model.config.max_action_dim == 128

    for value in action.values():
        assert np.asarray(value).shape[1] == horizon

    widths = {k: np.asarray(v).shape[-1] for k, v in action.items()}
    assert widths == {"left_arm": 7, "right_arm": 7, "left_hand": 6,
                      "right_hand": 6, "waist": 3}
    assert sum(widths.values()) == 29, "GR1 uses 29 of the 128 padded action dims"


def test_language_conditioning_changes_the_action(policy):
    """The instruction must reach the action.

    Cheap to get wrong — the processor's language key ("task") is named independently of
    the dataset's annotation key, so a mis-wired instruction still produces valid actions.
    """
    from run_policy_cpu import build_observation

    obs = build_observation(DATASET, policy)
    a, _ = policy.get_action(obs)

    obs2 = build_observation(DATASET, policy)
    obs2["language"][policy.language_key] = [["stand completely still and do nothing"]]
    b, _ = policy.get_action(obs2)

    assert any(
        not np.allclose(np.asarray(a[k]), np.asarray(b[k]), atol=1e-4) for k in a
    ), "action ignores the language instruction"

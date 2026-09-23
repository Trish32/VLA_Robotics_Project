"""The observation layout the node sends must be the one a real server validates.

The node originally sent a FLAT dict with a single leading axis:

    {"video.ego_view": (1, H, W, C), "state.single_arm": (1, D),
     "annotation.human.task_description": ["pick up the block"]}

Upstream `Gr00tPolicy.check_observation` requires a NESTED dict with bare keys and both a
batch and a time axis:

    {"video": {"ego_view": (1, 1, H, W, C)}, "state": {"single_arm": (1, 1, D)},
     "language": {"annotation.human.task_description": [["pick up the block"]]}}

Three defects in one payload — nesting, key naming, rank — and none of them was visible,
because the end-to-end test's fake server never looks at the observation it is sent. See
../../bug_log.txt [3].

The check that actually settles this is in
`grootN1_Robotics/tests/test_node_observation_against_real_policy.py`, which runs this encoder's
output through the genuine policy's `check_observation`. These tests pin the behaviour so
a future edit cannot drift back.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge.obs_encode import (  # noqa: E402
    DEFAULT_LANGUAGE_KEY,
    ModalityLayout,
    ObservationEncodeError,
    build_observation,
    parse_layout,
    reconcile,
    strip_modality_prefix,
)

H, W, D = 8, 6, 5
TASK = "pick up the red block"


def frame() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def gr00t_layout() -> ModalityLayout:
    return ModalityLayout(video_keys=("ego_view",), state_keys=("single_arm",),
                          language_key=DEFAULT_LANGUAGE_KEY)


# ------------------------------------------------------------------- the shape itself


def test_observation_is_nested_not_flat():
    obs = build_observation({"video.ego_view": frame()}, np.zeros(D), TASK, gr00t_layout())

    assert set(obs) == {"video", "state", "language"}
    assert "ego_view" in obs["video"], "keys inside a group are bare, not prefixed"
    assert "video.ego_view" not in obs, "the flat form is what the node used to send"


def test_video_has_both_a_batch_and_a_time_axis():
    obs = build_observation({"ego_view": frame()}, np.zeros(D), TASK, gr00t_layout())

    arr = obs["video"]["ego_view"]
    assert arr.ndim == 5, "check_observation asserts ndim == 5 for (B, T, H, W, C)"
    assert arr.shape == (1, 1, H, W, 3)
    assert arr.dtype == np.uint8


def test_state_has_both_axes_and_is_float32():
    obs = build_observation({"ego_view": frame()}, np.arange(D), TASK, gr00t_layout())

    arr = obs["state"]["single_arm"]
    assert arr.shape == (1, 1, D), "check_observation asserts ndim == 3 for (B, T, D)"
    assert arr.dtype == np.float32
    assert np.allclose(arr[0, 0], np.arange(D))


def test_language_is_a_list_of_lists_of_strings():
    """`[task]` passes an `isinstance(list)` check but fails the per-item horizon one."""
    obs = build_observation({"ego_view": frame()}, np.zeros(D), TASK, gr00t_layout())

    value = obs["language"][DEFAULT_LANGUAGE_KEY]
    assert value == [[TASK]]
    assert isinstance(value[0], list) and isinstance(value[0][0], str)


def test_language_horizon_greater_than_one_repeats_the_instruction():
    layout = ModalityLayout(video_keys=("ego_view",), state_keys=("single_arm",),
                            language_key="task", language_horizon=2)
    obs = build_observation({"ego_view": frame()}, np.zeros(D), TASK, layout)

    assert obs["language"]["task"] == [[TASK, TASK]]


def test_prefixed_and_bare_camera_keys_both_work():
    """Users write `video.ego_view` because that is the dataset column name."""
    assert strip_modality_prefix("video.ego_view", "video.") == "ego_view"
    assert strip_modality_prefix("ego_view", "video.") == "ego_view"


def test_multiple_cameras_are_all_present():
    layout = ModalityLayout(video_keys=("ego_view", "wrist"), state_keys=("s",),
                            language_key="task")
    obs = build_observation({"ego_view": frame(), "wrist": frame()},
                            np.zeros(D), TASK, layout)

    assert set(obs["video"]) == {"ego_view", "wrist"}


# ------------------------------------------------------------------------- refusals


def test_missing_frame_is_refused():
    layout = ModalityLayout(video_keys=("ego_view", "wrist"), state_keys=("s",),
                            language_key="task")
    with pytest.raises(ObservationEncodeError, match="no frame for video key"):
        build_observation({"ego_view": frame()}, np.zeros(D), TASK, layout)


def test_float_image_is_refused_not_cast():
    """Casting a float image to uint8 wraps values; the model would see noise."""
    with pytest.raises(ObservationEncodeError, match="uint8"):
        build_observation({"ego_view": np.zeros((H, W, 3), np.float32)},
                          np.zeros(D), TASK, gr00t_layout())


def test_wrong_channel_count_is_refused():
    with pytest.raises(ObservationEncodeError, match=r"expected \(H, W, 3\)"):
        build_observation({"ego_view": np.zeros((H, W), np.uint8)},
                          np.zeros(D), TASK, gr00t_layout())


def test_non_finite_state_is_refused():
    state = np.zeros(D)
    state[2] = np.nan
    with pytest.raises(ObservationEncodeError, match="non-finite"):
        build_observation({"ego_view": frame()}, state, TASK, gr00t_layout())


# --------------------------------------------------------------- layout negotiation


def test_parse_layout_reads_keys_and_horizons():
    layout = parse_layout({
        "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
        "state": {"modality_keys": ["left_arm", "waist"], "delta_indices": [0]},
        "language": {"modality_keys": ["annotation.human.task_description"],
                     "delta_indices": [0]},
    })

    assert layout.video_keys == ("ego_view",)
    assert layout.state_keys == ("left_arm", "waist")
    assert layout.language_key == "annotation.human.task_description"
    assert layout.video_horizon == 1


def test_parse_layout_returns_none_when_only_action_is_declared():
    """DiVLA declares only `action`; the node then uses its own parameters."""
    assert parse_layout({"action": {"modality_keys": ["action.joints"]}}) is None
    assert parse_layout(None) is None


def test_reconcile_falls_back_to_parameters_when_nothing_is_declared():
    layout = reconcile(["video.ego_view"], ["state.single_arm"], [], None)

    assert layout.video_keys == ("ego_view",)
    assert layout.state_keys == ("single_arm",)
    assert layout.language_key == DEFAULT_LANGUAGE_KEY


def test_reconcile_rejects_a_camera_key_the_server_does_not_know():
    """Caught at startup instead of as a model-side assertion mid-episode."""
    declared = parse_layout({"video": {"modality_keys": ["ego_view"],
                                       "delta_indices": [0]}})
    with pytest.raises(ObservationEncodeError, match="do not match"):
        reconcile(["video.front_cam"], ["state.single_arm"], [], declared)


def test_reconcile_rejects_a_state_key_the_server_does_not_know():
    declared = parse_layout({"state": {"modality_keys": ["left_arm", "right_arm"],
                                       "delta_indices": [0]}})
    with pytest.raises(ObservationEncodeError, match="state_modality_key"):
        reconcile(["ego_view"], ["single_arm"], [], declared)


def test_reconcile_refuses_a_multi_frame_history_rather_than_padding():
    """The node holds one frame. Repeating it would feed a stationary history the model
    never saw in training — plausible-looking input, silently wrong."""
    declared = parse_layout({"video": {"modality_keys": ["ego_view"],
                                       "delta_indices": [-1, 0]}})
    with pytest.raises(ObservationEncodeError, match="holds only the latest frame"):
        reconcile(["ego_view"], ["single_arm"], [], declared)


def test_reconcile_adopts_the_servers_language_key():
    """GR00T embodiments name the instruction differently; the server is authoritative."""
    declared = parse_layout({"language": {"modality_keys": ["task"],
                                          "delta_indices": [0]}})
    assert reconcile(["ego_view"], ["single_arm"], [], declared).language_key == "task"


def test_more_than_one_language_key_is_refused():
    with pytest.raises(ObservationEncodeError, match="exactly one"):
        parse_layout({"language": {"modality_keys": ["task", "other"],
                                   "delta_indices": [0]}})


# ------------------------------------------------------- splitting a joint vector
#
# The real GR1 embodiment does not have one state key. It has five, and their widths
# (7, 6, 7, 6, 3 = 29) are NOT carried on the wire — get_modality_config gives names and
# delta indices only. So the operator supplies them, and everything about that pairing is
# worth pinning: it decides which joints are an arm and which are a hand.

GR1_KEYS = ["left_arm", "left_hand", "right_arm", "right_hand", "waist"]
GR1_DIMS = [7, 6, 7, 6, 3]


def gr1_declared():
    return parse_layout({
        "video": {"modality_keys": ["ego_view"], "delta_indices": [0]},
        "state": {"modality_keys": GR1_KEYS, "delta_indices": [0]},
        "language": {"modality_keys": ["annotation.human.task_description"],
                     "delta_indices": [0]},
    })


def test_joint_vector_is_split_across_the_declared_state_keys():
    layout = reconcile(["ego_view"], GR1_KEYS, GR1_DIMS, gr1_declared())
    joints = np.arange(29, dtype=np.float32)

    obs = build_observation({"ego_view": frame()}, joints, TASK, layout)

    state = obs["state"]
    assert [state[k].shape for k in GR1_KEYS] == [(1, 1, d) for d in GR1_DIMS]
    assert np.allclose(state["left_arm"][0, 0], np.arange(0, 7))
    assert np.allclose(state["left_hand"][0, 0], np.arange(7, 13))
    assert np.allclose(state["waist"][0, 0], np.arange(26, 29))


def test_slices_follow_configured_order_not_sorted_order():
    """Sorting would put left_hand before left_arm and hand the arm's joints to the hand.

    `action_decode` DOES sort, because there the order is a column layout the node
    defines alone. Here it is a pairing with operator-supplied widths, so the configured
    order is authoritative — a difference worth stating in a test rather than a comment.
    """
    layout = reconcile(["ego_view"], GR1_KEYS, GR1_DIMS, gr1_declared())
    assert layout.state_keys == tuple(GR1_KEYS), "must not be sorted"
    assert list(layout.state_dims) == GR1_DIMS


def test_multi_key_state_without_dims_is_refused():
    """An even split would be well-formed and wrong; the model would run regardless."""
    with pytest.raises(ObservationEncodeError, match="state_dims"):
        reconcile(["ego_view"], GR1_KEYS, [], gr1_declared())


def test_dims_that_do_not_match_the_joint_count_are_refused():
    layout = reconcile(["ego_view"], GR1_KEYS, GR1_DIMS, gr1_declared())
    with pytest.raises(ObservationEncodeError, match="supplied 28 values"):
        build_observation({"ego_view": frame()}, np.zeros(28), TASK, layout)


def test_mismatched_dims_length_is_refused():
    with pytest.raises(ObservationEncodeError, match="paired positionally"):
        reconcile(["ego_view"], GR1_KEYS, [7, 6], gr1_declared())


def test_duplicate_state_keys_are_refused():
    with pytest.raises(ObservationEncodeError, match="duplicate"):
        reconcile(["ego_view"], ["a", "a"], [1, 1], None)

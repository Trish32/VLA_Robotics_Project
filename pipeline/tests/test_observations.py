"""The camera-frame -> world-frame step, which is where an object quietly ends up
somewhere it isn't.

Poses are composed by hand from known transforms so the expected world position is exact,
not a tolerance that happens to pass.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.observations import (  # noqa: E402
    NS_PER_MS,
    Observation,
    ObservationError,
    from_openmask3d,
    refine_with_pose,
    to_scene_nodes,
)
from pipeline.transforms import quaternion_to_matrix  # noqa: E402

STAMP = 1_000_000_000


def rigid(t=(0, 0, 0), quat=(0, 0, 0, 1)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(*quat)
    T[:3, 3] = t
    return T


def cloud_with_two_instances():
    rng = np.random.default_rng(0)
    table = rng.uniform([-0.6, -0.4, 0.70], [0.6, 0.4, 0.75], (500, 3))
    pear = rng.uniform([0.10, -0.04, 0.75], [0.17, 0.03, 0.83], (120, 3))
    points = np.vstack([table, pear])
    return points, [np.arange(500), np.arange(500, 620)]


# ------------------------------------------------------------------ OpenMask3D


def test_instances_become_observations_with_axis_aligned_extents():
    points, groups = cloud_with_two_instances()
    obs = from_openmask3d(groups, points, ["table", "pear"], [0.41, 0.33],
                          anchor_frame="keyframe_7", stamp_ns=STAMP)

    assert [o.label for o in obs] == ["table", "pear"]
    pear = obs[1]
    assert pear.anchor_frame == "keyframe_7"
    assert pear.extent[2] == pytest.approx(0.08, abs=0.01)
    assert pear.pose[2, 3] == pytest.approx(0.79, abs=0.01)


def test_a_segmentation_mask_claims_no_orientation():
    """A mask has no orientation. Inventing one would fabricate precision that
    FoundationPose is the stage actually supplying."""
    points, groups = cloud_with_two_instances()
    obs = from_openmask3d(groups, points, ["table", "pear"], [0.4, 0.3],
                          anchor_frame="kf", stamp_ns=STAMP)

    assert np.allclose(obs[1].pose[:3, :3], np.eye(3))


def test_an_empty_mask_is_refused_not_placed_at_the_origin():
    """An empty mask has no centroid; emitting one puts a 'detection' at the robot's own
    base with zero extent."""
    points, _ = cloud_with_two_instances()
    with pytest.raises(ObservationError, match="is empty"):
        from_openmask3d([np.array([], dtype=int)], points, ["pear"], [0.3],
                        anchor_frame="kf", stamp_ns=STAMP)


def test_parallel_arrays_must_agree():
    points, groups = cloud_with_two_instances()
    with pytest.raises(ObservationError, match="parallel arrays"):
        from_openmask3d(groups, points, ["table"], [0.4, 0.3],
                        anchor_frame="kf", stamp_ns=STAMP)


# --------------------------------------------------------------- FoundationPose


def base_observation() -> Observation:
    return Observation(
        node_id="pear_0", label="pear", pose=np.eye(4), extent=np.array([0.07, 0.07, 0.08]),
        anchor_frame="keyframe_7", stamp_ns=STAMP, score=0.33,
    )


def test_the_camera_frame_pose_lands_where_the_composition_says():
    """T_anchor_obj = T_anchor_cam @ T_cam_obj, checked against a hand-composed answer."""
    # Camera is 2 m along world +x, yawed 90 degrees about z.
    T_anchor_cam = rigid(t=(2.0, 0.0, 1.0), quat=(0, 0, 0.7071068, 0.7071068))
    # Object sits 0.5 m straight ahead of the camera (+z in the optical frame).
    T_cam_obj = rigid(t=(0.0, 0.0, 0.5))

    refined = refine_with_pose(base_observation(), T_cam_obj, T_anchor_cam,
                               stamp_ns=STAMP, camera_stamp_ns=STAMP)

    expected = T_anchor_cam @ T_cam_obj
    assert np.allclose(refined.pose, expected)
    assert np.allclose(refined.pose[:3, 3], expected[:3, 3])


def test_an_identity_camera_pose_passes_the_object_pose_through():
    T_cam_obj = rigid(t=(0.1, -0.2, 0.6), quat=(0, 0.3826834, 0, 0.9238795))
    refined = refine_with_pose(base_observation(), T_cam_obj, np.eye(4),
                               stamp_ns=STAMP, camera_stamp_ns=STAMP)
    assert np.allclose(refined.pose, T_cam_obj)


def test_identity_and_label_survive_refinement():
    """FoundationPose measures pose, not identity. Overwriting the label or extent would
    discard what OpenMask3D established."""
    refined = refine_with_pose(base_observation(), rigid(t=(0, 0, 0.5)), np.eye(4),
                               stamp_ns=STAMP, camera_stamp_ns=STAMP)

    assert refined.node_id == "pear_0" and refined.label == "pear"
    assert np.allclose(refined.extent, [0.07, 0.07, 0.08])
    assert refined.score == pytest.approx(0.33)


def test_composing_across_time_is_refused():
    """The guard that matters. A camera pose from a different instant displaces the
    object by however far the camera moved — small enough to read as calibration error,
    big enough to miss the grasp."""
    with pytest.raises(ObservationError, match="ms apart"):
        refine_with_pose(base_observation(), rigid(t=(0, 0, 0.5)), np.eye(4),
                         stamp_ns=STAMP, camera_stamp_ns=STAMP + 200 * NS_PER_MS)


def test_a_small_stamp_difference_is_tolerated():
    refined = refine_with_pose(base_observation(), rigid(t=(0, 0, 0.5)), np.eye(4),
                               stamp_ns=STAMP, camera_stamp_ns=STAMP + 5 * NS_PER_MS)
    assert refined.stamp_ns == STAMP


@pytest.mark.parametrize("bad,match", [
    (np.diag([2.0, 2.0, 2.0, 1.0]), "orthonormal"),
    (np.vstack([np.eye(4)[:3], [1.0, 0, 0, 1.0]]), "bottom row"),
])
def test_a_non_rigid_transform_is_refused(bad, match):
    """A scaled or transposed rotation places the object plausibly and wrongly."""
    with pytest.raises(ObservationError, match=match):
        refine_with_pose(base_observation(), bad, np.eye(4),
                         stamp_ns=STAMP, camera_stamp_ns=STAMP)


# ------------------------------------------------------------- into the graph


def test_observations_become_graph_nodes_with_the_right_kind():
    points, groups = cloud_with_two_instances()
    obs = from_openmask3d(groups, points, ["table", "pear"], [0.41, 0.33],
                          anchor_frame="keyframe_7", stamp_ns=STAMP)
    nodes = to_scene_nodes(obs)

    kinds = {n.label: n.kind.value for n in nodes}
    assert kinds == {"table": "surface", "pear": "object"}
    assert all(n.anchor_frame == "keyframe_7" for n in nodes)
    assert all(n.tf_frame.startswith("object/") for n in nodes)


def test_the_whole_chain_lands_in_a_queryable_graph():
    """End to end for the halves that can run: segmentation creates the node, pose
    refinement places it, the graph answers a referring expression."""
    from pipeline.scene_graph import SceneGraph

    points, groups = cloud_with_two_instances()
    obs = from_openmask3d(groups, points, ["table", "pear"], [0.41, 0.33],
                          anchor_frame="keyframe_7", stamp_ns=STAMP)
    refined = [obs[0], refine_with_pose(obs[1], rigid(t=(0.13, 0.0, 0.79)), np.eye(4),
                                        stamp_ns=STAMP, camera_stamp_ns=STAMP)]

    graph = SceneGraph()
    for node in to_scene_nodes(refined):
        graph.upsert(node)
    graph.infer_relations(STAMP)

    assert graph.resolve("pear", STAMP).label == "pear"
    assert "the pear is on the table" in graph.describe()


# ------------------------------------------------- fusion under real rotations

def test_fusion_round_trips_under_many_arbitrary_camera_rotations():
    """A static object must land in the same world spot from every viewpoint.

    The other composition tests use identity or single hand-built rotations, which a
    transposed rotation block or a missing inverse can survive: at identity, R and R.T
    are the same matrix. Sweeping 200 arbitrary orientations is what makes those
    mistakes show up, and it is the invariant the whole world model rests on — the
    same object seen from anywhere is one node, not a smear.
    """
    rng = np.random.default_rng(7)
    world_point = np.array([1.3, -0.4, 2.1])
    observation = Observation(
        node_id="cup_1", label="cup", pose=rigid(t=world_point),
        extent=[0.08, 0.08, 0.12], anchor_frame="map", stamp_ns=STAMP)

    for _ in range(200):
        quat = rng.normal(size=4)
        T_world_cam = rigid(t=rng.uniform(-3, 3, 3), quat=quat / np.linalg.norm(quat))
        T_cam_obj = np.linalg.inv(T_world_cam) @ rigid(t=world_point)

        fused = refine_with_pose(observation, T_cam_obj, T_world_cam,
                                 stamp_ns=STAMP, camera_stamp_ns=STAMP)
        assert np.allclose(fused.pose[:3, 3], world_point, atol=1e-9)


def test_camera_rotation_error_displaces_an_object_by_range_times_angle():
    """Placement error scales with RANGE, which is why ATE understates it.

    Measured on the real trajectory: 1.03 cm ATE puts objects 7.63 cm from truth,
    because a camera rotation error acts across the object's distance. Locking the
    mechanism here keeps that explanation honest if the fusion path ever changes.
    """
    angle = np.radians(2.0)
    axis_quat = (0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2))   # 2 deg about +y

    for distance in (1.0, 2.0, 4.0):
        observation = Observation(
            node_id="o", label="o", pose=rigid(t=(0, 0, distance)),
            extent=[0.1, 0.1, 0.1], anchor_frame="map", stamp_ns=STAMP)
        T_cam_obj = rigid(t=(0, 0, distance))

        fused = refine_with_pose(observation, T_cam_obj, rigid(quat=axis_quat),
                                 stamp_ns=STAMP, camera_stamp_ns=STAMP)
        displaced = np.linalg.norm(fused.pose[:3, 3] - np.array([0, 0, distance]))
        # Chord length for a rotation of `angle` at radius `distance`; d*theta is the
        # small-angle bound on it, approached from below.
        assert np.isclose(displaced, 2 * distance * np.sin(angle / 2), atol=1e-9)
        assert displaced < distance * angle

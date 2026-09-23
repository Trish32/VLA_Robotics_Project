"""The world model node's own logic — TF shape and the pre-grasp standoff.

`pipeline/tests` covers the scene graph library; none of it covers this node. That gap
is how the standoff bug below survived: the library was well tested, the node that
consumes it was not, and the two were reported together as "logic tested".

Needs rclpy, so it runs in the ROS2 container (see conftest).
"""

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped

from vla_bridge.world_model_node import WorldModelNode

from pipeline.identity import Frames
from pipeline.scene_graph import NodeKind, SceneNode


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = WorldModelNode()
    yield n
    n.destroy_node()


def make(node_id="monitor_4", extent=(0.5, 1.0, 0.94), centre=(0.0, 0.0, 2.0)):
    pose = np.eye(4)
    pose[:3, 3] = centre
    return SceneNode(
        node_id=node_id, kind=NodeKind.OBJECT, label="monitor",
        tf_frame=Frames.instance(node_id), anchor_frame=Frames.keyframe(0),
        pose=pose, extent=np.asarray(extent, float), stamp_ns=10**9,
    )


def test_standoff_clears_the_objects_surface_not_its_centre(node):
    """The bug this file was written for.

    `approach * standoff` measures from the object ORIGIN, so clearance shrinks as the
    object grows and goes negative once the object exceeds twice the standoff. On the
    real end-to-end instances the monitor is 0.94 m tall, so a 0.12 m standoff sat
    0.35 m inside its bounding box — a pre-grasp the planner would drive into.
    """
    obj = make(extent=(0.5, 1.0, 0.94))
    offset = node._standoff_offset(obj)

    half_height = 0.94 / 2
    assert offset[2] == pytest.approx(half_height + node.standoff)
    assert offset[2] > half_height, "pre-grasp is inside the object"


def test_clearance_is_constant_regardless_of_object_size(node):
    """A standoff that changes meaning with object size is not a standoff."""
    for height in (0.05, 0.4, 0.94, 2.0):
        offset = node._standoff_offset(make(extent=(0.3, 0.3, height)))
        assert offset[2] - height / 2 == pytest.approx(node.standoff)


def test_support_function_handles_a_diagonal_approach_axis(node):
    """Exact for any direction, not only the axis-aligned ones."""
    node.approach = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    obj = make(extent=(0.4, 0.4, 0.6))
    # Support of an axis-aligned box along `a` is sum |a_i| * half_extent_i.
    expected = (0.2 + 0.3) / np.sqrt(2) + node.standoff
    assert np.linalg.norm(node._standoff_offset(obj)) == pytest.approx(expected)


def test_pregrasp_is_a_child_of_the_object_so_it_tracks_it(node):
    """Anchoring to map would freeze the pre-grasp where the object used to be."""
    obj = make()
    tf = node._pregrasp(obj, rclpy.clock.Clock().now().to_msg())

    assert isinstance(tf, TransformStamped)
    assert tf.header.frame_id == obj.tf_frame
    assert tf.child_frame_id == "target/pregrasp"
    # A pure translation: the approach direction carries the orientation, and inventing
    # a grasp rotation here would claim precision segmentation never supplied.
    assert (tf.transform.rotation.x, tf.transform.rotation.y,
            tf.transform.rotation.z, tf.transform.rotation.w) == (0.0, 0.0, 0.0, 1.0)


def test_target_pose_and_pregrasp_frame_agree(node):
    """Two publishers of one fact must not drift apart."""
    obj = make()
    stamp = rclpy.clock.Clock().now().to_msg()
    tf = node._pregrasp(obj, stamp)
    pose = node._target_pose(obj, stamp)

    assert pose.header.frame_id == tf.header.frame_id
    assert pose.pose.position.z == pytest.approx(tf.transform.translation.z)

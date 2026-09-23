"""Publishes the scene graph as a shared world model: TF frames, detections, grounding.

This is the layer that makes `pipeline/scene_graph.py` shared rather than a library. It
owns one graph, folds in whatever the producers report, and republishes the result in the
forms other ROS nodes already know how to consume:

    tf2                          <anchor> -> object/<id>     live geometry
    /world_model/objects         vision_msgs/Detection3DArray
    /world_model/graph           std_msgs/String (JSON)      nodes + relations
    /world_model/target          geometry_msgs/PoseStamped   resolved pre-grasp
    tf2                          object/<id> -> target/pregrasp

Producers publish observations to `/world_model/observations`; SLAM's odometry updates
the robot node directly.

THE TF PARENT IS THE ANCHOR, NOT `map`
--------------------------------------
Objects are published as children of the keyframe they were observed from, not of `map`.
That is the whole point of anchoring: when the SLAM backend's bundle adjustment moves a
keyframe, TF2 recomposes `map -> anchor -> object` and every object follows the
correction for free. Publishing `map -> object` instead bakes in the pre-optimisation
estimate, and the object visibly slides off the table it is sitting on after a loop
closure — while every individual transform still looks perfectly well-formed.

STALE NODES STOP BEING PUBLISHED
--------------------------------
A frame that simply stops updating is worse than a missing one: consumers keep reading
the last transform and cannot tell it is old. Dropping it makes their `lookup_transform`
fail loudly at the moment the fact expired, which is the behaviour a planner can act on.

WHY THE PRE-GRASP IS A TF FRAME
-------------------------------
`target/pregrasp` is published as a child of the object's own frame with a pure
translation. Any consumer then asks TF for it in whatever frame it works in —
`lookup_transform("base_link", "target/pregrasp")` composes through the arm's kinematic
chain automatically. Computing a base-frame pose here instead would duplicate the
transform maths that TF2 exists to do, and would go stale the instant the robot moved.

The pose never enters the VLA. It resolves the referring expression and positions the
approach; the policy still receives images, state and language.
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

# `pipeline/` is the single source for the world model and is NOT vendored here —
# two copies of a data model diverge, and the divergence shows up as a planner acting on
# relations the producer never asserted. The ROS container mounts only
# /ws/src/vla_bridge, so the repo root is added explicitly when present. Running this
# node in the container needs one extra read-only mount:
#
#     - /Users/trish/VLAProjects/pipeline:/ws/src/pipeline:ro
#
# Without it the import below fails loudly at startup rather than silently degrading.
import sys as _sys
from pathlib import Path as _Path

for _candidate in (_Path(__file__).resolve().parents[3], _Path("/ws/src")):
    if (_candidate / "pipeline" / "scene_graph.py").exists():
        _sys.path.insert(0, str(_candidate))
        break

from pipeline.scene_graph import (  # noqa: E402
    NodeKind,
    SceneGraph,
    SceneNode,
    StaleFact,
)
from pipeline.transforms import matrix_to_quaternion as _quat_from_matrix  # noqa: E402
from pipeline.transforms import quaternion_to_matrix as _quat_to_matrix  # noqa: E402


def _pose_from_msg(pose_msg) -> np.ndarray:
    """geometry_msgs/Pose -> (4, 4). Quaternion is scalar-last, as ROS defines it."""
    q = pose_msg.orientation
    T = np.eye(4)
    T[:3, :3] = _quat_to_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = (pose_msg.position.x, pose_msg.position.y, pose_msg.position.z)
    return T


class WorldModelNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("world_model_node", **kwargs)

        p = self.declare_parameter
        p("map_frame", "map")
        p("odometry_topic", "/droid/odometry")
        p("observations_topic", "/world_model/observations")
        p("request_topic", "/world_model/target_request")
        p("publish_hz", 10.0)
        # Clearance from the object's SURFACE along the approach axis (see
        # `_standoff_offset`). Objects are approached from above on a tabletop;
        # `approach_axis` is in the OBJECT's frame so a side-approach is a parameter
        # change rather than a code change.
        p("pregrasp_standoff_m", 0.12)
        p("approach_axis", [0.0, 0.0, 1.0])
        p("publish_markers", False)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = g("map_frame")
        self.standoff = float(g("pregrasp_standoff_m"))
        self.approach = np.asarray([float(v) for v in g("approach_axis")], dtype=float)
        norm = float(np.linalg.norm(self.approach))
        if norm < 1e-9:
            raise ValueError("approach_axis must be non-zero; it is a direction")
        self.approach /= norm

        self.graph = SceneGraph()
        self._target_id: str | None = None

        self.create_subscription(Odometry, g("odometry_topic"), self._on_odometry, 10)
        self.create_subscription(
            Detection3DArray, g("observations_topic"), self._on_observations, 10
        )
        self.create_subscription(String, g("request_topic"), self._on_request, 10)

        self.objects_pub = self.create_publisher(Detection3DArray, "/world_model/objects", 10)
        self.graph_pub = self.create_publisher(String, "/world_model/graph", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/world_model/target", 10)

        self.tf_pub = None
        try:
            from tf2_ros import TransformBroadcaster

            self.tf_pub = TransformBroadcaster(self)
        except ImportError:
            self.get_logger().warn("tf2_ros unavailable; publishing topics only")

        self.create_timer(1.0 / float(g("publish_hz")), self._publish)
        self.get_logger().info(
            f"world_model_node up: frames anchored per-keyframe, standoff "
            f"{self.standoff:.3f} m along {self.approach.tolist()}"
        )

    # ------------------------------------------------------------------ inputs

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_odometry(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        self.graph.upsert(SceneNode(
            node_id="robot", kind=NodeKind.ROBOT, label="robot",
            tf_frame=msg.child_frame_id or "base_link",
            anchor_frame=msg.header.frame_id or self.map_frame,
            pose=_pose_from_msg(msg.pose.pose),
            extent=np.zeros(3),
            stamp_ns=int(stamp.sec) * 10**9 + int(stamp.nanosec),
        ))

    def _on_observations(self, msg: Detection3DArray) -> None:
        """Producers report instances here: OpenMask3D labels, FoundationPose poses.

        `header.frame_id` is taken as the ANCHOR frame, so a producer that reports
        relative to the keyframe it observed from gets loop-closure correction for free.
        """
        anchor = msg.header.frame_id or self.map_frame
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 10**9 + int(stamp.nanosec)

        for detection in msg.detections:
            if not detection.results:
                # No hypothesis means no label, and an unlabelled instance cannot be
                # resolved by a referring expression. Skip rather than invent "unknown".
                continue
            best = max(detection.results, key=lambda r: r.hypothesis.score)
            label = best.hypothesis.class_id
            node_id = detection.id or f"{label}_{len(self.graph.nodes)}"
            size = detection.bbox.size

            self.graph.upsert(SceneNode(
                node_id=node_id,
                kind=NodeKind.SURFACE if label in {"table", "floor", "shelf", "counter"}
                else NodeKind.OBJECT,
                label=label,
                tf_frame=f"object/{node_id}",
                anchor_frame=anchor,
                pose=_pose_from_msg(detection.bbox.center),
                extent=np.array([size.x, size.y, size.z]),
                stamp_ns=stamp_ns,
                clip_similarity=float(best.hypothesis.score),
            ))

    def _on_request(self, msg: String) -> None:
        """A referring expression -> a pre-grasp frame. The grounding step."""
        now = self._now_ns()
        try:
            node = self.graph.resolve(msg.data, now)
        except StaleFact as exc:
            self._target_id = None
            self.get_logger().warn(f"{msg.data!r}: {exc}")
            return
        except KeyError as exc:
            self._target_id = None
            self.get_logger().warn(str(exc))
            return

        self._target_id = node.node_id
        self.get_logger().info(
            f"target {msg.data!r} -> {node.node_id} ({node.label}), "
            f"frame {node.tf_frame}"
        )

    # ----------------------------------------------------------------- outputs

    def _publish(self) -> None:
        now = self._now_ns()
        self.graph.forget_older_than(now)
        self.graph.infer_relations(now)

        stamp = self.get_clock().now().to_msg()
        detections = Detection3DArray()
        detections.header.stamp = stamp
        detections.header.frame_id = self.map_frame
        transforms: list[TransformStamped] = []

        for node in self.graph.nodes.values():
            if node.kind is NodeKind.ROBOT:
                continue
            limit = self.graph.max_age_s[node.kind]
            if limit != float("inf") and node.age_s(now) > limit:
                # Deliberately not published: see the module docstring. A frame that
                # stops updating is read as current by every consumer.
                continue

            transforms.append(self._transform(node, stamp))
            detections.detections.append(self._detection(node, stamp))

        if self._target_id and self._target_id in self.graph.nodes:
            transforms.append(self._pregrasp(self.graph.nodes[self._target_id], stamp))
            self.target_pub.publish(self._target_pose(self.graph.nodes[self._target_id], stamp))

        if self.tf_pub is not None and transforms:
            self.tf_pub.sendTransform(transforms)

        self.objects_pub.publish(detections)
        snapshot = self.graph.snapshot(now)
        snapshot["target"] = self._target_id
        self.graph_pub.publish(String(data=json.dumps(snapshot)))

    def _transform(self, node: SceneNode, stamp) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = node.anchor_frame     # NOT map; see module docstring
        tf.child_frame_id = node.tf_frame
        t = node.pose[:3, 3]
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        qx, qy, qz, qw = _quat_from_matrix(node.pose[:3, :3])
        tf.transform.rotation.x, tf.transform.rotation.y = float(qx), float(qy)
        tf.transform.rotation.z, tf.transform.rotation.w = float(qz), float(qw)
        return tf

    def _detection(self, node: SceneNode, stamp) -> Detection3D:
        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = node.anchor_frame
        detection.id = node.node_id

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = node.label
        hypothesis.hypothesis.score = float(node.clip_similarity or node.confidence)
        detection.results.append(hypothesis)

        box = BoundingBox3D()
        t = node.pose[:3, 3]
        box.center.position.x = float(t[0])
        box.center.position.y = float(t[1])
        box.center.position.z = float(t[2])
        qx, qy, qz, qw = _quat_from_matrix(node.pose[:3, :3])
        box.center.orientation.x, box.center.orientation.y = float(qx), float(qy)
        box.center.orientation.z, box.center.orientation.w = float(qz), float(qw)
        box.size.x, box.size.y, box.size.z = (float(v) for v in node.extent)
        detection.bbox = box
        return detection

    def _standoff_offset(self, node: SceneNode) -> np.ndarray:
        """Standoff measured from the object's SURFACE, not from its origin.

        `approach * standoff` alone is a fixed offset from the object's centre, so the
        actual clearance shrinks as the object grows and goes negative once the object
        is bigger than twice the standoff. Caught by running the node on the real
        end-to-end instances: the monitor is 0.94 m tall, so a 0.12 m "standoff" sat
        0.35 m INSIDE its bounding box, and a planner would have driven the gripper
        into the thing it was reaching for.

        For an axis-aligned box the distance from the centre to the supporting plane
        along a unit direction `a` is the support function `sum |a_i| * half_extent_i`,
        which is exact for any approach axis, not just the axis-aligned ones.
        """
        support = float(np.abs(self.approach) @ (np.asarray(node.extent, float) / 2.0))
        return self.approach * (support + self.standoff)

    def _pregrasp(self, node: SceneNode, stamp) -> TransformStamped:
        """A pure translation off the object, expressed in the OBJECT's frame.

        Child of the object rather than of map, so it tracks the object automatically
        and TF composes it into whatever frame the caller works in.
        """
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = node.tf_frame
        tf.child_frame_id = "target/pregrasp"
        offset = self._standoff_offset(node)
        tf.transform.translation.x = float(offset[0])
        tf.transform.translation.y = float(offset[1])
        tf.transform.translation.z = float(offset[2])
        tf.transform.rotation.w = 1.0
        return tf

    def _target_pose(self, node: SceneNode, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = node.tf_frame
        offset = self._standoff_offset(node)
        pose.pose.position.x = float(offset[0])
        pose.pose.position.y = float(offset[1])
        pose.pose.position.z = float(offset[2])
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WorldModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

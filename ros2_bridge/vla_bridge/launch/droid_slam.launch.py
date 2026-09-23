"""DROID-SLAM against a remote SLAM server.

A different node from the VLA one — DROID consumes frames and produces camera poses, not
action chunks — but the same transport, because `lietorch` and `droid_backends` are
CUDA-compile-only and the system has to run off-board regardless.

`camera_intrinsics` is left as the [0.0] sentinel so the node takes fx/fy/cx/cy from
CameraInfo. Override it only if your driver does not publish CameraInfo; two sources of
truth for intrinsics is how one of them goes stale.

`publish_tf` broadcasts map -> camera_link. Turn it off if something else in the graph
already owns that edge, or tf2 will see two publishers fighting over one transform.

Arguments are resolved through an OpaqueFunction so each parameter reaches the node as a
real Python type. `publish_tf` is the clearest case: a substitution delivers the string
`'false'`, which is truthy, so the transform would be broadcast anyway.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *_args, **_kwargs):
    resolve = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    return [Node(
        package="vla_bridge",
        executable="droid_slam_node",
        name="droid_slam_node",
        output="screen",
        parameters=[{
            "host": resolve("host"),
            "port": int(resolve("port")),
            "image_topic": resolve("image_topic"),
            "camera_info_topic": resolve("camera_info_topic"),
            "camera_intrinsics": [0.0],   # sentinel: read them from CameraInfo
            "map_frame": resolve("map_frame"),
            "camera_frame": resolve("camera_frame"),
            "publish_tf": resolve("publish_tf").lower() in ("1", "true", "yes"),
            # DROID's dense BA is heavy; a 5s policy timeout would spuriously fail.
            "timeout_ms": 10000,
        }],
    )]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("host", default_value="localhost",
                              description="GPU host running droidSLAM_monocular/slam_server.py"),
        DeclareLaunchArgument("port", default_value="5557",
                              description="SLAM server port (5555 GR00T, 5556 DiVLA)"),
        DeclareLaunchArgument("image_topic",
                              default_value="/camera/color/image_raw",
                              description="RGB topic fed to DROID"),
        DeclareLaunchArgument("camera_info_topic",
                              default_value="/camera/color/camera_info",
                              description="source of fx, fy, cx, cy"),
        DeclareLaunchArgument("map_frame", default_value="map",
                              description="odometry parent frame"),
        DeclareLaunchArgument("camera_frame", default_value="camera_link",
                              description="odometry child frame"),
        DeclareLaunchArgument("publish_tf", default_value="true",
                              description="broadcast map -> camera_link"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])

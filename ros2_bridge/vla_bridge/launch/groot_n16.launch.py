"""GR00T N1.6 (GR1 humanoid) against a remote policy server.

Every value here was read off the real checkpoint, not chosen — see
`grootN1_Robotics/tests/test_node_observation_against_real_policy.py`, which negotiates this same
layout from `LocalGr00tPolicy` and runs the result through the policy's own
`check_observation`.

Two of them are load-bearing and neither is discoverable at runtime:

`state_dims`
    GR1's state is FIVE keys, not one: left_arm 7, left_hand 6, right_arm 7,
    right_hand 6, waist 3 = 29. `get_modality_config` carries the key names but no
    widths (they live in the checkpoint's `statistics.json`), so the split has to be
    declared here. Get the order or the widths wrong and the model receives a
    well-formed observation with hand joints in the arm slot.

`state_relative:=true`
    N1.6 predicts state-relative chunks for most embodiments. Set it false and the arm
    drives toward the origin instead of tracking a delta.

The 29 joint names below are placeholders in GR1's own ordering; replace them with the
names your robot's /joint_states actually publishes, keeping the order.

Launch arguments are resolved through an OpaqueFunction rather than passed as
`LaunchConfiguration` objects. Substitutions carry no type: a `LaunchConfiguration`
sitting inside a Python list is written to the params file as a bare string, and rclpy
then refuses it with "expecting type 'STRING_ARRAY'"; `port` arrives as `'5555'` and
fails the same way against an integer declaration. Resolving here makes every value a
real Python `str`, `int` or `bool` before it reaches the node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Order matters twice over: it is the order state_dims cuts, and the order the published
# JointTrajectory names its points.
GR1_JOINTS = (
    [f"left_arm_{i}" for i in range(7)]
    + [f"left_hand_{i}" for i in range(6)]
    + [f"right_arm_{i}" for i in range(7)]
    + [f"right_hand_{i}" for i in range(6)]
    + [f"waist_{i}" for i in range(3)]
)


def launch_setup(context, *_args, **_kwargs):
    resolve = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    return [Node(
        package="vla_bridge",
        executable="vla_policy_node",
        name="vla_policy_node",
        output="screen",
        parameters=[{
            "host": resolve("host"),
            "port": int(resolve("port")),
            "camera_topics": [resolve("camera_topic")],
            "camera_modality_keys": ["video.ego_view"],
            "state_modality_keys": ["state.left_arm", "state.left_hand",
                                    "state.right_arm", "state.right_hand",
                                    "state.waist"],
            "state_dims": [7, 6, 7, 6, 3],
            "joint_names": GR1_JOINTS,
            "task_description": resolve("task"),
            "state_relative": True,
            "execute_k": 8,       # of a 16-step chunk, so one is always in flight
            "control_dt": 1.0 / 30.0,   # N1.6 System 1 runs at 30Hz
            "max_obs_age_s": 0.25,
        }],
    )]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("host", default_value="localhost",
                              description="GPU host running the GR00T PolicyServer"),
        DeclareLaunchArgument("port", default_value="5555",
                              description="PolicyServer port"),
        DeclareLaunchArgument("camera_topic",
                              default_value="/camera/color/image_raw",
                              description="RGB topic feeding video.ego_view"),
        DeclareLaunchArgument("task", default_value="pick up the red block",
                              description="initial instruction; /vla/task overrides it"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])

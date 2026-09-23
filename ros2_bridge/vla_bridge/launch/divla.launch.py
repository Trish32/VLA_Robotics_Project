"""DiffusionVLA against `DexVLA_Robotics/policy_server.py`.

Same node as GR00T, three parameters different — which is the point of the design.

`state_relative:=false`
    DiVLA emits ABSOLUTE joint targets; N1.6 emits state-relative chunks. The node
    defaults to true for N1.6, so a DiVLA deployment must override it. Wrong value here
    adds the current pose to an already-absolute target and the arm drives off.

`state_dims` is empty
    DiVLA has no per-embodiment modality config the way GR00T does; its server declares
    only the single action key it emits. One state key therefore takes the whole
    /joint_states vector, and the node falls back to its own parameters for the rest of
    the layout.

`control_dt`
    DiVLA's ScaleDP head predicts a 50-step chunk. At 10Hz that is 5 seconds of motion
    per call, of which we execute 16 steps before re-querying.

Arguments are resolved through an OpaqueFunction so each parameter reaches the node as a
real Python type — see `groot_n16.launch.py` for why a bare LaunchConfiguration inside a
list is rejected as a STRING where a STRING_ARRAY is declared.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Bimanual Aloha/AgileX layout: 6 joints + 1 gripper per arm.
ALOHA_JOINTS = (
    [f"left_joint_{i}" for i in range(6)] + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
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
            "camera_modality_keys": ["video.top"],
            "state_modality_keys": ["state.joints"],
            "state_dims": [0],          # sentinel for "unset": one key, whole vector
            "joint_names": ALOHA_JOINTS,
            "task_description": resolve("task"),
            "state_relative": False,    # <-- the one that breaks a robot if wrong
            "execute_k": 16,            # of a 50-step chunk
            "control_dt": 0.1,
            "max_obs_age_s": 0.5,
        }],
    )]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("host", default_value="localhost",
                              description="GPU host running DexVLA_Robotics/policy_server.py"),
        DeclareLaunchArgument("port", default_value="5556",
                              description="DiVLA server port (5555 is GR00T's)"),
        DeclareLaunchArgument("camera_topic",
                              default_value="/camera/color/image_raw",
                              description="RGB topic feeding the model"),
        DeclareLaunchArgument("task", default_value="pick up the red block",
                              description="initial instruction; /vla/task overrides it"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])

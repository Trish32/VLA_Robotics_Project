# ROS2 bridge — GR00T N1.6, DiffusionVLA and DROID-SLAM

One ROS2 node drives both VLAs; switching between them is a launch file, not a code
change. A second node drives DROID-SLAM over the same transport, because `lietorch` and
`droid_backends` are CUDA-compile-only and the SLAM system has to run off-board anyway.

| Launch file | Node | Server | Port |
|---|---|---|---|
| `groot_n16.launch.py` | `vla_policy_node` | `gr00t.eval.run_gr00t_server` | 5555 |
| `divla.launch.py` | `vla_policy_node` | `DexVLA_Robotics/policy_server.py` | 5556 |
| `droid_slam.launch.py` | `droid_slam_node` | `droidSLAM_monocular/slam_server.py` | 5557 |

## Design: the model does not live in ROS

```
  ROS2 Jazzy container (Mac, arm64)          GPU host (GCP L4 spot / local CUDA)
  ┌──────────────────────────────┐           ┌────────────────────────────────┐
  │ /camera/color/image_raw ──┐  │           │  gr00t.policy.PolicyServer     │
  │ /joint_states ────────────┼──┤   ZMQ     │    ├── GR00T N1.6   (port 5555)│
  │ /vla/task ────────────────┘  │  msgpack  │    ├── DiVLA        (port 5556)│
  │                              │ ────────► │    └── DROID-SLAM   (port 5557)│
  │  vla_policy_node             │ ◄──────── │                                │
  │    └─► /vla/joint_trajectory │           │  action chunk (H, dof), or an  │
  │  droid_slam_node             │           │  SE3 pose for the SLAM node    │
  │    └─► /droid/odometry, tf   │           └────────────────────────────────┘
  └──────────────────────────────┘
     rclpy · pyzmq · msgpack · numpy                torch · transformers · CUDA
     NO torch
```

Importing `gr00t` pulls in torch, transformers and diffusers — several GB of wheels, none
of which can use a GPU from a Colima container on Apple Silicon anyway. So the bridge
restates GR00T's wire protocol in [`vla_bridge/wire.py`](vla_bridge/vla_bridge/wire.py)
(~100 lines: msgpack, with ndarrays carried as `.npy` bytes) and the ROS container stays
light enough to run on the Mac.

That duplication is only safe because it is checked:
`test/test_wire_matches_upstream.py` asserts **byte-level** agreement against upstream's
real `MsgSerializer`, and runs on the Mac where `gr00t` is installed. If NVIDIA changes
the encoding, that test fails before a robot does.

## Why chunking makes remote inference work

A VLA predicts `H` future actions per call. The node executes `execute_k < H` of them and
re-queries, so a fresh chunk is always in flight before the current one is exhausted. One
network round trip therefore covers `execute_k` control steps — which is what makes it
viable to run the policy on a GCP box while the arm is on your desk.

Deliberate safety behaviour:
- observations older than `max_obs_age_s` (default 0.25s) are refused — acting on a stale
  frame is worse than not acting;
- non-finite actions are dropped with an error rather than published;
- nothing is published until every configured input has been seen at least once;
- an action chunk whose width disagrees with the joint list is refused, not truncated.

## The two parameters that will bite you

**`state_relative`.** GR00T N1.6 predicts *state-relative* action chunks for most
embodiments; N1.5 and DiVLA predict absolute targets. Set it wrong and the arm drives
toward the origin instead of tracking a delta. Defaults to `true` for N1.6;
`divla.launch.py` sets it `false`.

**`state_dims`.** A real embodiment does not have one state key. GR1 has five —
`left_arm 7, left_hand 6, right_arm 7, right_hand 6, waist 3` = 29 — and the widths are
**not on the wire**: `get_modality_config` carries key names and delta indices only, while
the widths live in the checkpoint's `statistics.json`. So the launch file declares how
`/joint_states` is cut, paired positionally with `state_modality_keys`. The node refuses
to guess: an even split would produce a perfectly well-formed observation with hand joints
in the arm slot, and the model would run.

## The observation layout

Not obvious, and wrong here for a long time (`bug_log.txt` [3]). `Gr00tPolicy` requires:

```python
{"video":    {"ego_view":   np.uint8   (B, T, H, W, C)},   # nested, BARE keys,
 "state":    {"left_arm":   np.float32 (B, T, D)},         # batch AND time axes
 "language": {"annotation.human.task_description": [["pick up the block"]]}}
```

The flat spelling (`{"video.ego_view": ...}`) exists only inside `Gr00tSimPolicyWrapper`,
which converts it before calling the policy. `vla_bridge/obs_encode.py` builds the nested
form and is deliberately rclpy-free so it can be handed to a real policy's
`check_observation` — which is what `grootN1_Robotics/tests/test_node_observation_against_real_policy.py`
does.

## Running it

The container mounts this package at `/ws/src/vla_bridge` (see
`~/ROS2Projects/docker-compose.yml`); `pyzmq` and `msgpack` are baked into the image.

```bash
docker exec ros2-cpp-dev bash -c '
  source /opt/ros/jazzy/setup.bash && cd /ws && colcon build --packages-select vla_bridge'

# tests — check the COUNT, not just the colour (bug_log.txt [6])
docker exec ros2-cpp-dev bash -c '
  source /opt/ros/jazzy/setup.bash && cd /ws/src/vla_bridge && python3 -m pytest test -q'

# run
docker exec ros2-cpp-dev bash -c '
  source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash &&
  ros2 launch vla_bridge groot_n16.launch.py host:=<gpu-host> task:="pick up the block"'
```

On the GPU host:

```bash
python -m gr00t.eval.run_gr00t_server --port 5555                    # GR00T N1.6
python DexVLA_Robotics/policy_server.py --backbone <qwen2-vl-2b> --port 5556  # DiVLA
```

Set the task at runtime:
`ros2 topic pub /vla/task std_msgs/String "data: 'pick up the red block'" -1`


**→ [RESULTS.md](RESULTS.md)** — wire-format and live-graph verification
**→ [Plan.md](Plan.md)** — what is left

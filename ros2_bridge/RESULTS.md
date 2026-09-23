# ROS2 bridge — results

Wire format is **byte-identical** to upstream's `MsgSerializer` across 23 cases, and the
world model now runs live in a ROS2 Jazzy graph over DDS. Cross-cutting numbers are in
the root [RESULTS.md](../RESULTS.md) §4.3.

## Status

| Check | Result |
|---|---|
| `colcon build`, `ros2 run`, all three launch files | pass |
| Wire format vs upstream `MsgSerializer` (23 cases) | byte-identical |
| Bridge suite in the ROS container | 72 passed, 23 skipped |
| Node observation vs **real** 3.29B GR00T `check_observation` | 5 passed |
| Bridge client vs **real** GR00T `PolicyServer` (CPU) | 4 passed |
| Bridge client vs **real** DiVLA server over ZMQ (CPU) | 6 passed |
| SLAM node → `Odometry`, scripted server (container) | 9 passed |
| SLAM server preprocessing + CUDA guard (Mac) | 27 passed, 3 skipped |

**What is not verified: DROID's tracking itself.** `lietorch` and `droid_backends` are
CUDA-compile-only, so `droidSLAM_monocular/slam_server.py` raises on a non-CUDA box rather than
importing a stub — a stub would return well-shaped poses that are silently wrong, which
for a pose is the worst possible outcome. Everything around it is tested (preprocessing,
the intrinsics rescale, the endpoint wiring, the node's publishing and refusal paths); the
tracking is cloud-verified-pending. The pose *direction* — `video.poses` is
world-to-camera and the server inverts to camera-to-world — is read from upstream, not
measured. The quaternion convention IS settled: `depth_video.py` initialises poses to
`[0,0,0, 0,0,0,1]`, which is the identity only if w is last, and there is a test pinning
that literal so upstream cannot change it silently.

## Tests that mean something, and tests that do not

Three protocol bugs have hidden behind a hand-written fake server ([1], [2], [3] in
`bug_log.txt`) — the request envelope, the action key convention, and the observation
layout. A fake agrees with whatever you wrote it to agree with, so the fake here now
validates observations the way upstream does, and every claim about the wire is checked
against a real peer:

| Test | Peer | Env |
|---|---|---|
| `test_wire_matches_upstream.py` | upstream `MsgSerializer` | `grootN1_Robotics` |
| `test_wire_client_errors.py` | scripted REP socket | any |
| `grootN1_Robotics/tests/test_node_observation_against_real_policy.py` | real 3.29B policy | `grootN1_Robotics` |
| `grootN1_Robotics/tests/test_ros_bridge_against_real_policy.py` | real `PolicyServer` | `grootN1_Robotics` |
| `DexVLA_Robotics/tests/test_ros_bridge_against_real_divla.py` | real DiVLA over ZMQ | `DexVLA_Robotics` |

The corollary learned the hard way: running against the real peer also means sending the
real **input**. The bridge had been checked against the live GR00T server four times, but
always with an observation built by a test helper — never the one the node itself builds.

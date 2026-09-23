# ROS2 bridge — plan

| item | note |
|---|---|
| **container mount** | the world model runs live, but `pipeline/` is staged in by hand. A permanent setup needs one line: `- /Users/trish/VLAProjects/pipeline:/ws/src/pipeline:ro` |
| **live `/vla/joint_trajectory`** | the bridge is verified against a real GR00T `PolicyServer`, but no robot or sim consumes the topic end to end |
| **latency under load** | the ZMQ round trip was moved off the callback thread and measured against a 1.5 s server; not profiled with real perception running alongside |

Cross-cutting gaps are in the root [Plan.md](../Plan.md).

# Object-centric RGB-D perception stack

RGB-D frames in, an action chunk out. Camera localization, open-vocabulary instance
segmentation and 6-DoF object pose fused into a **hierarchical semantic scene graph**,
exposed through ROS2/TF2 and read by a VLA policy.

![end-to-end run](assets/e2e_demo.gif)

*Full resolution: [`e2e_demo.mp4`](assets/e2e_demo.mp4) · still:
[`e2e_demo.png`](assets/e2e_demo.png). Rendered from the run's own artifacts by
[`tools/render_demo.py`](tools/render_demo.py) — not a screen capture.*

One run on real TUM `fr1/xyz`, CPU only. Every panel is live the whole time and the
*state* moves — this stack's claim is that the stages feed each other, so showing them as
six separate pictures would misrepresent it:

| panel | shows |
|---|---|
| RGB + depth | the raw stream everything below is derived from |
| **3-D world model** (orbiting) | the cloud accumulating under a flying camera frustum, then instance-coloured points, then the target's box |
| points-fused sparkline | fusion progress, 0 → 319,570 unprojected points |
| scene graph | instances resolving with kind and CLIP similarity, then the `on`/`near` relations |
| prompt to policy | the exact string stage 5 serialises and stage 6 consumes |
| GR00T chunk | 16 × 29 DoF, with the execution cursor advancing at 30 Hz |

The stage banner tracks 1–2 → 3–4 → 5 → 6 as the run progresses. The 3-D view orbits
continuously, which is what makes a point cloud readable as geometry rather than a smear,
and it is drawn at true room proportions (4.90 × 3.71 × 1.56 m) rather than a cube.

Only the **target** gets a wireframe box. Six nested proposal boxes mostly showed how
much the proposals overlap, which on this scene is a lot and read as noise; colouring the
cloud points by instance shows the segmentation itself.

```
RGB-D ──▶ ORB-SLAM3 / DROID-SLAM ──▶ TSDF fusion ──▶ OpenMask3D ──▶ FoundationPose
                                                            │              │
                                                            ▼              ▼
                                                     scene graph ◀── 6-DoF pose
                                                            │
                                              ROS2 / TF2 ───┴──▶ GR00T N1.6 / DexVLA
```

| stage | component | measured |
|---|---|---|
| localize | ORB-SLAM3 (upstream `4452a3c`, 0 patches) | ATE **1.03 cm**, 798/798 frames, 41.4 FPS |
| localize (dynamic) | + YOLO/ByteTrack feature rejection | ATE 80.92 → **18.70 cm** (−76.9%) on `fr3/walking_xyz` |
| fuse | TSDF, keyframe-anchored | 70,236 points |
| segment | OpenMask3D — Mask3D + SAM + CLIP | **7 proposals → 6 instances**, scores 0.556–0.943 |
| pose | FoundationPose | **CUDA-gated**, never executed — see below |
| ground | scene graph → target | 2 relations incl. **`the monitor is on the desk`** → `monitor_6` |
| act | GR00T N1.6-3B | **16 steps × 29 DoF**, 3.0 s CPU |

**→ [Full numbers, ablations and what is *not* established](../RESULTS.md)**

## ATE understates world-model error by ~7×

The stack's central composition is `T_anchor_obj = T_anchor_cam @ T_cam_obj`, and the
camera pose on the left is an estimate. Fusing a perfect detection (synthesised from TUM
ground truth) with the ORB-SLAM3 estimate, through the real `refine_with_pose`, over 2172
observations in the gravity-levelled frame:

| | value |
|---|---|
| camera ATE RMSE | 1.03 cm |
| **fused object placement error** | **7.30 cm mean**, 12.30 cm p95, 17.24 cm max |
| attributable to camera *rotation* (2.29°) | ~100% |
| attributable to camera *translation* | ~0.9 cm |
| improvement from averaging 316–690 views | **~2%** |

Two things follow. The error is **rotational**, amplified by object range — 4.53 cm at
1.22 m, 9.21 cm at 3.36 m, bounded above by `d·θ` — so a stack tuned on ATE alone is
tuned on the term contributing 12% of its own error. And the drift is **systematic**:
averaging hundreds of views removes ~2%, so multi-view fusion is not a way out.

This is the measured argument for anchoring object frames to keyframes rather than to
`map`: a keyframe-anchored object inherits the rotational correction when bundle
adjustment moves its keyframe.

```bash
conda run -n foundationpose_vl python pipeline/tools/validate_pose_fusion.py
```

## The honest limit

`nvdiffrast` is CUDA-only, so **FoundationPose never ran** and no pose refinement is in
this result. Object poses are position-only with identity rotation: a segmentation mask
carries no orientation, and `observations.from_openmask3d` refuses to invent one rather
than hand a planner an authoritative-looking wrong pose. Every stage consumes the
previous stage's real output; the chain does not yet produce 6-DoF grasps.

**Recognition is the weak stage, and it did not improve.** Segmentation got much better
once the frame was levelled (scores 0.556–0.943), but CLIP labelled **6 of 7 instances
"a desk"**, similarities 0.198–0.239, with margins over the runner-up as thin as 0.198 vs
0.182. One 64-point, 7 cm-tall instance was also called a desk. That is close to a
constant prediction, so `the monitor is on the desk` is geometrically sound while its
*labels* rest on a 0.198 cosine similarity. These are open-vocabulary similarities, not
calibrated confidences; the mIoU/recall that would settle it needs ground-truth instance
annotations and is not measured.

## Live in a ROS2 graph

`world_model_node` runs as a real process in the ROS2 Jazzy container, fed the run's
instances over DDS by a separate producer node. The TF tree it publishes:

```
map
└─ keyframe/0
   ├─ object/desk_1 … object/desk_5
   └─ object/monitor_6
      └─ target/pregrasp
```

Objects are children of **`keyframe/0`, not `map`** — the design claim above, confirmed
on the wire — and `tf2_echo keyframe/0 target/pregrasp` resolves, so the chain composes
rather than merely publishing well-formed messages.

Running it live is what caught the pre-grasp bug: `approach * standoff` measured from the
object's **origin**, so a 0.12 m standoff resolved 0.35 m *inside* the 0.94 m monitor.
Standoff is now taken from the surface via the box support function, and `target/pregrasp`
moved from z = 2.084 to z = 2.554. See [RESULTS.md §4.3.1](../RESULTS.md).

## Three design decisions

**TF2 owns geometry; the graph owns semantics.** TF2 is a transform *tree* — `on`,
`inside` and `near` have no inverse transform and cannot live in it. Nodes *name* their
TF frame rather than owning a pose.

**Object frames are anchored to keyframes, not to `map`.** When bundle adjustment moves
a keyframe, TF recomposes and every object follows the correction. Publishing
`map → object` bakes in the pre-optimisation estimate, and objects slide off their
supports after loop closure while every transform still looks well-formed.

**One clock, one frame authority, one identity** ([`identity.py`](identity.py)).
Nanoseconds are canonical because float64 cannot hold ns at epoch scale — 2⁵³ ≈ 9e15
against 1.3e18 ns — so a stage that ingests float seconds is lossy exactly once, at
ingestion. `InstanceRegistry` mints ids by IoU + label rather than by proposal rank;
under rank ordering, re-running segmentation renamed every instance and every TF frame,
graph node and planner target silently pointed elsewhere. It also merges duplicates: two
desk proposals overlapping at IoU 0.498 collapse into one instance.

## Running it

Stages hand off through JSON rather than sharing a process: they need different
environments, and two OpenMP runtimes in one process abort.

```bash
# stages 1-2  SLAM + gravity-levelled TSDF fusion
conda run -n foundationpose_vl python pipeline/tools/e2e_tum.py
# stage 3     Mask3D proposals
conda run -n openmask3d_vl    python pipeline/tools/e2e_segment.py
# stage 4     SAM + CLIP open-vocabulary labels
conda run -n openmask3d_vl    python pipeline/tools/e2e_label.py
# stage 5     scene graph, relations, grounding
conda run -n foundationpose_vl python pipeline/tools/e2e_ground.py --query "the monitor"
# stage 6     GR00T, driven by the graph's own prompt
PYTHONPATH=grootN1_Robotics/upstream:. conda run -n groot_vl python \
    grootN1_Robotics/tools/run_policy_cpu.py \
    --checkpoint grootN1_Robotics/checkpoints/GR00T-N1.6-3B \
    --dataset grootN1_Robotics/upstream/demo_data/gr1.PickNPlace \
    --task "$(python -c "import json;print(json.load(open('pipeline/assets/e2e/ground.json'))['prompt'])")" \
    --save-json pipeline/assets/e2e/policy.json
```

```bash
# what SLAM error does to a fused object pose
conda run -n foundationpose_vl python pipeline/tools/validate_pose_fusion.py
# visualise: time-aligned Rerun replay, then the README media
conda run -n foundationpose_vl python pipeline/tools/replay_rerun.py
conda run -n foundationpose_vl python pipeline/tools/render_demo.py
```

`pipeline/tests` — 117 tests, no ROS and no GPU required.

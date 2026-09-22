# VLA Robotics Project

An **object-centric RGB-D perception stack for embodied AI** — camera localization, open-vocabulary
instance segmentation and 6-DoF object pose fused into a hierarchical semantic world model,
exposed through ROS2/TF2 and read by a VLA policy for action chunking.

![end-to-end run](pipeline/assets/e2e_demo.gif)

*One run on TUM `freiburg3_walking_xyz` — a cluttered office with people walking through it.
[Full resolution](pipeline/assets/e2e_demo.mp4) · [the stack in detail](pipeline/)*

CUDA-first **adaptations**, not reimplementations: upstream stays pinned and unvendored, our
edits live in each project's `patches/`, and the rule for every project is **load the official
weights at 0 missing / 0 unexpected before anything is changed.** An adaptation that was never
checked against the original is not measurable.

> **Sibling repo — [VLM-AD-Project](https://github.com/Trish32/VLM-AD-Project)** — pure-PyTorch
> ports of BEVFormer, BEVFusion, FlashOcc, Simple-BEV, Sparse4D v2/v3, QCNet, a closed-loop
> KBM simulator and **DiffusionDrive**, all running on Apple Silicon (MPS) without
> `mmcv`/`mmdet3d`/`spconv`. This repo is the deliberate inverse: CUDA-first, and it *adapts*
> upstream rather than reimplementing it.

---

## The stack

```
RGB-D ──▶ ORB-SLAM3 / DROID-SLAM ──▶ TSDF fusion ──▶ OpenMask3D ──▶ FoundationPose
                                                            │              │
                                                            ▼              ▼
                                                     scene graph ◀── 6-DoF pose
                                                            │
                                              ROS2 / TF2 ───┴──▶ GR00T N1.6 / DexVLA
```

| stage | measured |
|---|---|
| **localize** — ORB-SLAM3 (upstream `4452a3c`, 0 patches) | **ATE 1.03 cm**, 798/798 frames, 41.4 FPS CPU |
| **localize, dynamic scene** — + YOLOv8n/ByteTrack rejection | **ATE 80.92 → 18.70 cm (−76.9%)** on `fr3/walking_xyz` |
| **fuse** — TSDF, gravity-levelled | 133,928 points, 6.3 × 4.0 × 2.2 m |
| **segment** — Mask3D + SAM + CLIP | checkpoint **0/0/0** (469 tensors, 39.66 M) via a pure-PyTorch sparse conv |
| **pose** — FoundationPose | both checkpoints **0/0/0**; scorer 15.77 M + refiner 16.83 M **construct on a T4** |
| **ground** — scene graph → target | `"the chair"` → `chair_4`, sim 0.286, margin 0.050 |
| **act** — GR00T N1.6-3B | **16 steps × 29 DoF**, all finite, 4.4 s CPU |

**→ [Demo, full numbers, and what is *not* established](pipeline/)**
**→ [RESULTS.md](RESULTS.md)** — every measurement, labelled MEASURED / EXACT / MODELLED

### Results worth naming

**MinkowskiEngine replaced with a pure-PyTorch sparse convolution**, verified two independent
ways: dense-grid equivalence to `nn.Conv3d` at **atol 1e-10**, and kernel-offset enumeration
checked against upstream's own `kernel_region.hpp` compiled `-DCPU_ONLY` — **195 offsets, exact**.
The dense oracle alone cannot catch an ordering error, because it builds its reference from the
same offsets under test.

**ATE understates world-model error by ~7×.** Over 2,172 fused observations, a 1.03 cm camera
trajectory places objects **7.30 cm** from truth. Decomposition shows camera **rotation (2.29°)**
accounts for essentially all of it; translation contributes ~0.9 cm, and averaging over 316–690
views removes only **~2%** — the drift is systematic.

**The world frame was never gravity-aligned.** SLAM returns poses in the first keyframe's camera
frame; the supporting plane's normal sat **50.1° off** the axis every consumer treated as vertical.
Levelling it, then re-segmenting, took Mask3D scores from "low" to **0.556–0.943** and produced the
pipeline's first `on` relation.

---

## Projects

### [Object-centric RGB-D perception stack](pipeline/)

The stack above. Runs end to end on real TUM RGB-D on CPU. FoundationPose's model now loads and
constructs on a T4; `register()` on our own TSDF-derived mesh is queued, and the project page says
so rather than implying a pose result exists.

### [DexVLA](DexVLA_Robotics/) — VLM with a plug-in diffusion expert

Qwen2-VL-2B with a ScaleDP transformer diffusion action head, plus a reasoning channel that
conditions the denoiser on the model's own emitted reasoning. The released Stage-1 head loads
0-unexpected against upstream's class, and the assembled 3.18 B VLA runs end to end on CPU and
over ZMQ from the ROS2 bridge.

**This one is not a reproduction, and says so.** Only Stage 1 is public — there is no Stage-2/3
checkpoint — and upstream's only evaluation entry point drives a real AgileX robot, with no sim
harness of any kind. So the 0/0 + reproduce-the-metric bar is unreachable here no matter how much
is ported. The deliverable is a correct architecture, fine-tuned from the released Stage-1 head,
measured against our own GR00T baseline under an identical budget.

**→ [What is achievable, the architecture, and the free-GPU budget](DexVLA_Robotics/)**

---

## Roadmap

| Component | Status | Validation target |
|---|---|---|
| **Perception stack** — SLAM → graph → VLA | **runs end to end**, stages 1–6 | ORB-SLAM3 ATE 1.03 cm reproduced |
| **OpenMask3D** — sparse conv + Mask3D | **0/0/0**, 1e-10 vs `nn.Conv3d` | mIoU / open-vocab recall — needs ScanNet200 GT |
| **FoundationPose** | **0/0/0**, constructs on T4 | `register()` on a TSDF-derived mesh — queued |
| GR00T N1.6-3B | checkpoint loads 0/0/0 (3.29 B params) | LIBERO / SimplerEnv success rate |
| DexVLA | ScaleDP-H Stage-1 head loads 0-unexpected | **Stage 1 only, real-robot eval** — controlled baseline, not a reproduction |
| DROID-SLAM | `droid.pth` loads 0/0/0 | **never executed** — `lietorch`/`droid_backends` are CUDA-compile-only |
| ROS2 bridge | wire format byte-identical to upstream | live in a ROS2 Jazzy graph ✓ |

Every SLAM number in this repo is **ORB-SLAM3**. DROID-SLAM is integrated and its checkpoint
loads, but it has never run — that distinction is kept explicit rather than folded into a
"DROID-SLAM/ORB-SLAM3" credit.

## License

Our adaptation layer is MIT. Upstream ORB-SLAM3, OpenMask3D/Mask3D, FoundationPose, GR00T,
DiffusionVLA and DROID-SLAM remain under their own licenses; none of their source is vendored here.

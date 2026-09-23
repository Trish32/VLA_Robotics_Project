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
                                                            │              │
                                                            ▼              ▼
                                        action-conditioned world model ◀── K candidates
                                                            │
                                                  simulate N steps, score, execute one
```

| stage | measured |
|---|---|
| **localize** — ORB-SLAM3 | **ATE 1.03 cm**, 798/798 frames, 41.4 FPS CPU |
| **localize, dynamic** — + YOLOv8n/ByteTrack | **ATE 80.92 → 18.70 cm (−76.9%)** |
| **segment** — Mask3D + SAM + CLIP | checkpoint **0/0/0**, via a pure-PyTorch sparse conv verified at **1e-10** |
| **pose** — FoundationPose | `register()` + tracking on a T4; **2.08 cm** self-consistency, but **175.63° rotation error** vs our map — gated out, not claimed |
| **act** — GR00T N1.6-3B | **16 steps × 29 DoF**, 3.1 s CPU |
| **simulate** — action-conditioned world model | dynamics beats identity (0.0554) and constant velocity (0.0262) at **0.0202–0.0258** on held-out proprio; **not** validated on objects |

**→ [The stack, the demo, and the honest limit](pipeline/)**
**→ [RESULTS.md](RESULTS.md)** — every measurement, labelled MEASURED / EXACT / MODELLED
**→ [Plan.md](Plan.md)** — what is *not* done, why, and what would close it

---

## Projects

### [Object-centric RGB-D perception stack](pipeline/)

The stack above. Runs end to end on real TUM RGB-D on CPU. FoundationPose registers correctly on
upstream's own demo data (**2.06 cm**), so the port is validated — but on a mesh cut from our TSDF
its pose is **175.63°** off, so stage 4 gates it out and the chain stays position-only. The limit
is our 695 px mask over a one-sided shell, and the project page leads with that.

### [DexVLA](DexVLA_Robotics/) — VLM with a plug-in diffusion expert

Qwen2-VL-2B with a ScaleDP transformer diffusion action head, plus a reasoning channel that
conditions the denoiser on the model's own emitted reasoning. The released Stage-1 head loads
0-unexpected against upstream's class, and the assembled 3.18 B VLA runs end to end on CPU and
over ZMQ from the ROS2 bridge.

---

## Roadmap

| Component | Status | Validation target |
|---|---|---|
| **Perception stack** — SLAM → graph → VLA | **runs end to end**, stages 1–6 | ORB-SLAM3 ATE 1.03 cm reproduced |
| **OpenMask3D** — sparse conv + Mask3D | **0/0/0**, 1e-10 vs `nn.Conv3d` | mIoU / open-vocab recall — needs ScanNet200 GT |
| **FoundationPose** | **0/0/0**; correct pose on upstream `demo_data/mustard0` (**2.06 cm**) | pose on OUR mesh **refused** by the stage-4 gate (175.63°) — the input is the limit, not the port |
| GR00T N1.6-3B | checkpoint loads 0/0/0 (3.29 B params) | LIBERO / SimplerEnv success rate |
| DexVLA | ScaleDP-H Stage-1 head loads 0-unexpected | **Stage 1 only, real-robot eval** — controlled baseline, not a reproduction |
| DROID-SLAM | `droid.pth` loads 0/0/0 | **never executed** — `lietorch`/`droid_backends` are CUDA-compile-only |
| ROS2 bridge | wire format byte-identical to upstream | live in a ROS2 Jazzy graph ✓ |


## License

Our adaptation layer is MIT. Upstream ORB-SLAM3, OpenMask3D/Mask3D, FoundationPose, GR00T,
DiffusionVLA and DROID-SLAM remain under their own licenses; none of their source is vendored here.

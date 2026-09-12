# VLA Robotics Project

CUDA-first **adaptations** of end-to-end driving and robotics policies — DiffusionDrive,
GR00T N1.6, DiffusionVLA, DROID-SLAM — each validated against its official checkpoint
before anything is changed.

The rule for every project here: **load the official weights at 0 missing / 0 unexpected and
reproduce the published metric first.** An adaptation that was never checked against the
original is not measurable.

Upstream stays pinned and unvendored; our edits live in each project's `patches/` so upstream
can be re-pulled and diffed. Each project keeps its own `README.md` and `bug_log.txt`.

> **Sibling repo — [VLM-AD-Project](https://github.com/Trish32/VLM-AD-Project)** — pure-PyTorch
> ports of BEVFormer, BEVFusion, FlashOcc, Simple-BEV, Sparse4D v2/v3, QCNet and a closed-loop
> KBM simulator, all running on Apple Silicon (MPS) without `mmcv`/`mmdet3d`/`spconv`.
> This repo is the deliberate inverse: CUDA-first, and it *adapts* upstream rather than
> reimplementing it. The two are directly connected — DiffusionDrive's `nusc` branch is
> SparseDrive plus one file, and **SparseDrive is already ported and metric-validated**
> (mAP 0.463 / NDS 0.480) in
> [`sparse4d_vldrive`](https://github.com/Trish32/VLM-AD-Project/tree/main/sparse4d_vldrive).

---

## Projects

### [DiffusionDrive](diffusiondrive_planner/) — truncated diffusion planner (CVPR 2025 Highlight)

Denoises a driving trajectory in **2 steps**, seeded from anchor trajectories plus a little
noise, rather than ~100 steps from Gaussian noise. Reproduced on nuScenes mini_val against
upstream's own `PlanningMetric`, with the official checkpoint loading 0 missing / 0 unexpected.

**→ [Visualization, metric results and the full fidelity chain](diffusiondrive_planner/)**

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

| Project | Status | Validation target |
|---|---|---|
| **DiffusionDrive** — nuScenes | **done** — reproduced | published L2 avg 0.57 |
| DiffusionDrive — NAVSIM | next | 88.1 PDMS on navtest; 60M/ResNet-34, the one model that trains end-to-end on a 16 GB T4 |
| GR00T N1.6-3B | checkpoint loads 0/0/0 (3.29 B params) | LIBERO / SimplerEnv success rate |
| DexVLA | ScaleDP-H Stage-1 head loads 0-unexpected | **only Stage 1 is released and eval is real-robot only** — controlled baseline against GR00T, not a reproduction |
| DROID-SLAM | `droid.pth` loads 0/0/0 | ATE on TUM-RGBD monocular |
| ROS2 bridge | built, wire format byte-identical to upstream | live `/vla/joint_trajectory` |

Projects land in this repo as they clear the 0/0 + reproduce-the-metric bar.

DiffusionDrive is first because it is the cheapest. Its `nusc` branch is SparseDrive plus
essentially one new file — `motion_planning_head_v13.py`, the truncated-diffusion planner that
replaces the regression planner. Everything upstream of it (sparse perception, instance bank,
motion head) is SparseDrive, which is
**[already ported and metric-validated](https://github.com/Trish32/VLM-AD-Project/tree/main/sparse4d_vldrive)**
in the sibling repo. That makes it a planner-head swap on working code rather than a
from-scratch port.

## License

Our adaptation layer is MIT. Upstream DiffusionDrive, SparseDrive, GR00T, DiffusionVLA and
DROID-SLAM remain under their own licenses; none of their source is vendored here.

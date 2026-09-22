# Plan

What is **not** done, why, and what would close it. Measured results live in
[RESULTS.md](RESULTS.md); this page is its complement — the same discipline applied to
the gaps, so neither page has to hedge.

An item leaves this page by being measured, not by being argued.

---

## 1. Blocked on a resource, not on effort

These have a known route and a cost. Nothing here is a design question.

| item | blocker | what closes it |
|---|---|---|
| **OpenMask3D mIoU / open-vocab recall** | ScanNet200 is a gated ~1 TB download | ground-truth instance labels on any scene the stack can fuse; ScanNet is the standard but not the only option |
| **MobileSAM precision cost** | needs the above | the `vit_t` speedup is the one acceleration step that changes outputs, so it must be measured against `vit_h` on the same scene, not assumed free |
| **Wall-clock speedup, in seconds** | modelled only, never timed | a CUDA box running the real weights. Until then the acceleration numbers stay **MODELLED** and are never reported in seconds |
| **DROID-SLAM tracking** | `lietorch` / `droid_backends` are CUDA-compile-only | a GPU box. The checkpoint already loads 0/0/0; nothing else has run |
| **Monocular tracking-loss reduction** | RGB-D loses no frames on these sequences | the monocular variant — the experiment is ill-posed on RGB-D, where both baseline and filtered track 827/827 |

**Every SLAM number in this repo is ORB-SLAM3.** DROID-SLAM is integrated and its
checkpoint loads, but it has never executed. That distinction is kept explicit rather
than folded into a combined credit.

---

## 2. FoundationPose: from "loads" to "poses"

The furthest-along gap, and the one with the most moving parts already in place.

**Done:** nvdiffrast builds and rasterises on a T4; the full import chain resolves;
both official checkpoints load and construct on `cuda:0` (scorer 15.77 M, refiner
16.83 M). The mesh + mask bundle is built from our own TSDF — 9,657 triangles cut from
1,142 instance points, 8 frames with occlusion-tested masks.

**Not done:** `register()` has never returned a pose.

**How it gets judged when it does.** Two checks, neither of which needs ground truth:

1. **Agreement** — `register()`'s translation against what our own segmentation believed
   the centroid was. These are independent estimates; agreement is evidence, not proof,
   and disagreement does not say which is wrong.
2. **World-frame spread** — the tracked pose composed with each camera pose. A static
   object fused into a world frame must not move. This is the invariant that actually
   discriminates, because a drifting tracker cannot satisfy it by luck.

**Known weakness going in.** The `chair_4` instance is 1.58 × 1.18 × 0.99 m — a coarse
region, not a clean chair — and its masks are 655–954 px. A large spread would more
likely indict the mesh than the tracker, so a bad result needs the mesh ruled out before
it means anything about FoundationPose.

---

## 3. Recognition is the weak stage

Segmentation improved sharply once the world frame was levelled (Mask3D scores
0.556–0.943). **Recognition did not.**

| | measured |
|---|---|
| median top-1 CLIP margin | **0.011** |
| worst case | `table` 0.264 vs `desk` 0.263 — **0.001** |
| best grounding so far | `the chair` → `chair_4`, 0.286, margin **0.050** |

At a 0.011 margin an argmax over a 16-word vocabulary is close to arbitrary. Two things
follow, and only the second is a fix:

- **Prompt-template ensembling was not the answer.** It raised absolute similarities
  (0.198–0.239 → 0.21–0.28) and flipped one instance from `monitor` to `person`. Better
  numbers, no better decisions — recorded because it is the kind of change that looks
  like progress.
- **Grounding no longer depends on the argmax.** `e2e_ground.py` ranks the saved
  per-instance CLIP features against the query, so a lossy 16-way winner is out of the
  critical path. That is what makes the vocabulary genuinely open, and it is why
  `resolve("the monitor")` stopped raising `KeyError` on a graph that demonstrably held
  monitor-like evidence.

**What would actually close it:** mIoU against ground truth (§1), which would say
whether the crops or the vocabulary are at fault. Until then no claim about label
accuracy is supportable, and none is made.

---

## 4. Scene graph: `on` works, the rest is untested on real data

`the monitor is on the desk` and `the chair is inside the desk` both fire now. But
`the chair is inside the person` also fires, and that is not a real spatial claim — it
is an artifact of axis-aligned boxes over heavily overlapping proposals.

| | state |
|---|---|
| `on` | fires on real data, after gravity alignment |
| `inside` | fires, but produces at least one nonsense edge from AABB overlap |
| `near` | fires |
| `part_of` / place hierarchy | unit-tested only; no real scene has places |

The honest fix for `inside` is a support test that fits a plane to the instance rather
than trusting its bounding box. **Deliberately not implemented**: with no ground-truth
instance labels there is no way to check whether the relations it produced were
*correct*, and an `inside` edge a planner acts on is worse wrong than absent. It waits
on §1 too.

---

## 5. Loose ends

| item | note |
|---|---|
| **`to_scene_nodes` surface vocabulary** | a hardcoded six-word set (`table, floor, shelf, counter, desk, wall`) decides `SURFACE` vs `OBJECT` inside an open-vocabulary pipeline. It disagrees with the CLIP vocabulary it is fed — `counter` and `shelf` are surfaces never queried; `chair` is queried but is not a surface. Swap the vocabulary and surface detection silently degrades, taking every `on` relation with it |
| **ROS2 container mount** | the world model runs live, but staged into the container by hand. A permanent setup needs one line: `- /Users/trish/VLAProjects/pipeline:/ws/src/pipeline:ro` |
| **Instance density for meshing** | `monitor_4` was 578 points across 1.2 m — roughly 4–5 cm spacing. Fine for a bounding box, thin for a mesh. A denser fusion (smaller `--stride`) is the lever, at TSDF memory cost |
| **Kaggle token** | used across two sessions; rotate it |

---

## 6. Sequencing

Ordered by what unblocks the most, not by effort.

1. **FoundationPose `register()`** — every input exists; this is the last stage of the
   headline claim that has never run.
2. **Ground-truth instance labels** — unblocks mIoU, the MobileSAM trade, the `inside`
   support test, and any statement about recognition. One resource, four gaps.
3. **A CUDA box** — unblocks DROID-SLAM tracking and turns every MODELLED speedup into a
   measured one.
4. **The loose ends in §5** — small, and none of them gate anything above.

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

**Done since:** `register()` and `track_one()` run on a T4 against that mesh —
**2.08 cm world-frame spread** over 8 frames, so the tracker is strongly self-consistent.

**Not done:** the pose disagrees with our map by **72.14 cm**, and the disagreement is a
**175.63° rotation error** — the object is essentially flipped — not a translation one.
See `foundationpose_6dof/bug_log.txt` entry [4].

**What the three checks say.** None needs ground truth:

1. **World-frame spread — passes, and proves less than it looks.** A static object must
   stay static once each tracked pose is composed with its camera pose: **2.08 cm** over
   8 frames. That is a *consistency* check. A tracker locked onto a wrong pose holds it
   exactly as steadily as one locked onto the right pose.
2. **Agreement with the map — fails.** `register()` says z = 2.73 m; the segmentation
   centroid says 2.05 m; median depth under the mask is 2.10 m.
3. **Rotation — fails, and this is the real finding.** Moving the mesh origin a known
   52.9 cm moved FoundationPose's answer 53.5 cm — right magnitude, so the pose
   convention is understood — but **99.5° away** from the direction the map predicts,
   which lower-bounds the rotation error. Measured directly against `R_world_to_cam`:
   **175.63°**. The object is essentially flipped.
4. **The control — run, and it convicts our input.** Upstream's `demo_data/mustard0`
   through the identical code path returns a **correct** pose: origin 2.06 cm behind the
   measured surface, against a 9.6 cm half-depth. The port, the checkpoints and the
   adaptation layer are sound. The difference is the input — 695 px over a one-sided
   Poisson shell, against 3,252 px over a CAD mesh.

   *A metric that failed its own control, and its replacement:* r13 read the scorer's
   flat top (48/252 within 1%) as "orientation unidentifiable". mustard0 is flatter — an
   exact tie, 91/252 — and still correct, because scores are computed after refinement so
   converged hypotheses legitimately tie. r14 measures **pose clustering** instead, and it
   separates the cases cleanly:

   | | `chair_4` | mustard0 |
   |---|---|---|
   | median pairwise angle, top-16 | **127.31°** | **0.29°** |
   | within 15° of top-1 | 2/16 | 16/16 |

   This needs neither our map nor ground truth, so it is also a **pre-flight test**:
   whether an instance is poseable at all, before a registration is spent on it. Now the
   fourth gate check. See `foundationpose_6dof/bug_log.txt` [6].

An earlier version of this page blamed the 72 cm on a frame convention, on the strength
of `reset_object` subtracting the mesh's bbox centre. `estimater.py:233` undoes that
subtraction before returning. The correction is kept rather than quietly edited out
because the failed fix is what produced check 3.

**The mask is the cause, now with the control to back it.** 695 px, where a 1.58 m object
at 2 m under fx = 535 should subtend ~410 px across — a one-sided Poisson shell seen
through a sparse partial view. mustard0 gets 3,252 px over a CAD mesh and lands the pose.
**What would close it:** a denser fusion (smaller `--stride`) for bigger masks, and a
cleaner instance than a 1.58 m coarse proposal — `chair_4` is a region, not an object.

**Wired regardless:** `pipeline/tools/e2e_pose.py` consumes the pose and gates it on
translation, rotation and depth before it reaches the world model. Run against the real
r12 output it refuses **0/8 frames corroborating** — median 73.02 cm, 175.86°, origin
65 cm behind a surface only 13 cm deep — and the position-only pose from stage 3 stands.
22 tests cover the gate.

The depth check originally could not fire: its tolerance was half the instance's 3-D
extent, and `chair_4` spans 1.58 m, so a 60.8 cm error sat inside a 78.9 cm bound. It now
uses the depth the camera actually sees (5th–95th percentile under the mask, a 13.4 cm
slab), which is also what keeps it independent of the segmentation. See `bug_log.txt` [7].

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

## 4. Scene graph: relations now come from geometry, not bounding boxes

The prompt handed to the policy used to read:

> Scene: the chair is inside the desk, **the chair is inside the person**. Target: the chair.

Both edges were artifacts of axis-aligned boxes over heavily overlapping proposals. It
now reads:

> Scene: **the chair is near the desk**. Target: the chair.

**This did not need ground truth**, which is why it no longer waits on §1. Whether a
label is *correct* needs ground truth; whether one instance's points lie inside
another's is a geometric question the data already answers.

| | decided by | state |
|---|---|---|
| `on` | support band + footprint overlap | fires on real data, after gravity alignment |
| `inside` | **convex-hull containment** of the subject's hull vertices | 3 spurious edges removed |
| `near` | **surface separation** (5th-pct nearest-neighbour), not centre distance | recovered a true edge the old rule missed |
| `part_of` / place hierarchy | — | unit-tested only; no real scene has places |

**`inside`.** Testing hull vertices is exact rather than approximate: a convex hull is
the convex combination of its vertices, so if every vertex is inside a convex region,
everything between them is too. Measured on the three pairs the box rule called `inside`:

| pair | hull-vertex containment | verdict at 95% |
|---|---|---|
| chair ⊂ desk_5 | **11.6%** | decisively refuted |
| desk_0 ⊂ desk_5 | 68.1% | refuted |
| chair ⊂ person_2 | **89.9%** | refused, but borderline |

The third is worth being precise about: `person_2` is a 4.1 m over-segmented region that
very nearly does enclose the chair, so 89.9% is a *threshold* decision, not geometry
flatly refuting it. What makes that edge read as nonsense is the **label** — §3's
problem, not this one. The two were previously conflated.

**`near`.** Centre distance asks the wrong question about extended objects: the chair
sits **7 cm** from a desk and **79 cm** from its centre, so the true relation was missed
while nothing replaced it. Box-gap distance is degenerate in the other direction — all
15 pairs in this scene have overlapping boxes, so every pair would be "near". Surface
separation between sampled points is the measure that discriminates: 7 of 15 pairs.

**Cost.** Nodes carry the hull (35–80 vertices, 66–156 planes) plus a voxel-spread
sample capped at 200 points: **4–8 KB per node**, small enough to publish. Nodes built
without point sets keep the old box behaviour, so nothing that does not supply geometry
changes.

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

1. **Explain the ≥99.5° rotation error** — run upstream's `demo_data/mustard0` (queued
   as r12). It is the only test that separates our mesh from the model, and until it runs
   no pose accuracy is claimable. `refine_with_pose` is now wired behind a gate
   (`e2e_pose.py`), so the moment a pose passes, it flows; today it does not.
2. **Ground-truth instance labels** — unblocks mIoU, the MobileSAM trade, the `inside`
   support test, and any statement about recognition. One resource, four gaps.
3. **A CUDA box** — unblocks DROID-SLAM tracking and turns every MODELLED speedup into a
   measured one.
4. **The loose ends in §5** — small, and none of them gate anything above.

# Plan

What is **not** done, why, and what would close it. Measured results live in
[RESULTS.md](RESULTS.md); this page is its complement — the same discipline applied to
the gaps, so neither page has to hedge.

An item leaves this page by being measured, not by being argued.

---

## 0. Status, and which numbers are comparable to which

This page and [RESULTS.md](RESULTS.md) predate the simulator work in
[`pipeline/sim/`](pipeline/sim/), so a reader arriving at either could act on a
superseded conclusion. This table is the single place that says what is current, what
it was measured on, and what it replaces. **Last reconciled 2026-10-04.**

Two distinctions are load-bearing throughout and are never collapsed:

- **REAL** means TUM RGB-D or recorded robot episodes. **SIM** means the MuJoCo
  cube-to-bowl arena. A result on one is never reported as a result on the other.
- A number is comparable to another only if both carry the same **task fingerprint**
  (`loop.task_fingerprint()` — demonstrator constants, waypoint heights, arm geometry,
  arena XML, and step budget). Three separate comparability failures are logged as
  [S20], [S21] and [S22] in `pipeline/bug_log.txt`; in each the numbers were real and
  the comparison was not.

| area | data | current status | supersedes |
|---|---|---|---|
| **SLAM** | REAL | ORB-SLAM3 ATE **1.03 cm**, 41.4 FPS CPU. Dynamic rejection **80.9 → 18.7 cm**, at 38.1 → 18.5 FPS — *below* a 30 Hz loop, so not described as real-time | — |
| **DROID-SLAM** | — | checkpoint loads 0/0/0, **never executed**. Every SLAM number here is ORB-SLAM3 | unchanged |
| **Segmentation** | REAL | checkpoint 0/0/0; sparse conv 1e-10 vs `nn.Conv3d`; 4.8× / 32.4× fewer SAM passes, four of five steps bitwise-identical. **No mIoU** — blocked on ground truth | — |
| **6-DoF pose** | REAL | **FoundationPose has run.** 2.06 cm on upstream demo (port validated), 175.63° on our mesh (input is the limit), stage-4 gate **refuses**, chain is position-only | replaces "never run" in RESULTS.md §nvdiffrast |
| **Object dynamics** | REAL | **ties identity** on two episodes; the model never leaves its initialisation. "+26.2% vs constant velocity" was retracted by `world_model/RESULTS.md` as misleading | — |
| **Object dynamics** | SIM | **+37.4% vs identity / +60.0% vs constant velocity** on 320 episodes. This is *not* evidence about real data and is never quoted as such | — |
| **Candidate selection** | SIM | **headroom +0.0.** Perfect episode-depth selection 99.0% = policy alone 99.0%; a winner exists at 100% of decisions. The heuristic selector's 88.5% is *below* not selecting | replaces every earlier "the scorer ranks badly" reading — see W1, W2, W8, W11 |
| **Commitment** | SIM | fixed-K peaks at 79.2% (K=32) with the two failure curves crossing at **K ≈ 36.5 [24.9, 42.6]**. The episode-commit "curve" is the selector's override rate (r = **−0.955**), not a commitment effect | replaces W12's original framing |
| **Veto** | SIM | **9.1% of decisions are outcome-changing**, 90% of fires inert, conditional breakage flat at ~0.045. The rescue "doubling" is **withdrawn** — median-cell lift +0.054 [−0.044, +0.160] spans zero | replaces the 0.406 headline in EXPERIMENT.md §"What the gate does do" |
| **Veto, stage-restricted** | SIM | **withdrawn (C4).** +5.5 [+2.0, +9.7] on discovery was a best-of-seven; held-out −0.5 [−3.3, +2.2] | — |
| **Runtime signals** | SIM | nothing reaches AUC 0.65; change detectors **anti-aligned** with burst onset at 0.19–0.23. P12 **failed** — two triggers hit 1.32× — so the event-trigger clause is **open**, not closed | replaces "no trigger concentrates the signal" |
| **P11–P15** | SIM | **a robustness check, not a pre-registration.** The claim was pushed before the registration, and every threshold came from data already in hand | audit in `sim/PREREGISTERED.md` |
| **P16–P21** | SIM | **registered, not run.** The `piv*.json` sets were lost with the scratchpad, so running them now means regenerating seeds 0–2 first | — |
| **VLA task success** | — | **does not exist.** GR00T emits 16 × 29 DoF in 3.0 s CPU; no task-success benchmark has been run on any task | — |

**What this implies for sequencing, and it is not what §6 below says.** The simulator
result is "selection cannot help *in an arena where the policy already scores 99%*",
which is a statement about a saturated task. The measurement that would tell us what to
fix next — a real VLA task-success baseline — has never been taken. §6 was written
before that was clear and is ordered around the pose question alone.

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

### Surfaces are detected, not looked up

The last lexical decision in the chain. A hardcoded `{table, floor, shelf, counter,
desk, wall}` deciding what can support something, inside an open-vocabulary pipeline, is
a contradiction: it disagreed with the CLIP vocabulary it was fed, and changing the query
words silently changed which instances could be the *subject* of a relation — taking
every `on` edge with it.

Measured slab areas on `freiburg3_walking_xyz` fall into two groups with nothing between
them, so the threshold is not doing the work:

| instance | label | slab (m²) | lexical | geometric |
|---|---|---|---|---|
| `desk_2` | desk | 1.258 | SURFACE | SURFACE |
| `person_3` | person | 0.960 | OBJECT | **SURFACE** |
| `desk_1` | desk | 0.680 | SURFACE | SURFACE |
| `desk_5` | desk | 0.125 | **SURFACE** | OBJECT |
| `chair_4` | chair | 0.123 | OBJECT | OBJECT |

**This is a robustness fix, not an accuracy claim.** With no ground-truth labels there is
no way to show the geometric answer is more *correct* — only that it does not depend on
wording, which is tested by re-running the same geometry with the labels replaced by
German and by meaningless identifiers. The two disagreements are both about bad
instances: `person_3` is a 4.1 m over-segmented region that genuinely contains table
surface, and `desk_5` is 351 sparse points that support nothing. Callers supplying no
point sets keep the word list, so nothing without geometry changes.

**`near`.** Centre distance asks the wrong question about extended objects: the chair
sits **7 cm** from a desk and **79 cm** from its centre, so the true relation was missed
while nothing replaced it. Box-gap distance is degenerate in the other direction — all
15 pairs in this scene have overlapping boxes, so every pair would be "near". Surface
separation between sampled points is the measure that discriminates: 7 of 15 pairs.

**Cost.** Nodes carry the hull (35–80 vertices, 66–156 planes) plus a voxel-spread
sample capped at 200 points: **4–8 KB per node**, small enough to publish. Nodes built
without point sets keep the old box behaviour, so nothing that does not supply geometry
changes.

### Extents are taken after outlier rejection

An axis-aligned extent is a *maximum* over points, so it is decided by the single
furthest one in each direction — the statistic most sensitive to a stray. That extent
becomes the box published to TF2 **and** the support function the pre-grasp standoff is
measured from, so a flier does not merely look wrong, it moves the frame a planner
reaches to.

Rejecting points whose local neighbourhood is anomalously sparse (k = 20, 2σ) drops
**0.4–3.6%** of each instance and shrinks volumes to **0.52–0.89×**:

| instance | raw extent (m) | after rejection | pre-grasp shift |
|---|---|---|---|
| `chair_4` (the target) | 1.58 × 1.18 × **0.99** | 1.58 × 1.11 × **0.59** | **−20.1 cm** |
| `desk_1` | 2.27 × 1.65 × 1.19 | 2.00 × 1.26 × 1.10 | −4.5 cm |
| `desk_2` | 2.71 × 2.19 × 0.62 | 2.23 × 1.52 × 0.57 | −2.6 cm |

**13 stray points out of 1,142** were adding 40 cm to the chair's height and pushing its
pre-grasp frame a fifth of a metre too far back. Density-based rather than a percentile
trim, which would discard a fixed fraction whether or not anything is wrong. A guard
keeps the instance whole if the rule would ever reject more than half of it: a bad extent
is visible and recoverable, a silently truncated instance is neither.

---

## 5. Loose ends

| item | note |
|---|---|
| ~~**`to_scene_nodes` surface vocabulary**~~ | **Done.** `SURFACE` vs `OBJECT` is now decided by whether the instance has a horizontal slab of at least 0.25 m² — roughly a 50 × 50 cm patch, the smallest area that usefully supports something. Vocabulary-independent by construction and tested as such. See below |
| **ROS2 container mount** | the world model runs live, but staged into the container by hand. A permanent setup needs one line: `- /Users/trish/VLAProjects/pipeline:/ws/src/pipeline:ro` |
| **Instance density for meshing** | `monitor_4` was 578 points across 1.2 m — roughly 4–5 cm spacing. Fine for a bounding box, thin for a mesh. A denser fusion (smaller `--stride`) is the lever, at TSDF memory cost |
| **Kaggle token** | used across two sessions; rotate it |

---

## 6. Sequencing

> **Scope note, 2026-10-04.** The order below unblocks the *perception* line and is
> still right for it. It predates the simulator work and so does not contain the
> measurement that now gates everything else — a **real VLA task-success baseline**,
> which has never been run on any task. Treat §6 as the perception track of a two-track
> plan, not as the whole order. §0 has the reconciliation.

Ordered by what unblocks the most, not by effort.

1. ~~**Explain the ≥99.5° rotation error** — run upstream's `demo_data/mustard0`.~~
   **DONE, 2026-09-28, and it answered.** Upstream's own demo registers at **2.06 cm**
   through the same code path, while our mesh gives **175.63°** and 72 cm against the
   map. That separates mesh from model: the port is validated, the input is the limit.
   `refine_with_pose` is wired behind the stage-4 gate (`e2e_pose.py`) and the gate
   **refuses** on today's numbers, so the chain stays position-only. The successor item
   is not "explain the error" but **"improve the mask and mesh, then re-run"** — see the
   status table at the top of this file.
2. **Ground-truth instance labels** — unblocks mIoU, the MobileSAM trade, the `inside`
   support test, and any statement about recognition. One resource, four gaps.
3. **A CUDA box** — unblocks DROID-SLAM tracking and turns every MODELLED speedup into a
   measured one.
4. **The loose ends in §5** — small, and none of them gate anything above.

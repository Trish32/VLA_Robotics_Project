# FoundationPose — plan

`register()` and `track_one()` now run on a T4 against a mesh cut from our own TSDF — see
[RESULTS.md](RESULTS.md). What remains is explaining one number.

## The binding gap: a ≥99.5° rotation error

`register()` places `chair_4` at z = 2.73 m; our segmentation centroid says 2.05 m;
median depth under the mask says 2.10 m. The tracker looked self-consistent (**2.08 cm**
world-frame spread over 8 frames), but those frames spanned 3.4 cm and 1.1°, and over a
real 20.8 cm baseline the spread is **33.17 cm** — withdrawn.

**What the 72 cm actually is.** Moving the mesh origin a known 52.9 cm moved
FoundationPose's answer 53.5 cm, but **99.5° off** the direction our map predicts — same
offset vector, one rotated by the estimate, one by the true camera rotation. So the
orientation is essentially undetermined and the translation residual is its shadow. See
`bug_log.txt` [4], including the wrong diagnosis that produced this measurement.

**The hypothesis was that the mask is the cause: 695 px, where a 1.58 m object at 2 m
under fx = 535 should subtend ~410 px across.** Three tests were queued against it. Both
that have run have answered, and the second answered against the stated cause.

| test | status |
|---|---|
| run upstream's own `demo_data/mustard0` | **DONE 2026-09-28 — correct pose, 2.06 cm.** The port is validated; the error belongs to our input. `bug_log.txt` [5] |
| ~~denser fusion (smaller `--stride`) → bigger masks~~ | **REFUTED 2026-10-04, and never needed.** The fusion was never the limit — on a frame where the chair is in view, 73% of its points agree with measured depth to a −1.1 cm median. The 695 px came from the bundle keeping the first 8 frames clearing a 4.4% floor, which showed 11–17% of the object. `bug_log.txt` [8] |
| a cleaner instance (higher Mask3D score, tighter extent) | **DONE 2026-10-04, and it closes the lever.** `reject_outliers` now runs before meshing (14 strays, 0.99 → 0.59 m tall). `pipeline/tools/survey_instances.py` then scored all 5 labelled instances: `chair_4` is best on every column — 0.74 visible, −0.7 cm residual, **93 frames over the bar** against the runner-up's 13. Two of the five are not poseable at all. There is no cleaner instance in this scene to switch to |

Rebuilt on measured visibility, with a depth-verified dense mask replacing the point
splat: **mask 842 → 6,908 px** (mustard0, the working control, is 3,252), **camera
baseline 3.4 → 20.8 cm**, **instance corroborated 11–17% → 64–74%** — all on the same
fusion and the same mesh pipeline.

**Re-run on a T4 (v15): still refused, and that is the finding.** Rotation moved
175.63° → 124.27°, but the top-16 hypotheses still scatter (1/16 within 15°, median
pairwise 134°) while mustard0 in the same job converges 16/16. The input defects were
real and not binding. The 2.08 cm self-consistency is withdrawn: 33.17 cm over a real
baseline. [RESULTS.md](RESULTS.md) has the table.

**Next: separate mesh from object** — running as kernel v16. mustard0 turned out to ship
**no annotated poses**, so the reference is the CAD mesh's own `register()` on frame 0:
the validated pose (origin 2.06 cm behind the surface, 16/16 hypotheses agree), not
ground truth. The test mesh is built by `mesh_control.py` with the chair's recipe matched
*relative to object size* — ~79 TSDF voxels across, 1,128 points, the same normal radius —
and the chair's own `reject_outliers` and `poisson_mesh`, shipped into the job verbatim.
Two arms: **wide** (every tracked frame, the robot turns the bottle) and **one-sided**
(views within 6° of frame 0, as the chair had). Same frame, same upstream mask, same
estimator; only the mesh differs.

### Decision rule — written while v16 runs, before any of its output was read

Committed locally before the job finished, so the order is checkable. Thresholds are the
stage-4 gate's own (30°) and r14's clustering bar (≥ 8/16 within 15°), not new ones.
Rotation is judged **allowing the bottle's 2-fold flip about its long axis**; the raw
number is reported beside it.

An arm **passes** if rotation (mod flip) ≤ 30° **and** ≥ 8/16 hypotheses cluster within 15°.

| wide | one-sided | reading | next move |
|---|---|---|---|
| fail | fail | the mesh recipe itself (fusion → ~1k points → Poisson) is the limit, whatever the coverage | fix the mesh pipeline; the chair is not yet indicted |
| pass | fail | one-sidedness is the limit, not the recipe | more view coverage of the target, not a better mesher |
| pass | pass | our mesh pipeline is exonerated on this object | the chair is the problem — change target object |
| fail | pass | tracking drift smeared the wide fusion (the SLAM-drift analogue) | report as such; it does not exonerate the recipe |

**Void conditions** — the run says nothing about meshes if: the CAD track fails (fused
mesh extents more than 50% off the CAD extents), or the one-sided arm has fewer than 5
frames, or either arm's mesh fails to build. These are checked before the table is read.

### Result (v16b) — read against the rule above, which was not edited

**First, the reference.** v16's first attempt voided on a layout guard (737 rgb, 1,332
depth, 1 mask), which exposed that the CAD control had paired frame 0's depth by sorted
index. v16b pairs by name and printed: **same file**. The 2.06 cm validation stands.
`bug_log.txt` [11].

**Then the void conditions, literally.**

| | fused extents vs CAD [9.7, 6.7, 19.1] cm | frames | mesh built | void? |
|---|---|---|---|---|
| wide | +8 / +19 / +8 % | 150, over 76.4° | 3,457 v / 6,738 f | **no** |
| one-sided | +35 / **+75** / +3 % | 42, over 4.4° | 4,135 v / 8,082 f | **yes — the registered extents test** |

**Then the table.**

| arm | rotation vs CAD (mod flip) | translation | top-16 within 15° | median pairwise | verdict |
|---|---|---|---|---|---|
| wide | **2.95°** (2.95°) | **0.08 cm** | **16/16** | 0.49° | **PASS** |
| one-sided | 5.74° (5.74°) | 0.67 cm | 16/16 | 0.97° | passes on the numbers, **VOID** |
| *chair, same job* | *124.27°* | *113.13 cm* | *1/16* | *134.17°* | *fail — reproduced exactly from v15* |

**What this establishes.** Wide passes and is not void, which rules out two rows: our mesh
recipe — TSDF fusion, ~1.1k points, `reject_outliers`, `poisson_mesh`, at the chair's
relative resolution — **is not the limit**. A bottle meshed exactly the chair's way
registers to within 3° of its CAD pose and the estimator converges 16/16.

**What it does not establish.** Whether one-sidedness or the chair itself is the limit.
The one-sided arm passes on every number, but the rule says it is void, and it stays
void. The case that the void test misfired is real — it was a proxy for a failed CAD
track, the wide arm fused over a superset of the same track sits within 19%, and the
one-sided mesh lands 0.67 cm from the CAD pose — but that argument was made after reading
the data, which is exactly what the registration exists to stop. It is recorded as an
argument, not a result.

**One fact for the next step, measured locally:** over the whole TUM sequence the chair
is seen along view directions spanning at most **31.4°** (p95 19.8°) — between the two
arms, nearer the one-sided one. Its mesh already uses every one of those views.

**Next, registered before it runs:** a coverage sweep on mustard0 at 4°, 15° and **31°**,
with the void test replaced by what it was meant to measure — CAD-track agreement with
the measured depth, per fused frame — rather than an extent proxy. The 31° arm is the
decisive one: it gives the bottle the chair's coverage. If it passes, coverage is not
the chair's problem and the chair is; if it fails, coverage is, and the move is more
views of the target, not a different mesher.


## Then

- [x] Wire the pose into the scene graph — `pipeline/tools/e2e_pose.py` consumes the
      kernel's `pose_result.json` and calls `refine_with_pose`, behind a gate on
      translation, rotation and depth. On today's numbers it **refuses**, which is the
      point: the position-only pose is uninformative about rotation, a wrong 6-DoF pose
      is actively misleading, and a planner cannot tell them apart.
- [ ] End-to-end 6-DoF grasp: the chain currently produces position-only object poses.

### Coverage sweep (v17) — decision rule, committed before the run is launched

The question v16b left open: is the chair's failure explained by how little of it the
sequence shows (at most **31.4°** of view direction), or by the chair itself? mustard0 is
meshed our way, exactly as in v16b, from frames within **4°, 15° and 31°** of frame 0,
plus the **wide** arm again as a positive control.

**Pass** (unchanged): rotation ≤ 30° allowing the long-axis flip, **and** ≥ 8/16 of the
top hypotheses within 15°.

**Void, per arm** — decided and printed in the job *before* that arm's pose is computed:
- fewer than 5 fused frames;
- actual view span (largest pairwise angle) below 0.8 × the arm's target;
- more than 20% of its fused frames where the CAD model at the tracked pose disagrees
  with measured depth by more than **1.5 cm** (per-frame median, ray-cast). This replaces
  v16b's extents test, which fired on a one-sided shell's own shape rather than on a bad
  track. Calibrated on synthetic data before the run: a 1.0 cm pose offset reads 1.08 cm,
  so 1.5 cm leaves margin for sensor noise and still catches a lost track;
- the mesh fails to build.

Extents are still printed, but are no longer a void test.

**The run is void** if the wide arm fails or is void: it passed in v16b, so a failure now
means something else broke.

**The decisive arm is 31°** — the bottle given the chair's coverage:

| 31° arm | reading | next move |
|---|---|---|
| pass | coverage at the chair's level is enough for our recipe; it does not explain the chair's failure | the problem is the chair itself — change the target object |
| fail | coverage at the chair's level is not enough | more views of the target, not a different object |
| void | undecided | report it; no post-hoc override |

The 4° and 15° arms describe the curve and carry no decision weight.

**Limits, stated before the result:** one object, a compact bottle; the reference is the
validated CAD pose, not ground truth. A 31° pass says coverage does not *explain* the
chair's failure — it does not say which property of the chair (size, thin structure,
symmetry) does.

### Result (v17) — read against the rule above, which was not edited

**Void checks, before anything else.** None fired. The CAD track agrees with measured
depth to a median **0.19 cm** (p90 0.45 cm) and **0/300** frames exceed 1.5 cm. Every arm
reached its target span and has well over 5 frames.

| arm | frames | view span | rotation vs CAD (mod flip) | translation | top-16 within 15° | verdict |
|---|---|---|---|---|---|---|
| 4° | 81 | 5.8° | 3.99° | 0.33 cm | 16/16 | pass |
| 15° | 89 | 16.1° | 4.59° | 0.47 cm | 16/16 | pass |
| **31°** | 99 | **31.3°** | **4.79°** | **0.29 cm** | **16/16** | **PASS — decisive** |
| wide (control) | 300 | 77.4° | 1.74° | 0.12 cm | 16/16 | pass — run is valid |
| *chair, same job* | — | *≤ 31.4°* | *124.27°* | *113.13 cm* | *1/16* | *fail — identical to v15 and v16b* |

**Reading, by the registered row:** a compact object meshed exactly our way, from exactly
the chair's view coverage, registers within 5° of its CAD pose with every top hypothesis
agreeing. **Coverage does not explain the chair's failure.** The registered next move is
a different target object.

**What "the chair itself" covers — the limit the rule stated, made concrete.** Two
candidate causes are now cleared by controls (the mesh recipe, v16b; coverage, v17), and
four input defects were fixed without effect (frames, mask, outliers, instance choice).
What still differs between the chair case and the bottle case is everything else at once:

| | chair | mustard0 control |
|---|---|---|
| object | 1.6 m, thin legs and frame | 0.2 m, compact |
| distance / sensor | ~2.1 m, Kinect v1 (TUM) | ~0.78 m |
| trajectory the mesh was fused along | ORB-SLAM3 on a **dynamic** sequence, ATE 18.7 cm | CAD track, 0.19 cm against depth |
| mask | our projected TSDF instance | upstream's |

This run cannot rank those. The trajectory row is the one worth testing first, because it
is the only one the pipeline controls and the cheapest to change: the same stack already
fuses TUM `fr1/xyz` — a static desk scene with compact rigid objects, ORB-SLAM3 at ATE
**1.03 cm** — so a target there changes object *and* trajectory quality together. That is
not a clean separation, and it is said here so it is not later read as one.

## Map prior (v18) — decision rule, committed before the run is launched

**What prompted it, measured locally first.** Our chair mesh placed at the pose our map
gives it reproduces the measured depth to **1.3–6.0 cm** (median under the mask; median
over the 8 frames **3.87 cm**) and covers 92–98% of the mask. FoundationPose's `register()`
pose misses the *same pixels* by **74–166 cm**. A pose that explains the data exists;
`register()` does not choose it. (The mask is built from depth-agreeing points, so the
map pose's absolute residual is partly by construction — the comparison holds because both
poses are scored on identical pixels.)

**Gate change, already made.** `e2e_pose.py` now renders the mesh at each candidate pose
and refuses if it disagrees with measured depth by more than **8 cm** (median under the
mask) — the bundle's existing `--depth-tol`, not a value tuned to the numbers above. It
is the gate's only check that needs neither the map nor the segmentation centroid. For a
pose *initialised from* the map, translation/rotation agreement with the map is not
independent evidence, so it is accepted only if this check is available.

**Q1 — diagnosis (FoundationPose's own scorer, one batch).** Score `register()`'s best
and the map pose after one refinement, together.
- map pose scores higher → `register()`'s **search** failed;
- `register()`'s pose scores higher → the **scorer** is fooled by this object.

**Q2 — what goes into the chain (the gate decides, not the scorer).** Refine from the map
pose, track the 8 frames from there, write `pose_result_prior.json`, run the gate on it.

| gate on the prior-tracked poses | median depth residual vs the map pose's 3.87 cm | reading | next |
|---|---|---|---|
| accepted | **lower** | FoundationPose holds a correct start and sharpens it | wire the map-prior path into the chain — the first accepted 6-DoF pose |
| accepted | equal or higher | it holds the start but adds no precision on a static object | wire it as a tracker only; claim no precision gain |
| refused | — | its refiner cannot hold even a correct start on this object | object-scale limit; change target |

Q1 does not override Q2: the depth check is independent of FoundationPose's networks,
the scorer is not.

**Void:** the prior section fails to run, the result's fingerprint does not match, or the
gate's depth check is unavailable.

**Limits, stated now.** An accepted prior-initialised pose is consistent with the depth
the camera measured; it does not independently confirm the map's rotation beyond what
that depth constrains. The object is static, so this measures holding and sharpening a
pose — tracking a *moving* object, the case FoundationPose exists for, is untested.

### Result (v18) — read against the rule above, which was not edited

**Void checks.** The prior section ran (`PRIOR_OK`), its fingerprint matched, and the
gate's depth check was available on all 8 frames. Not void.

**Q1 — the scorer prefers `register()`'s pose:** 134.28 against 133.94 for the map pose,
in one batch. The map pose scored the same before and after refinement (133.9375 both —
the scores are fp16, quantised to 1/16 here). By the rule: the scorer is fooled.

**Q2 — refused, which is the deciding row.**

| | map pose (measured locally) | after one refinement call (5 iterations) | tracked 8 frames from there |
|---|---|---|---|
| depth residual, frame 0 | **3.77 cm** | **85.24 cm** | — |
| depth residual, median of 8 | 3.87 cm | — | **125.85 cm**, 3 frames off the object entirely |
| vs map, frame 0 | — | 84.28 cm, 14.11° | — |
| gate | — | — | **0/8, refused** |

**One refinement call — five internal iterations, not one step; corrected 2026-10-05 —
from a pose that fits the depth to 3.8 cm moves it 77 cm and makes it 22× worse.** Whether
the first iteration is already wrong or the later ones diverge is not known from this run;
that is P1 below. By the registered row: FoundationPose's refiner cannot hold even a
correct start on this object — **object-scale limit; change target.**

**A lead, not a cause.** With this checkpoint's config (`normalize_xyz: true`,
`trans_rep: tracknet`) the refiner's translation step is its raw output × mesh diameter / 2
(`predict_pose_refine.py:229`): **0.85 m per unit for the chair**, ~0.11 m for mustard0.
But scaling is not the explanation by itself — 77 cm is **0.9 units even in normalised
terms**, a step the same network does not take on the bottle. Its prediction is wrong on
this input: a 1.7 m object at 2 m, cropped to 160 × 160 (thin legs a pixel or two wide),
from Kinect-v1 depth. That is an out-of-distribution reading, and it is untested here.

**What this does and does not close** *(rewritten 2026-10-05 after review; it first said
v18 "removes two of them as candidates")*. On identical pixels, FoundationPose leaves a pose
that fits the measured depth far better than anything it chooses. That locates the
*behaviour* in the estimator on this input. It does **not** clear the mesh, trajectory or
mask: they were built from depth-agreeing points, so the map pose's 3.9 cm is partly by
construction, and the absolute number cannot certify them. Held-out validation frames that
took no part in building the mesh are what would.

**Gate defect found by this run, fixed.** A mesh rendered onto no measured pixel was
recorded as "check unavailable" rather than as the largest disagreement there is. It
changed nothing here, but it would have let a pose rendered entirely off the object pass
the depth check. `bug_log.txt` [12].


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

**Next: separate mesh from object.** Give mustard0 a mesh built the way ours is — fused
from its own depth, one-sided, through the same Poisson path. It has ground truth, so the
result is an accuracy. If it fails, the mesh pipeline is convicted; if it still lands,
the chair is, and the honest move is a different target object.

## Then

- [x] Wire the pose into the scene graph — `pipeline/tools/e2e_pose.py` consumes the
      kernel's `pose_result.json` and calls `refine_with_pose`, behind a gate on
      translation, rotation and depth. On today's numbers it **refuses**, which is the
      point: the position-only pose is uninformative about rotation, a wrong 6-DoF pose
      is actively misleading, and a planner cannot tell them apart.
- [ ] End-to-end 6-DoF grasp: the chain currently produces position-only object poses.

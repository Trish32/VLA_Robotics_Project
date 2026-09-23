# FoundationPose — plan

`register()` and `track_one()` now run on a T4 against a mesh cut from our own TSDF — see
[RESULTS.md](RESULTS.md). What remains is explaining one number.

## The binding gap: a ≥99.5° rotation error

`register()` places `chair_4` at z = 2.73 m; our segmentation centroid says 2.05 m;
median depth under the mask says 2.10 m. The tracker is self-consistent (**2.08 cm**
world-frame spread over 8 frames), which bounds drift and nothing else.

**What the 72 cm actually is.** Moving the mesh origin a known 52.9 cm moved
FoundationPose's answer 53.5 cm, but **99.5° off** the direction our map predicts — same
offset vector, one rotated by the estimate, one by the true camera rotation. So the
orientation is essentially undetermined and the translation residual is its shadow. See
`bug_log.txt` [4], including the wrong diagnosis that produced this measurement.

**Hypothesis to test first, not to assume:** the mask is 695 px, where a 1.58 m object at
2 m under fx = 535 should subtend ~410 px across. That is a sparse partial view of a
one-sided Poisson shell — little signal to fix an orientation. Ways to discriminate:

| test | what it would show |
|---|---|
| denser fusion (smaller `--stride`) → bigger masks | if the gap shrinks, the mask was the cause |
| run upstream's own `demo_data/mustard0` | a clean CAD mesh + full mask isolates our bundle from FoundationPose itself |
| a cleaner instance (higher Mask3D score, tighter extent) | `chair_4` is 1.58 × 1.18 × 0.99 m — a coarse region, not a clean object |

Running upstream's demo is the decisive one: it is the only test that separates "our mesh
is bad" from "FoundationPose is misbehaving", and it is the Fidelity Rule's own
prescription — validate against the original before trusting the adaptation.

## Then

- [x] Wire the pose into the scene graph — `pipeline/tools/e2e_pose.py` consumes the
      kernel's `pose_result.json` and calls `refine_with_pose`, behind a gate on
      translation, rotation and depth. On today's numbers it **refuses**, which is the
      point: the position-only pose is uninformative about rotation, a wrong 6-DoF pose
      is actively misleading, and a planner cannot tell them apart.
- [ ] End-to-end 6-DoF grasp: the chain currently produces position-only object poses.

# FoundationPose — plan

`register()` and `track_one()` now run on a T4 against a mesh cut from our own TSDF — see
[RESULTS.md](RESULTS.md). What remains is explaining one number.

## The binding gap: a 72 cm disagreement

`register()` places `chair_4` at z = 2.73 m; our segmentation centroid says 2.05 m. The
tracker is self-consistent (**2.08 cm** world-frame spread over 8 frames), so this is not
drift — the two methods disagree about where the object *is*, and neither is ground truth.

**Hypothesis to test first, not to assume:** the mask is 695 px, where a 1.58 m object at
2 m under fx = 535 should subtend ~410 px across. That is a sparse partial view, so the
two estimates may be centring on different subsets. Ways to discriminate:

| test | what it would show |
|---|---|
| denser fusion (smaller `--stride`) → bigger masks | if the gap shrinks, the mask was the cause |
| run upstream's own `demo_data/mustard0` | a clean CAD mesh + full mask isolates our bundle from FoundationPose itself |
| a cleaner instance (higher Mask3D score, tighter extent) | `chair_4` is 1.58 × 1.18 × 0.99 m — a coarse region, not a clean object |

Running upstream's demo is the decisive one: it is the only test that separates "our mesh
is bad" from "FoundationPose is misbehaving", and it is the Fidelity Rule's own
prescription — validate against the original before trusting the adaptation.

## Then

- [ ] Feed the refined 6-DoF pose back into the scene graph via `refine_with_pose`, which
      has never run on real FoundationPose output — only on synthesised poses.
- [ ] End-to-end 6-DoF grasp: the chain currently produces position-only object poses.

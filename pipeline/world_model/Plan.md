# Action-conditioned world model — plan

What is not done, and what would close it. Measurements are in [RESULTS.md](RESULTS.md).

## 1. The object pathway is signal-limited, not mask-limited — MEASURED

Previously this section said the object pathway needed per-instance masks and named
running our own OpenMask3D over the `cube_to_bowl_5` video as the cheaper route. That is
done (`extract_tracks.py`), the tracks are good, and **the model still ties identity**.

The cause is now measured rather than assumed: median single-step cube displacement is
0.00084 normalised image units — half a pixel — and a linear map from the action explains
R² = 0.016 of it. At a one-second step the same fit reaches R² = 0.139. The relationship
is real and buried under the extraction's own noise at the frame rate the model was being
asked to predict at.

**What would close it**, in order of expected effect:

| lever | why |
|---|---|
| more demonstration episodes | two training episodes; validation is best at epoch 0, so every run overfits immediately. This is the binding constraint |
| depth, or a calibrated camera | image-plane centroids conflate object motion with camera-relative geometry. Metric 3-D tracks would remove a whole noise source |
| predicting a longer step | already implemented (`--stride`); improves the signal but cannot substitute for data |

### The old section, kept because the constraint it names is still real

The binding gap. The architecture predicts per-object motion; the data to fit it does
not exist in this repo.

| have | why it is not enough |
|---|---|
| `cube_to_bowl_5_with_mask` | **one** episode, and a single binary foreground mask rather than per-instance segmentation — so one "object", no holdout episode, and the model scores worse than identity |
| TUM `freiburg3_walking_xyz` | no robot and no actions. A static dataset cannot supervise action-conditioned dynamics at all |
| `gr1.PickNPlace` | proprioception only; no masks shipped |

**What closes it:** episodes with per-instance masks and actions — several of them.
Either more of NVIDIA's masked demo data, or running our own OpenMask3D over the
existing `cube_to_bowl_5` videos to produce per-instance tracks, which is the cheaper
route and uses the stack already here. That second option is real work, not a download:
the instances must be associated across frames, which is what `InstanceRegistry` is for.

Until then the object slots roll forward near-static and the README says so.

## 2. The proprioceptive gain is not stable

+1.2% to +22.6% over constant velocity across three seeds. Three training episodes is
too few to distinguish "the model learned the action" from "this seed landed well".

**What closes it:** more episodes, and reporting a seed distribution rather than a
single run. Neither is a design question — the training script already takes both.

## 3. Per-candidate uncertainty is not calibrated — MEASURED

Closed as an open question, and the answer was the unflattering one. `evaluate.py` rolls
319 held-out windows 8 steps and compares predicted spread against realised error:
**rank correlation −0.071.** The spread does not say which rollout will be wrong, so the
risk term was decorative and its default weight is now zero.

It is not useless: the spread tracks error growth across horizons almost exactly (14.1×
predicted, 14.2× realised over 8 steps). It is a horizon discount, not a candidate
discriminator, and every candidate in a plan shares a horizon.

**What would close it properly:** a dynamics model good enough for member disagreement to
mean something — i.e. §1 and §2. Deep ensembles are calibrated when members are fit to
enough data to disagree *informatively*; four episodes is not that.

## 4. The value head is unused

`ValueHead` exists and is never trained. Five demonstrations would teach it to memorise
five demonstrations, so the planner runs on geometric terms only and the head waits for
real reward or preference data. Deliberate, and recorded here so it is not mistaken for
an oversight.

## 5. The planner selects, it does not optimise

There is no gradient through the dynamics into the action. With a model this weak,
optimising against it would find its errors rather than good actions — a well-known
failure mode of model-based control with a learned model. Ranking what the policy
already proposed is the honest use of it. Revisit when §1 and §2 are closed.

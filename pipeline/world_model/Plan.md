# Action-conditioned world model — plan

What is not done, and what would close it. Measurements are in [RESULTS.md](RESULTS.md).

## 1. The object pathway needs object data

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

## 3. Multi-step rollout error is unmeasured

Everything reported is **one-step**. The planner rolls out 8 steps, and compounding
error over that horizon is exactly what the ensemble uncertainty is supposed to stand in
for — but the relationship between predicted uncertainty and actual N-step error has
never been checked. If the uncertainty is badly calibrated, the risk term is decorative.

**What closes it:** an N-step held-out evaluation, comparing predicted spread against
realised error per horizon. Cheap, and it needs no new data — it is the next thing to do.

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

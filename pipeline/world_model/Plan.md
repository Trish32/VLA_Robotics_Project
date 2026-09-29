# Action-conditioned world model — plan

What is not done, and what would close it. Measurements are in [RESULTS.md](RESULTS.md).

## 1. The object pathway was signal-limited — CLOSED, in simulation

The diagnosis in this section was that the object pathway failed for want of data and
metric tracks, not for want of architecture. That has now been tested rather than
argued, by building the data it named: `pipeline/sim/` is a MuJoCo cube-to-bowl arena
producing episodes at ~0.03 s each with ground-truth metric object boxes.

| | real `cube_to_bowl_5` | simulated, 400 episodes |
|---|---|---|
| training episodes | 2 | 320 |
| object tracks | image-plane centroid + sqrt-area | centre **and** extent, in metres |
| median single-step object motion | 0.00084 image units (~half a pixel) | ~12 mm |
| object RMSE vs identity | **+0.1%** — a tie | **+39.5%** |
| object RMSE vs constant velocity | −39.9% (worse) | **+17.3%** |
| the cube alone, vs identity | tie | **+63.6%** |

The prediction the old section made was correct: with the signal above the measurement
floor and two orders of magnitude more episodes, the same architecture beats both
trivial predictors. Nothing about the model changed.

**What this does and does not license.** It licenses the claim that the object pathway
works when the data supports it. It does **not** license a claim about the real
demonstrations, which still have two usable episodes and still tie identity; that
result stands unchanged below. Simulated tracks are also exact, so the figures above
exclude tracking error entirely — the honest reading is an upper bound on what the
pathway can do, measured in the absence of the noise that defeated it on real video.

One incidental finding, recorded because it bit: the real tracks carry three channels
(centroid-x, centroid-y, sqrt-area), so after velocity augmentation the scorer's
`SLOT_EXTENT` channels were reading **velocities as if they were box sizes**. The
simulated tracks carry centre-then-extent, which is what the scorer has always assumed.

### The original section, kept because it is what was measured on real data

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

## 4. The value head is trained, in simulation — and it beats the heuristic

Previously: "`ValueHead` exists and is never trained. Five demonstrations would teach it
to memorise five demonstrations." That constraint was the data, and the arena removed
it. `pipeline/sim/train_value.py` fits the head on the discounted return of the sparse
success reward — 340 training episodes, 40 held out by episode.

| predictor of discounted return | held-out MSE |
|---|---|
| constant (train mean) | 0.054688 |
| cube-to-goal distance, least squares | 0.032412 |
| **learned value head** | **0.001417** (+95.6% vs distance) |

The distance row is the one that matters: the head exists to replace the scorer's
geometric `progress` term, so beating a constant proves only that state matters.
`load_value` refuses a checkpoint whose held-out MSE does not beat its own distance
baseline, so this cannot silently regress into a slower restatement of the heuristic.

Getting here required fixing the label first. `next.reward` fired on the policy
reaching its terminal waypoint stage, which `env.done()` almost never lets it reach, so
**377 of 380 successful episodes were labelled failures** and an earlier run of this
same script reported a meaningless +10.3%. See [bug_log.txt](../bug_log.txt) [S13];
the +95.6% above is from the corrected column.

Whether a better objective actually rescues *planning* is a separate question, measured
in [RESULTS.md](RESULTS.md) rather than assumed here.

## 5b. The binding constraint is commitment coverage, not the model — MEASURED

This supersedes §5, §6 and §8 as the thing to fix first. Three controls, 24 episodes
each with 95% bootstrap CIs:

| rule | candidates | success |
|---|---|---|
| no planning / `--pick first` / `--pick best` with **K=1** | — / 8 / 1 | **95.8% [87.5, 100.0]** |
| `--pick best` | 8 | 0.0% [0.0, 0.0] |
| **`--pick oracle`** (rewind, run each to completion, keep the winner) | 8 | **0.0% [0.0, 0.0]** |

Greedy with one candidate reproduces the policy, so the execution path is sound.
**Perfect per-decision selection is exactly as fatal as the model's**, so no amount of
ranking work can help.

`compare_traces.py` gives the mechanism: the chosen plan changes at **80.3%** of
consecutive decisions, 7.62 of 8 distinct plans are used per episode, no choice survives
more than ~4 decisions of 90, and each executed action is 1.02× the policy's own step
size away from what the policy wanted. Candidates are whole plans; selection re-decides
every `execute` steps; the arm never carries one out.

**Success tracks commitment coverage — the share of a waypoint stage a choice outlasts
— rather than breaking at a threshold.** Coverage 0.3% / 27.4% / 71.8% / ~97% at
K = 8 / 16 / 32 / 64 gives success 0.0% / 0.0% / 37.5% / 66.7%. The 28-step median dwell
is a landmark on that curve, not a cut-off: K=32 already exceeds it and still loses most
episodes, because half the stages are longer and `transfer` runs to 61 steps.

**The fix is structural: choose a plan once and hold it**, or make each decision a
continuation of the plan in progress rather than a fresh set of alternatives.
Until that is done, §6 (counterfactual coverage) and §8 (applicable-domain gating) are
improvements to a component that is not the bottleneck — worth having, and not worth
expecting a closed-loop gain from.

## 5. The planner selects, it does not optimise

There is no gradient through the dynamics into the action. With a model this weak,
optimising against it would find its errors rather than good actions — a well-known
failure mode of model-based control with a learned model. Ranking what the policy
already proposed is the honest use of it. Revisit when §1 and §2 are closed.

## 6. The dynamics has no counterfactual coverage — MEASURED, and it is the binding one

`sim/ranking_metrics.py` scores the ordering rather than the outcome, and the two rows
that matter differ only in **whose trajectory the episode follows** while the ranking is
measured. On the policy's own trajectory the scorer ranks affordance candidates at
Kendall τ **+0.291**; on the trajectory its own choices produce, the same model on the
same candidates ranks at τ **−0.157** with regret@1 up 4.8×. The ranking does not merely
degrade off-distribution, it **inverts**.

Counterfactual coverage fixes it, and the size of the fix depends on how it is
measured. Self-driven it reads τ **+0.671**; **state-matched** — both models scored at
the same decisions on the same candidates — it reads **+0.029** against the baseline's
−0.114, with regret@1 down 35%. The gap between those two numbers is the model steering
itself into states where ranking is easier, which `ranking_metrics --also-score` exists
to expose. A quantity control (1,903 on-policy episodes, five times the data and no
counterfactual branches) reaches only −0.065 state-matched: **coverage does about twice
what quantity does, and only coverage crosses zero.**

That is the defect named precisely. Every transition in the training set is one a
competent demonstrator took from a state that demonstrator reached, so evaluating "what
if we grasped 2 cm to the left" is extrapolation — and acting on the answer is what
carries the episode further into the region where the extrapolation is worse. It
re-reads the reachability collapse in [RESULTS.md](RESULTS.md) (91.7% under `--pick
first` → 40% under `--pick best`) as a compounding loop rather than a bad objective, and
it subsumes §5: a planner cannot be blamed for optimising against a model that is
inverted exactly where optimisation takes it.

`collect.py --counterfactual N --branch-length L` supplies the missing slice: from
states along an on-policy episode, branch under a *different* plan and record the
result. States stay on the demonstrator's distribution while the actions leave it, which
is what the demonstrations can never contain by construction — and which the domain
calibration independently confirms is the right axis, since state energy never leaves
its in-distribution range while the action does (§8).

**It does not close the loop.** Greedy selection is 0.0% with coverage alone and 16.7%
with coverage plus the applicable-domain gate, against 95.8% for simply running the
policy. Better ranking is necessary and demonstrably not sufficient.

## 8. The applicable domain is defined by the action, not the state — MEASURED

`sim/domain.py` gates the scorer: rank only where the dynamics was fitted, roll back to
the policy's own chunk everywhere else. Three signals were calibrated against
per-decision Kendall τ and only one survived.

| signal | best τ reachable by any threshold | shipped |
|---|---|---|
| state energy | −0.095 (from −0.133 ungated) | unarmed (∞) |
| ensemble spread | −0.106 | unarmed (∞) |
| **per-candidate action deviation** | **+0.062** | **armed at 0.68** |

Filtering candidates at deviation ≤ 0.68 keeps half of them, loses none of the 144
decisions, and takes the ordering from inverted to correct. The mechanism is that
deviation predicts the scorer's own error per candidate (r = +0.417).

State energy ranged 0.38–1.43 in exactly the regime the gate exists for — the scorer
steers into ordinary scenes — so it is retained as a backstop the caller must arm rather
than shipped with a number that would look calibrated and do nothing.

## 7. The veto's label was the problem — REOPENED, with a signal

`sim/calibrate_veto.py` measured every signal the model can produce against
rewind-and-execute ground truth. Every value-derived signal is at or below chance;
only the ensemble's disagreement with itself carries information (AUC 0.587), and its
precision peaks at 0.404 against a base rate of 0.285.

The cause is not the classifier, and the obvious structural explanation is **wrong**.
This section previously read "the latent carries the cube, the bowl and the ball and no
gripper". `sim/probe_latent.py` refutes it: a linear probe on the latent recovers the
finger opening at R² 1.000, the tip-to-cube offset at 0.999, the pad-to-cube clearance
at 0.973 and whether the cube is held at 0.903. The representation is there.

What is not established is whether the **dynamics** can propagate that geometry under an
action, or whether the **value head** can map it to an outcome. That is the open
question, and it is a narrower one than "add the gripper", which was the recommended
next step until the probe was run.

Rolling every available plan to completion at each decision shows recoverability is
bimodal: 15.5% of decisions have no surviving plan, 71% have all eight, and **62% of
"doomed action" labels come from states that were already lost**. The label was largely
a property of the state, not of the action.

Removing those decisions does not close the section — it **reopens** it. On the 105
pivotal decisions the predicted return drop reaches **AUC 0.653 [0.541, 0.764]**, an
interval excluding chance, while ensemble spread is *anti*-predictive at 0.371 [0.269,
0.490]. That is the reverse of what the pooled numbers said and the reverse of what
`veto.py` currently defaults to.

Open: re-measure on an independent sample before changing the default, and find whether
a run-time proxy for "is this decision pivotal" exists — without one, a gate cannot tell
the 21% of decisions where it could help from the 79% where it cannot.

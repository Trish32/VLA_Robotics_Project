# Action-conditioned world model — results

All numbers are held-out RMSE in standardised units, on **LeRobot demonstration data**
(`grootN1_Robotics/upstream/demo_data`). Splits are by **episode**, never by frame:
adjacent frames at 20–30 Hz are nearly identical, so a frame-wise holdout measures
interpolation and reports it as generalisation.

## Proprioceptive dynamics · MEASURED

`gr1.PickNPlace` — 5 episodes, 2,091 transitions, 44-dim state and action.
Split **3 train / 1 validation / 1 test episodes**. The epoch is chosen on the
validation episode; the test episode is touched once.

| predictor | global RMSE | **median sample** | uses the action? |
|---|---|---|---|
| identity — *nothing changes* | 0.05540 | 0.02698 | no |
| constant velocity | 0.02666 | **0.00167** | no |
| **learned dynamics** | **0.01929** | 0.00770 | **yes** |

**The two aggregations disagree about who wins, and both are true.** Global RMSE squares
before averaging, so it is dominated by the worst samples; the median describes the
typical one. On identical predictions:

- **constant velocity beats the model on 91% of samples**, and is 4.6× better on the
  median one;
- **the model is far better in the tail** — p99 0.083 vs 0.152, worst case 0.129 vs
  0.226, and on constant velocity's ten worst samples the model is 40% better.

Constant velocity is excellent during smooth motion and fails at direction changes and
contact — exactly where the action matters and where a planner needs to be right. That
is an argument for caring about the tail, not a measurement that the model is better;
the honest summary is that **it trades typical accuracy for tail robustness.**

An earlier revision of this page reported only the global figure, as "+26.2% vs constant
velocity". That is arithmetically correct and materially misleading on its own.

Across three seeds the global-RMSE gain over constant velocity ranges **+1.2% to +22.6%**
— on three training episodes that cannot separate a learned dynamics from a lucky init.

### The failure that produced the design

The first working version lost badly:

| | RMSE |
|---|---|
| constant velocity | 0.02717 |
| model, identity-initialised, no velocity in the latent | 0.05900 |

The cause was architectural, not a tuning problem: `z` carried only position, so the
model **could not represent constant-velocity motion even in principle** while being
asked to beat it. Adding velocity to the latent alone did not fix it (0.05795, still
worse than identity) — the model was initialised at identity and had to learn its way
past momentum on 1,760 samples, which it spent all its capacity doing. Making the
untrained model *be* the constant-velocity predictor, so the learned term only supplies
the action's correction, is what closed it.

Training loss reaches ~6e-5 against a test RMSE of ~2e-2, and validation error bottoms
out around epoch 15 and rises after. The model overfits 3 episodes comfortably; early
stopping is load-bearing, not hygiene.

## Multi-step rollout and uncertainty calibration · MEASURED

319 eight-step windows from the held-out episode. The planner rolls out 8 steps and
discounts by ensemble spread, so both of those needed checking and neither was implied
by a one-step number.

| step | model (global) | const-vel (global) | model (median) | const-vel (median) | win rate | spread | rank corr |
|---|---|---|---|---|---|---|---|
| 1 | 0.01945 | 0.02694 | 0.00760 | 0.00165 | 9% | 0.01063 | −0.071 |
| 4 | 0.10084 | 0.15829 | 0.07060 | 0.01514 | 12% | 0.05775 | −0.051 |
| 8 | 0.27626 | 0.38114 | 0.22405 | 0.05355 | 15% | 0.14977 | −0.120 |

The one-step pattern holds all the way out: the model beats constant velocity at **8/8
horizons on global RMSE and 0/8 on the median**, winning only 12% of individual
rollouts. Where it wins, it wins on the hard ones.

### The uncertainty is not calibrated per candidate

**Mean rank correlation between predicted spread and realised error: −0.071.** The
ensemble disagreement does not predict which rollout will be wrong. The risk term was
therefore weighting a quantity with no relationship to the thing it stands in for, and
its default weight is now **zero** — the term is kept because it is correct in principle,
but a measurement, not an opinion, decides whether it is on.

What the spread *does* track is horizon:

| | step 1 → 8 | growth |
|---|---|---|
| realised error (global RMSE) | 0.0195 → 0.2763 | **14.2×** |
| predicted spread | 0.0106 → 0.1498 | **14.1×** |

So it is a good horizon discount and a useless per-candidate discriminator. Since every
candidate in a plan shares a horizon, weighting it changed nothing except appearances.

## Object dynamics · MEASURED, AND NEGATIVE

`cube_to_bowl_5_with_mask` — the only dataset here with per-frame object masks.
**One episode**, so no episode-level holdout is possible; a temporal 25% tail is used
instead and it is the weaker claim.

| predictor | slot RMSE |
|---|---|
| identity — *the object does not move* | 0.35029 |
| **learned dynamics** | **0.35541** |

**The model is worse than assuming nothing moves.** One episode of a single binary
foreground mask does not supervise object dynamics, and nothing here should be read as
predicting how objects respond to actions. The architecture carries an object pathway;
the data to fit it does not exist in this repo.

Proprioception on the same run came out at 0.01572 against a constant-velocity 0.01582 —
a 0.6% difference, i.e. a tie. Six state dimensions and one episode.

## What the scoring loop does on a real scene · MEASURED

Stage 6.5 on `freiburg3_walking_xyz`, 5 instances, GR00T's 16 × 29 chunk expanded to
8 candidates over an 8-step horizon.

Absolute collision cost was **identical (0.2400) for every candidate**, and vetoed all
eight. The cause is that Mask3D's coarse proposals already interpenetrate — measured
separately, **all 15 instance pairs have overlapping boxes**. That is a property of the
segmentation, not of any action, and charging it to candidates makes the term incapable
of discriminating. Scoring only **induced** cost against the pre-action scene drops the
spurious vetoes to zero.

With the dynamics untrained for this action width (29 DoF here vs the checkpoint's 44),
the remaining discriminating term is effort, and the planner selects candidate 0 — the
policy's own unmodified chunk. That is the correct degenerate behaviour: with nothing
to predict, the planner defers to the policy rather than inventing a preference.

## Verified locally · 32 tests

Shape and gradient correctness on CPU with tiny inputs, per the repo's local bar. The
properties that are load-bearing rather than incidental:

- the untrained model is **exactly** the identity, and exactly constant-velocity with
  `integrate_velocity` — checked against hand-computed values;
- **padded slots never influence a prediction or a score** — the same two real objects
  predict identically whether or not blanks trail them, and padding at the origin is not
  scored as a pile of colliding objects;
- **ensemble uncertainty grows with horizon**, the property the planner relies on;
- collision and support terms match hand-computed volumes and drops
  (0.5 m³ overlap, 1.95 m drop);
- the planner **vetoes a collision the action causes** and **does not charge pre-existing
  overlap** — the two halves of the scene-baseline fix.

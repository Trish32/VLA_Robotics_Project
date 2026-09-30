# Pre-registered prediction — retiming the task should move the commitment fracture

Written BEFORE the sweep below was run. Measured dwells are already in hand; the
success rates are not.

## The structural claim being tested

Closed-loop success under selection is zero while the commitment K is shorter than a
waypoint stage, because the arm is redirected before it finishes the phase it is in.
At the shipped timing the median stage lasts 28 steps and the fracture sits between
K=16 and K=32.

If that explanation is right, the fracture is set by the **task's own timescale**, not
by any absolute number of steps, not by the chunk length, and not by the candidate set.

## The lever

`ScriptedPickPlace.retimed(r)` scales `max_step`, `grip_hold` and `settle_hold`
together, stretching every stage by r without changing what the task is. Measured
median stage duration, 60 episodes each:

| retime r | median stage (steps) | mean | p90 |
|---|---|---|---|
| 0.5 | **21** | 21.7 | 42 |
| 1.0 | **28** | 29.0 | 53 |
| 2.0 | **42** | 47.3 | 83 |
| 3.0 | **61** | 67.5 | 114 |

(Not proportional to r: `close` and `release` scale exactly as 14r, while the motion
stages are set by distance / speed against a fixed settle tolerance.)

## Predictions

**P1 — the fracture moves with the median.** Perfect-selection success stays at ~0 for
K < median and rises for K >= median. Specifically the fracture lies in:

| retime r | predicted fracture K* | bracket |
|---|---|---|
| 0.5 | ~21 | between K=11 and K=32 |
| 1.0 | ~28 | between K=14 and K=42 |
| 2.0 | ~42 | between K=21 and K=63 |

**P2 — the curves collapse.** Plotted against **K / median_dwell** rather than K, the
three retimings lie on one curve, with the rise at K/median ~ 1.

**P3 — the absolute-step alternative is refuted.** If the fracture were a fixed number
of steps (an artefact of the 360-step episode cap, the 8-step chunk the model was
trained on, or anything else independent of the task), all three curves would break at
the same K and would NOT collapse under rescaling.

## What would falsify this

- Success at K = 0.5 x median being well above zero, or success at K = 1.5 x median
  still at zero, for any retiming.
- The three curves breaking at the same absolute K (supports P3's alternative).
- No collapse under the K/median rescaling — i.e. the rescaled curves disagree by more
  than their bootstrap intervals.

## Method

`--pick oracle`, which is model-free, so the retimings do not need three retrained
dynamics models and the test measures the control scheme rather than a learned scorer.
`--horizon K --execute K` (full commitment), affordance candidates, 16 episodes per
cell, 95% bootstrap CI over episodes. `--max-env-steps` scaled with r so a slowed task
does not simply run out of time.

---

# Pre-registered prediction 2 — phase-aligned commitment beats any fixed K

Written BEFORE the runs below.

## The claim

A fixed commitment of K=32 still cuts the **28% of stages that outlast it**, and
`transfer` alone runs to 61 steps at p90. If the fracture is genuinely set by stage
boundaries rather than by a step count, then committing **until the stage changes** —
holding the chosen plan and regenerating its chunk as the buffer empties — should beat
every fixed K, including K=32.

`--commit phase` implements exactly that and changes nothing else: the same candidates,
the same selection rule, the same arena.

## Predictions

**P4 — phase-aligned > K=32.** Perfect-selection success under `--commit phase`
exceeds the 37.5% [16.7, 58.3] measured at K=32, and the intervals should not be
subsumed by it.

**P5 — it beats the best fixed K at matched chunk length.** Compared at the same
`--horizon`, phase alignment wins, so the gain is the alignment and not a longer chunk.

**P6 — decisions per episode drops toward the stage count.** Phase alignment should
make roughly one decision per stage — about 6-8 per episode — against 90 at K=4 and
~11 at K=32. If the decision count does not fall, the mechanism is not engaging and P4
is untestable rather than confirmed.

## What would falsify this

- Phase-aligned success inside the K=32 interval, or below it.
- Decision count per episode not falling to roughly the number of stages.
- A fixed K greater than 32 (say 64) matching phase alignment — which would mean the
  fracture is about duration alone and the boundaries are irrelevant.


---

# Outcome — P6 fired, in the direction nobody registered

**P4 and P5 are still untested.** The first phase-aligned run was invalid, and P6 is how
that was known within a minute of reading the output.

| arm | success | decisions / episode | steps / episode |
|---|---|---|---|
| fixed K=32 | 37.5% [16.7, 58.3] | 11.1 | 340 |
| fixed K=64 | **66.7% [45.8, 83.3]** | 4.0 | 245 |
| phase, horizon 32 | 0.0% | **2.6** | 360 (cap) |
| phase, horizon 16 | 0.0% | **1.0** | 360 (cap) |

P6 was registered as a guard against the mechanism *not engaging*: "if the decision
count does not fall, the mechanism is not engaging and P4 is untestable rather than
confirmed." It fell — to **one decision per episode**, far past the ~6–8 that one
decision per stage would give, with every episode running to the step cap.

Read as a result, phase alignment scoring 0.0% against fixed K=64's 66.7% would have
been a clean falsification of P4 and a genuinely interesting one: *boundaries do not
matter, only duration does*. It was wrong. The loop was deadlocked.

**The mechanism, once traced.** The commitment was keyed to the stage of the *nominal*
policy while the arm executed a *different* plan. The nominal machine advances when the
tip arrives within 8 mm of its own target; the arm was driving to an offset target, so
it never arrived, the stage never changed, the commitment never expired, and the plan
was refilled forever.

Fixing that exposed a second, physical deadlock underneath it. Once the driver adopts
the chosen plan, a `descend` target 14 mm off the cube's centre sits **inside the cube's
own footprint** — the gripper cannot reach it, the tip never comes within tolerance, and
that stage never ends either. The waypoint machine had no timeout because the nominal
demonstrator never needs one (p90 stage 53 steps); off-centre plans do.
`stage_timeout = 120` now caps it, and the demonstrator is unchanged at 10/10 canonical
and 40/40 randomised, which is the check that the cap changed the failure mode and not
the task.

**Why this is the argument for pre-registering.** Nothing about the headline number
looked wrong. 0.0% is a perfectly plausible result for an intervention, it had a tidy
story ready, and it would have been written up. What refused it was a *secondary*
quantity registered in advance for a reason unrelated to the outcome — and it fired in
the opposite direction to the one it was written for, which is only possible because it
was committed to before the data existed. A diagnostic chosen after seeing the result
would have been chosen to explain it.

That is the third time on this project that an instrument produced a plausible number
from a broken measurement (bug_log [S14], [S18], [S19]), and the first time one was
caught before it reached a document.

---

# P7–P10 · The veto's unit is the stage, not the decision

Registered **2026-09-28**, before any held-out episode exists. Discovery set is
`piv7.json`: 130 episodes at `--seed 0`, 1987 decisions, stage recorded per decision.
Held-out set will be 130 episodes at `--seed 1`, same checkpoints, same flags, nothing
else changed. No number below has been computed on the held-out set.

## What was found on the discovery set

Splitting decisions by the demonstrator's waypoint stage gives a sharply non-uniform
rescue-to-breakage ratio. A veto is worth firing where holding rescues a doomed action
more often than it breaks a working one:

| stage | decisions | rescue | breakage | ratio |
|---|---|---|---|---|
| transfer | 615 | 8.0% | 2.4% | **3.3 : 1** |
| descend | 474 | 6.1% | 4.9% | 1.2 : 1 |
| lower | 413 | 5.6% | 6.1% | 0.9 : 1 |
| lift | 240 | 1.7% | 2.9% | 0.6 : 1 |
| close | 130 | 0.0% | 0.8% | harm only |
| release | 39 | 0.0% | 0.0% | never decisive |

Vetoing during `transfer` alone scores **+5.5 net flips per 100 fires, 95% CI
[+2.0, +9.7]**, against the learned surface-median gate's +3.2 [−0.6, +7.6].

**This is a post-hoc selection over seven stages and is registered as such.** Being
best of seven is worth about what a one-in-seven maximum is worth, which is why it is
written down here before being tested rather than reported as a result.

## Predictions

**P7 · transfer-only clears zero on held-out episodes.** A veto restricted to
`transfer`, with no learned signal at all, scores net flips per 100 fires whose 95%
episode-bootstrap CI excludes 0.

**P8 · the stage ordering replicates.** Ranking stages by rescue-to-breakage ratio on
the held-out set puts `transfer` in the top two, and `close` and `release` in the bottom
two. Spearman between the discovery and held-out per-stage ratios is positive.

**P9 · the learned signal adds nothing inside the stage.** Within `transfer` alone,
adding the learned gate does not raise conditional rescue: the paired difference
between transfer-gated and transfer-plus-learned-gate spans 0. If P9 holds, the world
model is removable from the veto and the gate is a stage lookup.

**P10 · stage-conditional thresholds beat one global threshold.** A gate that loosens
on `transfer`, disables on `close`/`release`/`lift`, and tightens on `descend`/`lower`
beats a single global threshold at matched firing rate, paired over episodes, with the
difference excluding 0.

## What would falsify each

- **P7 fails** if the held-out CI includes 0. Then +5.5 was the one-in-seven maximum
  and there is no stage effect to exploit.
- **P8 fails** if `transfer` drops out of the top two, or the rank correlation is not
  positive. Then the ordering is noise and P10 has nothing to condition on.
- **P9 fails** if the learned gate *does* raise conditional rescue within `transfer`
  with the difference excluding 0. That would be the first evidence this session that
  the world model earns its place in the veto, and it should be reported as such
  rather than buried — the prediction is written expecting the opposite.
- **P10 fails** if the paired difference spans 0. Then per-stage tuning is overfitting
  to six numbers and one global threshold is the honest gate.

**Engagement check, separate from effect size:** the held-out set must contain at least
400 `transfer` decisions and at least 25 decisive ones there, or P7–P10 are
underpowered and report "not tested" rather than "not found".

## Outcome — P7, P8 and P10 all failed on held-out episodes

Run 2026-09-28, `--seed 1`, 130 episodes, 1978 decisions. Engagement check passed
(615 `transfer` decisions against a 400 floor, 45 decisive against 25), so these are
negatives and not underpowered non-results.

| | discovery | held-out | |
|---|---|---|---|
| **P7** transfer-only net/100 | +5.5 [+2.0, +9.7] | **−0.5 [−3.3, +2.2]** | **FAILED** |
| **P8** transfer rescue:breakage | 3.27 | **0.88**, Spearman −0.100 | **FAILED** |
| **P9** learned signal inside transfer | +0.073 [−0.066, +0.203] | +0.023 [−0.194, +0.382] | held |
| **P10** stage-conditional vs global | +2.0 [+0.2, +4.0] | **−1.2 [−3.3, +0.8]** | **FAILED** |

`transfer` does not merely lose its margin — its ratio inverts, from best of seven to
below parity, and the rank correlation between the two sets' per-stage ratios is
*negative*. The ordering was noise throughout.

**This is what the registration was for.** +5.5 [+2.0, +9.7] excluded zero, had a
mechanism ready — transfer has the lowest conditional breakage of any stage, so a veto
there should be cheap — and would have been written up. Nothing about it looked wrong.
It was a maximum over seven stages, reported with the interval of a single measurement,
and the only thing separating it from a result was a prediction committed to before the
held-out set existed.

It is also the exact failure mode of the (q1, q2) surface one step earlier: a headline
read off the argmax of a grid. That one was caught by plotting the surface; this one
needed held-out data, because the grid had seven cells and no shape to inspect.

P9 held, and holds in the same direction on both sets: the world model adds nothing
detectable inside a stage. That survives only as far as it goes — with the stage effect
withdrawn, there is no stage restriction left for it to be inside.

---

# P11–P15 · Constraint 3, as a confirmatory test

Registered **2026-09-28**, before the run exists. Every number in Constraint 3 came from
`piv7` (seed 0) and `piv8` (seed 1), and `piv8` has since been used three times — for
P7–P10, for the state-change alignment test, and for the burst-onset test. It is no
longer held out in any meaningful sense. Constraint 3 is a strong general claim
("decision importance is not a run-time attribute of the state") assembled from
exploratory work, so it gets one clean test.

Confirmatory set: **`--seed 2`, 130 episodes**, same checkpoints, same flags, nothing
else changed. This file is committed before the run is launched. The analysis below is
fixed in advance and will be run **once**; no threshold, signal or subgroup may be added
after seeing it.

## Predictions

**P11 · no run-time signal beats AUC 0.65 against the decisive label.** Scored over all
decisions, episode-bootstrapped: `plan spread ¼-horizon`, `return drop`, `mean plan
value`, `action deviation`, `ensemble spread`, `model surprise`, `|Δ plan spread|`,
`|Δ observation|`, `|Δ value head|`. Prediction: **every point estimate < 0.65**.

**P12 · event triggers do not concentrate the decisive signal.** At the 80th and 90th
percentile of each trigger, P(decisive | fired) divided by the base rate. Prediction:
**every lift < 1.2×**.

**P13 · change detectors stay anti-aligned with burst onset.** `model surprise`,
`|Δ observation|` and `|Δ value head|` scored as AUC against burst onset among decisive
decisions. Prediction: **all three below 0.5**, and at least two with the 95% interval
excluding 0.5 on the low side.

**P14 · the burst structure replicates.** P(decisive | previous decision decisive)
divided by the base rate. Prediction: **≥ 3.0×** (measured 5.95× on discovery).

**P15 · the decisive rate replicates.** Prediction: **between 7% and 12%** (8.7% at
n=68, 9.1% at n=130, 8.2% on `piv8`).

## What would falsify each

- **P11 fails** if any signal reaches 0.65. Constraint 3 is then wrong as stated and the
  veto's first stage is worth rebuilding around whatever cleared it.
- **P12 fails** if any trigger concentrates at 1.2× or better. Event-triggering was
  dismissed on two sets; a third disagreeing means the dismissal was premature.
- **P13 fails** if the detectors sit at or above chance. The anti-alignment is the
  strongest claim in Constraint 3 and the one most likely to be a two-set coincidence —
  it is the prediction I would bet against most readily.
- **P14 fails** below 3.0×. Then the burst structure was an artefact of probe spacing
  rather than of the task, and the explanation for why refractory triggering hurts goes
  with it.
- **P15 fails** outside 7–12%. Then the decisive rate depends on the seed and
  Constraint 1's "replicated" wording must be withdrawn.

**Engagement check, separate from effect size:** the run must yield at least 120 decisive
decisions and at least 40 burst onsets, or P11–P14 report "not tested" rather than
"confirmed". `piv7` gave 181 and 110.

**Committed in advance:** if P11–P15 all hold, Constraint 3 stands as written and no
further signal search is warranted on this arena. If P13 alone fails, the anti-alignment
sentence is struck and the rest survives — the detectors would then be uninformative
rather than misleading, which is a weaker claim but not a contradictory one.

## Outcome — P11, P13, P14, P15 held; **P12 failed**

Run 2026-09-28 on `--seed 2`, 130 episodes, 2031 decisions. Engagement check passed
(159 decisive against a 120 floor, 104 burst onsets against 40), so every verdict below
is a real one. The analysis was run once, exactly as registered.

| | prediction | measured | |
|---|---|---|---|
| **P11** | no signal ≥ AUC 0.65 | max **0.566** (plan spread ¼-H) | **holds** |
| **P12** | every trigger lift < 1.2× | **1.32×** — two triggers | **FAILED** |
| **P13** | detectors < 0.5, ≥2 excluding | 0.229 / 0.228 / **0.188**, all three excluding | **holds** |
| **P14** | burst lift ≥ 3.0× | **6.69×** (52.4% vs 7.8% base) | **holds** |
| **P15** | decisive rate 7–12% | **7.8%** | **holds** |

**P12 is falsified as registered.** At the 90th percentile, `plan-spread jump` reaches
**1.32×** and `near-contact` reaches **1.32×**, both above the 1.2× line. The registered
falsifier says this in plain terms: *"Event-triggering was dismissed on two sets; a third
disagreeing means the dismissal was premature."* It is premature, and the sentence in
Constraint 3 that event triggers "select decisions that matter no more than average" does
not survive as written.

What may **not** be done with this result, and is recorded so it cannot be done quietly
later: `near-contact` scored **0.50×** on `piv7` and **1.32×** here, a reversal across
seeds, and `plan-spread jump` went 1.05× → 1.32×. That instability is the obvious
counter-argument and it is exactly the post-hoc reinterpretation the registration exists
to block. Whether these triggers are unstable noise around 1.0 or a real effect the first
two sets missed is a **new question**, and it needs its own registration and its own
seed. It does not settle this one.

**The consequence committed to in advance was:** *"if P11–P15 all hold, Constraint 3
stands as written and no further signal search is warranted on this arena."* They did
not all hold. A further search on the P12 axis — trigger-based concentration at tight
quantiles — is therefore warranted, and Constraint 3 is amended rather than confirmed.

**P13 held, and held harder than on either prior set** — 0.188 to 0.229 with all three
intervals excluding chance, against 0.166–0.268 before. This was the prediction I
registered as the one I would bet against most readily. It is now the best-supported
claim in Constraint 3 and the only part of it tested three times in the same direction.

---

# Audit · the P11–P15 registration was formal, not real

Prompted by the question "did you reach a conclusion first, or make a prediction
first?". Checked against the git history rather than against memory, and the answer is
**conclusion first**.

```
e8f7514  sim: decision importance is not a run-time attribute of the state
0d3b368  sim: pre-register P11-P15, Constraint 3 as a confirmatory test
```

Constraint 3 was written into `EXPERIMENT.md` and **pushed** in `e8f7514`. P11–P15 were
registered in `0d3b368`, *after*. So the claim was already published when its "test" was
written. That is a confirmation exercise wearing the label of a pre-registration.

**It is worse than the ordering alone.** Every threshold was set from values already
measured on seeds 0 and 1:

| | registered bar | what had already been observed |
|---|---|---|
| P11 | no signal ≥ 0.65 | max 0.637 |
| P12 | every lift < 1.2× | max 1.05× |
| P13 | detectors < 0.5 | 0.166–0.268 |
| P14 | burst lift ≥ 3.0× | 5.95× |
| P15 | decisive rate 7–12% | 8.7%, 9.1%, 8.2% |

Each bar sits just outside the range already in hand. A prediction that cannot fail
unless the next seed departs from the last two is a robustness check, and calling it a
pre-registration overstates what four "holds" are worth. P11, P13, P14 and P15 holding
is evidence of *stability across seeds* — real, but weaker than presented.

**The one genuinely informative result is the failure.** P12 broke despite its bar being
drawn from prior data, which is the hardest way for a registered prediction to fail.
That asymmetry is now the reason to trust it over the four that held.

## The process rule this changes

1. **A registration written after the claim it tests is labelled a ROBUSTNESS CHECK, not
   a pre-registration.** The distinction is the commit order, and it is checkable.
2. **A threshold derived from the data being generalised from must be declared as such**,
   with the observed value printed beside the bar, so a reader can see how much room the
   prediction had to fail in.
3. **Registrations are committed before the claim is written into any document**, not
   after. Where that ordering was not achieved, say so in the registration itself.
4. Retroactively: P11–P15 are **relabelled a robustness check**. Constraint 3's
   confirmatory status in `EXPERIMENT.md` is downgraded to match.

---

# P16–P18 · seed 3, registered on VARIANCE rather than on means

Registered **2026-09-29**, before seed 3 exists, and before any claim about trigger
stability has been written into any document. The quantity being predicted is the
*spread* across seeds, which is not derivable from the per-seed means already in hand.

Observed on seeds 0/1/2, lift at the 90th percentile:

| trigger | s0 | s1 | s2 | mean | SD | range |
|---|---|---|---|---|---|---|
| near-contact | 0.50 | 1.23 | 1.32 | 1.02 | **0.45** | 0.82 |
| plan spread ¼-H | 1.60 | 1.29 | 1.01 | 1.30 | 0.30 | 0.59 |
| return drop | 1.10 | 0.80 | 0.50 | 0.80 | 0.30 | 0.60 |
| \|Δ plan spread\| | 0.99 | 1.11 | 1.32 | 1.14 | 0.17 | 0.33 |
| \|Δ value head\| | 0.99 | 1.23 | 1.07 | 1.10 | 0.12 | 0.24 |
| model surprise | 0.94 | 0.99 | 1.07 | 1.00 | 0.07 | 0.13 |
| \|Δ observation\| | 1.10 | 1.05 | 1.07 | 1.07 | **0.03** | 0.05 |

**P16 · the stability ordering survives a fourth seed.** Recomputing SD over four seeds,
`near-contact` stays in the top two by spread and `|Δ observation|` stays in the bottom
two. *Falsified* if either leaves its half of the ranking.

**P17 · the P12 failure does not replicate as a level.** The four-seed mean lift of
`plan-spread jump` is **below 1.2×**, and the four-seed mean of `near-contact` is below
1.2×. *Falsified* if either mean is 1.2× or above — which would make the seed-2 result a
real effect rather than the top of a noisy range, and event-triggering worth building on.

**P18 · two signals are drifting, not scattering.** `plan spread ¼-H` and `return drop`
fell monotonically across s0→s1→s2 (1.60→1.29→1.01 and 1.10→0.80→0.50). With three
points that is 1-in-3 by chance each. Prediction: **at least one of the two breaks
monotonicity on seed 3.** *Falsified* if both continue to fall — which would mean
something varies with the seed index itself, and every cross-seed comparison in this
project needs re-examining before anything else is believed.

**Engagement check:** ≥120 decisive decisions and ≥40 burst onsets, else "not tested".

**Declared in advance:** P17's bar of 1.2× is inherited from P12 and is therefore *not*
independent of prior data. P16 and P18 are about spread and shape, and no threshold in
either was read off the table above.

---

# P19–P21 · seed 4, the signal inversion at the decision boundary

Registered **2026-09-29**, before seed 4 exists. Motivated by a joint analysis of the
event triggers and the change detectors that had not been done when P11–P15 was written.

Asking each signal two questions instead of one separates them cleanly:

  **(a)** is this decision decisive?  — over all decisions
  **(b)** is this the START of the decisive run? — among decisive decisions only

| signal | mean (a) | mean (b) | inverted |
|---|---|---|---|
| \|Δ observation\| | 0.524 | **0.225** | 3/3 seeds |
| \|Δ plan spread\| | 0.537 | **0.232** | 3/3 seeds |
| \|Δ value head\| | 0.512 | **0.199** | 2/3 seeds |
| model surprise | 0.496 | **0.221** | 1/3 seeds |
| plan spread ¼-H | 0.585 | 0.441 | 2/3 |
| return drop | 0.497 | 0.460 | 1/3 |
| near-contact | 0.471 | 0.472 | 0/3 |

The change detectors sit at **chance on (a) and ~0.22 on (b)**, on every seed. They carry
no information about *whether* a decision is decisive, and strong information about
whether it is the *start* of one — pointing the wrong way. Read forwards that is a
failure; read backwards it is a usable statement: **when observation change is high, you
are mid-burst, not at its onset.**

That has a design consequence that has never been tested. Every gate built here fires
when a signal is HIGH. If the detectors are anti-aligned with onset, the gate that
catches onsets should fire when they are **LOW**.

**P19 · the inversion replicates.** On seed 4, `|Δ observation|` and `|Δ plan spread|`
both score **below 0.35** on question (b), with 95% episode-bootstrap intervals
excluding 0.5. *Falsified* if either reaches 0.35, or if either interval covers chance.

**P20 · the inverted gate beats the forward one at catching onsets.** Fire when
`|Δ observation|` is below its 30th percentile, versus above its 70th, at matched fire
count. Prediction: the low-side gate catches **more burst onsets**, and the paired
difference over resampled episodes **excludes 0**. *Falsified* if the difference spans 0
or favours the high-side gate — which would mean the inversion is an artefact of
conditioning on `decisive` and carries no operational content.

**P21 · the inverted gate still does not beat doing nothing on outcomes.** Net episode
flips per 100 fires for the low-side gate has a 95% interval that **includes 0**.
*Falsified* if it excludes 0 — which would be the first gate in this project to improve
outcomes at an operating point chosen in advance, and would reopen the veto entirely.

**Engagement check:** ≥120 decisive and ≥40 onsets, else "not tested".

**Declared in advance:** 0.35 in P19 is a round number below the observed 0.199–0.232 and
above chance; the 30th/70th percentiles in P20 were not tuned — no split other than
30/70 has been evaluated on any seed. **P21 is registered expecting to hold**, i.e. I
expect the inverted gate to find onsets and still not change outcomes, because the
decisive set is 8% of decisions and holding rescues only ~23% of doomed actions. If P21
fails, the right response is delight, not a rescue of Constraint 3.

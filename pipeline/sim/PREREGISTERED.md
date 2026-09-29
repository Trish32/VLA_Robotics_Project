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

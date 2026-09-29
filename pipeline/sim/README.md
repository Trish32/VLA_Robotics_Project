# MuJoCo pick-and-place arena

A cube, a bowl, a distractor ball, and a 4-DoF arm that has to put the cube in the
bowl. Small on purpose. It exists to supply the one thing the real demonstrations
cannot, and to make one claim checkable that was previously only assertable.

## Why this exists

`pipeline/world_model/Plan.md` §1 measured why the action-conditioned world model's
object pathway was dead, and the answer was not the architecture:

| constraint | real data | here |
|---|---|---|
| training episodes | **2** usable; validation best at epoch 0, every run overfits immediately | as many as you want to generate (~0.03 s each) |
| object tracks | normalised image-plane centroids, conflating object motion with camera geometry | metres in the world frame, centre **and** extent |
| single-step signal | cube moves 0.00084 image units — half a pixel, R² = 0.016 from the action | ~12 mm per step, metric |
| segmentation quality | unmeasurable, no ground truth exists | ground-truth instance masks every frame |
| task reward | none shipped; had to be inferred and believed | simulator reports it, so an inferred reward can be *scored* |

## What it is not

**There is no robot, so there is no sim-to-real result here.** Domain randomisation
supports a sim-to-**sim** robustness claim and nothing stronger. Anything framed as
transfer in this directory means transfer to a randomisation level the policy was not
trained at.

GR00T is not the policy here either, and the reason is embodiment rather than
preference: its action space is a 29-DoF GR1 humanoid, this arena has a 4-DoF Cartesian
gripper, and bridging them would need retargeting that changes what the experiment
measures. GR00T keeps the real-data path; the sim runs a policy sized to it.

## The arena

```
pipeline/sim/
  arena.xml       MJCF: table, 3-link polar arm on a yawing base, parallel gripper,
                  cube (free), ball (free, distractor), octagonal bowl (static)
  arm.py          closed-form IK/FK — pure numpy, tested without MuJoCo
  env.py          PickPlaceEnv: reset/step/render, RGB + metric depth + instance masks
  perception.py   frame -> world-frame boxes, oracle masks or SAM+CLIP
  policy.py       ScriptedPickPlace (the demonstrator) and ChunkPolicy (BC, chunked)
  collect.py      episodes in the LeRobot layout world_model/data.py already reads
  reward.py       success from perception alone, scored against simulator truth
  loop.py         perceive -> predict -> choose -> act, closed
```

The arm is a polar 3-link chain so the inverse kinematics is analytic — a Cartesian
command becomes joint targets with no solver between the policy and the robot. The
physics is not simplified: real actuators, real contacts, real gravity. Only the
kinematic redundancy is gone.

Throughput on an M-series CPU: **214 env-steps/s** (≈5× real time), ~5 ms per rendered
frame at 320×240 for RGB + depth + segmentation together.

## Results

### The demonstrator

| condition | success |
|---|---|
| canonical scene | 10/10 |
| full domain randomisation | 39/40 |

### Dynamics, trained on 400 simulated episodes

Held-out one-step RMSE, standardised units, 40-episode holdout (7,447 transitions):

| | identity | constant velocity | learned |
|---|---|---|---|
| whole latent | 0.41858 | 0.65529 | **0.26067** (+37.7% / +60.2%) |
| object slots | 0.03441 | 0.02517 | **0.02124** (+38.3% / +15.6%) |
| the cube alone | 0.03170 | — | **0.01181** (+62.7%) |

**This closes Plan.md §1.** On real demonstrations the object pathway *tied* an identity
predictor — the model could not beat "assume nothing moved". With metric 3-D tracks and
two orders of magnitude more episodes it beats both trivial predictors, which is the
outcome the measured diagnosis there predicted.

The static objects are reported honestly rather than flatteringly: the bowl is bolted
down, so its identity error is exactly zero and the model's error on it is purely
invented motion (0.00067), not an improvement.

### Perception-derived reward, scored against the simulator

`reward.py` labels success from perception alone — object positions from masks + depth,
gripper state from proprioception — which is exactly the information a real robot has.
32 episodes, a quarter of them deliberately sabotaged into near-misses:

| segmenter | precision | recall | **F1** | specificity | accuracy | tp / tn / fp / fn |
|---|---|---|---|---|---|---|
| ground-truth masks | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 23 / 9 / 0 / 0 |
| SAM + CLIP | 1.000 | 0.913 | **0.955** | 1.000 | 0.938 | 21 / 9 / 0 / 2 |
| OWLv2 + SAM | 1.000 | 0.696 | 0.821 | 1.000 | 0.781 | 16 / 9 / 0 / 7 |

F1 rather than agreement, because accuracy hides which error is happening — a false
positive teaches a policy that a miss was a success, a false negative merely discards a
good demonstration — and it drifts with the success rate of whatever policy generated
the episodes, so it is not comparable between runs. **Precision is 1.000 on all three:
the rule never calls a miss a success.**

The oracle row validates the *rule*: it rejects a cube 7.9 cm from the bowl centre
against a 4.9 cm threshold derived from the *perceived* bowl, and accepts sabotaged
runs that missed their aim point and succeeded anyway.

### The bowl was not a bowl

SAM+CLIP scored **0.625 accuracy** here until the arena was fixed, and chasing that
through three components found the cause in none of them.

SAM was cleared first: dumping every mask against ground truth showed it segmenting the
bowl at **IoU 0.977** while CLIP argmaxed that exact mask to "a small dark cube".
Reading the similarity matrix down its columns rather than across its rows recovered six
points ([bug_log.txt](../bug_log.txt) [S2]); a ten-prompt sweep moved it by one frame in
thirty. So the residue was written up as CLIP's weakness on small low-texture objects.

That was wrong. Swapping CLIP for an open-vocabulary **detector** — OWLv2, which should
be strictly better at localising from text — made the bowl *worse*: 4/24 against CLIP's
19/30, while the same detector found the cube at 0.599 and the ball at 0.874 in the same
frames. A cube is a cube and a sphere is a sphere. The bowl was eight upright boxes in a
ring, and a shape-honest sweep says so:

| query | old shape | rebuilt |
|---|---|---|
| "an orange plastic bowl" | 0.003, **0/12** | **0.367, 9/12** |
| "an orange bowl" | — | **0.424, 10/12** |
| "a ring of orange blocks" | **0.460, 11/12** | 0.390, 12/12 |

The detector was never failing; it was describing what was there. CLIP scored better
only because being a fuzzier matcher made it more forgiving of a label that did not fit.

Rebuilding the bowl as 12 segments flared 30 degrees, **with no change to any
perception code**, took SAM+CLIP from 0.625 accuracy to **F1 0.955**. The demonstrator
is unaffected (39/40), which is the check that the fix changed the appearance and not
the task.

Two things worth keeping from this. The perception number was largely measuring a
modelling error in the arena, and no further work on the segmenter could have found
that — what found it was swapping in a component that was *more* correct and reading its
disagreement as information. And the upgrade that exposed the bug is not the upgrade
worth keeping: OWLv2+SAM still trails SAM+CLIP (F1 0.821 vs 0.955), because its
box-prompted mask estimates the bowl's *extent* worse, and the seating test is computed
from that extent.

### The closed loop

Perceive -> predict -> choose -> act, with the chosen action executed and ground-truth
success counted. The loop reads simulator state only to report; everything it acts on
comes through masks and depth. Mean perception error **1.7 cm**, which is the
visible-surface bias of a single camera rather than a segmentation failure. (This read
1.4 cm before the bowl was rebuilt with flared walls; the wider rim gives the single
view more surface to miss.)

Driven by perception instead of ground truth, the pick-and-place solves **95.8% of
episodes [87.5, 100.0]** with no planning — 24 episodes, 95% bootstrap CI.

### Candidate generation, and what Oracle@k said about it

Because the simulator rewinds, `--pick oracle` can execute every candidate for real and
keep the one that turned out best — the **ceiling** of per-decision selection rather
than another model's opinion of it. At `--oracle-depth episode` each candidate is rolled
all the way to completion and scored on the simulator's own success flag, which removes
every proxy from the diagnostic.

| candidate source | a candidate succeeds | the chosen one succeeds | headroom | success |
|---|---|---|---|---|
| perturb 0.003 (default) | 96.7% | 96.7% | **+0.0** | 91.7% |
| perturb 0.012 (4x wider) | 70.0% | 66.7% | +3.3 | 41.7% |
| belief-conditioned | 36.7% | 35.0% | +1.7 | 0.0% |
| **affordance** | **91.7%** | 88.9% | **+2.8** | **91.7%** |

Perturbing a policy's chunk produces candidates with **identical outcomes** — headroom
exactly zero — so no ranking rule can do anything but pick one of eight equivalent
actions. Perturbing harder creates differences only by making candidates worse:
diversity and quality are the same knob turned in opposite directions.

`affordance.py` breaks that. It derives a **graspable radius** from the object's
perceived extent minus what the jaw needs, and a **placeable radius** from the
container's extent minus the object's diagonal, then spreads K plans on a ring inside
both. Every candidate is a valid plan — a real grasp point, a real release point — so
the set differs substantially without any member being a degraded copy. It buys nearly
the headroom of a 4x perturbation at 22 points more reachability.

Everything comes from the perceived boxes, never the arena's dimensions; an object wider
than the jaw is reported as having **no** graspable region rather than a small one,
because a plan that cannot execute is worse than no alternative plan.

**Selection still loses**, and with affordance candidates it loses much harder — greedy
distance-optimal picking scores 0.0%, because the candidates now differ enough for a bad
objective to do real damage. Full sequence and the four eliminated hypotheses in
[world_model/RESULTS.md](../world_model/RESULTS.md).

### How well does the scorer rank, and where does it stop being right

Success rate answers "did planning help" one bit at a time. `ranking_metrics.py` scores
the *ordering* against the simulator's own rewind-and-execute cost, at the decisions the
planner actually faced.

| candidate source | ranking measured along | pairwise | Kendall τ | regret@1 | SNR |
|---|---|---|---|---|---|
| ~~perturbation~~ *(invalid)* | the policy's trajectory | 0.564 | +0.127 | 0.00248 | 1.09 |
| **affordance** | the policy's trajectory | **0.646** | **+0.291** | 0.00403 | 1.18 |
| **affordance** | **the scorer's own trajectory** | **0.421** | **−0.157** | **0.01930** | 1.21 |

**The perturbation row is invalid and may not be cited against the planner or the
scorer.** Its candidates are outcome-equivalent — Oracle@k headroom +0.0, SNR 1.09 —
so a perfect ranker and a coin score the same on it, and a negative result is a property
of the test rather than of what is being tested. It is why four early ablations read
"no". **Affordance is the primary ranking test**; `ranking_metrics.py` prints an
invalidity warning when asked for `--candidates-from action`.

The last two rows differ only in **whose trajectory the episode follows**. Same model,
same candidates, same metric — and the ranking **inverts** once the scorer's own choices
are what produced the state. That is distribution shift, measured rather than inferred:
every transition the dynamics was trained on is one a competent demonstrator took from
a state that demonstrator reached, so "what if we grasped 2 cm left instead" is
extrapolation, and acting on the answer is what walks the episode further into it. It
also re-reads the earlier reachability collapse (91.7% under `--pick first` → 40% under
`--pick best`) as a compounding loop rather than an unlucky objective.

SNR is the third column that matters: in every configuration the spread between
candidates is only ~1.1–1.2× the model's own error, so even the good row sits on a
margin barely wider than the noise.

`collect.py --counterfactual N` adds the missing coverage — from states along an
on-policy episode it branches under a *different* plan, so the states stay on the
demonstrator's distribution while the actions leave it. Scored **state-matched**, with
both models ranking the same candidates at the same decisions:

| dynamics trained on | pairwise | Kendall τ | regret@1 |
|---|---|---|---|
| 400 on-policy *(driving)* | 0.443 | −0.114 | 0.01787 |
| 1,903 on-policy *(quantity control)* | 0.468 | −0.065 | 0.01517 |
| **400 seeds → 2,088 branched** | **0.514** | **+0.029** | **0.01154** |

Coverage does about twice what raw quantity does, and only coverage crosses zero. Each
model driving *itself* reads +0.671 instead of +0.029 — a model that chooses better ends
up where ranking is easier, and `--also-score` exists to strip that out.

### Ranking only where the model was fitted

`domain.py` filters candidates by how far they sit from the policy's own chunk, in
standardised action units, and rolls back to the policy when too few survive. Of three
calibrated signals only that one works: **state energy never leaves its in-distribution
range** (0.38–1.43 across the decisions this gate exists for), so what goes off-manifold
is the action, not the scene. Filtering at 0.68 keeps half the candidates, loses no
decisions, and moves τ from −0.134 to +0.062.

It does not rescue the loop:

| dynamics | policy-first | greedy | greedy + gate |
|---|---|---|---|
| 400 on-policy | **95.8%** | 0.0% | 0.0% |
| counterfactual | **95.8%** | 0.0% | 16.7% |

Both interventions are needed to move the cell at all, and 16.7% is still 79 points
below running the policy. The gate is a containment measure — it stops the scorer being
trusted where it is wrong — not a way to make greedy selection work.

### Why none of it helps: the scorer was never the problem

Three controls, 24 episodes each, 95% bootstrap CI over episodes:

| rule | candidates | success |
|---|---|---|
| no planning | — | **95.8% [87.5, 100.0]** |
| `--pick first` | 8 | **95.8% [87.5, 100.0]** |
| `--pick best`, **K=1** | 1 | **95.8% [87.5, 100.0]** |
| `--pick best`, K=8 | 8 | 0.0% [0.0, 0.0] |
| **`--pick oracle`**, K=8 | 8 | **0.0% [0.0, 0.0]** |

Greedy with a single candidate reproduces the policy exactly, so the machinery is sound.
And **perfect per-decision selection scores zero too** — the oracle rewinds, executes
every candidate to completion and keeps whichever really succeeded. No ranking rule can
help when the best possible one does not.

`compare_traces.py` says why. Against the policy run on the same seeds, the greedy run
changes its chosen plan at **80.3%** of consecutive decisions, picks 7.62 of 8 distinct
plans per episode, holds one choice for at most 3.96 decisions out of 90, and executes
actions **1.02× the policy's own step size** away from what the policy wanted. The grip
command never differs — the two runs disagree about where to go, not what to do there.

**The candidates are whole plans and the controller re-decides every `execute` steps**,
so the arm is redirected before any plan is carried out. Plan-level choices need
plan-level commitment.

### Read it as coverage, not as a threshold

The natural next sentence — "the fracture is at K=28, the median stage duration" — is
the wrong shape for what was measured. Success does not step from 0 to its ceiling at a
threshold; it climbs with the **fraction of a stage the commitment covers**:

| K | stages the choice outlasts | perfect-selection success |
|---|---|---|
| 8 | 0.3% | 0.0% |
| 16 | 27.4% | 0.0% |
| 32 | 71.8% | 37.5% [16.7, 58.3] |
| 64 | ~97% | **66.7% [45.8, 83.3]** |

**Coverage is the variable; the median dwell is a landmark on it, not a cut-off.** K=32
already exceeds the 28-step median and still loses most episodes, because a median
leaves half the stages uncovered and `transfer` alone runs to 61 steps. K=64 covers
nearly all of them and roughly doubles success again. Calling 28 a threshold would
predict a plateau just above it, and there is none.

What the median *is* good for is explaining the floor: below it, coverage is near zero
and so is success, which is why everything from K=1 to K=16 is flat at 0.0%.

### Refusing instead of choosing

`veto.py` lets the policy act and gives the world model one job: refuse a disaster.
`calibrate_veto.py` rewinds at sampled decisions, runs each action to completion, and
sweeps every available signal against that ground truth (492 decisions, 28.5% doomed):

| signal | AUC |
|---|---|
| **ensemble spread** | **0.587** |
| predicted collision | 0.466 |
| predicted return drop | 0.457 |
| predicted instability | 0.396 |
| predicted return at horizon | 0.353 |

**Every value-derived signal is at or below chance; the only informative one is the
ensemble disagreeing with itself** — which inverts the gate's own design, since spread
was written in as a *disqualifier*.

The structural explanation for that is **refuted**. An earlier version of this page said
the latent "carries the cube, the bowl and the ball and no gripper", so the model could
not represent the mechanism. `probe_latent.py` fits a linear probe and recovers finger
opening at R² 1.000, tip-to-cube offset at 0.999, pad-to-cube clearance at 0.973 and
"is the cube held" at 0.903. The gripper is in there; what fails is downstream of it.

**And the edge does not survive a confidence interval.** Precision peaks at 0.404
against a base rate of 0.285, and the best operating point nets about four flipped
episode outcomes per hundred fires — but bootstrapped over **episodes** rather than
decisions, every operating point's 95% CI includes zero (the best reads +4.1 [−1.5,
+10.0]). A 120-episode closed-loop ablation agrees: 79.2% without the veto, 80.8% with
it. There is a ceiling underneath either way — holding instead of acting rescues only
**10.0%** of doomed actions, so even a perfect detector would buy little.

### The small policy

`train_bc.py` clones the demonstrator into `ChunkPolicy`: a 2-layer MLP from
proprioception plus tip-relative object offsets to an 8-step action chunk.

| | held-out chunk MSE |
|---|---|
| zero action | 0.182147 |
| **behaviour cloning** | **0.002924** (+98.4%) |

Object offsets are relative to the tip, not absolute: handed absolute coordinates a
small MLP memorises the table region the cube was sampled from and stops working the
moment the layout is randomised.

## Traps

Things about this arena that are load-bearing, non-obvious, and cost real debugging time
when they were learned. Each one is a measurement, not a warning.

### The timestep is coupled to the policy's dwell counts

`ScriptedPickPlace` waits in **steps**, not seconds: `grip_hold = 14` means fourteen
`mj_step` calls for the fingers to close, `settle_hold = 8` means eight for a waypoint
to settle. `mujoco` `<option timestep>` is 0.002 s, so those are 28 ms and 16 ms — and
nothing in the code ties them together.

Change the timestep and the demonstrator silently stops working:

| timestep | blow-ups / 200 | **success** |
|---|---|---|
| 0.002 (as shipped) | 11 (5.5%) | **75.5%** |
| 0.001 | 1 (0.5%) | **31.0%** |

Halving the timestep looks like a clean fix for solver instability — it removes almost
every blow-up — and it destroys the task, because every dwell now lasts half as long in
simulated time. The fingers get 14 ms to close on a cube they need ~28 ms for, so the
grasp is released before it is made.

**If you change `timestep`, scale `grip_hold` and `settle_hold` by the same factor**, or
the arena will report a physics improvement that is a policy regression. The check that
catches it is one line: the demonstrator must stay at 10/10 canonical and ~40/40 under
domain randomisation. Any change to the arena that moves those numbers changed the task,
whatever else it also did.

The same coupling applies to `max_steps` (360 steps = 0.72 s of simulated time per
episode at the shipped timestep) and to `ChunkPolicy`'s horizon of 8.

### Contact stiffness is not a free knob either

The free objects once carried `solref="0.006 1"` — stiffer than MuJoCo's 0.02 default —
and that, not any grasp geometry, is what launched the cube at 83 m/s in 5.5% of
plan-randomised episodes. The arena now defaults to `solref="0.03 1"` and the finger pads
override back to 0.006, because that value is what stops a rigid pinch jittering.

**It is an 11x reduction, not an elimination**: 5.5% → **0.5%** over 2000 episodes. A
first check at 200 episodes read 0.0% and would have retired the `SANE_POSITION_M` guard
that still catches the remaining 1 in 200. Details and the refuted first hypothesis in
[bug_log.txt](../bug_log.txt) [S17].

### `--keep-failures` keeps task failures, never physics failures

The flag exists so the dynamics model sees failures. It was also keeping episodes where
the solver ejected an object to 9.6×10⁷ m, and since the loss is standardised over the
whole dataset, 30 such episodes set the scale for the other 470 — every real 12 mm cube
motion became ~1e-4 standardised and contributed no gradient. The physics check now runs
*before* the flag and the two counts are reported separately ([bug_log.txt](../bug_log.txt)
[S16]).

### A "done" marker must mean the work finished, not the process did

Long measurements here are launched in the background and waited on by watching for a
marker string. Written the obvious way, the marker lies:

```bash
set -e
conda run -n env python -m pipeline.sim.calibrate_veto ... | grep -v WARNING
echo VETO_GOAL_DONE            # fires whether or not the python above worked
```

`set -e` does not save it: a pipeline's exit status is its **last** command's, so `grep`
succeeding hides the Python process exiting non-zero. A 20-minute run that died on an
argparse error in under a second reported itself finished, and the waiter woke up with
no result ([bug_log.txt](../bug_log.txt) [S19]). Write it so the marker is bound to the
work:

```bash
set -euo pipefail
if conda run -n env python -m ... | grep -v WARNING; then
  echo TAG_DONE
else
  echo TAG_FAILED
fi
```

The general form of this rule shows up twice more in this project: `next.reward` had to
fire on the simulator's success flag rather than on the policy reaching its terminal
stage ([S13]), and a task marker reading `started` is not evidence of a result. **A
proxy for completion diverges from completion exactly in the cases worth knowing
about.**

### The env carries its own randomisation stream

`PickPlaceEnv(seed=k)` does not mean "episode k". Each `reset()` advances an internal
RNG, so episode 89's scene only exists after 89 resets. A probe that resets once and
expects to reproduce episode 89 reproduces episode 0 — which is exactly how a blow-up
that had just been measured failed to reproduce on the first attempt at isolating it.

## Running it

```bash
# 400 demonstrations with domain randomisation (~3 min, CPU)
conda run -n simple_bev_vldrive python -m pipeline.sim.collect \
    --out pipeline/sim/data/pick_place_400 --episodes 400

# fit the dynamics on them
conda run -n simple_bev_vldrive python -m pipeline.world_model.train \
    --data pipeline/sim/data/pick_place_400 --epochs 25 --holdout 40 \
    --out pipeline/assets/world_model_sim

# how good is a reward you could actually compute on a real robot?
conda run -n openmask3d_vl python -m pipeline.sim.reward --episodes 32 \
    --perception oracle          # or: --perception sam

# the closed loop, and its ablations
conda run -n simple_bev_vldrive python -m pipeline.sim.loop --episodes 24
conda run -n simple_bev_vldrive python -m pipeline.sim.loop --episodes 24 --no-plan

# the control that separates "the scorer ranks badly" from "the candidates are bad"
conda run -n simple_bev_vldrive python -m pipeline.sim.loop --episodes 24 \
    --policy-noise 0.004 --pick random     # or: --pick best / --pick first

# how the ranking is wrong, not just whether. --drive best measures it along the
# scorer's OWN trajectory, which is where distribution shift shows up.
conda run -n simple_bev_vldrive python -m pipeline.sim.ranking_metrics \
    --candidates-from affordance --drive first --episodes 12
conda run -n simple_bev_vldrive python -m pipeline.sim.ranking_metrics \
    --candidates-from affordance --drive best --episodes 12

# off-policy actions from on-policy states: the coverage the demonstrations lack
conda run -n simple_bev_vldrive python -m pipeline.sim.collect \
    --out pipeline/sim/data/cf_400 --episodes 400 --randomise-plans \
    --keep-failures --counterfactual 6

# read the veto threshold off a curve instead of guessing it
conda run -n simple_bev_vldrive python -m pipeline.sim.calibrate_veto \
    --episodes 24 --probes 8 --signal spread

# clone the demonstrator into the small policy, then run the loop on it
conda run -n simple_bev_vldrive python -m pipeline.sim.train_bc \
    --data pipeline/sim/data/pick_place_400 --epochs 40
conda run -n simple_bev_vldrive python -m pipeline.sim.loop --episodes 24 \
    --policy bc --candidates-from belief
```

Tests run without MuJoCo where they can (`test_sim_arm.py` parses `arena.xml` and
checks it still agrees with the IK's constants) and skip cleanly where they cannot.

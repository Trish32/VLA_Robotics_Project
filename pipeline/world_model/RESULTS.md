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

## Object dynamics · MEASURED, AND NEGATIVE — with the cause identified

The earlier version of this section blamed the data: one episode of a single binary
foreground mask. That was addressed — `extract_tracks.py` runs this repo's own SAM ViT-H
and CLIP over the raw `cube_to_bowl_5` video and produces **per-instance tracks for all
five episodes**. The object pathway still does not learn, and the reason turns out not to
be the one assumed.

### The tracks themselves are good

| | |
|---|---|
| instances found | `a small dark cube`, `a green rubber ball` (+ the bowl on one episode) |
| cube travel | **0.67 – 0.81** normalised image units — table → gripper → into the bowl |
| ball travel | **0.046 – 0.085** — correctly static |
| measured (not forward-filled) frames | 2,048 of 2,343 usable transitions |

Verified by drawing them on the frames, which is the only check that works: a track file
looks healthy whatever it contains.

### The model still ties identity, at every step size

Split 2 train / 1 val / 1 test **episodes**, after dropping one idle episode (below).

| stride | identity | constant velocity | learned |
|---|---|---|---|
| 1 frame | 0.09750 | 0.13638 | **0.09742** (+0.1%) |
| 10 frames | 0.31578 | 0.50117 | **0.31514** (+0.2%) |

Both runs restore **epoch 0** — the best validation score is at initialisation, so
training never improves held-out performance. The model ties identity because it *is*
identity; it never usefully leaves its initialisation.

### Why: the per-step signal is below the measurement noise

| | |
|---|---|
| median per-step cube displacement | **0.00084** — about half a pixel at 640 px |
| linear R² from the full action, 1 frame | **0.008** (x), **0.016** (y) |
| linear R² from the full action, 30 frames | 0.055 (x), **0.139** (y) |

The action-to-object relationship is not absent, it is **buried**: it grows monotonically
with horizon, from R² = 0.016 at one frame to 0.139 at one second. Asking the model to
predict one frame ahead is asking it to resolve a signal the extraction cannot measure.
Training at stride 10 confirmed the diagnosis without fixing the outcome — the signal
improves but two training episodes is still far too little to fit it.

**So per-instance tracking was necessary and not sufficient.** The binding constraints
are now named rather than guessed: single-step displacement under the tracker's noise
floor, and two training episodes.

### Constant velocity is the *wrong* baseline for these tracks

Worth recording because it inverts the robot case: for the object tracks identity
(0.0975) beats constant velocity (0.1364), because centroid jitter makes the velocity
channel mostly noise and carrying it forward amplifies it. For the robot the order is the
other way round. Velocity integration is therefore chosen **per stream**; applying one
setting to both started the object pathway from the worse of the two predictors and cost
it 19% against identity.

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

## In simulation, the object pathway works · MEASURED

The object result above is negative and its stated cause was data, not architecture.
`pipeline/sim/` tests that claim by supplying the data: a MuJoCo cube-to-bowl arena
(see its [README](../sim/README.md)) generating episodes at ~0.03 s each with
ground-truth metric object boxes. 400 episodes, 320 train / 40 validation / 40 test,
split by episode as everywhere else; 7,165 held-out transitions rather than the 170 a
one-episode holdout was giving.

| predictor | whole latent | object slots | the cube alone |
|---|---|---|---|
| identity | 0.41858 | 0.03441 | 0.03170 |
| constant velocity | 0.65529 | 0.02517 | — |
| **learned dynamics** | **0.26188** | **0.02080** | **0.01153** |
| | +37.4% / +60.0% | **+39.5% / +17.3%** | **+63.6%** vs identity |

Nothing about the model changed. What changed is that the per-step object signal is
~12 mm of real motion instead of half a pixel, and that there are 320 training episodes
instead of two.

The static objects are reported as static rather than as wins: the bowl is a fixed
body, so its identity error is exactly zero and the model's 0.00040 is **invented
motion**, not an improvement. Printing a percentage there produced "-34959.8%" before
the guard was fixed.

**Scope.** This is a measurement about the object pathway, not about
`cube_to_bowl_5` — those two episodes still tie identity and that result stands. And
simulated tracks are exact, so these figures exclude tracking error entirely. Read them
as the ceiling the pathway reaches when the measurement is not the limiting factor.

## Closing the loop · MEASURED

Everything above scores rollouts that never happen. In the arena the chosen action is
executed, so the ranking can be credited or not.

The loop runs entirely off perception — masks, depth, unprojection — and reads
simulator state only to report. Mean perception error **1.4 cm** (the visible-surface
bias of a single camera, not a segmentation failure).

### A reward the robot could actually compute

`sim/reward.py` labels success from perceived object positions plus the gripper
command, which is exactly a real robot's information set, and the simulator says
whether it was right. 32 episodes, a quarter sabotaged into near-misses:

| segmenter | agreement | tp | tn | fp | fn |
|---|---|---|---|---|---|
| ground-truth masks | **100%** | 22 | 10 | 0 | 0 |
| SAM + CLIP | 62.5% | 10 | 10 | **0** | 12 |

Reported as a confusion matrix with precision, recall and F1 rather than as accuracy.
Accuracy is the wrong headline twice over: it hides which error is happening — a false
positive teaches a policy that a miss was a success, a false negative merely discards a
good demonstration — and it drifts with the success rate of whatever policy generated
the episodes, so it is not comparable across runs. The oracle segmenter scores
**precision 1.000, recall 1.000, F1 1.000**.

That validates the **rule**: it rejects a cube 7.9 cm from the bowl centre against a
4.9 cm threshold derived from the *perceived* bowl, and accepts two sabotaged runs that
missed their aim point and succeeded anyway.

The SAM row is a perception result, and chasing it through three components found a
cause in none of them. SAM segments the bowl at IoU 0.977 while CLIP argmaxes that same
mask to "a small dark cube" ([bug_log.txt](../bug_log.txt) [S2]); swapping CLIP for an
open-vocabulary detector made it *worse*, while that same detector found the cube and
ball confidently; and a shape-honest query sweep scored "a ring of orange blocks" at
0.460 / 11-of-12 against "an orange plastic bowl" at 0.003 / 0-of-12. **The arena's bowl
is not bowl-shaped**, the detector was right, and the perception number was largely
measuring a modelling error — see [S11] and the
[arena README](../sim/README.md#the-bowl-is-not-a-bowl).

### The perturbation benchmark is INVALID as a ranking test · READ THIS FIRST

Everything below that ranks **perturbation candidates** — a policy chunk plus Gaussian
noise — measures nothing about a scorer, and none of it may be cited as evidence
against the planner or the scorer. The benchmark is invalid on its own measurements,
not on reinterpretation:

| | perturbation candidates | affordance candidates |
|---|---|---|
| Oracle@k headroom (episode depth) | **+0.0 pts** | +2.8 pts |
| a candidate succeeds / the chosen one does | 96.7% / 96.7% | 91.7% / 88.9% |
| pairwise ranking accuracy | 0.564 | **0.703** |
| Kendall τ | +0.127 | **+0.291** |
| SNR (candidate spread ÷ model error) | 1.09 | 1.21 |

**Headroom +0.0 is disqualifying by itself.** Executing every candidate to completion
and keeping whichever really turned out best produces the same success rate as taking
the first one: the candidates are outcome-equivalent, so *no* ordering of them can
change what happens. A test on which a perfect ranker and a random one score identically
cannot distinguish a good scorer from a bad one, and a negative result on it is a
property of the test.

Widening the perturbation does not repair this. It makes candidates differ only by
being worse — reachability 96.7% → 70.0% → 36.7% as the spread grows — so diversity and
quality are the same knob turned in opposite directions. There is no setting at which
perturbation is a valid ranking test.

**Concretely withdrawn.** "The scorer ranks at chance" was read off oracle-rank values
of 0.507 / 0.508 / 0.491 / 0.511 / 0.542 on perturbation candidates. Chance is what a
ranker *must* score when every candidate leads to the same outcome, so those numbers
measure the candidate set. On affordance candidates the same scorer ranks at pairwise
0.703 / τ +0.291 along the policy's trajectory — well above chance. The withdrawn claim
appeared in this file, in `sim/README.md`, and in the docstrings of `affordance.py` and
`veto.py`; all four now carry the correction.

**What perturbation results still support.** They remain valid evidence about the
*candidate set* and about closed-loop behaviour: that perturbing a chunk produces
outcome-equivalent alternatives, that widening the perturbation degrades reachability,
and the success rates of the loop itself. They are invalid only as a measurement of
ranking quality.

**The primary ranking test is `--candidates-from affordance`**, scored by
`sim/ranking_metrics.py` (pairwise accuracy, Kendall τ, Spearman ρ, regret@k, SNR) and
reported in "How well does the scorer actually rank?" below. `ranking_metrics.py` prints
an invalidity warning when run with `--candidates-from action`.

### Does planning before acting help? · PERTURBATION CANDIDATES — INVALID AS A RANKING TEST

The first version of this section reported a plan/no-plan grid where planning lost
everywhere, then a random-pick control showing the scorer still beat chance. That is
two of the three questions. The third — **is there anything in the candidate set worth
selecting at all** — needs an oracle, and the arena can supply one: rewind the
simulator, execute every candidate for real, keep the one that actually turned out best.

`--pick oracle` is not a policy. It is the **ceiling** of per-decision selection, and
the gap between it and "just run the policy" is the only honest evidence that better
selection was ever available to win.

Oracle rank is where the chosen candidate sat in the true ordering, normalised so 0 is
the best of k, **0.5 is chance**, and 1 is the worst. 16 episodes per cell.

| policy | candidates | rule | success | oracle rank |
|---|---|---|---|---|
| scripted | action | `first` — the policy's own chunk | **100.0%** | 0.453 |
| scripted | action | `random` | 56.2% | 0.511 |
| scripted | action | `best` — the scorer | 100.0% | 0.507 |
| scripted | action | **`oracle` — perfect selection** | **75.0%** | 0.000 |
| scripted | belief | `first` | 93.8% | 0.379 |
| scripted | belief | `random` | 6.2% | 0.488 |
| scripted | belief | `best` | 12.5% | 0.508 |
| scripted | belief | **`oracle`** | **50.0%** | 0.000 |
| bc | action | `first` | 50.0% | 0.487 |
| bc | action | `best` | 50.0% | 0.491 |
| bc | action | **`oracle`** | **50.0%** | 0.000 |
| bc | action | `best`, MPPI | 31.2% | 0.542 |

What this table supports, now that the benchmark above is marked invalid:

1. ~~**The scorer ranks at chance.**~~ **WITHDRAWN.** Oracle rank 0.507 / 0.508 / 0.491
   / 0.511 / 0.542 is what *any* ranker scores when the candidates are
   outcome-equivalent, including a perfect one. This row measures the candidate set.
   The scorer's actual ranking quality is in
   "How well does the scorer actually rank?", on affordance candidates.
2. **The policy's own chunk ranks better than chance** (0.453, 0.379, 0.487) — still
   valid, and still the reason the policy-first baseline is the one to beat.
3. **Perfect selection does not beat no selection** (75.0% vs 100.0%, 50.0% vs 93.8%)
   — still valid as a statement about *these* candidates, and it is exactly why they
   are the wrong candidates. It is not evidence that selection is worthless in
   general; `--candidates-from affordance` has headroom +2.8.

### It is not myopia — MEASURED, hypothesis refuted · PERTURBATION CANDIDATES

The obvious reading of the above is that a 4-step cost is too short-sighted. That is
testable: widen the window the oracle commits to and measures over, and it should
improve. It gets worse.

| horizon / execute | `first` | `best` | `oracle` |
|---|---|---|---|
| 8 / 4 | 87.5% | 68.8% | 75.0% |
| 16 / 16 | 87.5% | 56.2% | 25.0% |
| 32 / 32 | 75.0% | 18.8% | 25.0% |

Everything degrades with fewer replans, which is expected — but the oracle degrades
fastest, so a longer measurement window is not what it was missing.

**What the mechanism is has not been isolated, and is not claimed.** One probe found the
oracle exploiting a physics instability — a single episode in twelve where the cube was
flung 21 m — which shows the objective *is* exploitable, but at 1-in-12 it cannot
explain a 3-episode gap. Median lift under the oracle is 19.9 cm against the baseline's
20.1 cm, so its failures are otherwise ordinary. (An earlier draft of this paragraph
read the 190 cm *mean* as systematic launching. It was one episode — the same
mean-over-a-heavy-tail trap this page documents for the proprioceptive result.)

**What this licenses.** Stop tuning proposers and ranking rules: a hand-written
instantaneous cost over geometric distances is not a good enough proxy for task success
that greedily optimising it helps, and that is true of the scorer's terms as much as the
oracle's. The concrete next step joins this work to the reward work — train `ValueHead`
(§4, still deliberately untrained) on the **perception-derived reward**, and score
predicted task success instead of predicted distance.

### A better objective does not rescue it either · PERTURBATION CANDIDATES — CONCLUSION NARROWED

The section above concluded that the objective was the binding constraint, and named
the fix: score predicted task success instead of predicted distance. That is now built
(`sim/train_value.py`, Plan.md §4) and the value head is **23x better than the distance
term it replaces** at predicting discounted return — held-out MSE 0.001417 against
0.032412.

It changes nothing.

| scorer | rule | success | oracle rank | planner overrode |
|---|---|---|---|---|
| distance | `first` | 93.8% | 0.450 | — |
| distance | `best` | 93.8% | 0.478 | 71.0% |
| **value** | `first` | 93.8% | 0.450 | — |
| **value** | `best` | **87.5%** | **0.467** | 85.0% |
| **value** | MPPI | 93.8% | 0.518 | 100% |

`val_first` and `dist_first` agree to the digit — 93.8%, rank 0.450, 228 mean steps —
which is the check that the two arms differ in the scorer and nothing else. So the
comparison is clean: a scorer that predicts *outcome* 23x better than the one it
replaced does not improve success **on perturbation candidates**.

**The conclusion is narrower than it was written.** "Still ranks candidates at chance
(0.467 against 0.478)" cannot be concluded from this table at all — chance is the
correct score for every ranker on an outcome-equivalent candidate set, so this measures
the candidates. What survives is the success column: a better objective does not rescue
the loop *here*, which is a fact about a candidate set that has +0.0 headroom. The
objective hypothesis is neither refuted nor supported by it; the valid test is on
affordance candidates.

**What survives, and it is the durable part.** The candidates barely differ in outcome
— true-cost spread 0.0067 across eight of them — and every ranking runs through the
learned dynamics, whose one-step error on the object slots is 0.0259 in the same
standardised units. The differences being ranked are **smaller than the model's own
error**. `ranking_metrics.py` later put a number on exactly this as SNR = 1.09 for
perturbation candidates, and this is the measurement that makes the benchmark invalid
rather than merely unflattering: at SNR ≈ 1 the test has no discriminative power by
construction. Affordance candidates raise it to 1.21 — better, still tight, and enough
for the same scorer to reach τ +0.291.

### The candidate set is outcome-equivalent · MEASURED, and this closes it

`--oracle-depth episode` removes the proxy from the diagnostic entirely: roll every
candidate to completion, read the simulator's own success flag, and compare how often
**any** candidate reaches success against how often the **chosen** one does. The gap is
headroom that a better selection rule could actually capture. 12 episodes, 3 probed
decisions each, 8 candidates rolled to completion per probe.

| candidate source | a candidate succeeds | the chosen one succeeds | **headroom** | episode success |
|---|---|---|---|---|
| perturb 0.003 (default) | 96.7% | 96.7% | **+0.0 pts** | 91.7% |
| perturb 0.012 (4x wider) | 70.0% | 66.7% | +3.3 pts | 41.7% |
| belief-conditioned | 36.7% | 35.0% | +1.7 pts | 0.0% |

**Zero headroom at the default.** Every candidate leads to the same outcome, so no
selection rule — not the scorer, not a perfect oracle, not a better objective — can do
anything but pick one of eight equivalent actions. **This is the measurement that
invalidates the benchmark**, and it does more than explain the earlier negative results:
it withdraws them. A test on which a perfect ranker and a coin score the same cannot
have been evidence about a ranker, so "the scorer ranks at chance" and "a 23x better
objective changes nothing" are statements about the candidate set and are not citable
against the planner or the scorer.

Diversifying the candidates does create headroom, and shows why that is not the fix.
The trend is monotone and it runs the wrong way: reachability falls 96.7 -> 70.0 -> 36.7
while headroom moves 0.0 -> 3.3 -> 1.7. Widening the perturbation buys 3.3 points of
extractable difference and costs 26.7 points of ceiling; belief-conditioned proposals
cost 60 points of ceiling and end at 0% success.

**The perturbation that makes candidates distinguishable is the same perturbation that
ruins them.** That is a property of perturbing one policy's output, not a tuning
failure, and no amount of scale search escapes it — the search is over a single axis
where diversity and quality are the same knob turned in opposite directions.

### An affordance proposer breaks the trade — and moves the constraint

The measurement above said the remaining option was candidates that differ
**task-meaningfully** rather than randomly. `sim/affordance.py` builds them: a graspable
radius from the object's perceived extent minus what the jaw needs, a placeable radius
from the container's extent minus the object's diagonal, and K plans spread on a ring
inside both. Every candidate is a *valid* plan — a real grasp point, a real release
point — so the set can differ without any member being a degraded copy.

| candidate source | reachable | headroom | episode success (`first`) |
|---|---|---|---|
| perturb 0.003 | 96.7% | +0.0 pts | 91.7% |
| perturb 0.012 | 70.0% | +3.3 pts | 41.7% |
| belief-conditioned | 36.7% | +1.7 pts | 0.0% |
| **affordance** | **91.7%** | **+2.8 pts** | **91.7%** |

**The trade is broken.** Affordance proposal buys nearly the headroom of a 4x
perturbation at 22 points more reachability, and unlike every other diverse source it
costs the nominal trajectory nothing.

And that changes which component is binding:

| affordance + | success | reachable | chosen | oracle rank |
|---|---|---|---|---|
| `first` — never select | **91.7%** | 91.7% | 88.9% | 0.246 |
| `best` — the scorer | 0.0% | 43.3% | 38.3% | 0.683 |
| `oracle` — greedy distance-optimal | **0.0%** | 40.0% | 36.7% | 0.000 |

Greedy per-decision optimisation of the distance cost now scores **zero**. With
perturbation candidates it could not do much harm because there was nothing to choose
between; given real choice it exploits the objective catastrophically. Note also that
reachability is not a fixed property of the scene — it is 91.7% along the nominal
trajectory and ~40% along the ones selection produces, because **each bad choice moves
the episode into a state where fewer choices can recover.** The failure compounds.

So the conclusion one section above — "the objective is not the binding constraint" —
was correct *for a candidate set with no headroom* and is now false. With a proposer
that creates real choice, the objective is exactly what binds.

### Why the value head does not rescue it either, and what would

The value head predicts return 23x better than the distance term (Plan.md §4), and it
ranks affordance candidates no better. The reason is not subtle and is true by
construction rather than by inference: it was fitted on `pick_place_400c`, every episode
of which was generated by the demonstrator running the **nominal** plan — centre grasp,
centre release, fixed carry height. It has never seen an off-centre grasp. The
affordance proposer generates precisely the states the value head is out of
distribution on, so the better it proposes, the less the critic knows.

**The closing move is therefore a data problem, not a model one:** collect
demonstrations with randomised affordance plans, refit the value head on states that
actually cover the candidate manifold, and re-run this table. That is the next
experiment, and it is the first one in this sequence where the mechanism is identified
before the measurement rather than after.

### What this licenses, and what it asks for next

The sequence, so that the order is not rewritten later: the ranking rule was measured at
chance, the horizon hypothesis was refuted, the objective was replaced with something
23x better and changed nothing, and two proposal distributions were neutral-to-worse.
Only then did Oracle@k at episode depth show why — the candidates were
outcome-equivalent, so none of those components could have mattered.

An affordance proposer removed that, and the constraint moved to the objective, which is
now failing for an identifiable reason (its training data covers one plan). Each step
was wrong about something, and each was wrong in a way the next measurement named.

### Candidate generation · two new sources · PERTURBATION-ERA, SEE THE INVALIDITY NOTE

`perturbed_candidates` documents itself as a stand-in for sampling a policy K times,
and additive Cartesian noise makes every candidate but one strictly worse than the
policy's answer. Two replacements were built and measured:

- **belief-conditioned proposals** perturb the *perceived* object positions within the
  measured perception error and re-run the policy, so every candidate is a chunk the
  policy would genuinely have produced. They do diversify more — true-cost spread
  0.0217 against 0.0070 — and they score worse (12.5% against 100.0%), because a chunk
  aimed at a displaced goal is committed to for four steps.
- **MPPI** refines the chunk by reward-weighted averaging instead of selecting, so its
  output need not be any sampled candidate. It converges back onto the policy's own
  chunk (regret 0.0016, spread 0.0030) and matches it at 100.0% on the scripted policy
  — a graceful degradation rather than a gain — while costing the BC policy 19 points.

Both were read at the time as "no proposer distribution makes selection work". That
reading is withdrawn: both were scored against perturbation-era conclusions on a
candidate set with +0.0 headroom. What remains valid is the mechanism in each bullet —
a belief-perturbed chunk is committed to for four steps, and MPPI converges back onto
the policy's own chunk — which are statements about those proposers, not about
selection.

## How well does the scorer actually rank? · THE PRIMARY RANKING TEST

This is the ranking benchmark of record. Success rate answers "did planning help" with
one bit per episode, which is why four ablations in a row could all read "no" without
saying *how* the ranking was wrong. `sim/ranking_metrics.py` scores the ordering itself
against the simulator's own rewind-and-execute cost, at the same decisions the planner
faced. 12 episodes, 4 probed decisions each, 8 candidates.

| candidate source | ranking measured along | decisions | pairwise | Kendall τ | Spearman ρ | regret@1 | SNR |
|---|---|---|---|---|---|---|---|
| ~~perturbation~~ *(invalid)* | the policy's trajectory | 91 | 0.564 | +0.127 | +0.159 | 0.00248 | 1.09 |
| **affordance** | the policy's trajectory | 91 | **0.646** | **+0.291** | +0.340 | 0.00403 | 1.18 |
| **affordance** | **the scorer's own trajectory** | 144 | **0.421** | **−0.157** | −0.228 | **0.01930** | 1.21 |

The perturbation row is shown struck through and must not be cited: with +0.0 Oracle@k
headroom its candidates are outcome-equivalent, so the row is a property of the test.
**Rows two and three are the benchmark.**

Ties in the *truth* are excluded from the pairwise count rather than scored as correct.
That is not a detail on the invalid row — on perturbation candidates almost every pair
is tied, and counting ties as wins would have reported it at ~0.9.

**This corrects an earlier claim in this file.** "The scorer ranks at chance" was read
off perturbation candidates, where there is nothing to rank; given candidates that
genuinely differ it ranks well above chance. An earlier revision of this table quoted
τ +0.407 / −0.304 from 6-episode runs; the 12-episode numbers above supersede them and
the qualitative reading is unchanged. The 6-episode/12-episode gap is itself worth
noting — a τ read off 34 decisions moved by 0.12 when measured on 91.

### The distribution of value gaps vs the distribution of model error

The question underneath "why doesn't better ranking help" is whether the differences
between candidates are even large enough to see through the model's error. Both are
reported in the same units and scale-matched (the predicted cost is regressed onto the
true cost before the residual is taken, so an arbitrary offset or gain is not counted
as error):

| | spread between candidates | model RMSE | SNR |
|---|---|---|---|
| ~~perturbation~~ *(invalid)* | 0.00191 | 0.00175 | 1.09 |
| affordance, on-policy | 0.00365 | 0.00311 | 1.18 |
| affordance, off-policy | 0.01094 | 0.00901 | 1.21 |

**SNR near 1 in every configuration.** The model's error is the same size as the thing
it is being asked to discriminate, so even the on-distribution τ of +0.291 sits on a
margin barely wider than the noise. This is also the cleanest statement of why the
perturbation benchmark is invalid rather than merely hard: at SNR 1.09 the test cannot
resolve the thing it purports to measure. Affordance candidates raise the signal 91%
and the noise rises 78% with it — a real but small gain, and the quantitative version
of "the candidates now differ enough for a bad objective to do real damage."

### Counterfactual coverage: the model is only right where the demonstrator went

The third row is the important one, and it is a direct measurement rather than an
inference. The only difference between rows two and three is **whose trajectory the
episode follows** while the ranking is scored: `--drive first` executes the policy's own
chunk and asks the scorer to rank the alternatives it did not take; `--drive best`
executes what the scorer chose, so each decision is made from a state the scorer's
previous decisions produced.

Same model, same candidates, same metric. **Kendall τ goes +0.291 → −0.157 and regret@1
rises 4.8×.** The ranking is not merely worse off-distribution, it is *inverted*.

That is textbook distribution shift, and it names the defect precisely: the dynamics
model is trained on transitions from demonstrations, so every action it has ever seen is
one a competent demonstrator took from a state that demonstrator reached. Asked to
evaluate "what if we grasped 2 cm to the left instead", it is extrapolating — and acting
on its answer is what walks the episode into the region where it extrapolates worse.
It also explains the earlier reachability collapse (91.7% under `--pick first` → 40%
under `--pick best`) as a compounding loop rather than an unlucky objective.

`collect.py --counterfactual N --branch-length L` adds the missing coverage: from
states along an on-policy episode, it branches L steps under a *different* plan and
records the result. The states stay on the demonstrator's distribution while the actions
leave it, which is exactly the slice the model has never seen. 400 seeds expand to 2,088
episodes (1,709 branches). Measured three ways, because the obvious measurement is
confounded twice over.

**As measured, each model driving itself** — and this is the number not to quote:

| dynamics trained on | τ on the policy's trajectory | **τ on its own trajectory** | regret@1 |
|---|---|---|---|
| 400 on-policy episodes | +0.291 | **−0.157** | 0.01930 |
| 1,903 on-policy episodes *(quantity control)* | — | **+0.040** | 0.01554 |
| 400 seeds → 2,088 branched | +0.181 | **+0.671** | 0.00261 |

τ −0.157 → +0.671 looks decisive and is mostly an artifact: under `--drive best` each
model steers itself, so a model that chooses better ends up in states where ranking is
easier, and the comparison silently includes that.

**State-matched** — `ranking_metrics --also-score` runs one model and scores a second at
the *same* decisions, on the *same* candidates, against the *same* rewind-and-execute
truth. 144 decisions, baseline driving:

| dynamics trained on | pairwise | **Kendall τ** | regret@1 |
|---|---|---|---|
| 400 on-policy *(driving)* | 0.443 | **−0.114** | 0.01787 |
| 1,903 on-policy *(quantity control)* | 0.468 | **−0.065** | 0.01517 |
| 400 seeds → 2,088 branched | 0.514 | **+0.029** | 0.01154 |

**Coverage does roughly twice what quantity does, and only coverage crosses zero.**
Five times the on-policy data halves the inversion (−0.114 → −0.065) and leaves the
ordering inverted; counterfactual branches at the same scale take it positive (+0.029)
and cut regret@1 by 35%. That is the controlled version of the claim, and it is much
more modest than the self-driven +0.671 — which is exactly what `--also-score` was built
to expose.

The on-policy column falls (+0.291 → +0.181), the trade one expects from spending
capacity on states the demonstrator never visits.

> **Comparability notice.** A `stage_timeout` was added to the demonstrator partway
> through this work to break a deadlock in phase-aligned commitment. It changes what the
> robot does when a waypoint cannot be reached, so it is part of the **task definition**,
> not a tuning parameter — and it moved closed-loop success at K=32 from 37.5% [16.7,
> 58.3] to 79.2% [66.7, 89.6], intervals that do not overlap ([bug_log.txt](../bug_log.txt)
> [S20]). Every closed-loop number below that is driven by affordance candidates was
> measured on one side or the other and they may not be placed in one table. Runs now
> record `task_fingerprint()`; numbers carrying different fingerprints are marked **[pre]**
> and are being re-measured. The nominal demonstrator is unaffected (10/10 canonical,
> 40/40 randomised), so policy-first, data collection, the dwell distribution and the
> probes are unchanged.

## The scorer was never the problem · MEASURED, and this reframes the whole page

Four ablations, a ranking benchmark, a domain gate and a veto were all built on the
premise that greedy selection loses because the ranking is wrong. Three controls say it
does not.

**Control 1: give greedy exactly one candidate.** With nothing to choose between, greedy
selection must reproduce the policy — and if it does not, the execution path is broken
rather than the ranking. 24 episodes, 95% bootstrap CI over episodes:

| rule | candidates | success |
|---|---|---|
| no planning at all | — | **95.8% [87.5, 100.0]** |
| `--pick first` | 8 affordance | **95.8% [87.5, 100.0]** |
| **`--pick best`, K=1** | **1 affordance** | **95.8% [87.5, 100.0]** |
| `--pick best`, K=8 | 8 affordance | **0.0% [0.0, 0.0]** |

The machinery is sound. The only difference between the third row and the fourth is that
there is something to choose.

**Control 2: replace the scorer with a perfect one.** `--pick oracle` at episode depth
rewinds, executes every candidate to completion, and keeps whichever actually succeeded
— oracle rank 0.000 by construction:

| rule | success | reachability along its own trajectory |
|---|---|---|
| `--pick first` | 95.8% | 91.7% |
| `--pick best` (model) | 0.0% | — |
| **`--pick oracle`** | **0.0% [0.0, 0.0]** | **42.5%** |

**Perfect per-decision selection also scores zero.** That exonerates the scorer
completely and reframes every negative ranking result on this page: improving the
ranking cannot help when perfect ranking does not. It also shows the compounding
directly — reachability measured along an oracle-driven trajectory is 42.5% against
91.7% along the policy's, so a few optimal-per-decision choices are enough to reach
states no candidate recovers.

**Control 3: look at what was executed.** `loop.py --record` logs the executed action,
the action the policy wanted, and the tip path at every step; `compare_traces.py` lays
the greedy run against the policy run on the same seeds:

| | greedy vs policy |
|---|---|
| chosen candidate changes between consecutive decisions | **80.3%** |
| distinct candidates chosen per episode (of 8) | 7.62 |
| longest run of the same choice | 3.96 decisions of 90 |
| executed action vs the action the policy wanted | **11.95 mm — 1.02× the policy's own step** |
| tip path divergence by end of episode | 36.1 cm |
| episode length | 360 steps (never finishes) vs 194 |

**The mechanism is plan thrash.** Each candidate is a whole plan — a grasp point and a
release point — but selection runs every `execute` steps and re-decides from scratch.
The arm is redirected to a different plan at four decisions in five, each redirection
displacing it by a full step, so it oscillates around the ring of plans and never
completes an approach to any of them. The grip command never differs (0.0% of steps):
the two runs disagree about *where to go*, not about what to do when they arrive.

That is also why the oracle fails. "Best for this decision" is evaluated in isolation,
and a plan that is optimal to start is not carried out long enough to be executed.

**What this implies.** The defect is in the *control scheme*, not in any of the
components this page spent four sections eliminating. A plan-level choice needs
plan-level commitment: choose once and hold, or make the candidate set a continuation of
the plan already in progress rather than a fresh set of alternatives. The ranking work
below is not wasted — it is a prerequisite — but it was never the binding constraint,
and this section is here rather than at the end because it changes how everything after
it should be read.

## Applicable-domain gating: rank only where the model was fitted · MEASURED

The inversion above is a statement about two regimes, so the natural response is a
boundary between them rather than trusting the model less everywhere. `sim/domain.py`
is that boundary, and it keeps the policy-first baseline intact by construction: the
policy's own chunk is in-domain by definition, so abstaining is exactly "do what the
policy wanted" and never a third behaviour.

Three candidate signals were calibrated against per-decision Kendall τ, and **only one
survived**.

| signal | best τ any threshold could reach | armed? |
|---|---|---|
| state energy (is this scene in-distribution?) | −0.095 *(from −0.133 ungated)* | no |
| ensemble spread | −0.106 | no |
| **per-candidate action deviation** | **+0.062** | **yes** |

**The state never leaves the manifold; the action does.** State energy ranged 0.38–1.43
across the 144 decisions in exactly the regime this gate exists for — these are ordinary
scenes — and no threshold on it recovered a positive τ. It is downgraded to a backstop
defaulting to infinity, which the caller must arm; shipping a plausible-looking number
for a signal measured not to work is how a gate comes to look calibrated while doing
nothing. That the *action* is what leaves is also why the counterfactual collector
branches off-policy actions from on-policy states.

The action filter is per-candidate and it works on the measurement it was built for:

| keep deviation ≤ | decisions kept | candidates kept | mean τ of what survives |
|---|---|---|---|
| 0.474 | 131/144 | 345/1152 | +0.046 |
| **0.681** *(shipped)* | **144/144** | 576/1152 | **+0.062** |
| 0.822 | 144/144 | 807/1152 | −0.039 |
| 1.331 *(no filter)* | 144/144 | 1151/1152 | **−0.134** |

Half the candidates go and no decision is lost, and the ordering goes from inverted to
correct. The mechanism is direct: a candidate's deviation predicts the scorer's own
error on it (r = **+0.417**, binned error 0.004 → 0.009), so the model is measurably
less accurate on actions further from the ones it was fitted on.

### It fixes the ranking and does not fix the loop

| dynamics | policy-first (`--pick first`) | greedy (`--pick best`) | greedy + domain gate |
|---|---|---|---|
| 400 on-policy | **95.8%** | 0.0% | 0.0% |
| counterfactual | **95.8%** | 0.0% | **16.7%** |

**Ranking quality and closed-loop success are dissociated.** The gate takes τ from
−0.134 to +0.062 and buys nothing at all on the baseline model. What moves the loop is
commitment coverage, measured separately: success climbs with the fraction of a
waypoint stage a choice outlasts (0.3% → 27.4% → 71.8% → ~97% of stages, giving 0.0% →
0.0% → 37.5% → 66.7% success), with the 28-step median dwell as a landmark on that
curve rather than a threshold on it. The one cell that moves
needs *both* interventions — counterfactual coverage and the gate — and recovers 16.7%
from 0.0%, which is a real interaction and still 79 points below simply running the
policy.

So the policy-first baseline stands, and this is what the gate is for: it is a
containment measure that stops the scorer being trusted where it is wrong, not a
mechanism that makes greedy selection viable. An untested candidate mechanism for the
remaining gap is plan thrash — each decision re-picks among eight whole plans, so the
arm may never commit to one long enough to execute it. That is a hypothesis, not a
finding, and the way to test it is to hold the chosen plan across decisions.

## Verification instead of selection · THE LABEL WAS THE PROBLEM, AND REMOVING IT FINDS A SIGNAL

Selection was measured not to work, so the world model was re-pointed at the job it
might plausibly be good at: **refusing**. `sim/veto.py` is policy-first — the policy's
chunk executes unless the model is confident it is a disaster — and it refuses only when
a predicted return drop exceeds a threshold **and** the ensemble agrees, because a
prediction the ensemble disagrees about is evidence of ignorance rather than of danger.

The threshold was not guessed. `sim/calibrate_veto.py` rewinds the simulator at sampled
decisions, executes the action to episode completion, records whether it really was
doomed, and sweeps every signal the model can produce against that ground truth. 492
decisions across 24 episodes, 28.5% of them genuinely doomed:

| signal | AUC (0.5 = no information) |
|---|---|
| **ensemble spread** | **0.587** |
| predicted collision | 0.466 |
| predicted return drop | 0.457 |
| predicted instability | 0.396 |
| predicted return at horizon | 0.353 |

**Every signal built from the value head points the wrong way; the only one that carries
information is the ensemble's disagreement with itself.** That inverts the gate's own
design — the rule was written to treat spread as a *disqualifier*, and the measurement
says spread is the detector and the value drop is the noise. `veto.py` now defaults to
firing on spread, with the refuted rule kept runnable as `mode="drop"`.

**That edge does not survive the two checks below, so the section's conclusion is
negative.** The mode change stands because it is the right way round; what does not
stand is any claim that the gate is worth having.

#### The explanation this file gave for that was wrong · REFUTED BY PROBE

Two revisions of this page said the cause was structural: "the latent carries the cube,
the bowl and the ball, and **no gripper**", so the model could not represent the
mechanism — the cube slipping from the fingers — that it was being asked to predict.
That was an assertion about a representation, and `sim/probe_latent.py` tests it with a
**linear** probe on the latent the scorer actually receives. 10,440 samples, held-out R²:

| target | full latent | object slots only | proprioception only |
|---|---|---|---|
| finger opening *(positive control)* | **1.000** | 0.458 | 1.000 |
| tip-to-cube offset (3-D) | **0.999** | 0.411 | 0.710 |
| pad-to-cube clearance | **0.973** | 0.803 | 0.929 |
| is the cube held | **0.903** | 0.852 | 0.834 |

**The gripper is in the latent, and so is the grasp geometry.** The state vector carries
both finger joints, their velocities, the grip command and the tip; the slots carry the
cube. A linear readout recovers pad-to-cube clearance at R² 0.973 and whether the cube
is held at 0.903. The probe is linear on purpose: a deep probe would answer "could some
network recover this", and the question is whether the information is available to the
kind of readout the scorer already uses.

#### And the dynamics carries it forward too · SECOND HALF ALSO REFUTED

The obvious follow-up was that the information is present at t and destroyed by the
rollout. `--horizon H` probes the same targets H steps ahead three ways: from the true
future latent (the ceiling), from (latent, action chunk) with no learned dynamics
involved, and from the **model's own rolled-out latent** — what the scorer actually
receives. 6,961 rows, held-out R² on pad-to-cube clearance:

| H | true z(t+H) | z(t) + action chunk | **model rollout** |
|---|---|---|---|
| 8 | 0.974 | 0.907 | **0.906** |
| 16 | 0.976 | 0.912 | **0.876** |
| 32 | 0.977 | 0.899 | **0.821** |

At 8 steps the model's rollout matches the no-dynamics upper bound to the third decimal,
and at 32 steps — four times the planning horizon — it still carries the clearance at
R² 0.821. "Is the cube held" behaves the same way (0.907 / 0.904 / 0.840).

**So both halves of the structural explanation are refuted.** The geometry is in the
latent and the dynamics propagates it. Whatever prevents a veto signal is in the
objective — the value head fitted on sparse success — or in the fact, established
above, that per-decision judgements are the wrong unit for this control scheme
regardless of how good they are.

What is settled is that **"put the gripper in the latent" is not the fix**, and it was
the concrete next step this page recommended twice.

### Why: the per-decision problem is degenerate

The framing above — "no usable signal" — was still too generous, because it implies a
signal might exist and we failed to find it. Rolling every available plan to completion
at each probed decision says otherwise.

497 probed decisions over 32 episodes:

| plans that reach success (of 8) | decisions | of which the policy's action fails |
|---|---|---|
| **0** — the state is already lost | **77** | **100.0%** |
| 1–2 | 27 | 96.3% |
| 3–5 | 39 | 30–61% |
| 6–8 — anything works | 354 | **0–10.5%** |

**Recoverability is bimodal, and it — not the action — decides the outcome.** 15.5% of
probed decisions have no surviving plan at all, and **62% of every "doomed action" label
comes from those states**. Where six or more plans still work the policy's action almost
never fails. So the label the veto was trained and scored against is largely a property
of the *state*, fixed before the action was chosen: the gate was being graded on a
question it was not asked.

**Removing the trap does not kill the veto. It finds the signal — and it is the opposite
signal.** On the 105 genuinely **pivotal** decisions (47 doomed, 58 fine, 2,726 orderable
pairs), with 95% CIs bootstrapped over episodes:

| signal | AUC on pivotal decisions | 95% CI |
|---|---|---|
| **predicted return drop** | **0.653** | **[0.541, 0.764]** |
| predicted return at horizon | 0.550 | [0.424, 0.667] *(chance)* |
| action deviation | 0.499 | [0.368, 0.640] *(chance)* |
| failure-label head | 0.478 | [0.350, 0.616] *(chance)* |
| **ensemble spread** | **0.371** | **[0.269, 0.490]** *(anti-predictive)* |

Two intervals exclude 0.5 and they point in opposite directions. **The value head's
predicted return drop — the quantity the original gate was designed around and which the
pooled numbers had at or below chance — is the one that works.** Ensemble spread, which
the pooled numbers promoted to the detector and which `veto.py` now defaults to, is
*worse* than chance here.

That is a third reading of the same experiment, and the reason it changed is the label,
not the sample: scoring a gate on decisions where nothing could have helped buries a
real signal under three times as many uninformative positives.

**What this licenses, stated carefully.** A per-decision veto is feasible on the
minority of decisions that are pivotal — about 21% of probes — using the return drop.
It is not feasible as a gate applied to every decision, because on the other 79% there
is either nothing to save or nothing at risk, and the gate cannot tell which kind of
decision it is looking at without the rewind it does not have at run time. The practical
ceiling is unchanged: holding rescues 10–12% of doomed actions and none of the
already-lost ones.

**`veto.py`'s default is now pointing the wrong way** and is left as-is pending a
re-measurement, with this table as the reason. Changing it on one run would repeat
exactly the mistake that produced the spread default.

### The two checks that produced the earlier, weaker conclusion

**The edge also disappears under two checks that should have come first.**

*Check one: does it hold within an episode?* The AUCs above pool every decision from
every episode, so a signal can score above chance purely by separating doomed episodes
from healthy ones. The gate is never asked "is this a bad episode" — it is asked "is
this the decision to refuse". Recomputed **within** each episode and averaged by pair
count:

| signal | pooled AUC | **within-episode AUC** |
|---|---|---|
| ensemble spread | 0.528 | **0.516** |
| action deviation | 0.578 | **0.508** |
| predicted return drop | 0.473 | **0.506** |
| spread + action deviation *(leave-one-episode-out)* | — | **0.476** |

**Every signal is at chance where the decision is actually made, and combining two is
not better than the best one.** The pooled numbers were measuring between-episode
discrimination: some episodes run at higher spread throughout and are also more likely
to fail. That does not help a gate choose a moment.

*Check two: does the net benefit survive an interval?* Bootstrapping over **episodes**
(decisions inside one episode share a scene and a policy seed, so resampling decisions
would understate the spread):

| spread > | fires | net flips / 100 fires | 95% CI |
|---|---|---|---|
| 0.1261 | 200 | +3.5 | [−0.5, +8.3] |
| 0.1504 | 100 | +3.0 | [+0.0, +7.4] |
| 0.1961 | 74 | **+4.1** | **[−1.5, +10.0]** |
| 0.2149 | 50 | +4.0 | [−3.8, +13.6] |

**Every operating point includes zero.** A 120-episode closed-loop ablation agrees —
79.2% without the veto against 80.8% with it, across three seeds. An earlier revision of
this page reported "+4.2 flips per 100 fires" as a result; it is a point estimate with
an interval that spans nothing-to-useful, and the honest reading is that **the veto has
no measurable net benefit**.

The instrument that produced the within-episode table was itself wrong the first time:
pooling leave-one-out logistic scores across folds reported AUC **0.188** for a signal
that scores 0.528 in sample, because each fold's intercept anti-correlates with the
episode it held out ([bug_log.txt](../bug_log.txt) [S18]). That is the second time on
this page that an AUC manufactured a strong result out of no data — [S14] was the first.

**The ceiling underneath all of it, and it was there before any of the above:** holding
instead of acting rescues only **10.0-12.2%** of genuinely doomed actions. By the time an action is doomed, not acting mostly does not
recover it, so even a perfect detector on a perfect signal would buy little. That caps
any veto whose fallback is inaction, regardless of the classifier. (Holding is at least
cheap: among actions that were *not* doomed, holding still ends OK 96.9% of the time, so
a false positive is nearly free.)

Two concrete consequences, and the third has been withdrawn. Invert the gate to fire on
spread rather than veto on it; and find a fallback that recovers, because refusal is
only useful when the substitute is better. ~~Put the gripper in the latent~~ — refuted
by the probe above; it is already there.

**This table replaces an earlier one that said no signal carried any information.** That
version was computed from a checkpoint whose slot normaliser read 187.9 metres
([bug_log.txt](../bug_log.txt) [S16]) — the conclusion changed when the data was fixed,
not when the gate was re-tuned. Two instrument failures are worth recording alongside it:
the collision row first read **0.764** from a column that was identically zero, because
the AUC broke ties by sort order ([S14]); and the contaminated run sampled 164 decisions
where this one samples 492, so the weak signal was also under-powered.

## Provenance audit · which results came from which pipeline

Every number above was traced back to the dataset and checkpoint it was computed from,
because a fix that is present in the code is not the same as a fix that was present when
the measurement ran. Three things were checked: that the checkpoint carries the slot
normaliser, that the loaded weights are the post-training ones, and that the training
data was sane.

**The loading path cannot silently skip the fix.** `load_dynamics` raises on a checkpoint
without a slot normaliser rather than falling back to unit scaling, and `load_value`
refuses a value head whose held-out MSE does not beat the distance baseline. Both
checkpoints in use carry `slot_mean`/`slot_std`/`slot_labels`; both differ from
initialisation (max |Δw| 1.19) and from an untrained forward pass (max |Δz| 4.35).

**One measurement did not survive the audit.** The slot normaliser of the plan-randomised
model read **187.9 metres** on a 1.2 m table — 30 of 500 episodes had launched the cube
to 9.6×10⁷ m and nothing rejected them ([bug_log.txt](../bug_log.txt) [S16]).

| result | data | status |
|---|---|---|
| Oracle@k, headroom, reachability, rescue rate | simulator rewind only | ✅ no checkpoint involved |
| perception F1, the bowl-shape finding | segmenter + simulator masks | ✅ no checkpoint involved |
| affordance feasibility | perceived boxes only | ✅ no checkpoint involved |
| object dynamics in simulation (+37.7% / +60.2%) | `pick_place_400b/c` | ✅ clean (max 0.399 m, 0 blow-ups) |
| value head (+97.4% vs constant, +95.6% vs distance) | `pick_place_400b/c` | ✅ clean |
| ranking metrics, all three rows | `world_model_sim` | ✅ clean |
| ranking metrics, state-matched | `world_model_sim` / `_onpolicy` / `_cf` | ✅ clean |
| domain-gate calibration and ablation | `world_model_sim` / `_cf` | ✅ clean |
| latent probe (R² table) | `world_model_sim` | ✅ clean |
| plan-randomised dynamics/value | `plans_500` | ❌ **contaminated, re-run below** |
| veto AUC table | `plans_500` | ❌ **contaminated, re-run below** |

Re-trained twice. First on `plans_600`, where the guard discarded 33 of 600; then on
`plans_600fix`, collected after the *cause* of the blow-ups was fixed
([bug_log.txt](../bug_log.txt) [S17]) and 1 of 600 was discarded:

| | contaminated `plans_500` | guarded `plans_600` | root-caused `plans_600fix` |
|---|---|---|---|
| episodes discarded at collection | 0 *(no guard)* | 33 | **1** |
| slot normaliser, max abs | **187.91** | 0.27 | 0.27 |
| dynamics vs identity | +45.3% | +38.8% | **+50.9%** |
| dynamics vs constant velocity | +63.8% | +59.3% | **+65.6%** |
| value held-out MSE | 0.025241 | 0.013159 | **0.010274** |
| value vs distance baseline | +68.3% | +80.4% | **+84.0%** |

Each column is strictly better than the last, and the second step is the informative
one: discarding the blown-up episodes and *preventing* them are not the same fix, and
preventing them is worth another 12 points of dynamics accuracy.

The dynamics figure *fell*, which is the honest direction — the outliers inflated the
identity baseline they were being compared against. The value head improved 48%.

**Column-wise CLIP is deliberately not connected on the real path**, and that is a
result rather than an omission: it gains six points in simulation and loses the bowl to
the robot arm on the real video, so `--label-read` defaults to `row` there and to
`column` in the arena ([bug_log.txt](../bug_log.txt) [S15]).

## Verified locally · 32 tests, plus 56 for the arena

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

The arena adds 56 more (`test_sim_arm/env/perception/reward/loop.py`), each one guarding
a seam that failed at least once while it was being built:

- the IK round-trips to **1e-9 m** over the whole workspace, and the clamp can never
  hand it an unreachable point — checked at the corners, where radius and height are
  each legal and their combination is not;
- `test_sim_arm.py` **parses `arena.xml`** and asserts the link lengths still match the
  constants the IK duplicates, so an MJCF typo fails a test rather than a grasp;
- rendered depth **unprojects to where the simulator says objects are**, and the
  residual is shown to point at the camera — proving it is the single-view bias and not
  a flipped axis;
- success needs **all four** of its conditions, each broken in turn;
- a missing detection **abstains** rather than scoring a failure, the distinction that
  decides whether the reward can be trusted at all;
- the lookahead **does not advance the policy it interrogates**, and the waypoint
  machine **does** advance during a run — the two halves of a bug that made the robot
  hover over the cube for an entire episode without erroring;
- slots keep a **fixed order** regardless of detection order, because a slot index is
  the planner's goal target;
- a rollout is **converted back to metres** before scoring, since the scorer's
  thresholds are volumes and the model works in standardised units.

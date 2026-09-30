# Experiment log — claims made, and claims withdrawn

A record of every conclusion this project stated and later retracted, with what killed
it. It exists because the retractions are the most reusable thing here: each one is a
measurement that a plausible-sounding result had a mechanical cause, and the mechanism
recurs.

`bug_log.txt` records defects in the code. This file records defects in the
**conclusions** — cases where the code did what it was told and the claim was still
wrong. The two overlap only when a bug produced a headline number, and then both carry
it.

Conventions: every row names the measurement that withdrew it, not an argument. A claim
is downgraded to **NARROWED** when it survives in a smaller form and **WITHDRAWN** when
nothing of it survives.

---

## W1 · "The scorer ranks at chance" — WITHDRAWN

**Claimed:** across two policies, two candidate sources, two optimisers and four
horizons, the scorer's oracle rank sat at 0.507 / 0.508 / 0.491 / 0.511 / 0.542, i.e.
chance. Treated as settled enough to stop tuning the ranking rule.

**Withdrawn by:** Oracle@k at episode depth. Rolling every candidate to completion and
keeping whichever really turned out best gives the same success rate as taking the
first — **headroom +0.0 points**. The candidates are outcome-equivalent, so chance is
what *every* ranker scores on them, including a perfect one. `ranking_metrics.py` later
quantified the same thing as SNR 1.09: the differences being ranked are the size of the
model's own error.

**What is true instead:** on affordance candidates, which differ in outcome, the same
scorer reaches pairwise 0.646 / Kendall τ **+0.291** along the policy's trajectory.

**Consequence:** the entire perturbation benchmark is marked invalid as a ranking test
and may not be cited against the planner or the scorer. `ranking_metrics.py` prints a
warning when run on it, and the affordance benchmark is the default.

**Reusable form:** *a negative result on a test with no discriminative power is a
property of the test.* Check the ceiling before concluding from the floor.

---

## W2 · "A 23× better objective changes nothing" — NARROWED

**Claimed:** a value head with 23× lower held-out MSE than the distance term it replaced
ranked candidates at chance (0.467 vs 0.478) and did not improve success, refuting the
objective hypothesis.

**Narrowed by:** the same +0.0 headroom as W1. The success column stands — a better
objective does not rescue the loop *on outcome-equivalent candidates* — but the ranking
comparison measures the candidate set, not the objective.

**What is true instead:** the objective hypothesis is neither refuted nor supported by
that table. It has not been tested on a valid ranking benchmark.

---

## W3 · "The latent has no gripper" — WITHDRAWN

**Claimed:** the veto has no usable signal because the latent carries the cube, the bowl
and the ball and no gripper, so the model cannot represent the mechanism — the cube
slipping from the fingers — it is being asked to predict. Stated in `veto.py`,
RESULTS.md and Plan.md, and named as the concrete next step: *put the gripper in the
latent.*

**Withdrawn by:** `probe_latent.py`. A **linear** probe on the latent the scorer actually
receives, 10,440 samples, held-out R²:

| target | full latent | slots only | proprioception only |
|---|---|---|---|
| finger opening *(positive control)* | 1.000 | 0.458 | 1.000 |
| tip-to-cube offset | 0.999 | 0.411 | 0.710 |
| **pad-to-cube clearance** | **0.973** | 0.803 | 0.929 |
| is the cube held | 0.903 | 0.852 | 0.834 |

The state vector carries both finger joints, their velocities, the grip command and the
tip; the slots carry the cube. The grasp geometry is linearly decodable.

**Cost avoided:** the recommended next step was weeks of work on a representation change
aimed at a hypothesis one afternoon's probe refutes.

**Reusable form:** *a claim about what a representation contains is testable in an hour.*
Probe before redesigning.

---

## W4 · "The veto nets +4.2 flipped outcomes per 100 fires" — WITHDRAWN

**Claimed:** ensemble spread predicts a doomed action at AUC 0.587 against a base rate
of 0.285, and the best operating point nets about four flipped episode outcomes per
hundred fires.

**Withdrawn by two checks, either of which is sufficient.**

*Bootstrap over episodes* (decisions inside an episode share a scene, a randomisation
draw and a policy seed, so they are not independent):

| spread > | fires | net / 100 | 95% CI |
|---|---|---|---|
| 0.1261 | 200 | +3.5 | [−0.5, +8.3] |
| 0.1961 | 74 | **+4.1** | **[−1.5, +10.0]** |
| 0.2149 | 50 | +4.0 | [−3.8, +13.6] |

Every operating point includes zero. A 120-episode closed-loop ablation agrees: 79.2%
without the veto, 80.8% with it.

*Within-episode AUC.* The pooled AUC lets a signal score above chance by separating
doomed episodes from healthy ones, but the gate is never asked "is this a bad episode" —
it is asked "is this the decision to refuse":

| signal | pooled | **within-episode** |
|---|---|---|
| ensemble spread | 0.528 | **0.516** |
| action deviation | 0.578 | **0.508** |
| predicted return drop | 0.473 | **0.506** |
| spread + action deviation | — | **0.476** |

Every signal is at chance where the decision is made, and combining two is worse than
the better one alone.

**What is true instead:** the veto has no measurable net benefit. The `mode="spread"`
default stands only because it is the right way round, not because it is worth having.

**Reusable form:** *pooled AUC answers a different question from the one a per-decision
gate asks.* Condition on the unit the decision is made in.

---

## W5 · "Counterfactual coverage lifts Kendall τ to +0.671" — NARROWED

**Claimed:** training on off-policy branches takes off-manifold ranking from τ −0.157 to
**+0.671**, pairwise 0.421 → 0.835.

**Narrowed by:** `ranking_metrics --also-score`, which runs one model and scores a second
at the *same* decisions, on the *same* candidates, against the *same* rewind-and-execute
truth. Under `--drive best` each model steers itself, so a model that chooses better
ends up in states where ranking is easier, and the self-driven comparison silently
includes that.

| dynamics trained on | self-driven τ | **state-matched τ** |
|---|---|---|
| 400 on-policy *(driving)* | −0.157 | **−0.114** |
| 1,903 on-policy *(quantity control)* | +0.040 | **−0.065** |
| 400 seeds → 2,088 branched | **+0.671** | **+0.029** |

**What is true instead:** coverage does about twice what raw quantity does, and only
coverage crosses zero — a real effect, roughly a twentieth the size of the headline.

**Reusable form:** *when a policy's own choices determine the states it is evaluated in,
the evaluation is confounded by the policy.* Match the states.

---

## W6 · "The contact fix eliminates the blow-ups" — NARROWED

**Claimed:** 0 blow-ups in 200 plan-randomised episodes, against 5.5% before.

**Narrowed by:** 2000 episodes — 10 blow-ups, **0.5%**. An 11× reduction, not an
elimination. A check at n=200 would have justified retiring the `SANE_POSITION_M` guard
that still catches the remaining 1 in 200.

**Reusable form:** *zero events at n=200 is consistent with a rate of 1%.* Size the
sample to the rate you intend to claim.

---

## W7 · "The scorer's grasp offsets are launching the cube" — WITHDRAWN BEFORE SHIPPING

**Claimed (internally, for about an hour):** off-centre grasps pinch the cube at a
corner, so `graspable_radius` is too loose and needs an anisotropic fix accounting for
the gripper's closing axis. A fix was designed and not written.

**Withdrawn by:** decomposing 300 episodes' grasp offsets into the gripper's own frame.
Blow-up rate against |tangential| offset: 7.9 / 5.0 / 4.7 / 5.4 / 3.3 / 0.0 / 4.8 %.
Flat, and the smallest blowing-up offset was 0.44 mm. The real cause was a contact
`solref` of 0.006 — stiffer than MuJoCo's own default (`bug_log.txt` [S17]).

**Reusable form:** *a mechanism that explains the failure is not evidence that it causes
it.* Bin the failures by the proposed cause first.

---

## W8 · "Greedy selection loses because the ranking is wrong" — WITHDRAWN

**Claimed, implicitly, by four sections of RESULTS.md and three built components.** The
ranking benchmark, the applicable-domain gate and the veto were all built on the premise
that `--pick best` scores 0.0% against policy-first's 95.8% because the scorer orders
candidates badly.

**Withdrawn by three controls, any one of which is sufficient.**

*One candidate.* With nothing to choose between, greedy must reproduce the policy — and
it does: 95.8% [87.5, 100.0], identical to `--pick first` and to no planning at all. So
the execution path is sound and the loss is caused by *choosing*, not by the plumbing.
(Running this control found a latent crash: with K=1 the margin computation read a
runner-up that does not exist.)

*A perfect scorer.* `--pick oracle` at episode depth rewinds, runs every candidate to
completion and keeps whichever actually succeeded — oracle rank 0.000 by construction.
It scores **0.0% [0.0, 0.0]**. Perfect per-decision selection is exactly as fatal as the
model's.

*The executed trajectory.* `compare_traces.py` against the policy run on the same seeds:
the chosen candidate changes at **80.3%** of consecutive decisions, 7.62 of 8 distinct
plans are chosen per episode, the longest run of one choice is 3.96 decisions out of 90,
and each executed action sits **1.02× the policy's own step size** away from what the
policy wanted.

**What is true instead:** the defect is in the control scheme. Candidates are whole
plans, selection re-decides every `execute` steps, and the arm is redirected before any
plan is carried out. Plan-level choices need plan-level commitment.

**Reusable form:** *before improving a component, replace it with a perfect one and see
whether the outcome changes.* The oracle control costs one flag and would have preceded
four ablations.

---

## W9 · "The dynamics cannot propagate the grasp geometry" — WITHDRAWN

**Claimed:** after W3 refuted "the latent has no gripper", the remaining explanations
were that the dynamics destroys the geometry over a rollout, or that the value head
cannot map it to an outcome.

**Withdrawn by:** `probe_latent.py --horizon H`, which probes the same targets H steps
ahead from the true future latent, from (latent, action) with no learned dynamics, and
from the model's own rollout. Pad-to-cube clearance, held-out R²:

| H | true z(t+H) | z(t) + action | model rollout |
|---|---|---|---|
| 8 | 0.974 | 0.907 | **0.906** |
| 16 | 0.976 | 0.912 | **0.876** |
| 32 | 0.977 | 0.899 | **0.821** |

At the planning horizon the rollout matches the no-dynamics upper bound to three
decimals, and at four times the horizon it still carries the clearance at 0.821.

**What is true instead:** one explanation remains — the objective — and W8 suggests even
that may be beside the point, since per-decision judgement is the wrong unit here
however good it is.

---

## W10 · "A per-decision veto is structurally infeasible" — WITHDRAWN

The shortest-lived entry here: written into RESULTS.md and Plan.md and retracted within
the hour, which is the only reason it appears as a withdrawal rather than as a claim.

**Claimed:** the recoverability measurement showed 75% of "doomed action" labels coming
from states where no plan works, and on the 48 genuinely pivotal decisions every signal
sat at chance. Read together, that said the per-decision problem was degenerate — not
"we failed to find a signal" but "there is none to find".

**Withdrawn by:** the same measurement at a larger sample. 497 decisions over 32
episodes give 105 pivotal ones (47 doomed, 58 fine, 2,726 orderable pairs), and with
95% CIs bootstrapped over episodes:

| signal | AUC | 95% CI |
|---|---|---|
| **predicted return drop** | **0.653** | **[0.541, 0.764]** |
| ensemble spread | **0.371** | **[0.269, 0.490]** |
| action deviation | 0.499 | [0.368, 0.640] |
| failure-label head | 0.478 | [0.350, 0.616] |

Two intervals exclude chance and they point opposite ways. Removing the labelling trap
does not kill the veto, it **uncovers** the signal — and it is the predicted return
drop, the quantity the original gate was designed around and which the pooled numbers
had at or below chance. Ensemble spread, which the pooled numbers promoted to detector
and which `veto.py` still defaults to, is *worse* than chance on pivotal decisions.

**What is true instead:** a per-decision veto is feasible on the ~21% of decisions that
are pivotal, using the return drop. It is not feasible applied to every decision,
because on the other 79% there is either nothing to save or nothing at risk and the gate
cannot tell which without the rewind it does not have at run time.

**Why it happened.** I strengthened a claim from "unverified" to "impossible" on
48 pivotal decisions with no interval attached, at the same moment I was adding
intervals to everything else. The stronger claim is the one that most needed the
interval, and it was the one I did not put one on.

**Reusable form:** *the direction a claim is being strengthened in is where the error
bar is most needed.* An upgrade from "not shown" to "ruled out" is a much larger step
than the sample usually supports.

---

## C1 · "Phase-aligned commitment scores 0.0%" — CAUGHT BEFORE PUBLICATION

Not a withdrawal: a claim that was stopped on the way to becoming one, and the only
entry here that never reached a document.

**Would have been claimed:** holding the chosen plan until the waypoint stage changes
scores **0.0%**, against 66.7% for a fixed 64-step commitment. Clean falsification of a
pre-registered prediction, with a ready story — stage boundaries do not matter, only
duration does.

**Caught by:** P6, a secondary prediction registered for an unrelated reason —
"decisions per episode should fall to roughly the stage count; if it does not, the
mechanism is not engaging." It fell to **1.0 per episode** with every run hitting the
step cap. Not a result, a deadlock.

**The cause** was two stacked deadlocks: the commitment was keyed to the *nominal*
policy's stage while the arm executed a *different* plan, so the nominal machine never
reached its own waypoint and the stage never changed; and once that was fixed, a
`descend` target 14 mm off centre sits inside the cube's own footprint, so the gripper
cannot reach it and that stage never ends either.

**Why it belongs in this file.** Every other row here was withdrawn *after* being
written down. This one was refused by a diagnostic committed to before the data existed,
and it fired in the opposite direction to the one it was written for — which is a thing
a post-hoc diagnostic cannot do, because a post-hoc diagnostic is chosen to explain the
result it is looking at.

**Reusable form:** *register a secondary quantity that says whether the intervention
happened at all, separately from whether it worked.* An effect size and an engagement
check fail differently, and only the second one can tell you the experiment was invalid.

---

## Standing constraints on the per-decision veto

Not withdrawn claims — measured limits that bound what any second stage can achieve,
recorded here because both were discovered by looking at *what the gate fired on*
rather than at its score, and both are invisible in a net/100.

### Constraint 1 · the decisive set is tiny, and it does not grow with data

A veto changes an outcome only where holding and executing disagree. Everywhere else
it fires and changes nothing.

| | decisions | decisive | hold-better | hold-worse |
|---|---|---|---|---|
| 68 episodes | 1052 | **92** (8.7%) | 55 | 37 |
| 130 episodes | 1987 | **181** (9.1%) | 107 | 74 |

The fraction replicated at **9.1%** on double the data, so 8.7% was not a small-sample
artifact — it is the rate. This is a hard ceiling on the second stage in two ways.

*It caps the training signal.* A head fitted on the correct objective — "is holding
better than executing" rather than "will this action fail" — has 181 examples to learn
from, split 107 / 74, held out by episode. It scores **AUC 0.487 [0.386, 0.577]**: dead
chance. The objective is right and the estimator cannot be fitted, which are different
problems with different fixes and must not be conflated.

*It caps what the shipped signal can be shown to do.* Return drop scores **0.556
[0.468, 0.643]** on the same target — spanning chance. The signal the veto actually
uses has never been demonstrated to predict the thing the veto is for.

### Constraint 2 · the inert fires cannot be filtered out

At the headline operating point, 87% of fires change nothing, and 62% of them land on
states where every available plan succeeds. Tightening the first stage does not shed
them preferentially:

| stage-1 quantile | fires | on all-viable states | share |
|---|---|---|---|
| 0.50 | 496 | 308 | 62% |
| 0.70 | 298 | 185 | 62% |
| 0.80 | 199 | 116 | 58% |
| 0.94 | 60 | 34 | 57% |

The share is **flat across the entire sweep**. The proxy does separate pivotal from
all-viable states — AUC 0.623 [0.574, 0.671], excluding chance — but far too weakly to
change the mix: raising the threshold from 0.70 to 0.94 drops 82% of the inert fires
and 64% of the rescues along with them. The inert fires are the price of the rescues,
not a tuning failure.

**Why this is not merely wasteful.** The harm from firing is *not* where intuition puts
it. P(holding breaks a working action | it fired on one), by stratum:

| stratum | rate |
|---|---|
| all viable | **0.007** [0.001, 0.015] |
| pivotal | **0.261** [0.192, 0.335] |

Firing on a non-pivotal decision is nearly free — if every plan works, holding works
too. The exposure is concentrated on pivotal decisions, which is exactly where the gate
is supposed to fire. So this is not a precision problem that a better first stage
fixes; a perfect pivotal detector raises the average yield and leaves the per-fire risk
untouched.

### Constraint 3 · decisiveness is not a run-time attribute of the state

Constraints 1 and 2 bound how much a veto can do and how cleanly it can be aimed. This
one says why every attempt to aim it better has failed, and it is the most general of
the three.

A decision is *decisive* when holding and executing lead to different episode outcomes.
That is a property of how two branches diverge over the rest of the episode — and the
only instrument that measures it is rewinding the simulator and running both to the
end. Nothing observable at the moment of decision predicts it. Three families have been
tried against the same target, all with episode-bootstrap intervals:

**Instantaneous signals**, scored against the pivotal label:

| signal | AUC |
|---|---|
| plan spread, quarter horizon | 0.637 [0.591, 0.681] |
| state classifier, no rollout | 0.635 [0.588, 0.682] |
| return drop, on pivotal decisions | 0.601 [0.527, 0.672] |
| a head fitted on the correct objective | 0.487 [0.386, 0.577] |
| action deviation | 0.500 [0.395, 0.600] |
| ensemble spread | 0.405 [0.333, 0.490] — *anti*-predictive |

**Event triggers**, against a 9.1% base rate. On the two sets this was developed from,
none concentrated the signal and most selected decisions that mattered *less* than
average:

| trigger | P(decisive \| fired) | lift |
|---|---|---|
| plan-spread jump | 9.5% | 1.05× |
| approach rate | 8.8% | 0.97× |
| model surprise | 8.5% | 0.94× |
| grip change | 8.0% | 0.88× |
| near-contact | 4.5% | 0.50× |
| stage change | 0.0% | 0.00× |

**This half is amended, not confirmed.** P12 registered "every trigger lift < 1.2×" and
a third seed falsified it: `plan-spread jump` and `near-contact` both reach **1.32×** at
the 90th percentile. The claim that event triggers never concentrate the decisive signal
does not survive. `near-contact` scored 0.50× on the first set and 1.32× on the third,
so the honest reading is that these triggers are unstable across seeds rather than
reliably useless — and which of those it is needs its own registration, not a
reinterpretation of this one. **Constraint 3's instantaneous and change-detector halves
are unaffected** (P11 max AUC 0.566, P13 detectors at 0.188–0.229 with all intervals
excluding chance, both on the same confirmatory seed).

**Change detectors**, against the structure of a decisive burst. Decisive decisions are
genuinely bursty — P(decisive | previous decisive) is **54.2%** against the 9.1% base, a
**5.95×** autocorrelation, with runs averaging 1.65–1.69 — so a detector that found a
burst's edges would be worth having. They are anti-aligned instead, reproducibly on two
independent sets:

| signal | vs burst onset (disc / held-out) | vs burst end (held-out) |
|---|---|---|
| model surprise | 0.166 / 0.266 | 0.457 |
| \|Δ observation\| | 0.180 / 0.268 | 0.556 |
| \|Δ v0\|, the value head on the latent | — | **0.326** |

The value head registers its largest state change in the *middle* of a burst, and
surprise and observation-change fire *late* in one. Both are below chance on the wrong
side with intervals excluding it, which is a stronger statement than "no signal".

**The mechanism is that these measure a different thing.** The world model was fitted to
predict dynamics and return; its notion of "something changed" tracks *motion*.
Decisiveness tracks *branch divergence*, and nothing in any training objective here ties
the two. A cube sliding 2 cm is a large state change and usually decides nothing; a
grasp offset 3 mm off-centre is almost no state change and decides everything.

What *is* correctly timed is the level signals — `return drop` and `plan spread` fire on
the burst's first decision in 26–44 of the bursts they catch, at a median lead of
**exactly 0 steps**. They are coincident indicators, not predictors: they report that a
decisive stretch has begun, never that one is coming. And they catch only **33–46%** of
bursts at a useful threshold.

**Confirmatory status.** P11 (no signal reaches 0.65), P13 (detectors anti-aligned),
P14 (burst lift 6.69×) and P15 (decisive rate 7.8%) were pre-registered and held on a
clean third seed. P12 failed. The constraint stands on its instantaneous and
change-detector evidence; its event-trigger clause is open.

**Reusable form:** *when the label requires a counterfactual, check whether any
observable tracks it before building machinery that assumes one does.* The check is
cheap — one AUC against the post-hoc label — and it would have refused the veto's first
stage, the event triggers, and the state-change gate in an afternoon each. It is also
the reason a better first stage cannot rescue this: there is nothing for it to read.

### What the gate does do

Stated plainly because the two constraints above are easy to read as "it does
nothing". The gate roughly doubles the rate at which a fire is a rescue, without
raising the rate at which one is a breakage:

| | conditional rescue | conditional breakage |
|---|---|---|
| no gate, every decision | 0.226 [0.154, 0.312] | 0.049 [0.033, 0.067] |
| gated | **0.406** [0.237, 0.591] | **0.048** [0.023, 0.077] |

Conditional rescue is over fires on doomed actions, conditional breakage over fires on
fine ones. They are the symmetric pair, and neither is recoverable from a net/100.

### Coarser granularity relocates the problem rather than solving it

A veto asked at a coarser unit faces a denser signal, and a worse conflict rate with it.
Measured on real waypoint boundaries rather than on a step-gap proxy — the proxy
over-segments long stages and flatters the conflict rate by 16 points, so it is reported
here only as the mistake it was:

| granularity | units | decisive | contains a rescue | contains a breakage | **contains both** |
|---|---|---|---|---|---|
| per-decision | 1987 | 9.1% [7.9, 10.4] | 5.4% | 3.7% | — |
| per-phase, true stages | 498 | 17.5% [14.3, 20.9] | 11.8% | 10.6% | 5.0% |
| per-episode | 130 | 39.2% [30.8, 47.7] | 29.2% | 30.0% | 20.0% |

The conflict rate — P(a unit contains a breakage | it contains a rescue) — is **42.4%**
per phase and **68.4%** per episode. Two-thirds of the episodes worth firing in also
contain a decision where firing does harm, so a coarser veto still has to choose *which*
decision, with strictly less information to choose with. The density rises; the
discrimination problem is relocated, not solved.

---

## C4 · "A model-free stage-restricted veto nets +5.5 per 100 fires" — CAUGHT BEFORE PUBLICATION

**Claimed:** splitting veto decisions by waypoint stage gave `transfer` a 3.3:1
rescue-to-breakage ratio against 1.5:1 pooled, and a veto restricted to it scored
**+5.5 net flips per 100 fires, 95% CI [+2.0, +9.7]** — clearing zero where the learned
two-stage gate's interval did not, with no model involved at all.

**Caught by:** P7/P8/P10, registered before the held-out set existed. On 130 fresh
episodes transfer-only scores **−0.5 [−3.3, +2.2]**, its ratio inverts from 3.27 to
0.88, and the rank correlation between the two sets' per-stage ratios is **−0.100**.

**What is true instead:** nothing. The ordering was a maximum over seven stages reported
with the interval of a single measurement. P9 held on both sets — the learned signal
adds nothing inside a stage — but with the stage effect withdrawn there is no
restriction left for that to be a statement about.

**Reusable form:** *an interval that excludes zero says nothing about how many intervals
were computed.* A best-of-k needs held-out data or a correction, and the tell is not in
the number — it is in how the number was chosen. Same defect as the (q1, q2) surface one
step earlier; that one was visible in a plot, this one had seven cells and no shape.

---

## The pattern

Five of the seven were produced by a measurement that was working exactly as written and
answering a slightly different question than the one being asked:

| withdrawal | the question actually answered |
|---|---|
| W1, W2 | "how well can anything rank these candidates" |
| W4 | "which *episodes* go badly" |
| W5 | "how well does this model rank *where this model goes*" |
| W6 | "is the rate below ~1%" |
| W8 | "is this component good" — when no version of it would have helped |
| W10 | "is there a signal" — asked of a sample too small to answer it |

W3, W7 and W9 were ordinary wrong guesses, each refuted by a probe that cost an
afternoon. The rest looked like results because the number was real — it just belonged
to a different claim. The corresponding habit is to state, before reading a number, what
would have to be true for it to mean what it appears to mean.

**The one that never became a withdrawal is C1**, and it cost a single line in a
pre-registration file.

**The cheapest check on this list is the one that came last.** W8 was withdrawn by
replacing the scorer with a perfect one and observing that nothing improved. That is one
flag, it needs no model, and it invalidates four sections of ablation. Run the ceiling
before optimising toward it.

Two instrument failures sit alongside these and are recorded in `bug_log.txt` because
they were code defects: an AUC that broke ties by sort order and manufactured a 0.764
predictor from a column of zeros ([S14]), and a pooled leave-one-out AUC that returned
0.188 for a signal scoring 0.528 in sample, because each fold's intercept
anti-correlated with the episode it held out ([S18]). Both produced a strong result from
no data, and in both cases the tell was a number too good for the signal behind it.

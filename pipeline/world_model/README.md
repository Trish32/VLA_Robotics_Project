# Action-conditioned world model

Predict what a candidate action would do, **before** doing it. The perception stack
above this produces a scene; this package learns `z_{t+1} = f(z_t, a_t)` over that scene
and uses it to rank the VLA's action candidates instead of executing the first one.

```
GR00T chunk ──▶ K candidates ──▶ ensemble rollout, N steps ──▶ score ──▶ execute one
                                        │                        │
                                  uncertainty              collision · stability
                                  (grows with N)           · goal · effort
```

## The limit, up front

**The proprioceptive pathway beats both trivial baselines on aggregate error but loses
on the typical sample. The object pathway is not validated at all.** Constant velocity
is better on 91% of held-out samples; the learned model wins on global RMSE because it is
substantially better in the tail, where the action matters. Read both numbers, not one. On the only object-motion data available — one episode of a single
binary mask — the model scores *worse* than assuming the object does not move. Numbers
and the split that produced them are in [RESULTS.md](RESULTS.md).

So on a real scene this planner is rejecting candidates whose predicted **robot**
trajectory is bad, while the objects roll forward near-static. That is useful and it is
not the same as predicting how objects respond to being pushed.

## Three design decisions

**The latent is object-centric.** `z` is `slots (M, D)` plus `robot (R,)`, with slot
channels 0:6 held as raw metric centre and extent. A flat latent would make the
collision and stability terms unwritable — they ask object-level questions — and would
throw away the structure the whole stack upstream exists to produce.

**The untrained model is exactly the best baseline.** Output heads are zero-initialised
and the per-slot motion gate starts at `sigmoid(-3) ≈ 0.047`, so an untrained model
predicts *nothing moves*; with `integrate_velocity` it predicts *constant velocity*.
This is measured, not aesthetic: identity scores 0.0554 held out and constant velocity
0.0262, so a model initialised at identity must first learn its way past momentum, and
on four demonstration episodes it never got there — it lost to constant velocity 0.0590
to 0.0262. Starting at the stronger baseline, it wins.

**Uncertainty is a deep ensemble, not a variance head** — and it only half works.
Independently initialised members diverge where the data did not constrain them, and that
divergence tracks how error grows with horizon almost exactly (14.1× predicted against
14.2× realised over 8 steps). But within a horizon it does **not** predict which rollout
will be wrong: rank correlation against realised error is **−0.071**. So the risk term's
default weight is zero. The term is correct in principle; the measurement decides whether
it is switched on, and today it says no.

## Scoring measures what the action *caused*

Every cost is charged against the pre-action scene. This is not a refinement; without it
the terms cannot discriminate at all. On the real TUM scene the Mask3D proposals already
overlap each other, so absolute collision cost was **identical (0.2400) for all eight
candidates and vetoed every one of them** — a fact about the segmentation, not about any
action. Charging only induced cost makes the term silent when nothing was caused.

Collisions are **vetoed**, not weighted. A weight lets a large enough gain elsewhere buy
a collision, which is exactly what must never happen.

## Running it

```bash
# train the dynamics on real demonstrations, with baselines reported
conda run -n groot_vl python -m pipeline.world_model.train \
    --data grootN1_Robotics/upstream/demo_data/gr1.PickNPlace --epochs 30

# stage 6.5: score GR00T's candidates against the scene the stack built
conda run -n foundationpose_vl python pipeline/tools/e2e_plan.py
```

`e2e_plan.py` refuses to load a checkpoint whose action width or slot dimension does not
match the scene, and says so. Reshaping it would produce confident predictions from
weights that never saw anything like the input.

**→ [RESULTS.md](RESULTS.md)** — every measurement, including the negative one
**→ [Plan.md](Plan.md)** — what would make the object pathway real

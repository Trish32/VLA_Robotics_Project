# GR1 candidate outcome pilot

Protocol fixed on 2026-10-06 before candidate execution. This is a diagnostic on
known failures, not held-out evaluation or a deployable selection rule.

## Fixed intervention and controls

Use decision-capture v2 (rollout SHA256
`36d289982ecc9dfb5c02cc3f87038c754f65154e72d8d2d0719e61ec178e5eff`)
and the completed cold-replay gate (report SHA256
`17989233e5d2353535b68893523e3a8a94fe829c2070b4aefee7db40ed34170e`).
Do not modify weights, observation processing, execute 8 or the absolute 1,440-step
deadline. The six decisions are fixed from existing stage diagnostics:

| Seed | Decision step | Reason |
|---:|---:|---|
| 0 | 256 | Object already inside; incomplete drawer closure |
| 2 | 160 | Before failed placement |
| 7 | 160 | Before transport loss |
| 8 | 224 | Before the brief grasp contact |
| 3 | 160 | Successful transport control |
| 5 | 160 | Successful transport control |

At each point run the reference chunk and all four saved candidates in manifest
order. Execute only their first eight actions, then predict every subsequent chunk
from the actual new observation using the unchanged real GR00T checkpoint.
Recorded future actions must never substitute for closed-loop continuation.

Every arm reconstructs its prefix in a fresh seeded official environment and must
match the saved oracle at every prefix step, including native integration state.
The reference arm must also reproduce every future original decoded action and
every saved control frame through the original stopping step. All six reference
arms pass before any alternative arm starts. Any mismatch invalidates the run;
do not turn it into a candidate failure or relax exactness.

Continuation sampling starts from the captured reference post-call RNG, equally
for all five arms. Keep the policy RNG stream isolated from simulator RNG so
alternative action sampling does not alter later draws. Official policy reset is
stateless in the pinned upstream. Reference reproduction remains an empirical
gate on these assumptions.

## Decision rule

Primary measurement: official task success by the same absolute 1,440-step deadline.
Report all 30 arm outcomes, success steps, failure-stage signals and videos.
For each failed reference point, report whether any alternative succeeds. For each
successful reference point, report how many alternatives break success. An oracle
choice can establish an opportunity on these six selected points only; it is not
an implementable selector or a population success-rate gain.

- At least one rescued failure: implement and evaluate a selection hypothesis on
  separate scenes before making any improvement claim.
- No rescued failures: no rescue found in these four draws at these four failure
  points. This does not rule out other points, samples or multi-step interventions.
- Reference mismatch or incomplete inventory: no candidate outcome conclusion.

The worst-case cost is **40,887 simulator steps**, including repeated prefixes:
6,327 for original arms plus 24 × 1,440 for alternatives. Using the previous cold
gate's throughput, budget approximately 6 hours including policy inference and
setup, with margin below Kaggle's 9-hour cap. Stop at official success; do not add
time after the deadline to recover a candidate. No fine-tuning occurs.

## Verification and artifacts

Tiny local tests exercise reactive replanning, absolute budgets, mid-chunk stopping,
paired isolated RNG and corruption rejection. A real official two-step CPU smoke
must execute both the original and one saved CPU candidate before cloud staging.
This smoke does not establish later branch quality or full closed-loop fidelity.

`tools/analyze_candidate_outcomes.py` separately audits the saved interventions,
current-observation hashes, exact reference continuation, task signals, video frame
counts, start offsets and descriptive summary. Artifacts stay under ignored
`data/candidate_outcomes_smoke/` and `data/kaggle_candidate_outcomes/`.

Status, 2026-10-07: real two-arm CPU smoke and independent six-frame audit passed;
the exact source bundle passed 111 tests (3 skipped). Following explicit user
authorization, private job `trishli/gr00t-gr1-candidate-outcomes` version 1
(kernel 137527779) was submitted and is RUNNING. Server metadata verifies privacy,
GPU enabled, both expected private input mounts and exact reviewed script SHA256
`02ec2423ae66f52a085e56cef5d501a0e01a185b73b61a50b6e9e20516cb20ed`.
No candidate task outcomes have been measured. Current progress: [RESULTS.md](RESULTS.md).

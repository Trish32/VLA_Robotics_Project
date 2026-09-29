#!/usr/bin/env python
"""Perceive, predict, choose, act — the whole stack running against the simulator.

This is the part the real demonstrations cannot support. On recorded video the planner
can be *shown* ranking candidates, but nothing it chooses ever happens, so there is no
way to learn whether ranking them helped. Here the chosen action is executed, the
consequence is physical, and success is counted.

One pass of the loop:

  1. render RGB-D and perceive object boxes (`perception.py`, oracle or SAM+CLIP);
  2. assemble a `SceneLatent` — perceived boxes as slots, proprioception as the robot
     token — normalised the way the dynamics was trained;
  3. ask the policy for an action **chunk**, H steps at once;
  4. perturb it into K candidates and roll each through the learned dynamics;
  5. score the predicted rollouts and execute the winner's first few actions;
  6. repeat, then report ground-truth success, perception-derived success, and whether
     anything got knocked over.

The ablations are the point, not the demo. `--no-plan` executes the policy's own chunk
so planning can be credited or not; `--perception sam` swaps the oracle segmenter for
the real one so perception can be credited or not. Everything the loop reports is
measured against simulator state the loop never reads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pipeline.sim.env import PickPlaceEnv
from pipeline.sim.perception import make_perception
from pipeline.sim.policy import (CUBE, ChunkPolicy, ScriptedPickPlace,
                                 policy_observation)
from pipeline.sim.reward import BOWL, perceived_success
from pipeline.sim.domain import DomainGate, DomainThresholds
from pipeline.sim.veto import VetoGate, hold_chunk
from pipeline.world_model.dynamics import DynamicsEnsemble
from pipeline.world_model.latent import SceneLatent
from pipeline.world_model.planner import ActionPlanner
from pipeline.world_model.scorer import ScoreWeights, TrajectoryScorer

SLOT_ORDER = (CUBE, BOWL, "a green rubber ball")


def scripted_chunk(pol: ScriptedPickPlace, tip: np.ndarray, objects: dict,
                   horizon: int) -> np.ndarray:
    """H actions from the waypoint machine, by dead-reckoning its own deltas.

    The machine's internal stage is saved and restored around the lookahead. Without
    that, asking it what it would do for the next eight steps *advances* it eight
    steps, and the policy silently runs at H times the intended rate — a bug that looks
    like an unstable controller rather than like a lookahead error.
    """
    # `in_stage` belongs in here too: it is the timeout counter, so leaving it out
    # means every lookahead of H steps ages the machine by H and the timeout fires
    # after a couple of decisions instead of after a stalled stage.
    saved = (pol.stage, pol.hold, pol.grip, pol.in_stage)
    # The RNG is part of the state being borrowed. With action noise on, `act` draws
    # from it every call, so a lookahead of H steps consumes H draws — and a candidate
    # source that calls this K times per decision consumes K times as many as one that
    # calls it once. That made `action/first` and `belief/first` execute different noise
    # realisations of the same policy, which is not a difference either arm was meant
    # to have. Restoring it also means belief candidates differ by BELIEF alone.
    rng_state = pol.rng.bit_generator.state
    t = np.asarray(tip, float).copy()
    out = []
    for _ in range(horizon):
        a = pol.act(t, objects)
        out.append(a)
        t = t + a[:3]
    pol.stage, pol.hold, pol.grip, pol.in_stage = saved
    pol.rng.bit_generator.state = rng_state
    return np.stack(out).astype(np.float32)


class ScriptedDriver:
    """The waypoint machine, behind the interface the loop actually needs."""

    kind = "scripted"

    def __init__(self, *, noise: float, rng: np.random.Generator,
                 retime: float = 1.0) -> None:
        # `retime` is the knob that stretches the task in time without changing what
        # the task is. It turns "the commitment fracture sits at the stage duration"
        # from an observation into a prediction: move the durations, and the fracture
        # has to move with them or the explanation is wrong.
        self.pol = (ScriptedPickPlace(noise=noise, rng=rng) if retime == 1.0
                    else ScriptedPickPlace.retimed(retime, noise=noise, rng=rng))

    def chunk(self, tip, seen, state, horizon: int) -> np.ndarray:
        return scripted_chunk(self.pol, tip, seen, horizon)

    def advance(self, tip, seen) -> None:
        self.pol.act(tip, seen)

    def adopt(self, plan) -> None:
        """Become the chosen plan, rather than merely simulating it.

        Phase-aligned commitment needs a stage boundary, and the boundary has to belong
        to the plan actually being executed. Keying it to the *nominal* policy's stage
        deadlocks: the nominal machine waits for the tip to arrive within `tolerance`
        of ITS target while the arm is driving to a different one, so its stage never
        advances, the commitment never expires, and the loop makes one decision per
        episode and runs to the step cap. That is exactly what happened the first time
        (see PREREGISTERED.md, P6).
        """
        self.pol = self.pol.variant(**plan.as_kwargs())

    def chunk_plan(self, plan, tip, seen, state, horizon: int) -> np.ndarray:
        """A chunk for an alternative grasp/release/height, from the current stage.

        The variant is a shallow copy, so it carries the stage and dwell counter across
        and shares the RNG — `scripted_chunk` saves and restores that generator, so
        interrogating a variant costs the original nothing.
        """
        return scripted_chunk(self.pol.variant(**plan.as_kwargs()), tip, seen, horizon)

    def snapshot(self):
        return (self.pol.stage, self.pol.hold, self.pol.grip,
                self.pol.rng.bit_generator.state)

    def restore(self, snap) -> None:
        (self.pol.stage, self.pol.hold, self.pol.grip,
         self.pol.rng.bit_generator.state) = snap


class ChunkDriver:
    """The behaviour-cloned policy. Stateless, so `advance` has nothing to do.

    That difference is the point of the interface: the scripted machine carries a stage
    that reality has to keep up to date, and forgetting to advance it froze the loop for
    a whole episode once already. A learned policy re-reads the world every call and
    cannot go stale, so the loop should not have to know which it is holding.
    """

    kind = "bc"

    def __init__(self, policy: ChunkPolicy) -> None:
        self.policy = policy

    def chunk(self, tip, seen, state, horizon: int) -> np.ndarray:
        if horizon != self.policy.horizon:
            raise SystemExit(
                f"policy was trained for a {self.policy.horizon}-step chunk and the "
                f"loop asked for {horizon}. Retrain, or run at the trained horizon — "
                "truncating a chunk changes what the policy committed to.")
        return self.policy.chunk(policy_observation(state, tip, seen))

    def advance(self, tip, seen) -> None:
        return None

    def adopt(self, plan) -> None:
        """Become the chosen plan, rather than merely simulating it.

        Phase-aligned commitment needs a stage boundary, and the boundary has to belong
        to the plan actually being executed. Keying it to the *nominal* policy's stage
        deadlocks: the nominal machine waits for the tip to arrive within `tolerance`
        of ITS target while the arm is driving to a different one, so its stage never
        advances, the commitment never expires, and the loop makes one decision per
        episode and runs to the step cap. That is exactly what happened the first time
        (see PREREGISTERED.md, P6).
        """
        self.pol = self.pol.variant(**plan.as_kwargs())

    def chunk_plan(self, plan, tip, seen, state, horizon: int) -> np.ndarray:
        """The same plan, expressed the only way a learned policy can read it.

        `ChunkPolicy` has no waypoints to retarget, but it is conditioned on object
        positions **relative to the tip**, so shifting where it believes the cube and
        the bowl are is exactly equivalent to telling it to grasp and release
        elsewhere. The plan means the same thing to both drivers.
        """
        moved = dict(seen)
        if CUBE in moved:
            moved[CUBE] = np.asarray(moved[CUBE]) + np.array([*plan.grasp_offset, 0.0])
        if BOWL in moved:
            moved[BOWL] = np.asarray(moved[BOWL]) + np.array(
                [*plan.release_offset, 0.0])
        return self.chunk(tip, moved, state, horizon)

    def snapshot(self):
        return None

    def restore(self, snap) -> None:
        return None


@dataclass
class Normalisers:
    """The standardisation the dynamics was trained under.

    Carried explicitly because the model is useless without it: slots were standardised
    during training, so feeding raw metres at inference offsets every input by whatever
    the training mean happened to be. An untrained model does not care, which is why
    this went unnoticed while every consumer ran on one.
    """

    slot_mean: np.ndarray
    slot_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def from_checkpoint(ck: dict) -> "Normalisers | None":
        if ck.get("slot_mean") is None:
            return None
        return Normalisers(np.asarray(ck["slot_mean"]), np.asarray(ck["slot_std"]),
                           np.asarray(ck["state_mean"]), np.asarray(ck["state_std"]),
                           np.asarray(ck["action_mean"]), np.asarray(ck["action_std"]))

    @staticmethod
    def identity(slot_geom: int, state_dim: int, action_dim: int) -> "Normalisers":
        return Normalisers(np.zeros(slot_geom), np.ones(slot_geom),
                           np.zeros(state_dim), np.ones(state_dim),
                           np.zeros(action_dim), np.ones(action_dim))


@dataclass
class LoopConfig:
    camera: str = "front"
    horizon: int = 8
    execute: int = 4
    candidates: int = 8
    perturb: float = 0.003          # metres of Cartesian jitter per candidate
    plan: bool = True
    #: How a candidate is chosen once the set exists. "best" is the scorer's argmin;
    #: "random" is the control that isolates it. Comparing plan against no-plan alone
    #: is confounded — the candidate set is the policy chunk plus Cartesian jitter, so
    #: a loss there could be the scorer ranking badly OR just the jitter, which the
    #: no-plan arm never suffers. Against a random pick from the SAME set, only the
    #: ranking differs.
    #: "best" is the scorer's argmin. "random" and "first" are the controls that
    #: separate ranking quality from candidate quality. "oracle" cheats — it rewinds
    #: the simulator, tries every candidate for real, and keeps the one that actually
    #: turned out best. It is not a policy; it is the CEILING of the candidate set, and
    #: the gap between it and "first" is the only honest evidence that better selection
    #: was ever available to win.
    pick: str = "best"
    #: Where the candidate set comes from. "action" adds Cartesian noise to the
    #: policy's chunk — the original, and measurably the wrong choice: every candidate
    #: but one is then strictly worse than the policy's own answer, so the planner's
    #: best case is recovering what the perturbation destroyed. "belief" instead
    #: perturbs the PERCEIVED object positions within the measured perception error and
    #: re-runs the policy, so every candidate is a real policy output under a scene the
    #: robot might plausibly be in.
    candidates_from: str = "action"
    #: Standard deviation of that belief perturbation, metres. Defaults to the
    #: perception error the loop actually measures (~1.4 cm), not a tuned value.
    belief_sigma: float = 0.014
    #: Nominal carry height the affordance proposer varies around, metres.
    transit_z: float = 0.20
    perceive_every: int = 1
    policy_noise: float = 0.0
    unsafe_displacement: float = 0.03
    #: Evaluate every candidate against the real simulator each decision, even when not
    #: picking by it, so Oracle@k regret is recorded for whatever rule IS in use.
    oracle_diagnostics: bool = False
    #: Weight on disturbing the distractor in the oracle's true-state cost.
    oracle_ball_weight: float = 5.0
    #: How deep the oracle looks. "chunk" executes the candidate's committed steps and
    #: scores a distance cost — cheap, and it measures a PROXY. "episode" executes the
    #: candidate and then runs the policy to the end, scoring real success. Only the
    #: second answers "does the candidate set contain an action that leads to
    #: finishing", which is the actual ceiling the diagnostic is after; the first can
    #: only say which candidate looks best under a heuristic that may itself be wrong.
    oracle_depth: str = "chunk"
    #: Episode-depth probes per episode. Each costs K full continuations, so this is
    #: the knob that decides whether the measurement is affordable.
    oracle_probes: int = 3
    #: "select" generates K candidates once and picks one. "mppi" instead treats the
    #: chunk as a distribution and refines it: sample, score, reweight by
    #: exp(-cost/temperature), shift the mean toward the good samples, repeat. The
    #: difference matters given what the ablation found — selection can only return one
    #: of the K things it was handed, so if the candidate set is poor no ranking rule
    #: can rescue it. MPPI's output is a weighted average and need not be any of them.
    #: Policy-first mode: execute the policy's chunk, and let the world model only
    #: REFUSE. Selection was measured not to work; refusal uses the model for the thing
    #: it is measurably good at (predicting outcome) instead of the thing it is not
    #: (discriminating near-identical candidates).
    #: Applicable-domain gating on the SCORER (not the policy): rank only where the
    #: dynamics was fitted, and roll back to the policy's own chunk everywhere else.
    #: Off by default so every earlier ablation reruns unchanged.
    #: How a choice is held.
    #:
    #: "steps"   — re-decide every `execute` steps.
    #: "phase"   — hold the chosen plan until the waypoint stage changes.
    #: "episode" — choose once and never re-decide.
    #:
    #: "phase" looked like the right unit and is not. A plan here spans stages: the
    #: grasp offset is committed to in `descend` and paid for in `lift`, the release
    #: offset in `transfer` and paid for in `lower`. Re-choosing at a stage boundary
    #: therefore grasps under one plan and releases under another, and the measured
    #: consequence is severe — at horizon 32 it re-decides five times and never
    #: finishes, while at horizon 64 a single chunk carries the arm across three stages
    #: and succeeds. "episode" is the coherent limit of the same idea.
    commit: str = "steps"
    #: Stretch every waypoint stage by this factor, in steps. 1.0 is the shipped
    #: demonstrator. See `ScriptedPickPlace.retimed`.
    retime: float = 1.0
    #: Record the executed action, the action the policy wanted, and the tip position
    #: at every step. Off by default because it triples the size of a run's JSON.
    trace: bool = False
    domain_gate: bool = False
    domain_state: float = DomainThresholds.state
    domain_action: float = DomainThresholds.action
    domain_spread: float = DomainThresholds.spread
    veto: bool = False
    #: "spread" (calibrated) or "drop" (the original, refuted design). See veto.py.
    veto_mode: str = "spread"
    veto_drop: float = 0.15
    veto_spread: float = 0.05
    #: Ground-truth probes for the veto itself: roll the refused action to completion
    #: and see whether it really would have failed. Without this the veto can only be
    #: reported as a firing rate, which says nothing about whether it fires correctly.
    veto_audit: int = 0
    optimiser: str = "select"
    mppi_iters: int = 3
    #: Softmax temperature on the cost. Too low and one sample takes all the weight,
    #: which is selection again with extra steps; too high and every sample weighs the
    #: same and the mean never moves.
    mppi_temperature: float = 0.05


@dataclass
class EpisodeReport:
    success: bool
    perceived_success: bool
    steps: int
    unsafe: bool
    ball_displacement: float
    plans: int = 0
    replaced: int = 0               # planner preferred something other than the policy
    vetoed: int = 0
    margins: list[float] = field(default_factory=list)
    #: Per-decision Oracle@k diagnostics. `regret` is how much true cost the chosen
    #: candidate gave up against the best available one; `rank` is where the chosen
    #: candidate sat in the oracle's ordering, normalised to [0, 1] so 0 is the best of
    #: k, 0.5 is chance and 1 is the worst. `spread` is how much the candidate set
    #: differed at all — a set whose members all do the same thing has no ceiling to
    #: reach, and that is a fact about generation, not about selection.
    oracle_regret: list[float] = field(default_factory=list)
    oracle_rank: list[float] = field(default_factory=list)
    oracle_spread: list[float] = field(default_factory=list)
    #: Episode-depth results, one entry per probed decision: whether ANY candidate led
    #: to success, and whether the CHOSEN one did. The gap between the two is the
    #: headroom a better selection rule could actually capture.
    reachable: list[bool] = field(default_factory=list)
    chosen_solved: list[bool] = field(default_factory=list)
    #: (vetoed, the action would have failed, holding instead would have succeeded)
    #: per audited decision — enough for a confusion matrix on the veto, and for
    #: whether the substitution actually rescued anything.
    veto_audit: list[tuple[bool, bool, bool]] = field(default_factory=list)
    vetoes: int = 0
    veto_checks: int = 0
    #: (decisions where the scorer was allowed to rank, decisions it abstained on,
    #: candidates dropped as off-manifold, candidates offered)
    domain: tuple[int, int, int, int] = (0, 0, 0, 0)
    #: Which candidate index was chosen at each decision, in order. The churn between
    #: consecutive entries is the difference between "the scorer committed to a plan"
    #: and "the scorer re-picked a different plan every time it was asked", and those
    #: two fail for entirely different reasons.
    chosen_idx: list[int] = field(default_factory=list)
    #: How many of the K candidates reach success, at each episode-depth probe. The
    #: binary `reachable` only says whether ANY of them does, which cannot distinguish
    #: "one narrow escape survives" from "all eight still work" — and the difference
    #: between those is exactly what a trajectory that is quietly being ruined looks
    #: like from the inside.
    viable: list[int] = field(default_factory=list)
    #: Per committed chunk, how many waypoint-stage boundaries were crossed while it
    #: executed. A chunk that spans a boundary is running actions planned for the phase
    #: it was in against the phase it is now in, which is the candidate mechanism for a
    #: longer commitment being WORSE than a shorter one.
    crossings: list[int] = field(default_factory=list)
    #: The stage each committed chunk began in, so crossings can be attributed.
    commit_stage: list[str] = field(default_factory=list)
    #: Per executed step: the action actually taken and the one the policy wanted.
    #: Recorded so a greedy run can be laid against the policy run it lost to, rather
    #: than compared only on the one bit of final success.
    executed: list[list[float]] = field(default_factory=list)
    policy_wanted: list[list[float]] = field(default_factory=list)
    tip_path: list[list[float]] = field(default_factory=list)
    #: Ground-truth cube position per step. Recorded alongside the tip because a figure
    #: of where the gripper went is only half a pick-and-place — the question is whether
    #: the cube went with it.
    cube_path: list[list[float]] = field(default_factory=list)
    perception_error: list[float] = field(default_factory=list)
    missed: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {"success": self.success, "perceived_success": self.perceived_success,
                "steps": self.steps, "unsafe": self.unsafe,
                "ball_displacement": round(self.ball_displacement, 4),
                "plans": self.plans, "replaced": self.replaced, "vetoed": self.vetoed,
                "mean_margin": round(float(np.mean(self.margins)), 5)
                if self.margins else None,
                "perception_error_m": round(float(np.mean(self.perception_error)), 4)
                if self.perception_error else None,
                "oracle_regret": round(float(np.mean(self.oracle_regret)), 5)
                if self.oracle_regret else None,
                "oracle_rank": round(float(np.mean(self.oracle_rank)), 4)
                if self.oracle_rank else None,
                "oracle_spread": round(float(np.mean(self.oracle_spread)), 5)
                if self.oracle_spread else None,
                "reachable": round(float(np.mean(self.reachable)), 4)
                if self.reachable else None,
                "chosen_solved": round(float(np.mean(self.chosen_solved)), 4)
                if self.chosen_solved else None,
                "probes": len(self.reachable),
                "vetoes": self.vetoes, "veto_checks": self.veto_checks,
                "domain": list(self.domain),
                "chosen_idx": list(self.chosen_idx), "viable": list(self.viable),
                "crossings": list(self.crossings),
                "commit_stage": list(self.commit_stage),
                "mean_crossings": (float(np.mean(self.crossings))
                                   if self.crossings else None),
                "plan_churn": (
                    float(np.mean([a != b for a, b in zip(self.chosen_idx,
                                                          self.chosen_idx[1:])]))
                    if len(self.chosen_idx) > 1 else None),
                "executed": self.executed, "policy_wanted": self.policy_wanted,
                "tip_path": self.tip_path, "cube_path": self.cube_path,
                "veto_audit": [list(map(bool, a)) for a in self.veto_audit],
                "missed": self.missed}


class ClosedLoop:
    """Runs one episode of perceive-predict-choose-act."""

    def __init__(self, env: PickPlaceEnv, perception, dynamics: DynamicsEnsemble,
                 norms: Normalisers, *, cfg: LoopConfig | None = None,
                 planner: ActionPlanner | None = None,
                 make_driver=None, value=None,
                 n_slots: int = 3, slot_geom: int = 6) -> None:
        self.env = env
        self.perception = perception
        self.cfg = cfg or LoopConfig()
        # A factory rather than an instance: the scripted driver carries per-episode
        # state and has to be rebuilt on every reset.
        self.make_driver = make_driver or (
            lambda seed: ScriptedDriver(noise=self.cfg.policy_noise,
                                        rng=np.random.default_rng(seed),
                                        retime=self.cfg.retime))
        self.norms = norms
        self.n_slots = n_slots
        self.slot_geom = slot_geom
        self.value = value
        self.gate = VetoGate(mode=self.cfg.veto_mode, drop=self.cfg.veto_drop,
                             max_spread=self.cfg.veto_spread)
        self.domain = DomainGate(DomainThresholds(state=self.cfg.domain_state,
                                                  action=self.cfg.domain_action,
                                                  spread=self.cfg.domain_spread))
        self._last_spread = 0.0
        self._last_plans = None
        if self.cfg.veto and self.cfg.veto_mode == "drop" and value is None:
            raise SystemExit(
                "veto mode needs a trained value head (--value): the gate refuses on a "
                "predicted DROP IN RETURN, and there is nothing to predict it with.")
        self.planner = planner or ActionPlanner(
            dynamics,
            TrajectoryScorer(ScoreWeights(collision=40.0, stability=8.0, progress=6.0,
                                          risk=0.0, effort=0.05)),
            # No collision veto. Placing a cube into a bowl means the two boxes
            # necessarily overlap, so a veto on predicted overlap rejects the correct
            # action for the task. The weighted term still discourages gratuitous
            # contact; the hard constraint is simply wrong for this goal.
            veto_collision_m3=None)

    # ── latent assembly ────────────────────────────────────────────────────────

    def _slots(self, perceived: dict) -> tuple[np.ndarray, np.ndarray]:
        """(M, 6) boxes in metres and (M,) validity, in a fixed slot order.

        Fixed order, not detection order. A slot index means the same object on every
        frame or the dynamics is being handed a different scene each step; and the
        planner's `target` is an index, so a reordering silently re-aims the goal term
        at whichever object happened to be found first.
        """
        boxes = np.zeros((self.n_slots, self.slot_geom), np.float32)
        ok = np.zeros(self.n_slots, bool)
        for i, lab in enumerate(SLOT_ORDER[:self.n_slots]):
            p = perceived.get(lab)
            if p is None:
                continue
            boxes[i] = p.as_slot()[:self.slot_geom]
            ok[i] = True
        return boxes, ok

    def _latent(self, boxes, prev_boxes, ok, state, prev_state) -> SceneLatent:
        n = self.norms
        g_t = (boxes - n.slot_mean) / n.slot_std
        g_p = (prev_boxes - n.slot_mean) / n.slot_std
        s_t = (state - n.state_mean) / n.state_std
        s_p = (prev_state - n.state_mean) / n.state_std
        slots = np.concatenate([g_t, g_t - g_p], -1)[None]
        robot = np.concatenate([s_t, s_t - s_p])[None]
        return SceneLatent(torch.from_numpy(slots).float(),
                           torch.from_numpy(robot).float(),
                           torch.from_numpy(ok[None]).bool())

    def _to_metres(self, states: list[SceneLatent]) -> list[SceneLatent]:
        """Undo slot standardisation on the geometry channels of a rollout.

        The scorer integrates box volumes and compares them against thresholds in cubic
        metres; standardised units would make those numbers mean nothing in particular.
        Only channels 0:6 are converted — the rest are velocities the scorer never
        reads.
        """
        mean = torch.from_numpy(self.norms.slot_mean).float()
        std = torch.from_numpy(self.norms.slot_std).float()
        out = []
        for z in states:
            s = z.slots.clone()
            s[..., :self.slot_geom] = s[..., :self.slot_geom] * std + mean
            # An extent is a size. The dynamics has no constraint keeping it positive,
            # and a negative one would give the overlap term a negative volume.
            s[..., 3:6] = s[..., 3:6].clamp(min=0.0)
            out.append(SceneLatent(s, z.robot, z.mask))
        return out

    def _candidates(self, driver, tip, seen, state, rng,
                    boxes: np.ndarray | None = None) -> np.ndarray:
        """(K, H, 4) raw-action candidates, candidate 0 being the policy's own answer.

        Keeping the unperturbed chunk as candidate 0 means the planner can never do
        worse than the policy on the scorer's own terms — it is not forced to discard
        the policy's answer to have something to choose.
        """
        cfg = self.cfg
        self._last_plans = None
        base = driver.chunk(tip, seen, state, cfg.horizon)
        if cfg.candidates_from == "affordance":
            from pipeline.sim import affordance
            plans = affordance.propose(
                boxes[0, 3:6], boxes[1, 3:6], cfg.candidates, rng,
                base_transit=cfg.transit_z)
            out = [base]
            for plan in plans[1:]:
                out.append(driver.chunk_plan(plan, tip, seen, state, cfg.horizon))
            # Kept so a phase-aligned commitment can regenerate a chunk from the plan
            # it already chose. Holding raw actions cannot do that: a chunk runs out
            # after `horizon` steps and the only way to continue is to re-choose.
            self._last_plans = plans
            return np.stack(out).astype(np.float32)
        if cfg.candidates_from == "belief":
            out = [base]
            for _ in range(cfg.candidates - 1):
                # Perturb what the robot BELIEVES, then re-plan under that belief. The
                # result is a chunk the policy would genuinely have produced, rather
                # than a chunk it produced with noise added afterwards.
                jitter = {k: v + rng.normal(0, cfg.belief_sigma, 3)
                          for k, v in seen.items()}
                out.append(driver.chunk(tip, jitter, state, cfg.horizon))
            return np.stack(out)
        if cfg.candidates_from != "action":
            raise ValueError(
                f"unknown candidate source {cfg.candidates_from!r}; use 'action', "
                "'belief' or 'affordance'")
        noise = rng.normal(0, cfg.perturb, (cfg.candidates - 1, *base.shape))
        # The grip channel is a command in [0, 1], not a displacement; jittering it in
        # metres would open the fingers mid-carry for reasons unrelated to the task.
        noise[..., 3] = 0.0
        cands = np.concatenate([base[None], base[None] + noise])
        cands[..., 3] = np.clip(cands[..., 3], 0.0, 1.0)
        return cands.astype(np.float32)

    def _true_cost(self, ball0) -> float:
        """Task progress read off the simulator. The oracle's objective.

        One continuous measure that is monotone through both phases: the tip-to-cube
        term drives the reach and goes quiet once the cube is held, and the
        cube-to-bowl term drives the transport. A pure cube-to-bowl cost would be
        identical for every candidate during the whole approach, which is exactly when
        the candidates differ most.

        The distractor term is what stops the oracle looking good by swatting the ball
        out of the way.
        """
        e = self.env
        cube, bowl, ball = (e.body_pos("cube"), e.body_pos("bowl"),
                            e.body_pos("ball"))
        return (float(np.linalg.norm(cube - e.tip))
                + float(np.linalg.norm(cube[:2] - bowl[:2]))
                + self.cfg.oracle_ball_weight * float(np.linalg.norm(ball - ball0)))

    def _oracle_costs(self, raw: np.ndarray, ball0) -> np.ndarray:
        """Actually execute every candidate, measure, and rewind.

        This is only possible because the simulator can be rewound, and it is the whole
        reason the arena earns its keep as a diagnostic: a ranking can be compared
        against what would really have happened, rather than against another model's
        opinion of what would have happened.
        """
        snap = self.env.snapshot()
        costs = np.empty(len(raw))
        for k, chunk in enumerate(raw):
            for a in chunk[:self.cfg.execute]:
                self.env.step(a)
            costs[k] = self._true_cost(ball0)
            self.env.restore(snap)
        return costs

    def _score(self, raw: np.ndarray, z0: SceneLatent, goal_xyz: np.ndarray):
        """Predicted cost per candidate. The one place the world model is consulted."""
        k = len(raw)
        cands = torch.from_numpy(
            (raw - self.norms.action_mean) / self.norms.action_std).float()
        states, unc = self.planner.dynamics.rollout(z0.expand_batch(k), cands)
        # Stashed rather than returned so the applicable-domain gate can read the
        # epistemic spread without paying for a second rollout, and so no caller's
        # tuple unpacking changes.
        self._last_spread = float(unc.max()) if unc.numel() else 0.0
        return self.planner.scorer(
            self._to_metres(states), unc, cands, target=0,
            goal_xyz=torch.from_numpy(np.asarray(goal_xyz, np.float32)).float(),
            baseline=self._to_metres([z0])[0].expand_batch(k)), cands

    def _mppi(self, driver, tip, seen, state, z0, goal_xyz, rng) -> np.ndarray:
        """Refine the policy's chunk by reward-weighted averaging over samples.

        Returns a single chunk, not a set. That is the point: it can land between the
        samples, so it is not limited to the best draw the way a selection rule is.
        """
        cfg = self.cfg
        mean = driver.chunk(tip, seen, state, cfg.horizon).astype(np.float64)
        for _ in range(cfg.mppi_iters):
            noise = rng.normal(0, cfg.perturb, (cfg.candidates, *mean.shape))
            noise[..., 3] = 0.0          # grip is a command, not a displacement
            samples = mean[None] + noise
            samples[..., 3] = np.clip(samples[..., 3], 0.0, 1.0)
            scores, _ = self._score(samples.astype(np.float32), z0, goal_xyz)
            cost = scores.total.detach().numpy().astype(np.float64)
            # Shifted by the minimum before exponentiating: the costs are unnormalised
            # and can be large, and exp of a large negative number underflows every
            # weight to zero, leaving the mean update as 0/0.
            w = np.exp(-(cost - cost.min()) / max(1e-9, cfg.mppi_temperature))
            w = w / max(1e-12, w.sum())
            mean = mean + (w[:, None, None] * noise).sum(0)
        mean[..., 3] = np.clip(mean[..., 3], 0.0, 1.0)
        return mean.astype(np.float32)

    def _continue_to_end(self, driver, chunk: np.ndarray) -> bool:
        """Execute one chunk, then run the policy to the end. Did the episode finish?

        The primitive both the episode-depth oracle and the veto audit are built from.
        Rewinds afterwards, so asking the question costs the live episode nothing.
        """
        env, cfg = self.env, self.cfg
        snap, dsnap = env.snapshot(), driver.snapshot()
        obs = None
        for a in chunk[:cfg.execute]:
            obs = env.step(a)
        while not env.done():
            seen = env.objects
            act = driver.chunk(env.tip, seen, obs.state, cfg.horizon)[0]
            driver.advance(env.tip, seen)
            obs = env.step(act)
        ok = bool(obs.success)
        env.restore(snap)
        driver.restore(dsnap)
        return ok

    def _episode_oracle(self, driver, raw: np.ndarray) -> np.ndarray:
        """Per candidate: execute it, then run the policy to the end. Did it finish?

        This is the ceiling the whole diagnostic is after. The chunk-depth oracle can
        only rank candidates by a hand-written distance cost, so a low ceiling there is
        ambiguous — it could mean the candidates are all equivalent, or it could mean
        the cost is wrong. Running to completion and reading the simulator's own
        success flag removes the proxy from the measurement entirely.

        The continuation uses GROUND-TRUTH object positions rather than perception.
        That is deliberate: the question is whether a candidate *can* lead to success
        under a competent continuation, not whether this particular segmenter would
        also have got there. Folding perception failures into the ceiling would
        understate the headroom and blame it on selection.
        """
        env, cfg = self.env, self.cfg
        snap, dsnap = env.snapshot(), driver.snapshot()
        solved = np.zeros(len(raw), bool)
        for k, chunk in enumerate(raw):
            env.restore(snap)
            driver.restore(dsnap)
            obs = None
            for a in chunk[:cfg.execute]:
                obs = env.step(a)
            while not env.done():
                seen = env.objects
                act = driver.chunk(env.tip, seen, obs.state, cfg.horizon)[0]
                driver.advance(env.tip, seen)
                obs = env.step(act)
            solved[k] = bool(obs.success)
        env.restore(snap)
        driver.restore(dsnap)
        return solved

    # ── the loop ───────────────────────────────────────────────────────────────

    def run(self, *, seed: int = 0) -> EpisodeReport:
        cfg = self.cfg
        env = self.env
        obs = env.reset()
        driver = self.make_driver(seed)
        ball0 = env.body_pos("ball").copy()
        rep = EpisodeReport(False, False, 0, False, 0.0)
        rng = np.random.default_rng(seed + 7919)
        # The gate is shared across episodes so its thresholds and counters are one
        # object; the per-episode report takes a difference rather than the running
        # total, or every episode after the first would inherit the previous one's.
        d0 = (self.domain.decisions, self.domain.abstained, self.domain.dropped,
              self.domain.offered)

        frame = env.render(cfg.camera)
        perceived = self.perception(frame)
        boxes, ok = self._slots(perceived)
        prev_boxes, prev_state = boxes.copy(), obs.state.copy()
        pending: list[np.ndarray] = []
        # Phase-aligned commitment: the PLAN that was chosen and the stage it was
        # chosen in. The chunk is regenerated from that plan whenever the buffer runs
        # dry, so the choice survives longer than any fixed chunk length can.
        held_plan = None
        held_stage = getattr(getattr(driver, "pol", None), "stage_name", None)
        # Stage-crossing bookkeeping for the committed chunk currently executing.
        chunk_stage0 = held_stage
        chunk_cross = 0
        prev_stage = held_stage

        while not env.done():
            if obs.step % cfg.perceive_every == 0:
                frame = env.render(cfg.camera)
                perceived = self.perception(frame)
                new_boxes, new_ok = self._slots(perceived)
                # Carry a missing detection forward rather than zeroing it. A slot of
                # zeros is not "unknown", it is "this object is at the origin, 0 m
                # across", and the collision and goal terms would both believe it.
                boxes = np.where(new_ok[:, None], new_boxes, boxes)
                ok = new_ok | ok
                for lab in SLOT_ORDER[:self.n_slots]:
                    if lab not in perceived:
                        rep.missed[lab] = rep.missed.get(lab, 0) + 1
                truth = env.objects
                for lab, p in perceived.items():
                    if lab in truth:
                        rep.perception_error.append(
                            float(np.linalg.norm(p.centre - truth[lab])))

            seen = {lab: boxes[i, :3] for i, lab in
                    enumerate(SLOT_ORDER[:self.n_slots]) if ok[i]}
            stage_now = getattr(getattr(driver, "pol", None), "stage_name", None)
            if stage_now != prev_stage:
                chunk_cross += 1
                prev_stage = stage_now
            if cfg.commit in ("phase", "episode") and held_plan is not None \
                    and pending == []:
                if cfg.commit == "episode" or stage_now == held_stage:
                    # Same phase, buffer empty: continue the plan already chosen rather
                    # than re-opening the decision. This is the whole intervention.
                    # The driver has adopted the plan, so its own chunk continues it.
                    pending = list(driver.chunk(obs.tip, seen, obs.state,
                                                cfg.horizon))
                else:
                    held_plan, held_stage = None, stage_now
            if not pending:
                if cfg.veto and ok[0] and ok[1]:
                    # Policy-first. The chunk the policy wants is what executes unless
                    # the model is confident it is a disaster; there is no candidate
                    # set and nothing is ranked.
                    chunk = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
                    z0 = self._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                    norm = torch.from_numpy(
                        (chunk - self.norms.action_mean)
                        / self.norms.action_std).float()
                    d = self.gate(self.planner.dynamics, self.value, z0, norm)
                    rep.veto_checks += 1
                    audit_due = (cfg.veto_audit > 0 and any(
                        abs(obs.step - t) < cfg.execute
                        for t in np.linspace(0, env.max_steps,
                                             cfg.veto_audit + 2)[1:-1]))
                    held = hold_chunk(cfg.horizon, obs.grip)
                    if audit_due:
                        # Both counterfactuals, so the veto can be scored on whether
                        # the action really was a disaster AND on whether refusing it
                        # actually rescued anything.
                        allow_ok = self._continue_to_end(driver, chunk)
                        hold_ok = self._continue_to_end(driver, held)
                        rep.veto_audit.append(
                            (not d.allowed, not allow_ok, hold_ok))
                    if d.allowed:
                        chosen = chunk
                    else:
                        chosen = held
                        rep.vetoes += 1
                elif cfg.plan and ok[0] and ok[1]:
                    z0 = self._latent(boxes, prev_boxes, ok, obs.state, prev_state)
                    if cfg.optimiser == "mppi":
                        refined = self._mppi(driver, obs.tip, seen, obs.state, z0,
                                             boxes[1, :3], rng)
                        base = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
                        # Candidate 0 stays the untouched policy chunk so the same
                        # Oracle@k diagnostics apply and MPPI can be credited or not
                        # against the thing it was supposed to improve.
                        raw = np.stack([base, refined])
                    else:
                        raw = self._candidates(driver, obs.tip, seen, obs.state, rng,
                                               boxes)
                    scores, _ = self._score(raw, z0, boxes[1, :3])
                    want_oracle = (cfg.pick in ("oracle", "deep-oracle")
                                   or cfg.oracle_diagnostics)
                    true_costs = (self._oracle_costs(raw, ball0) if want_oracle
                                  else None)
                    # Episode-depth probes are sampled, not run every decision: each
                    # one costs K full continuations, so running them throughout would
                    # cost more than the experiment they inform.
                    deep = None
                    if cfg.pick == "deep-oracle":
                        # Selection BY the thing the other modes only diagnose. Every
                        # `--pick oracle` number is a hand-written distance cost read
                        # off privileged state over one chunk; this one rewinds each
                        # candidate to the end of the episode and reads the simulator's
                        # own success flag, so it is the only setting entitled to the
                        # word ceiling. Affordable only when decisions are rare — at
                        # `--commit episode` it is K continuations once per episode.
                        deep = self._episode_oracle(driver, raw)
                    elif cfg.oracle_depth == "episode" and cfg.oracle_probes > 0:
                        due = np.linspace(0, env.max_steps, cfg.oracle_probes + 2)[1:-1]
                        if any(abs(obs.step - d) < cfg.execute for d in due):
                            deep = self._episode_oracle(driver, raw)
                    # Applicable-domain gate. It sits between scoring and choosing so
                    # the scores are still computed and reported — an abstention has to
                    # be distinguishable from a scorer that happened to pick candidate
                    # 0, and those two have completely different fixes.
                    allowed = None
                    if cfg.domain_gate:
                        verdict = self.domain(z0, raw, raw[0],
                                              self.norms.action_std,
                                              spread=self._last_spread)
                        allowed = None if verdict.abstain else verdict.in_domain
                    if cfg.pick == "random":
                        idx = int(rng.integers(len(raw)))
                    elif cfg.pick == "first":
                        idx = 0
                    elif cfg.pick == "oracle":
                        idx = int(np.argmin(true_costs))
                    elif cfg.pick == "deep-oracle":
                        # Ties are broken toward the policy's own chunk (candidate 0)
                        # so that "several candidates finish" is not silently scored as
                        # "selection helped" — it has to beat doing nothing to count.
                        idx = 0 if deep[0] else (int(np.argmax(deep)) if deep.any()
                                                 else 0)
                    elif cfg.optimiser == "mppi":
                        idx = 1              # the refined mean
                    elif cfg.domain_gate and allowed is None:
                        # Roll back to the policy. Candidate 0 is the policy's own
                        # chunk by construction in every proposer here, so abstaining
                        # is exactly "do what the policy wanted" and never a third
                        # behaviour the ablation would have to account for.
                        idx = 0
                    else:
                        cost = scores.total.detach().numpy().astype(np.float64)
                        if allowed is not None:
                            cost = np.where(allowed, cost, np.inf)
                        idx = int(np.argmin(cost))
                    if deep is not None:
                        rep.reachable.append(bool(deep.any()))
                        rep.viable.append(int(deep.sum()))
                    if true_costs is not None:
                        order = np.argsort(true_costs)
                        rep.oracle_regret.append(
                            float(true_costs[idx] - true_costs[order[0]]))
                        # Normalised so 0 is the best of k, 1 the worst, and a rule
                        # that knows nothing averages 0.5. Reporting the raw index
                        # instead would make the number depend on k.
                        rep.oracle_rank.append(
                            float(np.where(order == idx)[0][0])
                            / max(1, len(order) - 1))
                        rep.oracle_spread.append(
                            float(true_costs.max() - true_costs.min()))
                    if deep is not None:
                        rep.chosen_solved.append(bool(deep[idx]))
                    rep.plans += 1
                    rep.replaced += int(idx != 0)
                    rep.chosen_idx.append(idx)
                    if chunk_stage0 is not None:
                        rep.crossings.append(chunk_cross)
                        rep.commit_stage.append(chunk_stage0)
                    chunk_stage0, chunk_cross = stage_now, 0
                    if cfg.commit in ("phase", "episode"):
                        plans = self._last_plans
                        held_plan = (plans[idx] if plans is not None
                                     and idx < len(plans) else None)
                        if held_plan is not None and hasattr(driver, "adopt"):
                            # The executed plan becomes the driver's own, so the stage
                            # that defines the commitment boundary is the stage of the
                            # motion actually being performed.
                            driver.adopt(held_plan)
                        held_stage = getattr(getattr(driver, "pol", None),
                                             "stage_name", None)
                    ranked = torch.sort(scores.total).values
                    # With a single candidate there is no runner-up and so no margin.
                    # Recorded as absent rather than 0.0: a margin of zero means "the
                    # top two were tied", which is the opposite of "nothing to compare".
                    if len(ranked) > 1:
                        rep.margins.append(float(ranked[1] - ranked[0]))
                    chosen = raw[idx]
                else:
                    chosen = driver.chunk(obs.tip, seen, obs.state, cfg.horizon)
                pending = list(chosen[:cfg.execute])

            # Advance the waypoint machine on the REAL observation, and throw its
            # action away — the executed action comes from the planner. Without this
            # the machine never moves at all: `scripted_chunk` restores whatever state
            # it borrowed, so the only thing that ever advanced it was the lookahead,
            # and the loop hovered over the cube for the full episode. Driving the
            # stage from reality rather than from the dead-reckoned chunk also means a
            # grasp that slips is noticed, instead of the machine proceeding on the
            # assumption that its own plan worked.
            driver.advance(obs.tip, seen)
            prev_boxes, prev_state = boxes.copy(), obs.state.copy()
            act = pending.pop(0)
            if cfg.trace:
                # `scripted_chunk` restores whatever state it borrows, so asking the
                # driver what it wanted does not advance it — the same property the
                # lookahead relies on.
                want = driver.chunk(obs.tip, seen, obs.state, 1)[0]
                rep.executed.append([float(v) for v in act])
                rep.policy_wanted.append([float(v) for v in want])
                rep.tip_path.append([float(v) for v in obs.tip])
                rep.cube_path.append([float(v) for v in env.body_pos("cube")])
            obs = env.step(act)

        rep.steps = obs.step
        rep.success = bool(obs.success)
        disp = float(np.linalg.norm(env.body_pos("ball") - ball0))
        rep.ball_displacement = disp
        rep.unsafe = disp > cfg.unsafe_displacement
        final = self.perception(env.render(cfg.camera))
        rep.perceived_success = bool(perceived_success(final, obs.grip).success)
        d = self.domain
        ranked = (d.decisions - d0[0]) - (d.abstained - d0[1])
        rep.domain = (ranked, d.abstained - d0[1], d.dropped - d0[2],
                      d.offered - d0[3])
        return rep


def task_fingerprint(max_env_steps: int) -> dict:
    """What the task IS, hashed, so two runs can be checked for comparability.

    Some constants are parameters of an experiment and some are part of the task
    definition. `--episodes` is the first kind; `stage_timeout` is the second — adding
    it changed closed-loop success at K=32 from 37.5% to 79.2% with non-overlapping
    intervals, because a stalled waypoint used to burn the rest of the episode. A run
    from either side of that change is not a measurement of the same thing, and
    comparing them is a category error rather than a noisy comparison.

    `max_env_steps` is here for the same reason and it was learned the same way, one
    bug later: the episode budget is not a knob on an experiment, it is how long the
    task gives you. Re-running the K sweep with 480 steps instead of 360 moved K=16
    from 18.8% to 83.3% while this function — which did not yet read it — reported both
    runs under the identical hash 3de92c99cbe5. A comparability check that omits a
    field does not stay silent about it; it actively certifies the wrong answer.

    Every run records this, so "are these two numbers comparable" is answered by the
    files rather than by remembering when a constant moved.
    """
    import hashlib

    from pipeline.sim.arm import ArmGeometry, WORKSPACE
    from pipeline.sim.policy import GRASP_DZ, RELEASE_Z, ScriptedPickPlace, TRANSIT_Z

    g = ArmGeometry()
    fields = {
        # how long the task gives you
        "max_env_steps": int(max_env_steps),
        # the demonstrator
        "max_step": ScriptedPickPlace.max_step,
        "grip_hold": ScriptedPickPlace.grip_hold,
        "settle_hold": ScriptedPickPlace.settle_hold,
        "tolerance": ScriptedPickPlace.tolerance,
        "stage_timeout": ScriptedPickPlace.stage_timeout,
        "transit_z": TRANSIT_Z, "grasp_dz": GRASP_DZ, "release_z": RELEASE_Z,
        # the arm
        "links": (g.shoulder_height, g.upper, g.fore, g.wrist_to_tip,
                  g.finger_open, g.finger_closed),
        "workspace": tuple(np.asarray(WORKSPACE).ravel().tolist())
        if not isinstance(WORKSPACE, dict) else tuple(sorted(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in WORKSPACE.items())),
        # the arena's physics, read from the XML that will actually be loaded
        "arena_sha": hashlib.sha256(
            (Path(__file__).with_name("arena.xml")).read_bytes()).hexdigest()[:12],
    }
    blob = repr(sorted(fields.items())).encode()
    return {"task_sha": hashlib.sha256(blob).hexdigest()[:12], **fields}


def bootstrap_ci(values, reps: int = 10000, seed: int = 0,
                 pct: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """95% bootstrap CI for a per-episode rate.

    Episodes are the independent unit — decisions inside one share a scene, a
    randomisation draw and a policy seed — so this resamples episodes and nothing
    finer. Reported on every closed-loop rate because at 24 episodes one flipped
    outcome is 4.2 points, and a difference smaller than that is not a difference:
    this project has already published a "+4.2 per 100" whose interval turned out to
    be [-1.5, +10.0].

    Percentile method rather than normal-approximate, because the statistic is a
    bounded proportion and a normal interval runs outside [0, 1] near the ends — which
    is exactly where a 95%-success arm sits.
    """
    v = np.asarray(values, float)
    if not len(v):
        return (float("nan"), float("nan"))
    rs = np.random.default_rng(seed)
    draws = v[rs.integers(0, len(v), (reps, len(v)))].mean(1)
    lo, hi = np.percentile(draws, pct)
    return (float(lo), float(hi))


class ValueCost(torch.nn.Module):
    """Wraps a return-predicting head so the scorer receives a COST.

    `ValueHead` is trained on discounted return, where **higher is better**.
    `TrajectoryScorer` adds its value term into a total it then MINIMISES, so handing it
    a return unchanged would make the planner steer toward exactly the states it should
    avoid — and it would do so quietly, since the number is finite and well-scaled and
    every diagnostic would keep working.

    Negated here rather than by training the head on `1 - return`, so the head's own
    held-out MSE stays comparable with the return baselines it was validated against.
    """

    def __init__(self, head, scale: float = 1.0) -> None:
        super().__init__()
        self.head = head
        self.scale = scale

    def forward(self, z: SceneLatent) -> torch.Tensor:
        return -self.scale * self.head(z)


def load_value(path: Path, scale: float = 1.0) -> ValueCost:
    """Load a trained value head, or refuse if it is not better than the heuristic.

    The head exists to replace the scorer's geometric distance term, so a head that
    does not beat a least-squares fit on that same distance is not an improvement — it
    is a larger, slower restatement of it. `train_value.py` records both numbers in the
    checkpoint precisely so this check can be made here rather than left to a reader.
    """
    from pipeline.world_model.scorer import ValueHead

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("distance_mse") is not None and ck["test_mse"] >= ck["distance_mse"]:
        raise SystemExit(
            f"{path}: held-out MSE {ck['test_mse']:.6f} is no better than the "
            f"cube-to-goal distance fit it replaces ({ck['distance_mse']:.6f}). "
            "Scoring with it would change the objective without improving it.")
    head = ValueHead(ck["slot_dim"], ck["robot_dim"], hidden=ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return ValueCost(head, scale)


def load_dynamics(path: Path) -> tuple[DynamicsEnsemble, Normalisers, int, int]:
    """Load a trained ensemble, or refuse. Never reshape to fit."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dyn = DynamicsEnsemble(ck["slot_dim"], ck["robot_dim"], ck["action_dim"],
                           n_members=ck["members"], hidden=ck["hidden"],
                           layers=ck["layers"],
                           integrate_velocity=ck["integrate_velocity"],
                           integrate_slot_velocity=ck["integrate_slot_velocity"])
    dyn.load_state_dict(ck["state_dict"])
    dyn.eval()
    norms = Normalisers.from_checkpoint(ck)
    if norms is None:
        raise SystemExit(
            f"{path} carries no slot normaliser, so its slot inputs cannot be scaled "
            "the way it was trained. Retrain with the current train.py rather than "
            "guessing the scaling.")
    n_slots = int(ck.get("n_slots") or 3)
    slot_geom = ck["slot_dim"] // 2 if ck["integrate_slot_velocity"] else ck["slot_dim"]
    return dyn, norms, n_slots, slot_geom


def load_chunk_policy(path: Path) -> ChunkPolicy:
    """Load the behaviour-cloned policy, or refuse. Never reshape to fit."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pol = ChunkPolicy(ck["obs_dim"], horizon=ck["horizon"], hidden=ck["hidden"])
    pol.net.load_state_dict(ck["state_dict"])
    pol.net.eval()
    pol.obs_mean = np.asarray(ck["obs_mean"], np.float32)
    pol.obs_std = np.asarray(ck["obs_std"], np.float32)
    return pol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("pipeline/assets/world_model_sim/dynamics.pt"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--perception", default="oracle", choices=("oracle", "sam"))
    ap.add_argument("--camera", default="front")
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--policy", default="scripted", choices=("scripted", "bc"))
    ap.add_argument("--policy-checkpoint", type=Path,
                    default=Path("pipeline/assets/sim_policy/chunk_policy.pt"))
    ap.add_argument("--candidates-from", default="action",
                    choices=("action", "belief", "affordance"),
                    help="'belief' perturbs perceived object positions and re-runs "
                         "the policy; 'affordance' proposes distinct grasp and release "
                         "points from the perceived geometry, so candidates differ "
                         "task-meaningfully rather than by noise")
    ap.add_argument("--belief-sigma", type=float, default=0.014)
    ap.add_argument("--value", type=Path, default=None,
                    help="score predicted task success with a trained ValueHead "
                         "instead of relying on geometric distance alone")
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--progress-weight", type=float, default=6.0,
                    help="set to 0 alongside --value to replace the distance term "
                         "rather than adding to it")
    ap.add_argument("--veto", action="store_true",
                    help="policy-first: execute the policy's chunk, and let the world "
                         "model only refuse high-confidence disasters")
    # NOT --trace: `conda run` swallows that one and turns on its own debug logging,
    # which is a confusing way to discover a flag collision.
    ap.add_argument("--commit", default="steps",
                    choices=("steps", "phase", "episode"),
                    help="'steps' re-decides every --execute steps; 'phase' holds the "
                         "chosen plan until the waypoint stage changes; 'episode' "
                         "chooses once. Only meaningful with --candidates-from "
                         "affordance, where a candidate IS a plan")
    ap.add_argument("--retime", type=float, default=1.0,
                    help="stretch every waypoint stage by this factor. The prediction "
                         "under test is that the commitment fracture moves with it")
    ap.add_argument("--max-env-steps", type=int, default=360,
                    help="episode cap. Scale it with --retime or a slowed task simply "
                         "runs out of time and every arm reads 0%%")
    ap.add_argument("--record", action="store_true",
                    help="record executed vs policy-wanted actions and the tip path, "
                         "so a greedy run can be laid against the policy run")
    ap.add_argument("--domain-gate", action="store_true",
                    help="rank only inside the region the dynamics was fitted on; "
                         "roll back to the policy's own chunk everywhere else")
    ap.add_argument("--domain-action", type=float, default=DomainThresholds.action,
                    help="per-candidate deviation from the policy's own chunk, in "
                         "standardised action units. This is the armed one")
    ap.add_argument("--domain-state", type=float, default=DomainThresholds.state,
                    help="unarmed backstop (inf): measured not to separate good "
                         "rankings from bad ones")
    ap.add_argument("--domain-spread", type=float, default=DomainThresholds.spread,
                    help="unarmed backstop (inf): predicts disaster, not ranking")
    ap.add_argument("--veto-mode", default="spread", choices=("spread", "drop"),
                    help="'spread' fires on ensemble disagreement (AUC 0.587); 'drop' "
                         "fires on predicted return loss, the original design the "
                         "calibration refuted (AUC 0.457)")
    ap.add_argument("--veto-drop", type=float, default=0.15)
    ap.add_argument("--veto-spread", type=float, default=0.05)
    ap.add_argument("--veto-audit", type=int, default=0,
                    help="ground-truth probes per episode: roll the refused action to "
                         "completion to see whether it really was a disaster")
    ap.add_argument("--optimiser", default="select", choices=("select", "mppi"))
    ap.add_argument("--mppi-iters", type=int, default=3)
    ap.add_argument("--mppi-temperature", type=float, default=0.05)
    ap.add_argument("--oracle-depth", default="chunk",
                    choices=("chunk", "episode"),
                    help="'episode' rolls each candidate to completion and reads real "
                         "success — the true ceiling, at K continuations per probe")
    ap.add_argument("--oracle-probes", type=int, default=3)
    ap.add_argument("--oracle-diagnostics", action="store_true",
                    help="record Oracle@k regret and rank whatever --pick is in use")
    ap.add_argument("--pick", default="best",
                    choices=("best", "random", "first", "oracle", "deep-oracle"),
                    help="'random' is the control that isolates ranking quality from "
                         "the noise the candidate set itself adds; 'oracle' ranks by a "
                         "distance cost on privileged state over one chunk, which is a "
                         "heuristic and can be wrong; 'deep-oracle' rewinds every "
                         "candidate to the end of the episode and ranks by the "
                         "simulator's own success flag, which is the real ceiling")
    ap.add_argument("--policy-noise", type=float, default=0.0)
    ap.add_argument("--perceive-every", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--execute", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--perturb", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-randomise", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    dyn, norms, n_slots, slot_geom = load_dynamics(a.checkpoint)
    cfg = LoopConfig(camera=a.camera, horizon=a.horizon, execute=a.execute,
                     candidates=a.candidates, perturb=a.perturb, plan=not a.no_plan,
                     commit=a.commit, retime=a.retime, trace=a.record,
                     domain_gate=a.domain_gate,
                     domain_state=a.domain_state,
                     domain_action=a.domain_action, domain_spread=a.domain_spread,
                     veto=a.veto, veto_mode=a.veto_mode, veto_drop=a.veto_drop,
                     veto_spread=a.veto_spread,
                     veto_audit=a.veto_audit,
                     pick=a.pick, oracle_diagnostics=a.oracle_diagnostics,
                     oracle_depth=a.oracle_depth, oracle_probes=a.oracle_probes,
                     candidates_from=a.candidates_from,
                     belief_sigma=a.belief_sigma, optimiser=a.optimiser,
                     mppi_iters=a.mppi_iters, mppi_temperature=a.mppi_temperature,
                     perceive_every=a.perceive_every,
                     policy_noise=a.policy_noise)
    env = PickPlaceEnv(seed=a.seed, randomise=not a.no_randomise, cameras=(),
                       max_steps=a.max_env_steps)
    if a.policy == "bc":
        bc = load_chunk_policy(a.policy_checkpoint)
        cfg.horizon = bc.horizon
        make_driver = lambda _seed: ChunkDriver(bc)          # noqa: E731
    else:
        make_driver = None
    planner = None
    if a.value is not None or a.progress_weight != 6.0:
        weights = ScoreWeights(collision=40.0, stability=8.0,
                               progress=a.progress_weight, risk=0.0, effort=0.05)
        scorer = TrajectoryScorer(weights, value_head=None)
        planner = ActionPlanner(dyn, scorer, veto_collision_m3=None)
    value = load_value(a.value, a.value_scale) if a.value else None
    loop = ClosedLoop(env, make_perception(a.perception), dyn, norms, cfg=cfg,
                      planner=planner, make_driver=make_driver, value=value,
                      n_slots=n_slots, slot_geom=slot_geom)

    rows = []
    for ep in range(a.episodes):
        r = loop.run(seed=a.seed * 1000 + ep)
        rows.append(r.summary())
        print(f"[loop] ep{ep:03d} {'OK  ' if r.success else 'FAIL'} "
              f"steps={r.steps:3d} unsafe={int(r.unsafe)} "
              f"replaced={r.replaced}/{r.plans} "
              f"perc_err={rows[-1]['perception_error_m']} "
              f"perceived={int(r.perceived_success)}", flush=True)
    env.close()

    n = max(1, len(rows))
    agree = sum(r["success"] == r["perceived_success"] for r in rows)


    errs = [r["perception_error_m"] for r in rows if r["perception_error_m"]]
    out = {"config": vars(a) | {"checkpoint": str(a.checkpoint)},
           "task": task_fingerprint(a.max_env_steps),
           "episodes": rows,
           "success_rate": sum(r["success"] for r in rows) / n,
           "unsafe_rate": sum(r["unsafe"] for r in rows) / n,
           "reward_agreement": agree / n,
           "replaced_frac": (sum(r["replaced"] for r in rows)
                             / max(1, sum(r["plans"] for r in rows))),
           "mean_perception_error_m": float(np.mean(errs)) if errs else None,
           "success_ci": bootstrap_ci([r["success"] for r in rows]),
           "unsafe_ci": bootstrap_ci([r["unsafe"] for r in rows]),
           "reward_agreement_ci": bootstrap_ci(
               [r["success"] == r["perceived_success"] for r in rows])}
    # The veto is a binary classifier, so it is scored like one. A firing rate alone
    # cannot distinguish a gate that catches disasters from one that blocks good
    # actions at the same frequency.
    audits = [tuple(a) for r in rows for a in r["veto_audit"]]
    if audits:
        tp = sum(1 for v, bad, _ in audits if v and bad)
        fp = sum(1 for v, bad, _ in audits if v and not bad)
        fn = sum(1 for v, bad, _ in audits if not v and bad)
        tn = sum(1 for v, bad, _ in audits if not v and not bad)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if tp and prec + rec > 0 else 0.0
        # Of the actions it refused, how many would holding actually have saved? A
        # veto that correctly identifies a doomed action but cannot improve on it is a
        # true positive that buys nothing.
        rescued = [hold for v, bad, hold in audits if v and bad]
        out["veto"] = {"confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                       "precision": prec, "recall": rec, "f1": f1,
                       "audited": len(audits),
                       "rescue_rate": (sum(rescued) / len(rescued)) if rescued else None}
        print(f"\n[loop] veto as a classifier ({len(audits)} audited decisions)")
        print("                  vetoed     allowed")
        print(f"    would fail {tp:8d} {fn:11d}")
        print(f"    would work {fp:8d} {tn:11d}")
        print(f"    precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}")
        if rescued:
            print(f"    of the disasters it caught, holding instead rescued "
                  f"{sum(rescued)}/{len(rescued)}")
    for key in ("oracle_regret", "oracle_rank", "oracle_spread", "reachable",
                "chosen_solved"):
        vals = [r[key] for r in rows if r[key] is not None]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else None
    dom = np.array([r["domain"] for r in rows], int).sum(0) if rows else None
    if dom is not None and dom.sum():
        ranked, abst, dropped, offered = (int(v) for v in dom)
        out["domain"] = {"ranked": ranked, "abstained": abst,
                         "candidates_dropped": dropped,
                         "candidates_offered": offered}
        print(f"\n[loop] applicable-domain gate: ranked {ranked}, abstained "
              f"{abst} ({abst / max(1, ranked + abst) * 100:.1f}% rolled back to the "
              f"policy), dropped {dropped}/{offered} candidates as off-manifold")
    fired = sum(r["vetoes"] for r in rows)
    checks = sum(r["veto_checks"] for r in rows)
    if checks:
        print(f"\n[loop] veto fired on {fired}/{checks} decisions "
              f"({fired / max(1, checks) * 100:.1f}%)")
    print(f"\n[loop] policy={a.policy} opt={a.optimiser} "
          f"value={'on' if a.value else 'off'} progress_w={a.progress_weight} "
          f"plan={not a.no_plan} pick={a.pick} "
          f"candidates={a.candidates_from} perception={a.perception} "
          f"noise={a.policy_noise}\n"
          f"       success {out['success_rate']*100:.1f}% "
          f"[{out['success_ci'][0]*100:.1f}, {out['success_ci'][1]*100:.1f}]   "
          f"unsafe {out['unsafe_rate']*100:.1f}% "
          f"[{out['unsafe_ci'][0]*100:.1f}, {out['unsafe_ci'][1]*100:.1f}]\n"
          f"       reward agreement {out['reward_agreement']*100:.1f}% "
          f"[{out['reward_agreement_ci'][0]*100:.1f}, "
          f"{out['reward_agreement_ci'][1]*100:.1f}]   "
          f"planner replaced policy on {out['replaced_frac']*100:.1f}% of decisions\n"
          f"       (95% bootstrap CI over {n} episodes; one flipped episode is "
          f"{100/n:.1f} points)\n"
          f"       task {out['task']['task_sha']} — comparable only with runs "
          f"carrying the same fingerprint")
    if out["mean_perception_error_m"] is not None:
        print(f"       perception error {out['mean_perception_error_m']*100:.2f} cm")
    viable = [v for r in rows for v in r.get("viable", [])]
    if viable:
        out["viable"] = viable
        k = max(viable) if viable else 0
        print(f"\n[loop] of {a.candidates} candidates, how many reach success "
              f"({len(viable)} probed decisions)")
        print(f"      {'viable':>8} {'decisions':>10} {'share':>7}")
        for n in range(0, max(2, k + 1)):
            c = sum(1 for v in viable if v == n)
            if c:
                print(f"      {n:>8} {c:>10} {c/len(viable)*100:6.1f}%")
        print(f"      mean {np.mean(viable):.2f} of {a.candidates}; "
              f"{sum(1 for v in viable if v == 0)/len(viable)*100:.1f}% of states have "
              f"NO surviving candidate")
    if out["mean_reachable"] is not None:
        print(f"       Oracle@k (episode depth): a candidate reaches success at "
              f"{out['mean_reachable']*100:.1f}% of probed decisions; the chosen one "
              f"does at {out['mean_chosen_solved']*100:.1f}% — headroom "
              f"{(out['mean_reachable'] - out['mean_chosen_solved'])*100:+.1f} pts")
    if out["mean_oracle_rank"] is not None:
        print(f"       Oracle@k: chosen candidate ranks "
              f"{out['mean_oracle_rank']:.3f} of 1 (0=best, 0.5=chance), "
              f"regret {out['mean_oracle_regret']:.4f}, "
              f"candidate spread {out['mean_oracle_spread']:.4f}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"[loop] -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

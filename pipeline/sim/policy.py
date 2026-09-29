"""The small policy that stands in for GR00T inside the simulator.

GR00T is not used here, and the reason is embodiment rather than preference: its action
space is a 29-DoF GR1 humanoid — two 7-DoF arms, two 6-DoF hands, a 3-DoF waist — and
this arena has a 4-DoF Cartesian gripper. Driving one with the other means retargeting,
which would change what the sim-to-sim result measures. GR00T keeps the real-data path;
the sim gets a policy sized to it.

Two policies live here and they do different jobs:

  * `ScriptedPickPlace` is the *demonstrator*. It reads object positions and runs a
    waypoint machine. It is not a baseline to beat — it is where training data comes
    from, and it is the thing whose behaviour a learned policy is cloned from.
  * `ChunkPolicy` is the *learned* policy: a small MLP that maps proprioception plus
    perceived object positions to an action **chunk**, H steps at once.

Both take object positions as an argument rather than reading the simulator, so the
same policy object runs on ground-truth state or on what SAM+CLIP actually perceived,
and the difference between those two runs is the perception ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CUBE = "a small dark cube"
BOWL = "an orange plastic bowl"

#: Heights the waypoint machine works between, in metres above the table.
TRANSIT_Z = 0.20
#: Tip height for the grasp, relative to the cube's centre. Slightly low so the pads
#: straddle the cube's upper half rather than its top corner.
GRASP_DZ = -0.002
#: Tip height at release. The pads are narrower than the bowl's inner radius at this
#: height, so they can enter it; going this low turns a 14 cm drop into 4 cm and stops
#: the cube bouncing back out over the rim, which is a failure the policy did not cause.
#: Tied to the rim (60 mm) — with the flared profile the walls lean outward, so a
#: release height set for the old upright walls would now clip them on the way in.
RELEASE_Z = 0.072


@dataclass
class ScriptedPickPlace:
    """Waypoint state machine. Emits one Cartesian delta per call.

    Deltas are capped per step rather than being a fraction of the remaining distance.
    A proportional controller's step size decays as it arrives, so it spends most of an
    episode barely moving and the transitions are soft — which reads as smooth but
    produces a dataset where nearly every action is near zero. A capped constant speed
    keeps the action distribution wide enough to learn from.
    """

    max_step: float = 0.012
    grip_hold: int = 14          # steps to wait for the fingers to actually move
    settle_hold: int = 8
    noise: float = 0.0
    tolerance: float = 0.008
    #: Hard cap on how long one stage may last before the machine moves on.
    #:
    #: The nominal demonstrator never reaches it — its p90 stage is 53 steps — so the
    #: shipped behaviour is unchanged. It exists for the off-centre plans an affordance
    #: proposer generates: a `descend` target 14 mm off the cube's centre sits inside
    #: the cube's own footprint, the gripper cannot physically reach it, so the tip
    #: never comes within `tolerance` and the stage never ends. Without a cap that is a
    #: deadlock, and it is what made the first phase-aligned commitment run produce one
    #: decision per episode (PREREGISTERED.md, P6).
    stage_timeout: int = 120
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    #: Where on the object to grasp, where in the bowl to release, and how high to
    #: carry — the three things about this task a competent operator could reasonably
    #: do differently. They exist so a proposer can generate candidates that differ
    #: **task-meaningfully** instead of by additive noise.
    #:
    #: The distinction is measured, not stylistic. Perturbing the chunk produces eight
    #: candidates with identical outcomes (Oracle@k headroom +0.0), and perturbing it
    #: harder produces candidates that differ only by being worse (reachability
    #: 96.7% -> 70.0%). Varying these instead keeps every candidate a valid plan.
    grasp_offset: tuple[float, float] = (0.0, 0.0)
    release_offset: tuple[float, float] = (0.0, 0.0)
    transit_z: float = TRANSIT_Z

    STAGES = ("hover", "descend", "close", "lift", "transfer",
              "lower", "release", "retreat", "done")

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stage = 0
        self.hold = 0
        self.grip = 0.0
        self.in_stage = 0

    @property
    def stage_name(self) -> str:
        return self.STAGES[self.stage]

    def _advance(self) -> None:
        self.stage = min(self.stage + 1, len(self.STAGES) - 1)
        self.hold = 0
        self.in_stage = 0

    def act(self, tip: np.ndarray, objects: dict[str, np.ndarray]) -> np.ndarray:
        """One action ``[dx, dy, dz, grip]`` from the tip and perceived objects.

        Returns a zero move with the grip held whenever the machine is waiting, so a
        recorded episode has a real action at every frame. Skipping frames during a
        wait would leave the dataset with gaps that a dynamics model reads as the arm
        teleporting.
        """
        if CUBE not in objects or BOWL not in objects:
            # Perception lost the object. Holding still is the correct response and is
            # also the honest one — inventing a target from the last known position
            # would hide a perception failure inside the policy.
            return np.array([0.0, 0.0, 0.0, self.grip])

        self.in_stage += 1
        if self.in_stage > self.stage_timeout and self.stage_name != "done":
            # Give up on a waypoint that cannot be reached rather than waiting forever.
            self._advance()
        cube = np.asarray(objects[CUBE]) + np.array([*self.grasp_offset, 0.0])
        bowl = np.asarray(objects[BOWL]) + np.array([*self.release_offset, 0.0])
        tz = self.transit_z
        name = self.stage_name
        if name == "hover":
            goal, grip = np.array([cube[0], cube[1], tz]), 0.0
        elif name == "descend":
            goal, grip = np.array([cube[0], cube[1], cube[2] + GRASP_DZ]), 0.0
        elif name == "close":
            goal, grip = tip.copy(), 1.0
        elif name == "lift":
            goal, grip = np.array([tip[0], tip[1], tz]), 1.0
        elif name == "transfer":
            goal, grip = np.array([bowl[0], bowl[1], tz]), 1.0
        elif name == "lower":
            goal, grip = np.array([bowl[0], bowl[1], RELEASE_Z]), 1.0
        elif name == "release":
            goal, grip = tip.copy(), 0.0
        elif name == "retreat":
            goal, grip = np.array([bowl[0], bowl[1], tz]), 0.0
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])

        self.grip = grip
        if name in ("close", "release"):
            self.hold += 1
            if self.hold >= self.grip_hold:
                self._advance()
            return np.array([0.0, 0.0, 0.0, grip])

        delta = goal - tip
        dist = float(np.linalg.norm(delta))
        if dist < self.tolerance:
            self.hold += 1
            if self.hold >= self.settle_hold:
                self._advance()
        else:
            self.hold = 0
        if dist > 1e-9:
            delta = delta * min(1.0, self.max_step / dist)
        if self.noise:
            delta = delta + self.rng.normal(0, self.noise, 3)
        return np.concatenate([delta, [grip]])


    @classmethod
    def retimed(cls, factor: float, **kw) -> "ScriptedPickPlace":
        """A demonstrator whose every stage lasts `factor` times as many steps.

        Three quantities set how long a stage takes and they have to move together.
        `max_step` sets how many steps a traverse needs; `grip_hold` and `settle_hold`
        are dwell counters that do not scale with speed at all. Changing speed alone
        moves the median stage duration from 28 steps only to 34 (at half speed) or 25
        (at double) — the fixed 14-step grip dwells dominate — which is far too weak a
        lever to test a prediction with.

        Scaling all three together multiplies every stage duration by `factor`, which
        is the same coupling the arena's timestep has and the reason the Traps section
        of sim/README.md tells you to scale the dwells alongside it.
        """
        return cls(max_step=cls.max_step / factor,
                   grip_hold=max(1, round(cls.grip_hold * factor)),
                   settle_hold=max(1, round(cls.settle_hold * factor)),
                   stage_timeout=max(1, round(cls.stage_timeout * factor)), **kw)

    def variant(self, **plan) -> "ScriptedPickPlace":
        """A copy aimed at a different grasp/release/height, at the same stage.

        The stage, dwell counter, grip and RNG are carried across so the variant
        continues the episode rather than restarting it — a candidate that silently
        rewinds the state machine to "hover" would be scored as a fresh approach
        instead of as an alternative continuation of the one in progress.
        """
        import copy

        out = copy.copy(self)
        for k, v in plan.items():
            setattr(out, k, v)
        return out


class ChunkPolicy:
    """Behaviour cloning to an action chunk: H actions from one observation.

    Chunking rather than single-step prediction for the reason it is standard in
    manipulation — a single-step BC policy has to re-decide the whole plan every frame,
    and small errors in that decision compound into jitter at exactly the moments
    (grasp, release) where the action distribution is multi-modal. Predicting H steps
    commits to a short plan and removes the per-frame re-decision.

    It is also what makes the world model useful. A chunk *is* the candidate the planner
    scores: one rollout per chunk, not one per frame.
    """

    def __init__(self, obs_dim: int, horizon: int = 8, hidden: int = 256,
                 seed: int = 0) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon * 4))
        # Zero the last layer so an untrained policy outputs "do nothing" rather than
        # random flailing. An untrained policy that thrashes is indistinguishable from a
        # broken env during the first debugging pass.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.obs_mean = np.zeros(obs_dim, np.float32)
        self.obs_std = np.ones(obs_dim, np.float32)

    def chunk(self, obs: np.ndarray) -> np.ndarray:
        """(H, 4) action chunk."""
        import torch

        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        with torch.no_grad():
            y = self.net(torch.from_numpy(x).float().unsqueeze(0))
        out = y.reshape(self.horizon, 4).numpy()
        # The grip channel is a command in [0, 1], not a delta; the network predicts it
        # unbounded and it is squashed here so a candidate perturbation cannot ask for
        # a grip of 3.
        out[:, 3] = np.clip(out[:, 3], 0.0, 1.0)
        return out


def policy_observation(state: np.ndarray, tip: np.ndarray,
                       objects: dict[str, np.ndarray]) -> np.ndarray:
    """What a learned policy sees: proprioception plus object positions *relative to
    the tip*.

    Relative rather than absolute because the arm is mounted at the origin and the task
    is defined by offsets. Handed absolute coordinates, a small MLP trained on a few
    hundred episodes memorises the table region the cube was sampled from; handed
    offsets it has to represent "move toward the thing", which is the part that
    transfers when the layout is randomised.
    """
    cube = np.asarray(objects.get(CUBE, tip))
    bowl = np.asarray(objects.get(BOWL, tip))
    seen = np.array([float(CUBE in objects), float(BOWL in objects)])
    return np.concatenate([state, cube - tip, bowl - tip,
                           [np.linalg.norm(cube - tip), np.linalg.norm(bowl - tip)],
                           seen]).astype(np.float32)

"""z_{t+1} = f(z_t, a_t) — action-conditioned latent dynamics over object slots.

Three design decisions carry this model, and each is there to make a specific failure
mode hard rather than to add capacity:

**Residual with a motion gate.** The model predicts `slots + gate * delta`, where `gate`
is a learned per-slot scalar in [0, 1] whose bias starts strongly negative. At
initialisation the model therefore predicts *nothing moves*, which is both the correct
prior for manipulation — one object moves, the rest of the scene does not — and a
well-conditioned starting point: training begins from a sensible predictor instead of
from noise, and the identity baseline is a floor the model can only improve on. The gate
is also readable: it says which object the model thinks the action will disturb.

**Attention across slots, conditioned on the action.** Objects are not independent; a
push transfers. A per-slot MLP cannot express contact at all, so the slots attend to each
other and to a robot token, with the action injected into every token. Nothing here
assumes a fixed object count or ordering.

**A deep ensemble, not a variance head.** Rolling out N steps compounds error, and a
single network reports the same confidence at step 10 as at step 1. Independently
initialised members disagree where the data did not constrain them, and that disagreement
grows with horizon — which is exactly the signal the planner needs to stop trusting a
long rollout. A variance head would learn aleatoric noise and stay flat under
extrapolation, which is the wrong quantity here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pipeline.world_model.latent import SceneLatent


class _Block(nn.Module):
    """One round of: attend across slots, then update each slot with the action."""

    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, tok: torch.Tensor, act: torch.Tensor,
                pad: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(tok)
        # `pad` marks positions to IGNORE. Padded slots must not contribute to any real
        # slot's update, or the prediction depends on how many blanks the batch carried.
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        tok = tok + a
        h = self.norm2(tok)
        return tok + self.mlp(torch.cat([h, act.unsqueeze(1).expand_as(h)], dim=-1))


class LatentDynamics(nn.Module):
    """One ensemble member. See the module docstring for why it is shaped this way."""

    def __init__(self, slot_dim: int, robot_dim: int, action_dim: int, *,
                 hidden: int = 128, layers: int = 2, heads: int = 4,
                 gate_bias: float = -3.0, integrate_velocity: bool = False,
                 integrate_slot_velocity: bool | None = None) -> None:
        """`integrate_velocity` makes the untrained model a CONSTANT-VELOCITY predictor
        rather than an identity one.

        The latent is then read as [position | velocity] in equal halves, and the base
        prediction carries the velocity into the position before any learned term is
        added. This matters because of a measurement, not a preference: on
        gr1.PickNPlace the identity predictor scores 0.0555 held-out and constant
        velocity scores 0.0267 — more than twice as good. A model initialised at
        identity has to learn its way past momentum before it learns anything about the
        action, and on four demonstration episodes it never gets there. Starting at the
        stronger baseline means the learned term only has to supply the action's
        correction.
        """
        super().__init__()
        self.slot_dim, self.robot_dim, self.action_dim = slot_dim, robot_dim, action_dim
        if integrate_velocity and robot_dim % 2:
            raise ValueError(
                "integrate_velocity reads the latent as [position | velocity] and needs "
                f"an even robot dim; got {robot_dim}")
        self.integrate_velocity = integrate_velocity
        # Velocity integration is chosen PER STREAM, because which baseline is stronger
        # is a property of the signal, not of the architecture. Measured on
        # cube_to_bowl_5: for the robot, constant velocity (0.0140) beats identity
        # (0.0339); for the mask-derived object tracks the order reverses — identity
        # 0.121 against constant velocity 0.169, because centroid jitter makes the
        # velocity channel mostly noise and carrying it forward amplifies it. Applying
        # one choice to both streams started the object pathway from the worse of the
        # two predictors.
        self.integrate_slot_velocity = (integrate_velocity
                                        if integrate_slot_velocity is None
                                        else integrate_slot_velocity)
        self.slot_in = nn.Linear(slot_dim, hidden)
        self.robot_in = nn.Linear(robot_dim, hidden)
        self.act_in = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([_Block(hidden, heads) for _ in range(layers)])
        self.slot_out = nn.Linear(hidden, slot_dim)
        self.slot_gate = nn.Linear(hidden, 1)
        self.robot_out = nn.Linear(hidden, robot_dim)
        # Start from "nothing moves". sigmoid(-3) ~= 0.047, so the initial prediction is
        # within a few percent of the identity and training improves on a sane baseline
        # instead of unlearning noise.
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.constant_(self.slot_gate.bias, gate_bias)
        nn.init.zeros_(self.slot_out.weight)
        nn.init.zeros_(self.slot_out.bias)
        nn.init.zeros_(self.robot_out.weight)
        nn.init.zeros_(self.robot_out.bias)

    def forward(self, z: SceneLatent, action: torch.Tensor
                ) -> tuple[SceneLatent, torch.Tensor]:
        """Returns the next latent and the per-slot motion gate (B, M)."""
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action has {action.shape[-1]} dims, model expects {self.action_dim}")
        act = self.act_in(action)
        # The robot is token 0 so objects can attend to it: what the arm does is the
        # cause of most of what the objects do.
        tok = torch.cat([self.robot_in(z.robot).unsqueeze(1), self.slot_in(z.slots)], 1)
        pad = None
        if z.mask is not None and not z.mask.all():
            real = torch.cat([torch.ones_like(z.mask[:, :1]), z.mask], dim=1)
            pad = ~real
        for blk in self.blocks:
            tok = blk(tok, act, pad)
        robot_tok, slot_tok = tok[:, 0], tok[:, 1:]

        gate = torch.sigmoid(self.slot_gate(slot_tok))           # (B, M, 1)
        delta = self.slot_out(slot_tok) * gate
        if z.mask is not None:
            # A padded slot must come out exactly as it went in.
            delta = delta * z.mask.unsqueeze(-1)

        base_slots, base_robot = z.slots, z.robot
        if self.integrate_slot_velocity and self.slot_dim % 2 == 0:
            base_slots = self._integrate(base_slots)
        if self.integrate_velocity:
            base_robot = self._integrate(base_robot)
        return (SceneLatent(base_slots + delta, base_robot + self.robot_out(robot_tok),
                            z.mask),
                gate.squeeze(-1))

    @staticmethod
    def _integrate(x: torch.Tensor) -> torch.Tensor:
        """[pos | vel] -> [pos + vel | vel]. Velocity is left for the learned term to
        correct; carrying it forward unchanged is what makes the base predictor exactly
        constant-velocity."""
        d = x.shape[-1] // 2
        pos, vel = x[..., :d], x[..., d:]
        return torch.cat([pos + vel, vel], dim=-1)


class DynamicsEnsemble(nn.Module):
    """`n_members` independently initialised dynamics models.

    Disagreement between members is the epistemic uncertainty the planner discounts long
    rollouts by. Members share nothing: a shared trunk would make them agree by
    construction in precisely the extrapolated regions where the disagreement matters.
    """

    def __init__(self, slot_dim: int, robot_dim: int, action_dim: int, *,
                 n_members: int = 4, **kw) -> None:
        super().__init__()
        if n_members < 1:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList([
            LatentDynamics(slot_dim, robot_dim, action_dim, **kw)
            for _ in range(n_members)])
        # Break symmetry in the TRUNK only. Members must differ or they disagree about
        # nothing and the uncertainty signal is dead — but perturbing the output heads
        # would destroy the identity initialisation, which is the property that makes
        # training start from a sane predictor instead of from noise. So the heads stay
        # exactly zero and every member starts by predicting "nothing changes"; they
        # diverge as soon as training moves the heads off zero, and the disagreement
        # then reflects what the data did not constrain rather than the seed.
        heads = {"slot_out", "slot_gate", "robot_out"}
        for i, m in enumerate(self.members):
            g = torch.Generator().manual_seed(1234 + i)
            for name, p in m.named_parameters():
                if p.dim() > 1 and name.split(".")[0] not in heads:
                    with torch.no_grad():
                        p.add_(torch.randn(p.shape, generator=g) * 0.02)

    def forward(self, z: SceneLatent, action: torch.Tensor) -> SceneLatent:
        """Mean prediction across members."""
        outs = [m(z, action)[0] for m in self.members]
        return SceneLatent(torch.stack([o.slots for o in outs]).mean(0),
                           torch.stack([o.robot for o in outs]).mean(0), z.mask)

    @torch.no_grad()
    def rollout(self, z: SceneLatent, actions: torch.Tensor
                ) -> tuple[list[SceneLatent], torch.Tensor]:
        """Roll `actions` (B, N, A) forward N steps.

        Returns the predicted states (length N) and per-step epistemic uncertainty
        (B, N): the mean standard deviation across members over whatever the latent
        actually carries — slot geometry for real slots (metres) and the robot state
        (its own units, standardised in practice).

        It must cover BOTH. An earlier version measured slot geometry only, so a model
        with no object slots — which is exactly what the trained proprioceptive
        checkpoint is — reported an uncertainty of identically zero at every horizon,
        and the planner's risk term silently stopped existing. Measuring only part of
        the latent means the parts it skips are treated as known.

        Each member is rolled along its OWN trajectory. Re-centring every member on the
        ensemble mean at each step would suppress exactly the divergence being measured,
        and uncertainty would stop growing with horizon.
        """
        if actions.dim() != 3:
            raise ValueError(f"actions must be (B, N, A), got {tuple(actions.shape)}")
        states = [SceneLatent(z.slots.clone(), z.robot.clone(), z.mask)
                  for _ in self.members]
        means, spreads = [], []
        for t in range(actions.shape[1]):
            a = actions[:, t]
            states = [m(s, a)[0] for m, s in zip(self.members, states)]
            all_slots = torch.stack([s.slots for s in states])          # (E, B, M, D)
            all_robot = torch.stack([s.robot for s in states])           # (E, B, R)
            means.append(SceneLatent(all_slots.mean(0), all_robot.mean(0), z.mask))
            if len(self.members) == 1:
                spreads.append(torch.zeros(z.batch, device=z.slots.device))
            else:
                m = z.mask.float()
                n_real = m.sum(-1)                                       # (B,)
                per_slot = all_slots[..., :6].std(0).mean(-1)            # (B, M)
                slot_unc = (per_slot * m).sum(-1) / n_real.clamp(min=1)
                robot_unc = all_robot.std(0).mean(-1)                    # (B,)
                # Average the components that exist. A scene with no slots is scored on
                # the robot alone rather than on nothing.
                have_slots = (n_real > 0).float()
                spreads.append((slot_unc * have_slots + robot_unc)
                               / (have_slots + 1.0))
        return means, torch.stack(spreads, dim=1)

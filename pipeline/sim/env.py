"""Cube-to-bowl pick-and-place in MuJoCo, with RGB-D and ground-truth instances.

Why a simulator at all. The measured obstacle to the action-conditioned world model is
not architecture, it is data: `cube_to_bowl_5` supplies two usable training episodes,
validation is best at epoch 0, and the object pathway ties an identity predictor. This
env is the cheapest honest source of the three things the real demonstrations cannot
give:

  * **episodes without limit**, which is the binding constraint;
  * **metric depth and camera intrinsics**, so object tracks are 3-D positions in metres
    rather than image-plane centroids that conflate object motion with camera geometry;
  * **ground-truth instance masks**, which turn "does SAM+CLIP segment this correctly"
    from an opinion into a measurable IoU.

What it cannot give is sim-to-*real* evidence. There is no robot. Domain randomisation
here supports a sim-to-sim robustness claim and nothing stronger, and
`pipeline/sim/README.md` says so where a reader will see it.

Control is Cartesian: an action moves the *commanded fingertip target*, and `arm.py`
turns that into joint targets analytically. The policy therefore never sees a joint
angle, which is what lets a 4-dimensional policy trained here mean the same thing as a
4-dimensional policy trained anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from pipeline.sim.arm import (WORKSPACE, ArmGeometry, clamp_to_workspace,
                              inverse_kinematics)

ARENA = Path(__file__).with_name("arena.xml")

#: Our instance vocabulary, and the geoms that belong to each. The CLIP prompts are the
#: same strings the real `cube_to_bowl_5` extraction uses, so a mask produced here is
#: comparable with one produced there.
INSTANCE_GEOMS = {
    "a small dark cube": ("cube",),
    "an orange plastic bowl": ("bowl_base",) + tuple(f"bowl_w{i}" for i in range(12)),
    "a green rubber ball": ("ball",),
    "a robot arm": ("pedestal", "yaw_link", "upper_arm", "forearm", "palm",
                    "finger_left", "finger_right"),
}

#: Vertical clearance kept above an object while travelling. Sized to the bowl rim
#: (44 mm) plus the cube (48 mm) plus margin: a transit height that clears the cube but
#: not the rim drags the payload through the wall on the way in.
TRANSIT_Z = 0.20


@dataclass
class DomainRandomisation:
    """Per-episode scene variation, and the axis a sim-to-sim claim is measured along.

    Every field is a symmetric fraction or an absolute range. `scale` multiplies all of
    them at once, so an eval can be run at a randomisation level the policy never saw
    without re-specifying the whole struct — which is the entire point of holding these
    in one place.
    """

    cube_size: tuple[float, float] = (0.021, 0.027)
    cube_density: tuple[float, float] = (420.0, 640.0)
    friction: tuple[float, float] = (0.7, 1.4)
    light_dir: float = 0.35
    light_level: tuple[float, float] = (0.35, 0.85)
    colour_jitter: float = 0.12
    camera_pos: float = 0.04
    camera_fov: float = 4.0
    scale: float = 1.0

    def scaled(self, scale: float) -> "DomainRandomisation":
        return DomainRandomisation(**{**self.__dict__, "scale": scale})


@dataclass
class Layout:
    """Where the objects start. Sampled, then checked for reachability and overlap."""

    cube_r: tuple[float, float] = (0.24, 0.40)
    cube_yaw: tuple[float, float] = (-0.85, -0.15)
    bowl_r: tuple[float, float] = (0.26, 0.40)
    bowl_yaw: tuple[float, float] = (0.15, 0.85)
    ball_r: tuple[float, float] = (0.22, 0.34)
    ball_yaw: tuple[float, float] = (-0.30, 0.30)
    min_separation: float = 0.13


@dataclass
class Observation:
    """One step of the env, as the rest of the repo wants to read it."""

    state: np.ndarray            # (16,) proprioception — the world model's robot half
    tip: np.ndarray              # (3,) fingertip, world frame
    objects: dict[str, np.ndarray]   # label -> (3,) centre, world frame, metres
    grip: float
    step: int
    success: bool
    frames: dict[str, dict] = field(default_factory=dict)   # camera -> rgb/depth/seg


class PickPlaceEnv:
    """Cartesian-controlled pick-and-place.

    Action is ``[dx, dy, dz, grip]``: the first three are metres of movement applied to
    the commanded tip target, the fourth is 0 (open) to 1 (closed). Deltas rather than
    absolute targets because a delta policy transfers across layouts — an absolute-target
    policy has to memorise where the bowl was.
    """

    #: Physics substeps per env step. 12 x 2 ms = 24 ms, so the env runs at ~42 Hz,
    #: close enough to the 30 Hz of the real demonstrations that a policy's notion of
    #: "one step of motion" carries over.
    SUBSTEPS = 12

    def __init__(self, *, seed: int = 0, randomise: bool = True,
                 cameras: tuple[str, ...] = ("front",),
                 width: int = 320, height: int = 240,
                 max_steps: int = 300,
                 dr: DomainRandomisation | None = None,
                 layout: Layout | None = None,
                 render_every: int = 1) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(ARENA))
        self.data = mujoco.MjData(self.model)
        self.geom = ArmGeometry()
        self.rng = np.random.default_rng(seed)
        self.randomise = randomise
        self.dr = dr or DomainRandomisation()
        self.layout = layout or Layout()
        self.max_steps = max_steps
        self.cameras = tuple(cameras)
        self.width, self.height = width, height
        self.render_every = max(1, int(render_every))

        self._renderers: dict[str, mujoco.Renderer] = {}
        self._pristine = {
            "geom_size": self.model.geom_size.copy(),
            "geom_rgba": self.model.geom_rgba.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "body_mass": self.model.body_mass.copy(),
            "light_dir": self.model.light_dir.copy(),
            "light_diffuse": self.model.light_diffuse.copy(),
            "cam_pos": self.model.cam_pos.copy(),
            "cam_fovy": self.model.cam_fovy.copy(),
        }

        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.gid = gid
        self.bid = bid
        self._geom_to_label = {gid(g): lab
                               for lab, geoms in INSTANCE_GEOMS.items() for g in geoms}
        self._arm_ctrl = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                          for n in ("base_yaw", "shoulder", "elbow", "wrist")]
        self._finger_ctrl = [mujoco.mj_name2id(self.model,
                                               mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in ("finger_left", "finger_right")]
        self._arm_qpos = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("base_yaw", "shoulder", "elbow", "wrist",
                      "finger_left", "finger_right")]
        self._arm_qvel = [self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("base_yaw", "shoulder", "elbow", "wrist",
                      "finger_left", "finger_right")]
        self._free_qpos = {
            n: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("cube", "ball")}

        self.target = np.array([0.30, 0.0, TRANSIT_Z])
        self.grip = 0.0
        self.step_count = 0
        self.reset()

    # ── setup ──────────────────────────────────────────────────────────────────

    def _sample_layout(self) -> dict[str, np.ndarray]:
        """Object start positions, rejection-sampled until they are legal.

        Rejection rather than a fixed grid because the constraints interact: a cube and
        a bowl can each be individually reachable and still be 4 cm apart, at which
        point the approach to one knocks over the other and the episode is unusable
        through no fault of the policy.
        """
        L = self.layout
        for _ in range(200):
            pos = {}
            for name, (rr, yy) in (("cube", (L.cube_r, L.cube_yaw)),
                                   ("bowl", (L.bowl_r, L.bowl_yaw)),
                                   ("ball", (L.ball_r, L.ball_yaw))):
                r, yaw = self.rng.uniform(*rr), self.rng.uniform(*yy)
                pos[name] = np.array([r * np.cos(yaw), r * np.sin(yaw), 0.0])
            pairs = [("cube", "bowl"), ("cube", "ball"), ("bowl", "ball")]
            if any(np.linalg.norm(pos[a][:2] - pos[b][:2]) < L.min_separation
                   for a, b in pairs):
                continue
            # The tip must be able to reach both the cube and a point above the bowl,
            # or the episode is unsolvable and would be scored as a policy failure.
            ok = True
            for key, z in (("cube", 0.020), ("bowl", 0.10)):
                probe = np.array([pos[key][0], pos[key][1], z])
                if np.linalg.norm(clamp_to_workspace(probe) - probe) > 1e-6:
                    ok = False
            if ok:
                return pos
        raise RuntimeError("layout sampler could not place three objects legally")

    def _apply_randomisation(self) -> None:
        """Perturb the model in place. Pristine values are restored first, always.

        Restoring first rather than perturbing the current values, because a
        multiplicative jitter applied to an already-jittered model random-walks: after a
        few hundred episodes the cube is either a speck or the size of the table.
        """
        m, p = self.model, self._pristine
        for k, v in p.items():
            getattr(m, k)[:] = v
        if not self.randomise:
            return
        dr, s = self.dr, self.dr.scale
        lerp = lambda lo, hi: lo + (hi - lo) * self.rng.uniform()  # noqa: E731
        mix = lambda base, val: base + (val - base) * s  # noqa: E731

        half = mix(float(p["geom_size"][self.gid("cube")][0]),
                   lerp(*dr.cube_size))
        m.geom_size[self.gid("cube")][:3] = half
        m.body_mass[self.bid("cube")] = mix(
            float(p["body_mass"][self.bid("cube")]),
            lerp(*dr.cube_density) * (2 * half) ** 3)

        f = mix(1.0, lerp(*dr.friction))
        for g in ("cube", "finger_left", "finger_right", "table"):
            m.geom_friction[self.gid(g)][0] = p["geom_friction"][self.gid(g)][0] * f

        for g in ("cube", "bowl_base", "ball", "table"):
            jitter = self.rng.uniform(-dr.colour_jitter, dr.colour_jitter, 3) * s
            m.geom_rgba[self.gid(g)][:3] = np.clip(
                p["geom_rgba"][self.gid(g)][:3] + jitter, 0.02, 1.0)
        for w in range(8):
            m.geom_rgba[self.gid(f"bowl_w{w}")][:3] = m.geom_rgba[
                self.gid("bowl_base")][:3]

        for li in range(m.nlight):
            d = p["light_dir"][li] + self.rng.normal(0, dr.light_dir, 3) * s
            m.light_dir[li] = d / max(1e-6, np.linalg.norm(d))
            # Scale the pristine triple rather than replacing it with a scalar. The
            # scalar form collapsed a coloured light to grey, so `scale=0` was not the
            # no-op it is documented to be — the fill light's (0.25, 0.25, 0.27) came
            # back as three copies of its mean.
            base = p["light_diffuse"][li]
            level = mix(1.0, lerp(*dr.light_level) / max(1e-6, float(base.mean())))
            m.light_diffuse[li] = np.clip(base * level, 0.02, 1.5)

        for ci in range(m.ncam):
            m.cam_pos[ci] = p["cam_pos"][ci] + self.rng.normal(
                0, dr.camera_pos, 3) * s
            m.cam_fovy[ci] = p["cam_fovy"][ci] + self.rng.normal(0, dr.camera_fov) * s

    def reset(self) -> Observation:
        self._apply_randomisation()
        mujoco.mj_resetData(self.model, self.data)
        pos = self._sample_layout()

        half = float(self.model.geom_size[self.gid("cube")][0])
        for name, z in (("cube", half + 0.001),
                        ("ball", float(self.model.geom_size[self.gid("ball")][0]))):
            a = self._free_qpos[name]
            self.data.qpos[a:a + 3] = [pos[name][0], pos[name][1], z]
            self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.model.body_pos[self.bid("bowl")] = [pos["bowl"][0], pos["bowl"][1], 0.0]

        self.target = clamp_to_workspace([pos["cube"][0], pos["cube"][1], TRANSIT_Z])
        self.grip = 0.0
        self._write_ctrl(self.target, self.grip)
        # Start the arm AT its command rather than at the zero pose, so the episode does
        # not open with a large transient that the policy did not cause and the dynamics
        # model would have to learn.
        q = inverse_kinematics(self.target, self.geom)
        for adr, val in zip(self._arm_qpos[:4], q):
            self.data.qpos[adr] = val
        for adr in self._arm_qpos[4:]:
            self.data.qpos[adr] = self.geom.finger_open
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._observe()

    # ── stepping ───────────────────────────────────────────────────────────────

    def _write_ctrl(self, target, grip: float) -> None:
        q = inverse_kinematics(target, self.geom)
        for a, v in zip(self._arm_ctrl, q):
            self.data.ctrl[a] = v
        opening = self.geom.finger_open + (
            self.geom.finger_closed - self.geom.finger_open) * float(
                np.clip(grip, 0.0, 1.0))
        for a in self._finger_ctrl:
            self.data.ctrl[a] = opening

    def step(self, action) -> Observation:
        a = np.asarray(action, np.float64).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"action must be [dx, dy, dz, grip], got shape {a.shape}")
        self.target = clamp_to_workspace(self.target + a[:3])
        self.grip = float(np.clip(a[3], 0.0, 1.0))
        self._write_ctrl(self.target, self.grip)
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        # `mj_step` integrates qpos but leaves the derived frames — xpos, geom_xpos,
        # xmat — describing the configuration it started from. Without this, `state`
        # (read from qpos) is current while `tip` and `objects` (read from xpos) are one
        # step behind, so a single observation mixes two timesteps. Only 0.43 mm on the
        # tip, but it is 0.43 mm of disagreement between a dynamics model's input and
        # its target, and it costs one forward pass to remove.
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        return self._observe()

    # ── reading the world ──────────────────────────────────────────────────────

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.bid(name)].copy()

    @property
    def tip(self) -> np.ndarray:
        """Midpoint between the fingertips, world frame.

        Read off the simulator rather than from forward kinematics of the *command*: a
        position servo under load lags its target, and a policy told where it asked to
        be instead of where it is cannot close the loop on contact.
        """
        left = self.data.geom_xpos[self.gid("finger_left")]
        right = self.data.geom_xpos[self.gid("finger_right")]
        mid = (left + right) / 2
        # Geom centres sit half a pad-length behind the tips, along the gripper axis.
        axis = self.data.xmat[self.bid("wrist")].reshape(3, 3)[:, 0]
        return mid + axis * 0.027

    @property
    def objects(self) -> dict[str, np.ndarray]:
        return {"a small dark cube": self.body_pos("cube"),
                "an orange plastic bowl": self.body_pos("bowl"),
                "a green rubber ball": self.body_pos("ball")}

    def object_extent(self, label: str) -> np.ndarray:
        """World-axis-aligned size of an object, in metres.

        For a rotated box the AABB is not the geom size: a cube tipped 45 degrees is
        1.41x wider along the world axes than along its own. `|R| @ h` is the exact
        half-extent of the rotated box's AABB, and it matters because these numbers
        become the `SceneLatent` extent channels the collision term integrates.
        """
        if label == "an orange plastic bowl":
            # Measured from the geoms rather than restated as constants. The walls are
            # tilted, so neither the base radius nor a wall's own half-size is the
            # bowl's extent, and hardcoding numbers here is how the extent silently
            # stops matching the shape the next time the profile changes.
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for g in INSTANCE_GEOMS[label]:
                i = self.gid(g)
                c = self.data.geom_xpos[i]
                R = self.data.geom_xmat[i].reshape(3, 3)
                h = np.abs(R) @ self.model.geom_size[i][:3]
                lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
            return hi - lo
        if label == "a green rubber ball":
            r = float(self.model.geom_size[self.gid("ball")][0])
            return np.array([2 * r, 2 * r, 2 * r])
        if label == "a small dark cube":
            h = self.model.geom_size[self.gid("cube")][:3]
            R = self.data.geom_xmat[self.gid("cube")].reshape(3, 3)
            return 2 * (np.abs(R) @ h)
        raise KeyError(label)

    def object_boxes(self) -> dict[str, np.ndarray]:
        """label -> (6,) centre + world-axis-aligned extent, metres.

        The slot format the scene latent and the scorer both assume: channels 0:3 are
        `SLOT_CENTRE` and 3:6 are `SLOT_EXTENT`. Emitting centre alone would leave the
        extent channels holding whatever the velocity augmentation put there, and the
        collision term would silently integrate velocities as if they were box sizes.
        """
        return {k: np.concatenate([v, self.object_extent(k)])
                for k, v in self.objects.items()}

    def _observe(self) -> Observation:
        q = np.array([self.data.qpos[a] for a in self._arm_qpos])
        v = np.array([self.data.qvel[a] for a in self._arm_qvel])
        tip = self.tip
        state = np.concatenate([q, v, tip, [self.grip]]).astype(np.float32)
        frames = {}
        if self.cameras and self.step_count % self.render_every == 0:
            frames = {c: self.render(c) for c in self.cameras}
        return Observation(state=state, tip=tip, objects=self.objects,
                           grip=self.grip, step=self.step_count,
                           success=self.success(), frames=frames)

    def success(self) -> bool:
        """Cube resting inside the bowl, gripper not holding it.

        All four conditions matter. Horizontal containment alone passes a cube held in
        the air above the bowl; adding height alone passes a cube balanced on the rim;
        adding rest alone passes a cube still in the fingers, which is a grasp, not a
        placement.
        """
        cube, bowl = self.body_pos("cube"), self.body_pos("bowl")
        planar = float(np.linalg.norm(cube[:2] - bowl[:2]))
        half = float(self.model.geom_size[self.gid("cube")][0])
        adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")]
        speed = float(np.linalg.norm(self.data.qvel[adr:adr + 3]))
        return bool(planar < 0.050
                    and 0.004 < cube[2] < 0.008 + 2 * half
                    and speed < 0.03
                    and self.grip < 0.5)

    def done(self) -> bool:
        return self.step_count >= self.max_steps or self.success()

    # ── rendering ──────────────────────────────────────────────────────────────

    def _renderer(self, camera: str) -> mujoco.Renderer:
        if camera not in self._renderers:
            self._renderers[camera] = mujoco.Renderer(self.model, self.height,
                                                      self.width)
        return self._renderers[camera]

    def intrinsics(self, camera: str) -> dict[str, float]:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy = np.deg2rad(self.model.cam_fovy[cid])
        f = (self.height / 2) / np.tan(fovy / 2)
        return {"fx": float(f), "fy": float(f),
                "cx": self.width / 2, "cy": self.height / 2}

    def extrinsics(self, camera: str) -> np.ndarray:
        """4x4 camera-to-world, in the OpenCV convention (+z forward, +y down).

        MuJoCo's camera frame looks down its own -z with +y up. Every consumer in this
        repo — the fusion stage, the pose gate, Open3D — assumes OpenCV. Flipping y and
        z here means the conversion happens once, in the one place that knows both
        conventions, instead of being rediscovered at each use site.
        """
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        R = self.data.cam_xmat[cid].reshape(3, 3).copy()
        R[:, 1] *= -1
        R[:, 2] *= -1
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.data.cam_xpos[cid]
        return T

    def render(self, camera: str) -> dict:
        """RGB, metric depth and a ground-truth instance-label map."""
        r = self._renderer(camera)
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        rgb = r.render().copy()

        r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        depth = r.render().copy()
        r.disable_depth_rendering()

        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=camera)
        seg = r.render().copy()
        r.disable_segmentation_rendering()

        # Segmentation comes back as (object type, object id); collapse the geom ids
        # that belong to one real thing — nine boxes are one bowl — into one label id.
        labels = np.zeros(seg.shape[:2], np.int32)
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        order = list(INSTANCE_GEOMS)
        for i, lab in enumerate(order, start=1):
            ids = [self.gid(g) for g in INSTANCE_GEOMS[lab]]
            labels[is_geom & np.isin(seg[..., 0], ids)] = i
        return {"rgb": rgb, "depth": depth, "labels": labels,
                "label_names": {i: lab for i, lab in enumerate(order, start=1)},
                "K": self.intrinsics(camera), "T_world_cam": self.extrinsics(camera)}

    def unproject(self, depth: np.ndarray, camera: str,
                  stride: int = 1) -> np.ndarray:
        """Depth image to an (N, 3) world-frame point cloud.

        Pixels at the far plane are dropped: MuJoCo reports the clip distance there, and
        a wall of points at 6 m dominates any centroid or bounding box computed from the
        cloud.
        """
        K, T = self.intrinsics(camera), self.extrinsics(camera)
        d = depth[::stride, ::stride]
        vs, us = np.nonzero(np.isfinite(d) & (d > 1e-4) & (d < 5.0))
        z = d[vs, us]
        u = us * stride + 0.5
        v = vs * stride + 0.5
        pts = np.stack([(u - K["cx"]) * z / K["fx"],
                        (v - K["cy"]) * z / K["fy"], z], -1)
        return pts @ T[:3, :3].T + T[:3, 3]

    # ── counterfactuals ────────────────────────────────────────────────────────

    #: The physics state MuJoCo itself considers complete for resuming integration.
    #: Hand-picking fields (qpos, qvel, act, ctrl, time) looks complete and is not: it
    #: omits `qacc_warmstart`, the solver's warm start, and a contact-rich scene
    #: resumed without it takes a different first solver iteration. The divergence is
    #: far below a millimetre and compounds chaotically — it showed up as a 16-episode
    #: run scoring 15/16 where an identical one scored 16/16, with the only difference
    #: being diagnostics that were supposed to leave no trace at all.
    _STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def snapshot(self) -> tuple:
        """Everything needed to rewind the episode, physics and controller alike.

        `target`, `grip` and `step_count` are as much of the state as `qpos` is: the
        Cartesian target is integrated across steps, so a rollback that restored only
        the physics would resume from the right pose with the wrong command and drift
        away from the trajectory it was supposed to be rejoining.
        """
        buf = np.empty(mujoco.mj_stateSize(self.model, self._STATE_SPEC))
        mujoco.mj_getState(self.model, self.data, buf, self._STATE_SPEC)
        return (buf, self.target.copy(), float(self.grip), int(self.step_count))

    def restore(self, snap: tuple) -> None:
        buf, target, grip, step = snap
        mujoco.mj_setState(self.model, self.data, buf, self._STATE_SPEC)
        self.target = target.copy()
        self.grip = grip
        self.step_count = step
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def instance_centroids(frame: dict, env: PickPlaceEnv,
                       *, min_pixels: int = 40) -> dict[str, np.ndarray]:
    """World-frame centroid per instance, from the ground-truth mask plus depth.

    This is the oracle the SAM+CLIP path is scored against, and it is deliberately built
    the same way the real path is — mask, then depth, then unproject — rather than read
    off `data.xpos`. Comparing a learned mask against a perfect *pipeline* isolates the
    segmentation error; comparing it against the simulator's own state would fold in
    every projection and depth-quantisation effect too, and blame those on SAM.
    """
    K, T = frame["K"], frame["T_world_cam"]
    depth, labels = frame["depth"], frame["labels"]
    out = {}
    for i, name in frame["label_names"].items():
        m = (labels == i) & np.isfinite(depth) & (depth > 1e-4) & (depth < 5.0)
        if int(m.sum()) < min_pixels:
            continue
        vs, us = np.nonzero(m)
        z = depth[vs, us]
        pts = np.stack([(us + 0.5 - K["cx"]) * z / K["fx"],
                        (vs + 0.5 - K["cy"]) * z / K["fy"], z], -1)
        out[name] = (pts @ T[:3, :3].T + T[:3, 3]).mean(0)
    return out

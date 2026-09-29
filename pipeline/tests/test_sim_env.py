"""The simulator's contract with the rest of the stack.

Skipped without MuJoCo so the suite still runs on a machine that cannot install it.
What is checked here is not physics — MuJoCo's physics is not ours to test — but the
seams we built: that rendered depth unprojects to where the simulator says objects are,
that the instance map names the right pixels, and that success means all four of the
things it claims to mean.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pipeline.sim.env import (INSTANCE_GEOMS, DomainRandomisation,  # noqa: E402
                              PickPlaceEnv, instance_centroids)
from pipeline.sim.policy import ScriptedPickPlace  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(seed=3, randomise=False, cameras=(), width=192, height=144,
                     max_steps=320)
    yield e
    e.close()


def test_reset_is_reproducible_from_the_seed():
    a = PickPlaceEnv(seed=11, randomise=True, cameras=()).reset()
    b = PickPlaceEnv(seed=11, randomise=True, cameras=()).reset()
    assert np.allclose(a.state, b.state)
    for k in a.objects:
        assert np.allclose(a.objects[k], b.objects[k])


def test_action_shape_is_validated_rather_than_broadcast(env):
    env.reset()
    with pytest.raises(ValueError, match="dx, dy, dz, grip"):
        env.step([0.0, 0.0, 0.0])


def test_commanded_target_never_leaves_the_workspace(env):
    """A policy pushing at the wall must not be able to drive IK out of its domain."""
    env.reset()
    for _ in range(80):
        env.step([0.05, 0.05, 0.05, 0.0])     # straight at the far corner, forever
    for _ in range(80):
        env.step([-0.05, -0.05, -0.05, 0.0])


def test_depth_unprojects_to_where_the_objects_are(env):
    """The camera convention is right, not merely self-consistent.

    Tolerance is 2 cm and the residual is expected, not slack: one camera sees one side
    of an object, so the centroid of the visible surface sits toward the camera. The
    test's job is to catch a flipped axis (which lands metres away), not to demand a
    bias that monocular RGB-D cannot avoid.
    """
    obs = env.reset()
    frame = env.render("front")
    seen = instance_centroids(frame, env)
    assert {"a small dark cube", "an orange plastic bowl"} <= set(seen)
    for label, centre in seen.items():
        if label in obs.objects:
            err = float(np.linalg.norm(centre[:2] - obs.objects[label][:2]))
            assert err < 0.02, f"{label} back-projects {err*100:.1f} cm away"


def test_visible_surface_bias_points_at_the_camera(env):
    """Confirms the residual above is the bias it is claimed to be, not noise."""
    obs = env.reset()
    frame = env.render("front")
    seen = instance_centroids(frame, env)
    cam = frame["T_world_cam"][:3, 3]
    cube = obs.objects["a small dark cube"]
    toward = (cam - cube)[:2] / np.linalg.norm((cam - cube)[:2])
    offset = (seen["a small dark cube"] - cube)[:2]
    assert float(offset @ toward) > 0


def test_segmentation_covers_every_declared_instance(env):
    env.reset()
    frame = env.render("front")
    names = set(frame["label_names"].values())
    assert names == set(INSTANCE_GEOMS)
    present = {frame["label_names"][i] for i in frame["label_names"]
               if (frame["labels"] == i).sum() > 30}
    assert {"a small dark cube", "an orange plastic bowl", "a green rubber ball",
            "a robot arm"} <= present


def test_bowl_is_one_instance_not_nine_geoms(env):
    env.reset()
    frame = env.render("front")
    bowl_id = [i for i, n in frame["label_names"].items()
               if n == "an orange plastic bowl"][0]
    assert (frame["labels"] == bowl_id).sum() > 200


def test_object_boxes_carry_centre_then_extent(env):
    env.reset()
    boxes = env.object_boxes()
    cube = boxes["a small dark cube"]
    assert cube.shape == (6,)
    assert np.allclose(cube[:3], env.body_pos("cube"))
    half = float(env.model.geom_size[env.gid("cube")][0])
    # Axis-aligned at reset, so the world AABB is the cube's own size.
    assert np.allclose(cube[3:], 2 * half, atol=1e-6)


def test_rotated_cube_has_a_larger_axis_aligned_extent(env):
    env.reset()
    adr = env._free_qpos["cube"]
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)
    env.data.qpos[adr + 3:adr + 7] = [c, 0, 0, s]      # 45 deg about z
    mujoco.mj_forward(env.model, env.data)
    half = float(env.model.geom_size[env.gid("cube")][0])
    assert env.object_extent("a small dark cube")[0] > 2 * half * 1.3


def test_success_needs_all_four_conditions(env):
    """Place the cube in the bowl by hand, then break one condition at a time."""
    env.reset()
    bowl = env.body_pos("bowl")
    adr = env._free_qpos["cube"]
    half = float(env.model.geom_size[env.gid("cube")][0])

    def place(xyz, grip=0.0):
        env.data.qpos[adr:adr + 3] = xyz
        env.data.qvel[:] = 0
        env.grip = grip
        mujoco.mj_forward(env.model, env.data)

    place([bowl[0], bowl[1], 0.008 + half])
    assert env.success()
    place([bowl[0] + 0.12, bowl[1], 0.008 + half])
    assert not env.success(), "containment ignored"
    place([bowl[0], bowl[1], 0.30])
    assert not env.success(), "height ignored — a held cube counts as placed"
    place([bowl[0], bowl[1], 0.008 + half], grip=1.0)
    assert not env.success(), "grip ignored — a grasp counts as a placement"


def test_snapshot_restore_is_exact(env):
    """Oracle@k rewinds the episode once per candidate; a leaky rewind would corrupt
    every measurement it makes, silently and in the flattering direction."""
    env.reset()
    for _ in range(30):
        env.step([0.01, 0.0, -0.01, 0.0])
    snap = env.snapshot()
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.target.copy(),
              env.grip, env.step_count, env.tip.copy(),
              env.data.qacc_warmstart.copy())
    for _ in range(25):
        env.step([-0.02, 0.02, 0.02, 1.0])
    assert not np.allclose(env.data.qpos, before[0]), "the detour changed nothing"
    env.restore(snap)
    assert np.allclose(env.data.qpos, before[0])
    assert np.allclose(env.data.qvel, before[1])
    assert np.allclose(env.target, before[2])
    assert env.grip == before[3] and env.step_count == before[4]
    assert np.allclose(env.tip, before[5])
    # The one that is easy to omit and impossible to notice: without it a rewind
    # resumes with a different solver warm start and slowly diverges.
    assert np.allclose(env.data.qacc_warmstart, before[6])


def test_restored_episode_continues_identically(env):
    """Stronger than field equality: the FUTURE must match, not just the state dump."""
    env.reset()
    for _ in range(20):
        env.step([0.01, 0.0, -0.005, 0.0])
    snap = env.snapshot()
    # A long plan through a grasp, so the comparison runs through contact — which is
    # where an incompletely restored state actually diverges. A short contact-free
    # replay matches even with a broken rewind.
    plan = ([[0.004, 0.002, -0.004, 0.0]] * 20 + [[0.0, 0.0, 0.0, 1.0]] * 20
            + [[0.0, 0.0, 0.006, 1.0]] * 20)
    straight = [env.step(a).tip.copy() for a in plan]
    env.restore(snap)
    for _ in range(12):
        env.step([-0.03, 0.02, 0.03, 1.0])    # a detour through contact, then rewind
    env.restore(snap)
    again = [env.step(a).tip.copy() for a in plan]
    assert np.allclose(straight, again, atol=1e-12), (
        f"max divergence {np.abs(np.array(straight) - np.array(again)).max():.3e} m")


def test_scripted_policy_solves_the_canonical_task(env):
    obs = env.reset()
    pol = ScriptedPickPlace()
    while not env.done():
        obs = env.step(pol.act(obs.tip, obs.objects))
    assert obs.success, f"stalled at stage {pol.stage_name}"


def test_domain_randomisation_does_not_random_walk():
    """Each reset perturbs the pristine model, never the previously perturbed one."""
    e = PickPlaceEnv(seed=5, randomise=True, cameras=(),
                     dr=DomainRandomisation(scale=1.0))
    sizes = []
    for _ in range(40):
        e.reset()
        sizes.append(float(e.model.geom_size[e.gid("cube")][0]))
    e.close()
    lo, hi = DomainRandomisation().cube_size
    assert min(sizes) >= lo - 1e-6 and max(sizes) <= hi + 1e-6
    assert np.std(sizes) > 1e-4, "randomisation is not actually varying anything"


def test_dr_scale_zero_leaves_the_model_pristine():
    """Scale 0 must perturb nothing, so it is a usable "no shift" eval condition.

    Only the MODEL is compared, not the layout: at scale 0 the randomiser still draws
    from the episode RNG, so the object placement stream differs from `randomise=False`
    even though every perturbation it applies is zero. That is the intended meaning —
    scale is the magnitude of the shift, not a switch for the whole sampler.
    """
    a = PickPlaceEnv(seed=2, randomise=True, cameras=(),
                     dr=DomainRandomisation().scaled(0.0))
    a.reset()
    b = PickPlaceEnv(seed=2, randomise=False, cameras=())
    b.reset()
    for field in ("geom_size", "geom_rgba", "geom_friction", "cam_fovy",
                  "light_dir", "light_diffuse"):
        assert np.allclose(getattr(a.model, field), getattr(b.model, field)), field
    a.close()
    b.close()


def test_layout_sampler_only_emits_reachable_separated_scenes():
    from pipeline.sim.arm import clamp_to_workspace
    e = PickPlaceEnv(seed=9, randomise=True, cameras=())
    for _ in range(30):
        o = e.reset()
        p = o.objects
        for a, b in (("a small dark cube", "an orange plastic bowl"),
                     ("a small dark cube", "a green rubber ball"),
                     ("an orange plastic bowl", "a green rubber ball")):
            assert np.linalg.norm(p[a][:2] - p[b][:2]) >= e.layout.min_separation - 1e-9
        probe = np.array([p["a small dark cube"][0], p["a small dark cube"][1], 0.020])
        assert np.allclose(clamp_to_workspace(probe), probe, atol=1e-9)
    e.close()

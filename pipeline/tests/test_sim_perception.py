"""Frame -> world-frame boxes. The oracle path only; SAM is measured, not unit-tested.

`SamPerception` needs a 2.4 GB ViT-H checkpoint and seconds per frame, so its quality
is reported by `pipeline/sim/reward.py --perception sam` as a number rather than
asserted here as a threshold. What is asserted here is the geometry both backends share
— the unprojection, the robust summary, and the fact that the oracle is a perfect
*segmenter* rather than a perfect observer.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pipeline.sim.env import PickPlaceEnv  # noqa: E402
from pipeline.sim.perception import (OraclePerception, mask_to_points,  # noqa: E402
                                     summarise)


@pytest.fixture(scope="module")
def frame_and_env():
    e = PickPlaceEnv(seed=4, randomise=False, cameras=(), width=256, height=192)
    obs = e.reset()
    yield e.render("front"), e, obs
    e.close()


def test_oracle_finds_the_task_objects(frame_and_env):
    frame, _, _ = frame_and_env
    seen = OraclePerception()(frame)
    assert {"a small dark cube", "an orange plastic bowl"} <= set(seen)
    assert "a robot arm" not in seen, "the arm is not an object to plan about"


def test_oracle_centres_are_close_to_the_truth(frame_and_env):
    frame, _, obs = frame_and_env
    for label, p in OraclePerception()(frame).items():
        err = float(np.linalg.norm(p.centre[:2] - obs.objects[label][:2]))
        assert err < 0.025, f"{label} off by {err*100:.1f} cm"


def test_oracle_is_a_perfect_segmenter_not_a_perfect_observer(frame_and_env):
    """It must go through depth, so it inherits the single-view bias SAM also has.

    If the oracle read simulator state instead, the oracle-versus-SAM difference would
    bundle in every projection effect and charge it to SAM.
    """
    frame, _, obs = frame_and_env
    cube = OraclePerception()(frame)["a small dark cube"]
    err = float(np.linalg.norm(cube.centre - obs.objects["a small dark cube"]))
    assert err > 1e-4, "oracle centre is exact — it is reading state, not depth"


def test_extent_is_a_lower_bound_along_the_viewing_axis(frame_and_env):
    """One camera sees one side, so depth extent is under-measured. Documented, not fixed."""
    frame, env, _ = frame_and_env
    seen = OraclePerception()(frame)["a small dark cube"]
    true = env.object_extent("a small dark cube")
    assert (seen.extent <= true + 0.012).all()


def test_as_slot_is_centre_then_extent(frame_and_env):
    frame, _, _ = frame_and_env
    p = OraclePerception()(frame)["an orange plastic bowl"]
    s = p.as_slot()
    assert s.shape == (6,)
    assert np.allclose(s[:3], p.centre) and np.allclose(s[3:], p.extent)


def test_min_pixels_suppresses_slivers(frame_and_env):
    frame, _, _ = frame_and_env
    assert OraclePerception(min_pixels=10**6)(frame) == {}


def test_summarise_ignores_a_silhouette_outlier():
    """A grazing ray lands on the table behind the object; min/max would follow it."""
    pts = np.random.default_rng(0).normal(0, 0.01, (400, 3))
    c0, e0 = summarise(pts)
    pts[0] = [4.0, 4.0, 4.0]
    c1, e1 = summarise(pts)
    assert np.linalg.norm(c1 - c0) < 0.01
    assert np.abs(e1 - e0).max() < 0.01
    assert np.abs(pts.max(0) - pts.min(0)).max() > 3.0, "the outlier is really there"


def test_summarise_survives_a_degenerate_cloud():
    c, e = summarise(np.zeros((2, 3)))
    assert np.isfinite(c).all() and np.isfinite(e).all()
    c, e = summarise(np.zeros((0, 3)))
    assert np.isfinite(c).all() and np.isfinite(e).all()


def test_extent_is_never_zero_so_volumes_stay_defined():
    """A zero extent makes every overlap volume zero and the collision term silent."""
    _, e = summarise(np.tile([0.3, 0.2, 0.1], (50, 1)))
    assert (e > 0).all()


def test_mask_to_points_drops_the_far_plane(frame_and_env):
    frame, _, _ = frame_and_env
    everything = np.ones(frame["depth"].shape, bool)
    pts = mask_to_points(everything, frame["depth"], frame["K"],
                         frame["T_world_cam"], max_range=0.8)
    T = frame["T_world_cam"]
    # Measured along the optical axis, which is what `max_range` filters. Euclidean
    # distance from the camera centre is larger for off-axis pixels by the field-of-
    # view factor, so asserting on that would demand a cut the function never makes.
    z = (pts - T[:3, 3]) @ T[:3, 2]
    assert len(pts) and float(z.max()) <= 0.8 + 1e-6
    assert float(np.linalg.norm(pts - T[:3, 3], axis=-1).max()) > 0.8, (
        "off-axis points should exceed the z-depth cut; otherwise this proves nothing")

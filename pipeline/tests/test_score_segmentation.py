"""The segmentation scorer: mutual-coverage F1, and the frame carry between fusions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from pipeline.tools.score_segmentation import coverage_f1  # noqa: E402


def _blob(c, n=400, r=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(c) + rng.uniform(-r, r, (n, 3))


def test_a_target_scored_against_itself_is_found():
    t = _blob([0, 0, 0])
    p, r, f1 = coverage_f1(t, t, 0.02)
    assert f1 == pytest.approx(1.0)


def test_a_huge_instance_that_swallows_the_target_has_low_precision():
    """This is the failure the current Mask3D shows: recall fine, precision near zero."""
    t = _blob([0, 0, 0])
    desk = np.concatenate([t, _blob([0.5, 0, 0], n=20000, r=0.5, seed=1)])
    p, r, f1 = coverage_f1(desk, t, 0.02)
    assert r > 0.9 and p < 0.1 and f1 < 0.5


def test_a_fragment_of_the_target_has_low_recall():
    t = _blob([0, 0, 0])
    frag = t[t[:, 0] > 0.04]
    p, r, f1 = coverage_f1(frag, t, 0.005)
    assert p > 0.9 and r < 0.3


def test_an_instance_elsewhere_scores_zero():
    assert coverage_f1(_blob([1, 1, 1]), _blob([0, 0, 0]), 0.02)[2] == 0.0


def test_empty_sets_score_zero():
    assert coverage_f1(np.zeros((0, 3)), _blob([0, 0, 0]), 0.02) == (0.0, 0.0, 0.0)

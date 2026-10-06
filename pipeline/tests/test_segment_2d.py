"""Cross-view merging for the 2-D proposal path: the nested-mask property."""

from __future__ import annotations

import numpy as np

from pipeline.tools.e2e_segment_2d import merge_proposals


def test_the_same_object_from_two_views_merges():
    a, b = np.arange(0, 100), np.arange(10, 110)
    out = merge_proposals([a, b], [0, 1], iou=0.5, min_views=2)
    assert len(out) == 1 and out[0][1] == {0, 1}


def test_a_small_object_is_not_merged_into_the_desk_that_contains_it():
    """overlap/min would call the mouse 100% inside the desk and merge it away."""
    desk = np.arange(0, 5000)
    mouse = np.arange(100, 160)
    out = merge_proposals([desk, mouse, desk + 1, mouse + 1], [0, 0, 1, 1],
                          iou=0.5, min_views=2)
    sizes = sorted(len(i) for i, _ in out)
    assert len(out) == 2 and sizes[0] < 100


def test_a_proposal_seen_once_is_dropped():
    out = merge_proposals([np.arange(50)], [3], iou=0.5, min_views=2)
    assert out == []


def test_disjoint_objects_stay_separate():
    out = merge_proposals([np.arange(50), np.arange(100, 150), np.arange(0, 50),
                           np.arange(100, 150)], [0, 0, 1, 1], iou=0.5, min_views=2)
    assert len(out) == 2

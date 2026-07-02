"""Tests for shape-aware ROI geometry (rect / circle / polygon)."""

from __future__ import annotations

import numpy as np

from abel.services.context_feature_service import ContextFeatureService
from abel.utils import roi_geometry as g


def test_rect_is_default_and_bbox_preserved():
    r = g.normalize_roi({"x": 5, "y": 6, "w": 10, "h": 20})
    assert g.roi_shape(r) == "rect"
    assert g.roi_bbox(r) == (5, 6, 10, 20)
    assert g.roi_center(r) == (10.0, 16.0)
    assert g.roi_has_area(r)


def test_missing_shape_treated_as_rect():
    assert g.roi_shape({"x": 0, "y": 0, "w": 1, "h": 1}) == "rect"
    assert g.roi_shape({}) == "rect"


def test_circle_bbox_center_and_containment():
    c = g.normalize_roi({"shape": "circle", "cx": 50, "cy": 50, "r": 10})
    assert g.roi_bbox(c) == (40, 40, 20, 20)
    assert g.roi_center(c) == (50.0, 50.0)
    xs = np.array([50.0, 50.0, 61.0])
    ys = np.array([50.0, 55.0, 50.0])
    # centre inside, 5px inside, 11px outside radius
    assert list(g.roi_contains(c, xs, ys)) == [True, True, False]


def test_polygon_bbox_center_and_containment():
    p = g.normalize_roi(
        {"shape": "polygon", "points": [[0, 0], [20, 0], [20, 20], [0, 20]]}
    )
    assert g.roi_bbox(p) == (0, 0, 20, 20)
    assert g.roi_center(p) == (10.0, 10.0)
    xs = np.array([10.0, 25.0, 10.0])
    ys = np.array([10.0, 10.0, -5.0])
    assert list(g.roi_contains(p, xs, ys)) == [True, False, False]


def test_degenerate_polygon_has_no_area():
    p = g.normalize_roi({"shape": "polygon", "points": [[0, 0], [1, 1]]})
    assert not g.roi_has_area(p)


def test_non_finite_points_excluded():
    xs = np.array([np.nan, 50.0])
    ys = np.array([50.0, np.nan])
    c = g.normalize_roi({"shape": "circle", "cx": 50, "cy": 50, "r": 10})
    assert list(g.roi_contains(c, xs, ys)) == [False, False]


def test_scale_roi_circle_and_polygon():
    c = g.scale_roi(g.normalize_roi({"shape": "circle", "cx": 50, "cy": 50, "r": 10}), 0.5)
    assert (c["cx"], c["cy"], c["r"]) == (25.0, 25.0, 5.0)
    p = g.scale_roi(
        g.normalize_roi({"shape": "polygon", "points": [[0, 0], [20, 0], [20, 20]]}), 0.5
    )
    assert g.roi_bbox(p) == (0, 0, 10, 10)


def test_roi_mask_area_matches_circle():
    c = g.normalize_roi({"shape": "circle", "cx": 50, "cy": 50, "r": 20})
    x, y, w, h = g.roi_bbox(c)
    mask = g.roi_mask(c, x, y, h, w)
    # In-shape fraction of the bounding box should approach pi/4.
    assert abs(mask.mean() - np.pi / 4) < 0.03


def test_roi_mean_masks_to_shape():
    # Frame: 1.0 inside a circle, 0.0 elsewhere.
    frame = np.zeros((100, 100), dtype=np.float32)
    ys, xs = np.mgrid[0:100, 0:100]
    frame[(xs - 50) ** 2 + (ys - 50) ** 2 <= 20 * 20] = 1.0

    circ = g.normalize_roi({"shape": "circle", "cx": 50, "cy": 50, "r": 20})
    rect = g.normalize_roi({"x": 30, "y": 30, "w": 40, "h": 40})
    # Circle ROI sees mostly the filled disc; rect bbox dilutes with corners.
    assert ContextFeatureService._roi_mean(frame, circ) > 0.9
    assert 0.7 < ContextFeatureService._roi_mean(frame, rect) < 0.85


def test_chaikin_and_rdp_shapes():
    pts = [[0, 0], [5, 1], [10, 0], [10, 10], [5, 9], [0, 10]]
    smoothed = g.chaikin_smooth(pts, iterations=2)
    assert len(smoothed) > len(pts)  # corner-cutting adds vertices
    simplified = g.rdp_simplify(pts, 2.0)
    assert 3 <= len(simplified) <= len(pts)

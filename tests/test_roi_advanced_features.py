"""Tests for the shape-aware (advanced) ROI geometry features."""

from __future__ import annotations

import numpy as np
import pytest

from abel.services.context_feature_service import ContextFeatureService
from abel.utils import roi_geometry as rg

RECT = {"shape": "rect", "x": 100, "y": 0, "w": 20, "h": 200}  # tall strip, like an EPM arm
CIRCLE = {"shape": "circle", "cx": 50.0, "cy": 50.0, "r": 10.0}
# Unit-ish square polygon (0,0)-(40,0)-(40,10)-(0,10): wide, so long axis is x.
POLY = {"shape": "polygon", "points": [[0, 0], [40, 0], [40, 10], [0, 10]]}


def arr(*vals):
    return np.asarray(vals, dtype=float)


class TestSignedDistance:
    def test_rect_inside_is_positive_distance_to_nearest_edge(self):
        # Centre of the strip: nearest edge is a side wall, 10 px away.
        d = rg.roi_signed_distance(RECT, arr(110.0), arr(100.0))
        assert d[0] == pytest.approx(10.0)

    def test_rect_outside_is_negative(self):
        d = rg.roi_signed_distance(RECT, arr(90.0), arr(100.0))
        assert d[0] == pytest.approx(-10.0)

    def test_rect_on_boundary_is_zero(self):
        d = rg.roi_signed_distance(RECT, arr(100.0), arr(100.0))
        assert d[0] == pytest.approx(0.0)

    def test_rect_outside_corner_uses_euclidean_distance(self):
        # Diagonally off the (100, 0) corner by (3, 4) -> 5.
        d = rg.roi_signed_distance(RECT, arr(97.0), arr(-4.0))
        assert d[0] == pytest.approx(-5.0)

    def test_circle_signed_distance(self):
        inside = rg.roi_signed_distance(CIRCLE, arr(50.0), arr(50.0))
        assert inside[0] == pytest.approx(10.0)  # centre -> r
        edge = rg.roi_signed_distance(CIRCLE, arr(60.0), arr(50.0))
        assert edge[0] == pytest.approx(0.0)
        outside = rg.roi_signed_distance(CIRCLE, arr(65.0), arr(50.0))
        assert outside[0] == pytest.approx(-5.0)

    def test_polygon_signed_distance(self):
        inside = rg.roi_signed_distance(POLY, arr(20.0), arr(5.0))
        assert inside[0] == pytest.approx(5.0)  # nearest edge is 5 away
        outside = rg.roi_signed_distance(POLY, arr(20.0), arr(-3.0))
        assert outside[0] == pytest.approx(-3.0)

    def test_sign_agrees_with_roi_contains(self):
        xs = np.linspace(80, 140, 50)
        ys = np.full(50, 100.0)
        signed = rg.roi_signed_distance(RECT, xs, ys)
        inside = rg.roi_contains(RECT, xs, ys)
        assert np.all((signed >= 0) == inside)

    def test_nonfinite_yields_nan(self):
        d = rg.roi_signed_distance(RECT, arr(np.nan), arr(100.0))
        assert np.isnan(d[0])


class TestCornerDistance:
    def test_rect_corner(self):
        # 3-4-5 from the (100, 0) corner, measured from inside.
        d = rg.roi_corner_distance(RECT, arr(103.0), arr(4.0))
        assert d[0] == pytest.approx(5.0)

    def test_circle_has_no_corners(self):
        d = rg.roi_corner_distance(CIRCLE, arr(50.0), arr(50.0))
        assert np.isnan(d[0])


class TestAxialLateral:
    def test_rect_axial_runs_along_the_long_axis(self):
        # RECT is 20 wide x 200 tall -> long axis is y, centre at (110, 100).
        axial, lateral = rg.roi_axial_lateral(RECT, arr(110.0, 110.0, 110.0), arr(100.0, 200.0, 0.0))
        assert axial[0] == pytest.approx(0.0)   # centre of the arm
        assert axial[1] == pytest.approx(1.0)   # far tip
        assert axial[2] == pytest.approx(-1.0)  # opposite tip
        assert np.allclose(lateral, 0.0)

    def test_lateral_runs_along_the_short_axis(self):
        _axial, lateral = rg.roi_axial_lateral(RECT, arr(120.0, 100.0), arr(100.0, 100.0))
        assert lateral[0] == pytest.approx(1.0)
        assert lateral[1] == pytest.approx(-1.0)

    def test_the_two_arm_tips_are_distinguishable_unlike_centre_distance(self):
        """The point of the whole feature: centre-distance conflates the tips."""
        tips_x, tips_y = arr(110.0, 110.0), arr(0.0, 200.0)
        cx, cy = rg.roi_center(RECT)
        centre_dist = np.hypot(tips_x - cx, tips_y - cy)
        assert centre_dist[0] == pytest.approx(centre_dist[1])  # identical -> ambiguous

        axial, _lat = rg.roi_axial_lateral(RECT, tips_x, tips_y)
        assert axial[0] != pytest.approx(axial[1])  # opposite signs -> resolved

    def test_polygon_axis_follows_the_shape_not_the_bbox(self):
        # POLY is 40 wide x 10 tall -> long axis is x.
        axial, _lat = rg.roi_axial_lateral(POLY, arr(40.0, 0.0), arr(5.0, 5.0))
        assert axial[0] == pytest.approx(1.0)
        assert axial[1] == pytest.approx(-1.0)

    def test_diagonal_polygon_gets_a_true_along_shape_axis(self):
        """A diagonally-drawn arm: PCA axis, not an axis-aligned approximation."""
        diag = {"shape": "polygon",
                "points": [[0, 0], [100, 100], [104, 96], [4, -4]]}
        # (102, 98) is the far end of the strip's *centreline* (not a vertex):
        # far out along the arm, but dead centre across it.
        axial, lateral = rg.roi_axial_lateral(diag, arr(102.0), arr(98.0))
        assert abs(axial[0]) == pytest.approx(1.0, abs=0.05)
        assert abs(lateral[0]) == pytest.approx(0.0, abs=0.05)

        # An axis-aligned bbox would have called this ROI "tall" and picked the
        # wrong axis; the PCA axis tracks the true diagonal instead.
        (_c, long_ax, _s, _hl, _hs) = rg.roi_axes(diag)
        assert abs(long_ax[0]) == pytest.approx(abs(long_ax[1]), abs=0.05)


class TestAdvancedRoiColumns:
    POINTS = {
        "nose": (arr(110.0, 90.0), arr(100.0, 100.0)),
        "body_centroid": (arr(110.0, 90.0), arr(100.0, 100.0)),
    }

    def test_emits_all_columns_per_point_and_scales_distances(self):
        cols = ContextFeatureService.advanced_roi_columns(
            RECT, 1, self.POINTS, dist_scale=2.0, n=2,
        )
        assert cols["in_roi_1_nose"].tolist() == [1.0, 0.0]
        # signed distance +10 px inside / -10 px outside, scaled by 2.0 -> mm
        assert cols["nose_to_roi_1_signed_dist"][0] == pytest.approx(20.0)
        assert cols["nose_to_roi_1_signed_dist"][1] == pytest.approx(-20.0)
        # edge distance is the unsigned magnitude
        assert cols["nose_to_roi_1_edge_dist"].tolist() == [20.0, 20.0]
        # axial/lateral are normalized -> unaffected by dist_scale
        assert cols["nose_roi_1_axial"][0] == pytest.approx(0.0)
        assert cols["nose_roi_1_lateral_abs"][1] == pytest.approx(2.0)
        for pt in ("nose", "body_centroid"):
            assert f"{pt}_to_roi_1_corner_dist" in cols

    def test_absent_roi_still_emits_a_stable_column_set(self):
        """Degenerate ROI must not drop columns — the matrix schema is fixed."""
        present = ContextFeatureService.advanced_roi_columns(
            RECT, 2, self.POINTS, dist_scale=1.0, n=2,
        )
        absent = ContextFeatureService.advanced_roi_columns(
            {"x": 0, "y": 0, "w": 0, "h": 0}, 2, self.POINTS, dist_scale=1.0, n=2,
        )
        assert set(present) == set(absent)
        # Inside flag is 0 (not NaN): "definitely not in a zone that isn't there".
        assert absent["in_roi_2_nose"].tolist() == [0.0, 0.0]
        assert np.all(np.isnan(absent["nose_to_roi_2_signed_dist"]))

    def test_column_names_are_indexed_per_roi(self):
        c1 = ContextFeatureService.advanced_roi_columns(RECT, 1, self.POINTS, 1.0, 2)
        c3 = ContextFeatureService.advanced_roi_columns(RECT, 3, self.POINTS, 1.0, 2)
        assert "in_roi_1_nose" in c1 and "in_roi_3_nose" in c3
        assert not set(c1) & set(c3)  # no collisions between ROI slots

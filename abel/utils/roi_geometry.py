"""Shape-aware ROI geometry helpers.

An ROI is a plain ``dict``.  Historically every ROI was an axis-aligned
rectangle stored as ``{"x", "y", "w", "h"}``.  ROIs may now also be circles or
freehand polygons, distinguished by an optional ``"shape"`` key:

* ``{"shape": "rect", "x", "y", "w", "h"}`` — default; a missing ``shape`` is
  treated as ``"rect"`` so every legacy config keeps working unchanged.
* ``{"shape": "circle", "cx", "cy", "r", ...}`` — centre + radius.
* ``{"shape": "polygon", "points": [[x, y], ...], ...}`` — ordered vertices.

To keep the many rectangle-only consumers working, :func:`normalize_roi`
*always* (re)computes the axis-aligned bounding box ``x/y/w/h`` from the shape
parameters and stores it alongside them.  Callers that only need a bbox (e.g.
"does this ROI have area?") keep reading ``w``/``h``; callers that need true
shape fidelity (pixel masks, point-in-shape tests) use the helpers here.

This module is the single source of truth for that math so the ROI service,
the optical-flow context extractor, and the analytics occupancy code all agree
on exactly what "inside ROI N" means.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_VALID_SHAPES = ("rect", "circle", "polygon")


def roi_shape(roi: Any) -> str:
    """Return the ROI's shape, defaulting to ``"rect"`` for legacy/unknown."""
    if not isinstance(roi, dict):
        return "rect"
    s = str(roi.get("shape", "rect") or "rect").lower()
    return s if s in _VALID_SHAPES else "rect"


def _polygon_points(roi: Any) -> list[list[float]]:
    """Return the polygon vertex list as ``[[x, y], ...]`` floats (may be empty)."""
    pts: list[list[float]] = []
    for p in (roi.get("points") or []) if isinstance(roi, dict) else []:
        try:
            pts.append([float(p[0]), float(p[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return pts


def normalize_roi(raw: Any) -> dict[str, Any]:
    """Return a canonical ROI dict with a valid ``shape`` and derived bbox.

    * ``circle`` — clamps ``r`` to ``>= 0`` and recomputes ``x/y/w/h`` as the
      inscribing bounding box.
    * ``polygon`` — keeps ``>= 3`` finite vertices and recomputes the bbox as
      the vertices' extent; a degenerate polygon collapses to an empty rect.
    * ``rect`` (or anything else) — clamps ``w``/``h`` to ``>= 0``.

    Pure Python (no NumPy) so the ROI service can call it cheaply on load/save.
    """
    src = raw if isinstance(raw, dict) else {}
    shape = roi_shape(src)

    if shape == "circle":
        cx = float(src.get("cx", 0) or 0)
        cy = float(src.get("cy", 0) or 0)
        r = max(0.0, float(src.get("r", 0) or 0))
        return {
            "shape": "circle",
            "cx": cx,
            "cy": cy,
            "r": r,
            "x": int(round(cx - r)),
            "y": int(round(cy - r)),
            "w": int(round(2 * r)),
            "h": int(round(2 * r)),
        }

    if shape == "polygon":
        pts = [p for p in _polygon_points(src) if _finite2(p)]
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            return {
                "shape": "polygon",
                "points": pts,
                "x": int(x0 // 1),
                "y": int(y0 // 1),
                "w": max(0, int(-(-(x1 - x0) // 1))),  # ceil
                "h": max(0, int(-(-(y1 - y0) // 1))),
            }
        # Degenerate polygon → empty rectangle (no area).
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    # Rectangle (default / legacy).
    return {
        "x": int(src.get("x", 0) or 0),
        "y": int(src.get("y", 0) or 0),
        "w": max(0, int(src.get("w", 0) or 0)),
        "h": max(0, int(src.get("h", 0) or 0)),
    }


def _finite2(p: list[float]) -> bool:
    return len(p) >= 2 and np.isfinite(p[0]) and np.isfinite(p[1])


def roi_bbox(roi: Any) -> tuple[int, int, int, int]:
    """Return the axis-aligned bounding box ``(x, y, w, h)`` in pixels."""
    n = normalize_roi(roi)
    return int(n["x"]), int(n["y"]), int(n["w"]), int(n["h"])


def roi_has_area(roi: Any) -> bool:
    """True when the ROI encloses at least one pixel."""
    _x, _y, w, h = roi_bbox(roi)
    return w > 0 and h > 0


def roi_center(roi: Any) -> tuple[float, float]:
    """Return the ROI's representative centre used for distance/angle features.

    Circle → its centre; polygon → vertex centroid; rect → bbox centre.
    """
    shape = roi_shape(roi)
    if shape == "circle":
        return float(roi.get("cx", 0) or 0), float(roi.get("cy", 0) or 0)
    if shape == "polygon":
        pts = _polygon_points(roi)
        if pts:
            arr = np.asarray(pts, dtype=float)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    x, y, w, h = roi_bbox(roi)
    return x + w / 2.0, y + h / 2.0


def roi_contains(roi: Any, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorized point-in-shape test.

    ``xs``/``ys`` are broadcast-compatible coordinate arrays; the result is a
    boolean array of the same shape.  Non-finite coordinates yield ``False``.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    shape = roi_shape(roi)

    if shape == "circle":
        cx = float(roi.get("cx", 0) or 0)
        cy = float(roi.get("cy", 0) or 0)
        r = max(0.0, float(roi.get("r", 0) or 0))
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        return inside & finite

    if shape == "polygon":
        pts = _polygon_points(roi)
        if len(pts) < 3:
            return np.zeros(np.broadcast(xs, ys).shape, dtype=bool)
        poly = np.asarray(pts, dtype=float)
        inside = _point_in_polygon(xs, ys, poly)
        return inside & finite

    # Rectangle (inclusive bounds, matching the legacy occupancy test).
    x, y, w, h = roi_bbox(roi)
    inside = (xs >= x) & (xs <= x + w) & (ys >= y) & (ys <= y + h)
    return inside & finite


def _point_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray-casting test, vectorized over ``xs``/``ys``.

    ``poly`` is an ``(m, 2)`` array of vertices (implicitly closed).  Points on
    an edge may fall either way; that ambiguity is immaterial at pixel scale.
    """
    xs_b, ys_b = np.broadcast_arrays(xs, ys)
    inside = np.zeros(xs_b.shape, dtype=bool)
    m = len(poly)
    j = m - 1
    for i in range(m):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Edge straddles the horizontal ray from the test point?
        straddle = (yi > ys_b) != (yj > ys_b)
        # X coordinate of the edge at the point's Y (guard divide-by-zero).
        denom = (yj - yi)
        denom = np.where(denom == 0.0, np.nan, denom)
        x_cross = (xj - xi) * (ys_b - yi) / denom + xi
        cond = straddle & (xs_b < x_cross)
        inside ^= np.where(np.isnan(x_cross), False, cond)
        j = i
    return inside


def roi_mask(roi: Any, x0: int, y0: int, height: int, width: int) -> np.ndarray:
    """Return a boolean ``(height, width)`` mask for a crop whose top-left pixel
    is ``(x0, y0)`` in ROI/frame coordinates.

    Pixel centres are tested, so the mask marks exactly the pixels whose centre
    lies inside the shape.  For a rectangle this is a full-True block (the crop
    is already the intersection with the bbox), so callers can skip masking.
    """
    if height <= 0 or width <= 0:
        return np.zeros((max(0, height), max(0, width)), dtype=bool)
    if roi_shape(roi) == "rect":
        return np.ones((height, width), dtype=bool)
    ys, xs = np.mgrid[y0:y0 + height, x0:x0 + width]
    return roi_contains(roi, xs.astype(float) + 0.5, ys.astype(float) + 0.5)


def scale_roi(roi: Any, factor: float) -> dict[str, Any]:
    """Return a copy of *roi* with every coordinate multiplied by *factor*.

    Used for spatial downsampling in the optical-flow path.  The bounding box is
    recomputed from the scaled shape parameters via :func:`normalize_roi`.
    """
    if factor == 1.0:
        return normalize_roi(roi)
    shape = roi_shape(roi)
    if shape == "circle":
        return normalize_roi({
            "shape": "circle",
            "cx": float(roi.get("cx", 0) or 0) * factor,
            "cy": float(roi.get("cy", 0) or 0) * factor,
            "r": float(roi.get("r", 0) or 0) * factor,
        })
    if shape == "polygon":
        pts = [[p[0] * factor, p[1] * factor] for p in _polygon_points(roi)]
        return normalize_roi({"shape": "polygon", "points": pts})
    x, y, w, h = roi_bbox(roi)
    return normalize_roi({
        "x": int(x * factor),
        "y": int(y * factor),
        "w": max(1, int(w * factor)),
        "h": max(1, int(h * factor)),
    })


# ── Polygon post-processing (freehand cleanup) ────────────────────────────────

def chaikin_smooth(points: list[list[float]], iterations: int = 2) -> list[list[float]]:
    """Corner-cutting smoothing of a closed polygon (Chaikin's algorithm).

    Each iteration replaces every vertex with two points at 1/4 and 3/4 along
    its outgoing edge, rounding off sharp corners.  Produces a smoother, more
    uniform outline from a jagged freehand trace.
    """
    pts = [[float(p[0]), float(p[1])] for p in points if _finite2(p)]
    if len(pts) < 3:
        return pts
    for _ in range(max(0, iterations)):
        new_pts: list[list[float]] = []
        m = len(pts)
        for i in range(m):
            p0 = pts[i]
            p1 = pts[(i + 1) % m]
            q = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]]
            r = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]
            new_pts.append(q)
            new_pts.append(r)
        pts = new_pts
    return pts


def rdp_simplify(points: list[list[float]], epsilon: float) -> list[list[float]]:
    """Ramer–Douglas–Peucker simplification of a closed polygon.

    Drops vertices that lie within *epsilon* pixels of the line joining their
    neighbours, "angularizing" a dense freehand trace into a cleaner polygon
    with fewer, more deliberate corners.  Always keeps at least 3 vertices.
    """
    pts = [[float(p[0]), float(p[1])] for p in points if _finite2(p)]
    if len(pts) <= 3 or epsilon <= 0:
        return pts
    # Simplify as an open chain between the two farthest-apart anchor vertices,
    # then recombine — a closed ring has no natural endpoints.
    arr = np.asarray(pts, dtype=float)
    # Anchor: the vertex farthest from the centroid, and the one farthest from it.
    c = arr.mean(axis=0)
    a0 = int(np.argmax(((arr - c) ** 2).sum(axis=1)))
    a1 = int(np.argmax(((arr - arr[a0]) ** 2).sum(axis=1)))
    lo, hi = sorted((a0, a1))
    chain_a = pts[lo:hi + 1]
    chain_b = pts[hi:] + pts[:lo + 1]
    simp_a = _rdp_open(chain_a, epsilon)
    simp_b = _rdp_open(chain_b, epsilon)
    # Merge, dropping the shared endpoints of the two open chains.
    merged = simp_a[:-1] + simp_b[:-1]
    if len(merged) < 3:
        return pts
    return merged


def _rdp_open(points: list[list[float]], epsilon: float) -> list[list[float]]:
    if len(points) < 3:
        return points
    start = np.asarray(points[0], dtype=float)
    end = np.asarray(points[-1], dtype=float)
    line = end - start
    line_len = float(np.hypot(*line))
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        p = np.asarray(points[i], dtype=float)
        if line_len < 1e-9:
            d = float(np.hypot(*(p - start)))
        else:
            d = abs(float(np.cross(line, p - start))) / line_len
        if d > dmax:
            dmax, index = d, i
    if dmax > epsilon:
        left = _rdp_open(points[:index + 1], epsilon)
        right = _rdp_open(points[index:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]

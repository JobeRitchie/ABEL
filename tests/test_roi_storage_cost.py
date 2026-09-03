"""Hand-drawn ROIs must not make the ROI tab crawl.

A freehand trace arrives as one sample per ~2 canvas pixels of mouse travel, so
storing it verbatim put hundreds to thousands of vertices per zone per subject
into ``config/environment_rois.yaml``.  That file is parsed several times per
subject switch, so a project drawn by hand went from a 10 KB config to megabytes
and from a 125 ms switch to tens of seconds — while rectangle projects, which is
everything drawn before, stayed fast and hid the problem.

Three defences, one test group each: decimate the trace at capture, cache the
parse so repeated loads are free, and compact the files already written.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.services.roi_service import ROIService, clear_roi_config_cache  # noqa: E402
from abel.ui.tabs.roi_definition_tab import _ROICanvas  # noqa: E402
from abel.utils import roi_geometry  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_roi_config_cache()
    yield
    clear_roi_config_cache()


def _traced_circle(n_pts, cx=320, cy=240, r=150, jitter=0.6):
    """A hand-drawn outline: a circle plus the wobble a real mouse trace has."""
    rng = np.random.default_rng(11)
    return [
        [
            cx + (r + rng.uniform(-jitter, jitter)) * math.cos(2 * math.pi * i / n_pts),
            cy + (r + rng.uniform(-jitter, jitter)) * math.sin(2 * math.pi * i / n_pts),
        ]
        for i in range(n_pts)
    ]


# ── Capture-time decimation ──────────────────────────────────────────────


def test_simplify_freehand_drops_most_of_a_dense_trace():
    pts = _traced_circle(900)
    out = roi_geometry.simplify_freehand(pts)
    assert len(out) < len(pts) / 5, "dense trace should collapse substantially"
    assert len(out) >= 3


def test_simplify_freehand_keeps_the_outline_where_it_was():
    """Decimation is a storage win, not a redraw — the shape must not move."""
    pts = _traced_circle(900)
    out = roi_geometry.simplify_freehand(pts)
    before = roi_geometry.normalize_roi({"shape": "polygon", "points": pts})
    after = roi_geometry.normalize_roi({"shape": "polygon", "points": out})
    for key in ("x", "y", "w", "h"):
        assert abs(before[key] - after[key]) <= 2, f"bbox moved on {key}"

    # Every dropped vertex must still lie inside the simplified outline (within
    # the tolerance), which is the property that makes this invisible to the user.
    kept = np.asarray(out, dtype=float)
    dists = roi_geometry.roi_signed_distance(
        {"shape": "polygon", "points": out},
        np.asarray([p[0] for p in pts]),
        np.asarray([p[1] for p in pts]),
    )
    assert np.max(np.abs(dists)) <= roi_geometry.FREEHAND_SIMPLIFY_EPS_PX + 1.5
    assert len(kept) >= 3


def test_simplify_freehand_leaves_small_polygons_alone():
    tri = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]]
    assert roi_geometry.simplify_freehand(tri) == tri


def test_freehand_capture_stores_a_decimated_polygon(qapp):
    """The canvas must decimate before the trace ever reaches the config."""
    canvas = _ROICanvas()
    canvas.resize(640, 480)
    canvas.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    canvas.set_shape_mode("polygon")
    canvas.set_draw_mode("roi_0")

    captured = {}
    canvas.roi_n_changed.connect(lambda i, roi: captured.update(roi))

    def _ev(kind, x, y, button, buttons):
        return QMouseEvent(kind, QPointF(x, y), button, buttons,
                           Qt.KeyboardModifier.NoModifier)

    pts = _traced_circle(600, cx=320, cy=240, r=150)
    canvas.mousePressEvent(_ev(QMouseEvent.Type.MouseButtonPress, *pts[0],
                               Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton))
    for x, y in pts[1:]:
        canvas.mouseMoveEvent(_ev(QMouseEvent.Type.MouseMove, x, y,
                                  Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(_ev(QMouseEvent.Type.MouseButtonRelease, *pts[-1],
                                 Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))

    assert captured.get("shape") == "polygon"
    stored = captured["points"]
    assert 3 <= len(stored) < 120, f"stored {len(stored)} vertices for a traced circle"
    assert roi_geometry.roi_has_area(captured)


# ── Parse caching ────────────────────────────────────────────────────────


def _write_config(tmp_path, n_subjects=3, n_pts=40):
    svc = ROIService()
    zones = [{"shape": "polygon", "points": _traced_circle(n_pts)}]
    svc.save(tmp_path, {
        "roi_count": 1,
        "project_rois": {"target_zones": zones,
                         "subject_crop": {"x": 0, "y": 0, "w": 0, "h": 0}},
        "subject_rois": {
            f"M{i}::s{i}": {
                "target_zones": [{"shape": "polygon", "points": _traced_circle(n_pts)}],
                "subject_crop": {"x": 5, "y": 5, "w": 50, "h": 50},
            } for i in range(n_subjects)
        },
    })
    return svc


def test_repeat_loads_do_not_reparse(tmp_path, monkeypatch):
    svc = _write_config(tmp_path)
    clear_roi_config_cache()
    svc.load(tmp_path)

    import abel.services.roi_service as roi_mod
    calls = []
    real = roi_mod.read_yaml
    monkeypatch.setattr(roi_mod, "read_yaml",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for _ in range(5):
        svc.load(tmp_path)
    assert calls == [], "cached config was re-parsed"


def test_cache_is_shared_between_service_instances(tmp_path, monkeypatch):
    """Each tab builds its own ROIService; they must not each pay for a parse."""
    _write_config(tmp_path)
    clear_roi_config_cache()
    ROIService().load(tmp_path)

    import abel.services.roi_service as roi_mod
    calls = []
    real = roi_mod.read_yaml
    monkeypatch.setattr(roi_mod, "read_yaml",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ROIService().load(tmp_path)
    assert calls == []


def test_external_edit_invalidates_the_cache(tmp_path):
    svc = _write_config(tmp_path)
    assert svc.local_motion_radius(tmp_path) == 36

    cfg = svc.load(tmp_path)
    cfg["motion"]["local_radius_px"] = 99
    # Write behind the service's back, the way a hand-edited YAML or a second
    # process would, and bump mtime past the filesystem's resolution.
    path = tmp_path / ROIService.ROI_FILE
    from abel.storage.file_store import write_yaml
    write_yaml(path, cfg)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2_000_000))

    assert svc.local_motion_radius(tmp_path) == 99


def test_loaded_config_is_isolated_from_the_cache(tmp_path):
    """Mutating a loaded config must not poison what the next caller sees."""
    svc = _write_config(tmp_path)
    first = svc.load(tmp_path)
    first["subject_rois"]["M0::s0"]["subject_crop"]["w"] = 4242
    first["project_rois"]["target_zones"][0]["points"].append([1.0, 2.0])

    second = svc.load(tmp_path)
    assert second["subject_rois"]["M0::s0"]["subject_crop"]["w"] == 50
    assert [1.0, 2.0] not in second["project_rois"]["target_zones"][0]["points"]


def test_save_refreshes_the_cache(tmp_path):
    svc = _write_config(tmp_path)
    cfg = svc.load(tmp_path)
    cfg["subject_rois"]["M1::s1"]["subject_crop"] = {"x": 1, "y": 2, "w": 30, "h": 40}
    svc.save(tmp_path, cfg)
    assert svc.resolve_subject_crop_roi(tmp_path, "M1::s1")["w"] == 30


# ── Compaction of files already written ──────────────────────────────────


def test_compact_polygons_shrinks_an_existing_file(tmp_path):
    svc = _write_config(tmp_path, n_subjects=4, n_pts=400)
    path = tmp_path / ROIService.ROI_FILE
    before_bytes = path.stat().st_size
    before_rois = svc.resolve_target_rois(tmp_path, "M2::s2")

    stats = svc.compact_polygons(tmp_path)

    assert stats["polygons"] == 5, "project block plus one per subject"
    assert stats["points_after"] < stats["points_before"] / 5
    assert path.stat().st_size < before_bytes / 3
    assert stats["bytes_after"] == path.stat().st_size

    # Same zone, same place.
    after_rois = svc.resolve_target_rois(tmp_path, "M2::s2")
    for key in ("x", "y", "w", "h"):
        assert abs(before_rois[0][key] - after_rois[0][key]) <= 2


def test_compact_polygons_is_a_no_op_on_rectangles(tmp_path):
    svc = ROIService()
    svc.save(tmp_path, {
        "roi_count": 1,
        "project_rois": {"target_zones": [{"x": 10, "y": 10, "w": 100, "h": 80}],
                         "subject_crop": {"x": 0, "y": 0, "w": 0, "h": 0}},
        "subject_rois": {},
    })
    path = tmp_path / ROIService.ROI_FILE
    before = path.read_bytes()

    stats = svc.compact_polygons(tmp_path)

    assert stats["polygons"] == 0
    assert path.read_bytes() == before, "untouched file must not be rewritten"


def test_compact_polygons_survives_a_missing_file(tmp_path):
    stats = ROIService().compact_polygons(tmp_path / "no_such_project")
    assert stats["polygons"] == 0

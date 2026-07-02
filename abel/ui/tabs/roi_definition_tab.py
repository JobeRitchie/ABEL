"""ROI configuration tab for target location and local motion settings.

Provides a visual frame-picker canvas so users can draw the Target Zone ROI
directly on a video frame, with per-project or per-subject overrides.

The subject-override workflow uses a dedicated subject list with prev/next
navigation, auto-save on subject change, and visual status indicators so
the user can quickly step through every subject drawing ROIs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QSize, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from abel.services.import_service import ImportService
from abel.services.roi_service import ROI_COLORS, MAX_ROIS, ROIService
from abel.utils import roi_geometry

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Visual ROI canvas
# ---------------------------------------------------------------------------

# Fill alpha for drawn ROI overlays (0-255)
_ROI_FILL_ALPHA = 40

# Pre-computed QColor pairs (border, fill) for each ROI slot.
# Generated lazily at first use to avoid constructing Qt objects at import time.
_ROI_QCOLORS: list[tuple] = []

def _roi_qcolors() -> list[tuple]:
    """Return list of (border_QColor, fill_QColor) for each ROI slot."""
    if not _ROI_QCOLORS:
        for hex_color in ROI_COLORS:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            _ROI_QCOLORS.append((QColor(hex_color), QColor(r, g, b, _ROI_FILL_ALPHA)))
    return _ROI_QCOLORS


class _ROICanvas(QWidget):
    """Displays a video frame; drag left-button to draw ROI overlays.

    Supports up to MAX_ROIS named target zones and one *subject_crop* zone.
    The active drawing slot is selected via :py:meth:`set_draw_mode`.

    Signals
    -------
    roi_n_changed(index, roi_dict)
        Emitted when target zone at *index* is redrawn.  ``index`` is 0-based.
    roi_changed(roi_dict)
        Legacy alias — emitted for index 0 only.
    crop_changed(roi_dict)
        Emitted when the subject crop zone is redrawn.
    """

    roi_n_changed = Signal(int, dict)   # (roi_index, roi_shape_dict)
    roi_changed   = Signal(dict)        # legacy — index 0 only
    crop_changed  = Signal(dict)        # subject crop

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rois: list[dict] = [{"x": 0, "y": 0, "w": 0, "h": 0}]
        self._crop: dict[str, int] = {"x": 0, "y": 0, "w": 0, "h": 0}
        # draw_mode is "roi_<index>" or "subject_crop"
        self._draw_mode: str = "roi_0"
        # shape_mode is "rect", "circle", or "polygon" (applies to roi_* targets;
        # the subject crop is always a rectangle).
        self._shape_mode: str = "rect"
        self._drag_origin: QPoint | None = None
        self._drag_rect: QRect | None = None
        # In-progress freehand polygon trace (canvas-space QPoints).
        self._freehand_pts: list[QPoint] | None = None
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._img_w: int = 1
        self._img_h: int = 1
        self._pixmap: QPixmap | None = None
        # Optional user zoom (multiplier on the fit-to-viewport scale).  Only
        # active once :py:meth:`set_zoom` is given a viewport size; until then
        # the canvas keeps its default auto-fit behaviour.
        self._zoom: float = 1.0
        self._viewport: "QSize | None" = None

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #0a0e18;")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

    # ── Frame loading ─────────────────────────────────────────────────

    def set_frame(self, bgr_array) -> None:
        """Set the displayed frame from an OpenCV BGR ndarray."""
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        if bgr_array is None:
            self._pixmap = None
            self.update()
            return
        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._img_w, self._img_h = w, h
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._recalculate_transform()
        # Re-apply any active user zoom now that the image dimensions are known.
        self._apply_zoom()
        self.update()

    # ── User zoom ─────────────────────────────────────────────────────

    def set_zoom(self, zoom: float, viewport: "QSize | None" = None) -> None:
        """Enlarge the displayed frame by *zoom* × the fit-to-viewport scale.

        *viewport* is the size of the scroll-area viewport the canvas lives in;
        it must be supplied at least once to activate zooming.  ``zoom == 1.0``
        fits the frame to the viewport (no letterboxing); larger values make the
        frame bigger so a surrounding scroll area can pan around it.
        """
        self._zoom = max(1.0, float(zoom))
        if viewport is not None:
            self._viewport = viewport
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        # No viewport supplied → zoom feature unused (e.g. ROIDefinitionTab),
        # keep the original auto-fit/Expanding behaviour untouched.
        if self._viewport is None:
            return
        if self._pixmap is None or not self._img_w or not self._img_h:
            self.setMinimumSize(320, 240)
            self.setMaximumSize(16_777_215, 16_777_215)
            return
        vw = max(1, self._viewport.width())
        vh = max(1, self._viewport.height())
        fit = min(vw / self._img_w, vh / self._img_h)
        scale = fit * self._zoom
        disp_w = max(1, int(self._img_w * scale))
        disp_h = max(1, int(self._img_h * scale))
        self.setFixedSize(disp_w, disp_h)
        self._recalculate_transform()
        self.update()

    # ── ROI state setters ─────────────────────────────────────────────

    def set_rois(self, rois: list[dict]) -> None:
        """Set all target-zone overlays at once (does not emit signals)."""
        self._rois = [roi_geometry.normalize_roi(r) for r in rois]
        self.update()

    def set_n_rois(self, n: int) -> None:
        """Resize the internal ROI list to *n* slots, padding with zeros."""
        n = max(1, min(n, MAX_ROIS))
        while len(self._rois) < n:
            self._rois.append({"x": 0, "y": 0, "w": 0, "h": 0})
        self._rois = self._rois[:n]
        self.update()

    def set_roi(self, roi: dict) -> None:
        """Backward-compat: update ROI slot 0 without emitting."""
        if not self._rois:
            self._rois = [{"x": 0, "y": 0, "w": 0, "h": 0}]
        self._rois[0] = roi_geometry.normalize_roi(roi)
        self.update()

    def set_roi_at(self, index: int, roi: dict) -> None:
        """Update a specific target-zone slot without emitting."""
        while len(self._rois) <= index:
            self._rois.append({"x": 0, "y": 0, "w": 0, "h": 0})
        self._rois[index] = roi_geometry.normalize_roi(roi)
        self.update()

    def set_shape_mode(self, mode: str) -> None:
        """Set the active shape for target-zone drawing (rect/circle/polygon)."""
        self._shape_mode = mode if mode in ("rect", "circle", "polygon") else "rect"
        # Abandon any in-progress freehand trace when switching shapes.
        self._freehand_pts = None
        self.update()

    def set_crop(self, roi: dict[str, int]) -> None:
        """Update the subject-crop overlay without emitting crop_changed."""
        self._crop = {
            "x": int(roi.get("x", 0) or 0),
            "y": int(roi.get("y", 0) or 0),
            "w": int(roi.get("w", 0) or 0),
            "h": int(roi.get("h", 0) or 0),
        }
        self.update()

    def set_draw_mode(self, mode: str) -> None:
        """Set the active draw mode.

        Accepted values: ``"roi_<N>"`` (0-based index) or ``"subject_crop"``.
        Legacy ``"target_zone"`` is mapped to ``"roi_0"``.
        """
        if mode == "target_zone":
            mode = "roi_0"
        self._draw_mode = mode

    # ── Layout helpers ────────────────────────────────────────────────

    def _recalculate_transform(self) -> None:
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        cw, ch = self.width(), self.height()
        if pw == 0 or ph == 0 or cw == 0 or ch == 0:
            return
        self._scale = min(cw / pw, ch / ph)
        self._offset_x = int((cw - pw * self._scale) / 2)
        self._offset_y = int((ch - ph * self._scale) / 2)

    def _canvas_to_image(self, pt: QPoint) -> tuple[int, int]:
        if self._scale == 0:
            return 0, 0
        ix = round((pt.x() - self._offset_x) / self._scale)
        iy = round((pt.y() - self._offset_y) / self._scale)
        return max(0, min(ix, self._img_w)), max(0, min(iy, self._img_h))

    def _image_roi_to_canvas_rect(self, roi: dict[str, int]) -> QRect:
        x = int(roi["x"] * self._scale + self._offset_x)
        y = int(roi["y"] * self._scale + self._offset_y)
        w = max(1, int(roi["w"] * self._scale))
        h = max(1, int(roi["h"] * self._scale))
        return QRect(x, y, w, h)

    def _image_pt_to_canvas(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._scale + self._offset_x, y * self._scale + self._offset_y)

    def _paint_roi_shape(
        self, painter: QPainter, roi: dict, border_c: QColor, fill_c: QColor
    ) -> QPoint:
        """Draw *roi* (rect/circle/polygon) and return the label anchor point."""
        shape = roi_geometry.roi_shape(roi)
        pen = QPen(border_c, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        if shape == "circle":
            c = self._image_pt_to_canvas(
                float(roi.get("cx", 0) or 0), float(roi.get("cy", 0) or 0)
            )
            rr = float(roi.get("r", 0) or 0) * self._scale
            path = QPainterPath()
            path.addEllipse(c, rr, rr)
            painter.fillPath(path, fill_c)
            painter.drawEllipse(c, rr, rr)
        elif shape == "polygon":
            poly = QPolygonF([
                self._image_pt_to_canvas(px, py)
                for px, py in roi.get("points", [])
            ])
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            painter.fillPath(path, fill_c)
            painter.drawPolygon(poly)
        else:
            crect = self._image_roi_to_canvas_rect(roi)
            painter.drawRect(crect)
            painter.fillRect(crect, fill_c)
        bx = self._image_roi_to_canvas_rect(roi)
        return bx.topLeft()

    # ── Qt events ─────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        self._recalculate_transform()
        super().resizeEvent(event)

    def _active_shape(self) -> str:
        """Shape to draw for the current target (crop is always a rectangle)."""
        return "rect" if self._draw_mode == "subject_crop" else self._shape_mode

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = event.position().toPoint()
        if self._active_shape() == "polygon":
            self._freehand_pts = [pt]
        else:
            self._drag_origin = pt
            self._drag_rect = QRect(pt, pt)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pt = event.position().toPoint()
        if self._freehand_pts is not None:
            # Sample the trace, skipping near-duplicate points to bound size.
            last = self._freehand_pts[-1]
            if abs(pt.x() - last.x()) + abs(pt.y() - last.y()) >= 2:
                self._freehand_pts.append(pt)
            self.update()
        elif self._drag_origin is not None:
            self._drag_rect = QRect(self._drag_origin, pt).normalized()
            self.update()

    def _emit_roi(self, roi: dict) -> None:
        """Store *roi* into the active target and emit the matching signal."""
        if self._draw_mode == "subject_crop":
            self._crop = roi
            self.crop_changed.emit(dict(roi))
            return
        idx = 0
        if self._draw_mode.startswith("roi_"):
            try:
                idx = int(self._draw_mode[4:])
            except ValueError:
                idx = 0
        while len(self._rois) <= idx:
            self._rois.append({"x": 0, "y": 0, "w": 0, "h": 0})
        self._rois[idx] = roi
        self.roi_n_changed.emit(idx, dict(roi))
        if idx == 0:
            self.roi_changed.emit(dict(roi))

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # ── Freehand polygon ──────────────────────────────────────────────
        if self._freehand_pts is not None:
            pts_canvas = self._freehand_pts
            self._freehand_pts = None
            if len(pts_canvas) >= 3:
                img_pts = [list(self._canvas_to_image(p)) for p in pts_canvas]
                roi = roi_geometry.normalize_roi({"shape": "polygon", "points": img_pts})
                if roi_geometry.roi_has_area(roi):
                    self._emit_roi(roi)
            self.update()
            return

        # ── Rectangle / circle (drag defines a bounding box) ──────────────
        if self._drag_origin is None:
            return
        rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
        ix, iy = self._canvas_to_image(rect.topLeft())
        ix2, iy2 = self._canvas_to_image(rect.bottomRight())
        w = max(1, ix2 - ix)
        h = max(1, iy2 - iy)
        self._drag_origin = None
        self._drag_rect = None
        if self._active_shape() == "circle":
            r = min(w, h) / 2.0
            roi = roi_geometry.normalize_roi(
                {"shape": "circle", "cx": ix + w / 2.0, "cy": iy + h / 2.0, "r": r}
            )
        else:
            roi = {"x": ix, "y": iy, "w": w, "h": h}
        self._emit_roi(roi)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a0e18"))

        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(self._offset_x, self._offset_y, scaled)

        # Target Zone ROI overlays (one per slot, each with its own colour)
        colors = _roi_qcolors()
        for idx, roi in enumerate(self._rois):
            if roi.get("w", 0) > 0 and roi.get("h", 0) > 0:
                border_c, fill_c = colors[idx % len(colors)]
                anchor = self._paint_roi_shape(painter, roi, border_c, fill_c)
                painter.setPen(border_c)
                label = f"ROI {idx + 1}" if idx > 0 else "ROI 1 (Target Zone)"
                label_pt = anchor + QPoint(4, -6)
                if label_pt.y() < 12:
                    label_pt = anchor + QPoint(4, 14)
                painter.drawText(label_pt, label)

        # Subject Crop overlay (green)
        if self._crop.get("w", 0) > 0 and self._crop.get("h", 0) > 0:
            crect = self._image_roi_to_canvas_rect(self._crop)
            pen = QPen(QColor("#66BB6A"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(crect)
            painter.fillRect(crect, QColor(102, 187, 106, 35))
            painter.setPen(QColor("#66BB6A"))
            label_pt = crect.topLeft() + QPoint(4, -6)
            if label_pt.y() < 12:
                label_pt = crect.topLeft() + QPoint(4, 14)
            painter.drawText(label_pt, "Subject Crop")

        # Active preview (colour matches current draw mode) — rectangle/circle
        # rubber-band or the live freehand-polygon trace.
        if self._drag_rect is not None or self._freehand_pts is not None:
            if self._draw_mode == "subject_crop":
                band_color = QColor("#66BB6A")
                fill_color = QColor(102, 187, 106, 30)
            else:
                idx = 0
                if self._draw_mode.startswith("roi_"):
                    try:
                        idx = int(self._draw_mode[4:])
                    except ValueError:
                        pass
                c = _roi_qcolors()[idx % len(_roi_qcolors())]
                band_color = c[0]
                r, g, b = band_color.red(), band_color.green(), band_color.blue()
                fill_color = QColor(r, g, b, 30)
            pen = QPen(band_color, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            if self._freehand_pts is not None and len(self._freehand_pts) >= 2:
                painter.drawPolyline(QPolygonF([QPointF(p) for p in self._freehand_pts]))
            elif self._drag_rect is not None and self._active_shape() == "circle":
                cx = self._drag_rect.center()
                rr = min(self._drag_rect.width(), self._drag_rect.height()) / 2.0
                painter.drawEllipse(QPointF(cx), rr, rr)
                path = QPainterPath()
                path.addEllipse(QPointF(cx), rr, rr)
                painter.fillPath(path, fill_color)
            elif self._drag_rect is not None:
                painter.drawRect(self._drag_rect)
                painter.fillRect(self._drag_rect, fill_color)

        if self._pixmap is None:
            painter.setPen(QColor("#546E7A"))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No frame loaded.\n"
                "Select a session and click 'Load Frame',\n"
                "then drag to draw an ROI or the Subject Crop zone.\n"
                "Use the draw mode selector to switch between ROI slots.",
            )

        painter.end()


# ---------------------------------------------------------------------------
# ROI Definition Tab
# ---------------------------------------------------------------------------

class ROIDefinitionTab(QWidget):
    """Define project/subject ROIs used by context and fusion features.

    Left panel: visual video-frame canvas — drag to draw the Target Zone ROI.
    Right panel: subject list with quick navigation, spinboxes, and save controls.

    Subject-override mode shows a scrollable subject list with colour-coded
    status indicators (green = configured, grey = pending).  Prev/Next buttons
    and keyboard shortcuts (Alt+Up / Alt+Down) let the user rapidly step
    through subjects.  The current subject's ROI is auto-saved when the user
    navigates away, so the workflow is just: draw → next → draw → next.
    """

    def __init__(
        self,
        roi_service: ROIService,
        import_service: ImportService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._rois = roi_service
        self._imports = import_service
        self._project_root: Path | None = None
        self._video_cap = None
        self._n_video_frames: int = 0
        self._auto_save_enabled: bool = True
        self._current_subject_dirty: bool = False
        self._previous_subject_id: str = ""  # tracks which subject owns unsaved edits

        # ── Status label ──────────────────────────────────────────────
        self._status = QLabel("Open a project to configure ROIs.")
        self._status.setWordWrap(True)

        # ── Visual canvas ─────────────────────────────────────────────
        self._canvas = _ROICanvas()
        self._canvas.roi_n_changed.connect(self._on_canvas_roi_n_changed)
        self._canvas.crop_changed.connect(self._on_canvas_crop_changed)

        # Frame navigation controls
        self._session_combo = QComboBox()
        self._session_combo.setToolTip("Session whose video file will be used for frame preview")
        self._session_combo.currentIndexChanged.connect(self._on_video_session_changed)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(80)

        self._load_frame_btn = QPushButton("Load Frame")
        self._load_frame_btn.setToolTip("Open the selected session's video and display the current frame")
        self._load_frame_btn.clicked.connect(self._load_video_frame)

        frame_nav_row = QHBoxLayout()
        frame_nav_row.addWidget(QLabel("Session:"))
        frame_nav_row.addWidget(self._session_combo, 1)
        frame_nav_row.addWidget(self._load_frame_btn)

        frame_slider_row = QHBoxLayout()
        frame_slider_row.addWidget(self._frame_slider, 1)
        frame_slider_row.addWidget(self._frame_label)

        canvas_hint = QLabel(
            "Drag on the frame to draw the selected ROI type. "
            "The outline updates the spinboxes on release."
        )
        canvas_hint.setWordWrap(True)
        canvas_hint.setStyleSheet("font-size: 11px; color: #78909C; padding: 2px 0;")

        self._draw_mode_combo = QComboBox()
        # Populated dynamically by _rebuild_draw_mode_combo()
        self._draw_mode_combo.currentIndexChanged.connect(self._on_draw_mode_changed)

        draw_mode_row = QHBoxLayout()
        draw_mode_row.addWidget(QLabel("Draw mode:"))
        draw_mode_row.addWidget(self._draw_mode_combo, 1)

        # Shape selector — applies to target-zone ROIs (the subject crop is
        # always a rectangle).
        self._shape_combo = QComboBox()
        self._shape_combo.addItem("Rectangle", userData="rect")
        self._shape_combo.addItem("Circle", userData="circle")
        self._shape_combo.addItem("Freehand polygon", userData="polygon")
        self._shape_combo.setToolTip(
            "ROI shape to draw:\n"
            "• Rectangle — drag a box.\n"
            "• Circle — drag a box; the inscribed circle is used.\n"
            "• Freehand polygon — drag to trace an outline, then optionally\n"
            "  Smooth or Angularize it. Applies to target zones only."
        )
        self._shape_combo.currentIndexChanged.connect(self._on_shape_changed)

        self._smooth_btn = QPushButton("Smooth")
        self._smooth_btn.setToolTip(
            "Round off a freehand polygon's corners (Chaikin). Operates on the\n"
            "currently selected target-zone ROI if it is a polygon."
        )
        self._smooth_btn.clicked.connect(self._smooth_active_polygon)
        self._angularize_btn = QPushButton("Angularize")
        self._angularize_btn.setToolTip(
            "Simplify a freehand polygon to fewer, cleaner corners\n"
            "(Douglas–Peucker). Operates on the selected target-zone ROI."
        )
        self._angularize_btn.clicked.connect(self._angularize_active_polygon)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shape:"))
        shape_row.addWidget(self._shape_combo, 1)
        shape_row.addWidget(self._smooth_btn)
        shape_row.addWidget(self._angularize_btn)

        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._canvas, 1)
        canvas_layout.addWidget(canvas_hint)
        canvas_layout.addLayout(draw_mode_row)
        canvas_layout.addLayout(shape_row)
        canvas_layout.addLayout(frame_nav_row)
        canvas_layout.addLayout(frame_slider_row)

        # ── Scope controls ────────────────────────────────────────────
        self._scope = QComboBox()
        self._scope.addItem("Project default", userData="project")
        self._scope.addItem("Subject override", userData="subject")
        self._scope.setToolTip(
            "Project default applies to all subjects. "
            "Subject override lets you set a per-subject Target Zone."
        )
        self._scope.currentIndexChanged.connect(self._on_scope_changed)

        scope_box = QGroupBox("Scope")
        scope_form = QFormLayout(scope_box)
        scope_form.addRow("Apply to:", self._scope)

        # ── Subject list with navigation ──────────────────────────────
        self._subject_list = QListWidget()
        self._subject_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._subject_list.currentRowChanged.connect(self._on_subject_list_changed)
        self._subject_list.setMinimumHeight(100)
        self._subject_list.setMaximumHeight(220)

        self._subject_counter = QLabel("0 / 0 subjects")
        self._subject_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subject_counter.setStyleSheet("font-size: 11px; color: #B0BEC5;")

        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setToolTip("Save current subject and go to previous (Left / Up arrow)")
        self._prev_btn.clicked.connect(self._go_prev_subject)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setToolTip("Save current subject and go to next (Right / Down arrow)")
        self._next_btn.clicked.connect(self._go_next_subject)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._subject_counter, 1)
        nav_row.addWidget(self._next_btn)

        self._auto_load_video_cb = QPushButton("Auto-load subject video")
        self._auto_load_video_cb.setCheckable(True)
        self._auto_load_video_cb.setChecked(True)
        self._auto_load_video_cb.setToolTip(
            "When checked, switching subjects automatically loads that subject's "
            "video and jumps to the current slider frame."
        )

        subject_box = QGroupBox("Subjects")
        subject_layout = QVBoxLayout(subject_box)
        subject_layout.addWidget(self._subject_list)
        subject_layout.addLayout(nav_row)
        subject_layout.addWidget(self._auto_load_video_cb)
        self._subject_box = subject_box

        # ── ROI count selector ────────────────────────────────────────
        self._roi_count_spin = QSpinBox()
        self._roi_count_spin.setRange(1, MAX_ROIS)
        self._roi_count_spin.setValue(1)
        self._roi_count_spin.setToolTip(
            f"Number of target-zone ROIs to define per subject (1–{MAX_ROIS}).\n"
            "Each zone gets a distinct colour on the canvas."
        )
        self._roi_count_spin.valueChanged.connect(self._on_roi_count_changed)

        roi_count_box = QGroupBox("Target Zone ROIs")
        roi_count_form = QFormLayout(roi_count_box)
        roi_count_form.addRow("ROIs per subject:", self._roi_count_spin)
        roi_note = QLabel(
            "Target-dependent features (flow magnitude, distance, angle) are "
            "computed for each ROI with a non-zero size."
        )
        roi_note.setWordWrap(True)
        roi_note.setStyleSheet("font-size: 11px; color: #78909C;")
        roi_count_form.addRow(roi_note)

        # Container for dynamically generated per-ROI spinbox groups
        self._roi_spinbox_groups: list[tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox]] = []
        self._roi_boxes: list[QGroupBox] = []
        # Authoritative per-slot ROI shape dicts (rect/circle/polygon).  The
        # spinboxes mirror each shape's bounding box; the canvas is the source
        # of truth for non-rectangular geometry.
        self._roi_shapes: list[dict] = [{"x": 0, "y": 0, "w": 0, "h": 0}]
        self._roi_spins_container = QWidget()
        self._roi_spins_layout = QVBoxLayout(self._roi_spins_container)
        self._roi_spins_layout.setContentsMargins(0, 0, 0, 0)
        self._roi_spins_layout.setSpacing(4)
        # Built on first call to _rebuild_roi_spinboxes below

        clear_rois_btn = QPushButton("Clear All Target Zones")
        clear_rois_btn.clicked.connect(self._clear_target_roi)
        roi_count_form.addRow(clear_rois_btn)

        # ── Subject Crop spinboxes ────────────────────────────────────
        self._crop_x = QSpinBox()
        self._crop_x.setRange(0, 10000)
        self._crop_y = QSpinBox()
        self._crop_y.setRange(0, 10000)
        self._crop_w = QSpinBox()
        self._crop_w.setRange(0, 10000)
        self._crop_h = QSpinBox()
        self._crop_h.setRange(0, 10000)

        crop_box = QGroupBox("Subject Crop ROI (for fusion)")
        crop_form = QFormLayout(crop_box)
        crop_form.addRow("x:", self._crop_x)
        crop_form.addRow("y:", self._crop_y)
        crop_form.addRow("width:", self._crop_w)
        crop_form.addRow("height:", self._crop_h)

        # ── Local Motion settings ─────────────────────────────────────
        self._local_radius = QSpinBox()
        self._local_radius.setRange(8, 2048)
        self._local_radius.setSingleStep(4)
        self._local_radius.setToolTip(
            "Pixel radius around each tracked body part used for local optical-flow "
            "and substrate-motion calculations.  Larger values capture a wider "
            "neighbourhood around the animal."
        )

        motion_box = QGroupBox("Local Motion Settings")
        motion_form = QFormLayout(motion_box)
        motion_form.addRow("Local radius (px):", self._local_radius)

        # ── Day Exclusions ────────────────────────────────────────────
        self._day_exclusions_list = QListWidget()
        self._day_exclusions_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._day_exclusions_list.setMaximumHeight(110)
        self._day_exclusions_list.setToolTip(
            "Sessions whose day label matches any entry here will have ROI features \n"
            "suppressed (all ROI distances, angles, and flow set to NaN).\n"
            "Match is case-insensitive. Example: \"Acclimation\""
        )

        self._day_excl_edit = QLineEdit()
        self._day_excl_edit.setPlaceholderText("Day label to exclude (e.g. Acclimation)")

        day_add_btn = QPushButton("Add")
        day_add_btn.clicked.connect(self._add_day_exclusion)
        day_remove_btn = QPushButton("Remove Selected")
        day_remove_btn.clicked.connect(self._remove_day_exclusion)

        day_excl_add_row = QHBoxLayout()
        day_excl_add_row.addWidget(self._day_excl_edit, 1)
        day_excl_add_row.addWidget(day_add_btn)

        day_excl_box = QGroupBox("ROI Day Exclusions")
        day_excl_layout = QVBoxLayout(day_excl_box)
        day_excl_note = QLabel(
            "Sessions whose day label contains a listed string will have all ROI "
            "features set to NaN (no object present). Re-run feature extraction after changes."
        )
        day_excl_note.setWordWrap(True)
        day_excl_note.setStyleSheet("font-size: 11px; color: #78909C;")
        day_excl_layout.addWidget(day_excl_note)
        day_excl_layout.addWidget(self._day_exclusions_list)
        day_excl_layout.addLayout(day_excl_add_row)
        day_excl_layout.addWidget(day_remove_btn)

        # ── Save / Reload / Copy / Clear buttons ────────────────────
        save_btn = QPushButton("Save ROI Settings")
        save_btn.clicked.connect(self._save)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        copy_all_btn = QPushButton("Copy to All Subjects")
        copy_all_btn.setToolTip(
            "Copy the current Target Zone and Subject Crop values to every subject.\n"
            "Useful when all subjects share the same camera position."
        )
        copy_all_btn.clicked.connect(self._copy_to_all_subjects)
        clear_all_btn = QPushButton("Clear All ROI Data")
        clear_all_btn.setToolTip(
            "Remove all project and per-subject ROI settings and start fresh."
        )
        clear_all_btn.setStyleSheet("color: #EF5350;")
        clear_all_btn.clicked.connect(self._clear_all_roi_data)
        button_row = QHBoxLayout()
        button_row.addWidget(save_btn)
        button_row.addWidget(reload_btn)
        button_row.addWidget(copy_all_btn)
        button_row.addStretch(1)
        button_row2 = QHBoxLayout()
        button_row2.addWidget(clear_all_btn)
        button_row2.addStretch(1)

        # ── Settings scroll panel (right side) ───────────────────────
        settings_inner = QWidget()
        settings_layout = QVBoxLayout(settings_inner)
        settings_layout.addWidget(scope_box)
        settings_layout.addWidget(subject_box)
        settings_layout.addWidget(roi_count_box)
        settings_layout.addWidget(self._roi_spins_container)
        settings_layout.addWidget(crop_box)
        settings_layout.addWidget(motion_box)
        settings_layout.addWidget(day_excl_box)
        settings_layout.addLayout(button_row)
        settings_layout.addLayout(button_row2)
        settings_layout.addStretch(1)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings_inner)
        settings_scroll.setMinimumWidth(280)
        settings_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # ── Splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(settings_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(splitter, 1)

        # ── Keyboard shortcuts ────────────────────────────────────────
        for key in ("Alt+Down", "Down", "Right"):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(self._go_next_subject)
        for key in ("Alt+Up", "Up", "Left"):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(self._go_prev_subject)
        sc_tab = QShortcut(QKeySequence("Tab"), self)
        sc_tab.activated.connect(self._cycle_draw_mode)

        # Build initial (1 ROI) spinbox groups and draw mode combo
        self._rebuild_roi_spinboxes(1)
        self._rebuild_draw_mode_combo()

        # Start with subject panel hidden until scope is "subject"
        self._subject_box.setVisible(False)

    # ── Project / scope management ────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._status.setText(f"Project: {project_root}")
        self._reload()

    def _on_scope_changed(self) -> None:
        is_subject = self._scope.currentData() == "subject"
        self._subject_box.setVisible(is_subject)
        if is_subject:
            self._refresh_subject_list()
        self._load_scope_values()

    def _subject_ids(self) -> list[str]:
        if not self._project_root:
            return []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return []
        out = sorted({str(s.subject_id).strip() for s in manifest.linked_sessions if s.subject_id})
        return [x for x in out if x]

    def _subject_sessions(self) -> list[tuple[str, str]]:
        """Return (subject_id, session_id) pairs for every linked session, sorted."""
        if not self._project_root:
            return []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return []
        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[str, str]] = []
        for s in manifest.linked_sessions:
            sid = str(s.subject_id or "").strip()
            sess = str(s.session_id or "").strip()
            if sid and sess and (sid, sess) not in seen:
                seen.add((sid, sess))
                pairs.append((sid, sess))
        pairs.sort(key=lambda p: (p[0], p[1]))
        return pairs

    @staticmethod
    def _subject_session_key(subject_id: str, session_id: str) -> str:
        """Composite key used as the subject_rois dict key and list UserRole."""
        return f"{subject_id}::{session_id}"

    @staticmethod
    def _split_subject_key(key: str) -> tuple[str, str]:
        """Split composite key back into (subject_id, session_id)."""
        if "::" in key:
            subj, sess = key.split("::", 1)
            return subj, sess
        return key, ""

    def _subject_to_session_video(self) -> dict[str, tuple[str, str]]:
        """Map session_id → (session_id, video_path_str) for auto-loading."""
        if not self._project_root:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        result: dict[str, tuple[str, str]] = {}
        for s in manifest.linked_sessions:
            sess = str(s.session_id or "").strip()
            if not sess or sess in result:
                continue
            vp = self._imports.video_path_for_session(manifest, s.session_id)
            result[sess] = (s.session_id, str(vp) if vp else "")
        return result

    def _all_sessions_with_video(self) -> list[tuple[str, str]]:
        """Return (session_id, video_path_str) for all imported sessions."""
        if not self._project_root:
            return []
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return []
        results = []
        for s in manifest.linked_sessions:
            vp = self._imports.video_path_for_session(manifest, s.session_id)
            results.append((s.session_id, str(vp) if vp else ""))
        return results

    def _reload(self) -> None:
        if not self._project_root:
            return

        # --- Session/video combo ---
        sessions = self._all_sessions_with_video()
        current_data = self._session_combo.currentData()
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        for sid, vp in sessions:
            self._session_combo.addItem(sid, userData=(sid, vp))
        self._session_combo.blockSignals(False)
        if sessions:
            new_idx = next(
                (i for i, (sid, vp) in enumerate(sessions) if (sid, vp) == current_data),
                0,
            )
            self._session_combo.setCurrentIndex(new_idx)

        # --- Subject list ---
        if self._scope.currentData() == "subject":
            self._refresh_subject_list()

        self._on_scope_changed()

    # ── Subject list ──────────────────────────────────────────────────

    def _refresh_subject_list(self) -> None:
        """Rebuild the subject list — one entry per subject × session."""
        if not self._project_root:
            return

        pairs = self._subject_sessions()
        cfg = self._rois.load(self._project_root)
        subject_rois = cfg.get("subject_rois", {})

        current_row = self._subject_list.currentRow()
        current_key = None
        if current_row >= 0:
            item = self._subject_list.item(current_row)
            if item:
                current_key = item.data(Qt.ItemDataRole.UserRole)

        self._subject_list.blockSignals(True)
        self._subject_list.clear()

        for subject_id, session_id in pairs:
            key = self._subject_session_key(subject_id, session_id)
            sroi = subject_rois.get(key, {})
            zones = sroi.get("target_zones", [sroi.get("target_zone", {})])
            first = zones[0] if zones else {}
            has_roi = (first.get("w", 0) or 0) > 0 and (first.get("h", 0) or 0) > 0

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, key)
            label = f"{subject_id}  /  {session_id}"
            if has_roi:
                item.setText(f"  ✓  {label}")
                item.setForeground(QColor("#66BB6A"))
            else:
                item.setText(f"  ○  {label}")
                item.setForeground(QColor("#78909C"))
            font = item.font()
            font.setPointSize(10)
            item.setFont(font)
            self._subject_list.addItem(item)

        self._subject_list.blockSignals(False)

        # Restore selection
        new_row = 0
        if current_key:
            for i in range(self._subject_list.count()):
                it = self._subject_list.item(i)
                if it and it.data(Qt.ItemDataRole.UserRole) == current_key:
                    new_row = i
                    break
        if self._subject_list.count() > 0:
            self._subject_list.setCurrentRow(new_row)
            self._previous_subject_id = self._current_subject_id()

        self._update_subject_counter()

    def _update_subject_counter(self) -> None:
        total = self._subject_list.count()
        current = self._subject_list.currentRow() + 1 if total > 0 else 0
        # Count how many have ROIs configured
        cfg = self._rois.load(self._project_root) if self._project_root else {}
        subject_rois = cfg.get("subject_rois", {})
        done = 0
        for i in range(total):
            item = self._subject_list.item(i)
            if item:
                sid = item.data(Qt.ItemDataRole.UserRole)
                s_block = subject_rois.get(sid, {})
                zones = s_block.get("target_zones", [s_block.get("target_zone", {})])
                first = zones[0] if zones else {}
                if (first.get("w", 0) or 0) > 0 and (first.get("h", 0) or 0) > 0:
                    done += 1
        self._subject_counter.setText(f"{current} / {total}  ({done} configured)")

    def _current_subject_id(self) -> str:
        """Return the subject ID currently selected in the list."""
        item = self._subject_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _on_subject_list_changed(self, row: int) -> None:
        """Auto-save previous subject, load new subject's ROI, optionally load video."""
        if self._auto_save_enabled and self._current_subject_dirty and self._previous_subject_id:
            self._save_subject_quiet(self._previous_subject_id)

        self._current_subject_dirty = False
        self._previous_subject_id = self._current_subject_id()
        self._load_scope_values()
        self._update_subject_counter()

        # Auto-load the selected subject's video
        if self._auto_load_video_cb.isChecked() and self._project_root:
            self._auto_load_subject_video()

    def _auto_load_subject_video(self) -> None:
        """Open the selected entry's video directly, bypassing the session combo."""
        key = self._current_subject_id()
        if not key:
            return
        _subject_id, session_id = self._split_subject_key(key)
        if not session_id:
            return
        session_map = self._subject_to_session_video()
        entry = session_map.get(session_id)
        if not entry:
            return
        _sess_id, video_path_str = entry
        if not video_path_str:
            return

        # Sync the session combo silently (no signal → won't reset video)
        self._session_combo.blockSignals(True)
        for i in range(self._session_combo.count()):
            data = self._session_combo.itemData(i)
            if data and data[0] == session_id:
                self._session_combo.setCurrentIndex(i)
                break
        self._session_combo.blockSignals(False)

        # Open the video file directly
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        self._close_video()
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            return
        self._video_cap = cap
        self._n_video_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._frame_slider.setMaximum(self._n_video_frames - 1)
        current_frame = self._frame_slider.value()
        if current_frame >= self._n_video_frames:
            self._frame_slider.setValue(0)
            current_frame = 0
        self._frame_label.setText(f"{current_frame} / {self._n_video_frames - 1}")
        self._show_frame(current_frame)

    def _go_next_subject(self) -> None:
        """Save current and advance to next subject."""
        if self._scope.currentData() != "subject":
            return
        count = self._subject_list.count()
        if count == 0:
            return
        row = self._subject_list.currentRow()
        if row < count - 1:
            self._subject_list.setCurrentRow(row + 1)

    def _go_prev_subject(self) -> None:
        """Save current and go to previous subject."""
        if self._scope.currentData() != "subject":
            return
        count = self._subject_list.count()
        if count == 0:
            return
        row = self._subject_list.currentRow()
        if row > 0:
            self._subject_list.setCurrentRow(row - 1)

    def _save_current_subject_quiet(self) -> None:
        """Silently save the current subject's ROI without dialogs."""
        self._save_subject_quiet(self._current_subject_id())

    def _save_subject_quiet(self, subject_id: str) -> None:
        """Silently save the given subject's ROI without dialogs."""
        if not self._project_root or not subject_id:
            return

        cfg = self._rois.load(self._project_root)
        cfg.setdefault("subject_rois", {})
        cfg.setdefault("motion", {})

        target_zones = self._read_all_target_zones()
        crop = self._read_roi_spins(
            (self._crop_x, self._crop_y, self._crop_w, self._crop_h)
        )
        cfg["subject_rois"][subject_id] = {
            "target_zones": target_zones,
            "subject_crop": crop,
        }
        self._rois.save(self._project_root, cfg)
        logger.info("Auto-saved ROI for subject '%s'", subject_id)

        # Update the list item's status indicator
        self._refresh_subject_list_item(subject_id)
        self._current_subject_dirty = False

    def _refresh_subject_list_item(self, key: str) -> None:
        """Update a single item's visual status without rebuilding the whole list."""
        if not self._project_root:
            return
        cfg = self._rois.load(self._project_root)
        subject_rois = cfg.get("subject_rois", {})
        zones = subject_rois.get(key, {}).get("target_zones", [])
        has_roi = bool(zones) and (zones[0].get("w", 0) or 0) > 0 and (zones[0].get("h", 0) or 0) > 0

        subject_id, session_id = self._split_subject_key(key)
        label = f"{subject_id}  /  {session_id}" if session_id else subject_id

        for i in range(self._subject_list.count()):
            item = self._subject_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == key:
                if has_roi:
                    item.setText(f"  ✓  {label}")
                    item.setForeground(QColor("#66BB6A"))
                else:
                    item.setText(f"  ○  {label}")
                    item.setForeground(QColor("#78909C"))
                break
        self._update_subject_counter()

    # ── Dynamic ROI spinbox builder ───────────────────────────────────

    def _rebuild_roi_spinboxes(self, n: int) -> None:
        """Create exactly *n* ROI spinbox groups inside _roi_spins_container."""
        n = max(1, min(n, MAX_ROIS))

        # Remove existing widgets
        while self._roi_spins_layout.count():
            item = self._roi_spins_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._roi_spinbox_groups = []
        self._roi_boxes = []

        # Resize the shape list to n slots, preserving existing shapes.
        while len(self._roi_shapes) < n:
            self._roi_shapes.append({"x": 0, "y": 0, "w": 0, "h": 0})
        self._roi_shapes = self._roi_shapes[:n]

        colors = _roi_qcolors()
        for i in range(n):
            hex_c = ROI_COLORS[i % len(ROI_COLORS)]
            border_c = colors[i % len(colors)][0]
            label = f"ROI {i + 1}" + (" (Target Zone)" if i == 0 else "")

            sx = QSpinBox(); sx.setRange(0, 10000)
            sy = QSpinBox(); sy.setRange(0, 10000)
            sw = QSpinBox(); sw.setRange(0, 10000)
            sh = QSpinBox(); sh.setRange(0, 10000)

            # Capture index for the lambda
            idx = i
            for sp in (sx, sy, sw, sh):
                sp.valueChanged.connect(lambda _val, _i=idx: self._on_roi_spinbox_changed(_i))

            box = QGroupBox(label)
            box.setStyleSheet(
                f"QGroupBox {{ color: {hex_c}; font-weight: 600; "
                f"border: 1px solid {hex_c}33; border-radius: 4px; "
                f"margin-top: 6px; }} "
                f"QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 2px; }}"
            )
            form = QFormLayout(box)
            form.setContentsMargins(6, 14, 6, 6)
            form.addRow("x:", sx)
            form.addRow("y:", sy)
            form.addRow("width:", sw)
            form.addRow("height:", sh)

            self._roi_spinbox_groups.append((sx, sy, sw, sh))
            self._roi_boxes.append(box)
            self._roi_spins_layout.addWidget(box)

        # Sync canvas
        self._canvas.set_n_rois(n)

    def _rebuild_draw_mode_combo(self) -> None:
        """Repopulate the draw-mode combo based on current ROI count."""
        current_data = self._draw_mode_combo.currentData()
        self._draw_mode_combo.blockSignals(True)
        self._draw_mode_combo.clear()
        n = len(self._roi_spinbox_groups)
        for i in range(n):
            label = f"Draw ROI {i + 1}" + (" (Target Zone)" if i == 0 else "")
            self._draw_mode_combo.addItem(label, userData=f"roi_{i}")
        self._draw_mode_combo.addItem("Draw Subject Crop", userData="subject_crop")
        # Restore selection if possible
        idx = self._draw_mode_combo.findData(current_data)
        self._draw_mode_combo.setCurrentIndex(max(0, idx))
        self._draw_mode_combo.blockSignals(False)
        self._on_draw_mode_changed()

    def _on_roi_count_changed(self, n: int) -> None:
        """User changed the ROI count spinbox."""
        self._rebuild_roi_spinboxes(n)
        self._rebuild_draw_mode_combo()
        # Reload values so the new spinboxes are populated
        self._load_scope_values()
        if self._project_root:
            # Persist the new roi_count
            cfg = self._rois.load(self._project_root)
            cfg["roi_count"] = n
            self._rois.save(self._project_root, cfg)

    # ── Spinbox helpers ───────────────────────────────────────────────

    @staticmethod
    def _set_roi_spins(
        spins: tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox], roi: dict[str, int]
    ) -> None:
        x, y, w, h = spins
        x.setValue(int(roi.get("x", 0) or 0))
        y.setValue(int(roi.get("y", 0) or 0))
        w.setValue(int(roi.get("w", 0) or 0))
        h.setValue(int(roi.get("h", 0) or 0))

    @staticmethod
    def _read_roi_spins(
        spins: tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox]
    ) -> dict[str, int]:
        x, y, w, h = spins
        return {"x": int(x.value()), "y": int(y.value()), "w": int(w.value()), "h": int(h.value())}

    def _block_all_roi_spins(self, block: bool) -> None:
        for grp in self._roi_spinbox_groups:
            for sp in grp:
                sp.blockSignals(block)

    def _read_all_target_zones(self) -> list[dict]:
        """Return the authoritative shape dict for every ROI slot.

        Rectangular slots are refreshed from their spinboxes (so manual numeric
        edits are honoured); circle/polygon slots keep the geometry drawn on the
        canvas.
        """
        zones: list[dict] = []
        for i, grp in enumerate(self._roi_spinbox_groups):
            shape = self._roi_shapes[i] if i < len(self._roi_shapes) else {}
            if roi_geometry.roi_shape(shape) == "rect":
                zones.append(self._read_roi_spins(grp))
            else:
                zones.append(roi_geometry.normalize_roi(shape))
        return zones

    def _load_target_zones(self, zones: list[dict]) -> None:
        """Populate ROI shapes, spinboxes (bbox mirror), and canvas from dicts."""
        self._block_all_roi_spins(True)
        for i, grp in enumerate(self._roi_spinbox_groups):
            raw = zones[i] if i < len(zones) else {"x": 0, "y": 0, "w": 0, "h": 0}
            shape = roi_geometry.normalize_roi(raw)
            if i < len(self._roi_shapes):
                self._roi_shapes[i] = shape
            self._set_roi_spins(grp, shape)  # spinboxes mirror the bbox
        self._block_all_roi_spins(False)

    def _load_scope_values(self) -> None:
        if not self._project_root:
            return
        cfg = self._rois.load(self._project_root)

        # Sync ROI count from project config
        saved_count = max(1, int(cfg.get("roi_count", 1)))
        if saved_count != self._roi_count_spin.value():
            self._roi_count_spin.blockSignals(True)
            self._roi_count_spin.setValue(saved_count)
            self._roi_count_spin.blockSignals(False)
            self._rebuild_roi_spinboxes(saved_count)
            self._rebuild_draw_mode_combo()

        is_subject = self._scope.currentData() == "subject"
        subject_id = self._current_subject_id() if is_subject else ""

        if is_subject and subject_id:
            src = cfg.get("subject_rois", {}).get(subject_id, {})
        else:
            src = cfg.get("project_rois", {})

        target_zones = src.get("target_zones", [src.get("target_zone", {})])
        if not isinstance(target_zones, list):
            target_zones = [target_zones]

        self._load_target_zones(target_zones)
        self._set_roi_spins(
            (self._crop_x, self._crop_y, self._crop_w, self._crop_h),
            src.get("subject_crop", {}),
        )
        self._canvas.set_rois(list(self._roi_shapes))
        self._canvas.set_crop(src.get("subject_crop", {}))
        # Day exclusions are project-level — always load from project config
        excl_days = cfg.get("roi_excluded_day_labels", [])
        self._day_exclusions_list.blockSignals(True)
        self._day_exclusions_list.clear()
        for label in excl_days:
            if label:
                self._day_exclusions_list.addItem(str(label))
        self._day_exclusions_list.blockSignals(False)

    # ── Canvas ↔ spinbox bidirectional sync ───────────────────────────

    def _on_canvas_roi_n_changed(self, index: int, roi: dict) -> None:
        """Canvas draw finished — store the shape and mirror its bbox to spins."""
        if index >= len(self._roi_spinbox_groups):
            return
        shape = roi_geometry.normalize_roi(roi)
        if index < len(self._roi_shapes):
            self._roi_shapes[index] = shape
        grp = self._roi_spinbox_groups[index]
        self._block_all_roi_spins(True)
        self._set_roi_spins(grp, shape)
        self._block_all_roi_spins(False)
        self._current_subject_dirty = True

    def _on_canvas_crop_changed(self, roi: dict) -> None:
        """Canvas drag finished (subject crop) — push result to spinboxes."""
        self._set_roi_spins(
            (self._crop_x, self._crop_y, self._crop_w, self._crop_h), roi
        )
        self._current_subject_dirty = True

    def _on_draw_mode_changed(self) -> None:
        mode = str(self._draw_mode_combo.currentData() or "roi_0")
        self._canvas.set_draw_mode(mode)
        # The subject crop must stay rectangular; disable shape controls for it.
        is_crop = mode == "subject_crop"
        for w in (self._shape_combo, self._smooth_btn, self._angularize_btn):
            w.setEnabled(not is_crop)

    def _on_shape_changed(self) -> None:
        shape = str(self._shape_combo.currentData() or "rect")
        self._canvas.set_shape_mode(shape)

    def _active_roi_slot(self) -> int:
        """Index of the ROI slot the draw-mode combo currently targets."""
        data = str(self._draw_mode_combo.currentData() or "")
        if data.startswith("roi_"):
            try:
                return int(data[4:])
            except ValueError:
                return 0
        return 0

    def _cleanup_active_polygon(self, mode: str) -> None:
        """Apply Chaikin smoothing or RDP simplification to the active polygon."""
        idx = self._active_roi_slot()
        if idx >= len(self._roi_shapes):
            return
        shape = self._roi_shapes[idx]
        if roi_geometry.roi_shape(shape) != "polygon":
            QMessageBox.information(
                self,
                "Not a Polygon",
                "Select a target-zone ROI that was drawn as a freehand polygon, "
                "then use Smooth / Angularize.",
            )
            return
        pts = shape.get("points", [])
        if mode == "smooth":
            new_pts = roi_geometry.chaikin_smooth(pts, iterations=2)
        else:
            # Epsilon scales with the polygon's size so simplification is
            # resolution-independent (≈2% of the bounding-box diagonal).
            _x, _y, w, h = roi_geometry.roi_bbox(shape)
            eps = max(1.0, 0.02 * (w ** 2 + h ** 2) ** 0.5)
            new_pts = roi_geometry.rdp_simplify(pts, eps)
        new_shape = roi_geometry.normalize_roi({"shape": "polygon", "points": new_pts})
        if not roi_geometry.roi_has_area(new_shape):
            return
        self._roi_shapes[idx] = new_shape
        self._block_all_roi_spins(True)
        self._set_roi_spins(self._roi_spinbox_groups[idx], new_shape)
        self._block_all_roi_spins(False)
        self._canvas.set_roi_at(idx, new_shape)
        self._current_subject_dirty = True

    def _smooth_active_polygon(self) -> None:
        self._cleanup_active_polygon("smooth")

    def _angularize_active_polygon(self) -> None:
        self._cleanup_active_polygon("angularize")

    def _cycle_draw_mode(self) -> None:
        """Advance the draw-mode combo to the next item, wrapping around."""
        n = self._draw_mode_combo.count()
        if n == 0:
            return
        next_idx = (self._draw_mode_combo.currentIndex() + 1) % n
        self._draw_mode_combo.setCurrentIndex(next_idx)

    def _on_roi_spinbox_changed(self, roi_index: int) -> None:
        """ROI spinbox edited manually — the slot reverts to a rectangle."""
        if roi_index < len(self._roi_spinbox_groups):
            roi = self._read_roi_spins(self._roi_spinbox_groups[roi_index])
            if roi_index < len(self._roi_shapes):
                self._roi_shapes[roi_index] = roi
            self._canvas.set_roi_at(roi_index, roi)
        self._current_subject_dirty = True

    def _clear_target_roi(self) -> None:
        """Clear all target-zone spinboxes, shapes, and canvas overlays."""
        self._block_all_roi_spins(True)
        for grp in self._roi_spinbox_groups:
            for sp in grp:
                sp.setValue(0)
        self._block_all_roi_spins(False)
        self._roi_shapes = [{"x": 0, "y": 0, "w": 0, "h": 0} for _ in self._roi_spinbox_groups]
        self._canvas.set_rois([{"x": 0, "y": 0, "w": 0, "h": 0}] * len(self._roi_spinbox_groups))
        self._current_subject_dirty = True

    # ── Day exclusion helpers ─────────────────────────────────────────

    def _read_day_exclusions(self) -> list[str]:
        return [
            self._day_exclusions_list.item(i).text()
            for i in range(self._day_exclusions_list.count())
            if self._day_exclusions_list.item(i).text().strip()
        ]

    def _add_day_exclusion(self) -> None:
        label = self._day_excl_edit.text().strip()
        if not label:
            return
        existing = self._read_day_exclusions()
        if label.lower() in {e.lower() for e in existing}:
            self._day_excl_edit.clear()
            return
        self._day_exclusions_list.addItem(label)
        self._day_excl_edit.clear()
        self._save_day_exclusions()

    def _remove_day_exclusion(self) -> None:
        row = self._day_exclusions_list.currentRow()
        if row >= 0:
            self._day_exclusions_list.takeItem(row)
            self._save_day_exclusions()

    def _save_day_exclusions(self) -> None:
        """Persist only the day-exclusion list without touching ROI zones."""
        if not self._project_root:
            return
        cfg = self._rois.load(self._project_root)
        cfg["roi_excluded_day_labels"] = self._read_day_exclusions()
        self._rois.save(self._project_root, cfg)

    # ── Video frame loading ───────────────────────────────────────────

    def _on_video_session_changed(self) -> None:
        self._close_video()
        self._frame_slider.setMaximum(0)
        self._frame_slider.setValue(0)
        self._frame_label.setText("0 / 0")

    def _close_video(self) -> None:
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        self._n_video_frames = 0

    def _load_video_frame(self) -> None:
        data = self._session_combo.currentData()
        if not data:
            QMessageBox.warning(self, "No Session", "Select a session first.")
            return
        _sid, video_path_str = data
        if not video_path_str:
            QMessageBox.warning(self, "No Video", "This session has no associated video file.")
            return
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            QMessageBox.warning(
                self, "OpenCV Missing", "OpenCV is required to preview video frames."
            )
            return

        self._close_video()
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            QMessageBox.warning(self, "Cannot Open Video", f"Could not open:\n{video_path_str}")
            return

        self._video_cap = cap
        self._n_video_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._frame_slider.setMaximum(self._n_video_frames - 1)
        # Keep slider at its current position if valid, otherwise reset to 0
        current_frame = self._frame_slider.value()
        if current_frame >= self._n_video_frames:
            self._frame_slider.setValue(0)
            current_frame = 0
        self._frame_label.setText(f"{current_frame} / {self._n_video_frames - 1}")
        self._show_frame(current_frame)

    def _on_frame_slider_changed(self, value: int) -> None:
        if self._video_cap is not None:
            self._show_frame(value)
        self._frame_label.setText(f"{value} / {max(0, self._n_video_frames - 1)}")

    def _show_frame(self, frame_idx: int) -> None:
        if self._video_cap is None:
            return
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self._video_cap.read()
        self._canvas.set_frame(bgr if ok else None)

    # ── Clear all ROI data ─────────────────────────────────────────

    def _clear_all_roi_data(self) -> None:
        """Wipe all project and per-subject ROI settings back to defaults."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        reply = QMessageBox.question(
            self,
            "Clear All ROI Data",
            "This will erase ALL project-level and per-subject ROI settings "
            "(all target zones, subject crop) and reset to empty defaults.\n\n"
            "Local motion radius and ROI count will be kept.  This cannot be undone.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cfg = self._rois.load(self._project_root)
        empty_roi = {"x": 0, "y": 0, "w": 0, "h": 0}
        n = self._roi_count_spin.value()
        cfg["project_rois"] = {
            "target_zones": [dict(empty_roi) for _ in range(n)],
            "subject_crop": dict(empty_roi),
        }
        cfg["subject_rois"] = {}
        self._rois.save(self._project_root, cfg)
        self._current_subject_dirty = False

        # Reset spinboxes and canvas
        self._block_all_roi_spins(True)
        for grp in self._roi_spinbox_groups:
            for sp in grp:
                sp.setValue(0)
        self._block_all_roi_spins(False)
        for sp in (self._crop_x, self._crop_y, self._crop_w, self._crop_h):
            sp.setValue(0)
        self._roi_shapes = [dict(empty_roi) for _ in range(n)]
        self._canvas.set_rois([dict(empty_roi)] * n)
        self._canvas.set_crop(empty_roi)

        if self._scope.currentData() == "subject":
            self._refresh_subject_list()
        QMessageBox.information(self, "Cleared", "All ROI data has been reset.")

    # ── Copy to all subjects ─────────────────────────────────────────

    def _copy_to_all_subjects(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        pairs = self._subject_sessions()
        if not pairs:
            QMessageBox.warning(self, "No Subjects", "No subjects found in the import manifest.")
            return

        target_zones = self._read_all_target_zones()
        crop = self._read_roi_spins(
            (self._crop_x, self._crop_y, self._crop_w, self._crop_h)
        )

        reply = QMessageBox.question(
            self,
            "Copy to All Subject / Sessions",
            f"This will overwrite ROI settings for all {len(pairs)} subject × session entries "
            f"with the current {len(target_zones)} target zone(s) and Subject Crop.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cfg = self._rois.load(self._project_root)
        cfg.setdefault("subject_rois", {})
        for subject_id, session_id in pairs:
            key = self._subject_session_key(subject_id, session_id)
            cfg["subject_rois"][key] = {
                "target_zones": [dict(z) for z in target_zones],
                "subject_crop": dict(crop),
            }
        self._rois.save(self._project_root, cfg)
        self._current_subject_dirty = False
        self._refresh_subject_list()
        QMessageBox.information(
            self, "Copy Complete", f"ROI settings copied to {len(pairs)} subject/session entries."
        )

    # ── Save ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        cfg = self._rois.load(self._project_root)
        cfg.setdefault("project_rois", {})
        cfg.setdefault("subject_rois", {})
        cfg.setdefault("motion", {})

        target_zones = self._read_all_target_zones()
        crop = self._read_roi_spins(
            (self._crop_x, self._crop_y, self._crop_w, self._crop_h)
        )

        is_subject = self._scope.currentData() == "subject"
        subject_id = self._current_subject_id()

        if is_subject:
            if not subject_id:
                QMessageBox.warning(
                    self, "No Subject", "No subject IDs available for subject override scope."
                )
                return
            cfg["subject_rois"][subject_id] = {
                "target_zones": [dict(z) for z in target_zones],
                "subject_crop": dict(crop),
            }
            _subj, _sess = self._split_subject_key(subject_id)
            scope_label = f"{_subj}  /  {_sess}" if _sess else subject_id
        else:
            cfg["project_rois"]["target_zones"] = [dict(z) for z in target_zones]
            cfg["project_rois"]["subject_crop"] = dict(crop)
            scope_label = "project defaults"

        cfg["roi_count"] = len(target_zones)
        cfg["roi_excluded_day_labels"] = self._read_day_exclusions()
        self._rois.save(self._project_root, cfg)
        self._current_subject_dirty = False
        if is_subject:
            self._refresh_subject_list_item(subject_id)
        QMessageBox.information(self, "ROI Settings", f"Saved ROI settings for {scope_label}.")

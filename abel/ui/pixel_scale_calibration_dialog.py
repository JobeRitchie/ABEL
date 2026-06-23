"""Interactive per-session pixel/mm calibration dialog."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import ImportManifest
from abel.services.import_service import ImportService


class _ClickableFrameLabel(QLabel):
    clicked = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._display_rect: QRect | None = None
        self._image_size: tuple[int, int] | None = None

    def set_mapping(self, display_rect: QRect | None, image_size: tuple[int, int] | None) -> None:
        self._display_rect = display_rect
        self._image_size = image_size

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._display_rect is None or self._image_size is None:
            return

        rect = self._display_rect
        pos = event.position().toPoint()
        if not rect.contains(pos):
            return

        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0 or rect.width() <= 0 or rect.height() <= 0:
            return

        rel_x = (pos.x() - rect.x()) / float(rect.width())
        rel_y = (pos.y() - rect.y()) / float(rect.height())
        x = int(np.clip(round(rel_x * (img_w - 1)), 0, img_w - 1))
        y = int(np.clip(round(rel_y * (img_h - 1)), 0, img_h - 1))
        self.clicked.emit(x, y)


class PixelScaleCalibrationDialog(QDialog):
    """Pick two points in a frame and convert known mm distance to pixels/mm."""

    def __init__(
        self,
        import_service: ImportService,
        manifest: ImportManifest,
        default_session_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate Pixels per mm")
        self.setMinimumSize(980, 760)

        self._imports = import_service
        self._manifest = manifest
        self._frame_bgr: np.ndarray | None = None
        self._p1: tuple[int, int] | None = None
        self._p2: tuple[int, int] | None = None

        self.result_session_id: str | None = None
        self.result_pixels_per_mm: float | None = None

        self._session = QComboBox()
        for linked in manifest.linked_sessions:
            sid = str(linked.session_id)
            subject = str(linked.subject_id or sid)
            self._session.addItem(f"{subject} ({sid})", userData=sid)
        if default_session_id:
            idx = self._session.findData(default_session_id)
            if idx >= 0:
                self._session.setCurrentIndex(idx)

        self._frame_index = QSpinBox()
        self._frame_index.setMinimum(0)
        self._frame_index.setMaximum(0)
        self._frame_index.setValue(0)

        self._distance_mm = QDoubleSpinBox()
        self._distance_mm.setDecimals(4)
        self._distance_mm.setRange(0.0001, 1_000_000.0)
        self._distance_mm.setValue(10.0)
        self._distance_mm.setSingleStep(1.0)

        self._frame_label = _ClickableFrameLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setStyleSheet("background: #111; color: #777;")
        self._frame_label.setText("Load a frame, then click two points.")
        self._frame_label.setMinimumSize(900, 620)

        self._status = QLabel("Click two points along a known-length reference.")
        self._status.setWordWrap(True)
        self._result = QLabel("pixels/mm: -")

        load_btn = QPushButton("Load Frame")
        load_btn.clicked.connect(self._load_frame)

        reset_btn = QPushButton("Reset Points")
        reset_btn.clicked.connect(self._reset_points)

        apply_btn = QPushButton("Apply px/mm")
        apply_btn.clicked.connect(self._apply)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        self._frame_label.clicked.connect(self._on_frame_clicked)
        self._distance_mm.valueChanged.connect(lambda _v: self._recompute())

        form = QFormLayout()
        form.addRow("Session:", self._session)
        form.addRow("Frame:", self._frame_index)
        form.addRow("Known distance (mm):", self._distance_mm)

        top_row = QHBoxLayout()
        top_row.addLayout(form, 1)
        top_row.addWidget(load_btn)
        top_row.addWidget(reset_btn)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._result)
        bottom_row.addStretch(1)
        bottom_row.addWidget(apply_btn)
        bottom_row.addWidget(cancel_btn)

        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(self._frame_label, 1)
        root.addWidget(self._status)
        root.addLayout(bottom_row)

        self._load_frame()

    def _session_id(self) -> str:
        return str(self._session.currentData() or "").strip()

    def _load_frame(self) -> None:
        import cv2  # noqa: PLC0415

        sid = self._session_id()
        if not sid:
            self._status.setText("No session selected.")
            return

        video_path = self._imports.video_path_for_session(self._manifest, sid)
        if not video_path or not Path(video_path).exists():
            self._status.setText("Video path not found for selected session.")
            self._frame_label.setText("Video path not found.")
            self._frame_bgr = None
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._status.setText("Unable to open video file.")
            self._frame_label.setText("Unable to open video file.")
            self._frame_bgr = None
            return

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                self._frame_index.setMaximum(max(0, frame_count - 1))
            idx = int(self._frame_index.value())
            if frame_count > 0:
                idx = int(np.clip(idx, 0, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                self._status.setText("Could not read frame from selected video.")
                self._frame_label.setText("Could not read frame.")
                self._frame_bgr = None
                return
        finally:
            cap.release()

        self._frame_bgr = frame
        self._reset_points(silent=True)
        self._render_frame()
        self._status.setText("Click first point, then second point on a known distance.")

    def _reset_points(self, silent: bool = False) -> None:
        self._p1 = None
        self._p2 = None
        self._recompute()
        if self._frame_bgr is not None:
            self._render_frame()
        if not silent:
            self._status.setText("Points reset. Click two points to measure.")

    def _on_frame_clicked(self, x: int, y: int) -> None:
        if self._frame_bgr is None:
            return

        if self._p1 is None:
            self._p1 = (x, y)
            self._status.setText(f"Point 1 set at ({x}, {y}). Click point 2.")
        elif self._p2 is None:
            self._p2 = (x, y)
            self._status.setText(f"Point 2 set at ({x}, {y}). Enter mm and apply.")
        else:
            self._p1 = (x, y)
            self._p2 = None
            self._status.setText(f"Restarted: new point 1 at ({x}, {y}). Click point 2.")

        self._recompute()
        self._render_frame()

    def _recompute(self) -> None:
        if self._p1 is None or self._p2 is None:
            self._result.setText("pixels/mm: -")
            return

        dx = float(self._p2[0] - self._p1[0])
        dy = float(self._p2[1] - self._p1[1])
        pixel_dist = math.hypot(dx, dy)
        mm = float(self._distance_mm.value())
        ppm = pixel_dist / max(mm, 1e-12)
        self._result.setText(
            f"pixel distance: {pixel_dist:.3f} px    |    pixels/mm: {ppm:.6f}"
        )

    def _render_frame(self) -> None:
        import cv2  # noqa: PLC0415

        if self._frame_bgr is None:
            return

        draw = self._frame_bgr.copy()
        if self._p1 is not None:
            cv2.circle(draw, self._p1, 6, (0, 255, 255), -1)
        if self._p2 is not None:
            cv2.circle(draw, self._p2, 6, (0, 255, 255), -1)
        if self._p1 is not None and self._p2 is not None:
            cv2.line(draw, self._p1, self._p2, (0, 255, 255), 2)

        rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888).copy()

        target = self._frame_label.size()
        if target.width() <= 1 or target.height() <= 1:
            pix = QPixmap.fromImage(qimg)
            self._frame_label.setPixmap(pix)
            self._frame_label.set_mapping(QRect(0, 0, pix.width(), pix.height()), (w, h))
            return

        scaled = QPixmap.fromImage(qimg).scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)

        x_off = max(0, (self._frame_label.width() - scaled.width()) // 2)
        y_off = max(0, (self._frame_label.height() - scaled.height()) // 2)
        self._frame_label.set_mapping(
            QRect(x_off, y_off, scaled.width(), scaled.height()),
            (w, h),
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._frame_bgr is not None:
            self._render_frame()

    def _apply(self) -> None:
        if self._p1 is None or self._p2 is None:
            QMessageBox.warning(self, "Calibration", "Click two points first.")
            return

        dx = float(self._p2[0] - self._p1[0])
        dy = float(self._p2[1] - self._p1[1])
        pixel_dist = math.hypot(dx, dy)
        mm = float(self._distance_mm.value())
        if mm <= 0 or pixel_dist <= 0:
            QMessageBox.warning(
                self,
                "Calibration",
                "Measured pixel distance and mm value must be positive.",
            )
            return

        self.result_session_id = self._session_id()
        self.result_pixels_per_mm = pixel_dist / mm
        self.accept()

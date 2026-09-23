"""Preview of the per-animal identity dots as they will appear in a clip."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# (percent, frame offset within the window) -> BGR frame or None
RenderFn = Callable[[int, int], "object | None"]


class IdentityMarkerPreviewDialog(QDialog):
    """Show one frame of a real clip with the identity dots at a chosen size.

    The frame comes from the same writer extraction uses, so the dot radius,
    crop and output resolution are exactly what the extracted clip will have.
    """

    _ZOOM_FIT = 0

    def __init__(
        self,
        render_fn: RenderFn,
        percent: int,
        n_frames: int,
        radius_fn: Callable[[int, int], float],
        caption: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Identity Marker Size")
        self._render_fn = render_fn
        self._radius_fn = radius_fn
        self._frame = None

        self._size_spin = QSpinBox()
        self._size_spin.setRange(10, 400)
        self._size_spin.setSingleStep(10)
        self._size_spin.setSuffix(" %")
        self._size_spin.setValue(int(percent))
        self._size_spin.setToolTip("Dot size relative to the default (100%).")

        self._zoom_combo = QComboBox()
        self._zoom_combo.addItem("Fit window", self._ZOOM_FIT)
        for z in (1, 2, 3, 4):
            self._zoom_combo.addItem(f"{z}x (clip pixels)" if z == 1 else f"{z}x", z)
        self._zoom_combo.setToolTip(
            "1x shows the clip at its real resolution. The Review player scales "
            "clips up to fill its panel, which is closer to 'Fit window'."
        )

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, int(n_frames) - 1))
        self._frame_slider.setValue(max(0, int(n_frames) - 1) // 2)
        self._frame_slider.setToolTip("Frame within the clip window.")

        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fm = self.fontMetrics()
        self._image.setMinimumSize(fm.height() * 20, fm.height() * 15)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._image)
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._info = QLabel()
        self._info.setWordWrap(True)
        self._caption = QLabel(caption)
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet("color: #78909C;")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Marker size:"))
        controls.addWidget(self._size_spin)
        controls.addSpacing(fm.averageCharWidth() * 2)
        controls.addWidget(QLabel("Zoom:"))
        controls.addWidget(self._zoom_combo)
        controls.addSpacing(fm.averageCharWidth() * 2)
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self._frame_slider, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use This Size")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._caption)
        layout.addLayout(controls)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._info)
        layout.addWidget(buttons)

        # Rendering decodes video, so wait for the spinbox/slider to settle.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(150)
        self._render_timer.timeout.connect(self._rerender)
        self._size_spin.valueChanged.connect(lambda _v: self._render_timer.start())
        self._frame_slider.valueChanged.connect(lambda _v: self._render_timer.start())
        self._zoom_combo.currentIndexChanged.connect(lambda _i: self._show_frame())

        self.resize(fm.height() * 45, fm.height() * 36)
        self._rerender()

    def percent(self) -> int:
        return int(self._size_spin.value())

    def _rerender(self) -> None:
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self._frame = self._render_fn(self.percent(), int(self._frame_slider.value()))
        except Exception as exc:  # show the reason instead of a blank dialog
            self._frame = None
            self._info.setText(f"Could not render a preview frame: {exc}")
        finally:
            self.unsetCursor()
        self._show_frame()

    def _show_frame(self) -> None:
        frame = self._frame
        if frame is None:
            self._image.setPixmap(QPixmap())
            if not self._info.text().startswith("Could not"):
                self._info.setText("Could not read a frame from the source video.")
            return
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        zoom = self._zoom_combo.currentData()
        if zoom == self._ZOOM_FIT:
            vp = self._scroll.viewport().size()
            pix = pix.scaled(
                max(1, vp.width() - 4), max(1, vp.height() - 4),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        elif int(zoom) > 1:
            # Nearest-neighbor so the real pixel size of the dot stays visible.
            pix = pix.scaled(
                w * int(zoom), h * int(zoom),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self._image.setPixmap(pix)
        r = self._radius_fn(w, self.percent())
        self._info.setText(
            f"Clip is {w} x {h} px. Each identity dot has a {r:.1f} px radius "
            f"({2 * r + 2:.1f} px across with its black outline, "
            f"{100.0 * (2 * r + 2) / max(1, w):.1f}% of the clip width)."
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom_combo.currentData() == self._ZOOM_FIT:
            self._show_frame()

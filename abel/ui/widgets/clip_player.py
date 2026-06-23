"""Reusable frame-by-frame video clip player.

A trimmed, self-contained player adapted from the Review tab's
``CandidateVideoPlayer``.  Supports play/pause, frame stepping, a scrub slider,
adjustable playback speed, and clip looping — everything the Validation quiz
needs to reuse the proven review-style playback experience.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_SPEED_LABELS = ["0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x"]
_SPEED_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


class ClipPlayer(QWidget):
    """Minimal frame-by-frame video player for short clips."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap = None
        self._n_frames = 0
        self._fps = 30.0
        self._cur_frame = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._last_bgr = None
        self._loop_enabled = True
        self._speed_multiplier = 1.0

        self._display = QLabel("No clip loaded")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(320, 240)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setStyleSheet("background: #0a0e18; color: #546e7a;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self.toggle_play)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(lambda: self.seek(self._cur_frame - 1))

        self._next_btn = QPushButton("▶▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(lambda: self.seek(self._cur_frame + 1))

        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setStyleSheet("font-size: 11px; font-weight: 600;")

        self._loop_chk = QCheckBox("Loop")
        self._loop_chk.setToolTip("Automatically restart the clip from the beginning when it ends")
        self._loop_chk.setChecked(True)
        self._loop_chk.toggled.connect(self._on_loop_toggled)

        self._speed_combo = QComboBox()
        for lbl in _SPEED_LABELS:
            self._speed_combo.addItem(lbl)
        self._speed_combo.setCurrentIndex(3)  # 1x
        self._speed_combo.setFixedWidth(64)
        self._speed_combo.setToolTip("Playback speed multiplier")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._slider, 1)
        ctrl.addWidget(self._frame_label)
        ctrl.addWidget(self._loop_chk)
        ctrl.addWidget(self._speed_combo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display, 1)
        layout.addLayout(ctrl)

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    def set_loop(self, enabled: bool) -> None:
        self._loop_chk.setChecked(bool(enabled))

    def load_clip(self, path: str, autoplay: bool = True) -> bool:
        self.close_clip()
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            self._display.setText("OpenCV not installed.\nCannot preview video.")
            return False

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._display.setText(f"Cannot open:\n{path}")
            return False

        self._cap = cap
        self._n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._slider.setMaximum(self._n_frames - 1)
        self._set_controls_enabled(True)
        self.seek(0)
        if autoplay and not self._playing:
            self.toggle_play()
        return True

    def close_clip(self) -> None:
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("▶")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._n_frames = 0
        self._cur_frame = 0
        self._last_bgr = None
        self._slider.setMaximum(0)
        self._display.setText("No clip loaded")
        self._frame_label.setText("Frame: 0 / 0")
        self._set_controls_enabled(False)

    @property
    def current_frame(self) -> int:
        return self._cur_frame

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def seek(self, frame: int) -> None:
        if self._cap is None:
            return
        frame = max(0, min(frame, self._n_frames - 1))
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, bgr = self._cap.read()
        if ret:
            self._cur_frame = frame
            self._render(bgr)
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._frame_label.setText(f"Frame: {frame} / {self._n_frames - 1}")
        self.frame_changed.emit(frame)

    def toggle_play(self) -> None:
        if self._cap is None:
            return
        self._playing = not self._playing
        self._play_btn.setText("⏸" if self._playing else "▶")
        if self._playing:
            interval = max(1, int(1000 / (self._fps * self._speed_multiplier)))
            self._timer.start(interval)
        else:
            self._timer.stop()

    def stop(self) -> None:
        if self._playing:
            self.toggle_play()

    def _advance(self) -> None:
        if self._cur_frame >= self._n_frames - 1:
            if self._loop_enabled:
                self.seek(0)
            else:
                self.toggle_play()
            return
        self.seek(self._cur_frame + 1)

    def _on_loop_toggled(self, checked: bool) -> None:
        self._loop_enabled = checked

    def _on_speed_changed(self, index: int) -> None:
        if 0 <= index < len(_SPEED_VALUES):
            self._speed_multiplier = _SPEED_VALUES[index]
        if self._playing:
            interval = max(1, int(1000 / (self._fps * self._speed_multiplier)))
            self._timer.start(interval)

    def _on_slider_moved(self, value: int) -> None:
        if self._cap is not None:
            self.seek(value)

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self._last_bgr is not None:
            self._render(self._last_bgr)
        super().resizeEvent(event)

    def _render(self, bgr) -> None:
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        self._last_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        dw = max(1, self._display.width())
        dh = max(1, self._display.height())
        pix = QPixmap.fromImage(qimg).scaled(
            dw,
            dh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._display.setPixmap(pix)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self._play_btn, self._prev_btn, self._next_btn, self._slider):
            w.setEnabled(enabled)

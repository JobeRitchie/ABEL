"""Overlay appearance customisation dialog with live video-frame preview."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Default overlay settings (also used as factory fallback)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    # Panel geometry
    "font_scale_factor": 1.0,        # multiplier on auto-computed font scale
    "row_spacing_factor": 1.0,       # multiplier on row height
    "dot_radius_offset": 0,          # pixels added to auto-computed dot radius
    "panel_opacity": 0.72,           # 0-1, background darkness blend
    "panel_border_gray": 60,         # 0-255, border brightness

    # Active-behaviour highlighting
    "highlight_intensity": 0.30,     # 0-1, colour tint blend
    "active_glow_ring_px": 4,        # extra radius for glow ring
    "accent_bar_enabled": True,      # show coloured bar under active row
    "accent_bar_height_factor": 0.18,  # fraction of row height

    # Text colours (BGR lists for JSON compat)
    "active_text_color": [255, 255, 255],
    "inactive_text_color": [160, 160, 160],

    # Text weight
    "text_thickness_offset": 0,       # added to auto-computed thickness (-2..+4)

    # Basic-mode label
    "basic_label_scale_factor": 1.0,

    # Keypoint dots
    "keypoint_radius_offset": 0,          # pixels added to auto-computed radius
    "keypoint_outline_enabled": True,      # draw black outline around dots
    "keypoint_confidence_threshold": 0.20, # 0-1, minimum confidence to draw
}

CONFIG_FILENAME = "overlay_appearance.json"


def _defaults() -> dict[str, Any]:
    """Return a fresh copy of the default settings."""
    return json.loads(json.dumps(_DEFAULTS))


def load_overlay_settings(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / CONFIG_FILENAME
    defaults = _defaults()
    if not path.exists():
        return defaults
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            defaults.update(stored)
    except Exception:
        pass
    return defaults


def save_overlay_settings(project_root: Path, settings: dict[str, Any]) -> None:
    cfg_dir = project_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / CONFIG_FILENAME
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper: grab a sample frame from a project video
# ---------------------------------------------------------------------------

def _grab_sample_frame(project_root: Path) -> np.ndarray | None:
    """Return a single BGR frame from a random project video, or *None*."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None

    from abel.services.import_service import ImportService  # noqa: PLC0415

    svc = ImportService()
    manifest = svc.load_manifest(project_root)
    if manifest is None or not manifest.linked_sessions:
        return None

    sessions = list(manifest.linked_sessions)
    random.shuffle(sessions)
    for session in sessions:
        vpath = svc.video_path_for_session(manifest, session.session_id)
        if vpath is None or not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 10:
            cap.release()
            continue
        target = random.randint(total // 4, 3 * total // 4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
    return None


def _numpy_to_qpixmap(bgr: np.ndarray, max_width: int = 720) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap, optionally downscaling."""
    import cv2  # noqa: PLC0415

    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# The dialog itself
# ---------------------------------------------------------------------------

class OverlaySettingsDialog(QDialog):
    """Live-preview overlay customisation dialog."""

    settings_saved = Signal()

    def __init__(
        self,
        project_root: Path,
        behavior_info: dict[str, dict],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay Appearance Settings")
        self.resize(980, 720)

        self._project_root = project_root
        self._behavior_info = behavior_info
        self._settings = load_overlay_settings(project_root)
        self._sample_frame: np.ndarray | None = _grab_sample_frame(project_root)

        # Debounce timer so rapid control changes don't freeze the UI
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._refresh_preview)

        # -- Preview area (left/top) --
        self._preview_label = QLabel("Loading preview…")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(480, 300)
        self._preview_label.setStyleSheet("background: #111; border: 1px solid #333;")

        # -- Controls (right/bottom) --
        controls = self._build_controls()
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(320)
        controls_scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # -- Action buttons --
        self._save_btn = QPushButton("Save Settings")
        self._save_btn.setStyleSheet("font-weight: bold;")
        self._save_btn.clicked.connect(self._save)

        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._reset)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(close_btn)

        # -- Layout --
        top = QHBoxLayout()
        top.addWidget(self._preview_label, 3)
        top.addWidget(controls_scroll, 2)

        root = QVBoxLayout(self)
        root.addLayout(top, 1)
        root.addLayout(btn_row)

        self._refresh_preview()

    # ----- control builders -----

    def _build_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Panel section ---
        panel_group = QGroupBox("Panel")
        panel_form = QFormLayout(panel_group)

        self._font_scale = self._dbl_spin(0.50, 2.00, self._settings["font_scale_factor"], 0.05)
        self._font_scale.valueChanged.connect(lambda v: self._set("font_scale_factor", v))
        panel_form.addRow("Font scale:", self._font_scale)

        self._row_spacing = self._dbl_spin(0.60, 2.00, self._settings["row_spacing_factor"], 0.05)
        self._row_spacing.valueChanged.connect(lambda v: self._set("row_spacing_factor", v))
        panel_form.addRow("Row spacing:", self._row_spacing)

        self._dot_offset = self._int_spin(-4, 12, self._settings["dot_radius_offset"])
        self._dot_offset.valueChanged.connect(lambda v: self._set("dot_radius_offset", v))
        panel_form.addRow("Dot size offset (px):", self._dot_offset)

        self._panel_opacity = self._slider(0, 100, int(self._settings["panel_opacity"] * 100))
        self._panel_opacity.valueChanged.connect(lambda v: self._set("panel_opacity", v / 100.0))
        opacity_row = QHBoxLayout()
        self._opacity_label = QLabel(f"{self._settings['panel_opacity']:.0%}")
        opacity_row.addWidget(self._panel_opacity, 1)
        opacity_row.addWidget(self._opacity_label)
        panel_form.addRow("Background opacity:", opacity_row)

        self._border_gray = self._int_spin(0, 255, self._settings["panel_border_gray"])
        self._border_gray.valueChanged.connect(lambda v: self._set("panel_border_gray", v))
        panel_form.addRow("Border brightness:", self._border_gray)

        self._thickness_offset = self._int_spin(-2, 4, self._settings.get("text_thickness_offset", 0))
        self._thickness_offset.valueChanged.connect(lambda v: self._set("text_thickness_offset", v))
        panel_form.addRow("Text boldness:", self._thickness_offset)

        layout.addWidget(panel_group)

        # --- Active highlighting section ---
        hl_group = QGroupBox("Active Highlighting")
        hl_form = QFormLayout(hl_group)

        self._hl_intensity = self._slider(0, 100, int(self._settings["highlight_intensity"] * 100))
        self._hl_intensity.valueChanged.connect(lambda v: self._set("highlight_intensity", v / 100.0))
        hl_row = QHBoxLayout()
        self._hl_label = QLabel(f"{self._settings['highlight_intensity']:.0%}")
        hl_row.addWidget(self._hl_intensity, 1)
        hl_row.addWidget(self._hl_label)
        hl_form.addRow("Highlight intensity:", hl_row)

        self._glow_ring = self._int_spin(0, 12, self._settings["active_glow_ring_px"])
        self._glow_ring.valueChanged.connect(lambda v: self._set("active_glow_ring_px", v))
        hl_form.addRow("Glow ring (px):", self._glow_ring)

        self._accent_bar_cb = QCheckBox("Show accent bar")
        self._accent_bar_cb.setChecked(bool(self._settings["accent_bar_enabled"]))
        self._accent_bar_cb.toggled.connect(lambda v: self._set("accent_bar_enabled", v))
        hl_form.addRow(self._accent_bar_cb)

        self._accent_bar_h = self._dbl_spin(0.05, 0.50, self._settings["accent_bar_height_factor"], 0.02)
        self._accent_bar_h.valueChanged.connect(lambda v: self._set("accent_bar_height_factor", v))
        hl_form.addRow("Accent bar height:", self._accent_bar_h)

        layout.addWidget(hl_group)

        # --- Text colours section ---
        clr_group = QGroupBox("Text Colours")
        clr_form = QFormLayout(clr_group)

        self._active_clr_btn = self._color_button(self._settings["active_text_color"])
        self._active_clr_btn.clicked.connect(lambda: self._pick_color("active_text_color", self._active_clr_btn))
        clr_form.addRow("Active text:", self._active_clr_btn)

        self._inactive_clr_btn = self._color_button(self._settings["inactive_text_color"])
        self._inactive_clr_btn.clicked.connect(lambda: self._pick_color("inactive_text_color", self._inactive_clr_btn))
        clr_form.addRow("Inactive text:", self._inactive_clr_btn)

        layout.addWidget(clr_group)

        # --- Basic mode section ---
        basic_group = QGroupBox("Basic Mode")
        basic_form = QFormLayout(basic_group)

        self._basic_scale = self._dbl_spin(0.50, 2.00, self._settings["basic_label_scale_factor"], 0.05)
        self._basic_scale.valueChanged.connect(lambda v: self._set("basic_label_scale_factor", v))
        basic_form.addRow("Label scale:", self._basic_scale)

        layout.addWidget(basic_group)

        # --- Keypoint dots section ---
        kp_group = QGroupBox("Keypoint Dots")
        kp_form = QFormLayout(kp_group)

        self._kp_radius_offset = self._int_spin(-4, 16, self._settings.get("keypoint_radius_offset", 0))
        self._kp_radius_offset.valueChanged.connect(lambda v: self._set("keypoint_radius_offset", v))
        kp_form.addRow("Radius offset (px):", self._kp_radius_offset)

        self._kp_outline_cb = QCheckBox("Draw outline")
        self._kp_outline_cb.setChecked(bool(self._settings.get("keypoint_outline_enabled", True)))
        self._kp_outline_cb.toggled.connect(lambda v: self._set("keypoint_outline_enabled", v))
        kp_form.addRow(self._kp_outline_cb)

        self._kp_conf = self._dbl_spin(0.00, 1.00, self._settings.get("keypoint_confidence_threshold", 0.20), 0.05)
        self._kp_conf.valueChanged.connect(lambda v: self._set("keypoint_confidence_threshold", v))
        kp_form.addRow("Confidence threshold:", self._kp_conf)

        layout.addWidget(kp_group)

        layout.addStretch()
        return container

    # ----- widget factory helpers -----

    @staticmethod
    def _dbl_spin(lo: float, hi: float, val: float, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.setDecimals(2)
        sb.setMinimumWidth(90)
        sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        return sb

    @staticmethod
    def _int_spin(lo: int, hi: int, val: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setMinimumWidth(90)
        sb.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        return sb

    @staticmethod
    def _slider(lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    @staticmethod
    def _color_button(bgr_list: list[int]) -> QPushButton:
        r, g, b = bgr_list[2], bgr_list[1], bgr_list[0]
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #666;")
        return btn

    # ----- actions -----

    def _set(self, key: str, value: Any) -> None:
        self._settings[key] = value
        # Keep percentage labels in sync
        if key == "panel_opacity":
            self._opacity_label.setText(f"{value:.0%}")
        elif key == "highlight_intensity":
            self._hl_label.setText(f"{value:.0%}")
        # Debounced preview so rapid arrow clicks stay responsive
        self._debounce.start()

    def _pick_color(self, key: str, btn: QPushButton) -> None:
        cur = self._settings[key]
        initial = QColorDialog.getColor(
            initial=QColor(cur[2], cur[1], cur[0]),
            parent=self,
            title=f"Choose colour for {key.replace('_', ' ')}",
        )
        if not initial.isValid():
            return
        bgr = [initial.blue(), initial.green(), initial.red()]
        self._settings[key] = bgr
        r, g, b = initial.red(), initial.green(), initial.blue()
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #666;")
        self._refresh_preview()

    def _save(self) -> None:
        save_overlay_settings(self._project_root, self._settings)
        self.settings_saved.emit()
        QMessageBox.information(self, "Saved", "Overlay appearance settings saved.")

    def _reset(self) -> None:
        self._settings = _defaults()
        # Update all widgets to default values
        self._font_scale.setValue(self._settings["font_scale_factor"])
        self._row_spacing.setValue(self._settings["row_spacing_factor"])
        self._dot_offset.setValue(self._settings["dot_radius_offset"])
        self._panel_opacity.setValue(int(self._settings["panel_opacity"] * 100))
        self._opacity_label.setText(f"{self._settings['panel_opacity']:.0%}")
        self._border_gray.setValue(self._settings["panel_border_gray"])
        self._hl_intensity.setValue(int(self._settings["highlight_intensity"] * 100))
        self._hl_label.setText(f"{self._settings['highlight_intensity']:.0%}")
        self._glow_ring.setValue(self._settings["active_glow_ring_px"])
        self._accent_bar_cb.setChecked(self._settings["accent_bar_enabled"])
        self._accent_bar_h.setValue(self._settings["accent_bar_height_factor"])
        self._basic_scale.setValue(self._settings["basic_label_scale_factor"])
        self._thickness_offset.setValue(self._settings.get("text_thickness_offset", 0))
        self._kp_radius_offset.setValue(self._settings.get("keypoint_radius_offset", 0))
        self._kp_outline_cb.setChecked(bool(self._settings.get("keypoint_outline_enabled", True)))
        self._kp_conf.setValue(self._settings.get("keypoint_confidence_threshold", 0.20))
        self._update_color_button(self._active_clr_btn, self._settings["active_text_color"])
        self._update_color_button(self._inactive_clr_btn, self._settings["inactive_text_color"])
        self._refresh_preview()

    @staticmethod
    def _update_color_button(btn: QPushButton, bgr: list[int]) -> None:
        r, g, b = bgr[2], bgr[1], bgr[0]
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #666;")

    # ----- preview rendering -----

    def _refresh_preview(self) -> None:
        if self._sample_frame is None:
            self._preview_label.setText("No video found in project.\nSettings will still be saved.")
            return
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            self._preview_label.setText("OpenCV not available for preview.")
            return

        frame = self._sample_frame.copy()

        # Build synthetic data so the overlay looks realistic
        names = sorted(self._behavior_info.keys(), key=str.lower) or ["Behavior A", "Behavior B", "Behavior C"]
        # Simulate: first behaviour is "active"
        active = [names[0]] if names else []
        cumulative = {}
        for i, name in enumerate(names):
            cumulative[name] = random.randint(30 * (i + 1), 30 * (i + 1) + 600)

        info = dict(self._behavior_info)
        if not info:
            palette = [(60, 180, 255), (80, 220, 80), (255, 200, 70), (200, 120, 255), (255, 110, 110)]
            for i, n in enumerate(names):
                info[n] = {"color_bgr": palette[i % len(palette)], "threshold": 0.5, "short_name": n}

        # Draw synthetic keypoint dots so the user can see their settings
        h, w = frame.shape[:2]
        kp_base_r = max(2, int(round(h / 250)))
        kp_offset = int(self._settings.get("keypoint_radius_offset", 0))
        kp_r = max(1, kp_base_r + kp_offset)
        kp_outline = bool(self._settings.get("keypoint_outline_enabled", True))
        kp_t = max(1, kp_r // 2)
        palette = [
            (60, 180, 255), (80, 220, 80), (255, 200, 70),
            (200, 120, 255), (255, 110, 110), (220, 220, 220),
            (255, 170, 0), (180, 255, 255),
        ]
        # Scatter keypoints in the centre-ish area of the frame
        rng = random.Random(42)  # deterministic so dots don't jump on refresh
        for i in range(12):
            px = int(w * rng.uniform(0.25, 0.75))
            py = int(h * rng.uniform(0.20, 0.70))
            cv2.circle(frame, (px, py), kp_r, palette[i % len(palette)], -1, cv2.LINE_AA)
            if kp_outline:
                cv2.circle(frame, (px, py), kp_r, (0, 0, 0), kp_t, cv2.LINE_AA)

        from abel.services.export_service import ExportService  # noqa: PLC0415
        ExportService._draw_advanced_overlay(
            frame, 30.0, names, active, cumulative, info,
            overlay_settings=self._settings,
        )

        pix = _numpy_to_qpixmap(frame, max_width=self._preview_label.width() or 720)
        self._preview_label.setPixmap(pix)

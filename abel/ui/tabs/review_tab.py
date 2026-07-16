"""Comprehensive Review Tab for candidate clip evaluation.

Displays candidate clips with:
- Video playback with frame stepping
- Keypoint overlay (optional)
- Score display
- Seed examples side-by-side
- Quick decision buttons
- Keyboard shortcuts for rapid review

Pipeline position:
    Candidate Retrieval → Clip Extraction
    → **Review Interface** ← here
    → Human Decision → Active Learning
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import (
    CandidateWindow,
    ReviewDecision,
    ReviewDecisionType,
    ReviewerLabelRecord,
)
from abel.services.behavior_service import BehaviorService
from abel.services.candidate_service import CandidateGenerationService
from abel.services.import_service import ImportService
from abel.services.preprocessing_service import ClipExtractionService, regenerate_clips_for_windows
from abel.services.review_service import ReviewService
from abel.services.dissimilarity_service import run_dissimilarity_analysis, DissimilarityReport
from abel.services.clip_metrics_service import (
    ClipMetricsService,
    ClipRef,
    Criterion,
    EssenceCheckResult,
)
from abel.ui.clip_mining_dialog import ClipMiningDialog
from abel.storage.file_store import read_yaml
from abel.workers.task_worker import TaskWorker

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("abel")


NO_BEHAVIOR_ID = "no_behavior"

# Distinct, deterministic BGR colors for pose body-part dots (matches the
# labeled-video export palette in ExportService._part_color).
_KEYPOINT_PALETTE = [
    (60, 180, 255),
    (80, 220, 80),
    (255, 200, 70),
    (200, 120, 255),
    (255, 110, 110),
    (220, 220, 220),
    (255, 170, 0),
    (180, 255, 255),
]
UNASSIGNED_BEHAVIOR_ID = "__unassigned__"


@dataclass
class _ReviewListRow:
    """Unified row model for current candidates and historic reviewed entries."""

    window_id: str
    session_id: str
    behavior_id: str | None
    start_frame: int
    end_frame: int
    total_score: float = 0.0
    clip_path: str | None = None


# ---------------------------------------------------------------------------
# Video player widget for candidate clips
# ---------------------------------------------------------------------------

class CandidateVideoPlayer(QWidget):
    """Minimal frame-by-frame video player for candidate clips."""

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
        self._show_keypoints = False
        self._last_bgr = None  # cached for re-render on resize
        # Optional pose overlay data.  When set, the "Show Keypoints" button
        # draws body-part dots on each frame.  Coordinates are indexed by the
        # video's own frame number plus _pose_frame_offset (0 when the player is
        # showing a full session whose frame indices match the pose arrays).
        self._pose_x: "np.ndarray | None" = None
        self._pose_y: "np.ndarray | None" = None
        self._pose_conf: "np.ndarray | None" = None
        self._pose_frame_offset = 0
        self._pose_conf_thresh = 0.20
        self._loop_enabled = False
        self._speed_multiplier = 1.0
        _SPEED_LABELS = ["0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x"]
        self._SPEED_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

        # Display
        self._display = QLabel("No clip loaded")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(320, 240)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setStyleSheet("background: #0a0e18; color: #546e7a;")

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)

        # Controls
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

        self._keypoints_chk = QPushButton("Show Keypoints")
        self._keypoints_chk.setCheckable(True)
        self._keypoints_chk.setFixedWidth(100)
        self._keypoints_chk.setToolTip("Overlay pose keypoints on the video (requires pose data).")
        self._keypoints_chk.toggled.connect(self._on_keypoints_toggled)

        self._loop_chk = QCheckBox("Loop")
        self._loop_chk.setToolTip("Automatically restart the clip from the beginning when it ends")
        self._loop_chk.setChecked(False)
        self._loop_chk.toggled.connect(self._on_loop_toggled)

        self._speed_combo = QComboBox()
        for _lbl in _SPEED_LABELS:
            self._speed_combo.addItem(_lbl)
        self._speed_combo.setCurrentIndex(3)  # default 1x
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
        ctrl.addWidget(self._keypoints_chk)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display, 1)
        layout.addLayout(ctrl)

        self._set_controls_enabled(False)

    def load_clip(self, path: str) -> bool:
        """Load a video clip."""
        self.close_clip()
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            self._display.setText("OpenCV not installed.\nCannot preview video.")
            return False

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._display.setText(f"Cannot open:\n{path}")
            return False

        self._cap = cap
        self._n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._slider.setMaximum(self._n_frames - 1)
        self._set_controls_enabled(True)
        self.seek(0)
        return True

    def close_clip(self) -> None:
        """Close the current clip."""
        self._playing = False
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._n_frames = 0
        self._cur_frame = 0
        self._slider.setMaximum(0)
        self._display.setText("No clip loaded")
        self._frame_label.setText("Frame: 0 / 0")
        self._set_controls_enabled(False)
        self.clear_pose_overlay()

    @property
    def current_frame(self) -> int:
        return self._cur_frame

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def seek(self, frame: int) -> None:
        """Seek to a specific frame."""
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
        """Toggle playback."""
        if self._cap is None:
            return
        self._playing = not self._playing
        self._play_btn.setText("⏸" if self._playing else "▶")
        if self._playing:
            interval = max(1, int(1000 / (self._fps * self._speed_multiplier)))
            self._timer.start(interval)
        else:
            self._timer.stop()

    def _advance(self) -> None:
        """Advance to next frame during playback."""
        if self._cur_frame >= self._n_frames - 1:
            if self._loop_enabled:
                self.seek(0)
            else:
                self.toggle_play()
            return
        self.seek(self._cur_frame + 1)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Enable or disable clip looping."""
        self._loop_enabled = checked

    def _on_speed_changed(self, index: int) -> None:
        """Update playback speed and restart the timer if currently playing."""
        if 0 <= index < len(self._SPEED_VALUES):
            self._speed_multiplier = self._SPEED_VALUES[index]
        if self._playing:
            interval = max(1, int(1000 / (self._fps * self._speed_multiplier)))
            self._timer.start(interval)

    def _on_slider_moved(self, value: int) -> None:
        """Handle slider movement."""
        if self._cap is not None:
            self.seek(value)

    def resizeEvent(self, event) -> None:
        """Re-render the current frame at the new widget size."""
        if self._last_bgr is not None:
            self._render(self._last_bgr)
        super().resizeEvent(event)

    def _on_keypoints_toggled(self, checked: bool) -> None:
        """Toggle the pose-keypoint overlay and redraw the current frame."""
        self._show_keypoints = bool(checked)
        if self._last_bgr is not None:
            self._render(self._last_bgr)

    def set_pose_overlay(
        self,
        x_vals,
        y_vals,
        conf_vals,
        conf_thresh: float = 0.20,
        frame_offset: int = 0,
    ) -> None:
        """Attach pose data so the "Show Keypoints" button can draw body parts.

        *x_vals*, *y_vals*, *conf_vals* are ``(n_frames, n_parts)`` arrays.  Frame
        ``f`` of the loaded video is drawn using row ``f + frame_offset`` (use 0
        when the player shows a full session that shares the pose indexing).
        """
        self._pose_x = x_vals
        self._pose_y = y_vals
        self._pose_conf = conf_vals
        self._pose_conf_thresh = float(conf_thresh)
        self._pose_frame_offset = int(frame_offset)
        if self._last_bgr is not None:
            self._render(self._last_bgr)

    def clear_pose_overlay(self) -> None:
        """Drop any attached pose data (keypoints can no longer be drawn)."""
        self._pose_x = None
        self._pose_y = None
        self._pose_conf = None
        self._pose_frame_offset = 0

    def _draw_keypoints(self, bgr, frame_idx: int):
        """Return a copy of *bgr* with pose dots drawn for *frame_idx*."""
        if self._pose_x is None or self._pose_y is None or self._pose_conf is None:
            return bgr
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:
            return bgr
        row = frame_idx + self._pose_frame_offset
        if row < 0 or row >= self._pose_x.shape[0]:
            return bgr
        n_parts = min(self._pose_x.shape[1], self._pose_y.shape[1], self._pose_conf.shape[1])
        if n_parts <= 0:
            return bgr
        h = bgr.shape[0]
        radius = max(2, int(round(h / 250)))
        thickness = max(1, radius // 2)
        out = bgr.copy()
        xr = self._pose_x[row]
        yr = self._pose_y[row]
        cr = self._pose_conf[row]
        for p in range(n_parts):
            conf = cr[p]
            if not np.isfinite(conf) or conf < self._pose_conf_thresh:
                continue
            x = xr[p]
            y = yr[p]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            color = _KEYPOINT_PALETTE[p % len(_KEYPOINT_PALETTE)]
            center = (int(round(x)), int(round(y)))
            cv2.circle(out, center, radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, center, radius, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        return out

    def _render(self, bgr) -> None:
        """Render frame to display, scaled to the current display area."""
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return
        self._last_bgr = bgr
        if self._show_keypoints:
            bgr = self._draw_keypoints(bgr, self._cur_frame)
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
        """Enable/disable controls."""
        for w in (self._play_btn, self._prev_btn, self._next_btn, self._slider):
            w.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Bulk assign / reassign dialog
# ---------------------------------------------------------------------------


class _BulkAssignDialog(QDialog):
    """Pick one behavior (or several, in co-occurring mode) to assign to a batch."""

    def __init__(
        self,
        behaviors: "list[tuple[str, str]]",
        allow_multi: bool,
        n_clips: int,
        preselect: "set[str] | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bulk Assign Behavior")
        self.setMinimumWidth(340)
        self._allow_multi = allow_multi
        self._buttons: list[tuple[str, QWidget]] = []
        self._selected: list[str] = []
        preselect = preselect or set()

        layout = QVBoxLayout(self)
        header = QLabel(f"Assign a behavior to {n_clips} clip(s):")
        header.setStyleSheet("font-weight: 600;")
        layout.addWidget(header)
        hint = QLabel(
            "Co-occurring mode — select one or more behaviors."
            if allow_multi
            else "Select one behavior."
        )
        hint.setStyleSheet("color: #607D8B; font-size: 11px;")
        layout.addWidget(hint)

        host = QWidget()
        vb = QVBoxLayout(host)
        vb.setContentsMargins(6, 6, 6, 6)
        vb.setSpacing(4)
        group = QButtonGroup(self)
        group.setExclusive(not allow_multi)
        for bid, name in behaviors:
            btn = QCheckBox(name) if allow_multi else QRadioButton(name)
            if bid in preselect:
                btn.setChecked(True)
            group.addButton(btn)
            vb.addWidget(btn)
            self._buttons.append((bid, btn))
        vb.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setMinimumHeight(min(300, 80 + 26 * len(behaviors)))
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        selected = [bid for bid, btn in self._buttons if btn.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No Behavior", "Select at least one behavior to assign.")
            return
        self._selected = selected
        self.accept()

    def selected_ids(self) -> list[str]:
        return list(self._selected)


# ---------------------------------------------------------------------------
# Main Review Tab
# ---------------------------------------------------------------------------

class ReviewTab(QWidget):
    """Comprehensive review interface for candidate clip evaluation."""

    # Friendly labels for the CandidateWindow.source provenance field, shown in
    # the "Source" column. Anything not listed falls back to a title-cased form of
    # the raw source, or to the active-learning selection_reason when no source was
    # recorded (the common candidate-generation case, whose source is left blank).
    _SOURCE_DISPLAY_LABELS = {
        "active_learning_uncertainty": "Active Learning",
        "umap_interactive_selection": "UMAP Selection",
        "umap_selection": "UMAP Selection",
        "temporal_bout_review": "Temporal Bouts",
        "quality_check": "Quality Check",
        "clip_mining": "Clip Mining",
    }

    # Cross-thread progress relay for background clip regeneration.
    _regen_progress = Signal(int, int)
    # Cross-thread progress relay for extracting mined clips into the queue.
    _mining_extract_progress = Signal(int, int)

    def __init__(
        self,
        review_service: ReviewService,
        candidate_service: CandidateGenerationService,
        import_service: ImportService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._review_service = review_service
        self._candidate_service = candidate_service
        self._imports = import_service
        self._behavior_service = behavior_service
        self._project_root: Path | None = None
        self._all_candidates = []
        self._visible_candidates = []
        self._decision_by_clip_id: dict[str, ReviewDecision] = {}
        self._current_candidate_idx = -1
        self._session_order_index: dict[str, int] = {}
        self._display_subject_map: dict[str, str] = {}
        self._display_behavior_name_map: dict[str, str] = {}
        self._display_occurrence: dict[str, int] = {}
        self._behavior_shortcuts: list[QShortcut] = []
        self._soundboard = None  # lazily-created BehaviorSoundboard window
        # Structured multi-animal labels captured via the soundboard, keyed by
        # window_id. In-memory for now (Phase 2b: persist to reviewer_labels).
        self._structured_labels: dict[str, list[dict]] = {}
        self._pool = QThreadPool.globalInstance()
        self._dissimilarity_scores: dict[str, float] = {}
        # window_id -> reasons a clip failed the essence-range test (Targeted Clip
        # Mining "Flag failing clips in review queue"). Highlights the current
        # queue's out-of-range clips; cleared implicitly on the next re-filter.
        self._essence_fail_violations: dict[str, list[str]] = {}
        self._essence_check_busy = False
        self._al_fp_ids: set[str] = set()
        self._al_fn_ids: set[str] = set()
        # Targeted Clip Mining: when non-None, the queue is narrowed to these
        # window IDs (ranked by their match score), independent of other filters.
        self._mined_ids: set[str] | None = None
        self._mined_scores: dict[str, float] = {}
        self._mining_dialog = None  # modeless ClipMiningDialog, when open
        # Background extraction of mined windows into the review queue.
        self._mining_extract_busy = False
        self._mining_progress_dialog = None
        self._pending_mined_ids: set[str] = set()
        self._pending_mined_scores: dict[str, float] = {}
        # Maps segment_id → list of target behavior IDs for which this segment is FP or FN.
        # A segment can be FP for one behavior and FN for a *different* behavior simultaneously.
        self._al_fp_behavior_map: dict[str, list[str]] = {}
        self._al_fn_behavior_map: dict[str, list[str]] = {}
        # Segments whose latest reviewer label was written AFTER the model predictions were
        # generated — i.e., they have already been re-reviewed in this FP/FN context.
        # Populated from timestamps on every _apply_filter call; persistent across restarts.
        self._al_post_pred_reviewed_ids: set[str] = set()

        # Co-occurring behavior mode state
        self._co_occurring_enabled: bool = False
        self._pending_labels: set[str] = set()

        # Arrow-key guard: track whether the user has explicitly interacted with
        # any decision/label control for the current clip.  Only save on navigation
        # when this is True so unvisited clips are never accidentally labelled.
        self._review_dirty: bool = False
        self._loading_candidate: bool = False

        self._empty_label = QLabel("Open a project, generate candidates, and extract clips to start review.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        self._candidate_label = QLabel("No candidates loaded")
        self._candidate_label.setStyleSheet("font-weight: 600;")

        self._summary_label = QLabel("Decisions: 0")
        self._summary_label.setStyleSheet("font-size: 11px; color: #78909C;")

        # ── Filter popup panel ──────────────────────────────────────────────
        self._filter_panel = QFrame(self, Qt.WindowType.Popup)
        self._filter_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self._filter_panel.setFrameShadow(QFrame.Shadow.Raised)
        self._filter_panel.setStyleSheet(
            "QFrame { background: #263238; border: 1px solid #546E7A; border-radius: 6px; padding: 4px; }"
        )

        panel_layout = QVBoxLayout(self._filter_panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(10, 8, 10, 10)

        _panel_title = QLabel("Display Filters")
        _panel_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #CFD8DC; margin-bottom: 2px;")
        panel_layout.addWidget(_panel_title)

        _sep1 = QFrame()
        _sep1.setFrameShape(QFrame.Shape.HLine)
        _sep1.setStyleSheet("color: #546E7A;")
        panel_layout.addWidget(_sep1)

        self._show_reviewed_chk = QCheckBox("Show reviewed candidates")
        self._show_reviewed_chk.setChecked(False)
        self._show_reviewed_chk.setToolTip("Include already-reviewed candidates in the list.")
        self._show_reviewed_chk.toggled.connect(self._apply_filter)
        panel_layout.addWidget(self._show_reviewed_chk)

        self._reviewed_with_clips_btn = QPushButton("Reviewed + Clips Only")
        self._reviewed_with_clips_btn.setCheckable(True)
        self._reviewed_with_clips_btn.setChecked(False)
        self._reviewed_with_clips_btn.setToolTip(
            "Show only reviewed rows with an available clip file. "
            "Reviewed rows missing clips are excluded."
        )
        self._reviewed_with_clips_btn.toggled.connect(self._apply_filter)
        panel_layout.addWidget(self._reviewed_with_clips_btn)

        self._soundboard_btn = QPushButton("🎹 Behavior Soundboard")
        self._soundboard_btn.setToolTip(
            "Open a pop-out window with one button per behavior for labeling the "
            "current clip. Arrow keys / Space / Enter still work while it's focused."
        )
        self._soundboard_btn.clicked.connect(self._open_soundboard)
        panel_layout.addWidget(self._soundboard_btn)

        self._show_missing_clips_chk = QCheckBox("Show candidates with missing clips")
        self._show_missing_clips_chk.setChecked(False)
        self._show_missing_clips_chk.setToolTip(
            "When enabled, include candidates whose extracted clip file is missing. "
            "Reviewed rows are always shown when 'Show reviewed' is enabled."
        )
        self._show_missing_clips_chk.toggled.connect(self._apply_filter)
        panel_layout.addWidget(self._show_missing_clips_chk)

        _sep2 = QFrame()
        _sep2.setFrameShape(QFrame.Shape.HLine)
        _sep2.setStyleSheet("color: #546E7A;")
        panel_layout.addWidget(_sep2)

        self._show_fp_fn_btn = QPushButton("Temporal Bout Review Clips")
        self._show_fp_fn_btn.setCheckable(True)
        self._show_fp_fn_btn.setChecked(False)
        self._show_fp_fn_btn.setToolTip(
            "Show only clips generated from detected temporal bouts. "
            "Use the Review label combo and decision buttons to label each clip."
        )
        self._show_fp_fn_btn.toggled.connect(self._apply_filter)
        panel_layout.addWidget(self._show_fp_fn_btn)

        self._show_al_fp_fn_btn = QPushButton("Active Learning FP / FN")
        self._show_al_fp_fn_btn.setCheckable(True)
        self._show_al_fp_fn_btn.setChecked(False)
        self._show_al_fp_fn_btn.setToolTip(
            "Show active-learning candidates that are likely mislabeled:\n"
            "  • FP (False Positive) — you previously labeled this segment as negative,\n"
            "    but the model still scores it high (hard negative).\n"
            "  • FN (False Negative) — you previously labeled this segment as a positive\n"
            "    behavior, but the current model now gives it low probability.\n\n"
            "Only segments with prior human labels qualify. Unreviewed low-confidence\n"
            "candidates are NOT shown here."
        )
        self._show_al_fp_fn_btn.toggled.connect(self._apply_filter)
        panel_layout.addWidget(self._show_al_fp_fn_btn)

        self._al_fp_fn_status_label = QLabel("")
        self._al_fp_fn_status_label.setStyleSheet("color: #80CBC4; font-size: 11px; padding-left: 4px;")
        self._al_fp_fn_status_label.setVisible(False)
        panel_layout.addWidget(self._al_fp_fn_status_label)

        _close_row = QHBoxLayout()
        _close_row.addStretch()
        _close_panel_btn = QPushButton("Close")
        _close_panel_btn.setFixedWidth(70)
        _close_panel_btn.clicked.connect(self._filter_panel.hide)
        _close_row.addWidget(_close_panel_btn)
        panel_layout.addLayout(_close_row)

        self._filter_panel.adjustSize()
        self._filter_panel.hide()

        # ── Filters toggle button ────────────────────────────────────────────
        self._filters_btn = QPushButton("Filters")
        self._filters_btn.setToolTip("Open display filter options")
        self._filters_btn.setCheckable(False)
        self._filters_btn.clicked.connect(self._toggle_filter_panel)

        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Video order (continuous)", userData="video_order")
        self._sort_combo.addItem("Score (high to low)", userData="score_desc")
        self._sort_combo.addItem("Score (low to high)", userData="score_asc")
        self._sort_combo.addItem("Window start (ascending)", userData="start_asc")
        self._sort_combo.setCurrentIndex(max(0, self._sort_combo.findData("score_desc")))
        self._sort_combo.currentIndexChanged.connect(self._apply_filter)

        self._behavior_filter_combo = QComboBox()
        self._behavior_filter_combo.addItem("All behaviors", userData="all")
        self._behavior_filter_combo.currentIndexChanged.connect(self._apply_filter)

        self._reviewer_input = QLineEdit("reviewer")
        self._reviewer_input.setMaximumWidth(140)

        self._autoplay_chk = QCheckBox("Auto-advance on save")
        self._autoplay_chk.setChecked(True)
        self._autoplay_chk.setToolTip(
            "When checked, automatically advance to the next clip after saving a decision.\n"
            "Uncheck to stay on the current clip (useful for co-occurring behavior labeling)."
        )

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_candidates)
        self._prev_candidate_btn = QPushButton("Previous")
        self._prev_candidate_btn.clicked.connect(self._load_previous)
        self._next_candidate_btn = QPushButton("Next")
        self._next_candidate_btn.clicked.connect(self._load_next)

        top = QHBoxLayout()
        top.addWidget(self._candidate_label, 1)
        top.addWidget(self._filters_btn)
        top.addWidget(QLabel("Sort:"))
        top.addWidget(self._sort_combo)
        top.addWidget(QLabel("Behavior:"))
        top.addWidget(self._behavior_filter_combo)
        top.addWidget(QLabel("Reviewer:"))
        top.addWidget(self._reviewer_input)
        top.addWidget(self._autoplay_chk)
        top.addWidget(self._prev_candidate_btn)
        top.addWidget(self._next_candidate_btn)
        top.addWidget(self._refresh_btn)

        self._candidate_table = QTableWidget(0, 8)
        self._candidate_table.setHorizontalHeaderLabels(["Subject", "Behavior", "#", "Score", "Source", "Clip", "FP/FN", "Decision"])
        self._candidate_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._candidate_table.verticalHeader().setVisible(False)
        _hdr = self._candidate_table.horizontalHeader()
        _hdr.setStretchLastSection(True)
        _hdr.setSectionResizeMode(0, _hdr.ResizeMode.Interactive)
        _hdr.setSectionResizeMode(1, _hdr.ResizeMode.Stretch)
        _hdr.setSectionResizeMode(2, _hdr.ResizeMode.ResizeToContents)
        _hdr.setSectionResizeMode(3, _hdr.ResizeMode.ResizeToContents)
        _hdr.setSectionResizeMode(4, _hdr.ResizeMode.ResizeToContents)
        _hdr.setSectionResizeMode(5, _hdr.ResizeMode.ResizeToContents)
        _hdr.setSectionResizeMode(6, _hdr.ResizeMode.ResizeToContents)
        self._candidate_table.setColumnWidth(0, 120)
        self._candidate_table.itemSelectionChanged.connect(self._on_table_selection_changed)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Candidates:"))
        left_layout.addWidget(self._candidate_table, 1)
        # ── Compact action toolbar ─────────────────────────────────────────
        # Actions are grouped into short rows and pop-up menus instead of one
        # tall stack of full-width buttons.  Every former button is still
        # reachable; low-frequency and maintenance actions live under ▾ menus.
        def _expand(widget: QWidget) -> QWidget:
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return widget

        actions_box = QVBoxLayout()
        actions_box.setContentsMargins(0, 4, 0, 0)
        actions_box.setSpacing(4)

        _bulk_hdr = QLabel("Bulk actions (selected rows)")
        _bulk_hdr.setStyleSheet("font-size: 11px; font-weight: 600; color: #78909C;")
        actions_box.addWidget(_bulk_hdr)

        # Row 1 — decisions applied to the selected rows.
        self._accept_selected_btn = QPushButton("Accept")
        self._accept_selected_btn.setToolTip("Mark selected candidate rows as Accepted")
        self._accept_selected_btn.setStyleSheet(
            "background-color: #388E3C; color: white; font-weight: 600; padding: 4px;"
        )
        self._accept_selected_btn.clicked.connect(lambda: self._apply_batch_decision(ReviewDecisionType.ACCEPT))

        self._reject_selected_btn = QPushButton("Reject")
        self._reject_selected_btn.setToolTip("Mark selected candidate rows as Rejected")
        self._reject_selected_btn.setStyleSheet("padding: 4px;")
        self._reject_selected_btn.clicked.connect(lambda: self._apply_batch_decision(ReviewDecisionType.REJECT))

        self._remove_selected_btn = QPushButton("Remove")
        self._remove_selected_btn.setToolTip(
            "Permanently remove selected candidates from the review queue.\n"
            "Also deletes their clip files and any saved decisions."
        )
        self._remove_selected_btn.setStyleSheet(
            "background-color: #B71C1C; color: white; font-weight: 600; padding: 4px;"
        )
        self._remove_selected_btn.clicked.connect(self._remove_selected_candidates)

        _row1 = QHBoxLayout()
        _row1.setSpacing(4)
        _row1.addWidget(_expand(self._accept_selected_btn))
        _row1.addWidget(_expand(self._reject_selected_btn))
        _row1.addWidget(_expand(self._remove_selected_btn))
        actions_box.addLayout(_row1)

        # Row 2 — assign/relabel and cleanup grouped under pop-up menus.
        self._assign_menu_btn = QToolButton()
        self._assign_menu_btn.setText("Assign / Relabel")
        self._assign_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._assign_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._assign_menu_btn.setStyleSheet(
            "QToolButton{background-color:#6A1B9A;color:white;font-weight:600;"
            "padding:4px 8px;border-radius:3px;}"
        )
        self._assign_menu_btn.setToolTip(
            "Assign or relabel behaviors in bulk (accept every visible candidate, "
            "reassign selected rows to the Review label, or open the multi-behavior "
            "Bulk Assign dialog)."
        )
        _assign_menu = QMenu(self._assign_menu_btn)
        _act_accept_all = QAction("Accept All Visible", self._assign_menu_btn)
        _act_accept_all.setToolTip("Mark every currently visible candidate as Accepted")
        _act_accept_all.triggered.connect(self._accept_all)
        _assign_menu.addAction(_act_accept_all)
        _act_reassign = QAction("Reassign selected → Review label", self._assign_menu_btn)
        _act_reassign.setToolTip("Assign selected candidate rows to the behavior in 'Review label'.")
        _act_reassign.triggered.connect(self._reassign_selected)
        _assign_menu.addAction(_act_reassign)
        _act_bulk_assign = QAction("Bulk Assign / Reassign…", self._assign_menu_btn)
        _act_bulk_assign.setToolTip(
            "Pick the behavior(s) to assign to a batch of clips.\n"
            "Applies to selected rows, or the whole visible queue if none are selected.\n"
            "In co-occurring mode you can assign several behaviors at once."
        )
        _act_bulk_assign.triggered.connect(self._bulk_assign_behavior)
        _assign_menu.addAction(_act_bulk_assign)
        _assign_menu.setToolTipsVisible(True)
        self._assign_menu_btn.setMenu(_assign_menu)

        self._cleanup_menu_btn = QToolButton()
        self._cleanup_menu_btn.setText("Cleanup")
        self._cleanup_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._cleanup_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._cleanup_menu_btn.setStyleSheet("QToolButton{padding:4px 8px;border-radius:3px;}")
        self._cleanup_menu_btn.setToolTip("Queue and clip-file maintenance actions.")
        _cleanup_menu = QMenu(self._cleanup_menu_btn)
        _act_dismiss = QAction("Dismiss Undecided", self._cleanup_menu_btn)
        _act_dismiss.setToolTip(
            "Remove candidate entries that have no saved decision from the review queue.\n"
            "Does not delete clip files or affect any saved decisions."
        )
        _act_dismiss.triggered.connect(self._dismiss_undecided_candidates)
        _cleanup_menu.addAction(_act_dismiss)
        _act_clear_unrev = QAction("Clear Unreviewed Clips", self._cleanup_menu_btn)
        _act_clear_unrev.setToolTip(
            "Delete extracted clip files for candidates that do not have saved decisions."
        )
        _act_clear_unrev.triggered.connect(self._clear_unreviewed_clips)
        _cleanup_menu.addAction(_act_clear_unrev)
        _act_clear_bouts = QAction("Clear Bouts Missing Clips", self._cleanup_menu_btn)
        _act_clear_bouts.setToolTip(
            "Remove candidate bouts that do not have an extracted clip file."
        )
        _act_clear_bouts.triggered.connect(self._clear_bouts_missing_clips)
        _cleanup_menu.addAction(_act_clear_bouts)
        _act_remove_mined = QAction("Remove Mined Windows", self._cleanup_menu_btn)
        _act_remove_mined.setToolTip(
            "Remove candidates that were added by Targeted Clip Mining\n"
            "(and were not already part of the review queue)."
        )
        _act_remove_mined.triggered.connect(self._remove_mined_windows)
        _cleanup_menu.addAction(_act_remove_mined)
        _cleanup_menu.setToolTipsVisible(True)
        self._cleanup_menu_btn.setMenu(_cleanup_menu)

        _row2 = QHBoxLayout()
        _row2.setSpacing(4)
        _row2.addWidget(_expand(self._assign_menu_btn))
        _row2.addWidget(_expand(self._cleanup_menu_btn))
        actions_box.addLayout(_row2)

        # Row 3 — clip regeneration + outlier analysis.  These stay as real
        # buttons because their label doubles as a live progress indicator.
        self._regenerate_clips_btn = QPushButton("Regenerate Missing Clips")
        self._regenerate_clips_btn.setToolTip(
            "Re-extract clip files for candidates whose clips are missing.\n"
            "Uses the selected rows if any are selected, otherwise every candidate "
            "missing a clip. Extraction settings (preset, crop, padding) match the "
            "Clip Extraction tab."
        )
        self._regenerate_clips_btn.setStyleSheet("padding: 4px;")
        self._regenerate_clips_btn.clicked.connect(self._regenerate_missing_clips)
        self._regen_progress.connect(self._on_regen_progress)

        self._flag_outliers_btn = QPushButton("Flag Outliers (Dissimilarity)")
        self._flag_outliers_btn.setToolTip(
            "Compute dissimilarity scores for accepted clips of the currently filtered behavior.\n"
            "Outlier clips are highlighted in orange — they may warrant re-review."
        )
        self._flag_outliers_btn.setStyleSheet(
            "background-color: #5C6BC0; color: white; font-weight: 600; padding: 4px;"
        )
        self._flag_outliers_btn.clicked.connect(self._run_dissimilarity_analysis)

        _row3 = QHBoxLayout()
        _row3.setSpacing(4)
        _row3.addWidget(_expand(self._regenerate_clips_btn))
        _row3.addWidget(_expand(self._flag_outliers_btn))
        actions_box.addLayout(_row3)

        # Row 4 — Targeted Clip Mining (construction owned elsewhere; only the
        # container it is added to changed here).
        self._clip_mining_btn = QPushButton("Targeted Clip Mining…")
        self._clip_mining_btn.setToolTip(
            "Hunt for clips by pose criteria (nose past edge, tail position, speed…).\n"
            "Select exemplar rows first to auto-fill criteria via Extract Essence."
        )
        self._clip_mining_btn.setStyleSheet(
            "background-color: #00796B; color: white; font-weight: 600; padding: 4px;"
        )
        self._clip_mining_btn.clicked.connect(self._open_clip_mining)

        self._clear_mining_btn = QPushButton("Clear Mining Filter")
        self._clear_mining_btn.setToolTip("Return to the full candidate queue.")
        self._clear_mining_btn.clicked.connect(self._clear_mining_filter)
        self._clear_mining_btn.setVisible(False)

        _row4 = QHBoxLayout()
        _row4.setSpacing(4)
        _row4.addWidget(_expand(self._clip_mining_btn))
        _row4.addWidget(_expand(self._clear_mining_btn))
        actions_box.addLayout(_row4)

        left_layout.addLayout(actions_box)

        self._player = CandidateVideoPlayer()
        self._id_label = QLabel("Candidate ID: N/A")
        self._score_label = QLabel("Score: N/A")
        self._clip_label = QLabel("Clip: N/A")
        self._clip_label.setWordWrap(True)

        self._start_frame_spin = QSpinBox()
        self._start_frame_spin.setMinimum(0)
        self._start_frame_spin.setMaximum(10_000_000)
        self._start_frame_spin.setKeyboardTracking(False)
        self._start_frame_spin.valueChanged.connect(self._on_frame_override_changed)

        self._end_frame_spin = QSpinBox()
        self._end_frame_spin.setMinimum(0)
        self._end_frame_spin.setMaximum(10_000_000)
        self._end_frame_spin.setKeyboardTracking(False)
        self._end_frame_spin.valueChanged.connect(self._on_frame_override_changed)

        self._apply_frame_btn = QPushButton("Apply Frame Range")
        self._apply_frame_btn.clicked.connect(self._apply_frame_overrides)

        self._reset_frame_btn = QPushButton("Reset")
        self._reset_frame_btn.clicked.connect(self._reset_frame_overrides)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start:"))
        frame_row.addWidget(self._start_frame_spin)
        frame_row.addWidget(QLabel("End:"))
        frame_row.addWidget(self._end_frame_spin)
        frame_row.addWidget(self._apply_frame_btn)
        frame_row.addWidget(self._reset_frame_btn)
        frame_row.addStretch(1)

        self._decision_combo = QComboBox()
        for decision in [
            ReviewDecisionType.ACCEPT,
            ReviewDecisionType.REJECT,
            ReviewDecisionType.RELABEL,
            ReviewDecisionType.SKIP,
        ]:
            self._decision_combo.addItem(decision.value, userData=decision)
        self._decision_combo.currentIndexChanged.connect(self._on_review_control_changed)

        self._label_combo = QComboBox()
        self._label_combo.addItem("No Behavior", userData=NO_BEHAVIOR_ID)
        self._label_combo.addItem("boundary_error", userData="boundary_error")
        self._label_combo.currentIndexChanged.connect(self._on_review_control_changed)

        # In co-occurring mode: "Add Label" queues the combo selection into pending labels
        self._add_label_btn = QPushButton("＋ Add Label")
        self._add_label_btn.setToolTip(
            "Add the selected 'Review label' to the active label set for this clip.\n"
            "You can add multiple behaviors before saving."
        )
        self._add_label_btn.setStyleSheet(
            "QPushButton{background:#0D2B3E;color:#4FC3F7;border:1px solid #0288D1;"
            "border-radius:4px;font-weight:700;padding:4px 10px;}"
            "QPushButton:hover{background:#163D6E;}"
        )
        self._add_label_btn.clicked.connect(self._add_pending_label_from_combo)
        self._add_label_btn.setVisible(False)

        # Pending labels display for co-occurring behavior mode
        self._pending_labels_label = QLabel("Active labels:")
        self._pending_labels_label.setStyleSheet("font-weight: 600; font-size: 11px; margin-top: 4px;")
        self._pending_labels_label.setVisible(False)
        self._pending_labels_display = QLabel("None")
        self._pending_labels_display.setWordWrap(True)
        self._pending_labels_display.setStyleSheet(
            "background: #37474F; border: 1px solid #546E7A; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; color: #B0BEC5; min-height: 22px;"
        )
        self._pending_labels_display.setVisible(False)
        self._clear_pending_btn = QPushButton("Clear Labels")
        self._clear_pending_btn.setToolTip("Remove all pending behavior labels for this clip")
        self._clear_pending_btn.setFixedWidth(100)
        self._clear_pending_btn.clicked.connect(self._clear_pending_labels)
        self._clear_pending_btn.setVisible(False)

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(1.0)

        self._notes_edit = QTextEdit()
        self._notes_edit.setMaximumHeight(90)
        self._notes_edit.setPlaceholderText("Optional notes for this candidate")

        self._save_btn = QPushButton("Save Decision (Enter)")
        self._save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: 600; padding: 6px;")
        self._save_btn.clicked.connect(self._save_decision)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(3)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.clicked.connect(lambda: self._save_with_decision(ReviewDecisionType.ACCEPT))
        self._reject_btn = QPushButton("Reject")
        self._reject_btn.clicked.connect(lambda: self._save_with_decision(ReviewDecisionType.REJECT))
        self._no_behavior_btn = QPushButton("No Behavior")
        self._no_behavior_btn.clicked.connect(self._save_no_behavior_accept)
        self._relabel_btn = QPushButton("Relabel")
        self._relabel_btn.clicked.connect(lambda: self._save_with_decision(ReviewDecisionType.RELABEL))
        self._boundary_btn = QPushButton("Boundary Error")
        self._boundary_btn.clicked.connect(self._mark_boundary_error)
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(lambda: self._save_with_decision(ReviewDecisionType.SKIP))
        for _qbtn in (
            self._accept_btn, self._reject_btn, self._no_behavior_btn,
            self._relabel_btn, self._boundary_btn, self._skip_btn,
        ):
            _qbtn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            quick_row.addWidget(_qbtn)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(QLabel("Clip Preview:"))
        right_layout.addWidget(self._player, 1)
        right_layout.addWidget(QLabel("Adjust Candidate Frame Bounds:"))
        right_layout.addLayout(frame_row)

        # Candidate id / score share one row to save vertical space.
        _meta_row = QHBoxLayout()
        _meta_row.addWidget(self._id_label)
        _meta_row.addWidget(self._score_label)
        _meta_row.addStretch(1)
        right_layout.addLayout(_meta_row)
        right_layout.addWidget(self._clip_label)

        # Decision + Confidence on a single row.
        _decision_row = QHBoxLayout()
        _decision_row.addWidget(QLabel("Decision:"))
        _decision_row.addWidget(self._decision_combo, 1)
        _decision_row.addWidget(QLabel("Confidence:"))
        _decision_row.addWidget(self._confidence_spin)
        right_layout.addLayout(_decision_row)

        right_layout.addWidget(QLabel("Review label:"))
        right_layout.addWidget(self._label_combo)
        right_layout.addWidget(self._add_label_btn)
        right_layout.addWidget(self._pending_labels_label)
        right_layout.addWidget(self._pending_labels_display)
        right_layout.addWidget(self._clear_pending_btn)
        right_layout.addWidget(QLabel("Notes:"))
        right_layout.addWidget(self._notes_edit)
        right_layout.addLayout(quick_row)
        right_layout.addWidget(self._save_btn)

        # The right pane stacks many controls below the player; wrap it in a
        # scroll area so they scroll instead of clipping on short windows.
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setWidget(right_pane)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_scroll)
        splitter.setSizes([520, 760])

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._empty_label)
        root.addLayout(top)
        root.addWidget(splitter, 1)
        root.addWidget(self._summary_label)
        self._empty_label.hide()

        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._player.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._load_previous)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._load_next)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self._load_previous)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self._load_next)
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(lambda: self._player.seek(self._player.current_frame - 1))
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(lambda: self._player.seek(self._player.current_frame + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Return), self).activated.connect(self._save_decision)
        QShortcut(QKeySequence("Ctrl+A"), self).activated.connect(lambda: self._apply_batch_decision(ReviewDecisionType.ACCEPT))
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(lambda: self._apply_batch_decision(ReviewDecisionType.REJECT))

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        # Close any mining window from the previous project (its scope is stale).
        if getattr(self, "_mining_dialog", None) is not None:
            try:
                self._mining_dialog.close()
            except Exception:
                pass
            self._mining_dialog = None
        # A mining filter from a previous project must not carry over.
        if self._mined_ids is not None:
            self._mined_ids = None
            self._mined_scores = {}
            if hasattr(self, "_clear_mining_btn"):
                self._clear_mining_btn.setVisible(False)
        # Defer I/O to avoid blocking the UI thread during tab switch.
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        self._review_service.set_project(project_root)
        self._candidate_service.set_project(project_root)
        self._load_co_occurring_setting()
        self._refresh_label_options()
        self._register_behavior_shortcuts()
        self.refresh()
        logger.info("Review tab ready: %s", project_root)

    def _load_co_occurring_setting(self) -> None:
        """Read allow_co_occurring_behaviors from project config."""
        self._co_occurring_enabled = False
        if not self._project_root:
            return
        cfg_path = self._project_root / "project.yaml"
        if not cfg_path.exists():
            return
        raw = read_yaml(cfg_path, {})
        model = raw.get("behavior_model") or {}
        self._co_occurring_enabled = bool(model.get("allow_co_occurring_behaviors", False))
        # Show/hide co-occurring UI elements
        self._pending_labels_label.setVisible(self._co_occurring_enabled)
        self._pending_labels_display.setVisible(self._co_occurring_enabled)
        self._clear_pending_btn.setVisible(self._co_occurring_enabled)
        self._add_label_btn.setVisible(self._co_occurring_enabled)
        self._autoplay_chk.setToolTip(
            "When checked, automatically advance to the next clip after saving a decision.\n"
            "Uncheck to stay on the current clip (useful for co-occurring behavior labeling)."
        )
        self._pending_labels.clear()
        self._update_pending_labels_display()

    def _refresh_label_options(self) -> None:
        current = self._normalize_behavior_id(str(self._label_combo.currentData() or ""))
        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        for behavior in self._behavior_service.behaviors:
            self._label_combo.addItem(behavior.name, userData=behavior.behavior_id)
        self._label_combo.addItem("boundary_error", userData="boundary_error")
        idx = self._label_combo.findData(current)
        if idx < 0:
            idx = self._label_combo.findData(self._default_behavior_id())
        self._label_combo.setCurrentIndex(max(0, idx))
        self._label_combo.blockSignals(False)

    def _register_behavior_shortcuts(self) -> None:
        for shortcut in self._behavior_shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._behavior_shortcuts.clear()

        used_keys: set[str] = set()
        for behavior in self._behavior_service.behaviors:
            behavior_id = self._normalize_behavior_id(str(behavior.behavior_id or ""))
            key = str(behavior.keyboard_shortcut or "").strip()
            if not behavior_id or not key:
                continue

            key_norm = key.lower()
            if key_norm in used_keys:
                continue
            used_keys.add(key_norm)

            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda bid=behavior_id: self._accept_with_behavior_shortcut(bid))
            self._behavior_shortcuts.append(shortcut)

    def _open_soundboard(self) -> None:
        """Open (or refresh) the pop-out behavior soundboard window."""
        from abel.ui.behavior_soundboard import BehaviorSoundboard

        if self._soundboard is None:
            self._soundboard = BehaviorSoundboard(self)

        behaviors: list[tuple] = []
        seen: set[str] = set()
        for behavior in self._behavior_service.behaviors:
            bid = self._normalize_behavior_id(str(behavior.behavior_id or ""))
            if not bid or bid in seen:
                continue
            seen.add(bid)
            behaviors.append((
                bid,
                str(behavior.name or bid),
                str(behavior.keyboard_shortcut or "").strip(),
                bool(getattr(behavior, "is_social", False)),
                str(getattr(behavior, "directionality", "none") or "none"),
            ))

        nav = {
            "next": self._load_next,
            "prev": self._load_previous,
            "play": self._player.toggle_play,
            "save": self._save_decision,
            "frame_back": lambda: self._player.seek(self._player.current_frame - 1),
            "frame_fwd": lambda: self._player.seek(self._player.current_frame + 1),
            "accept_all": lambda: self._apply_batch_decision(ReviewDecisionType.ACCEPT),
            "reject_all": lambda: self._apply_batch_decision(ReviewDecisionType.REJECT),
        }
        self._soundboard.configure(
            behaviors, self._accept_with_behavior_shortcut, nav,
            on_structured=self._on_structured_label,
            on_commit=self._commit_structured_labels,
        )
        self._soundboard.set_animals(self._current_clip_animals())
        self._soundboard.load_labels(self._current_structured_payload())
        self._soundboard.show()
        self._soundboard.raise_()
        self._soundboard.activateWindow()

    def _current_window_id(self) -> "str | None":
        i = self._current_candidate_idx
        if 0 <= i < len(self._visible_candidates):
            return self._visible_candidates[i].window_id
        return None

    def _current_clip_animals(self) -> "list[tuple]":
        """(animal_id, display_name, (r,g,b)) for the current clip's session.

        The first tuple element MUST be the *resolved* animal id used to key the
        per-individual segment features — ``individual_subject_map[ind]`` when a
        subject mapping exists, else ``f"{subject_id or session_id}:{ind}"`` — so
        that committed soundboard labels (segment id ``seg_{animal_id}_…``) join
        to the correct segment rows. Using the raw individual key here would make
        every label miss the training join. Mirrors
        ``FeaturerepService._process_one`` (feature_prep_service.py).
        """
        from abel.utils.individual_colors import color_for
        i = self._current_candidate_idx
        if not (self._project_root and 0 <= i < len(self._visible_candidates)):
            return []
        cand = self._visible_candidates[i]
        try:
            manifest = self._imports.load_manifest(self._project_root)
        except Exception:
            manifest = None
        if manifest is None:
            return []
        sess = next((s for s in manifest.linked_sessions if s.session_id == cand.session_id), None)
        if sess is None or not getattr(sess, "individuals", None):
            return []
        imap = dict(getattr(sess, "individual_subject_map", {}) or {})
        subject_id = getattr(sess, "subject_id", None) or sess.session_id
        return [
            (
                imap.get(ind) or f"{subject_id}:{ind}",   # resolved animal_id (join key)
                str(imap.get(ind) or ind),                # display name
                color_for(idx),
            )
            for idx, ind in enumerate(sess.individuals)
        ]

    def _on_structured_label(self, behavior_id: str, focal_animal_id: str, partner_animal_id: "str | None") -> None:
        """Receive a structured (animal-aware) label from the soundboard."""
        fields = self._behavior_service.label_animal_fields(behavior_id, focal_animal_id, partner_animal_id)
        wid = self._current_window_id() or ""
        rec = {"behavior_id": behavior_id, **fields}
        self._structured_labels.setdefault(wid, []).append(rec)
        logger.info("Structured label on window %s: %s", wid, rec)

    def _current_structured_payload(self) -> "list[dict]":
        """Previously-committed soundboard labels for the current clip, if any."""
        wid = self._current_window_id()
        if not wid:
            return []
        return self._review_service.get_structured_labels(wid)

    def _refresh_soundboard_for_clip(self) -> None:
        if self._soundboard is not None and self._soundboard.isVisible():
            self._soundboard.set_animals(self._current_clip_animals())
            # Repopulate any labels this clip was already committed with, so the
            # reviewer can revisit and edit them (set_animals clears first).
            self._soundboard.load_labels(self._current_structured_payload())

    def _commit_structured_labels(self, labels: "list[dict]") -> None:
        """Persist the soundboard's collected structured labels for the current clip.

        ``labels`` is a list of ``{behavior_id, focal_animal_id,
        partner_animal_id}`` dicts. Labels are keyed to each focal animal's own
        segment (``seg_{animal}_{session}_{start}_{end}``) so instances pool by
        identity-agnostic behavior id at training time ("a mouse is a mouse").

        Multiple behaviors on the *same* animal-segment are collapsed into one
        pipe-joined :class:`ReviewerLabelRecord` (e.g. ``"grooming|rearing"``) —
        the co-occurring-label convention the trainer expands into per-behavior
        positives — rather than separate rows that would resolve to
        ``ambiguous`` and be dropped. Symmetric (mutual) social behaviors are
        exhibited by both animals, so they label the partner's segment too;
        directed behaviors label only the actor. Finally the window is marked
        ACCEPTED so it leaves the review queue on refresh.
        """
        if not labels:
            return
        if not (self._project_root and 0 <= self._current_candidate_idx < len(self._visible_candidates)):
            QMessageBox.warning(self, "No Candidate", "No candidate selected to commit labels for.")
            return
        cand = self._visible_candidates[self._current_candidate_idx]
        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        start = int(cand.start_frame)
        end = int(cand.end_frame)

        # Normalize behavior ids, then let the behavior service fan the labels out
        # to per-animal segments (solo/directed/mutual + co-occurring merge).
        normalized = [
            {
                "behavior_id": self._normalize_behavior_id(str(lab.get("behavior_id") or "")),
                "focal_animal_id": lab.get("focal_animal_id"),
                "partner_animal_id": lab.get("partner_animal_id"),
            }
            for lab in labels
        ]
        records = self._behavior_service.aggregate_clip_labels(
            normalized, str(cand.session_id), start, end,
        )

        # Re-commit safety: purge any prior label rows for this clip's
        # animal-segments before appending the fresh set, so editing a
        # previously-committed clip replaces its labels instead of duplicating
        # or leaving stale rows (e.g. a label the user just removed).
        purge_ids = {
            f"seg_{aid}_{cand.session_id}_{start}_{end}"
            for (aid, _n, _c) in self._current_clip_animals()
        }
        purge_ids.update(r["segment_id"] for r in records)
        self._review_service.remove_segment_labels(list(purge_ids))

        committed = 0
        multi_label_segments = 0
        for rec_spec in records:
            if "|" in rec_spec["review_label"]:
                multi_label_segments += 1
            self._review_service.append_segment_label(
                ReviewerLabelRecord(
                    segment_id=rec_spec["segment_id"],
                    review_label=rec_spec["review_label"],
                    reviewer_id=reviewer,
                    notes="soundboard",
                    **rec_spec["fields"],
                )
            )
            committed += 1

        if committed == 0:
            return

        # Mark the whole window reviewed so it leaves the queue on refresh.
        rec = self._review_service.upsert_decision(
            clip_id=cand.window_id,
            reviewer=reviewer,
            decision=ReviewDecisionType.ACCEPT,
            behavior_label=self._normalize_behavior_id(str(labels[0].get("behavior_id") or "")),
            notes="soundboard",
            adjusted_start_frame=start,
            adjusted_end_frame=end,
        )
        self._decision_by_clip_id[cand.window_id] = rec
        self._structured_labels.pop(cand.window_id, None)
        # Persist the exact payload so revisiting this clip repopulates the
        # soundboard for editing (the training-side rows are lossy).
        self._review_service.save_structured_labels(cand.window_id, normalized)
        self._update_decision_cell(self._current_candidate_idx, rec)
        self._update_summary()
        logger.info("Committed %d structured label(s) for window %s", committed, cand.window_id)

        # Co-occurring (pipe-joined) labels are only expanded into per-behavior
        # positives when the project enables co-occurring behaviors; otherwise
        # the trainer treats "a|b" as one junk class. Warn once per commit.
        if multi_label_segments and not getattr(self, "_co_occurring_enabled", False):
            logger.warning(
                "Committed %d segment(s) with multiple behaviors on one animal, but "
                "'allow co-occurring behaviors' is OFF — these will not split into "
                "per-behavior training instances. Enable it in the Behavior tab.",
                multi_label_segments,
            )
            QMessageBox.information(
                self, "Co-occurring behaviors disabled",
                f"{multi_label_segments} animal(s) in this clip were labeled with more "
                "than one behavior. For those to train correctly, enable “Allow "
                "co-occurring behaviors” in the Behavior tab; otherwise only "
                "single-behavior labels are used.",
            )

        # Advance to the next clip and refresh the soundboard's animal roster.
        if self._autoplay_chk.isChecked():
            self._load_next()
        self._refresh_soundboard_for_clip()

    def _accept_with_behavior_shortcut(self, behavior_id: str) -> None:
        if self._co_occurring_enabled:
            # In co-occurring mode: pressing a shortcut adds the behavior to the pending set.
            # If auto-advance is on, also save+advance immediately (single-label fast review).
            # If auto-advance is off, just queue it so the user can add more labels first.
            if behavior_id in self._pending_labels:
                self._pending_labels.discard(behavior_id)
            else:
                if behavior_id != NO_BEHAVIOR_ID:
                    self._pending_labels.discard(NO_BEHAVIOR_ID)
                self._pending_labels.add(behavior_id)
            self._update_pending_labels_display()
            label_idx = self._label_combo.findData(behavior_id)
            if label_idx >= 0:
                self._label_combo.setCurrentIndex(label_idx)
            if self._autoplay_chk.isChecked():
                self._save_decision()
            return
        label_idx = self._label_combo.findData(behavior_id)
        if label_idx >= 0:
            self._label_combo.setCurrentIndex(label_idx)
        self._save_with_decision(ReviewDecisionType.ACCEPT)

    @staticmethod
    def _behavior_tag_html(name: str, color: str, suffix: str = "") -> str:
        """One colored behavior tag; ``suffix`` (e.g. an animal name) is appended."""
        text = f"{name} · {suffix}" if suffix else name
        return (
            f'<span style="background:{color}; color:white; padding:2px 6px; '
            f'border-radius:3px; margin:1px; font-weight:600;">{text}</span>'
        )

    @classmethod
    def _format_structured_tags(
        cls, structured: "list[dict]", name_by_id: dict, behavior_service,
    ) -> list[str]:
        """Build animal-aware behavior tags (behavior · animal, with →/⇄ for social)."""
        tags: list[str] = []
        for lab in structured:
            bid = str(lab.get("behavior_id") or "").strip()
            focal = lab.get("focal_animal_id")
            partner = lab.get("partner_animal_id")
            bdef = behavior_service.get(bid)
            name = bdef.name if bdef else (bid or "?")
            color = (getattr(bdef, "color", None) or "#4A90E2")
            focal_nm = name_by_id.get(focal, str(focal))
            if partner:
                directed = bool(bdef and str(getattr(bdef, "directionality", "none")) == "directed")
                arrow = "→" if directed else "⇄"
                who = f"{focal_nm} {arrow} {name_by_id.get(partner, str(partner))}"
            else:
                who = focal_nm
            tags.append(cls._behavior_tag_html(name, color, who))
        return tags

    def _structured_label_tags(self) -> "list[str] | None":
        """Animal-aware tags for the current clip's committed structured labels.

        Returns ``None`` when the clip has no structured (soundboard) labels, so
        the caller falls back to the plain behavior-only rendering. Each tag
        names the behavior *and* the animal it applies to (with →/⇄ for directed
        / mutual social behaviors), so a per-mouse label no longer displays as a
        bare behavior name.
        """
        structured = self._current_structured_payload()
        if not structured:
            return None
        name_by_id = {aid: nm for (aid, nm, _c) in self._current_clip_animals()}
        return self._format_structured_tags(structured, name_by_id, self._behavior_service)

    def _update_pending_labels_display(self) -> None:
        """Refresh the pending-labels tag display."""
        # Prefer animal-aware structured labels when this clip was committed via
        # the soundboard, so "Groom" reads e.g. "Groom · black female".
        structured_tags = self._structured_label_tags()
        if structured_tags:
            self._pending_labels_display.setText("  ".join(structured_tags))
            self._pending_labels_display.setStyleSheet(
                "background: #263238; border: 1px solid #4CAF50; border-radius: 4px; "
                "padding: 4px 8px; font-size: 11px; min-height: 22px;"
            )
            return
        if not self._pending_labels:
            self._pending_labels_display.setText("None — press behavior hotkeys to add labels")
            self._pending_labels_display.setStyleSheet(
                "background: #37474F; border: 1px solid #546E7A; border-radius: 4px; "
                "padding: 4px 8px; font-size: 11px; color: #78909C; min-height: 22px;"
            )
            return
        # Build colored tag display
        parts: list[str] = []
        for bid in sorted(self._pending_labels):
            name = bid
            color = "#4A90E2"
            for b in self._behavior_service.behaviors:
                if b.behavior_id == bid:
                    name = b.name
                    color = b.color or "#4A90E2"
                    break
            parts.append(self._behavior_tag_html(name, color))
        self._pending_labels_display.setText("  ".join(parts))
        self._pending_labels_display.setStyleSheet(
            "background: #263238; border: 1px solid #4CAF50; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; min-height: 22px;"
        )

    def _clear_pending_labels(self) -> None:
        self._pending_labels.clear()
        self._update_pending_labels_display()

    def _add_pending_label_from_combo(self) -> None:
        """Queue the currently selected label combo entry into the pending-labels set."""
        bid = self._normalize_behavior_id(str(self._label_combo.currentData() or ""))
        if not bid:
            return
        if bid != NO_BEHAVIOR_ID:
            self._pending_labels.discard(NO_BEHAVIOR_ID)
        self._pending_labels.add(bid)
        self._update_pending_labels_display()

    def _default_behavior_id(self) -> str:
        for behavior in self._behavior_service.behaviors:
            bid = str(behavior.behavior_id).strip()
            if bid and bid != NO_BEHAVIOR_ID:
                return bid
        if self._behavior_service.behaviors:
            return str(self._behavior_service.behaviors[0].behavior_id)
        return NO_BEHAVIOR_ID

    def _normalize_behavior_id(self, behavior_id: str | None) -> str:
        bid = str(behavior_id or "").strip()
        if not bid or bid == "target_behavior":
            return ""
        return bid

    def _effective_behavior_id(self, candidate) -> str:
        decision = self._decision_by_clip_id.get(candidate.window_id)
        if decision and decision.behavior_label:
            bid = self._normalize_behavior_id(str(decision.behavior_label))
            return bid or UNASSIGNED_BEHAVIOR_ID
        bid = self._normalize_behavior_id(getattr(candidate, "behavior_id", ""))
        return bid or UNASSIGNED_BEHAVIOR_ID

    def _resolve_behavior_display_name(self, behavior_id_str: str) -> str:
        """Resolve a behavior ID string (possibly pipe-separated) to display names."""
        if "|" not in behavior_id_str:
            return self._display_behavior_name_map.get(behavior_id_str, behavior_id_str)
        parts = sorted(bid.strip() for bid in behavior_id_str.split("|") if bid.strip())
        names = [self._display_behavior_name_map.get(bid, bid) for bid in parts]
        return " + ".join(names)

    @staticmethod
    def _canonical_multi_label(bid: str) -> str:
        """Return a canonical, order-independent form of a (possibly pipe-separated) label.

        Sorts constituent behavior IDs alphabetically so that 'rearing|grooming'
        and 'grooming|rearing' are treated as identical.
        """
        if "|" not in bid:
            return bid
        parts = sorted(b.strip() for b in bid.split("|") if b.strip())
        return "|".join(parts)

    def refresh(self) -> None:
        self._refresh_candidates()

    def _refresh_candidates(self) -> None:
        if not self._project_root:
            self._all_candidates = []
            self._apply_filter()
            return

        # Re-read co-occurring setting in case it was changed in the Behaviors tab.
        self._load_co_occurring_setting()

        # Reload behaviors from project files so newly added labels appear without reopening the tab.
        self._behavior_service.set_project(self._project_root)
        self._refresh_label_options()
        self._register_behavior_shortcuts()

        self._refresh_session_order_index()
        self._all_candidates = self._candidate_service.load_candidates()
        decisions = self._review_service.load_decisions()
        self._decision_by_clip_id = {d.clip_id: d for d in decisions}
        self._rebuild_display_maps()
        self._refresh_behavior_filter_options()
        self._apply_filter()

    def _refresh_session_order_index(self) -> None:
        self._session_order_index = {}
        if not self._project_root:
            return
        manifest = self._imports.load_manifest(self._project_root)
        if not manifest:
            return
        for idx, session in enumerate(manifest.linked_sessions):
            self._session_order_index[session.session_id] = idx

    def _refresh_behavior_filter_options(self) -> None:
        current = self._behavior_filter_combo.currentData()
        behavior_ids = {
            self._canonical_multi_label(self._effective_behavior_id(c))
            for c in self._all_candidates
            if self._effective_behavior_id(c)
        }
        for decision in self._decision_by_clip_id.values():
            bid = self._normalize_behavior_id(str(decision.behavior_label or ""))
            if bid:
                behavior_ids.add(self._canonical_multi_label(bid))
        behavior_ids = sorted(behavior_ids)
        self._behavior_filter_combo.blockSignals(True)
        self._behavior_filter_combo.clear()
        self._behavior_filter_combo.addItem("All behaviors", userData="all")
        for bid in behavior_ids:
            label = self._resolve_behavior_display_name(bid)
            self._behavior_filter_combo.addItem(label, userData=bid)
        idx = self._behavior_filter_combo.findData(current)
        if idx >= 0:
            self._behavior_filter_combo.setCurrentIndex(idx)
        self._behavior_filter_combo.blockSignals(False)

    def _load_al_fp_fn_ids(
        self,
    ) -> tuple[set[str], set[str], dict[str, list[str]], dict[str, list[str]], set[str]]:
        """Return (fp_ids, fn_ids, fp_behavior_map, fn_behavior_map, post_pred_reviewed_ids).

        For every behavior model that has a ``segment_predictions.parquet`` and
        ``run_settings.json``, joins predictions with the most-recent human label
        per segment from ``reviewer_labels.parquet`` — the exact same join used
        inside ``_evaluate_if_possible`` to build the confusion matrix.

        - **FP**: human labeled the segment as *not* the target behavior, but the
          model gives ``prediction_prob ≥ 0.5`` (top-right cell of the matrix).
        - **FN**: human labeled the segment *as* the target behavior, but the model
          gives ``prediction_prob < 0.5`` (bottom-left cell of the matrix).

        The behavior maps store the target **behavior_id** (e.g. ``"dig"``) so that
        ``_load_candidate`` can pre-fill the label combo with the relevant behavior.

        ``post_pred_reviewed_ids`` contains segments whose most-recent label timestamp
        is newer than the prediction file — i.e., already re-reviewed after the model ran.
        """
        if not self._project_root:
            logger.debug("_load_al_fp_fn_ids: project_root not set, returning empty")
            return set(), set(), {}, {}, set()

        import pandas as _pd  # noqa: PLC0415
        from abel.storage.file_store import read_json  # noqa: PLC0415

        label_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not label_path.exists():
            logger.debug("_load_al_fp_fn_ids: reviewer_labels.parquet not found at %s", label_path)
            return set(), set(), {}, {}, set()

        try:
            labels_df = _pd.read_parquet(label_path)
        except Exception as exc:
            logger.warning("_load_al_fp_fn_ids: failed to read reviewer_labels.parquet: %s", exc)
            return set(), set(), {}, {}, set()

        if (
            labels_df.empty
            or "segment_id" not in labels_df.columns
            or "review_label" not in labels_df.columns
        ):
            logger.debug("_load_al_fp_fn_ids: labels_df empty or missing columns: %s", list(labels_df.columns))
            return set(), set(), {}, {}, set()

        logger.debug("_load_al_fp_fn_ids: loaded %d label rows from parquet", len(labels_df))

        # Deduplicate: keep most-recent label per segment_id.
        has_timestamp = "timestamp" in labels_df.columns
        if has_timestamp:
            labels_df = labels_df.sort_values("timestamp", ascending=False)
        labels_df = labels_df.drop_duplicates(subset=["segment_id"], keep="first")
        logger.debug("_load_al_fp_fn_ids: %d unique segments after dedup", len(labels_df))

        # Build a per-segment latest-label Unix timestamp dict for post-prediction detection.
        latest_label_ts: dict[str, float] = {}
        if has_timestamp:
            for _, _row in labels_df[["segment_id", "timestamp"]].iterrows():
                try:
                    _ts = _row["timestamp"]
                    _unix = float(
                        _ts.timestamp() if hasattr(_ts, "timestamp")
                        else _pd.Timestamp(_ts).timestamp()
                    )
                    latest_label_ts[str(_row["segment_id"])] = _unix
                except Exception:
                    pass

        fp_ids: set[str] = set()
        fn_ids: set[str] = set()
        # segment_id → list of behavior *names* for which it is FP / FN
        fp_behavior_map: dict[str, list[str]] = {}
        fn_behavior_map: dict[str, list[str]] = {}

        models_root = self._project_root / "derived" / "models"
        if not models_root.exists():
            logger.debug("_load_al_fp_fn_ids: models_root not found at %s", models_root)
            return set(), set(), {}, {}, set()

        max_pred_mtime: float = 0.0

        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            pred_path = model_dir / "segment_predictions.parquet"
            settings_path = model_dir / "run_settings.json"
            if not pred_path.exists() or not settings_path.exists():
                logger.debug("_load_al_fp_fn_ids: skipping %s (missing pred or settings)", model_dir.name)
                continue
            try:
                settings = read_json(settings_path, {})
                target_behavior = str(settings.get("target_behavior", "") or "").strip()
                if not target_behavior:
                    logger.debug("_load_al_fp_fn_ids: %s has empty target_behavior, skipping", model_dir.name)
                    continue
                pred_df = _pd.read_parquet(pred_path)
                if "segment_id" not in pred_df.columns or "prediction_prob" not in pred_df.columns:
                    logger.debug("_load_al_fp_fn_ids: %s pred missing columns: %s", model_dir.name, list(pred_df.columns))
                    continue
                pred_df = pred_df[["segment_id", "prediction_prob"]]
            except Exception as exc:
                logger.warning("_load_al_fp_fn_ids: error loading %s: %s", model_dir.name, exc)
                continue

            # Inner join — only segments that have both a prediction and a human label.
            merged = pred_df.merge(
                labels_df[["segment_id", "review_label"]], on="segment_id", how="inner"
            )
            if merged.empty:
                logger.debug("_load_al_fp_fn_ids: %s inner join returned 0 rows", model_dir.name)
                continue

            label_true = (merged["review_label"].astype(str) == target_behavior).astype(int)
            label_pred = (merged["prediction_prob"].astype(float) >= 0.5).astype(int)

            # Skip degenerate models. A behavior trained with no (or near-zero)
            # positive labels collapses into an all-positive predictor — it scores
            # ~every segment >= 0.5, so every human-labeled segment becomes a
            # "false positive" against it. That floods al_all_ids with the entire
            # labeled set, which (a) pins every reviewed clip visible even with
            # "Show reviewed" off — it stays an FP no matter how it's relabeled —
            # and (b) crowds the queue so real FP/FN disagreements are lost. Such a
            # model has no ground truth to triage against, so exclude it entirely.
            n_pos_labels = int(label_true.sum())
            pred_pos_frac = float(label_pred.mean()) if len(label_pred) else 0.0
            if n_pos_labels == 0 or pred_pos_frac >= 0.9:
                logger.warning(
                    "_load_al_fp_fn_ids: skipping degenerate model %s (target=%s): "
                    "%d positive human labels, predicts positive for %.0f%% of %d "
                    "labeled segments — excluded from FP/FN triage",
                    model_dir.name, target_behavior[:8], n_pos_labels,
                    pred_pos_frac * 100.0, len(merged),
                )
                continue

            # This model contributes real FP/FN — count its predictions toward the
            # "predictions ran at" cutoff used to detect post-run re-reviews.
            try:
                _mtime = pred_path.stat().st_mtime
                if _mtime > max_pred_mtime:
                    max_pred_mtime = _mtime
            except Exception:
                pass

            fp_mask = (label_true == 0) & (label_pred == 1)
            fn_mask = (label_true == 1) & (label_pred == 0)
            n_fp = int(fp_mask.sum())
            n_fn = int(fn_mask.sum())
            logger.debug(
                "_load_al_fp_fn_ids: %s → %d merged, %d FP, %d FN (target=%s)",
                model_dir.name, len(merged), n_fp, n_fn, target_behavior[:8],
            )
            for seg_id in merged.loc[fp_mask, "segment_id"].astype(str):
                fp_ids.add(seg_id)
                fp_behavior_map.setdefault(seg_id, []).append(target_behavior)
            for seg_id in merged.loc[fn_mask, "segment_id"].astype(str):
                fn_ids.add(seg_id)
                fn_behavior_map.setdefault(seg_id, []).append(target_behavior)

        # Determine which FP/FN clips were already re-reviewed after the model ran.
        post_pred_reviewed_ids: set[str] = set()
        if max_pred_mtime > 0.0:
            for seg_id in (fp_ids | fn_ids):
                seg_ts = latest_label_ts.get(seg_id, 0.0)
                if seg_ts > max_pred_mtime:
                    post_pred_reviewed_ids.add(seg_id)

        logger.info(
            "_load_al_fp_fn_ids: total unique FP=%d, FN=%d, already-re-reviewed=%d",
            len(fp_ids), len(fn_ids), len(post_pred_reviewed_ids),
        )
        return fp_ids, fn_ids, fp_behavior_map, fn_behavior_map, post_pred_reviewed_ids

    @staticmethod
    def _is_bout_review_candidate(candidate) -> bool:
        """Return True if the candidate originated from temporal bout review."""
        # CandidateWindow objects carry an explicit source field.
        source = getattr(candidate, "source", "")
        if source == "temporal_bout_review":
            return True
        # Fallback: window IDs generated by the bout-to-clip pipeline start with "bout_".
        return str(getattr(candidate, "window_id", "")).startswith("bout_")

    @staticmethod
    def _is_umap_selection_candidate(candidate) -> bool:
        """Return True if the candidate originated from UMAP interactive selection."""
        source = getattr(candidate, "source", "") or ""
        return source in ("umap_interactive_selection", "umap_selection")

    # ── Filter popup helpers ─────────────────────────────────────────────────

    def _toggle_filter_panel(self) -> None:
        """Show or hide the filter popup positioned below the Filters button."""
        if self._filter_panel.isVisible():
            self._filter_panel.hide()
            return
        self._filter_panel.adjustSize()
        btn_global = self._filters_btn.mapToGlobal(self._filters_btn.rect().bottomLeft())
        self._filter_panel.move(btn_global)
        self._filter_panel.show()
        self._filter_panel.raise_()

    def _update_filter_btn_text(self) -> None:
        """Append an indicator to the Filters button text when any filter is active."""
        active = (
            self._show_reviewed_chk.isChecked()
            or self._show_missing_clips_chk.isChecked()
            or self._reviewed_with_clips_btn.isChecked()
            or self._show_fp_fn_btn.isChecked()
            or self._show_al_fp_fn_btn.isChecked()
        )
        self._filters_btn.setText("Filters \u25cf" if active else "Filters")
        self._filters_btn.setStyleSheet(
            "font-weight: 700; color: #80CBC4;" if active else ""
        )

    # ── Targeted Clip Mining ─────────────────────────────────────────────────

    def _open_clip_mining(self) -> None:
        """Open the clip-mining dialog (modeless) over the full scored-segment pool."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        # Bring an already-open mining window forward instead of stacking a new one.
        existing = getattr(self, "_mining_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        # Scope is the full feature-extraction segment pool, not just the loaded
        # review queue, so the user can surface clips that were never candidates.
        pool_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        if not pool_path.exists():
            QMessageBox.information(
                self, "Targeted Clip Mining",
                "No scored segments were found. Run Feature Extraction first.",
            )
            return

        scope_label = "All feature-extraction segments"

        def _live_exemplars() -> list[ClipRef]:
            # Read the review list's current selection each time essence is used.
            rows = sorted({idx.row() for idx in self._candidate_table.selectionModel().selectedRows()})
            return [
                ClipRef(
                    str(self._visible_candidates[r].window_id),
                    str(self._visible_candidates[r].session_id),
                    int(self._visible_candidates[r].start_frame),
                    int(self._visible_candidates[r].end_frame),
                )
                for r in rows if 0 <= r < len(self._visible_candidates)
            ]

        dlg = ClipMiningDialog(
            project_root=self._project_root,
            exemplar_provider=_live_exemplars,
            scope_label=scope_label,
            on_apply=self._apply_mined_ids,
            parent=self,
            on_flag_queue=self._flag_queue_by_essence,
        )
        # Modeless so the main window (session selection, clip list) stays usable.
        dlg.setModal(False)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        self._mining_dialog = dlg
        # Keep the essence button's live count in sync with the review selection.
        sel_model = self._candidate_table.selectionModel()
        sel_model.selectionChanged.connect(dlg.refresh_exemplar_count)

        def _on_closed(*_a) -> None:
            try:
                sel_model.selectionChanged.disconnect(dlg.refresh_exemplar_count)
            except Exception:
                pass
            self._mining_dialog = None

        dlg.finished.connect(_on_closed)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _apply_mined_ids(self, matched_refs: list, scores: dict) -> None:
        """Inject mined windows as review candidates, extracting any missing clips.

        ``matched_refs`` are :class:`ClipRef`-shaped windows from the full scored
        segment pool. Windows that are not already review candidates are persisted
        as external candidates; those without an extracted clip are re-extracted in
        a background worker. The queue is then narrowed to the mined matches,
        ranked by match score.
        """
        if not self._project_root or not matched_refs:
            return
        if getattr(self, "_mining_extract_busy", False):
            QMessageBox.information(
                self, "Targeted Clip Mining",
                "A previous mining batch is still extracting clips. Please wait.",
            )
            return

        matched_ids = [str(r.window_id) for r in matched_refs]
        scores_map = {str(k): float(v) for k, v in (scores or {}).items()}

        existing_ids = {c.window_id for c in self._all_candidates}
        existing_with_clip = {c.window_id for c in self._all_candidates if self._candidate_clip_path(c)}

        windows = [
            CandidateWindow(
                window_id=str(r.window_id),
                session_id=str(r.session_id),
                start_frame=int(r.start_frame),
                end_frame=int(r.end_frame),
                source="clip_mining",
                selection_reason="Targeted clip mining match",
            )
            for r in matched_refs
        ]
        to_persist = [w for w in windows if w.window_id not in existing_ids]
        to_extract = [w for w in windows if w.window_id not in existing_with_clip]

        if to_extract:
            answer = QMessageBox.question(
                self,
                "Targeted Clip Mining",
                f"{len(matched_ids)} segment(s) matched. "
                f"{len(to_extract)} need clip extraction (decodes video and may take a "
                "while for large batches).\n\nExtract and load them into the review queue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # Persist pool-only windows so they survive a reload and appear in the queue.
        if to_persist:
            self._candidate_service.upsert_external_window_candidates(to_persist)

        # Remember what to narrow to once extraction (if any) finishes.
        self._pending_mined_ids = set(matched_ids)
        self._pending_mined_scores = scores_map

        if not to_extract:
            self._finalize_mined_view()
            return

        self._mining_extract_busy = True
        progress = QProgressDialog(
            "Extracting mined clips…", "Cancel", 0, len(to_extract), self
        )
        progress.setWindowTitle("Targeted Clip Mining")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._mining_progress_dialog = progress
        cancel_flag = [False]
        progress.canceled.connect(lambda: cancel_flag.__setitem__(0, True))

        try:
            self._mining_extract_progress.disconnect(self._on_mining_extract_progress)
        except (TypeError, RuntimeError):
            pass
        self._mining_extract_progress.connect(self._on_mining_extract_progress)

        def _progress(done: int, total: int) -> None:
            self._mining_extract_progress.emit(int(done), int(total))

        worker = TaskWorker(
            regenerate_clips_for_windows,
            self._project_root,
            to_extract,
            progress_callback=_progress,
            cancel_flag=cancel_flag,
        )
        worker.signals.finished.connect(self._on_mining_extract_complete)
        worker.signals.failed.connect(self._on_mining_extract_failed)
        self._pool.start(worker)

    @Slot(int, int)
    def _on_mining_extract_progress(self, done: int, total: int) -> None:
        dlg = getattr(self, "_mining_progress_dialog", None)
        if dlg is not None:
            dlg.setMaximum(max(1, total))
            dlg.setValue(done)

    @Slot(object)
    def _on_mining_extract_complete(self, summary: dict) -> None:
        self._mining_extract_busy = False
        dlg = getattr(self, "_mining_progress_dialog", None)
        if dlg is not None:
            dlg.close()
            self._mining_progress_dialog = None
        extracted = int((summary or {}).get("extracted", 0))
        requested = int((summary or {}).get("requested", 0))
        self._finalize_mined_view()
        if extracted < requested:
            QMessageBox.information(
                self, "Targeted Clip Mining",
                f"Extracted {extracted} of {requested} clip(s). Some could not be "
                "rebuilt (missing source video, unresolved session, or decode error).",
            )

    @Slot(str)
    def _on_mining_extract_failed(self, tb: str) -> None:
        self._mining_extract_busy = False
        dlg = getattr(self, "_mining_progress_dialog", None)
        if dlg is not None:
            dlg.close()
            self._mining_progress_dialog = None
        logger.error("Mined-clip extraction failed:\n%s", tb)
        # Still show whatever already had clips.
        self._finalize_mined_view()
        QMessageBox.critical(
            self, "Targeted Clip Mining",
            "Clip extraction failed for the mined batch — see log for details.",
        )

    def _finalize_mined_view(self) -> None:
        """Reload candidates and narrow the queue to the pending mined match set."""
        self._mined_ids = set(getattr(self, "_pending_mined_ids", set()) or set())
        self._mined_scores = dict(getattr(self, "_pending_mined_scores", {}) or {})
        self._pending_mined_ids = set()
        self._pending_mined_scores = {}
        self._clear_mining_btn.setVisible(True)
        # Reload so freshly extracted / persisted windows are in _all_candidates.
        self._refresh_candidates()

    def _clear_mining_filter(self) -> None:
        self._mined_ids = None
        self._mined_scores = {}
        self._clear_mining_btn.setVisible(False)
        self._apply_filter()

    def _remove_mined_windows(self) -> None:
        """Delete candidates that were injected by Targeted Clip Mining.

        Mined pool windows persist as external candidates (``source='clip_mining'``)
        so their clips stay reviewable after the mining filter is cleared. This
        purges those windows on demand; already-reviewed windows keep their saved
        decisions untouched.
        """
        if not self._project_root:
            return
        removed = self._candidate_service.remove_external_candidates_by_source("clip_mining")
        if not removed:
            QMessageBox.information(
                self, "Remove Mined Windows", "No mined windows to remove."
            )
            return
        # Drop the mining filter if active so we return to the normal queue.
        self._mined_ids = None
        self._mined_scores = {}
        self._clear_mining_btn.setVisible(False)
        self._refresh_candidates()
        QMessageBox.information(
            self, "Remove Mined Windows", f"Removed {removed} mined window(s)."
        )

    def _apply_mining_view(self) -> None:
        """Build the visible queue from the mined ID set, ranked by match score."""
        mined = self._mined_ids or set()
        rows = [c for c in self._all_candidates if c.window_id in mined]
        have = {c.window_id for c in rows}
        # Surface mined clips that are already reviewed (not in the live queue).
        for wid in mined:
            if wid in have:
                continue
            decision = self._decision_by_clip_id.get(wid)
            if decision is not None:
                rows.append(self._candidate_from_decision(decision))
                have.add(wid)
        rows.sort(
            key=lambda c: (
                -self._mined_scores.get(c.window_id, 0.0),
                self._session_order_index.get(c.session_id, 10_000_000),
                c.start_frame,
            )
        )
        self._visible_candidates = rows
        self._populate_candidate_table()
        self._update_summary()
        self._update_filter_btn_text()

        has_rows = bool(rows)
        self._empty_label.setVisible(not has_rows)
        if not has_rows:
            self._candidate_label.setText("No mined clips match")
            self._player.close_clip()
            self._current_candidate_idx = -1
            return
        self._candidate_label.setText(f"Mined queue: {len(rows)} clip(s)")
        next_idx = self._current_candidate_idx if 0 <= self._current_candidate_idx < len(rows) else 0
        self._load_candidate(next_idx, select_row=True)

    def _apply_filter(self) -> None:
        if not hasattr(self, "_behavior_filter_combo"):
            return
        # Targeted Clip Mining takes over the queue entirely when active.
        if self._mined_ids is not None:
            self._apply_mining_view()
            return
        sort_mode = str(self._sort_combo.currentData() or "score_desc")
        behavior_mode = str(self._behavior_filter_combo.currentData() or "all")
        show_reviewed = bool(self._show_reviewed_chk.isChecked())
        show_missing_clips = bool(self._show_missing_clips_chk.isChecked())
        reviewed_with_clips_only = bool(self._reviewed_with_clips_btn.isChecked())
        show_fp_fn = bool(self._show_fp_fn_btn.isChecked())
        show_al_fp_fn = bool(self._show_al_fp_fn_btn.isChecked())
        if reviewed_with_clips_only:
            show_reviewed = True

        # Load active-learning FP/FN index if the filter is on.
        al_fp_ids: set[str] = set()
        al_fn_ids: set[str] = set()
        al_all_ids: set[str] = set()
        al_fp_behavior_map: dict[str, list[str]] = {}
        al_fn_behavior_map: dict[str, list[str]] = {}
        al_post_pred_reviewed_ids: set[str] = set()
        if show_al_fp_fn:
            al_fp_ids, al_fn_ids, al_fp_behavior_map, al_fn_behavior_map, al_post_pred_reviewed_ids = self._load_al_fp_fn_ids()
        self._al_fp_ids = al_fp_ids
        self._al_fn_ids = al_fn_ids
        self._al_fp_behavior_map = al_fp_behavior_map
        self._al_fn_behavior_map = al_fn_behavior_map
        self._al_post_pred_reviewed_ids = al_post_pred_reviewed_ids

        rows = list(self._all_candidates)

        # Build the set of bout-review window IDs for narrowing later.
        bout_review_ids: set[str] = set()
        if show_fp_fn:
            bout_review_ids = {
                str(c.window_id)
                for c in rows
                if self._is_bout_review_candidate(c)
            }

        # UMAP-selected clips are hand-picked for accuracy checks — always
        # keep them visible regardless of review status.
        umap_selection_ids: set[str] = {
            str(c.window_id)
            for c in rows
            if self._is_umap_selection_candidate(c)
        }
        if show_reviewed:
            existing_ids = {str(c.window_id) for c in rows}
            for decision in self._decision_by_clip_id.values():
                clip_id = str(decision.clip_id or "").strip()
                if not clip_id or clip_id in existing_ids:
                    continue
                rows.append(self._candidate_from_decision(decision))

        # Inject any AL FP/FN segments that are not already in the list.
        # These are reviewed segments (they have labels) that may not be in
        # the candidate queue at all — they need to be surfaced explicitly.
        al_all_ids = al_fp_ids | al_fn_ids
        if show_al_fp_fn and al_all_ids:
            existing_ids = {c.window_id for c in rows}
            for seg_id in al_all_ids:
                if seg_id in existing_ids:
                    continue
                # Prefer decision record (carries behavior label).
                if seg_id in self._decision_by_clip_id:
                    rows.append(self._candidate_from_decision(self._decision_by_clip_id[seg_id]))
                else:
                    # Parse session_id and frames from segment_id format:
                    # seg_{animal}_{session_id}_{start}_{end}
                    m_seg = re.match(
                        r"^seg_.+_(session_[^_]+)_([0-9]+)_([0-9]+)$", seg_id
                    )
                    if m_seg:
                        rows.append(
                            _ReviewListRow(
                                window_id=seg_id,
                                session_id=m_seg.group(1),
                                behavior_id=None,
                                start_frame=int(m_seg.group(2)),
                                end_frame=int(m_seg.group(3)),
                                total_score=0.0,
                                clip_path=None,
                            )
                        )
                existing_ids.add(seg_id)

        # Narrow to AL FP/FN candidates when that filter is active.
        if show_al_fp_fn and al_all_ids:
            rows = [c for c in rows if c.window_id in al_all_ids]

        # Narrow to bout-review candidates when that filter is active.
        if show_fp_fn:
            rows = [c for c in rows if c.window_id in bout_review_ids]

        # Update the status label with FP/FN counts.
        if hasattr(self, "_al_fp_fn_status_label"):
            if show_al_fp_fn:
                self._al_fp_fn_status_label.setText(
                    f"Found: {len(al_fp_ids)} FP  •  {len(al_fn_ids)} FN  •  {len(rows)} shown"
                )
                self._al_fp_fn_status_label.setVisible(True)
            else:
                self._al_fp_fn_status_label.setVisible(False)

        # Behavior-mode filter: when AL FP/FN is active the rows already span
        # multiple human-assigned labels (FP rows are labeled as non-target,
        # FN rows as the target), so skip the behavior filter in that mode.
        if behavior_mode != "all" and not show_al_fp_fn:
            rows = [c for c in rows if self._canonical_multi_label(self._effective_behavior_id(c)) == behavior_mode]
        if reviewed_with_clips_only:
            rows = [
                c
                for c in rows
                if c.window_id in self._decision_by_clip_id and self._candidate_clip_path(c)
            ]
        elif not show_reviewed:
            # AL FP/FN clips stay visible until the user re-labels them; after saving
            # a new decision, _load_al_fp_fn_ids drops the clip from al_all_ids and it
            # disappears naturally — advancing the panel like any other reviewed clip.
            # Bout-review clips always stay visible when the temporal bout review
            # filter is on so the user can see what was sent over and what decision
            # has been recorded for each clip.
            # Reviewed clips (including UMAP-selected ones) disappear here so the
            # queue only shows unreviewed work; use "Show reviewed" to see them.
            rows = [
                c
                for c in rows
                if c.window_id not in self._decision_by_clip_id
                or (show_al_fp_fn and c.window_id in al_all_ids)
                or (show_fp_fn and c.window_id in bout_review_ids)
            ]
        if (not reviewed_with_clips_only) and (not show_missing_clips):
            if show_reviewed or show_fp_fn or show_al_fp_fn:
                # Keep reviewed rows visible even if clip files were pruned after review.
                # Bout-review and AL FP/FN rows may have no clip files — always keep them.
                rows = [
                    c
                    for c in rows
                    if c.window_id in self._decision_by_clip_id
                    or self._candidate_clip_path(c)
                    or (show_fp_fn and c.window_id in bout_review_ids)
                    or (show_al_fp_fn and c.window_id in al_all_ids)
                    or c.window_id in umap_selection_ids
                ]
            else:
                rows = [c for c in rows if self._candidate_clip_path(c)]

        if sort_mode == "video_order":
            rows.sort(
                key=lambda c: (
                    self._session_order_index.get(c.session_id, 10_000_000),
                    c.start_frame,
                    c.end_frame,
                    c.window_id,
                )
            )
        elif sort_mode == "score_asc":
            rows.sort(key=lambda c: c.total_score)
        elif sort_mode == "start_asc":
            rows.sort(key=lambda c: (c.session_id, c.start_frame, c.end_frame))
        else:
            rows.sort(key=lambda c: c.total_score, reverse=True)

        self._visible_candidates = rows
        self._populate_candidate_table()
        self._update_summary()
        self._update_filter_btn_text()

        has_rows = bool(self._visible_candidates)
        self._empty_label.setVisible(not has_rows)
        if not has_rows:
            self._candidate_label.setText("No candidates match this filter")
            self._player.close_clip()
            self._id_label.setText("Candidate ID: N/A")
            self._score_label.setText("Score: N/A")
            self._clip_label.setText("Clip: N/A")
            self._current_candidate_idx = -1
            return

        next_idx = self._current_candidate_idx if 0 <= self._current_candidate_idx < len(self._visible_candidates) else 0
        self._load_candidate(next_idx, select_row=True)

    @staticmethod
    def _candidate_from_decision(decision: ReviewDecision) -> _ReviewListRow:
        clip_id = str(decision.clip_id or "").strip()
        start_frame = int(decision.adjusted_start_frame or 0)
        end_frame = int(decision.adjusted_end_frame or start_frame)

        # Extract session_id from clip_id — handles seg_feedback_*, rand_*, and
        # any other format that embeds session_<hex> somewhere in the ID.
        ms = re.search(r"(session_[a-f0-9]+)", clip_id)
        session_id = ms.group(1) if ms else "unknown_session"

        return _ReviewListRow(
            window_id=clip_id,
            session_id=session_id,
            behavior_id=str(decision.behavior_label or "") or None,
            start_frame=start_frame,
            end_frame=end_frame,
            total_score=0.0,
            clip_path=None,
        )

    def _populate_candidate_table(self) -> None:
        self._candidate_table.setRowCount(0)
        al_fp_fn_active = bool(
            hasattr(self, "_show_al_fp_fn_btn") and self._show_al_fp_fn_btn.isChecked()
        )
        for cand in self._visible_candidates:
            row = self._candidate_table.rowCount()
            self._candidate_table.insertRow(row)
            subject = self._display_subject_map.get(cand.session_id, cand.session_id) or cand.session_id
            effective_bid = self._effective_behavior_id(cand)
            bname = self._resolve_behavior_display_name(effective_bid) if effective_bid else "—"
            occ_item = QTableWidgetItem(str(self._display_occurrence.get(cand.window_id, "")))
            occ_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            score_item = QTableWidgetItem(f"{cand.total_score:.3f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            dec = self._decision_by_clip_id.get(cand.window_id)
            # When the AL FP/FN filter is active, label FP/FN status takes priority
            # over the review decision so the user can see which rows are which.
            is_fp = cand.window_id in self._al_fp_ids
            is_fn = cand.window_id in self._al_fn_ids

            # FP/FN column: show FP, FN, FP/FN, or blank
            fp_fn_parts: list[str] = []
            if is_fp:
                fp_fn_parts.append("FP")
            if is_fn:
                fp_fn_parts.append("FN")
            fp_fn_text = "/".join(fp_fn_parts)

            # Decision column: stale FP/FN clips (label predates the model run) show blank
            # so the user knows these need to be re-evaluated. Clips already re-reviewed
            # after the model ran show their actual saved decision.
            post_reviewed = cand.window_id in getattr(self, "_al_post_pred_reviewed_ids", set())
            is_stale_fp_fn = al_fp_fn_active and (is_fp or is_fn) and not post_reviewed
            if is_stale_fp_fn:
                dec_text = "—"
            elif dec:
                dec_text = dec.decision.value
            else:
                dec_text = "—"
            self._candidate_table.setItem(row, 0, QTableWidgetItem(subject))
            self._candidate_table.setItem(row, 1, QTableWidgetItem(bname))
            self._candidate_table.setItem(row, 2, occ_item)
            self._candidate_table.setItem(row, 3, score_item)
            source_text, source_color, source_tip = self._candidate_source_display(cand)
            source_item = QTableWidgetItem(source_text)
            source_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if source_color is not None:
                source_item.setForeground(source_color)
            source_item.setToolTip(source_tip)
            self._candidate_table.setItem(row, 4, source_item)
            self._candidate_table.setItem(row, 5, QTableWidgetItem("yes" if self._candidate_clip_path(cand) else "no"))
            fp_fn_item = QTableWidgetItem(fp_fn_text)
            fp_fn_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._candidate_table.setItem(row, 6, fp_fn_item)
            self._candidate_table.setItem(row, 7, QTableWidgetItem(dec_text))

    def _candidate_source_display(self, candidate) -> "tuple[str, QColor | None, str]":
        """Return (label, color, tooltip) for a candidate's Source column.

        Prefers the explicit provenance ``source`` field (mapped to a friendly
        label); falls back to the active-learning ``selection_reason`` when no
        source was recorded — the common candidate-generation case whose windows
        leave ``source`` blank.
        """
        src = (getattr(candidate, "source", "") or "").strip()
        reason = (getattr(candidate, "selection_reason", "") or "").strip()
        if src:
            label = self._SOURCE_DISPLAY_LABELS.get(src) or src.replace("_", " ").title()
        else:
            label = reason or "—"
        color = None
        if src == "clip_mining":
            color = QColor("#00796B")
        elif src in ("umap_interactive_selection", "umap_selection"):
            color = QColor("#6A1B9A")
        elif src == "temporal_bout_review":
            color = QColor("#00838F")
        elif src == "quality_check":
            color = QColor("#00838F")
        elif not src and reason in ("hard_negative", "confound_boundary"):
            color = QColor("#E65100")
        elif not src and reason == "uncertainty":
            color = QColor("#1565C0")
        elif not src and reason == "disagreement":
            color = QColor("#6A1B9A")
        elif not src and reason == "exploration":
            color = QColor("#2E7D32")
        tip = f"Source: {src or '—'}\nSelection reason: {reason or '—'}"
        return label, color, tip

    def _rebuild_display_maps(self) -> None:
        """Build subject, behavior-name, and occurrence maps from all loaded candidates."""
        self._display_subject_map = {}
        self._display_behavior_name_map = {}
        self._display_occurrence = {}

        if self._project_root:
            manifest = self._imports.load_manifest(self._project_root)
            if manifest:
                video_by_id = {v.asset_id: v for v in manifest.videos}
                for session in manifest.linked_sessions:
                    subject = (session.subject_id or "").strip()
                    if not subject:
                        video = video_by_id.get(session.video_asset_id)
                        subject = (video.subject_id or "").strip() if video else ""
                    self._display_subject_map[session.session_id] = subject or session.session_id

        self._display_behavior_name_map = {
            b.behavior_id: b.name for b in self._behavior_service.behaviors
        }
        self._display_behavior_name_map[UNASSIGNED_BEHAVIOR_ID] = "(unassigned)"

        sorted_cands = sorted(
            self._all_candidates,
            key=lambda c: (c.session_id, c.behavior_id or "", int(c.start_frame)),
        )
        counters: dict[tuple[str, str], int] = {}
        for c in sorted_cands:
            key = (c.session_id, c.behavior_id or "")
            counters[key] = counters.get(key, 0) + 1
            self._display_occurrence[c.window_id] = counters[key]

    def _accept_all(self) -> None:
        """Accept every currently visible candidate, with a confirmation prompt."""
        visible = self._visible_candidates
        if not visible:
            return
        answer = QMessageBox.question(
            self,
            "Accept All Visible",
            f"Accept all {len(visible)} visible candidate(s)?\n\n"
            "This will overwrite any existing decisions for these candidates.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        for cand in visible:
            rec = self._review_service.upsert_decision(
                clip_id=cand.window_id,
                reviewer=reviewer,
                decision=ReviewDecisionType.ACCEPT,
                behavior_label=cand.behavior_id,
                notes="",
                adjusted_start_frame=int(cand.start_frame),
                adjusted_end_frame=int(cand.end_frame),
            )
            self._decision_by_clip_id[cand.window_id] = rec
        self._apply_filter()

    def _dismiss_undecided_candidates(self) -> None:
        """Remove or confirm candidates that show "—" in the Decision column.

        Handles three kinds of "undecided" entries:

        1. Persisted candidates with no saved decision: removed from the
           candidate store via remove_segment_candidates.
        2. Legacy virtual temporal-feedback rows (seg_feedback_* from
           feedback_intervals.json): source intervals are pruned from disk.
        3. Stale AL FP/FN entries: they have decisions but those decisions
           pre-date the last prediction run, so the table shows "—".
           Dismissing re-saves their existing decision with a fresh timestamp
           so they transition from "—" to showing their actual decision.

        Clip files are not deleted.
        """
        if not self._project_root:
            return

        # ── Case 1: persisted candidates with no decision ──────────────────
        undecided_persisted = [
            c for c in self._all_candidates
            if c.window_id not in self._decision_by_clip_id
        ]

        # ── Case 2: virtual seg_feedback_* rows with no decision ───────────
        persisted_ids = {c.window_id for c in self._all_candidates}
        undecided_virtual = [
            c for c in self._visible_candidates
            if c.window_id.startswith("seg_feedback_")
            and c.window_id not in persisted_ids
            and c.window_id not in self._decision_by_clip_id
        ]

        # ── Case 3: stale AL FP/FN entries ─────────────────────────────────
        # These show "—" because their label timestamp is older than the last
        # prediction file.  Re-saving the existing decision touches the
        # timestamp, making them "post-run reviewed" so the table shows the
        # decision rather than "—".
        al_fp_fn_active = bool(
            hasattr(self, "_show_al_fp_fn_btn") and self._show_al_fp_fn_btn.isChecked()
        )
        al_all_ids = getattr(self, "_al_fp_ids", set()) | getattr(self, "_al_fn_ids", set())
        post_reviewed_ids = getattr(self, "_al_post_pred_reviewed_ids", set())
        stale_al_entries = [
            c for c in self._visible_candidates
            if c.window_id in al_all_ids
            and c.window_id not in post_reviewed_ids
            and c.window_id in self._decision_by_clip_id
        ] if al_fp_fn_active else []

        if not undecided_persisted and not undecided_virtual and not stale_al_entries:
            QMessageBox.information(
                self, "Dismiss Undecided", "No undecided or stale entries found."
            )
            return

        total = len(undecided_persisted) + len(undecided_virtual) + len(stale_al_entries)
        lines = []
        if undecided_persisted:
            lines.append(f"  • {len(undecided_persisted)} unreviewed candidate(s) removed from queue")
        if undecided_virtual:
            lines.append(f"  • {len(undecided_virtual)} temporal-feedback tile(s) cleared")
        if stale_al_entries:
            lines.append(
                f"  • {len(stale_al_entries)} stale FP/FN label(s) confirmed as-is\n"
                "    (existing decisions re-saved with a fresh timestamp)"
            )
        answer = QMessageBox.question(
            self,
            "Dismiss Undecided",
            f"Dismiss {total} unresolved entry(ies)?\n\n" + "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        reviewer = (self._reviewer_input.text() or "reviewer").strip()

        # ── Case 1: remove persisted undecided ─────────────────────────────
        removed_persisted = 0
        if undecided_persisted:
            removed_persisted = self._candidate_service.remove_segment_candidates(
                [c.window_id for c in undecided_persisted]
            )

        # ── Case 2: prune temporal-feedback source intervals ───────────────
        removed_virtual = 0
        if undecided_virtual:
            import json as _json
            import re as _re

            tr_root = self._project_root / "derived" / "temporal_refinement"
            folder_map: dict[str, Path] = {
                p.name: p for p in tr_root.iterdir() if p.is_dir()
            }

            def _safe(val: str) -> str:
                return "".join(
                    ch if ch.isalnum() or ch in {"_", "-"} else "_"
                    for ch in str(val).strip()
                )

            from collections import defaultdict
            by_concept: dict[str, list] = defaultdict(list)
            for row in undecided_virtual:
                cid = str(row.behavior_id or "").strip()
                by_concept[cid].append(row)

            for concept_id, concept_rows in by_concept.items():
                folder = folder_map.get(_safe(concept_id)) or folder_map.get(concept_id)
                if folder is None:
                    removed_virtual += len(concept_rows)
                    continue
                fb_path = folder / "feedback_intervals.json"
                if not fb_path.exists():
                    removed_virtual += len(concept_rows)
                    continue
                try:
                    payload = _json.loads(fb_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                to_remove: set[tuple] = set()
                pat = _re.compile(
                    r"^seg_feedback_(session_[a-f0-9]+)_(-?\d+)_(-?\d+)$"
                )
                for row in concept_rows:
                    m = pat.match(str(row.window_id))
                    if m:
                        to_remove.add((m.group(1), int(m.group(2)), int(m.group(3))))
                        removed_virtual += 1

                def _touches(src_start: int, src_end: int, session: str) -> bool:
                    ws = 60
                    for ts in range(src_start, src_end, ws):
                        te = ts + ws - 1
                        if (session, ts, te) in to_remove:
                            return True
                    return False

                for key in ("false_positive_intervals_by_session",
                            "false_negative_intervals_by_session"):
                    by_session = payload.get(key) or {}
                    new_by_session: dict[str, list] = {}
                    for session, intervals in by_session.items():
                        kept = [
                            iv for iv in intervals
                            if not _touches(int(iv[0]), int(iv[1]), session)
                        ]
                        if kept:
                            new_by_session[session] = kept
                    payload[key] = new_by_session

                fb_path.write_text(
                    _json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        # ── Case 3: re-confirm stale AL FP/FN labels ───────────────────────
        dismissed_stale = 0
        if stale_al_entries:
            for cand in stale_al_entries:
                dec = self._decision_by_clip_id[cand.window_id]
                behavior_label = str(dec.behavior_label or "").strip()
                rec = self._review_service.upsert_decision(
                    clip_id=cand.window_id,
                    reviewer=reviewer,
                    decision=dec.decision,
                    behavior_label=behavior_label,
                    notes=str(dec.notes or ""),
                    adjusted_start_frame=int(cand.start_frame),
                    adjusted_end_frame=int(cand.end_frame),
                )
                self._decision_by_clip_id[cand.window_id] = rec
                review_label = self._decision_to_review_label(dec.decision, behavior_label)
                if review_label:
                    self._review_service.append_segment_label(
                        ReviewerLabelRecord(
                            segment_id=cand.window_id,
                            review_label=review_label,
                            reviewer_id=reviewer,
                            confidence=1.0,
                            notes=str(dec.notes or ""),
                        )
                    )
                dismissed_stale += 1

        self._refresh_candidates()
        total_done = removed_persisted + removed_virtual + dismissed_stale
        QMessageBox.information(
            self,
            "Dismiss Undecided",
            f"Done: {total_done} entry(ies) resolved.",
        )

    def _clear_unreviewed_clips(self) -> None:
        """Delete extracted clip files for candidates without a saved decision."""
        if not self._project_root:
            return
        undecided = [c for c in self._all_candidates if c.window_id not in self._decision_by_clip_id]
        if not undecided:
            QMessageBox.information(self, "Clear Unreviewed Clips", "All candidates already have decisions.")
            return

        clip_paths: list[Path] = []
        for cand in undecided:
            clip = self._candidate_clip_path(cand)
            if clip:
                clip_paths.append(Path(clip))

        existing_paths = [p for p in clip_paths if p.exists()]
        if not existing_paths:
            QMessageBox.information(
                self,
                "Clear Unreviewed Clips",
                "No extracted clip files were found for undecided candidates.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Clear Unreviewed Clips",
            f"Delete {len(existing_paths)} extracted clip file(s) for undecided candidates?\n\n"
            "This does not remove candidates or decisions.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        failed = 0
        touched_dirs: set[Path] = set()
        for path in existing_paths:
            try:
                path.unlink()
                removed += 1
                touched_dirs.add(path.parent)
            except Exception:
                failed += 1

        # Best-effort cleanup of empty session clip folders.
        for d in sorted(touched_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except Exception:
                pass

        self._refresh_candidates()
        QMessageBox.information(
            self,
            "Clear Unreviewed Clips",
            f"Removed {removed} clip file(s)."
            + (f" Could not remove {failed} file(s)." if failed else ""),
        )

    def _clear_bouts_missing_clips(self) -> None:
        """Remove persisted candidates whose extracted clip files are missing."""
        if not self._project_root:
            return

        missing = [c for c in self._all_candidates if not self._candidate_clip_path(c)]
        if not missing:
            QMessageBox.information(
                self,
                "Clear Bouts Missing Clips",
                "No candidate bouts are missing clip files.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Clear Bouts Missing Clips",
            f"Remove {len(missing)} candidate bout(s) that are missing extracted clips?\n\n"
            "This updates the candidate list only and does not delete saved decisions.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed = self._candidate_service.remove_segment_candidates([c.window_id for c in missing])
        self._refresh_candidates()
        QMessageBox.information(
            self,
            "Clear Bouts Missing Clips",
            f"Removed {removed} candidate bout(s) from the review list.",
        )

    def _regenerate_missing_clips(self) -> None:
        """Re-extract clip files for candidates whose clips are missing.

        Operates on the selected rows when any are selected, otherwise on every
        candidate in the queue that is missing a clip file. Runs in a background
        thread so the UI stays responsive.
        """
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return

        # Only treat rows as "selected" when the user explicitly highlighted
        # them; otherwise regenerate every missing clip in the queue.
        selected_rows = sorted({idx.row() for idx in self._candidate_table.selectionModel().selectedRows()})
        selected = [self._visible_candidates[r] for r in selected_rows if 0 <= r < len(self._visible_candidates)]
        scope_all = not selected
        pool = self._all_candidates if scope_all else selected
        missing = [c for c in pool if not self._candidate_clip_path(c)]

        if not missing:
            QMessageBox.information(
                self,
                "Regenerate Missing Clips",
                "No missing clips found for "
                + ("the review queue." if scope_all else "the selected candidate(s)."),
            )
            return

        scope_text = (
            f"all {len(missing)} candidate(s) in the queue that are missing clips"
            if scope_all
            else f"{len(missing)} of the selected candidate(s) that are missing clips"
        )
        answer = QMessageBox.question(
            self,
            "Regenerate Missing Clips",
            f"Re-extract clips for {scope_text}?\n\n"
            "This decodes video for each window and may take a while for large "
            "batches. Extraction settings match the Clip Extraction tab.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._regenerate_clips_btn.setEnabled(False)
        self._regenerate_clips_btn.setText(f"Regenerating 0/{len(missing)}…")

        # CandidateWindow-shaped copies are safe to hand to the worker thread.
        windows = list(missing)

        def _progress(done: int, total: int) -> None:
            # Emitted from the worker thread; relayed to the UI thread via signal.
            self._regen_progress.emit(int(done), int(total))

        worker = TaskWorker(
            regenerate_clips_for_windows,
            self._project_root,
            windows,
            progress_callback=_progress,
        )
        worker.signals.finished.connect(self._on_regenerate_complete)
        worker.signals.failed.connect(self._on_regenerate_failed)
        self._pool.start(worker)

    @Slot(int, int)
    def _on_regen_progress(self, done: int, total: int) -> None:
        self._regenerate_clips_btn.setText(f"Regenerating {done}/{total}…")

    @Slot(object)
    def _on_regenerate_complete(self, summary: dict) -> None:
        self._regenerate_clips_btn.setEnabled(True)
        self._regenerate_clips_btn.setText("Regenerate Missing Clips")

        extracted = int(summary.get("extracted", 0))
        requested = int(summary.get("requested", 0))
        warnings = [str(w) for w in summary.get("warnings", [])]

        self._refresh_candidates()

        message = f"Regenerated {extracted} of {requested} requested clip(s)."
        if extracted < requested:
            message += (
                "\n\nSome clips could not be rebuilt (missing source video, "
                "unresolved session, or decode error)."
            )
        if warnings:
            preview = "\n".join(f"• {w}" for w in warnings[:8])
            extra = f"\n…and {len(warnings) - 8} more." if len(warnings) > 8 else ""
            message += f"\n\nWarnings:\n{preview}{extra}"

        QMessageBox.information(self, "Regenerate Missing Clips", message)

    @Slot(str)
    def _on_regenerate_failed(self, tb: str) -> None:
        self._regenerate_clips_btn.setEnabled(True)
        self._regenerate_clips_btn.setText("Regenerate Missing Clips")
        logger.error("Clip regeneration failed:\n%s", tb)
        QMessageBox.critical(
            self,
            "Regenerate Missing Clips",
            "Clip regeneration failed — see log for details.",
        )

    def _selected_visible_candidates(self) -> list:
        selected = self._candidate_table.selectionModel().selectedRows()
        if not selected and 0 <= self._current_candidate_idx < len(self._visible_candidates):
            return [self._visible_candidates[self._current_candidate_idx]]
        rows = sorted({idx.row() for idx in selected})
        return [self._visible_candidates[r] for r in rows if 0 <= r < len(self._visible_candidates)]

    def _apply_batch_decision(self, decision: ReviewDecisionType) -> None:
        candidates = self._selected_visible_candidates()
        if not candidates:
            QMessageBox.warning(self, "No Selection", "Select one or more candidate rows first.")
            return
        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        notes = self._notes_edit.toPlainText().strip()
        confidence = float(self._confidence_spin.value())
        selected_label = self._normalize_behavior_id(str(self._label_combo.currentData() or "")) or self._default_behavior_id()

        for cand in candidates:
            rec = self._review_service.upsert_decision(
                clip_id=cand.window_id,
                reviewer=reviewer,
                decision=decision,
                behavior_label=selected_label,
                notes=notes,
                confidence_override=confidence,
                adjusted_start_frame=int(cand.start_frame),
                adjusted_end_frame=int(cand.end_frame),
            )
            self._decision_by_clip_id[cand.window_id] = rec

            review_label = self._decision_to_review_label(decision, selected_label)
            if review_label:
                self._review_service.append_segment_label(
                    ReviewerLabelRecord(
                        segment_id=cand.window_id,
                        review_label=review_label,
                        reviewer_id=reviewer,
                        confidence=confidence,
                        notes=notes,
                    )
                )

        self._apply_filter()

    def _remove_selected_candidates(self) -> None:
        """Delete selected candidates, their clip files, and all saved decisions."""
        candidates = self._selected_visible_candidates()
        if not candidates:
            QMessageBox.warning(self, "No Selection", "Select one or more candidate rows first.")
            return

        clip_ids = [c.window_id for c in candidates]
        has_decisions = any(cid in self._decision_by_clip_id for cid in clip_ids)
        clip_paths = [p for c in candidates for p in [self._candidate_clip_path(c)] if p]

        detail_lines = [f"  • {len(candidates)} candidate(s) removed from review queue"]
        if clip_paths:
            detail_lines.append(f"  • {len(clip_paths)} clip file(s) deleted")
        if has_decisions:
            detail_lines.append("  • Saved decisions and reviewer labels deleted")

        answer = QMessageBox.question(
            self,
            "Remove Selected",
            "This will permanently erase the selected entries as if they were never reviewed:\n\n"
            + "\n".join(detail_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Delete clip files
        deleted_clips = 0
        touched_dirs: set[Path] = set()
        for p in clip_paths:
            try:
                Path(p).unlink(missing_ok=True)
                deleted_clips += 1
                touched_dirs.add(Path(p).parent)
            except Exception:
                pass
        for d in sorted(touched_dirs, key=lambda x: len(x.parts), reverse=True):
            try:
                d.rmdir()
            except Exception:
                pass

        # Delete decisions + reviewer labels
        self._review_service.delete_decisions(clip_ids)
        for cid in clip_ids:
            self._decision_by_clip_id.pop(cid, None)

        # Remove candidate entries
        removed = self._candidate_service.remove_segment_candidates(clip_ids)
        self._refresh_candidates()
        QMessageBox.information(
            self,
            "Remove Selected",
            f"Removed {removed} candidate(s), deleted {deleted_clips} clip file(s).",
        )

    def _reassign_selected(self) -> None:
        candidates = self._selected_visible_candidates()
        if not candidates:
            QMessageBox.warning(self, "No Selection", "Select one or more candidate rows first.")
            return
        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        notes = self._notes_edit.toPlainText().strip()
        confidence = float(self._confidence_spin.value())
        selected_label = self._normalize_behavior_id(str(self._label_combo.currentData() or "")) or self._default_behavior_id()

        for cand in candidates:
            rec = self._review_service.upsert_decision(
                clip_id=cand.window_id,
                reviewer=reviewer,
                decision=ReviewDecisionType.RELABEL,
                behavior_label=selected_label,
                notes=notes,
                confidence_override=confidence,
                adjusted_start_frame=int(cand.start_frame),
                adjusted_end_frame=int(cand.end_frame),
            )
            self._decision_by_clip_id[cand.window_id] = rec
            self._review_service.append_segment_label(
                ReviewerLabelRecord(
                    segment_id=cand.window_id,
                    review_label=selected_label,
                    reviewer_id=reviewer,
                    confidence=confidence,
                    notes=notes,
                )
            )

        self._populate_candidate_table()
        self._update_summary()
        self._load_candidate(self._current_candidate_idx if self._current_candidate_idx >= 0 else 0, select_row=True)

    def _bulk_assign_behavior(self) -> None:
        """Assign a chosen behavior (or several, in co-occurring mode) to a batch.

        Scope is the explicitly-selected rows, or the entire visible queue when
        nothing is selected — so the whole mined queue can be reclassified in one
        click. Persists in a single batched write.
        """
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return

        selected_rows = sorted({idx.row() for idx in self._candidate_table.selectionModel().selectedRows()})
        scope = [self._visible_candidates[r] for r in selected_rows if 0 <= r < len(self._visible_candidates)]
        if not scope:
            scope = list(self._visible_candidates)
        if not scope:
            QMessageBox.information(self, "Bulk Assign", "No clips in the queue to assign.")
            return

        behaviors = [
            (str(b.behavior_id), b.name)
            for b in self._behavior_service.behaviors
            if getattr(b, "is_active", True)
        ]
        if not behaviors:
            QMessageBox.warning(self, "Bulk Assign", "No active behaviors are defined.")
            return

        preselect = {self._normalize_behavior_id(str(self._label_combo.currentData() or ""))}
        dlg = _BulkAssignDialog(
            behaviors,
            allow_multi=bool(self._co_occurring_enabled),
            n_clips=len(scope),
            preselect={p for p in preselect if p},
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        chosen = [self._normalize_behavior_id(b) or b for b in dlg.selected_ids()]
        non_none = sorted({b for b in chosen if b and b != NO_BEHAVIOR_ID})
        if non_none:
            label = "|".join(non_none) if len(non_none) > 1 else non_none[0]
            decision = ReviewDecisionType.ACCEPT
        else:
            label = NO_BEHAVIOR_ID
            decision = ReviewDecisionType.REJECT

        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        notes = self._notes_edit.toPlainText().strip()
        confidence = float(self._confidence_spin.value())
        review_label = self._decision_to_review_label(decision, label)

        specs: list[dict] = []
        label_records: list[ReviewerLabelRecord] = []
        for cand in scope:
            specs.append(
                {
                    "clip_id": cand.window_id,
                    "reviewer": reviewer,
                    "decision": decision,
                    "behavior_label": label,
                    "notes": notes,
                    "confidence_override": confidence,
                    "adjusted_start_frame": int(cand.start_frame),
                    "adjusted_end_frame": int(cand.end_frame),
                }
            )
            if review_label:
                label_records.append(
                    ReviewerLabelRecord(
                        segment_id=cand.window_id,
                        review_label=review_label,
                        reviewer_id=reviewer,
                        confidence=confidence,
                        notes=notes,
                    )
                )

        records = self._review_service.upsert_decisions_bulk(specs)
        for rec in records:
            self._decision_by_clip_id[rec.clip_id] = rec
        if label_records:
            self._review_service.replace_segment_labels(label_records)

        self._apply_filter()
        display = self._resolve_behavior_display_name(label)
        QMessageBox.information(
            self, "Bulk Assign", f"Assigned “{display}” to {len(scope)} clip(s)."
        )

    def _candidate_clip_path(self, candidate) -> str | None:
        clip_path = (candidate.clip_path or "").strip() if candidate.clip_path else ""
        if clip_path and Path(clip_path).exists():
            return clip_path
        if not self._project_root:
            return None
        legacy = self._project_root / "derived" / "clips" / candidate.session_id / f"{candidate.window_id}.mp4"
        if legacy.exists():
            return str(legacy)
        safe_name = ClipExtractionService.clip_filename_for_id(candidate.window_id)
        guessed = self._project_root / "derived" / "clips" / candidate.session_id / f"{safe_name}.mp4"
        return str(guessed) if guessed.exists() else None

    def _on_table_selection_changed(self) -> None:
        selected = self._candidate_table.selectionModel().selectedRows()
        if not selected:
            return
        self._load_candidate(selected[0].row(), select_row=False)

    def _on_review_control_changed(self) -> None:
        """Mark that the user has intentionally changed a review control for the current clip."""
        if not self._loading_candidate:
            self._review_dirty = True

    def _load_candidate(self, idx: int, select_row: bool = False) -> None:
        if idx < 0 or idx >= len(self._visible_candidates):
            return

        self._review_dirty = False
        self._loading_candidate = True
        self._current_candidate_idx = idx
        candidate = self._visible_candidates[idx]
        subject = self._display_subject_map.get(candidate.session_id, candidate.session_id) or candidate.session_id
        occ = self._display_occurrence.get(candidate.window_id, idx + 1)
        self._candidate_label.setText(f"Candidate {idx + 1} / {len(self._visible_candidates)}")
        self._id_label.setText(f"Subject: {subject}  |  Segment: {occ}")
        self._id_label.setToolTip(f"Internal segment id: {candidate.window_id}")
        self._score_label.setText(f"Score: {candidate.total_score:.3f}  |  Subject: {subject}")

        clip = self._candidate_clip_path(candidate)
        self._clip_label.setText(f"Clip: {clip or 'missing (extract clips first)'}")
        if clip:
            self._player.load_clip(clip)
            if self._autoplay_chk.isChecked() and not self._player._playing:
                self._player.toggle_play()
        else:
            self._player.close_clip()

        self._refresh_soundboard_for_clip()

        decision = self._decision_by_clip_id.get(candidate.window_id)
        start_value = int(candidate.start_frame)
        end_value = int(candidate.end_frame)

        # If this is a stale FP/FN clip (label predates the current model run), treat it
        # as unreviewed so the user makes a fresh, deliberate decision each time.
        show_al_fp_fn = bool(
            hasattr(self, "_show_al_fp_fn_btn") and self._show_al_fp_fn_btn.isChecked()
        )
        is_fp = candidate.window_id in self._al_fp_ids
        is_fn = candidate.window_id in self._al_fn_ids
        post_reviewed = candidate.window_id in getattr(self, "_al_post_pred_reviewed_ids", set())
        treat_as_fresh = show_al_fp_fn and (is_fp or is_fn) and not post_reviewed

        if decision and not treat_as_fresh:
            decision_order = [
                ReviewDecisionType.ACCEPT,
                ReviewDecisionType.REJECT,
                ReviewDecisionType.RELABEL,
                ReviewDecisionType.SKIP,
            ]
            fallback_idx = next(
                (i for i, d in enumerate(decision_order) if d == ReviewDecisionType.SKIP),
                0,
            )
            combo_idx = next(
                (
                    i
                    for i, d in enumerate(decision_order)
                    if d == decision.decision
                ),
                fallback_idx,
            )
            self._decision_combo.setCurrentIndex(combo_idx)
            self._notes_edit.setPlainText(decision.notes or "")
            label_idx = self._label_combo.findData(self._normalize_behavior_id(decision.behavior_label))
            if label_idx >= 0:
                self._label_combo.setCurrentIndex(label_idx)
            start_value = int(decision.adjusted_start_frame) if decision.adjusted_start_frame is not None else start_value
            end_value = int(decision.adjusted_end_frame) if decision.adjusted_end_frame is not None else end_value
        else:
            self._decision_combo.setCurrentIndex(0)
            self._notes_edit.clear()
            # For stale FP/FN clips, pre-fill the label combo with the target behavior
            # being questioned (e.g. "dig" for a FP/FN against the dig model) so the
            # user just confirms or corrects it rather than hunting through the list.
            if treat_as_fresh:
                target_ids = (
                    self._al_fp_behavior_map.get(candidate.window_id)
                    or self._al_fn_behavior_map.get(candidate.window_id)
                    or []
                )
                fallback_label = target_ids[0] if target_ids else self._default_behavior_id()
            else:
                fallback_label = self._effective_behavior_id(candidate)
                if fallback_label == UNASSIGNED_BEHAVIOR_ID:
                    fallback_label = self._default_behavior_id()
            label_idx = self._label_combo.findData(fallback_label)
            if label_idx >= 0:
                self._label_combo.setCurrentIndex(label_idx)

        self._start_frame_spin.blockSignals(True)
        self._end_frame_spin.blockSignals(True)
        self._start_frame_spin.setValue(start_value)
        self._end_frame_spin.setValue(max(start_value, end_value))
        self._start_frame_spin.blockSignals(False)
        self._end_frame_spin.blockSignals(False)

        # Restore pending labels for co-occurring mode
        if self._co_occurring_enabled:
            self._pending_labels.clear()
            if decision and not treat_as_fresh and decision.behavior_label:
                for bid in decision.behavior_label.split("|"):
                    bid = bid.strip()
                    if bid:
                        self._pending_labels.add(bid)
            self._update_pending_labels_display()

        if select_row:
            self._candidate_table.blockSignals(True)
            self._candidate_table.selectRow(idx)
            self._candidate_table.blockSignals(False)

        self._loading_candidate = False
        self._review_dirty = False

    def _load_previous(self) -> None:
        if self._current_candidate_idx > 0:
            self._save_decision_silent()
            self._load_candidate(self._current_candidate_idx - 1, select_row=True)

    def _load_next(self) -> None:
        if self._current_candidate_idx + 1 < len(self._visible_candidates):
            self._save_decision_silent()
            self._load_candidate(self._current_candidate_idx + 1, select_row=True)

    def _save_decision_silent(self) -> None:
        """Save the current decision without showing any warning dialogs.

        Called automatically when the user navigates away with arrow keys so
        the current combo/label state is always persisted before the clip changes.
        Auto-advance is suppressed so the arrow key controls where we go, not
        the save logic.
        Only saves if the user explicitly changed a control (_review_dirty is True).
        """
        if not self._visible_candidates or self._current_candidate_idx < 0:
            return
        if not self._review_dirty:
            return
        # Temporarily disable auto-advance so _save_decision does not also
        # move the index — the caller handles navigation.
        prev = self._autoplay_chk.isChecked()
        self._autoplay_chk.blockSignals(True)
        self._autoplay_chk.setChecked(False)
        try:
            self._save_decision()
        finally:
            self._autoplay_chk.setChecked(prev)
            self._autoplay_chk.blockSignals(False)

    def _save_with_decision(self, decision: ReviewDecisionType) -> None:
        idx = next((i for i in range(self._decision_combo.count()) if self._decision_combo.itemData(i) == decision), 0)
        self._decision_combo.setCurrentIndex(idx)
        self._save_decision()

    def _save_no_behavior_accept(self) -> None:
        label_idx = self._label_combo.findData(NO_BEHAVIOR_ID)
        if label_idx >= 0:
            self._label_combo.setCurrentIndex(label_idx)
        self._save_with_decision(ReviewDecisionType.ACCEPT)

    def _update_decision_cell(self, row_idx: int, decision: ReviewDecision) -> None:
        """Patch only the Decision column for the given row without rebuilding the table."""
        if row_idx < 0 or row_idx >= self._candidate_table.rowCount():
            return
        al_fp_fn_active = bool(
            hasattr(self, "_show_al_fp_fn_btn") and self._show_al_fp_fn_btn.isChecked()
        )
        cand = self._visible_candidates[row_idx] if row_idx < len(self._visible_candidates) else None
        post_reviewed = cand is not None and cand.window_id in getattr(self, "_al_post_pred_reviewed_ids", set())
        is_stale = (
            al_fp_fn_active
            and cand is not None
            and (cand.window_id in self._al_fp_ids or cand.window_id in self._al_fn_ids)
            and not post_reviewed
        )
        dec_text = "—" if is_stale else decision.decision.value
        item = self._candidate_table.item(row_idx, 7)
        if item is None:
            self._candidate_table.setItem(row_idx, 7, QTableWidgetItem(dec_text))
        else:
            item.setText(dec_text)

    def _save_decision(self) -> None:
        if not self._visible_candidates or self._current_candidate_idx < 0:
            QMessageBox.warning(self, "No Candidate", "No candidate selected.")
            return

        candidate = self._visible_candidates[self._current_candidate_idx]
        decision_type: ReviewDecisionType = self._decision_combo.currentData()
        reviewer = (self._reviewer_input.text() or "reviewer").strip()
        notes = self._notes_edit.toPlainText().strip()
        confidence = float(self._confidence_spin.value())
        adjusted_start = int(self._start_frame_spin.value())
        adjusted_end = int(self._end_frame_spin.value())
        if adjusted_end < adjusted_start:
            adjusted_end = adjusted_start
            self._end_frame_spin.setValue(adjusted_end)

        # Determine the label(s) to save
        if self._co_occurring_enabled and self._pending_labels:
            selected_label = "|".join(sorted(self._pending_labels))
        else:
            selected_label = self._normalize_behavior_id(str(self._label_combo.currentData() or "")) or self._default_behavior_id()

        rec = self._review_service.upsert_decision(
            clip_id=candidate.window_id,
            reviewer=reviewer,
            decision=decision_type,
            behavior_label=selected_label,
            notes=notes,
            confidence_override=confidence,
            adjusted_start_frame=adjusted_start,
            adjusted_end_frame=adjusted_end,
        )
        self._decision_by_clip_id[candidate.window_id] = rec

        review_label = self._decision_to_review_label(decision_type, selected_label)
        if review_label:
            self._review_service.append_segment_label(
                ReviewerLabelRecord(
                    segment_id=candidate.window_id,
                    review_label=review_label,
                    reviewer_id=reviewer,
                    confidence=confidence,
                    notes=notes,
                )
            )

        # Clear pending labels after save in co-occurring mode
        if self._co_occurring_enabled:
            self._pending_labels.clear()
            self._update_pending_labels_display()

        # Update the Decision column in-place so the clip stays in the list.
        # Do NOT call _apply_filter() here — the list only gets pruned when
        # the user explicitly clicks Refresh.
        self._update_decision_cell(self._current_candidate_idx, rec)
        self._update_summary()

        # Advance to the next clip only when auto-advance is enabled.
        if self._autoplay_chk.isChecked():
            next_idx = self._current_candidate_idx + 1
            if next_idx < len(self._visible_candidates):
                self._current_candidate_idx = next_idx
                self._load_candidate(self._current_candidate_idx, select_row=True)

    @staticmethod
    def _decision_to_review_label(decision: ReviewDecisionType, selected_label: str) -> str | None:
        if decision == ReviewDecisionType.ACCEPT:
            return selected_label
        if decision == ReviewDecisionType.REJECT:
            return NO_BEHAVIOR_ID
        if decision == ReviewDecisionType.AMBIGUOUS:
            return "ambiguous"
        if decision == ReviewDecisionType.RELABEL:
            return selected_label
        return None

    def _mark_boundary_error(self) -> None:
        idx = self._label_combo.findData("boundary_error")
        if idx >= 0:
            self._label_combo.setCurrentIndex(idx)
        self._decision_combo.setCurrentIndex(max(0, self._decision_combo.findData(ReviewDecisionType.RELABEL)))
        notes = self._notes_edit.toPlainText().strip()
        if "boundary" not in notes.lower():
            self._notes_edit.setPlainText((notes + "\nBoundary error reported.").strip())
        self._save_decision()

    def _update_summary(self) -> None:
        decisions = list(self._decision_by_clip_id.values())
        summary = self._review_service.summary(decisions)
        total_candidates = len(self._all_candidates)
        reviewed = summary.total
        visible_ids = {c.window_id for c in self._all_candidates}
        reviewed_not_in_candidates = sum(1 for d in decisions if d.clip_id not in visible_ids)
        pct = (100.0 * reviewed / max(1, total_candidates))
        skipped_total = int(summary.skipped) + int(summary.ambiguous)
        self._summary_label.setText(
            f"Progress: {reviewed}/{total_candidates} ({pct:.1f}%)  |  "
            f"accept={summary.accepted} reject={summary.rejected} "
            f"skipped={skipped_total}"
            + (
                f"  |  reviewed-not-in-current-candidates={reviewed_not_in_candidates}"
                if reviewed_not_in_candidates > 0
                else ""
            )
        )

    def _on_frame_override_changed(self) -> None:
        if self._end_frame_spin.value() < self._start_frame_spin.value():
            self._end_frame_spin.blockSignals(True)
            self._end_frame_spin.setValue(self._start_frame_spin.value())
            self._end_frame_spin.blockSignals(False)

    def _apply_frame_overrides(self) -> None:
        pass

    def _reset_frame_overrides(self) -> None:
        if not self._visible_candidates or self._current_candidate_idx < 0:
            return
        candidate = self._visible_candidates[self._current_candidate_idx]
        self._start_frame_spin.setValue(int(candidate.start_frame))
        self._end_frame_spin.setValue(int(candidate.end_frame))

    # ------------------------------------------------------------------
    # Dissimilarity analysis
    # ------------------------------------------------------------------

    def _run_dissimilarity_analysis(self) -> None:
        """Collect accepted clips for the current behavior filter and run
        dissimilarity analysis in a background thread."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return

        behavior_id = str(self._behavior_filter_combo.currentData() or "all")
        if behavior_id == "all":
            QMessageBox.information(
                self,
                "Select a Behavior",
                "Filter to a specific behavior before running dissimilarity analysis.",
            )
            return

        # Gather accepted clips for this behavior from visible candidates
        # (includes both loaded candidates and decision-sourced entries).
        reviewed_clips: list[dict] = []
        seen_ids: set[str] = set()
        for cand in self._visible_candidates:
            if cand.window_id in seen_ids:
                continue
            decision = self._decision_by_clip_id.get(cand.window_id)
            if not decision or decision.decision != ReviewDecisionType.ACCEPT:
                continue
            if self._effective_behavior_id(cand) != behavior_id:
                continue
            seen_ids.add(cand.window_id)
            reviewed_clips.append(
                {
                    "window_id": cand.window_id,
                    "session_id": cand.session_id,
                    "start_frame": int(cand.start_frame),
                    "end_frame": int(cand.end_frame),
                }
            )
        # Also check _all_candidates for accepted clips not in visible set.
        for cand in self._all_candidates:
            if cand.window_id in seen_ids:
                continue
            decision = self._decision_by_clip_id.get(cand.window_id)
            if not decision or decision.decision != ReviewDecisionType.ACCEPT:
                continue
            if self._effective_behavior_id(cand) != behavior_id:
                continue
            seen_ids.add(cand.window_id)
            reviewed_clips.append(
                {
                    "window_id": cand.window_id,
                    "session_id": cand.session_id,
                    "start_frame": int(cand.start_frame),
                    "end_frame": int(cand.end_frame),
                }
            )

        if len(reviewed_clips) < 3:
            QMessageBox.information(
                self,
                "Not Enough Clips",
                f"Need at least 3 accepted clips for this behavior — found {len(reviewed_clips)}.",
            )
            return

        self._flag_outliers_btn.setEnabled(False)
        self._flag_outliers_btn.setText("Analyzing…")

        worker = TaskWorker(
            run_dissimilarity_analysis,
            self._project_root,
            reviewed_clips,
            behavior_id,
        )
        worker.signals.finished.connect(self._on_dissimilarity_complete)
        worker.signals.failed.connect(self._on_dissimilarity_failed)
        self._pool.start(worker)

    @Slot(object)
    def _on_dissimilarity_complete(self, report: DissimilarityReport) -> None:
        self._flag_outliers_btn.setEnabled(True)
        self._flag_outliers_btn.setText("Flag Outliers (Dissimilarity)")

        if report.error:
            QMessageBox.warning(self, "Dissimilarity Error", report.error)
            return

        # Store scores keyed by window_id for table colouring.
        self._dissimilarity_scores = {r.window_id: r.score for r in report.results}

        # Re-sort visible candidates: scored clips from most to least dissimilar,
        # then unscored clips at the end in their original order.
        self._visible_candidates.sort(
            key=lambda c: (-self._dissimilarity_scores.get(c.window_id, -1.0),)
        )
        self._current_candidate_idx = 0
        self._populate_candidate_table()
        self._apply_dissimilarity_highlights()
        if self._visible_candidates:
            self._candidate_table.selectRow(0)
            self._load_candidate(0)

        behavior_name = self._display_behavior_name_map.get(
            report.behavior_id, report.behavior_id[:8]
        )
        unmatched = report.n_clips - report.n_matched
        unmatched_note = (
            f"\n\n{unmatched} clip(s) could not be matched to segment features "
            "(their sessions may not have features extracted yet)."
            if unmatched > 0
            else ""
        )
        QMessageBox.information(
            self,
            "Dissimilarity Analysis Complete",
            f"Behavior: {behavior_name}\n"
            f"Matched: {report.n_matched}/{report.n_clips} clips\n"
            f"Outliers flagged: {report.n_outliers}\n\n"
            "Outlier rows are highlighted in orange. "
            "Scores are shown in the Score column (0 = typical, 1 = most dissimilar)."
            + unmatched_note,
        )

    @Slot(str)
    def _on_dissimilarity_failed(self, tb: str) -> None:
        self._flag_outliers_btn.setEnabled(True)
        self._flag_outliers_btn.setText("Flag Outliers (Dissimilarity)")
        logger.error("Dissimilarity analysis failed:\n%s", tb)
        QMessageBox.critical(self, "Dissimilarity Failed", "Analysis failed — see log for details.")

    def _apply_dissimilarity_highlights(self) -> None:
        """Colour table rows by dissimilarity score and update the Score column."""
        if not self._dissimilarity_scores:
            return
        for row_idx, cand in enumerate(self._visible_candidates):
            if row_idx >= self._candidate_table.rowCount():
                break
            score = self._dissimilarity_scores.get(cand.window_id)
            if score is None:
                continue
            # Update Score column (col 3) to show dissimilarity score.
            score_item = self._candidate_table.item(row_idx, 3)
            if score_item:
                score_item.setText(f"D:{score:.2f}")
            # Highlight: orange for outliers (>=0.8), light yellow for elevated (>=0.5).
            if score >= 0.8:
                bg = QColor(255, 152, 0, 80)  # orange
            elif score >= 0.5:
                bg = QColor(255, 235, 59, 60)  # yellow
            else:
                bg = QColor(0, 0, 0, 0)  # transparent
            for col in range(self._candidate_table.columnCount()):
                item = self._candidate_table.item(row_idx, col)
                if item:
                    item.setBackground(bg)

    # -- essence-range flagging (Targeted Clip Mining → review queue) ---------

    def _flag_queue_by_essence(self, criteria: list[Criterion], match_all: bool) -> None:
        """Audit the clips in the current review filter against essence ranges.

        Called by the Targeted Clip Mining dialog. Computes pose metrics for the
        clips currently visible in the review queue and highlights the ones whose
        values fall *outside* the acceptable ranges (fail the essence test), then
        sorts those to the top so they can be re-reviewed. Unlike Flag Outliers
        (Dissimilarity), the cutoffs are the explicit target ranges the reviewer
        set — not a statistical distance.
        """
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        if self._essence_check_busy:
            QMessageBox.information(
                self, "Flag Failing Clips",
                "An essence check is already running. Please wait for it to finish.",
            )
            return

        # Snapshot the visible queue (deduped) as clips to test.
        refs: list[ClipRef] = []
        seen: set[str] = set()
        for cand in self._visible_candidates:
            wid = str(cand.window_id)
            if wid in seen:
                continue
            seen.add(wid)
            refs.append(
                ClipRef(wid, str(cand.session_id), int(cand.start_frame), int(cand.end_frame))
            )
        if not refs:
            QMessageBox.information(
                self, "Flag Failing Clips",
                "No clips in the current review filter to check.",
            )
            return

        self._essence_check_busy = True
        progress = QProgressDialog(
            f"Checking {len(refs)} clip(s) against the essence ranges…", None, 0, 0, self
        )
        progress.setWindowTitle("Flag Failing Clips")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._essence_progress_dialog = progress
        progress.show()

        worker = TaskWorker(
            self._run_essence_check, self._project_root, refs, criteria, match_all
        )
        worker.signals.finished.connect(self._on_essence_check_complete)
        worker.signals.failed.connect(self._on_essence_check_failed)
        self._pool.start(worker)

    @staticmethod
    def _run_essence_check(
        project_root, refs: list[ClipRef], criteria: list[Criterion], match_all: bool
    ) -> EssenceCheckResult:
        """Worker body: compute metrics for ``refs`` and audit them (off the UI thread)."""
        svc = ClipMetricsService()
        svc.set_project(project_root)
        df = svc.compute(refs)
        return svc.check_essence(df, criteria, match_all)

    def _close_essence_progress(self) -> None:
        dlg = getattr(self, "_essence_progress_dialog", None)
        if dlg is not None:
            try:
                dlg.close()
            except RuntimeError:
                pass
            self._essence_progress_dialog = None

    @Slot(object)
    def _on_essence_check_complete(self, result: EssenceCheckResult) -> None:
        self._essence_check_busy = False
        self._close_essence_progress()

        failed = {wid for wid in result.failed_ids}
        self._essence_fail_violations = dict(result.violations)

        # Sort failing clips to the top of the queue (stable within each group).
        self._visible_candidates.sort(
            key=lambda c: 0 if str(c.window_id) in failed else 1
        )
        self._current_candidate_idx = 0
        self._populate_candidate_table()
        self._apply_essence_highlights()
        if self._visible_candidates:
            self._candidate_table.selectRow(0)
            self._load_candidate(0)

        n_fail = len(failed)
        n_pass = len(result.passed_ids)
        no_data = len(result.no_data_ids)
        no_data_note = (
            f"\n\n{no_data} clip(s) had no pose data for the target metrics and "
            "could not be judged."
            if no_data
            else ""
        )
        if n_fail:
            QMessageBox.information(
                self,
                "Flag Failing Clips",
                f"{n_fail} of {result.n_evaluated} clip(s) fall OUTSIDE the essence "
                f"ranges and are highlighted in orange (sorted to the top).\n"
                f"{n_pass} clip(s) pass. Hover a flagged row to see which metric failed."
                + no_data_note,
            )
        else:
            QMessageBox.information(
                self,
                "Flag Failing Clips",
                f"All {n_pass} judged clip(s) fall within the essence ranges — "
                "nothing to flag." + no_data_note,
            )

    @Slot(str)
    def _on_essence_check_failed(self, tb: str) -> None:
        self._essence_check_busy = False
        self._close_essence_progress()
        logger.error("Essence-range flagging failed:\n%s", tb)
        QMessageBox.critical(
            self, "Flag Failing Clips", "Essence check failed — see log for details."
        )

    def _apply_essence_highlights(self) -> None:
        """Colour rows whose clips fell outside the essence ranges and add tooltips."""
        if not self._essence_fail_violations:
            return
        bg = QColor(255, 152, 0, 90)  # orange — matches the dissimilarity outlier tint
        for row_idx, cand in enumerate(self._visible_candidates):
            if row_idx >= self._candidate_table.rowCount():
                break
            viols = self._essence_fail_violations.get(str(cand.window_id))
            if not viols:
                continue
            tip = "Fails essence test:\n• " + "\n• ".join(viols)
            for col in range(self._candidate_table.columnCount()):
                item = self._candidate_table.item(row_idx, col)
                if item:
                    item.setBackground(bg)
                    item.setToolTip(tip)

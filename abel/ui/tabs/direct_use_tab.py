"""Direct Use workflow tab — select new data files, apply a trained snapshot."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from abel.services.direct_run_service import (
    STEP_IDS,
    STEP_LABELS,
    DirectRunProgress,
    DirectRunService,
)
from abel.services.import_service import ImportService
from abel.services import keypoint_mapping
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.roi_service import MAX_ROIS, ROI_COLORS, ROIService
from abel.ui.pixel_scale_calibration_dialog import PixelScaleCalibrationDialog
from abel.services.workflow_snapshot_service import (
    WorkflowSnapshot,
    WorkflowSnapshotService,
)
from abel.ui.tabs.roi_definition_tab import _ROICanvas, _roi_qcolors

logger = logging.getLogger("abel")

# ── Visual constants ────────────────────────────────────────────────
_CARD_PENDING = (
    "background: #0A1929; border: 1px solid #1E3A5F; border-radius: 6px;"
)
_CARD_ACTIVE = (
    "background: #0D2137; border: 2px solid #1976D2; border-radius: 6px;"
)
_CARD_DONE = (
    "background: #0A2E1A; border: 1px solid #2E7D32; border-radius: 6px;"
)
_CARD_ERROR = (
    "background: #2E0A0A; border: 1px solid #C62828; border-radius: 6px;"
)
_ICON_PENDING = "○"
_ICON_ACTIVE  = "◉"
_ICON_DONE    = "✓"
_ICON_ERROR   = "✕"

_BTN = (
    "QPushButton { background: #1A2A3A; color: #B0BEC5; font-size: 12px;"
    " border: 1px solid #2A4060; border-radius: 4px; padding: 4px 12px; }"
    "QPushButton:hover { background: #1E3A5F; }"
    "QPushButton:disabled { color: #546E7A; border-color: #1A2A3A; }"
)
_BTN_PRIMARY = (
    "QPushButton { background: #1565C0; color: white; font-size: 13px;"
    " font-weight: 700; border: none; border-radius: 5px; padding: 8px 20px; }"
    "QPushButton:hover { background: #1976D2; }"
    "QPushButton:disabled { background: #263238; color: #546E7A; }"
)


class _CollapsibleSection(QWidget):
    """A titled section whose body collapses to a single header row.

    Used to declutter the Direct Use tab: each pipeline step lives in its own
    collapsible section so the user can focus on one step at a time.  Build the
    step's widgets into :pyattr:`content_layout`.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._toggle.setStyleSheet(
            "QToolButton { color: #90CAF9; font-size: 13px; font-weight: 700;"
            " border: none; background: transparent; padding: 5px 2px;"
            " text-align: left; }"
            "QToolButton:hover { color: #BBDEFB; }"
        )
        self._toggle.clicked.connect(self._on_toggle)

        self._content = QWidget()
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(16, 2, 4, 8)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)
        self.setStyleSheet(
            "_CollapsibleSection { border-top: 1px solid #1E3A5F; }"
        )

    def _on_toggle(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)
        self._on_toggle(expanded)


class _StepCard(QWidget):
    """Single pipeline step card with icon, label, progress bar, and timing."""

    def __init__(self, step_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_id = step_id
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_CARD_PENDING)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._icon = QLabel(_ICON_PENDING)
        self._icon.setFixedWidth(26)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet(
            "font-size: 14px; color: #546E7A; background: transparent; border: none;"
        )

        self._label = QLabel(STEP_LABELS.get(step_id, step_id))
        self._label.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #90A4AE;"
            " background: transparent; border: none;"
        )

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        self._progress.setStyleSheet(
            "QProgressBar { background: #1A2027; border-radius: 2px; border: none; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #1565C0, stop:1 #42A5F5"
            "); border-radius: 2px; }"
        )

        self._detail = QLabel("")
        self._detail.setStyleSheet(
            "font-size: 10px; color: #78909C; background: transparent; border: none;"
        )
        self._detail.setWordWrap(True)

        self._timing = QLabel("")
        self._timing.setStyleSheet(
            "font-size: 10px; color: #546E7A; background: transparent; border: none;"
        )
        self._timing.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)
        top_row.addWidget(self._icon)
        top_row.addWidget(self._label, 1)
        top_row.addWidget(self._timing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(3)
        layout.addLayout(top_row)
        layout.addWidget(self._progress)
        layout.addWidget(self._detail)

    def set_pending(self, estimate_s: float = 0.0) -> None:
        self.setStyleSheet(_CARD_PENDING)
        self._icon.setText(_ICON_PENDING)
        self._icon.setStyleSheet(
            "font-size: 14px; color: #546E7A; background: transparent; border: none;"
        )
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            "QProgressBar { background: #1A2027; border-radius: 2px; border: none; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #1565C0, stop:1 #42A5F5"
            "); border-radius: 2px; }"
        )
        self._detail.setText("")
        self._detail.setStyleSheet(
            "font-size: 10px; color: #78909C; background: transparent; border: none;"
        )
        self._timing.setText(f"~{_fmt(estimate_s)}" if estimate_s > 0 else "")

    def set_active(self, pct: float, msg: str) -> None:
        self.setStyleSheet(_CARD_ACTIVE)
        self._icon.setText(_ICON_ACTIVE)
        self._icon.setStyleSheet(
            "font-size: 14px; color: #42A5F5; background: transparent; border: none;"
        )
        self._progress.setValue(int(pct * 100))
        self._detail.setText(msg)

    def set_done(self, elapsed_s: float) -> None:
        self.setStyleSheet(_CARD_DONE)
        self._icon.setText(_ICON_DONE)
        self._icon.setStyleSheet(
            "font-size: 14px; color: #66BB6A; background: transparent; border: none;"
        )
        self._progress.setValue(100)
        self._progress.setStyleSheet(
            "QProgressBar { background: #1A2027; border-radius: 2px; border: none; }"
            "QProgressBar::chunk { background: #2E7D32; border-radius: 2px; }"
        )
        self._detail.setText("Done")
        self._timing.setText(_fmt(elapsed_s))

    def set_error(self, msg: str) -> None:
        self.setStyleSheet(_CARD_ERROR)
        self._icon.setText(_ICON_ERROR)
        self._icon.setStyleSheet(
            "font-size: 14px; color: #EF5350; background: transparent; border: none;"
        )
        self._detail.setText(msg)
        self._detail.setStyleSheet(
            "font-size: 10px; color: #EF9A9A; background: transparent; border: none;"
        )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        return _fmt(seconds)


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


class _DirectRunWorker(QThread):
    """Background thread for running the direct-use pipeline."""

    progress = Signal(object)  # DirectRunProgress
    # NB: named ``done`` rather than ``finished`` — QThread already defines a
    # built-in ``finished`` signal, and shadowing it interferes with Qt's
    # thread-lifecycle management.
    done = Signal(dict)

    def __init__(self, target_root: Path, source_root: Path, snapshot: WorkflowSnapshot) -> None:
        super().__init__()
        self._target_root = target_root
        self._source_root = source_root
        self._snapshot    = snapshot
        self._service     = DirectRunService()

    def run(self) -> None:
        result = self._service.run(
            target_project_root=self._target_root,
            source_project_root=self._source_root,
            snapshot=self._snapshot,
            progress_cb=self.progress.emit,
        )
        self.done.emit(result)

    def cancel(self) -> None:
        self._service.cancel()


class DirectUseTab(QWidget):
    """Three-step Direct Use workflow:
      1. Source project (snapshot / trained model).
      2. Input data — pick video files + DLC pose files, choose output folder.
      3. Run pipeline — progress tracking with step cards.
    """

    pipeline_complete = Signal(Path)  # emitted with the output project root

    # ── Supported file extensions ─────────────────────────────────────
    _VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv)"
    _POSE_FILTER  = "DLC pose files (*.csv *.h5 *.hdf5)"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot:    WorkflowSnapshot | None = None
        self._source_root: Path | None             = None
        self._output_root: Path | None             = None
        self._video_paths: list[Path]              = []
        self._pose_paths:  list[Path]              = []
        self._worker:      _DirectRunWorker | None = None
        self._snap_svc    = WorkflowSnapshotService()
        self._import_svc  = ImportService()
        self._roi_svc     = ROIService()
        self._roi_config: dict = ROIService.default_config()
        self._roi_video_cap = None
        self._roi_n_frames: int = 0

        # Per-subject ROI state
        self._subject_rois: dict[str, dict] = {}  # subject_id → {"target_zones": [{…}], "subject_crop": {…}}
        self._previous_roi_subject_id: str = ""
        self._roi_subject_dirty: bool = False

        # Per-subject pixel/mm calibration (subject_id → pixels_per_mm).
        self._subject_pxmm: dict[str, float] = {}

        # Keypoint mapping: model expects these keypoints; new data provides
        # _data_keypoints.  _kp_combos maps model_keypoint → its QComboBox.
        self._pose_svc = PoseProcessingService()
        self._data_keypoints: list[str] = []
        self._kp_combos: dict[str, QComboBox] = {}

        # ── Header ───────────────────────────────────────────────────
        header = QLabel("Direct Use Workflow")
        header.setStyleSheet("font-size: 16px; font-weight: 800; color: #90CAF9;")
        desc = QLabel(
            "Apply a trained model from an existing project to new, unannotated data. "
            "Select the source project, add your video and DLC pose files, "
            "choose an output folder, then run."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #607D8B; padding-bottom: 2px;")

        # ── Step 1: Source project ────────────────────────────────────
        source_box = _CollapsibleSection("Step 1 — Source Project (trained model)")
        source_lay = source_box.content_layout
        source_lay.setSpacing(4)

        src_row = QHBoxLayout()
        self._source_btn = QPushButton("Browse…")
        self._source_btn.setStyleSheet(_BTN)
        self._source_btn.clicked.connect(self._select_source_project)
        self._source_info = QLabel("No source project selected.")
        self._source_info.setWordWrap(True)
        self._source_info.setStyleSheet("color: #78909C; font-size: 11px;")
        src_row.addWidget(self._source_btn)
        src_row.addWidget(self._source_info, 1)
        source_lay.addLayout(src_row)

        self._snap_detail = QLabel("")
        self._snap_detail.setWordWrap(True)
        self._snap_detail.setStyleSheet(
            "font-size: 11px; color: #B0BEC5; padding: 4px 0 0 2px;"
        )
        self._snap_detail.hide()
        source_lay.addWidget(self._snap_detail)

        # ── Step 2: Input data ────────────────────────────────────────
        data_box = _CollapsibleSection("Step 2 — Input Data")
        data_lay = data_box.content_layout
        data_lay.setSpacing(6)

        instr = QLabel(
            "Add the video file(s) and their matching DLC pose CSV/H5 file(s). "
            "Files are paired automatically by filename stem."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("font-size: 11px; color: #78909C;")
        data_lay.addWidget(instr)

        file_btns = QHBoxLayout()
        self._add_video_btn = QPushButton("+ Add Video(s)…")
        self._add_video_btn.setStyleSheet(_BTN)
        self._add_video_btn.clicked.connect(self._add_videos)
        self._add_pose_btn = QPushButton("+ Add DLC Pose File(s)…")
        self._add_pose_btn.setStyleSheet(_BTN)
        self._add_pose_btn.clicked.connect(self._add_poses)
        self._clear_btn = QPushButton("✕ Clear All")
        self._clear_btn.setStyleSheet(_BTN)
        self._clear_btn.clicked.connect(self._clear_input_files)
        file_btns.addWidget(self._add_video_btn)
        file_btns.addWidget(self._add_pose_btn)
        file_btns.addWidget(self._clear_btn)
        file_btns.addStretch(1)
        data_lay.addLayout(file_btns)

        # Session match table
        self._session_table = QTableWidget(0, 3)
        self._session_table.setHorizontalHeaderLabels(["Video", "Pose File", ""])
        hdr = self._session_table.horizontalHeader()
        hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, hdr.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
        self._session_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._session_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.setStyleSheet(
            "QTableWidget { background: #0A1929; color: #B0BEC5; font-size: 11px;"
            " gridline-color: #1E3A5F; alternate-background-color: #0D2137;"
            " border: 1px solid #1E3A5F; }"
            "QHeaderView::section { background: #0F2744; color: #78909C; font-size: 11px;"
            " font-weight: 600; padding: 3px 6px; border-bottom: 1px solid #1E3A5F;"
            " border-right: 1px solid #1E3A5F; }"
        )
        self._session_table.setMinimumHeight(80)
        self._session_table.setMaximumHeight(140)
        data_lay.addWidget(self._session_table)

        self._file_status = QLabel("")
        self._file_status.setStyleSheet("font-size: 11px; color: #78909C;")
        data_lay.addWidget(self._file_status)

        # Output folder
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        out_lbl = QLabel("Output folder:")
        out_lbl.setStyleSheet("font-size: 11px; color: #78909C;")
        out_lbl.setFixedWidth(90)
        self._out_btn = QPushButton("Browse…")
        self._out_btn.setStyleSheet(_BTN)
        self._out_btn.clicked.connect(self._browse_output)
        self._out_lbl = QLabel("Not selected  (results subfolder will be created here).")
        self._out_lbl.setWordWrap(True)
        self._out_lbl.setStyleSheet("font-size: 11px; color: #78909C;")
        out_row.addWidget(out_lbl)
        out_row.addWidget(self._out_btn)
        out_row.addWidget(self._out_lbl, 1)
        data_lay.addLayout(out_row)

        # ── Pixel/mm calibration ──────────────────────────────────────
        # px/mm is camera-specific and cannot be inherited from the source
        # project — it must be measured on the new videos.  Spatial context
        # features (distance-to-target) and physical-unit analytics depend
        # on it, so we surface it as an explicit step.
        pxmm_row = QHBoxLayout()
        pxmm_row.setSpacing(6)
        pxmm_lbl = QLabel("Pixels / mm:")
        pxmm_lbl.setStyleSheet("font-size: 11px; color: #78909C;")
        pxmm_lbl.setFixedWidth(90)
        self._pxmm_spin = QDoubleSpinBox()
        self._pxmm_spin.setDecimals(6)
        self._pxmm_spin.setRange(0.0, 1_000_000.0)
        self._pxmm_spin.setValue(0.0)
        self._pxmm_spin.setSpecialValueText("not set")
        self._pxmm_spin.setFixedWidth(110)
        self._pxmm_spin.setToolTip(
            "Pixels per millimetre for the current subject. 'not set' (0) "
            "leaves spatial features in pixel units."
        )
        self._pxmm_spin.valueChanged.connect(self._on_pxmm_spin_changed)
        self._pxmm_calib_btn = QPushButton("Calibrate (2-point)…")
        self._pxmm_calib_btn.setStyleSheet(_BTN)
        self._pxmm_calib_btn.setToolTip(
            "Measure pixels/mm by clicking two points a known distance apart."
        )
        self._pxmm_calib_btn.clicked.connect(self._calibrate_pxmm)
        self._pxmm_all_btn = QPushButton("Apply to All")
        self._pxmm_all_btn.setStyleSheet(_BTN)
        self._pxmm_all_btn.setToolTip("Apply the current px/mm value to every subject.")
        self._pxmm_all_btn.clicked.connect(self._apply_pxmm_to_all)
        self._pxmm_status = QLabel("")
        self._pxmm_status.setStyleSheet("font-size: 11px; color: #78909C;")
        pxmm_row.addWidget(pxmm_lbl)
        pxmm_row.addWidget(self._pxmm_spin)
        pxmm_row.addWidget(self._pxmm_calib_btn)
        pxmm_row.addWidget(self._pxmm_all_btn)
        pxmm_row.addWidget(self._pxmm_status, 1)
        data_lay.addLayout(pxmm_row)

        # ── Step 3: Keypoint Mapping ──────────────────────────────────
        kp_box = _CollapsibleSection("Step 3 — Keypoint Mapping")
        kp_lay = kp_box.content_layout
        kp_lay.setSpacing(6)
        kp_desc = QLabel(
            "The model was trained on specific pose keypoints. If your new DLC "
            "files name keypoints differently, map each model keypoint to the "
            "matching one in your data so features line up. Mismatched keypoints "
            "would otherwise be treated as missing and wreck predictions. "
            "Suggestions are auto-filled; review and correct as needed."
        )
        kp_desc.setWordWrap(True)
        kp_desc.setStyleSheet("font-size: 11px; color: #78909C;")
        kp_lay.addWidget(kp_desc)

        kp_btn_row = QHBoxLayout()
        self._kp_auto_btn = QPushButton("Auto-map")
        self._kp_auto_btn.setStyleSheet(_BTN)
        self._kp_auto_btn.setToolTip("Re-run automatic keypoint matching.")
        self._kp_auto_btn.clicked.connect(self._auto_map_keypoints)
        self._kp_status = QLabel("")
        self._kp_status.setStyleSheet("font-size: 11px; color: #78909C;")
        kp_btn_row.addWidget(self._kp_auto_btn)
        kp_btn_row.addWidget(self._kp_status, 1)
        kp_lay.addLayout(kp_btn_row)

        self._kp_table = QTableWidget(0, 2)
        self._kp_table.setHorizontalHeaderLabels(["Model Keypoint", "Your Data's Keypoint"])
        kp_hdr = self._kp_table.horizontalHeader()
        kp_hdr.setSectionResizeMode(0, kp_hdr.ResizeMode.Stretch)
        kp_hdr.setSectionResizeMode(1, kp_hdr.ResizeMode.Stretch)
        self._kp_table.verticalHeader().setVisible(False)
        self._kp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._kp_table.setStyleSheet(
            "QTableWidget { background: #0A1929; color: #B0BEC5; font-size: 11px;"
            " gridline-color: #1E3A5F; alternate-background-color: #0D2137;"
            " border: 1px solid #1E3A5F; }"
            "QHeaderView::section { background: #0F2744; color: #78909C; font-size: 11px;"
            " font-weight: 600; padding: 3px 6px; border-bottom: 1px solid #1E3A5F; }"
        )
        self._kp_table.setMinimumHeight(120)
        self._kp_table.setMaximumHeight(240)
        kp_lay.addWidget(self._kp_table)

        # ── Step 3: Define ROIs ───────────────────────────────────────
        roi_box = _CollapsibleSection("Step 4 — Define ROIs")
        roi_lay = roi_box.content_layout
        roi_lay.setSpacing(6)

        roi_desc = QLabel(
            "Define the environment ROIs for your new data. These are used by "
            "context features (target-relative distance, angle, etc.). "
            "Draw on the video frame below, or copy ROIs from the source project."
        )
        roi_desc.setWordWrap(True)
        roi_desc.setStyleSheet("font-size: 11px; color: #78909C;")
        roi_lay.addWidget(roi_desc)

        # ── ROI Scope selector ────────────────────────────────────────
        self._roi_scope = QComboBox()
        self._roi_scope.addItem("Project default (all subjects)", userData="project")
        self._roi_scope.addItem("Per-subject override", userData="subject")
        self._roi_scope.setToolTip(
            "Project default applies the same ROI to all subjects. "
            "Per-subject override lets you set ROIs individually."
        )
        self._roi_scope.currentIndexChanged.connect(self._on_roi_scope_changed)

        roi_scope_row = QHBoxLayout()
        roi_scope_row.setSpacing(6)
        roi_scope_row.addWidget(QLabel("Scope:"))
        roi_scope_row.addWidget(self._roi_scope)
        roi_scope_row.addStretch(1)
        roi_lay.addLayout(roi_scope_row)

        # ── Subject list with navigation ──────────────────────────────
        self._roi_subject_list = QListWidget()
        self._roi_subject_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._roi_subject_list.currentRowChanged.connect(self._on_roi_subject_list_changed)
        self._roi_subject_list.setMinimumHeight(60)
        self._roi_subject_list.setMaximumHeight(160)

        self._roi_subject_counter = QLabel("0 / 0 subjects")
        self._roi_subject_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_subject_counter.setStyleSheet("font-size: 11px; color: #B0BEC5;")

        self._roi_prev_btn = QPushButton("◀ Prev")
        self._roi_prev_btn.setToolTip("Save current subject ROI and go to previous (Left / Up)")
        self._roi_prev_btn.setStyleSheet(_BTN)
        self._roi_prev_btn.clicked.connect(self._roi_go_prev_subject)

        self._roi_next_btn = QPushButton("Next ▶")
        self._roi_next_btn.setToolTip("Save current subject ROI and go to next (Right / Down)")
        self._roi_next_btn.setStyleSheet(_BTN)
        self._roi_next_btn.clicked.connect(self._roi_go_next_subject)

        roi_nav_row = QHBoxLayout()
        roi_nav_row.addWidget(self._roi_prev_btn)
        roi_nav_row.addWidget(self._roi_subject_counter, 1)
        roi_nav_row.addWidget(self._roi_next_btn)

        self._roi_copy_all_btn = QPushButton("Copy Current to All Subjects")
        self._roi_copy_all_btn.setStyleSheet(_BTN)
        self._roi_copy_all_btn.setToolTip(
            "Copy the current Target Zone and Subject Crop values to every subject."
        )
        self._roi_copy_all_btn.clicked.connect(self._roi_copy_to_all_subjects)

        self._roi_subject_box = QGroupBox("Subjects")
        roi_subject_layout = QVBoxLayout(self._roi_subject_box)
        roi_subject_layout.addWidget(self._roi_subject_list)
        roi_subject_layout.addLayout(roi_nav_row)
        roi_subject_layout.addWidget(self._roi_copy_all_btn)
        self._roi_subject_box.setVisible(False)  # hidden until scope = "subject"
        roi_lay.addWidget(self._roi_subject_box)

        # Canvas (inside a scroll area so it can be zoomed beyond the viewport)
        self._roi_canvas = _ROICanvas()
        self._roi_canvas.roi_n_changed.connect(self._on_roi_n_drawn)
        self._roi_canvas.crop_changed.connect(self._on_roi_crop_drawn)
        self._roi_scroll = QScrollArea()
        self._roi_scroll.setWidgetResizable(True)
        self._roi_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._roi_scroll.setMinimumHeight(300)
        self._roi_scroll.setMaximumHeight(460)
        self._roi_scroll.setWidget(self._roi_canvas)
        # Re-fit the zoom whenever the viewport is resized.
        self._roi_scroll.viewport().installEventFilter(self)
        roi_lay.addWidget(self._roi_scroll)

        # Zoom control row
        roi_zoom_row = QHBoxLayout()
        roi_zoom_row.setSpacing(6)
        roi_zoom_lbl = QLabel("Zoom:")
        roi_zoom_lbl.setStyleSheet("font-size: 11px; color: #78909C;")
        self._roi_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._roi_zoom_slider.setMinimum(100)   # 1.0×  (fit to view)
        self._roi_zoom_slider.setMaximum(600)   # 6.0×
        self._roi_zoom_slider.setValue(100)
        self._roi_zoom_slider.setSingleStep(25)
        self._roi_zoom_slider.setPageStep(50)
        self._roi_zoom_slider.setToolTip(
            "Enlarge the frame to draw ROIs more precisely. Kept when you move "
            "to the next subject."
        )
        self._roi_zoom_slider.valueChanged.connect(self._on_roi_zoom_changed)
        self._roi_zoom_value = QLabel("Fit")
        self._roi_zoom_value.setFixedWidth(46)
        self._roi_zoom_value.setStyleSheet("font-size: 11px; color: #B0BEC5;")
        roi_zoom_row.addWidget(roi_zoom_lbl)
        roi_zoom_row.addWidget(self._roi_zoom_slider, 1)
        roi_zoom_row.addWidget(self._roi_zoom_value)
        roi_lay.addLayout(roi_zoom_row)

        # Frame controls row
        roi_ctrl_row = QHBoxLayout()
        roi_ctrl_row.setSpacing(6)
        self._roi_load_btn = QPushButton("Load Frame")
        self._roi_load_btn.setStyleSheet(_BTN)
        self._roi_load_btn.setToolTip(
            "Load a frame from the first selected video to draw ROIs on"
        )
        self._roi_load_btn.clicked.connect(self._load_roi_frame)

        self._roi_copy_src_btn = QPushButton("Copy from Source")
        self._roi_copy_src_btn.setStyleSheet(_BTN)
        self._roi_copy_src_btn.setToolTip(
            "Copy ROI settings from the source project"
        )
        self._roi_copy_src_btn.clicked.connect(self._copy_roi_from_source)

        self._roi_draw_mode = QComboBox()
        # Populated dynamically by _rebuild_draw_mode_du()
        self._roi_draw_mode.currentIndexChanged.connect(
            self._on_roi_draw_mode_changed
        )

        roi_ctrl_row.addWidget(QLabel("Draw mode:"))
        roi_ctrl_row.addWidget(self._roi_draw_mode)
        roi_ctrl_row.addWidget(self._roi_load_btn)
        roi_ctrl_row.addWidget(self._roi_copy_src_btn)
        roi_ctrl_row.addStretch(1)
        roi_lay.addLayout(roi_ctrl_row)

        # Frame slider
        roi_slider_row = QHBoxLayout()
        self._roi_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._roi_frame_slider.setMinimum(0)
        self._roi_frame_slider.setMaximum(0)
        self._roi_frame_slider.valueChanged.connect(self._on_roi_frame_slider)
        self._roi_frame_label = QLabel("0 / 0")
        self._roi_frame_label.setMinimumWidth(80)
        self._roi_frame_label.setStyleSheet("font-size: 11px; color: #B0BEC5;")
        roi_slider_row.addWidget(self._roi_frame_slider, 1)
        roi_slider_row.addWidget(self._roi_frame_label)
        roi_lay.addLayout(roi_slider_row)

        # ── ROI count selector ─────────────────────────────────────────
        roi_count_row = QHBoxLayout()
        roi_count_row.setSpacing(6)
        roi_count_lbl = QLabel("Target Zone ROIs:")
        roi_count_lbl.setStyleSheet("font-weight: 600;")
        self._roi_count_spin_du = QSpinBox()
        self._roi_count_spin_du.setRange(1, MAX_ROIS)
        self._roi_count_spin_du.setValue(1)
        self._roi_count_spin_du.setFixedWidth(60)
        self._roi_count_spin_du.valueChanged.connect(self._on_roi_count_changed_du)
        roi_count_row.addWidget(roi_count_lbl)
        roi_count_row.addWidget(self._roi_count_spin_du)
        roi_count_row.addStretch(1)
        roi_lay.addLayout(roi_count_row)

        # Dynamic ROI spinbox container
        self._roi_spinbox_groups_du: list[tuple[QSpinBox, QSpinBox, QSpinBox, QSpinBox]] = []
        self._roi_boxes_du: list[QGroupBox] = []
        self._roi_spins_container_du = QWidget()
        self._roi_spins_layout_du = QVBoxLayout(self._roi_spins_container_du)
        self._roi_spins_layout_du.setContentsMargins(0, 0, 0, 0)
        self._roi_spins_layout_du.setSpacing(4)
        roi_lay.addWidget(self._roi_spins_container_du)

        # Spinboxes — subject crop / motion side by side
        roi_spins_row = QHBoxLayout()
        roi_spins_row.setSpacing(12)

        sc_group = QGroupBox("Subject Crop")
        sc_form = QFormLayout(sc_group)
        sc_form.setContentsMargins(6, 6, 6, 6)
        self._roi_sc_x = QSpinBox()
        self._roi_sc_x.setRange(0, 10000)
        self._roi_sc_y = QSpinBox()
        self._roi_sc_y.setRange(0, 10000)
        self._roi_sc_w = QSpinBox()
        self._roi_sc_w.setRange(0, 10000)
        self._roi_sc_h = QSpinBox()
        self._roi_sc_h.setRange(0, 10000)
        for sp in (self._roi_sc_x, self._roi_sc_y, self._roi_sc_w, self._roi_sc_h):
            sp.valueChanged.connect(self._on_roi_crop_spins_changed)
        sc_form.addRow("x:", self._roi_sc_x)
        sc_form.addRow("y:", self._roi_sc_y)
        sc_form.addRow("w:", self._roi_sc_w)
        sc_form.addRow("h:", self._roi_sc_h)

        mr_group = QGroupBox("Local Motion")
        mr_form = QFormLayout(mr_group)
        mr_form.setContentsMargins(6, 6, 6, 6)
        self._roi_motion_radius = QSpinBox()
        self._roi_motion_radius.setRange(8, 2048)
        self._roi_motion_radius.setSingleStep(4)
        self._roi_motion_radius.setValue(36)
        self._roi_motion_radius.setToolTip(
            "Pixel radius around body parts for local motion features"
        )
        mr_form.addRow("Radius (px):", self._roi_motion_radius)

        roi_spins_row.addWidget(sc_group)
        roi_spins_row.addWidget(mr_group)
        roi_lay.addLayout(roi_spins_row)

        # ── Step 4: Run ───────────────────────────────────────────────
        run_box = _CollapsibleSection("Step 5 — Run Pipeline")
        run_lay = run_box.content_layout
        run_lay.setSpacing(6)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Direct Use Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setMinimumHeight(34)
        self._run_btn.setStyleSheet(_BTN_PRIMARY)
        self._run_btn.clicked.connect(self._start_pipeline)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet(_BTN)
        self._cancel_btn.clicked.connect(self._cancel_pipeline)
        btn_row.addWidget(self._run_btn, 1)
        btn_row.addWidget(self._cancel_btn)
        run_lay.addLayout(btn_row)

        # Overall progress bar + timing on one row
        prog_row = QHBoxLayout()
        prog_row.setSpacing(8)
        self._overall_progress = QProgressBar()
        self._overall_progress.setRange(0, 100)
        self._overall_progress.setValue(0)
        self._overall_progress.setMinimumHeight(8)
        self._overall_progress.setTextVisible(False)
        self._overall_progress.setStyleSheet(
            "QProgressBar { background: #1A2027; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #1565C0, stop:0.5 #42A5F5, stop:1 #66BB6A"
            "); border-radius: 4px; }"
        )
        self._time_label = QLabel("")
        self._time_label.setFixedWidth(110)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._time_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #90CAF9;")
        self._eta_label = QLabel("")
        self._eta_label.setFixedWidth(140)
        self._eta_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._eta_label.setStyleSheet("font-size: 11px; color: #546E7A;")
        prog_row.addWidget(self._overall_progress, 1)
        prog_row.addWidget(self._time_label)
        prog_row.addWidget(self._eta_label)
        run_lay.addLayout(prog_row)

        # Step cards in a 2-column grid
        cards_widget = QWidget()
        cards_grid = QGridLayout(cards_widget)
        cards_grid.setSpacing(5)
        cards_grid.setContentsMargins(0, 0, 0, 0)
        cards_grid.setColumnStretch(0, 1)
        cards_grid.setColumnStretch(1, 1)
        self._step_cards: dict[str, _StepCard] = {}
        for i, step_id in enumerate(STEP_IDS):
            card = _StepCard(step_id)
            self._step_cards[step_id] = card
            cards_grid.addWidget(card, i // 2, i % 2)
        run_lay.addWidget(cards_widget)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")
        run_lay.addWidget(self._status)

        # ── Keyboard shortcuts for subject navigation ─────────────────
        for key in ("Alt+Down", "Down", "Right"):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(self._roi_go_next_subject)
        for key in ("Alt+Up", "Up", "Left"):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(self._roi_go_prev_subject)

        # ── Outer layout ──────────────────────────────────────────────
        content = QWidget()
        clayout = QVBoxLayout(content)
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(10)
        clayout.addWidget(header)
        clayout.addWidget(desc)
        clayout.addWidget(source_box)
        clayout.addWidget(data_box)
        clayout.addWidget(kp_box)
        clayout.addWidget(roi_box)
        clayout.addWidget(run_box)
        clayout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setWidget(content)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)
        root.addWidget(scroll)

        # Initialise dynamic ROI spinboxes and draw-mode combo
        self._rebuild_roi_spinboxes_du(1)
        self._rebuild_draw_mode_du()

        # Declutter: collapse the two tallest steps by default so the workflow
        # reads as a compact checklist; the user expands them when needed.
        kp_box.set_expanded(False)
        roi_box.set_expanded(False)

    # ── Public API ─────────────────────────────────────────────────────

    def set_project(self, project_root: Path) -> None:
        """Called when the currently open project changes.
        For Direct Use the target is always chosen explicitly — this is a no-op."""
        pass

    def set_source_from_current(self, project_root: Path) -> None:
        """Pre-populate the source project from the currently open project."""
        self._source_root = project_root
        self._load_snapshot(project_root)

    # ── Step 1 — source project ───────────────────────────────────────

    def _select_source_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Source ABEL Project Folder",
        )
        if path:
            self._source_root = Path(path)
            self._load_snapshot(Path(path))

    def _load_snapshot(self, project_root: Path) -> None:
        snapshot = self._snap_svc.load(project_root)
        if snapshot is None:
            self._source_info.setText(
                f"{project_root.name} — No workflow snapshot found. "
                "Create one via Home → Create Snapshot Workflow."
            )
            self._snap_detail.hide()
            self._snapshot = None
            self._update_run_button()
            return

        valid, reason = self._snap_svc.is_valid(project_root, snapshot)
        self._snapshot = snapshot
        self._source_info.setText(
            f"{project_root.name}"
            + (f" — {reason}" if not valid else " — Snapshot loaded.")
        )

        # Build snapshot summary line
        sbm      = snapshot.selected_behavior_models or {}
        excluded = set(snapshot.excluded_behavior_ids or [])
        btab: dict[str, str] = {}
        for b in (snapshot.behavior_definitions or []):
            bid = b.get("behavior_id", b.get("name", ""))
            btab[bid] = b.get("name", b.get("short_name", bid))

        active_names = [btab.get(bid, bid) for bid in sbm if bid not in excluded]
        win = snapshot.segment_window_frames
        stride = snapshot.segment_stride_frames
        fps = snapshot.fps or "auto"

        # Extract temporal refinement inference stride if available
        tr_settings = snapshot.temporal_refinement_settings or {}
        tr_global = tr_settings.get("__all__") or {}
        tr_step = tr_global.get("inference_step_seconds")

        settings_parts: list[str] = [f"Window {win} fr · Stride {stride} fr · FPS {fps}"]
        if tr_step is not None:
            settings_parts.append(f"TR Stride {tr_step}s")
        settings_parts.append(
            "Video features: ON"
            if getattr(snapshot, "use_video_features", False)
            else "Video features: OFF"
        )
        parts: list[str] = [
            " · ".join(settings_parts),
            ("Behaviors: " + ", ".join(active_names)) if active_names else "",
        ]
        if snapshot.step_timings:
            total = sum(snapshot.step_timings.values())
            parts.append(f"Est. run time: ~{_fmt(total)}")
        self._snap_detail.setText("  ·  ".join(p for p in parts if p))
        self._snap_detail.show()

        # Seed step cards with reference timings
        for step_id, card in self._step_cards.items():
            ref = (snapshot.step_timings or {}).get(step_id, 0.0)
            card.set_pending(estimate_s=ref)

        # ROIs are NOT pre-populated from the source: they must be drawn fresh
        # on the new footage (Step 4).  Users can still pull the source values in
        # explicitly via "Copy from Source" when the camera setup is identical.

        self._refresh_keypoint_mapping()
        self._update_run_button()

    # ── Step 2 — input data ────────────────────────────────────────────

    def _add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", "", self._VIDEO_FILTER,
        )
        if not paths:
            return
        for p in paths:
            pp = Path(p)
            if pp not in self._video_paths:
                self._video_paths.append(pp)
        self._rebuild_table()

    def _add_poses(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DLC Pose File(s)", "", self._POSE_FILTER,
        )
        if not paths:
            return
        for p in paths:
            pp = Path(p)
            if pp not in self._pose_paths:
                self._pose_paths.append(pp)
        self._rebuild_table()

    def _clear_input_files(self) -> None:
        self._video_paths.clear()
        self._pose_paths.clear()
        self._session_table.setRowCount(0)
        self._file_status.setText("")
        self._close_roi_video()
        self._roi_frame_slider.setMaximum(0)
        self._roi_frame_label.setText("0 / 0")
        self._subject_rois.clear()
        self._roi_subject_list.clear()
        self._roi_subject_counter.setText("0 / 0 subjects")
        self._previous_roi_subject_id = ""
        self._roi_subject_dirty = False
        self._subject_pxmm.clear()
        self._pxmm_spin.blockSignals(True)
        self._pxmm_spin.setValue(0.0)
        self._pxmm_spin.blockSignals(False)
        self._update_pxmm_status()
        self._data_keypoints = []
        self._refresh_keypoint_mapping()
        self._update_run_button()

    def _rebuild_table(self) -> None:
        if not self._video_paths and not self._pose_paths:
            self._session_table.setRowCount(0)
            self._file_status.setText("")
            self._update_run_button()
            return

        tmp = self._import_svc.build_manifest(self._video_paths, self._pose_paths)
        vid_by_id  = {v.asset_id: v for v in tmp.videos}
        pose_by_id = {p.asset_id: p for p in tmp.poses}
        matched_vid_ids  = {s.video_asset_id for s in tmp.linked_sessions}
        matched_pose_ids = {s.pose_asset_id  for s in tmp.linked_sessions}

        rows: list[tuple[str, str, str]] = []
        for s in tmp.linked_sessions:
            vn = Path(vid_by_id[s.video_asset_id].source_path).name
            pn = Path(pose_by_id[s.pose_asset_id].source_path).name
            rows.append((vn, pn, "✓"))
        for v in tmp.videos:
            if v.asset_id not in matched_vid_ids:
                rows.append((Path(v.source_path).name, "—", "⚠ no pose"))
        for p in tmp.poses:
            if p.asset_id not in matched_pose_ids:
                rows.append(("—", Path(p.source_path).name, "⚠ no video"))

        self._session_table.setRowCount(len(rows))
        for r, (vname, pname, status) in enumerate(rows):
            for col, text in enumerate((vname, pname, status)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 2:
                    item.setForeground(
                        Qt.GlobalColor.darkGreen if "✓" in status else Qt.GlobalColor.yellow
                    )
                self._session_table.setItem(r, col, item)

        n_matched = len(tmp.linked_sessions)
        self._file_status.setText(
            f"{len(self._video_paths)} video(s), {len(self._pose_paths)} pose file(s)"
            f" — {n_matched} matched session(s)"
        )
        self._refresh_roi_subject_list()
        self._update_pxmm_status()
        self._detect_data_keypoints()
        self._refresh_keypoint_mapping()
        self._update_run_button()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Results",
        )
        if not path:
            return
        self._output_root = Path(path)
        self._out_lbl.setText(str(self._output_root))
        self._update_run_button()

    def _matched_session_count(self) -> int:
        if not self._video_paths and not self._pose_paths:
            return 0
        return len(
            self._import_svc.build_manifest(self._video_paths, self._pose_paths).linked_sessions
        )

    def _update_run_button(self) -> None:
        ready = (
            self._snapshot is not None
            and self._source_root is not None
            and self._output_root is not None
            and self._matched_session_count() > 0
            and self._worker is None
        )
        self._run_btn.setEnabled(ready)

    # ── Step 3 — ROI scope & subject management ──────────────────────

    def _matched_subjects(self) -> list[tuple[str, Path]]:
        """Return sorted (subject_id, video_path) pairs from matched sessions."""
        if not self._video_paths or not self._pose_paths:
            return []
        manifest = self._import_svc.build_manifest(self._video_paths, self._pose_paths)
        vid_by_id = {v.asset_id: v for v in manifest.videos}
        seen: dict[str, Path] = {}
        for s in manifest.linked_sessions:
            v = vid_by_id.get(s.video_asset_id)
            # Use subject_id if available, otherwise fall back to video filename stem
            sid = str(s.subject_id or "").strip()
            if not sid and v:
                sid = Path(v.source_path).stem
            if not sid or sid in seen:
                continue
            seen[sid] = Path(v.source_path) if v else Path()
        return sorted(seen.items(), key=lambda x: x[0])

    def _on_roi_scope_changed(self) -> None:
        is_subject = self._roi_scope.currentData() == "subject"
        self._roi_subject_box.setVisible(is_subject)
        if is_subject:
            self._refresh_roi_subject_list()
            # Load the first subject's values
            if self._roi_subject_list.count() > 0:
                self._roi_subject_list.setCurrentRow(0)

    def _refresh_roi_subject_list(self) -> None:
        """Rebuild the subject list from matched video/pose pairs."""
        subjects = self._matched_subjects()

        current_row = self._roi_subject_list.currentRow()
        current_sid = None
        if current_row >= 0:
            item = self._roi_subject_list.item(current_row)
            if item:
                current_sid = item.data(Qt.ItemDataRole.UserRole)

        self._roi_subject_list.blockSignals(True)
        self._roi_subject_list.clear()

        for sid, _vpath in subjects:
            sroi = self._subject_rois.get(sid, {})
            zones = sroi.get("target_zones", [sroi.get("target_zone", {})])
            first = zones[0] if zones else {}
            has_roi = (first.get("w", 0) or 0) > 0 and (first.get("h", 0) or 0) > 0

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, sid)
            if has_roi:
                item.setText(f"  ✓  {sid}")
                item.setForeground(QColor("#66BB6A"))
            else:
                item.setText(f"  ○  {sid}")
                item.setForeground(QColor("#78909C"))
            font = item.font()
            font.setPointSize(10)
            item.setFont(font)
            self._roi_subject_list.addItem(item)

        self._roi_subject_list.blockSignals(False)

        # Restore selection
        new_row = 0
        if current_sid:
            for i in range(self._roi_subject_list.count()):
                it = self._roi_subject_list.item(i)
                if it and it.data(Qt.ItemDataRole.UserRole) == current_sid:
                    new_row = i
                    break
        if self._roi_subject_list.count() > 0:
            self._roi_subject_list.setCurrentRow(new_row)
            self._previous_roi_subject_id = self._current_roi_subject_id()

        self._update_roi_subject_counter()

    def _update_roi_subject_counter(self) -> None:
        total = self._roi_subject_list.count()
        current = self._roi_subject_list.currentRow() + 1 if total > 0 else 0
        done = 0
        for sid, sroi in self._subject_rois.items():
            zones = sroi.get("target_zones", [sroi.get("target_zone", {})])
            first = zones[0] if zones else {}
            if (first.get("w", 0) or 0) > 0 and (first.get("h", 0) or 0) > 0:
                done += 1
        self._roi_subject_counter.setText(f"{current} / {total}  ({done} configured)")

    def _current_roi_subject_id(self) -> str:
        item = self._roi_subject_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _on_roi_subject_list_changed(self, row: int) -> None:
        """Auto-save previous subject, load new subject's ROI, load video."""
        if self._roi_subject_dirty and self._previous_roi_subject_id:
            self._save_roi_subject_quiet(self._previous_roi_subject_id)

        self._roi_subject_dirty = False
        sid = self._current_roi_subject_id()
        self._previous_roi_subject_id = sid
        self._load_roi_subject_values(sid)
        self._load_pxmm_for_subject(sid)
        self._update_roi_subject_counter()

        # Auto-load this subject's video
        self._auto_load_roi_subject_video(sid)

    def _load_roi_subject_values(self, subject_id: str) -> None:
        """Populate spinboxes from the in-memory per-subject ROI dict."""
        sroi = self._subject_rois.get(subject_id, {})
        zones = sroi.get("target_zones", [sroi.get("target_zone", {})])
        if not isinstance(zones, list):
            zones = [zones]
        sc = sroi.get("subject_crop", {})

        # Block crop spinboxes
        for sp in (self._roi_sc_x, self._roi_sc_y, self._roi_sc_w, self._roi_sc_h):
            sp.blockSignals(True)

        # Block all roi spinboxes
        for grp in self._roi_spinbox_groups_du:
            for sp in grp:
                sp.blockSignals(True)

        # Load target zones into spinbox groups
        for i, grp in enumerate(self._roi_spinbox_groups_du):
            roi = zones[i] if i < len(zones) else {"x": 0, "y": 0, "w": 0, "h": 0}
            x, y, w, h = grp
            x.setValue(roi.get("x", 0))
            y.setValue(roi.get("y", 0))
            w.setValue(roi.get("w", 0))
            h.setValue(roi.get("h", 0))

        # Load crop spinboxes
        self._roi_sc_x.setValue(sc.get("x", 0))
        self._roi_sc_y.setValue(sc.get("y", 0))
        self._roi_sc_w.setValue(sc.get("w", 0))
        self._roi_sc_h.setValue(sc.get("h", 0))

        for grp in self._roi_spinbox_groups_du:
            for sp in grp:
                sp.blockSignals(False)
        for sp in (self._roi_sc_x, self._roi_sc_y, self._roi_sc_w, self._roi_sc_h):
            sp.blockSignals(False)

        rois_for_canvas = [
            {"x": grp[0].value(), "y": grp[1].value(), "w": grp[2].value(), "h": grp[3].value()}
            for grp in self._roi_spinbox_groups_du
        ]
        self._roi_canvas.set_rois(rois_for_canvas)
        self._roi_canvas.set_crop(sc if sc else {"x": 0, "y": 0, "w": 0, "h": 0})

    def _save_roi_subject_quiet(self, subject_id: str) -> None:
        """Save the current spinbox values into the in-memory per-subject dict."""
        if not subject_id:
            return
        target_zones = [
            {"x": grp[0].value(), "y": grp[1].value(), "w": grp[2].value(), "h": grp[3].value()}
            for grp in self._roi_spinbox_groups_du
        ]
        self._subject_rois[subject_id] = {
            "target_zones": target_zones,
            "subject_crop": {
                "x": self._roi_sc_x.value(),
                "y": self._roi_sc_y.value(),
                "w": self._roi_sc_w.value(),
                "h": self._roi_sc_h.value(),
            },
        }
        self._roi_subject_dirty = False
        logger.info("Saved ROI for direct-use subject '%s'", subject_id)
        self._refresh_roi_subject_list_item(subject_id)

    def _refresh_roi_subject_list_item(self, subject_id: str) -> None:
        """Update a single item's visual status without rebuilding the whole list."""
        sroi = self._subject_rois.get(subject_id, {})
        zones = sroi.get("target_zones", [sroi.get("target_zone", {})])
        first = zones[0] if zones else {}
        has_roi = (first.get("w", 0) or 0) > 0 and (first.get("h", 0) or 0) > 0

        for i in range(self._roi_subject_list.count()):
            item = self._roi_subject_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == subject_id:
                if has_roi:
                    item.setText(f"  ✓  {subject_id}")
                    item.setForeground(QColor("#66BB6A"))
                else:
                    item.setText(f"  ○  {subject_id}")
                    item.setForeground(QColor("#78909C"))
                break
        self._update_roi_subject_counter()

    def _auto_load_roi_subject_video(self, subject_id: str) -> None:
        """Load the video for the given subject into the ROI canvas."""
        if not subject_id:
            return
        subjects = dict(self._matched_subjects())
        vpath = subjects.get(subject_id)
        if not vpath or not vpath.exists():
            return

        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return

        self._close_roi_video()
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return
        self._roi_video_cap = cap
        self._roi_n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._roi_frame_slider.setMaximum(self._roi_n_frames - 1)
        current_frame = self._roi_frame_slider.value()
        if current_frame >= self._roi_n_frames:
            self._roi_frame_slider.setValue(0)
            current_frame = 0
        self._roi_frame_label.setText(
            f"{current_frame} / {max(0, self._roi_n_frames - 1)}"
        )
        self._show_roi_frame(current_frame)

    def _roi_go_next_subject(self) -> None:
        """Save current subject and advance to next."""
        if self._roi_scope.currentData() != "subject":
            return
        count = self._roi_subject_list.count()
        if count == 0:
            return
        row = self._roi_subject_list.currentRow()
        if row < count - 1:
            self._roi_subject_list.setCurrentRow(row + 1)

    def _roi_go_prev_subject(self) -> None:
        """Save current subject and go to previous."""
        if self._roi_scope.currentData() != "subject":
            return
        count = self._roi_subject_list.count()
        if count == 0:
            return
        row = self._roi_subject_list.currentRow()
        if row > 0:
            self._roi_subject_list.setCurrentRow(row - 1)

    def _roi_copy_to_all_subjects(self) -> None:
        """Copy the current spinbox ROI values to every subject."""
        subjects = self._matched_subjects()
        if not subjects:
            return
        target_zones = [
            {"x": grp[0].value(), "y": grp[1].value(), "w": grp[2].value(), "h": grp[3].value()}
            for grp in self._roi_spinbox_groups_du
        ]
        sc = {
            "x": self._roi_sc_x.value(),
            "y": self._roi_sc_y.value(),
            "w": self._roi_sc_w.value(),
            "h": self._roi_sc_h.value(),
        }
        for sid, _ in subjects:
            self._subject_rois[sid] = {
                "target_zones": [dict(z) for z in target_zones],
                "subject_crop": dict(sc),
            }
        self._refresh_roi_subject_list()
        logger.info("Copied current ROI to all %d subjects", len(subjects))

    # ── Step 3 — ROI definition ──────────────────────────────────────

    def _rebuild_roi_spinboxes_du(self, n: int) -> None:
        """Create exactly *n* ROI spinbox groups in the direct-use container."""
        n = max(1, min(n, MAX_ROIS))

        while self._roi_spins_layout_du.count():
            item = self._roi_spins_layout_du.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._roi_spinbox_groups_du = []
        self._roi_boxes_du = []

        colors = _roi_qcolors()
        for i in range(n):
            hex_c = ROI_COLORS[i % len(ROI_COLORS)]
            label = f"ROI {i + 1}" + (" (Target Zone)" if i == 0 else "")

            sx = QSpinBox(); sx.setRange(0, 10000)
            sy = QSpinBox(); sy.setRange(0, 10000)
            sw = QSpinBox(); sw.setRange(0, 10000)
            sh = QSpinBox(); sh.setRange(0, 10000)

            idx = i
            for sp in (sx, sy, sw, sh):
                sp.valueChanged.connect(lambda _val, _i=idx: self._on_roi_spinbox_changed_du(_i))

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

            self._roi_spinbox_groups_du.append((sx, sy, sw, sh))
            self._roi_boxes_du.append(box)
            self._roi_spins_layout_du.addWidget(box)

        self._roi_canvas.set_n_rois(n)

    def _rebuild_draw_mode_du(self) -> None:
        """Repopulate the draw-mode combo for the current ROI count."""
        current_data = self._roi_draw_mode.currentData()
        self._roi_draw_mode.blockSignals(True)
        self._roi_draw_mode.clear()
        n = len(self._roi_spinbox_groups_du)
        for i in range(n):
            label = f"Draw ROI {i + 1}" + (" (Target Zone)" if i == 0 else "")
            self._roi_draw_mode.addItem(label, userData=f"roi_{i}")
        self._roi_draw_mode.addItem("Draw Subject Crop", userData="subject_crop")
        idx = self._roi_draw_mode.findData(current_data)
        self._roi_draw_mode.setCurrentIndex(max(0, idx))
        self._roi_draw_mode.blockSignals(False)
        self._on_roi_draw_mode_changed()

    def _on_roi_count_changed_du(self, n: int) -> None:
        """User changed the ROI count spinbox (direct use tab)."""
        self._rebuild_roi_spinboxes_du(n)
        self._rebuild_draw_mode_du()

    def _load_roi_frame(self) -> None:
        """Load a frame from the current subject's video (or the first video)."""
        if not self._video_paths:
            QMessageBox.information(
                self, "Define ROIs",
                "Add video files in Step 2 first, then load a frame here.",
            )
            return
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            QMessageBox.warning(
                self, "Define ROIs",
                "OpenCV (cv2) is required for frame preview.",
            )
            return

        # In subject mode, prefer the currently selected subject's video
        video_path = self._video_paths[0]
        if self._roi_scope.currentData() == "subject":
            sid = self._current_roi_subject_id()
            if sid:
                subjects = dict(self._matched_subjects())
                vp = subjects.get(sid)
                if vp and vp.exists():
                    video_path = vp

        if not video_path.exists():
            QMessageBox.warning(
                self, "Define ROIs",
                f"Video file not found:\n{video_path}",
            )
            return

        self._close_roi_video()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(
                self, "Define ROIs",
                f"Cannot open video:\n{video_path}",
            )
            return

        self._roi_video_cap = cap
        self._roi_n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._roi_frame_slider.setMaximum(self._roi_n_frames - 1)
        self._roi_frame_slider.setValue(0)
        self._show_roi_frame(0)

    def _close_roi_video(self) -> None:
        """Release the ROI video capture resource."""
        if self._roi_video_cap is not None:
            self._roi_video_cap.release()
            self._roi_video_cap = None
            self._roi_n_frames = 0

    def _on_roi_frame_slider(self, value: int) -> None:
        self._roi_frame_label.setText(
            f"{value} / {max(0, self._roi_n_frames - 1)}"
        )
        self._show_roi_frame(value)

    def _show_roi_frame(self, frame_idx: int) -> None:
        """Seek to *frame_idx* and display the frame in the ROI canvas."""
        if self._roi_video_cap is None:
            return
        import cv2  # noqa: PLC0415

        self._roi_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._roi_video_cap.read()
        if ret and frame is not None:
            self._roi_canvas.set_frame(frame)
            # Re-apply the current zoom so it persists across frame/subject loads.
            self._apply_roi_zoom()

    # ── ROI zoom ─────────────────────────────────────────────────────

    def _on_roi_zoom_changed(self, value: int) -> None:
        zoom = value / 100.0
        self._roi_zoom_value.setText("Fit" if value <= 100 else f"{zoom:.2g}×")
        self._apply_roi_zoom()

    def _apply_roi_zoom(self) -> None:
        """Push the current zoom factor + viewport size to the ROI canvas."""
        zoom = self._roi_zoom_slider.value() / 100.0
        self._roi_canvas.set_zoom(zoom, self._roi_scroll.viewport().size())

    def eventFilter(self, obj, event) -> bool:
        # Keep the frame fitted/zoomed correctly when the viewport resizes.
        if (
            obj is self._roi_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._apply_roi_zoom()
        return super().eventFilter(obj, event)

    def _on_roi_draw_mode_changed(self) -> None:
        mode = self._roi_draw_mode.currentData() or "roi_0"
        self._roi_canvas.set_draw_mode(mode)

    def _on_roi_n_drawn(self, index: int, roi: dict) -> None:
        """Canvas emitted a new target zone rectangle — update matching spinbox group."""
        if index >= len(self._roi_spinbox_groups_du):
            return
        x, y, w, h = self._roi_spinbox_groups_du[index]
        for sp in (x, y, w, h):
            sp.blockSignals(True)
        x.setValue(roi.get("x", 0))
        y.setValue(roi.get("y", 0))
        w.setValue(roi.get("w", 0))
        h.setValue(roi.get("h", 0))
        for sp in (x, y, w, h):
            sp.blockSignals(False)
        if self._roi_scope.currentData() == "subject":
            self._roi_subject_dirty = True

    def _on_roi_crop_drawn(self, roi: dict) -> None:
        """Canvas emitted a new subject crop rectangle — update spinboxes."""
        for sp in (self._roi_sc_x, self._roi_sc_y, self._roi_sc_w, self._roi_sc_h):
            sp.blockSignals(True)
        self._roi_sc_x.setValue(roi.get("x", 0))
        self._roi_sc_y.setValue(roi.get("y", 0))
        self._roi_sc_w.setValue(roi.get("w", 0))
        self._roi_sc_h.setValue(roi.get("h", 0))
        for sp in (self._roi_sc_x, self._roi_sc_y, self._roi_sc_w, self._roi_sc_h):
            sp.blockSignals(False)
        if self._roi_scope.currentData() == "subject":
            self._roi_subject_dirty = True

    def _on_roi_spinbox_changed_du(self, roi_index: int) -> None:
        """ROI spinbox edited — update the matching canvas overlay."""
        if roi_index < len(self._roi_spinbox_groups_du):
            x, y, w, h = self._roi_spinbox_groups_du[roi_index]
            roi = {"x": x.value(), "y": y.value(), "w": w.value(), "h": h.value()}
            self._roi_canvas.set_roi_at(roi_index, roi)
        if self._roi_scope.currentData() == "subject":
            self._roi_subject_dirty = True

    def _on_roi_crop_spins_changed(self) -> None:
        """Subject crop spinbox edited — update the canvas crop overlay."""
        sc = {
            "x": self._roi_sc_x.value(),
            "y": self._roi_sc_y.value(),
            "w": self._roi_sc_w.value(),
            "h": self._roi_sc_h.value(),
        }
        self._roi_canvas.set_crop(sc)
        if self._roi_scope.currentData() == "subject":
            self._roi_subject_dirty = True

    def _copy_roi_from_source(self) -> None:
        """Copy ROI settings from the source project."""
        if not self._source_root:
            QMessageBox.information(
                self, "Define ROIs",
                "Select a source project in Step 1 first.",
            )
            return
        cfg = self._roi_svc.load(self._source_root)
        self._apply_roi_config(cfg)

    def _apply_roi_config(self, cfg: dict) -> None:
        """Populate the ROI spinboxes/count from a loaded ROI config dict."""
        proj_rois = cfg.get("project_rois", {})
        # Support both new target_zones list and legacy target_zone dict
        zones = proj_rois.get("target_zones", [proj_rois.get("target_zone", {})])
        if not isinstance(zones, list):
            zones = [zones]
        roi_count = max(1, min(len(zones), MAX_ROIS))
        sc = proj_rois.get("subject_crop", {})
        motion = cfg.get("motion", {})

        # Update count spinbox and rebuild if needed
        self._roi_count_spin_du.setValue(roi_count)  # triggers _on_roi_count_changed_du if changed

        # Ensure spinboxes match the new count
        if len(self._roi_spinbox_groups_du) != roi_count:
            self._rebuild_roi_spinboxes_du(roi_count)
            self._rebuild_draw_mode_du()

        for i, grp in enumerate(self._roi_spinbox_groups_du):
            zone = zones[i] if i < len(zones) else {"x": 0, "y": 0, "w": 0, "h": 0}
            x, y, w, h = grp
            x.setValue(zone.get("x", 0))
            y.setValue(zone.get("y", 0))
            w.setValue(zone.get("w", 0))
            h.setValue(zone.get("h", 0))

        self._roi_sc_x.setValue(sc.get("x", 0))
        self._roi_sc_y.setValue(sc.get("y", 0))
        self._roi_sc_w.setValue(sc.get("w", 0))
        self._roi_sc_h.setValue(sc.get("h", 0))
        self._roi_motion_radius.setValue(motion.get("local_radius_px", 36))

    # ── Pixel/mm calibration ─────────────────────────────────────────

    _PXMM_DEFAULT_KEY = "__all__"

    def _current_pxmm_key(self) -> str:
        """Subject the px/mm spin currently edits, or the project default."""
        if self._roi_scope.currentData() == "subject":
            sid = self._current_roi_subject_id()
            if sid:
                return sid
        return self._PXMM_DEFAULT_KEY

    def _on_pxmm_spin_changed(self, value: float) -> None:
        key = self._current_pxmm_key()
        if value > 0:
            self._subject_pxmm[key] = float(value)
        else:
            self._subject_pxmm.pop(key, None)
        self._update_pxmm_status()

    def _load_pxmm_for_subject(self, subject_id: str) -> None:
        """Reflect the stored px/mm for *subject_id* in the spinbox."""
        val = self._subject_pxmm.get(
            subject_id, self._subject_pxmm.get(self._PXMM_DEFAULT_KEY, 0.0)
        )
        self._pxmm_spin.blockSignals(True)
        self._pxmm_spin.setValue(float(val or 0.0))
        self._pxmm_spin.blockSignals(False)

    def _apply_pxmm_to_all(self) -> None:
        val = float(self._pxmm_spin.value())
        if val <= 0:
            QMessageBox.information(
                self, "Pixels/mm",
                "Enter or calibrate a positive px/mm value first.",
            )
            return
        self._subject_pxmm[self._PXMM_DEFAULT_KEY] = val
        for sid, _ in self._matched_subjects():
            self._subject_pxmm[sid] = val
        self._update_pxmm_status()

    def _calibrate_pxmm(self) -> None:
        if not self._video_paths or not self._pose_paths:
            QMessageBox.information(
                self, "Pixels/mm",
                "Add video and pose files in Step 2 first.",
            )
            return
        manifest = self._import_svc.build_manifest(self._video_paths, self._pose_paths)
        if not manifest.linked_sessions:
            QMessageBox.information(
                self, "Pixels/mm", "No matched video/pose sessions to calibrate.",
            )
            return
        dlg = PixelScaleCalibrationDialog(
            import_service=self._import_svc,
            manifest=manifest,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        ppm = dlg.result_pixels_per_mm
        if ppm is None or ppm <= 0:
            QMessageBox.warning(
                self, "Pixels/mm", "Calibration did not produce a valid value.",
            )
            return
        # Map the calibrated session back to its subject so the value lands
        # on the right key (the calibrated subject in per-subject scope, or
        # the project default otherwise).
        sid = str(dlg.result_session_id or "").strip()
        subject = ""
        for s in manifest.linked_sessions:
            if str(s.session_id) == sid:
                subject = str(s.subject_id or "").strip()
                break
        if subject and self._roi_scope.currentData() == "subject":
            key = subject
        else:
            key = self._PXMM_DEFAULT_KEY
        self._subject_pxmm[key] = float(ppm)

        # Reflect in the spinbox only when it represents the same key, so we
        # don't overwrite a different subject's value via the spin handler.
        if key == self._current_pxmm_key():
            self._pxmm_spin.blockSignals(True)
            self._pxmm_spin.setValue(float(ppm))
            self._pxmm_spin.blockSignals(False)
        self._update_pxmm_status()

    def _update_pxmm_status(self) -> None:
        subjects = [sid for sid, _ in self._matched_subjects()]
        default = self._subject_pxmm.get(self._PXMM_DEFAULT_KEY)
        if subjects:
            n_done = sum(
                1 for sid in subjects
                if self._subject_pxmm.get(sid, default) and
                self._subject_pxmm.get(sid, default) > 0
            )
            self._pxmm_status.setText(f"{n_done}/{len(subjects)} subjects calibrated")
        elif default:
            self._pxmm_status.setText(f"default {default:.4g} px/mm")
        else:
            self._pxmm_status.setText("not calibrated (pixel units)")

    def _pxmm_for_subject(self, subject_id: str) -> float | None:
        val = self._subject_pxmm.get(
            subject_id, self._subject_pxmm.get(self._PXMM_DEFAULT_KEY)
        )
        return float(val) if val and val > 0 else None

    # ── Step 3 — keypoint mapping ────────────────────────────────────

    _KP_NONE = "(none)"

    def _model_keypoints(self) -> list[str]:
        return list(self._snapshot.pose_keypoints) if self._snapshot else []

    def _detect_data_keypoints(self) -> None:
        """Read body-part names from the first readable pose file."""
        self._data_keypoints = []
        for pp in self._pose_paths:
            try:
                pose = self._pose_svc.load(Path(pp))
                self._data_keypoints = list(pose.body_parts)
                break
            except Exception:
                continue

    def _refresh_keypoint_mapping(self) -> None:
        """Rebuild the mapping table from the model + detected data keypoints."""
        model_kps = self._model_keypoints()
        self._kp_combos = {}
        self._kp_table.setRowCount(len(model_kps))
        if not model_kps:
            self._kp_status.setText(
                "Snapshot has no keypoint info — re-create it to enable mapping."
            )
            return

        # Initial mapping: a saved map for this source wins; otherwise suggest.
        mapping: dict[str, str] = {}
        if self._source_root:
            mapping = keypoint_mapping.load_saved(self._source_root)
        if not mapping and self._data_keypoints:
            mapping = keypoint_mapping.suggest_mapping(model_kps, self._data_keypoints)

        options = [self._KP_NONE] + list(self._data_keypoints)
        for r, mk in enumerate(model_kps):
            name_item = QTableWidgetItem(mk)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._kp_table.setItem(r, 0, name_item)
            combo = QComboBox()
            combo.addItems(options)
            sel = mapping.get(mk, "")
            idx = combo.findText(sel) if sel else -1
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.currentIndexChanged.connect(self._update_kp_status)
            self._kp_combos[mk] = combo
            self._kp_table.setCellWidget(r, 1, combo)
        self._update_kp_status()

    def _auto_map_keypoints(self) -> None:
        if not self._data_keypoints:
            QMessageBox.information(
                self, "Keypoint Mapping", "Add DLC pose files in Step 2 first.")
            return
        mapping = keypoint_mapping.suggest_mapping(
            self._model_keypoints(), self._data_keypoints)
        for mk, combo in self._kp_combos.items():
            sel = mapping.get(mk, "")
            idx = combo.findText(sel) if sel else -1
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._update_kp_status()

    def _collect_keypoint_map(self) -> dict[str, str]:
        """Return {model_keypoint: data_keypoint} from the dropdowns."""
        out: dict[str, str] = {}
        for mk, combo in self._kp_combos.items():
            val = combo.currentText()
            if val and val != self._KP_NONE:
                out[mk] = val
        return out

    def _update_kp_status(self) -> None:
        model_kps = self._model_keypoints()
        if not model_kps:
            return
        mapping = self._collect_keypoint_map()
        data_vals = list(mapping.values())
        dupes = len(data_vals) != len(set(data_vals))
        msg = f"{len(mapping)}/{len(model_kps)} model keypoints mapped"
        if dupes:
            msg += "   ⚠ duplicate assignments"
            self._kp_status.setStyleSheet("font-size: 11px; color: #EF9A9A;")
        elif len(mapping) < len(model_kps):
            self._kp_status.setStyleSheet("font-size: 11px; color: #FFB74D;")
        else:
            self._kp_status.setStyleSheet("font-size: 11px; color: #66BB6A;")
        self._kp_status.setText(msg)

    @staticmethod
    def _any_zone_nonzero(zones: list) -> bool:
        """True when at least one target zone has a positive width and height."""
        return any(
            (z.get("w", 0) or 0) > 0 and (z.get("h", 0) or 0) > 0
            for z in (zones or [])
            if isinstance(z, dict)
        )

    def _user_provided_roi(self) -> bool:
        """True when the user has drawn at least one non-empty ROI in this tab."""
        cfg = self._build_roi_config()
        if self._any_zone_nonzero(cfg.get("project_rois", {}).get("target_zones", [])):
            return True
        for sroi in (cfg.get("subject_rois", {}) or {}).values():
            if self._any_zone_nonzero((sroi or {}).get("target_zones", [])):
                return True
        return False

    def _source_has_roi(self) -> bool:
        """True when the source project defines a real (non-zero) ROI."""
        if not self._source_root:
            return False
        try:
            cfg = self._roi_svc.load(self._source_root)
        except Exception:
            return False
        if self._any_zone_nonzero(cfg.get("project_rois", {}).get("target_zones", [])):
            return True
        for sroi in (cfg.get("subject_rois", {}) or {}).values():
            if self._any_zone_nonzero((sroi or {}).get("target_zones", [])):
                return True
        return False

    def _build_roi_config(self) -> dict:
        """Build an ROI config dict from the current spinbox/subject values."""
        # Flush any pending per-subject edits
        if self._roi_scope.currentData() == "subject":
            sid = self._current_roi_subject_id()
            if sid and self._roi_subject_dirty:
                self._save_roi_subject_quiet(sid)

        target_zones = [
            {"x": grp[0].value(), "y": grp[1].value(), "w": grp[2].value(), "h": grp[3].value()}
            for grp in self._roi_spinbox_groups_du
        ]
        return {
            "schema_version": "0.3.0",
            "roi_count": len(target_zones),
            "project_rois": {
                "target_zones": target_zones,
                "subject_crop": {
                    "x": self._roi_sc_x.value(),
                    "y": self._roi_sc_y.value(),
                    "w": self._roi_sc_w.value(),
                    "h": self._roi_sc_h.value(),
                },
            },
            "subject_rois": dict(self._subject_rois),
            "motion": {
                "local_radius_px": self._roi_motion_radius.value(),
            },
        }

    # ── Step 4 — run ──────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        if self._snapshot is None or self._source_root is None:
            return
        if self._output_root is None:
            QMessageBox.warning(self, "Direct Use", "Please select an output folder.")
            return
        n = self._matched_session_count()
        if n == 0:
            QMessageBox.warning(
                self, "Direct Use",
                "No matched video/pose pairs found.\n"
                "Add video files and their matching DLC pose files.",
            )
            return

        # Build target project root
        target_root = self._output_root / f"direct_use_{self._source_root.name}"
        target_root.mkdir(parents=True, exist_ok=True)

        # Save import manifest into target project
        manifest = self._import_svc.build_manifest(self._video_paths, self._pose_paths)

        # Ensure every linked session has a subject_id so per-subject ROIs
        # can be resolved downstream.  Fall back to the video filename stem
        # (same logic as _matched_subjects) when the regex didn't extract one.
        vid_by_id = {v.asset_id: v for v in manifest.videos}
        for s in manifest.linked_sessions:
            sid = str(s.subject_id or "").strip()
            if not sid:
                v = vid_by_id.get(s.video_asset_id)
                if v:
                    s.subject_id = Path(v.source_path).stem

        # Apply pixel/mm calibration into the manifest (per subject, falling
        # back to the project default) so context features and analytics use
        # physical units that match the source project.
        for s in manifest.linked_sessions:
            ppm = self._pxmm_for_subject(str(s.subject_id or "").strip())
            if ppm is not None:
                self._import_svc.update_session_pixels_per_mm(
                    manifest, str(s.session_id), ppm,
                )

        self._import_svc.save_manifest(target_root, manifest)

        # Save ROI configuration into target project so context features
        # and downstream services can resolve ROIs from the target config.
        self._roi_svc.save(target_root, self._build_roi_config())

        # Persist keypoint mapping: the {data:model} rename map into the target
        # for the pipeline to apply on pose load, and the {model:data} map back
        # to the source project so it auto-loads next time.
        kp_map = self._collect_keypoint_map()
        keypoint_mapping.write_target_aliases(
            target_root, keypoint_mapping.to_rename_map(kp_map)
        )
        if self._source_root:
            keypoint_mapping.save(self._source_root, kp_map)

        # Behavior list for confirmation
        sbm      = self._snapshot.selected_behavior_models or {}
        excluded = set(self._snapshot.excluded_behavior_ids or [])
        btab: dict[str, str] = {}
        for b in (self._snapshot.behavior_definitions or []):
            bid = b.get("behavior_id", b.get("name", ""))
            btab[bid] = b.get("name", b.get("short_name", bid))
        active = [btab.get(bid, bid) for bid in sbm if bid not in excluded]

        # Context features: report whether they will run, and warn when the
        # model needs them but no px/mm calibration was supplied.
        use_video = bool(getattr(self._snapshot, "use_video_features", False))
        n_calibrated = sum(
            1 for s in manifest.linked_sessions
            if self._pxmm_for_subject(str(s.subject_id or "").strip()) is not None
        )
        feature_line = (
            "Features:  pose + video/context (optical flow)"
            if use_video else "Features:  pose only"
        )
        warn_line = ""
        if use_video and n_calibrated < len(manifest.linked_sessions):
            missing = len(manifest.linked_sessions) - n_calibrated
            warn_line = (
                f"\n\n⚠  {missing} session(s) have no pixels/mm calibration. "
                "Spatial context features will be in pixel units, which may not "
                "match the source model and can degrade accuracy. Calibrate in "
                "Step 2 for best results."
            )

        # Keypoint mapping: warn if the model's keypoints are not all mapped —
        # unmapped keypoints make their features missing (zero-filled), which
        # severely degrades predictions.
        model_kps = self._model_keypoints()
        if model_kps:
            unmapped = [mk for mk in model_kps if mk not in kp_map]
            if unmapped:
                warn_line += (
                    f"\n\n⚠  {len(unmapped)} of {len(model_kps)} model keypoints are "
                    f"unmapped ({', '.join(unmapped[:5])}"
                    f"{'…' if len(unmapped) > 5 else ''}). Their features will be "
                    "missing and predictions will be poor. Map them in Step 3."
                )

        # ROI: the source model relies on ROIs, but they must be redrawn on the
        # new footage (pixel coordinates don't transfer between camera setups).
        # Warn if the source defines real ROIs and the user hasn't drawn any.
        if self._source_has_roi() and not self._user_provided_roi():
            warn_line += (
                "\n\n⚠  The source project defines ROIs, but you haven't drawn any "
                "for this new data. ROIs are not copied over (they're specific to "
                "each camera setup). Context features (target-zone distance/angle) "
                "will be zero/incorrect and ROI-dependent behaviors will score "
                "poorly. Draw them in Step 4 — or use 'Copy from Source' only if "
                "the camera framing is identical."
            )

        reply = QMessageBox.question(
            self, "Run Direct Use Pipeline",
            f"Analyze {n} session(s) using the model from '{self._source_root.name}'.\n\n"
            f"Behaviors: {', '.join(active) or 'all'}\n"
            f"{feature_line}\n"
            f"Output:    {target_root}"
            f"{warn_line}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Reset UI
        for step_id, card in self._step_cards.items():
            ref = (self._snapshot.step_timings or {}).get(step_id, 0.0)
            card.set_pending(estimate_s=ref)
        self._overall_progress.setValue(0)
        self._overall_progress.setStyleSheet(
            "QProgressBar { background: #1A2027; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #1565C0, stop:0.5 #42A5F5, stop:1 #66BB6A"
            "); border-radius: 4px; }"
        )
        self._time_label.setText("")
        self._eta_label.setText("")
        self._status.setText("Starting pipeline…")
        self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._worker = _DirectRunWorker(target_root, self._source_root, self._snapshot)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_finished)
        self._worker.start()

    def _cancel_pipeline(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling…")

    def _on_progress(self, state: DirectRunProgress) -> None:
        completed_frac = len(state.completed_steps) / state.total_steps
        step_frac      = state.step_progress / state.total_steps
        self._overall_progress.setValue(int((completed_frac + step_frac) * 100))

        self._time_label.setText(f"Elapsed: {_fmt(state.elapsed_seconds)}")
        if state.estimated_remaining_seconds > 0:
            self._eta_label.setText(f"~{_fmt(state.estimated_remaining_seconds)} left")
        else:
            self._eta_label.setText("")

        for step_id, card in self._step_cards.items():
            if step_id in state.completed_steps:
                card.set_done(state.step_timings.get(step_id, 0.0))
            elif step_id == state.current_step:
                card.set_active(state.step_progress, state.step_message)

        self._status.setText(state.step_message)

    def _on_finished(self, result: dict) -> None:
        # ``done`` is emitted as the last statement of run(); wait() for the OS
        # thread to fully terminate before dropping our reference, otherwise the
        # QThread can be garbage-collected while still running ("QThread:
        # Destroyed while thread is still running") and hard-abort the process.
        if self._worker is not None:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
        self._cancel_btn.setEnabled(False)
        self._update_run_button()

        status = result.get("status", "error")
        if status == "success":
            self._overall_progress.setValue(100)
            self._overall_progress.setStyleSheet(
                "QProgressBar { background: #1A2027; border-radius: 4px; border: none; }"
                "QProgressBar::chunk { background: #2E7D32; border-radius: 4px; }"
            )
            elapsed  = result.get("elapsed_seconds", 0)
            bouts    = result.get("bout_count", 0)
            sessions = result.get("session_count", 0)
            self._time_label.setText(f"Done in {_fmt(elapsed)}")
            self._eta_label.setText("")
            self._status.setStyleSheet("font-size: 11px; color: #66BB6A; padding-top: 2px;")
            self._status.setText(
                f"Complete — {sessions} session(s), {bouts} bout(s). "
                "Open the output folder or switch to the Analytics tab."
            )
            if self._output_root and self._source_root:
                self.pipeline_complete.emit(
                    self._output_root / f"direct_use_{self._source_root.name}"
                )
        elif status == "cancelled":
            self._status.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 2px;")
            self._status.setText("Pipeline cancelled.")
            self._time_label.setText("")
        else:
            error = result.get("error", "Unknown error")
            self._status.setStyleSheet("font-size: 11px; color: #EF5350; padding-top: 2px;")
            self._status.setText(f"Pipeline failed: {error}")
            for step_id, card in self._step_cards.items():
                if card._icon.text() == _ICON_ACTIVE:
                    card.set_error(error)
                    break

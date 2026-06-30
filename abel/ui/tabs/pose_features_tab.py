"""Pose Features tab — extract features and pre-build the Active-Learning cache.

This is the feature-preparation step of the pipeline.  It runs in two phases:

1. **Kinematic windows** (``.npz``) — fast, pose-only, used by motif/syllable
   discovery.  No video is decoded in this phase.
2. **Active-Learning prep** — frame-pose parquet, video context features (when
   "Include video features" is enabled), and the cached frame/segment
   representations.  This is the heavy work that used to run on the first
   Active-Learning training run; doing it here makes that run fast.

Everything is content/config-cached, so re-running only rebuilds genuinely
changed sessions (new clips, changed window/stride).

Pipeline:
    Data Import → Behavior Definitions → Seed Examples
    → **Features** ← here
    → Active Learning (trains on the cache) → Candidate Generation → Review
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QThreadPool

from abel.models.schemas import PoseFeaturePreset, PoseSmoothingSettings
from abel.services.behavior_service import BehaviorService
from abel.services.feature_prep_service import (
    STAGE_CONSOLIDATE,
    STAGE_PREPROCESS,
    STAGE_REPRESENTATIONS,
    FeaturePrepService,
    PrepConfig,
    SessionJob,
)
from abel.services.import_service import ImportService
from abel.services.pose_features_service import PoseFeaturesService, PoseFeatureConfig
from abel.services.roi_service import ROIService
from abel.storage.file_store import read_json, read_yaml, write_yaml
from abel.ui.smoothing_preview_dialog import SmoothingPreviewDialog
from abel.ui.widgets.progress_panel import ProgressPanel
from abel.utils.run_timeline import RunTimeline, Stage
from abel.workers.task_worker import TaskWorker

# Timeline stage for the fast pose-only (.npz) kinematic extraction that the
# Features tab has always done, before the heavier Active-Learning prep stages.
STAGE_KINEMATICS = "kinematics"


class _SignalPrepObserver:
    """Bridges :class:`FeaturePrepService` progress (worker thread) to the tab's
    Qt signals, which Qt delivers safely on the GUI thread.
    """

    def __init__(self, tab: "PoseFeaturesTab") -> None:
        self._tab = tab

    def stage_start(self, key: str, label: str, total_units: int) -> None:
        self._tab._prep_stage_start.emit(key, label, int(total_units))

    def stage_advance(self, key: str, done_units: int, message: str) -> None:
        self._tab._prep_stage_advance.emit(key, int(done_units), message)

    def stage_done(self, key: str) -> None:
        self._tab._prep_stage_done.emit(key)

    def stage_skip(self, key: str, message: str) -> None:
        self._tab._prep_stage_skip.emit(key, message)

    def log(self, message: str) -> None:
        self._tab._prep_log.emit(message)

logger = logging.getLogger("abel")


class PoseFeaturesTab(QWidget):
    """Configure and run pose feature extraction across all imported sessions."""

    # Emitted from the background worker thread; Qt delivers it to the GUI thread.
    _progress_updated = Signal(int, str)  # (value, format_text)
    segmentation_completed = Signal()  # emitted after a successful extraction run

    # Structured prep-progress signals — emitted from the worker thread and
    # delivered (queued) to GUI-thread slots that drive the timeline + panel.
    _prep_stage_start = Signal(str, str, int)   # (key, label, total_units)
    _prep_stage_advance = Signal(str, int, str)  # (key, done_units, message)
    _prep_stage_done = Signal(str)               # (key)
    _prep_stage_skip = Signal(str, str)          # (key, message)
    _prep_log = Signal(str)                      # (message)

    def __init__(
        self,
        pose_features_service: PoseFeaturesService,
        import_service: ImportService,
        behavior_service: BehaviorService | None = None,
        roi_service: ROIService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = pose_features_service
        self._imports = import_service
        self._behavior_service = behavior_service
        self._rois = roi_service or ROIService()
        self._project_root: Path | None = None
        # Guards against settings being written back while a project is being
        # loaded — restoring presets/spinboxes during init would otherwise fire
        # valueChanged and clobber the saved settings before they're read.
        self._suspend_settings_save = False
        self._manifest = None
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._current_preset: PoseFeaturePreset | None = None
        self._last_run_preset: PoseFeaturePreset | None = None
        self._last_run_session_ids: list[str] = []
        self._preview_dialog: SmoothingPreviewDialog | None = None
        self._prep = FeaturePrepService()
        self._timeline: RunTimeline | None = None
        self._progress_updated.connect(self._apply_progress)
        self._prep_stage_start.connect(self._on_prep_stage_start)
        self._prep_stage_advance.connect(self._on_prep_stage_advance)
        self._prep_stage_done.connect(self._on_prep_stage_done)
        self._prep_stage_skip.connect(self._on_prep_stage_skip)
        self._prep_log.connect(self._append_log)

        # ── No-project placeholder ──────────────────────────────────────
        self._no_project = QLabel(
            "Open or create a project and import sessions before extracting pose features."
        )
        self._no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_project.setWordWrap(True)
        self._no_project.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        # ── Session table ───────────────────────────────────────────────
        self._session_table = QTableWidget(0, 4)
        self._session_table.setHorizontalHeaderLabels(["", "Session", "Pose File", "Status"])
        self._session_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._session_table.verticalHeader().setVisible(False)
        _st_hdr = self._session_table.horizontalHeader()
        _st_hdr.setSectionResizeMode(0, _st_hdr.ResizeMode.ResizeToContents)
        _st_hdr.setSectionResizeMode(1, _st_hdr.ResizeMode.Interactive)
        _st_hdr.setSectionResizeMode(2, _st_hdr.ResizeMode.Stretch)
        _st_hdr.setSectionResizeMode(3, _st_hdr.ResizeMode.ResizeToContents)
        self._session_table.setColumnWidth(1, 120)

        sel_all_btn = QPushButton("Select All")
        sel_none_btn = QPushButton("Select None")
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload sessions and extraction status from project files")
        clear_cache_btn = QPushButton("Clear Cached Features")
        clear_cache_btn.setToolTip(
            "Delete all cached pose, context, and representation features for this "
            "project so the next extraction run rebuilds everything from scratch.\n\n"
            "Use this when feature extraction is wrongly skipping work because of "
            "stale caches. Source pose/video files and settings are not touched."
        )
        sel_all_btn.clicked.connect(self._select_all)
        sel_none_btn.clicked.connect(self._select_none)
        refresh_btn.clicked.connect(self._refresh_clicked)
        clear_cache_btn.clicked.connect(self._clear_cached_features)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Sessions to process:"))
        sel_row.addStretch()
        sel_row.addWidget(clear_cache_btn)
        sel_row.addWidget(refresh_btn)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)

        session_box = QGroupBox("Sessions")
        session_layout = QVBoxLayout(session_box)
        session_layout.addLayout(sel_row)
        session_layout.addWidget(self._session_table)

        # ── Preset selector ─────────────────────────────────────────────
        self._preset_combo = QComboBox()
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        save_preset_btn = QPushButton("Save as Custom")
        save_preset_btn.clicked.connect(self._save_preset)
        suggest_btn = QPushButton("✦  Suggest Settings")
        suggest_btn.setToolTip(
            "Analyse your behavior definitions and imported sessions to recommend "
            "window duration, stride, and smoothing settings"
        )
        suggest_btn.setStyleSheet(
            "QPushButton { background-color: #1A3A52; color: #4FC3F7; "
            "border: 1px solid #0288D1; border-radius: 4px; padding: 4px 10px; font-weight: 600; }"
            "QPushButton:hover { background-color: #0D2B3E; }"
        )
        suggest_btn.clicked.connect(self._suggest_settings)
        est_btn = QPushButton("Estimate Window Count")
        est_btn.setToolTip("Show how many windows will be produced with the current parameters")
        est_btn.clicked.connect(self._estimate_windows)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self._preset_combo, 1)
        preset_row.addWidget(save_preset_btn)
        preset_row.addWidget(suggest_btn)
        preset_row.addWidget(est_btn)

        # ── Parameter form ──────────────────────────────────────────────
        param_box = QGroupBox("Parameters")
        param_form = QFormLayout(param_box)

        self._p_win_dur = QDoubleSpinBox()
        self._p_win_dur.setRange(0.1, 30.0)
        self._p_win_dur.setSuffix(" s")
        self._p_win_dur.setValue(2.0)
        self._p_win_dur.setToolTip(
            "Duration of each feature window in seconds.\n\n"
            "Shorter windows (0.5–1 s) capture brief behaviors (e.g. head dips, freezing bouts).\n"
            "Longer windows (2–4 s) are better for sustained or multi-phase behaviors.\n\n"
            "Rule of thumb: set to roughly 2× the shortest behavior you want to detect. "
            "Use the Suggest button to auto-calculate from your behavior definitions."
        )

        self._p_stride = QDoubleSpinBox()
        self._p_stride.setRange(0.05, 30.0)
        self._p_stride.setSuffix(" s")
        self._p_stride.setValue(1.0)
        self._p_stride.setToolTip(
            "Time between the start of consecutive windows.\n\n"
            "Stride < window duration means windows overlap—this increases resolution "
            "but produces more features (larger dataset, longer extraction).\n"
            "Stride = window duration means no overlap.\n\n"
            "Smaller strides (e.g. 0.25 s) help detect short events precisely. "
            "Larger strides (e.g. 1–2 s) reduce dataset size and speed up training."
        )

        self._p_fps = QDoubleSpinBox()
        self._p_fps.setRange(1.0, 240.0)
        self._p_fps.setSuffix(" fps")
        self._p_fps.setValue(30.0)
        self._p_fps.setToolTip("Acquisition frame rate (used to convert seconds → frames)")

        self._p_likelihood = QDoubleSpinBox()
        self._p_likelihood.setRange(0.0, 1.0)
        self._p_likelihood.setSingleStep(0.05)
        self._p_likelihood.setValue(0.2)
        self._p_likelihood.setToolTip("Body-part detections below this confidence are masked")

        self._p_interp = QCheckBox("Interpolate short pose dropouts")
        self._p_interp.setChecked(True)

        self._p_smooth = QSpinBox()
        self._p_smooth.setRange(1, 31)
        self._p_smooth.setSuffix(" frames")
        self._p_smooth.setValue(5)
        self._p_smooth.setToolTip("Rolling-mean smoothing width applied after interpolation")

        param_form.addRow("Window duration:", self._p_win_dur)
        param_form.addRow("Stride (overlap):", self._p_stride)
        param_form.addRow("Source FPS:", self._p_fps)
        param_form.addRow("Min likelihood:", self._p_likelihood)
        param_form.addRow("", self._p_interp)
        param_form.addRow("Smoothing window:", self._p_smooth)

        # Wire auto-save on every parameter change
        self._p_win_dur.valueChanged.connect(self._save_extraction_settings)
        self._p_stride.valueChanged.connect(self._save_extraction_settings)
        self._p_fps.valueChanged.connect(self._save_extraction_settings)
        self._p_likelihood.valueChanged.connect(self._save_extraction_settings)
        self._p_interp.stateChanged.connect(self._save_extraction_settings)
        self._p_smooth.valueChanged.connect(self._save_extraction_settings)

        self._preview_btn = QPushButton("Preview Video Settings…")
        self._preview_btn.setToolTip(
            "Open a side-by-side video comparison showing raw DLC tracking\n"
            "on the left and the current smoothing, interpolation, and\n"
            "local motion radius settings applied on the right."
        )
        self._preview_btn.clicked.connect(self._open_smoothing_preview)
        param_form.addRow("", self._preview_btn)

        # ── Video features toggle ───────────────────────────────────────
        self._p_use_video = QCheckBox("Include video-derived features (optical flow, motion)")
        self._p_use_video.setChecked(False)
        self._p_use_video.setToolTip(
            "When enabled, context features (optical flow, gradient magnitude, substrate motion) "
            "are extracted from raw video and fusion re-scoring is available.\n\n"
            "When disabled, only pose-derived kinematic features are used. "
            "This is faster and sufficient for many behaviors like freezing or rearing."
        )
        param_form.addRow("", self._p_use_video)
        self._p_use_video.stateChanged.connect(self._save_extraction_settings)

        # ── Local Motion settings ───────────────────────────────────────
        motion_box = QGroupBox("Local Motion Settings")
        motion_form = QFormLayout(motion_box)

        self._local_radius = QSpinBox()
        self._local_radius.setRange(8, 2048)
        self._local_radius.setSingleStep(4)
        self._local_radius.setValue(36)
        self._local_radius.setToolTip(
            "Pixel radius around each tracked body part used for local optical-flow "
            "and substrate-motion calculations.  Larger values capture a wider "
            "neighbourhood around the animal."
        )
        self._local_radius.valueChanged.connect(self._update_motion_area_preview)
        motion_form.addRow("Local radius (px):", self._local_radius)

        self._motion_area_label = QLabel("")
        self._motion_area_label.setWordWrap(True)
        self._motion_area_label.setStyleSheet(
            "color: #90A4AE; font-size: 11px; padding: 2px 0;"
        )
        motion_form.addRow("", self._motion_area_label)

        # ── Feature Selection ───────────────────────────────────────────
        feature_box = QGroupBox("Feature Selection")
        feature_box.setToolTip(
            "Choose which kinematic feature groups are included during\n"
            "representation building and active learning.\n"
            "Unchecked groups are excluded from training."
        )
        feature_layout = QVBoxLayout(feature_box)

        self._feat_per_keypoint = QCheckBox("Per-keypoint kinematics (velocity, speed, acceleration, jerk)")
        self._feat_per_keypoint.setChecked(True)
        self._feat_per_keypoint.setToolTip(
            "Per-body-part velocity, speed, acceleration, and jerk.\n"
            "Produces 5 columns per tracked keypoint."
        )
        self._feat_global_speed = QCheckBox("Global movement (centroid velocity, forepaw/nose speed)")
        self._feat_global_speed.setChecked(True)
        self._feat_global_speed.setToolTip(
            "Centroid velocity, forepaw speed, nose velocity,\n"
            "and vertical velocity components."
        )
        self._feat_oscillation = QCheckBox("Oscillation / rhythm (autocorrelation, FFT frequency, energy)")
        self._feat_oscillation.setChecked(True)
        self._feat_oscillation.setToolTip(
            "Windowed autocorrelation peaks, dominant FFT frequencies,\n"
            "and oscillation energy for forepaw and nose."
        )
        self._feat_orientation = QCheckBox("Spatial / orientation (head pitch, body orientation)")
        self._feat_orientation.setChecked(True)
        self._feat_orientation.setToolTip(
            "Head pitch angle, body axis orientation, and\n"
            "vertical velocity components."
        )
        for cb in (self._feat_per_keypoint, self._feat_global_speed,
                    self._feat_oscillation, self._feat_orientation):
            feature_layout.addWidget(cb)
            cb.stateChanged.connect(self._save_feature_selection)

        self._feat_status_label = QLabel("")
        self._feat_status_label.setStyleSheet("color: #90A4AE; font-size: 11px;")
        feature_layout.addWidget(self._feat_status_label)

        # ── Robustness Features group ───────────────────────────────────
        robustness_box = QGroupBox("Robustness Features")
        robustness_box.setToolTip(
            "Invariant features that improve generalization across sessions,\n"
            "animals, and recording conditions.  These features are computed\n"
            "alongside the standard kinematics above."
        )
        robustness_layout = QVBoxLayout(robustness_box)

        self._feat_egocentric = QCheckBox("Egocentric kinematics (body-frame forward/lateral velocity)")
        self._feat_egocentric.setChecked(True)
        self._feat_egocentric.setToolTip(
            "Adds forward/lateral velocity in the body-centred frame for every keypoint\n"
            "(tail-base origin, nose→tail forward axis).\n"
            "Makes velocity direction features invariant to camera orientation and animal heading."
        )

        self._feat_body_length_norm = QCheckBox("Body-length normalization (body_length_px reference column)")
        self._feat_body_length_norm.setChecked(True)
        self._feat_body_length_norm.setToolTip(
            "Saves the per-frame nose-to-tail body length estimate as a reference column.\n"
            "Also used to normalize pairwise distance features when relative geometry is enabled.\n"
            "Makes spatial features invariant to animal size and camera zoom."
        )

        self._feat_relative_geometry = QCheckBox("Relative geometry (all-pairs normalized distances)")
        self._feat_relative_geometry.setChecked(True)
        self._feat_relative_geometry.setToolTip(
            "Computes pairwise Euclidean distances between all tracked body parts.\n"
            "When body-length normalization is enabled, normalized variants are also added.\n"
            "Translation- and scale-invariant posture descriptors.\n"
            "⚠  Number of columns grows quadratically with keypoint count."
        )

        self._feat_head_direction = QCheckBox("Head direction (ear/nose heading angle + angular velocity)")
        self._feat_head_direction.setChecked(True)
        self._feat_head_direction.setToolTip(
            "Estimates head orientation from ear midpoint → nose vector.\n"
            "Falls back to body axis when ear keypoints are absent.\n"
            "Adds head angle, angular velocity, and head-frame forward/lateral speed.\n"
            "Useful for orienting, approach, and investigatory behaviors."
        )

        self._feat_joint_angles = QCheckBox("Joint angles (spine flexion + limb flexion angles)")
        self._feat_joint_angles.setChecked(True)
        self._feat_joint_angles.setToolTip(
            "Computes angles at joint triplets — e.g. nose-body-tail spine flexion,\n"
            "and elbow/shoulder/knee angles when limb keypoints are present.\n"
            "Rotation-invariant posture descriptors for rearing, grooming, and locomotion.\n"
            "Only triplets where all three keypoints are detected are computed."
        )

        self._feat_spine_curvature = QCheckBox("Spine curvature (requires 3+ midline keypoints)")
        self._feat_spine_curvature.setChecked(False)
        self._feat_spine_curvature.setToolTip(
            "Estimates spine curvature from the mean angular change along ordered midline\n"
            "keypoints (nose → spine1 → spine2 → … → tail_base).\n"
            "Returns zeros / empty when fewer than 3 midline keypoints are present.\n"
            "Useful for models that track spine1/spine2 explicitly."
        )

        for cb in (self._feat_egocentric, self._feat_body_length_norm, self._feat_relative_geometry,
                    self._feat_head_direction, self._feat_joint_angles, self._feat_spine_curvature):
            robustness_layout.addWidget(cb)
            cb.stateChanged.connect(self._save_robustness_feature_selection)

        self._robustness_status_label = QLabel("")
        self._robustness_status_label.setStyleSheet("color: #90A4AE; font-size: 11px;")
        robustness_layout.addWidget(self._robustness_status_label)

        # ── Clip-wise Deltas group ──────────────────────────────────────
        clipdelta_box = QGroupBox("Clip-wise Deltas")
        clipdelta_box.setToolTip(
            "Capture how posture changes across each clip, rather than its average.\n"
            "Computed at the window-aggregation stage from the angle and proximity\n"
            "features above — they require the relevant robustness features to be enabled."
        )
        clipdelta_layout = QVBoxLayout(clipdelta_box)

        self._feat_clipwise_deltas = QCheckBox(
            "Clip-wise deltas (start→end change in angles & proximities)"
        )
        self._feat_clipwise_deltas.setChecked(False)
        self._feat_clipwise_deltas.setToolTip(
            "For every joint/head/body angle and every inter-keypoint distance, adds two\n"
            "per-window statistics:\n"
            "  • _delta : last-frame minus first-frame value (signed net change)\n"
            "  • _trend : least-squares slope across the clip (noise-robust)\n\n"
            "Captures posture evolution — e.g. an animal rising from a crouch, or two\n"
            "body parts drawing together — that mean/std aggregates discard.\n"
            "Requires Relative geometry / Joint angles / Head direction to be enabled\n"
            "so the underlying angle and proximity columns exist."
        )
        clipdelta_layout.addWidget(self._feat_clipwise_deltas)
        self._feat_clipwise_deltas.stateChanged.connect(self._save_robustness_feature_selection)

        self._clipdelta_status_label = QLabel("")
        self._clipdelta_status_label.setStyleSheet("color: #90A4AE; font-size: 11px;")
        clipdelta_layout.addWidget(self._clipdelta_status_label)

        # ── Social / Interaction group (multi-animal) ───────────────────
        social_box = QGroupBox("Social / Interaction (multi-animal)")
        social_box.setToolTip(
            "Inter-animal features for projects that track more than one animal.\n"
            "Has no effect on single-animal sessions."
        )
        social_layout = QVBoxLayout(social_box)
        self._feat_social = QCheckBox(
            "Interaction features (inter-animal distance, orientation, contact)"
        )
        self._feat_social.setChecked(False)
        self._feat_social.setToolTip(
            "For each focal animal, computes distance / facing-angle / approach-velocity /\n"
            "bounding-box overlap to every other animal, reduced over conspecifics into a\n"
            "fixed set of social_* columns (nearest + mean). Drives social behaviors\n"
            "(e.g. dominance displacement). Requires a multi-animal pose file."
        )
        social_layout.addWidget(self._feat_social)
        self._feat_social.stateChanged.connect(self._save_robustness_feature_selection)

        self._run_btn = QPushButton("▶  Extract Pose Features")
        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setEnabled(False)
        run_row = QHBoxLayout()
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch()
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn.clicked.connect(self._cancel)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")
        self._progress.setValue(0)

        # Rich stage-aware progress + ETA panel (hidden until a run starts).
        self._prep_panel = ProgressPanel("Feature preparation")
        self._prep_panel.hide()

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)
        self._log.setMaximumHeight(200)
        self._log.setPlaceholderText("Extraction log will appear here…")

        # ── Results table ───────────────────────────────────────────────
        self._result_table = QTableWidget(0, 5)
        self._result_table.setHorizontalHeaderLabels(
            ["Session", "Frames", "Windows", "Body Parts", "Status"]
        )
        self._result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.horizontalHeader().setStretchLastSection(True)

        # ── Left panel ──────────────────────────────────────────────────
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(session_box)
        left_layout.addLayout(preset_row)
        left_layout.addWidget(param_box)
        left_layout.addWidget(motion_box)
        left_layout.addWidget(feature_box)
        left_layout.addWidget(robustness_box)
        left_layout.addWidget(clipdelta_box)
        left_layout.addWidget(social_box)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)

        # ── Right panel ─────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Feature explanation banner
        info_label = QLabel(
            "ℹ  Running here also prepares everything Active Learning needs: pose-feature\n"
            "tables, video context (when enabled), and the cached frame/segment\n"
            "representations. Active Learning then just trains on the cache — so the\n"
            "first training run is fast. Re-runs are cheap; only changed clips/settings rebuild."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "background: #0D2B3E; color: #4FC3F7; border: 1px solid #0288D1; "
            "border-radius: 4px; padding: 8px; font-size: 11px; font-weight: 600;"
        )

        right_layout.addWidget(info_label)
        right_layout.addLayout(run_row)
        right_layout.addWidget(self._prep_panel)
        right_layout.addWidget(self._progress)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self._log)
        right_layout.addWidget(QLabel("Results:"))
        right_layout.addWidget(self._result_table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([440, 440])
        self._splitter = splitter

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._no_project)
        root.addWidget(splitter)
        splitter.hide()

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._no_project.hide()
        self._splitter.show()
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        # Suppress settings writes for the duration of the load: restoring
        # presets/spinboxes fires valueChanged, which would otherwise persist
        # default values over the project's saved settings (e.g. flipping the
        # "Include video features" checkbox back off) before they're loaded.
        self._suspend_settings_save = True
        try:
            self._service.set_project(project_root)
            self._manifest = self._imports.load_manifest(project_root)
            self._refresh_presets()
            self._refresh_sessions()
            # Restore all extraction settings from project config
            self._load_extraction_settings()
            # Load local motion radius from ROI config
            try:
                radius = self._rois.local_motion_radius(project_root)
                self._local_radius.setValue(radius)
            except Exception:
                pass
            self._update_motion_area_preview()
            self._load_feature_selection()
            self._load_robustness_feature_selection()
        finally:
            self._suspend_settings_save = False
    def _refresh_presets(self) -> None:
        presets = self._service.load_project_presets()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for p in presets:
            self._preset_combo.addItem(p.name, userData=p)
        self._preset_combo.blockSignals(False)
        if presets:
            self._on_preset_changed(0)

    def _refresh_sessions(self) -> None:
        """Reload manifest and repopulate session table, marking already-extracted sessions."""
        if self._project_root:
            self._manifest = self._imports.load_manifest(self._project_root)
        self._session_table.setRowCount(0)
        if not self._manifest:
            return

        summaries = {s.session_id: s for s in self._service.load_all_summaries()}

        for s in self._manifest.linked_sessions:
            pose = next((p for p in self._manifest.poses if p.asset_id == s.pose_asset_id), None)
            row = self._session_table.rowCount()
            self._session_table.insertRow(row)

            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked)
            chk.setData(Qt.ItemDataRole.UserRole, s.session_id)
            self._session_table.setItem(row, 0, chk)
            self._session_table.setItem(row, 1, QTableWidgetItem(s.session_id))
            self._session_table.setItem(row, 2, QTableWidgetItem(
                Path(pose.source_path).name if pose else "—"
            ))
            summary = summaries.get(s.session_id)
            status = f"✓ {summary.n_windows} windows" if summary else "Not extracted"
            self._session_table.setItem(row, 3, QTableWidgetItem(status))

    def _refresh_clicked(self) -> None:
        """Manual refresh after re-importing sessions or regenerating feature files."""
        if not self._project_root:
            QMessageBox.information(self, "No Project", "Open a project first.")
            return
        self._refresh_sessions()
        self._update_motion_area_preview()
        self._append_log("Session list refreshed.")

    def _clear_cached_features(self) -> None:
        """Delete all cached features so the next run rebuilds from scratch."""
        if not self._project_root:
            QMessageBox.information(self, "No Project", "Open a project first.")
            return
        if not self._run_btn.isEnabled():
            QMessageBox.information(
                self, "Extraction Running",
                "Wait for the current extraction to finish before clearing caches.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Clear Cached Features",
            "Delete all cached pose, context, and representation features for this "
            "project?\n\n"
            "The next extraction run will rebuild everything from scratch. This fixes "
            "stale caches that cause feature extraction to be skipped incorrectly.\n\n"
            "Your source pose/video files and extraction settings are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            info = FeaturePrepService.clear_feature_caches(self._project_root)
        except Exception as exc:
            logger.error("Failed to clear feature caches", exc_info=True)
            QMessageBox.critical(self, "Clear Failed", f"Could not clear caches:\n{exc}")
            return

        mb = info.get("n_bytes", 0) / (1024 * 1024)
        removed = ", ".join(info.get("removed", [])) or "nothing (no caches found)"
        self._append_log(
            f"Cleared cached features: removed {removed} "
            f"({info.get('n_files', 0)} files, {mb:.1f} MB freed)."
        )
        self._refresh_sessions()
        QMessageBox.information(
            self,
            "Caches Cleared",
            f"Removed {info.get('n_files', 0)} cached file(s) ({mb:.1f} MB).\n\n"
            "Run “Extract Pose Features” to rebuild them.",
        )

    # ------------------------------------------------------------------
    # Extraction settings persistence
    # ------------------------------------------------------------------

    _SETTINGS_KEY = "feature_extraction"

    def _save_extraction_settings(self, _value: object = None) -> None:
        """Persist all extraction parameters to project.yaml on every change."""
        if not self._project_root or self._suspend_settings_save:
            return
        try:
            path = self._project_root / "project.yaml"
            raw = read_yaml(path, {})
            raw[self._SETTINGS_KEY] = {
                "window_duration_sec": round(self._p_win_dur.value(), 4),
                "stride_sec": round(self._p_stride.value(), 4),
                "source_fps": round(self._p_fps.value(), 4),
                "likelihood_threshold": round(self._p_likelihood.value(), 4),
                "interpolate_dropouts": self._p_interp.isChecked(),
                "smoothing_window": self._p_smooth.value(),
                "use_video_features": self._p_use_video.isChecked(),
            }
            write_yaml(path, raw)
        except Exception:
            logger.debug("Failed to save extraction settings", exc_info=True)

    def _load_extraction_settings(self) -> None:
        """Restore extraction parameters from project.yaml on project open."""
        if not self._project_root:
            return
        try:
            raw = read_yaml(self._project_root / "project.yaml", {})
            cfg = raw.get(self._SETTINGS_KEY)
            if not cfg:
                # Fall back to legacy keys in behavior_model
                model = raw.get("behavior_model") or {}
                self._p_use_video.setChecked(bool(model.get("use_video_features", False)))
                return

            # Block signals while bulk-loading to avoid N redundant writes
            widgets = [
                self._p_win_dur, self._p_stride, self._p_fps,
                self._p_likelihood, self._p_interp, self._p_smooth,
                self._p_use_video,
            ]
            for w in widgets:
                w.blockSignals(True)

            self._p_win_dur.setValue(float(cfg.get("window_duration_sec", 2.0)))
            self._p_stride.setValue(float(cfg.get("stride_sec", 1.0)))
            self._p_fps.setValue(float(cfg.get("source_fps", 30.0)))
            self._p_likelihood.setValue(float(cfg.get("likelihood_threshold", 0.2)))
            self._p_interp.setChecked(bool(cfg.get("interpolate_dropouts", True)))
            self._p_smooth.setValue(int(cfg.get("smoothing_window", 5)))
            self._p_use_video.setChecked(bool(cfg.get("use_video_features", False)))

            for w in widgets:
                w.blockSignals(False)
        except Exception:
            logger.debug("Failed to load extraction settings", exc_info=True)

    # ------------------------------------------------------------------
    # Local motion helpers
    # ------------------------------------------------------------------

    def _update_motion_area_preview(self, _value: int = 0) -> None:
        """Update the label showing the local motion area relative to video resolution."""
        radius = self._local_radius.value()
        diameter = radius * 2
        lines: list[str] = [f"Sampling area: {diameter} \u00d7 {diameter} px ({diameter**2:,} px\u00b2)"]

        if self._manifest:
            for v in self._manifest.videos:
                if v.width and v.height:
                    pct_w = diameter / v.width * 100
                    pct_h = diameter / v.height * 100
                    lines.append(
                        f"  \u2192 {pct_w:.1f}% \u00d7 {pct_h:.1f}% of {v.width}\u00d7{v.height} "
                        f"({Path(v.source_path).name})"
                    )
                    break  # show first video with known resolution
            else:
                lines.append("  (video resolution unknown \u2014 import sessions to see preview)")

        self._motion_area_label.setText("\n".join(lines))

        # Persist to ROI config whenever the value changes
        if self._project_root:
            try:
                cfg = self._rois.load(self._project_root)
                cfg.setdefault("motion", {})["local_radius_px"] = radius
                self._rois.save(self._project_root, cfg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Feature selection helpers
    # ------------------------------------------------------------------

    # Maps each UI checkbox to the feature-name patterns it controls.
    # Patterns are matched with startswith / contains checks against column names
    # in frame_pose.parquet.
    _FEATURE_GROUPS: dict[str, list[str]] = {
        "per_keypoint": ["_velocity_x", "_velocity_y", "_speed", "_acceleration", "_jerk"],
        "global_speed": [
            "centroid_velocity", "forepaw_speed", "forepaw_vertical_velocity",
            "nose_velocity", "nose_vertical_velocity",
        ],
        "oscillation": [
            "forepaw_oscillation_power", "nose_oscillation_power",
            "forepaw_autocorr_peak", "nose_autocorr_peak",
            "forepaw_movement_frequency", "nose_movement_frequency",
            "oscillation_energy", "nose_oscillation_energy",
        ],
        "orientation": ["head_pitch", "body_orientation"],
    }

    def _feature_checkbox_for_group(self, group: str) -> QCheckBox:
        return {
            "per_keypoint": self._feat_per_keypoint,
            "global_speed": self._feat_global_speed,
            "oscillation": self._feat_oscillation,
            "orientation": self._feat_orientation,
        }[group]

    def _save_feature_selection(self, _state: int = 0) -> None:
        """Persist disabled feature groups to config/feature_exclusions.json."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_json, write_json
            excl_path = self._project_root / "config" / "feature_exclusions.json"
            existing = read_json(excl_path, {}) if excl_path.exists() else {}

            # Start from any manually excluded columns from the Feature Audit tab
            manual_excl = set(existing.get("excluded_feature_cols", []))
            # Remove any patterns that belong to our managed groups, then re-add disabled ones
            all_managed_patterns: list[str] = []
            for patterns in self._FEATURE_GROUPS.values():
                all_managed_patterns.extend(patterns)
            managed_prefix = "__feat_group:"
            manual_excl = {e for e in manual_excl if not e.startswith(managed_prefix)}

            disabled_groups: list[str] = []
            for group, patterns in self._FEATURE_GROUPS.items():
                cb = self._feature_checkbox_for_group(group)
                if not cb.isChecked():
                    disabled_groups.append(group)
                    for p in patterns:
                        manual_excl.add(f"{managed_prefix}{p}")

            existing["excluded_feature_cols"] = sorted(manual_excl)
            existing["disabled_feature_groups"] = disabled_groups
            config_dir = self._project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            write_json(excl_path, existing)

            n_enabled = sum(
                1 for g in self._FEATURE_GROUPS
                if self._feature_checkbox_for_group(g).isChecked()
            )
            self._feat_status_label.setText(
                f"{n_enabled}/{len(self._FEATURE_GROUPS)} feature groups enabled"
            )
        except Exception:
            pass

    def _load_feature_selection(self) -> None:
        """Restore feature group checkboxes from config/feature_exclusions.json."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_json
            excl_path = self._project_root / "config" / "feature_exclusions.json"
            if not excl_path.exists():
                return
            data = read_json(excl_path, {})
            disabled = set(data.get("disabled_feature_groups", []))
            for group in self._FEATURE_GROUPS:
                cb = self._feature_checkbox_for_group(group)
                cb.blockSignals(True)
                cb.setChecked(group not in disabled)
                cb.blockSignals(False)
            n_enabled = sum(
                1 for g in self._FEATURE_GROUPS
                if self._feature_checkbox_for_group(g).isChecked()
            )
            self._feat_status_label.setText(
                f"{n_enabled}/{len(self._FEATURE_GROUPS)} feature groups enabled"
            )
        except Exception:
            pass

    # ── Robustness feature save/load ────────────────────────────────────

    def _save_robustness_feature_selection(self, _state: int = 0) -> None:
        """Persist InvariantFeatureConfig toggles to config/experiment.yaml."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_yaml, write_yaml  # noqa: PLC0415
            cfg_path = self._project_root / "config" / "experiment.yaml"
            data = read_yaml(cfg_path, {}) if cfg_path.exists() else {}
            bm = data.setdefault("behavior_model", {})
            inv = bm.setdefault("invariant_features", {})
            inv["enable_egocentric_kinematics"] = self._feat_egocentric.isChecked()
            inv["enable_body_length_normalization"] = self._feat_body_length_norm.isChecked()
            inv["enable_relative_geometry"] = self._feat_relative_geometry.isChecked()
            inv["enable_head_direction"] = self._feat_head_direction.isChecked()
            inv["enable_joint_angles"] = self._feat_joint_angles.isChecked()
            inv["enable_spine_curvature"] = self._feat_spine_curvature.isChecked()
            inv["enable_clipwise_deltas"] = self._feat_clipwise_deltas.isChecked()
            inv["enable_social_features"] = self._feat_social.isChecked()
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            write_yaml(cfg_path, data)
            n_enabled = sum(
                1 for cb in (self._feat_egocentric, self._feat_body_length_norm,
                              self._feat_relative_geometry, self._feat_head_direction,
                              self._feat_joint_angles, self._feat_spine_curvature)
                if cb.isChecked()
            )
            self._robustness_status_label.setText(f"{n_enabled}/6 robustness features enabled")
            self._clipdelta_status_label.setText(
                "Clip-wise deltas enabled" if self._feat_clipwise_deltas.isChecked()
                else "Clip-wise deltas disabled"
            )
        except Exception:
            pass

    def _load_robustness_feature_selection(self) -> None:
        """Restore InvariantFeatureConfig checkboxes from config/experiment.yaml."""
        if not self._project_root:
            return
        try:
            from abel.storage.file_store import read_yaml  # noqa: PLC0415
            cfg_path = self._project_root / "config" / "experiment.yaml"
            if not cfg_path.exists():
                return
            data = read_yaml(cfg_path, {})
            inv = data.get("behavior_model", {}).get("invariant_features", {})
            mapping = {
                "enable_egocentric_kinematics": self._feat_egocentric,
                "enable_body_length_normalization": self._feat_body_length_norm,
                "enable_relative_geometry": self._feat_relative_geometry,
                "enable_head_direction": self._feat_head_direction,
                "enable_joint_angles": self._feat_joint_angles,
                "enable_spine_curvature": self._feat_spine_curvature,
                "enable_clipwise_deltas": self._feat_clipwise_deltas,
                "enable_social_features": self._feat_social,
            }
            for field, cb in mapping.items():
                if field in inv:
                    cb.blockSignals(True)
                    cb.setChecked(bool(inv[field]))
                    cb.blockSignals(False)
            robustness_cbs = (
                self._feat_egocentric, self._feat_body_length_norm,
                self._feat_relative_geometry, self._feat_head_direction,
                self._feat_joint_angles, self._feat_spine_curvature,
            )
            n_enabled = sum(1 for cb in robustness_cbs if cb.isChecked())
            self._robustness_status_label.setText(f"{n_enabled}/6 robustness features enabled")
            self._clipdelta_status_label.setText(
                "Clip-wise deltas enabled" if self._feat_clipwise_deltas.isChecked()
                else "Clip-wise deltas disabled"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Suggest Settings
    # ------------------------------------------------------------------

    def _suggest_settings(self) -> None:
        """Analyse behaviors and imported sessions, then recommend parameters."""
        if not self._project_root:
            QMessageBox.information(self, "No Project", "Open a project first.")
            return

        # ── Gather behavior info ────────────────────────────────────────
        behaviors = []
        if self._behavior_service:
            behaviors = [b for b in self._behavior_service.behaviors if b.is_active]

        durations = [b.min_duration_sec for b in behaviors if b.min_duration_sec > 0]
        shortest_dur = min(durations) if durations else None
        shortest_name = None
        if shortest_dur is not None:
            for b in behaviors:
                if b.min_duration_sec == shortest_dur:
                    shortest_name = b.name
                    break

        # ── Probe session FPS from pose metadata ────────────────────────
        fps_values: list[float] = []
        if self._manifest:
            for s in self._manifest.linked_sessions:
                pose = next(
                    (p for p in self._manifest.poses if p.asset_id == s.pose_asset_id), None
                )
                if pose:
                    pose_path = Path(pose.source_path)
                    if pose_path.exists():
                        try:
                            meta = self._service._pose_service.probe_metadata(pose_path)
                            fps_values.append(float(meta.get("fps", 30.0) or 30.0))
                        except Exception:
                            pass

        detected_fps = round(sum(fps_values) / len(fps_values), 2) if fps_values else None
        working_fps = detected_fps or self._p_fps.value()

        # ── Compute recommendations ─────────────────────────────────────
        # Window duration: needs to be long enough to *contain* the shortest
        # target behavior while still being short enough to localise events.
        # Rule: 2× the shortest min_duration, clamped [0.75 s, 6.0 s],
        # rounded to nearest 0.25 s.
        if shortest_dur is not None:
            raw_win = max(0.75, min(6.0, shortest_dur * 2.0))
        else:
            raw_win = 2.0  # fallback default
        # Round to nearest 0.25
        rec_window = round(round(raw_win / 0.25) * 0.25, 2)

        # Stride: 50 % overlap gives good temporal resolution without
        # excessive redundancy.  Round to nearest 0.25 s.
        raw_stride = rec_window * 0.5
        rec_stride = round(round(raw_stride / 0.25) * 0.25, 2)
        rec_stride = max(0.25, rec_stride)

        # Smoothing: scale with window frames — roughly 1 frame per 4 frames
        # of window, kept odd, clamped [3, 15].
        win_frames = int(rec_window * working_fps)
        raw_smooth = max(3, min(15, win_frames // 8 * 2 + 1))
        # Ensure it is odd
        rec_smooth = raw_smooth if raw_smooth % 2 == 1 else raw_smooth + 1

        # ── Build explanation ───────────────────────────────────────────
        lines: list[str] = ["<b>Recommended settings based on your project:</b><br><br>"]

        if behaviors:
            dur_rows = "".join(
                f"<li>{b.name}: {b.min_duration_sec:.2f} s</li>"
                for b in sorted(behaviors, key=lambda x: x.min_duration_sec)
            )
            lines.append(f"<b>Active behaviors ({len(behaviors)}):</b><ul>{dur_rows}</ul>")
        else:
            lines.append(
                "<i>No behavior definitions found — using defaults. "
                "Define behaviors on the Behaviors tab to get tailored recommendations.</i><br><br>"
            )

        if shortest_dur is not None:
            lines.append(
                f"<b>Shortest behavior:</b> {shortest_name} @ {shortest_dur:.2f} s<br>"
                f"Window set to <b>2\u00d7 {shortest_dur:.2f} s = {rec_window:.2f} s</b> so each "
                "window is long enough to contain one complete instance of even your briefest behavior.<br><br>"
            )

        lines.append(
            f"<b>Stride ({rec_stride:.2f} s = 50% overlap):</b> "
            "Half-window overlap means every point in the recording falls inside at least two "
            "consecutive windows, so brief events near window boundaries are not missed.<br><br>"
        )

        if detected_fps is not None:
            lines.append(
                f"<b>Source FPS:</b> Detected {detected_fps} fps from your pose files.<br><br>"
            )
        else:
            lines.append(
                "<b>Source FPS:</b> Could not detect from pose metadata — "
                f"using current value ({working_fps:.1f} fps).<br><br>"
            )

        lines.append(
            f"<b>Smoothing ({rec_smooth} frames):</b> "
            f"~1 frame per 8 window-frames, kept at an odd value for a symmetric kernel. "
            "Larger windows benefit from more smoothing to suppress single-frame jitter.<br><br>"
        )

        lines.append(
            "<hr><b>Summary of recommended values:</b><br>"
            f"&nbsp;&nbsp;Window duration: <b>{rec_window:.2f} s</b><br>"
            f"&nbsp;&nbsp;Stride: <b>{rec_stride:.2f} s</b><br>"
            f"&nbsp;&nbsp;Source FPS: <b>{working_fps:.2f} fps</b><br>"
            f"&nbsp;&nbsp;Smoothing window: <b>{rec_smooth} frames</b><br>"
        )

        # ── Dialog ──────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Suggested Settings")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)

        text_label = QLabel("".join(lines))
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.TextFormat.RichText)
        text_label.setStyleSheet("font-size: 12px; padding: 4px;")
        layout.addWidget(text_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.setText("Apply These Settings")
        buttons.rejected.connect(dlg.reject)
        apply_btn.clicked.connect(dlg.accept)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._p_win_dur.setValue(rec_window)
            self._p_stride.setValue(rec_stride)
            if detected_fps is not None:
                self._p_fps.setValue(working_fps)
            self._p_smooth.setValue(rec_smooth)
            self._append_log(
                f"Settings updated from suggestion: window={rec_window}s, "
                f"stride={rec_stride}s, smooth={rec_smooth} frames."
            )

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------

    def _on_preset_changed(self, idx: int) -> None:
        preset: PoseFeaturePreset | None = self._preset_combo.itemData(idx)
        if not preset:
            return
        self._current_preset = preset
        self._p_win_dur.setValue(preset.window_duration_sec)
        self._p_stride.setValue(preset.stride_sec)
        self._p_fps.setValue(preset.source_fps)
        self._p_likelihood.setValue(preset.likelihood_threshold)
        self._p_interp.setChecked(preset.interpolate_dropouts)
        self._p_smooth.setValue(preset.smoothing_window)

    def _current_params_as_preset(self) -> PoseFeaturePreset:
        return PoseFeaturePreset(
            preset_id=getattr(self._current_preset, "preset_id", "custom"),
            name=getattr(self._current_preset, "name", "Custom"),
            window_duration_sec=self._p_win_dur.value(),
            stride_sec=self._p_stride.value(),
            source_fps=self._p_fps.value(),
            likelihood_threshold=self._p_likelihood.value(),
            interpolate_dropouts=self._p_interp.isChecked(),
            smoothing_window=self._p_smooth.value(),
        )

    def _save_preset(self) -> None:
        preset = self._current_params_as_preset()
        self._service.save_project_preset(preset)
        self._append_log(f"Preset '{preset.name}' saved.")

    def _estimate_windows(self) -> None:
        if not self._manifest:
            QMessageBox.information(self, "No Sessions", "Import sessions first.")
            return
        fps = self._p_fps.value()
        win = self._p_win_dur.value()
        stride = self._p_stride.value()
        win_f = max(1, int(win * fps))
        stride_f = max(1, int(stride * fps))
        lines = [f"Window: {win}s = {win_f} frames,  Stride: {stride}s = {stride_f} frames\n"]
        for s in self._manifest.linked_sessions:
            pose = next((p for p in self._manifest.poses if p.asset_id == s.pose_asset_id), None)
            pose_path = Path(pose.source_path) if pose else None
            if pose_path and pose_path.exists():
                try:
                    meta = self._service._pose_service.probe_metadata(pose_path)
                    n = meta.get("n_frames", 0)
                    wins = max(0, (n - win_f) // stride_f + 1) if n >= win_f else 0
                    lines.append(f"  {s.session_id}: {n} frames → {wins} windows")
                except Exception as exc:
                    lines.append(f"  {s.session_id}: error — {exc}")
            else:
                lines.append(f"  {s.session_id}: pose file not found")
        QMessageBox.information(self, "Estimated Window Counts", "\n".join(lines))

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _select_all(self) -> None:
        for row in range(self._session_table.rowCount()):
            item = self._session_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

    def _select_none(self) -> None:
        for row in range(self._session_table.rowCount()):
            item = self._session_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _selected_session_ids(self) -> list[str]:
        ids = []
        for row in range(self._session_table.rowCount()):
            item = self._session_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._project_root or not self._manifest:
            QMessageBox.warning(self, "No Project", "Open a project and import sessions first.")
            return
        session_ids = self._selected_session_ids()
        if not session_ids:
            QMessageBox.warning(self, "No Sessions", "Select at least one session to process.")
            return

        preset = self._current_params_as_preset()
        configs: list[PoseFeatureConfig] = []
        for sid in session_ids:
            pose_path = self._imports.pose_path_for_session(self._manifest, sid)
            if not pose_path:
                self._append_log(f"[SKIP] {sid}: no pose file found in manifest.")
                continue
            configs.append(PoseFeatureConfig(
                session_id=sid,
                pose_path=pose_path,
                preset=preset,
            ))

        if not configs:
            QMessageBox.warning(self, "Nothing to Run", "No valid pose paths found.")
            return

        self._cancel_flag[0] = False
        self._last_run_preset = preset
        self._last_run_session_ids = [c.session_id for c in configs]
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._result_table.setRowCount(0)
        self._progress.setMaximum(len(configs))
        self._progress.setValue(0)
        self._progress.setFormat(f"0 / {len(configs)} sessions")

        # ── Build the run timeline + show the rich progress panel ─────────
        use_video = self._p_use_video.isChecked()
        n = len(configs)
        stages = [
            Stage(STAGE_KINEMATICS, "Kinematic windows (.npz)", weight=2.0, total_units=n),
            Stage(STAGE_PREPROCESS,
                  "Pose + context features" if use_video else "Pose features (parquet)",
                  weight=(8.0 if use_video else 3.0), total_units=n),
            Stage(STAGE_CONSOLIDATE, "Consolidate feature caches", weight=0.5),
            Stage(STAGE_REPRESENTATIONS, "Build representations", weight=3.0),
        ]
        self._timeline = RunTimeline(stages, history=self._load_timeline_history())
        self._prep_panel.set_stages([(s.key, s.label) for s in stages])
        self._prep_panel.set_snapshot_provider(
            lambda: self._timeline.snapshot() if self._timeline else None
        )
        self._prep_panel.show()
        self._timeline.start()
        self._timeline.start_stage(STAGE_KINEMATICS, total_units=n)
        self._refresh_prep_panel()

        self._append_log(
            f"Starting feature preparation: {n} session(s), "
            f"preset '{preset.name}' ({preset.window_duration_sec}s windows, "
            f"{preset.stride_sec}s stride){', video context ON' if use_video else ''}."
        )

        worker = TaskWorker(self._run_all, configs, preset)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_error)
        self._pool.start(worker)

    def _run_all(self, configs: list[PoseFeatureConfig], preset: PoseFeaturePreset) -> list:
        from abel.services.pose_features_service import PoseFeatureResult  # noqa: PLC0415
        results: list[PoseFeatureResult] = []
        for i, cfg in enumerate(configs):
            if self._cancel_flag[0]:
                break
            result = self._service.extract_features(cfg, cancel_flag=self._cancel_flag)
            results.append(result)
            done = i + 1
            # Signals cross the thread boundary safely; Qt queues GUI-thread delivery.
            self._progress_updated.emit(done, f"{done} / {len(configs)} sessions")
            self._prep_stage_advance.emit(
                STAGE_KINEMATICS, done, f"Kinematic windows {done}/{len(configs)}: {cfg.session_id}."
            )
        return results

    def _on_finished(self, results: list) -> None:
        """Phase 1 (.npz extraction) done — record it, then start the heavy prep."""
        self._prep_stage_done.emit(STAGE_KINEMATICS)
        self._sync_behavior_model_segment_settings(self._last_run_preset, results)
        total_windows = sum(r.n_windows for r in results)
        self._progress.setFormat(
            f"Kinematics done — {len(results)} session(s), {total_windows} windows"
        )
        self._progress.setValue(self._progress.maximum())
        self._result_table.setRowCount(0)
        for r in results:
            row = self._result_table.rowCount()
            self._result_table.insertRow(row)
            self._result_table.setItem(row, 0, QTableWidgetItem(r.session_id))
            self._result_table.setItem(row, 1, QTableWidgetItem(str(r.n_frames)))
            self._result_table.setItem(row, 2, QTableWidgetItem(str(r.n_windows)))
            bp_text = ", ".join(r.body_parts)
            bp_item = QTableWidgetItem(bp_text)
            bp_item.setToolTip(bp_text)
            self._result_table.setItem(row, 3, bp_item)
            status = "✓ Complete" if r.success else "✗ Failed"
            self._result_table.setItem(row, 4, QTableWidgetItem(status))
            for w in r.warnings:
                self._append_log(f"  ⚠ [{r.session_id}] {w}")
        self._append_log(
            f"Kinematic extraction complete: {len(results)} session(s), "
            f"{total_windows} windows."
        )
        QTimer.singleShot(0, self._refresh_sessions)
        self.segmentation_completed.emit()

        # ── Phase 2: build the cacheable Active-Learning inputs ───────────
        if self._cancel_flag[0]:
            self._finish_prep_ui("Cancelled.")
            return
        jobs = self._build_prep_jobs(self._last_run_session_ids, self._last_run_preset)
        if not jobs:
            self._finish_prep_ui("No sessions available for representation prep.")
            return
        cfg = PrepConfig(
            use_video_features=self._p_use_video.isChecked(),
            segment_window_frames=max(8, int(round(
                float(self._last_run_preset.window_duration_sec) * float(self._last_run_preset.source_fps)))),
            segment_stride_frames=max(1, int(round(
                float(self._last_run_preset.stride_sec) * float(self._last_run_preset.source_fps)))),
            reuse_cached=True,
        )
        self._append_log("Building Active-Learning inputs (pose parquet, context, representations)…")
        worker = TaskWorker(self._run_prep_task, jobs, cfg)
        worker.signals.finished.connect(self._on_prep_finished)
        worker.signals.failed.connect(self._on_error)
        self._pool.start(worker)

    def _run_prep_task(self, jobs: list, cfg: PrepConfig):
        observer = _SignalPrepObserver(self)
        return self._prep.prepare(
            self._project_root, jobs, cfg,
            observer=observer, cancel_flag=self._cancel_flag,
        )

    def _on_prep_finished(self, result) -> None:
        self._save_timeline_history()
        msg = (
            f"Preparation complete — {result.n_segment_rows} segment row(s) ready. "
            f"Reused {result.n_sessions_reused} cached session(s), "
            f"processed {result.n_sessions_processed}."
        )
        self._append_log("✓ " + msg)
        self._finish_prep_ui(msg)
        QTimer.singleShot(0, self._refresh_sessions)

    def _finish_prep_ui(self, status: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._refresh_prep_panel()
        self._prep_panel.stop()
        self._progress.setFormat(status)

    def _on_error(self, traceback_text: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setFormat("Error")
        self._prep_panel.stop()
        self._append_log("Feature preparation failed:")
        self._append_log(traceback_text[:600])
        logger.error("Feature preparation error:\n%s", traceback_text)

    # ── Prep timeline plumbing ──────────────────────────────────────────

    def _build_prep_jobs(self, session_ids: list[str], preset: PoseFeaturePreset | None) -> list:
        if not session_ids or self._manifest is None or preset is None:
            return []
        session_by_id = {
            str(s.session_id): s
            for s in getattr(self._manifest, "linked_sessions", [])
        }
        jobs: list[SessionJob] = []
        for sid in session_ids:
            pose_path = self._imports.pose_path_for_session(self._manifest, sid)
            if not pose_path:
                continue
            video_path = self._imports.video_path_for_session(self._manifest, sid)
            sess = session_by_id.get(str(sid))
            subject_id = str(getattr(sess, "subject_id", "") or sid) if sess else str(sid)
            individuals = list(getattr(sess, "individuals", []) or []) if sess else []
            ind_map = dict(getattr(sess, "individual_subject_map", {}) or {}) if sess else {}
            corrections = list(getattr(sess, "identity_corrections", []) or []) if sess else []
            jobs.append(SessionJob(
                session_id=str(sid),
                subject_id=subject_id,
                pose_path=pose_path,
                video_path=video_path,
                fps=float(preset.source_fps),
                individuals=individuals,
                individual_subject_map=ind_map,
                identity_corrections=corrections,
            ))
        return jobs

    def _refresh_prep_panel(self) -> None:
        if self._timeline is not None:
            self._prep_panel.update_snapshot(self._timeline.snapshot())

    @Slot(str, str, int)
    def _on_prep_stage_start(self, key: str, label: str, total_units: int) -> None:
        if self._timeline is not None:
            self._timeline.start_stage(key, total_units=total_units)
            self._refresh_prep_panel()

    @Slot(str, int, str)
    def _on_prep_stage_advance(self, key: str, done_units: int, message: str) -> None:
        if self._timeline is not None:
            self._timeline.advance(key, done_units)
            self._refresh_prep_panel()

    @Slot(str)
    def _on_prep_stage_done(self, key: str) -> None:
        if self._timeline is not None:
            self._timeline.complete_stage(key)
            self._refresh_prep_panel()

    @Slot(str, str)
    def _on_prep_stage_skip(self, key: str, message: str) -> None:
        if self._timeline is not None:
            self._timeline.skip_stage(key)
            if message:
                self._append_log(message)
            self._refresh_prep_panel()

    def _timeline_history_path(self) -> Path | None:
        if self._project_root is None:
            return None
        return self._project_root / "derived" / "evaluation" / "feature_prep_timeline.json"

    def _load_timeline_history(self) -> dict:
        path = self._timeline_history_path()
        if path is None or not path.exists():
            return {}
        try:
            return read_json(path, {}) or {}
        except Exception:
            return {}

    def _save_timeline_history(self) -> None:
        path = self._timeline_history_path()
        if path is None or self._timeline is None:
            return
        try:
            from abel.storage.file_store import write_json  # noqa: PLC0415
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, self._timeline.to_history())
        except Exception:
            pass

    @Slot(int, str)
    def _apply_progress(self, value: int, text: str) -> None:
        """Slot that receives progress updates from the background thread safely."""
        self._progress.setValue(value)
        self._progress.setFormat(text)

    def _cancel(self) -> None:
        self._cancel_flag[0] = True
        self._append_log("Cancellation requested…")

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Smoothing preview
    # ------------------------------------------------------------------

    def _smoothing_settings_from_ui(self) -> PoseSmoothingSettings:
        return PoseSmoothingSettings(
            likelihood_threshold=self._p_likelihood.value(),
            interpolate_dropouts=self._p_interp.isChecked(),
            smoothing_window=self._p_smooth.value(),
        )

    def _open_smoothing_preview(self) -> None:
        if not self._manifest or not self._manifest.linked_sessions:
            QMessageBox.information(
                self,
                "No Sessions",
                "Import and link at least one session before previewing smoothing.",
            )
            return

        if self._preview_dialog is not None and self._preview_dialog.isVisible():
            self._preview_dialog.raise_()
            self._preview_dialog.activateWindow()
            return

        self._preview_dialog = SmoothingPreviewDialog(
            import_service=self._imports,
            manifest=self._manifest,
            get_smoothing_fn=self._smoothing_settings_from_ui,
            get_local_radius_fn=lambda: self._local_radius.value(),
            project_root=self._project_root,
            parent=self,
        )
        self._preview_dialog.show()

    def _sync_behavior_model_segment_settings(self, preset: PoseFeaturePreset | None, results: list) -> None:
        if self._project_root is None or preset is None:
            return
        if not any(getattr(r, "success", False) for r in results):
            return

        window_frames = max(8, int(round(float(preset.window_duration_sec) * float(preset.source_fps))))
        stride_frames = max(1, int(round(float(preset.stride_sec) * float(preset.source_fps))))
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        model = dict(raw.get("behavior_model") or {})
        model["segment_window_frames"] = int(window_frames)
        model["segment_stride_frames"] = int(stride_frames)
        model["use_video_features"] = self._p_use_video.isChecked()
        raw["behavior_model"] = model
        write_yaml(path, raw)
        self._append_log(
            "Updated project behavior model segment settings from pose extraction: "
            f"window={window_frames} frames, stride={stride_frames} frames."
        )

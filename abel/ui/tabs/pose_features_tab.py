"""Pose Features tab — extract kinematic feature windows from DLC pose files.

This is step 1 of the analysis pipeline.  No video is decoded here —
pose cleaning and feature computation run directly on the tracking CSV/H5.

Pipeline:
    Data Import → Behavior Definitions → Seed Examples
    → **Pose Features** ← here
    → Behavior Representations → Candidate Generation → Review
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
from abel.services.import_service import ImportService
from abel.services.pose_features_service import PoseFeaturesService, PoseFeatureConfig
from abel.services.roi_service import ROIService
from abel.storage.file_store import read_yaml, write_yaml
from abel.ui.smoothing_preview_dialog import SmoothingPreviewDialog
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")


class PoseFeaturesTab(QWidget):
    """Configure and run pose feature extraction across all imported sessions."""

    # Emitted from the background worker thread; Qt delivers it to the GUI thread.
    _progress_updated = Signal(int, str)  # (value, format_text)
    segmentation_completed = Signal()  # emitted after a successful extraction run

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
        self._manifest = None
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._current_preset: PoseFeaturePreset | None = None
        self._last_run_preset: PoseFeaturePreset | None = None
        self._preview_dialog: SmoothingPreviewDialog | None = None
        self._progress_updated.connect(self._apply_progress)

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
        sel_all_btn.clicked.connect(self._select_all)
        sel_none_btn.clicked.connect(self._select_none)
        refresh_btn.clicked.connect(self._refresh_clicked)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Sessions to process:"))
        sel_row.addStretch()
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
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)

        # ── Right panel ─────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Feature explanation banner
        info_label = QLabel(
            "ℹ  Pose feature extraction runs on tracking files only — no video is decoded.\n"
            "Feature matrices are saved to derived/pose_features/ and used by\n"
            "downstream representation modeling and candidate generation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "background: #0D2B3E; color: #4FC3F7; border: 1px solid #0288D1; "
            "border-radius: 4px; padding: 8px; font-size: 11px; font-weight: 600;"
        )

        right_layout.addWidget(info_label)
        right_layout.addLayout(run_row)
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

    # ------------------------------------------------------------------
    # Extraction settings persistence
    # ------------------------------------------------------------------

    _SETTINGS_KEY = "feature_extraction"

    def _save_extraction_settings(self, _value: object = None) -> None:
        """Persist all extraction parameters to project.yaml on every change."""
        if not self._project_root:
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
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._result_table.setRowCount(0)
        self._progress.setMaximum(len(configs))
        self._progress.setValue(0)
        self._progress.setFormat(f"0 / {len(configs)} sessions")
        self._append_log(
            f"Starting pose feature extraction: {len(configs)} session(s), "
            f"preset '{preset.name}' ({preset.window_duration_sec}s windows, "
            f"{preset.stride_sec}s stride)."
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
            # Signal crosses the thread boundary safely; Qt queues delivery on the GUI thread.
            self._progress_updated.emit(done, f"{done} / {len(configs)} sessions")
        return results

    def _on_finished(self, results: list) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._sync_behavior_model_segment_settings(self._last_run_preset, results)
        total_windows = sum(r.n_windows for r in results)
        self._progress.setFormat(
            f"Done — {len(results)} session(s), {total_windows} total windows"
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
            f"Feature extraction complete: {len(results)} session(s), "
            f"{total_windows} windows ready for downstream modeling."
        )
        # Refresh session status column on next event cycle to avoid painter conflict
        QTimer.singleShot(0, self._refresh_sessions)
        self.segmentation_completed.emit()

    def _on_error(self, traceback_text: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setFormat("Error")
        self._append_log("Extraction failed:")
        self._append_log(traceback_text[:600])
        logger.error("Pose feature extraction error:\n%s", traceback_text)

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

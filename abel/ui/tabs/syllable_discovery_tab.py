"""Syllable Discovery tab — Keypoint-MoSeq based behavior syllable discovery.

Runs Keypoint-MoSeq on imported pose trajectories to discover syllables,
then builds behavior signatures from seed examples.

Pipeline position:
    Data Import → Behavior Definitions → Seed Examples → Pose Features
    → **Syllable Discovery** ← here
    → Behavior Signature Builder → Candidate Retrieval
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.services.behavior_service import BehaviorService
from abel.models.schemas import BehaviorSignature
from abel.services.behavior_signature_builder import (
    BehaviorSignatureBuilder,
    BehaviorSignatureBuilderConfig,
    BehaviorSignatureBuilderResult,
)
from abel.services.import_service import ImportService
from abel.services.keypoint_moseq_service import (
    KeypointMoSeqConfig,
    KeypointMoSeqResult,
    KeypointMoSeqService,
)
from abel.services.seed_service import SeedService
from abel.services.umap_qc_service import (
    QCResult,
    UMAPConfig,
    UMAPQCService,
)
from abel.services.umap_export import QCExportConfig
from abel.services.syllable_clip_service import (
    SyllableClipConfig,
    SyllableClipResult,
    SyllableClipService,
)
from abel.services.preprocessing_service import DEFAULT_PRESETS, ClipExtractionService
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")


class SyllableDiscoveryTab(QWidget):
    """Keypoint-MoSeq based syllable discovery and behavior signature building."""

    progress_update_requested = Signal(int, int, str)
    umap_ready = Signal(object, object)  # (xy: np.ndarray, labels: np.ndarray)
    umap_failed = Signal(str)
    qc_complete = Signal(object)   # QCResult
    qc_failed = Signal(str)
    clips_ready = Signal(object)   # SyllableClipResult
    clips_failed = Signal(str)

    def __init__(
        self,
        syllable_service: KeypointMoSeqService,
        signature_builder: BehaviorSignatureBuilder,
        import_service: ImportService,
        seed_service: SeedService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._syllable_service = syllable_service
        self._signature_builder = signature_builder
        self._imports = import_service
        self._seed_service = seed_service
        self._behavior_service = behavior_service
        self._project_root: Path | None = None
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._qc_cancel_flag: list[bool] = [False]
        self._qc_service = UMAPQCService()
        self._clip_cancel_flag: list[bool] = [False]
        self._clip_service = SyllableClipService()
        self._clip_svc_presets = ClipExtractionService().default_presets
        self.progress_update_requested.connect(self._on_progress_update)
        self.umap_ready.connect(self._on_umap_complete)
        self.umap_failed.connect(self._on_umap_failed)
        self.qc_complete.connect(self._on_qc_complete)
        self.qc_failed.connect(self._on_qc_failed)
        self.clips_ready.connect(self._on_clips_complete)
        self.clips_failed.connect(self._on_clips_failed)

        # ── No-project placeholder ──────────────────────────────────────
        self._no_project = QLabel(
            "Open or create a project, import sessions, and extract pose features "
            "before discovering syllables."
        )
        self._no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_project.setWordWrap(True)
        self._no_project.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        # ── Left: Discovery parameters ─────────────────────────────────
        left_layout = QVBoxLayout()

        # Keypoint-MoSeq discovery group
        discovery_group = QGroupBox("Keypoint-MoSeq Syllable Discovery")
        discovery_form = QFormLayout(discovery_group)

        self._model_name_input = QLineEdit("moseq_model_v1")
        self._backend_combo = QComboBox()
        self._backend_combo.addItem("Keypoint-MoSeq", userData="keypoint_moseq")
        self._backend_combo.addItem("Temporal K-Means", userData="temporal_kmeans")
        self._backend_combo.setToolTip(
            "Select the syllable discovery engine for this run.\n"
            "Temporal K-Means: lightweight temporal clustering.\n"
            "Keypoint-MoSeq: full AR-HMM based keypoint modeling."
        )

        self._n_syllables_spin = QSpinBox()
        self._n_syllables_spin.setRange(2, 200)
        self._n_syllables_spin.setValue(50)
        self._n_syllables_spin.setToolTip(
            "How many distinct movement syllables to discover.\n\n"
            "Typical ranges by assay complexity:\n"
            "  Simple / constrained (one task, few behaviors)  →  20–30\n"
            "  Standard assay (open field, EPM, social)         →  40–70\n"
            "  Rich / long recording (home cage, full day)      →  80–120\n\n"
            "Too few: distinct postures get merged together, hiding real behaviors.\n"
            "Too many: single movements get split into near-identical fragments.\n\n"
            "Start at 50 and adjust after reviewing the distribution tab.\n"
            "Syllables that are never enriched in any seed signature can be ignored."
        )

        self._n_lags_spin = QSpinBox()
        self._n_lags_spin.setRange(1, 10)
        self._n_lags_spin.setValue(2)
        self._n_lags_spin.setToolTip("Number of time lags for AR model")

        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(10, 5000)
        self._max_iterations_spin.setValue(200)
        self._max_iterations_spin.setSingleStep(50)
        self._max_iterations_spin.setToolTip(
            "Total Gibbs sampling iterations (split evenly between AR-HMM phase and full model phase).\n\n"
            "Lower values are faster but less converged:\n"
            "  Quick test / debugging  →  50–100\n"
            "  Standard run            →  200–400\n"
            "  High-quality final run  →  500–1000\n\n"
            "Note: JAX JIT-compiles the model on the first run, which can take\n"
            "several minutes regardless of iteration count."
        )

        self._learning_rate = QDoubleSpinBox()
        self._learning_rate.setRange(0.00001, 0.1)
        self._learning_rate.setValue(0.0001)
        self._learning_rate.setSingleStep(0.0001)
        self._learning_rate.setDecimals(5)

        self._overwrite_chk = QCheckBox("Overwrite existing results")
        self._overwrite_chk.setChecked(True)
        self._overwrite_chk.setToolTip(
            "If checked, any previously saved results.h5 entries for these sessions\n"
            "will be overwritten. Uncheck to skip sessions that have already been processed."
        )

        discovery_form.addRow("Model name:", self._model_name_input)
        discovery_form.addRow("Backend:", self._backend_combo)
        discovery_form.addRow("N syllables:", self._n_syllables_spin)
        discovery_form.addRow("N lags:", self._n_lags_spin)
        discovery_form.addRow("Max iterations:", self._max_iterations_spin)
        discovery_form.addRow("Learning rate:", self._learning_rate)
        discovery_form.addRow("", self._overwrite_chk)

        left_layout.addWidget(discovery_group)

        # Behavior signature group
        signature_group = QGroupBox("Behavior Signature Builder")
        signature_form = QFormLayout(signature_group)

        self._behavior_combo = QComboBox()
        self._behavior_combo.setToolTip("Select target behavior to build signature for")

        self._refresh_behaviors_btn = QPushButton("↻")
        self._refresh_behaviors_btn.setFixedWidth(30)
        self._refresh_behaviors_btn.setToolTip("Refresh behavior list")
        self._refresh_behaviors_btn.clicked.connect(self._refresh_behaviors)

        behavior_row = QHBoxLayout()
        behavior_row.addWidget(self._behavior_combo, 1)
        behavior_row.addWidget(self._refresh_behaviors_btn)
        behavior_row_widget = QWidget()
        behavior_row_widget.setLayout(behavior_row)

        self._build_signature_btn = QPushButton("Build Signature from Seeds")
        self._build_signature_btn.clicked.connect(self._build_signature)

        signature_form.addRow("Behavior:", behavior_row_widget)
        signature_form.addRow("", self._build_signature_btn)

        left_layout.addWidget(signature_group)

        # Run discovery button
        self._run_discovery_btn = QPushButton("▶ Run Syllable Discovery")
        self._run_discovery_btn.setMinimumHeight(40)
        self._run_discovery_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; font-size: 13px;"
        )
        self._run_discovery_btn.clicked.connect(self._run_discovery)

        self._clear_btn = QPushButton("Clear Existing")
        self._clear_btn.clicked.connect(self._clear_existing)

        self._cancel_btn = QPushButton("Stop")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_discovery)

        self._umap_btn = QPushButton("🗺  Generate UMAP")
        self._umap_btn.setEnabled(False)
        self._umap_btn.setToolTip(
            "Compute a 2-D UMAP of the pose-feature space coloured by syllable.\n"
            "Requires umap-learn. Saves a PNG to derived/syllables/ and opens a viewer."
        )
        self._umap_btn.clicked.connect(self._generate_umap)

        self._qc_btn = QPushButton("📊  Full Model QC Export")
        self._qc_btn.setEnabled(False)
        self._qc_btn.setToolTip(
            "Generate a comprehensive QC report for the trained syllable model:\n"
            "  • Standard UMAP (dark + light theme, high-res)\n"
            "  • Density UMAP (hexbin)\n"
            "  • Per-syllable highlight panels\n"
            "  • Compactness / occupancy chart\n"
            "  • Transition graph\n"
            "  • Composite QC dashboard\n"
            "  • QC metrics CSV + JSON report\n\n"
            "Outputs are saved to results/model_qc/<model_name>/."
        )
        self._qc_btn.clicked.connect(self._generate_full_qc)
        self._qc_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold;"
        )

        self._qc_cancel_btn = QPushButton("Stop QC")
        self._qc_cancel_btn.setEnabled(False)
        self._qc_cancel_btn.clicked.connect(self._cancel_qc)

        # ── QC settings widgets (shown via settings dialog) ───────────────
        self._qc_settings_btn = QPushButton("⚙️  QC Settings…")
        self._qc_settings_btn.setToolTip("Configure QC export options.")
        self._qc_settings_btn.clicked.connect(self._open_qc_settings)

        # All QC settings widgets live inside a persistent panel parented to self.
        # This prevents them from being destroyed when the settings dialog closes
        # (re-parenting into a dialog and back is safe; orphaned local QWidget
        # containers that held checkboxes were being GC'd after __init__).
        self._qc_settings_panel = QWidget(self)
        self._qc_settings_panel.hide()
        _sp_form = QFormLayout(self._qc_settings_panel)
        _sp_form.setContentsMargins(8, 8, 8, 8)
        _sp_form.setSpacing(8)

        self._qc_theme_combo = QComboBox()
        self._qc_theme_combo.addItem("Dark", userData=True)
        self._qc_theme_combo.addItem("Light", userData=False)
        _sp_form.addRow("Theme:", self._qc_theme_combo)

        self._qc_dpi_combo = QComboBox()
        self._qc_dpi_combo.addItem("150 dpi  (screen/preview)", userData=150)
        self._qc_dpi_combo.addItem("300 dpi  (publication)", userData=300)
        self._qc_dpi_combo.addItem("600 dpi  (high-res print)", userData=600)
        self._qc_dpi_combo.setCurrentIndex(1)  # 300 dpi default
        _sp_form.addRow("Resolution:", self._qc_dpi_combo)

        self._qc_format_png = QCheckBox("PNG")
        self._qc_format_png.setChecked(True)
        self._qc_format_svg = QCheckBox("SVG")
        self._qc_format_svg.setChecked(True)
        self._qc_format_pdf = QCheckBox("PDF")
        _fmt_row = QHBoxLayout()
        _fmt_row.addWidget(self._qc_format_png)
        _fmt_row.addWidget(self._qc_format_svg)
        _fmt_row.addWidget(self._qc_format_pdf)
        _fmt_row.addStretch()
        _fmt_widget = QWidget()
        _fmt_widget.setLayout(_fmt_row)
        _sp_form.addRow("Formats:", _fmt_widget)

        self._qc_subsample_combo = QComboBox()
        self._qc_subsample_combo.addItem("Stratified (recommended)", userData="stratified")
        self._qc_subsample_combo.addItem("Uniform random", userData="uniform")
        self._qc_subsample_combo.addItem("All frames (slow)", userData="all")
        _sp_form.addRow("Subsample:", self._qc_subsample_combo)

        self._qc_max_frames_spin = QSpinBox()
        self._qc_max_frames_spin.setRange(10_000, 500_000)
        self._qc_max_frames_spin.setValue(60_000)
        self._qc_max_frames_spin.setSingleStep(10_000)
        self._qc_max_frames_spin.setToolTip(
            "Maximum frames used for the UMAP embedding.\n"
            "Larger values capture more structure but are slower."
        )
        _sp_form.addRow("Max frames:", self._qc_max_frames_spin)

        self._qc_include_labels = QCheckBox("Show syllable labels")
        self._qc_include_labels.setChecked(True)
        _sp_form.addRow("", self._qc_include_labels)

        self._qc_density = QCheckBox("Density UMAP")
        self._qc_density.setChecked(True)
        self._qc_per_syllable = QCheckBox("Per-syllable panels")
        self._qc_per_syllable.setChecked(True)
        _plots_w1 = QWidget()
        _plots_row1 = QHBoxLayout()
        _plots_row1.setContentsMargins(0, 0, 0, 0)
        _plots_row1.addWidget(self._qc_density)
        _plots_row1.addWidget(self._qc_per_syllable)
        _plots_row1.addStretch()
        _plots_w1.setLayout(_plots_row1)
        _sp_form.addRow("Plots:", _plots_w1)

        self._qc_compactness = QCheckBox("Compactness chart")
        self._qc_compactness.setChecked(True)
        self._qc_transition = QCheckBox("Transition graph")
        self._qc_transition.setChecked(True)
        self._qc_dashboard = QCheckBox("QC dashboard")
        self._qc_dashboard.setChecked(True)
        _plots_w2 = QWidget()
        _plots_row2 = QHBoxLayout()
        _plots_row2.setContentsMargins(0, 0, 0, 0)
        _plots_row2.addWidget(self._qc_compactness)
        _plots_row2.addWidget(self._qc_transition)
        _plots_row2.addWidget(self._qc_dashboard)
        _plots_row2.addStretch()
        _plots_w2.setLayout(_plots_row2)
        _sp_form.addRow("", _plots_w2)

        # ── Representative clip extraction group ────────────────────
        clip_group = QGroupBox("Representative Clip Extraction")
        clip_form = QFormLayout(clip_group)

        self._clip_n_spin = QSpinBox()
        self._clip_n_spin.setRange(1, 50)
        self._clip_n_spin.setValue(3)
        self._clip_n_spin.setToolTip(
            "Number of clips to extract per syllable.\n"
            "Clips are selected by highest syllable enrichment —\n"
            "windows where the largest fraction of frames belong to that syllable."
        )

        self._clip_min_bout_spin = QSpinBox()
        self._clip_min_bout_spin.setRange(1, 500)
        self._clip_min_bout_spin.setValue(10)
        self._clip_min_bout_spin.setToolTip(
            "Minimum number of syllable frames required within the clip window.\n"
            "Windows with fewer than this many frames of the target syllable are skipped."
        )

        self._clip_frames_spin = QSpinBox()
        self._clip_frames_spin.setRange(10, 1000)
        self._clip_frames_spin.setValue(90)
        self._clip_frames_spin.setSingleStep(10)
        self._clip_frames_spin.setToolTip(
            "Width of the sliding window in source-rate frames.\n"
            "e.g. 90 frames ≈ 3 s at 30 fps. Each extracted clip will be this length."
        )

        self._clip_preset_combo = QComboBox()
        for preset in self._clip_svc_presets:
            self._clip_preset_combo.addItem(preset.name, userData=preset)
        self._clip_preset_combo.setToolTip(
            "Video output preset — controls resolution, FPS, and crop margin."
        )

        clip_form.addRow("Clips / syllable:", self._clip_n_spin)
        clip_form.addRow("Min syl frames:", self._clip_min_bout_spin)
        clip_form.addRow("Clip length (frames):", self._clip_frames_spin)
        clip_form.addRow("Output preset:", self._clip_preset_combo)

        self._clip_btn = QPushButton("🎬  Extract Representative Clips")
        self._clip_btn.setEnabled(False)
        self._clip_btn.setToolTip(
            "For each syllable, scan all sessions with a sliding window and\n"
            "extract the top-N clips with the highest syllable enrichment.\n"
            "The syllable label is burned into the top-left corner of each clip.\n\n"
            "Clips are saved to results/syllable_clips/<model>/syllable_NNN/."
        )
        self._clip_btn.clicked.connect(self._extract_representative_clips)
        self._clip_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold;"
        )

        self._clip_cancel_btn = QPushButton("Stop Extraction")
        self._clip_cancel_btn.setEnabled(False)
        self._clip_cancel_btn.clicked.connect(self._cancel_clip_extraction)

        left_layout.addStretch()
        left_layout.addWidget(self._run_discovery_btn)
        left_layout.addWidget(self._umap_btn)
        left_layout.addWidget(self._qc_btn)
        qc_row = QHBoxLayout()
        qc_row.addWidget(self._qc_settings_btn)
        qc_row.addWidget(self._qc_cancel_btn)
        left_layout.addLayout(qc_row)
        left_layout.addWidget(clip_group)
        left_layout.addWidget(self._clip_btn)
        left_layout.addWidget(self._clip_cancel_btn)
        left_layout.addWidget(self._clear_btn)
        left_layout.addWidget(self._cancel_btn)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ── Right: Progress + log + results ───────────────────────────
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(100)
        self._progress.setValue(0)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(80)
        self._log_area.setMaximumHeight(200)

        # ── Results tabs ───────────────────────────────────────────────
        self._syllable_tab = QWidget()
        self._syllable_tab_layout = QVBoxLayout(self._syllable_tab)
        self._syllable_tab_layout.addWidget(
            QLabel("Run syllable discovery to see the frequency distribution here.")
        )

        self._signature_tab = QWidget()
        self._signature_tab_layout = QVBoxLayout(self._signature_tab)
        self._signature_tab_layout.addWidget(
            QLabel("Build a behavior signature to see enrichment and duration stats here.")
        )

        self._results_tabs = QTabWidget()
        self._results_tabs.addTab(self._syllable_tab, "Syllable Distribution")
        self._results_tabs.addTab(self._signature_tab, "Behavior Signature")

        # QC results tab
        self._qc_tab = QWidget()
        self._qc_tab_layout = QVBoxLayout(self._qc_tab)
        self._qc_tab_layout.addWidget(
            QLabel(
                "Run \"Full Model QC Export\" to generate high-resolution UMAP figures,\n"
                "compactness metrics, transition graph, and a QC dashboard.\n"
                "Outputs are saved to results/model_qc/<model_name>/."
            )
        )
        self._results_tabs.addTab(self._qc_tab, "Model QC")

        # Syllable clips results tab
        self._clips_tab = QWidget()
        self._clips_tab_layout = QVBoxLayout(self._clips_tab)
        self._clips_tab_layout.addWidget(
            QLabel(
                'Click \"Extract Representative Clips\" to sample the longest bouts\n'
                "for each syllable and save mp4 clips to\n"
                "results/syllable_clips/<model_name>/syllable_NNN/."
            )
        )
        self._results_tabs.addTab(self._clips_tab, "Syllable Clips")

        right_layout.addWidget(QLabel("Progress:"))
        right_layout.addWidget(self._progress)
        right_layout.addWidget(QLabel("Status:"))
        right_layout.addWidget(self._status_label)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self._log_area)
        right_layout.addWidget(self._results_tabs, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # ── Main splitter ──────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 600])

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._no_project)
        root.addWidget(splitter)
        splitter.hide()
        self._splitter = splitter

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._syllable_service.set_project(project_root)
        self._signature_builder.set_project(project_root)
        self._qc_service.set_project(project_root)
        self._clip_service.set_project(project_root)
        self._no_project.hide()
        self._splitter.show()
        self._refresh_behaviors()
        self._load_existing_results()
        logger.info("Syllable Discovery tab ready: %s", project_root)

    def _load_existing_results(self) -> None:
        existing = self._syllable_service.load_existing_result()
        if not existing or not existing.success:
            self._umap_btn.setEnabled(False)
            self._qc_btn.setEnabled(False)
            self._clip_btn.setEnabled(False)
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self._status_label.setText(
                f"Loaded existing discovery ({existing.n_syllables} syllables, "
                f"{len(existing.syllable_assignments)} session(s))"
            )
            self._status_label.setStyleSheet("color: #4FC3F7; font-weight: bold;")
            self._umap_btn.setEnabled(True)
            self._qc_btn.setEnabled(True)
            self._clip_btn.setEnabled(True)
            self._log_area.append(
                f"Loaded existing syllable results from project storage: "
                f"{existing.n_syllables} syllables, {len(existing.syllable_assignments)} session(s)."
            )
            self._plot_syllable_distribution(existing)

        self._refresh_signature_visualization()

    def _refresh_behaviors(self) -> None:
        """Populate behavior dropdown."""
        self._behavior_combo.blockSignals(True)
        current = self._behavior_combo.currentData()
        self._behavior_combo.clear()
        self._behavior_combo.addItem("(all behaviors)", userData=None)
        for behavior in self._behavior_service.behaviors:
            self._behavior_combo.addItem(behavior.name, userData=behavior.behavior_id)
        if current is not None:
            idx = self._behavior_combo.findData(current)
            if idx >= 0:
                self._behavior_combo.setCurrentIndex(idx)
        elif self._behavior_combo.count() > 0:
            self._behavior_combo.setCurrentIndex(0)
        self._behavior_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Discovery workflow
    # ------------------------------------------------------------------

    def _run_discovery(self) -> None:
        """Run Keypoint-MoSeq syllable discovery."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        # Get session IDs from import manifest
        manifest = self._imports.load_manifest(self._project_root)
        if not manifest or not manifest.linked_sessions:
            QMessageBox.warning(self, "No Sessions", "Import sessions first.")
            return

        session_ids = [s.session_id for s in manifest.linked_sessions]
        config = KeypointMoSeqConfig(
            session_ids=session_ids,
            model_name=self._model_name_input.text(),
            backend=str(self._backend_combo.currentData()),
            n_syllables=self._n_syllables_spin.value(),
            n_lags=self._n_lags_spin.value(),
            max_iterations=self._max_iterations_spin.value(),
            learning_rate=self._learning_rate.value(),
            overwrite=self._overwrite_chk.isChecked(),
        )

        # Check for existing results and prompt user before overwriting
        results_path = (
            self._project_root
            / "derived"
            / "syllables"
            / "kpm_fit"
            / config.model_name
            / "results.h5"
        )
        if results_path.exists():
            answer = QMessageBox.question(
                self,
                "Existing Results Found",
                f"Results already exist for model '{config.model_name}':\n"
                f"{results_path}\n\n"
                "Delete the existing results and run a fresh analysis?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            results_path.unlink()

        # Run in background
        self._run_discovery_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_flag[0] = False
        backend_label = self._backend_combo.currentText()
        self._log_area.append(f"Starting syllable discovery ({backend_label})...")

        def run_discovery_task():
            self.progress_update_requested.emit(0, 8, "Starting syllable discovery...")
            result = self._syllable_service.run_discovery(
                config,
                progress_callback=lambda done, total, msg: self.progress_update_requested.emit(
                    done, total, msg
                ),
                cancel_flag=self._cancel_flag,
            )
            return result

        worker = TaskWorker(run_discovery_task)
        worker.signals.finished.connect(self._on_discovery_complete)
        worker.signals.failed.connect(self._on_discovery_failed)
        self._pool.start(worker)

    def _cancel_discovery(self) -> None:
        """Cancel running discovery."""
        self._cancel_flag[0] = True
        self._cancel_btn.setEnabled(False)

    def _clear_existing(self) -> None:
        if not self._project_root:
            return
        answer = QMessageBox.question(
            self,
            "Clear Existing Syllables",
            "Delete the saved syllable model and assignments for this project?\n\n"
            "This removes files in derived/syllables/.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed = self._syllable_service.clear_results()
        self._clear_layout(self._syllable_tab_layout)
        self._syllable_tab_layout.addWidget(
            QLabel("Run syllable discovery to see the frequency distribution here.")
        )
        self._status_label.setText("Syllable results cleared")
        self._status_label.setStyleSheet("color: #F57C00; font-weight: bold;")
        if removed > 0:
            self._log_area.append(f"Cleared existing syllable results ({removed} file(s)).")
        else:
            self._log_area.append("No saved syllable results were found to clear.")

    @Slot(object)
    def _on_discovery_complete(self, result: KeypointMoSeqResult) -> None:
        """Handle discovery completion."""
        self._run_discovery_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(100 if result.success else self._progress.value())
        if result.warnings:
            for warning in result.warnings:
                self._log_area.append(f"⚠  {warning}")
        if result.success:
            n_sessions = len(result.syllable_assignments)
            self._log_area.append(
                f"Discovery complete: {result.n_syllables} syllables found across {n_sessions} session(s)"
            )
            self._status_label.setText("Syllable discovery complete ✓")
            self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self._umap_btn.setEnabled(True)
            self._qc_btn.setEnabled(True)
            self._clip_btn.setEnabled(True)
            self._plot_syllable_distribution(result)
            # Clear stale signatures built on the old syllable model
            n_cleared = self._signature_builder.clear_all_signatures()
            if n_cleared:
                self._log_area.append(
                    f"ℹ️  Cleared {n_cleared} behavior signature(s) — they were built on the "
                    "previous model. Use 'Build Signature from Seeds' to rebuild."
                )
            self._refresh_signature_visualization()
        else:
            self._log_area.append("Discovery did not complete successfully.")
            self._status_label.setText("Discovery failed")
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    @Slot(str)
    def _on_discovery_failed(self, traceback_str: str):
        """Handle unhandled exception from the worker."""
        self._run_discovery_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._log_area.append("ERROR during discovery:")
        self._log_area.append(traceback_str)
        self._status_label.setText("Discovery error")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    def _build_signature(self) -> None:
        """Build behavior signature from seeds."""
        behavior_id = self._behavior_combo.currentData()
        behavior_targets: list[tuple[str, str]] = []
        if behavior_id:
            behavior_targets.append((str(behavior_id), self._behavior_combo.currentText()))
        else:
            for behavior in self._behavior_service.behaviors:
                behavior_targets.append((behavior.behavior_id, behavior.name))

        if not behavior_targets:
            QMessageBox.warning(self, "No Behaviors", "Define behaviors first.")
            return

        model_id = self._model_name_input.text() or "moseq_model_v1"
        successes = 0
        failures = 0
        last_success_result: BehaviorSignatureBuilderResult | None = None
        last_success_name = ""

        for target_id, target_name in behavior_targets:
            seeds = self._seed_service.seeds_for_behavior(target_id)
            if not seeds:
                self._log_area.append(f"↷ Skipping {target_name}: no seed examples.")
                continue

            config = BehaviorSignatureBuilderConfig(
                behavior_id=target_id,
                syllable_model_id=model_id,
                seed_examples=seeds,
            )

            self._log_area.append(f"Building signature for {target_name}...")
            result = self._signature_builder.build_signature(config)
            if result.success:
                successes += 1
                last_success_result = result
                last_success_name = target_name
                n = result.n_syllable_windows_analyzed
                self._log_area.append(f"✓ Signature built for {target_name}: {n} windows analyzed")
            else:
                failures += 1
                self._log_area.append(f"✗ Signature build failed for {target_name}")
                for warning in result.warnings:
                    self._log_area.append(f"  {warning}")

        if successes > 0:
            self._status_label.setText(
                f"Signature build complete ✓ ({successes} succeeded, {failures} failed/skipped)"
            )
            self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if last_success_result is not None:
                self._refresh_signature_visualization(
                    preferred_behavior_id=last_success_result.behavior_id
                )
        else:
            self._status_label.setText("Signature building failed")
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            QMessageBox.warning(
                self,
                "No Signatures Built",
                "No behavior signatures were built. Add seed examples and retry.",
            )

    def _refresh_signature_visualization(self, preferred_behavior_id: str | None = None) -> None:
        signatures = self._signature_builder.load_all_signatures()
        if not signatures:
            self._clear_layout(self._signature_tab_layout)
            self._signature_tab_layout.addWidget(
                QLabel("Build a behavior signature to see enrichment and duration stats here.")
            )
            return

        names_by_id = {b.behavior_id: b.name for b in self._behavior_service.behaviors}

        if len(signatures) == 1:
            behavior_id, sig = next(iter(signatures.items()))
            behavior_name = names_by_id.get(behavior_id, behavior_id)
            single_result = BehaviorSignatureBuilderResult(
                behavior_id=behavior_id,
                signature=sig,
                n_seed_examples=int(sig.n_seed_examples),
                n_syllable_windows_analyzed=0,
                success=True,
            )
            self._plot_behavior_signature(single_result, behavior_name)
            return

        self._plot_behavior_signature_comparison(signatures, names_by_id, preferred_behavior_id)

    def _plot_behavior_signature_comparison(
        self,
        signatures: dict[str, BehaviorSignature],
        names_by_id: dict[str, str],
        preferred_behavior_id: str | None,
    ) -> None:
        """Comparative enrichment analysis across multiple behavior signatures."""
        self._clear_layout(self._signature_tab_layout)
        self._results_tabs.setCurrentIndex(1)

        behavior_ids = sorted(signatures.keys(), key=lambda bid: names_by_id.get(bid, bid).lower())
        if preferred_behavior_id and preferred_behavior_id in behavior_ids:
            behavior_ids.remove(preferred_behavior_id)
            behavior_ids.insert(0, preferred_behavior_id)

        top_syllables: set[str] = set()
        for bid in behavior_ids:
            sig = signatures[bid]
            enriched = sorted(sig.enriched_syllables.items(), key=lambda x: x[1], reverse=True)
            for sid, _score in enriched[:6]:
                top_syllables.add(str(sid))

        if not top_syllables:
            self._signature_tab_layout.addWidget(
                QLabel("No comparative enrichment data found yet for saved signatures.")
            )
            return

        syllable_ids = sorted(top_syllables, key=lambda s: int(s) if str(s).isdigit() else str(s))[:15]
        matrix = np.zeros((len(behavior_ids), len(syllable_ids)), dtype=float)

        for r, bid in enumerate(behavior_ids):
            sig = signatures[bid]
            for c, sid in enumerate(syllable_ids):
                matrix[r, c] = float(sig.enriched_syllables.get(sid, 0.0)) - float(
                    sig.depleted_syllables.get(sid, 0.0)
                )

        summary = QLabel(
            f"<b>Comparative signature analysis:</b> {len(behavior_ids)} behaviors are available. "
            "Each cell shows net enrichment (enriched - depleted) for a syllable. "
            "Brighter teal means stronger enrichment for that behavior."
        )
        summary.setWordWrap(True)
        summary.setTextFormat(Qt.TextFormat.RichText)
        summary.setStyleSheet("color: #90A4AE; font-size: 11px; padding: 4px 4px 8px 4px;")
        self._signature_tab_layout.addWidget(summary)

        try:
            import matplotlib.figure as mfig
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

            fig_h = max(2.8, 0.55 * len(behavior_ids) + 1.1)
            fig_w = max(5.0, 0.5 * len(syllable_ids) + 2.8)
            fig = mfig.Figure(figsize=(fig_w, fig_h), tight_layout=True)
            fig.patch.set_facecolor("#1a2333")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1e2d3d")

            vmax = max(0.1, float(np.max(np.abs(matrix))))
            im = ax.imshow(matrix, cmap="rainbow", aspect="auto", vmin=-vmax, vmax=vmax)

            ax.set_xticks(range(len(syllable_ids)))
            ax.set_xticklabels([f"S{sid}" for sid in syllable_ids], rotation=45, ha="right", fontsize=8, color="#B0BEC5")
            ax.set_yticks(range(len(behavior_ids)))
            ax.set_yticklabels([names_by_id.get(bid, bid) for bid in behavior_ids], fontsize=8, color="#B0BEC5")
            ax.set_title("Behavior vs Syllable Enrichment (comparative)", color="#ECEFF1", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#37474F")

            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(colors="#B0BEC5", labelsize=7)
            cbar.set_label("Net enrichment", color="#B0BEC5", fontsize=8)

            self._signature_tab_layout.addWidget(FigureCanvas(fig))
        except Exception as exc:
            logger.warning("Could not render comparative signature chart: %s", exc)
            self._signature_tab_layout.addWidget(QLabel(f"Comparative chart unavailable: {exc}"))

        lines = []
        for bid in behavior_ids:
            sig = signatures[bid]
            top = sorted(sig.enriched_syllables.items(), key=lambda x: x[1], reverse=True)
            top_label = f"S{top[0][0]} ({top[0][1]:.2f})" if top else "none"
            mean_dur = float(sig.duration_stats.get("mean", 0.0)) if sig.duration_stats else 0.0
            lines.append(f"{names_by_id.get(bid, bid)}: top enriched {top_label}, mean duration {mean_dur:.2f}s")

        details = QLabel("<br>".join(lines))
        details.setWordWrap(True)
        details.setTextFormat(Qt.TextFormat.RichText)
        details.setStyleSheet("color: #80CBC4; font-size: 11px; padding: 8px 4px 2px 4px;")
        self._signature_tab_layout.addWidget(details)

    # ------------------------------------------------------------------
    # Progress updates
    # ------------------------------------------------------------------

    @Slot(int, int, str)
    def _on_progress_update(self, done: int, total: int, message: str) -> None:
        """Update progress bar and status."""
        if total > 0:
            self._progress.setValue(int(100 * done / max(total, 1)))
        self._status_label.setText(message)

        if message.startswith("↺ "):
            # Iteration ticker: replace the last line if it was also an iteration
            # update (same ↺ prefix), so we get a single updating line rather than
            # 50 appended lines flooding the log.
            cursor = self._log_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.LineUnderCursor)
            if cursor.selectedText().startswith("↺ "):
                cursor.removeSelectedText()
                cursor.deletePreviousChar()   # remove the preceding newline
                self._log_area.setTextCursor(cursor)
            self._log_area.append(message)
        else:
            self._log_area.append(message)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _plot_syllable_distribution(self, result: KeypointMoSeqResult) -> None:
        """Bar chart: how many windows were assigned to each syllable."""
        self._clear_layout(self._syllable_tab_layout)
        self._results_tabs.setCurrentIndex(0)

        try:
            import matplotlib.figure as mfig
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

            all_syllables: list[int] = []
            for path in result.syllable_assignments.values():
                data = np.load(path, allow_pickle=True)
                all_syllables.extend(data["syllables"].tolist())

            # Derive syllable count from actual data so stale metadata never
            # produces an incorrect title or phantom zero-height bars.
            observed_max = max(all_syllables) if all_syllables else -1
            n_syllables = (observed_max + 1) if observed_max >= 0 else (result.n_syllables or 0)
            counts = Counter(all_syllables)
            x = list(range(n_syllables))
            y = [counts.get(i, 0) for i in x]
            total_windows = sum(y)
            top_idx = int(np.argmax(y)) if y else 0

            fig = mfig.Figure(figsize=(5, 3.2), tight_layout=True)
            fig.patch.set_facecolor("#1a2333")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1e2d3d")

            bar_colors = ["#42A5F5" if i == top_idx else "#1E88E5" for i in x]
            ax.bar(x, y, color=bar_colors, edgecolor="#0D47A1", linewidth=0.5)
            ax.set_xlabel("Syllable #", color="#B0BEC5", fontsize=9)
            ax.set_ylabel("Windows", color="#B0BEC5", fontsize=9)
            ax.set_title(
                f"{n_syllables} syllables  ·  {total_windows:,} total windows  ·  "
                f"{len(result.syllable_assignments)} session(s)",
                color="#ECEFF1", fontsize=9,
            )
            ax.set_xticks(x)
            ax.tick_params(colors="#B0BEC5", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#37474F")

            self._syllable_tab_layout.addWidget(FigureCanvas(fig))
        except Exception as exc:
            logger.warning("Could not render syllable distribution chart: %s", exc)
            self._syllable_tab_layout.addWidget(QLabel(f"Chart unavailable: {exc}"))

        note = QLabel(
            "Each bar is one syllable — a distinct movement pattern discovered by the model.\n"
            "Taller bars mean that movement pattern appeared more often across your sessions.\n"
            "The highlighted bar is the most common syllable."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #78909C; font-size: 11px; padding: 6px 4px 2px 4px;")
        self._syllable_tab_layout.addWidget(note)

    def _plot_behavior_signature(
        self, result: BehaviorSignatureBuilderResult, behavior_name: str
    ) -> None:
        """Enrichment bar chart + duration stats for the behavior signature."""
        self._clear_layout(self._signature_tab_layout)
        self._results_tabs.setCurrentIndex(1)

        sig = result.signature
        if sig is None:
            self._signature_tab_layout.addWidget(QLabel("No signature data available."))
            return

        # ── Plain-language explanation of "windows analyzed" ───────────
        n = result.n_syllable_windows_analyzed
        explanation = QLabel(
            f"<b>What does '{n} windows analyzed' mean?</b><br>"
            f"Your video data is automatically divided into short overlapping time segments "
            f"called <i>windows</i>. The model labels each window with a syllable — a "
            f"distinct movement pattern. Of all the windows in your sessions, <b>{n}</b> of "
            f"them overlapped with your seed examples for <i>{behavior_name}</i>. The chart "
            f"below shows which syllables were <b>enriched</b> (appeared more during those "
            f"seed moments) vs expected by chance."
        )
        explanation.setWordWrap(True)
        explanation.setTextFormat(Qt.TextFormat.RichText)
        explanation.setStyleSheet("color: #90A4AE; font-size: 11px; padding: 4px 4px 10px 4px;")
        self._signature_tab_layout.addWidget(explanation)

        # ── Enrichment chart ──────────────────────────────────────────
        enriched = sig.enriched_syllables
        depleted = sig.depleted_syllables

        if enriched or depleted:
            try:
                import matplotlib.figure as mfig
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

                all_syls = sorted(
                    set(list(enriched.keys()) + list(depleted.keys())),
                    key=lambda s: -enriched.get(s, 0),
                )
                scores = [enriched.get(s, -depleted.get(s, 0)) for s in all_syls]
                colors = ["#26C6DA" if v >= 0 else "#EF5350" for v in scores]
                labels = [f"Syllable {s}" for s in all_syls]

                fig_h = max(2.5, len(all_syls) * 0.38 + 0.8)
                fig = mfig.Figure(figsize=(5, fig_h), tight_layout=True)
                fig.patch.set_facecolor("#1a2333")
                ax = fig.add_subplot(111)
                ax.set_facecolor("#1e2d3d")

                y_pos = list(range(len(all_syls)))
                ax.barh(y_pos, scores, color=colors, edgecolor="#263238", linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=8, color="#B0BEC5")
                ax.axvline(0, color="#546E7A", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Enrichment score", color="#B0BEC5", fontsize=8)
                ax.set_title(
                    f"Syllable Enrichment — {behavior_name}  "
                    f"(teal = enriched, red = depleted)",
                    color="#ECEFF1", fontsize=9,
                )
                ax.tick_params(colors="#B0BEC5", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#37474F")

                self._signature_tab_layout.addWidget(FigureCanvas(fig))
            except Exception as exc:
                logger.warning("Could not render enrichment chart: %s", exc)
                self._signature_tab_layout.addWidget(QLabel(f"Chart unavailable: {exc}"))
        else:
            self._signature_tab_layout.addWidget(
                QLabel("No enrichment data yet — add seed examples and rebuild the signature.")
            )

        # ── Duration stats ────────────────────────────────────────────
        ds = sig.duration_stats
        if ds:
            mean = ds.get("mean", 0)
            std = ds.get("std", 0)
            mn = ds.get("min", 0)
            mx = ds.get("max", 0)
            stats_label = QLabel(
                f"<b>Seed behavior duration:</b> &nbsp;"
                f"{mean:.2f}s ± {std:.2f}s &nbsp; "
                f"(range {mn:.2f}s – {mx:.2f}s &nbsp;·&nbsp; "
                f"{result.n_seed_examples} seed example(s))"
            )
            stats_label.setWordWrap(True)
            stats_label.setTextFormat(Qt.TextFormat.RichText)
            stats_label.setStyleSheet(
                "color: #80CBC4; font-size: 11px; padding: 8px 4px 2px 4px;"
            )
            self._signature_tab_layout.addWidget(stats_label)

    # ------------------------------------------------------------------
    # UMAP generation
    # ------------------------------------------------------------------

    def _generate_umap(self) -> None:
        """Warn the user, then run UMAP in a background worker."""
        result = self._syllable_service.load_existing_result()
        if not result or not result.success:
            QMessageBox.warning(self, "No Results", "Run syllable discovery first.")
            return

        answer = QMessageBox.question(
            self,
            "Generate UMAP",
            "Computing a UMAP projection can take <b>several minutes</b> on CPU, "
            "depending on the number of sessions and frames.<br><br>"
            "The process will re-load all pose tracks, build temporal embeddings, "
            "then run UMAP dimensionality reduction.<br><br>"
            "Progress will be logged below. Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._umap_btn.setEnabled(False)
        self._log_area.append("─── Starting UMAP generation ───")

        def _run():
            def _prog(msg: str) -> None:
                self.progress_update_requested.emit(0, 0, msg)
                logger.info("UMAP: %s", msg)

            xy, labels = self._syllable_service.build_umap_embeddings(
                result,
                progress_callback=_prog,
            )
            return xy, labels

        worker = TaskWorker(_run)
        worker.signals.finished.connect(self._on_umap_complete)
        worker.signals.failed.connect(self._on_umap_failed)
        self._pool.start(worker)

    @Slot(object)
    def _on_umap_complete(self, payload) -> None:
        """Render and display the UMAP, save PNG to derived/syllables/."""
        xy, labels = payload
        self._umap_btn.setEnabled(True)
        self._log_area.append("UMAP complete — rendering figure...")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            import matplotlib.figure as mfig
            from matplotlib.lines import Line2D

            n_syllables = int(labels.max()) + 1

            # ── Colour palette ─────────────────────────────────────────────
            cmap = plt.colormaps.get_cmap("turbo")
            colours = [cmap(i / max(n_syllables - 1, 1)) for i in range(n_syllables)]

            # ── Publication-quality figure ─────────────────────────────────
            fig = mfig.Figure(figsize=(9, 7.5), dpi=150)
            fig.patch.set_facecolor("#0d1117")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#0d1117")

            # Plot each syllable as a separate scatter so the legend is clean
            alpha = max(0.08, min(0.45, 8_000 / max(len(xy), 1)))
            label_arr = labels.astype(int)
            for syl_id in range(n_syllables):
                mask = label_arr == syl_id
                if not mask.any():
                    continue
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    c=[colours[syl_id]],
                    s=1.2,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )

            # ── Cluster centroid labels ────────────────────────────────────
            for syl_id in range(n_syllables):
                mask = label_arr == syl_id
                if not mask.any():
                    continue
                cx, cy = float(xy[mask, 0].mean()), float(xy[mask, 1].mean())
                ax.text(
                    cx, cy,
                    str(syl_id),
                    fontsize=5 if n_syllables > 40 else 6.5,
                    color="white",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    alpha=0.90,
                    zorder=5,
                )

            # ── Legend (compact, max 30 entries before hiding) ─────────────
            if n_syllables <= 30:
                legend_handles = [
                    Line2D([0], [0], marker="o", color="none",
                           markerfacecolor=colours[i], markersize=5,
                           label=f"Syllable {i}")
                    for i in range(n_syllables)
                ]
                legend = ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    fontsize=5.5,
                    framealpha=0.25,
                    facecolor="#1a2333",
                    edgecolor="#37474F",
                    labelcolor="#ECEFF1",
                    ncol=max(1, n_syllables // 15),
                    markerscale=1.4,
                )

            # ── Axes styling — clean, publication-ready ────────────────────
            ax.set_xlabel("UMAP 1", color="#B0BEC5", fontsize=10, labelpad=8)
            ax.set_ylabel("UMAP 2", color="#B0BEC5", fontsize=10, labelpad=8)
            ax.set_title(
                f"Syllable UMAP  ·  {n_syllables} syllables  ·  {len(xy):,} frames",
                color="#ECEFF1",
                fontsize=11,
                pad=12,
            )
            ax.tick_params(colors="#546E7A", labelsize=7, length=3)
            for spine in ax.spines.values():
                spine.set_visible(False)   # no box frame
            ax.set_xticks([])
            ax.set_yticks([])

            fig.tight_layout(pad=1.4)

            # ── Save PNG ───────────────────────────────────────────────────
            save_path: Path | None = None
            if self._project_root:
                syllables_dir = self._project_root / "derived" / "syllables"
                syllables_dir.mkdir(parents=True, exist_ok=True)
                save_path = syllables_dir / "umap.png"
                fig.savefig(
                    save_path,
                    dpi=200,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                )
                self._log_area.append(f"UMAP saved → {save_path}")

            # ── Pop-up viewer ──────────────────────────────────────────────
            dlg = QDialog(self)
            dlg.setWindowTitle("Syllable UMAP")
            dlg.resize(900, 780)
            dlg.setStyleSheet("background: #0d1117;")

            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(700, 580)

            note = QLabel(
                "Each point is one pose window, coloured by syllable. Numbers mark cluster centroids.\n"
                + (f"Saved to: {save_path}" if save_path else "")
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #78909C; font-size: 10px; padding: 6px 8px 2px 8px;")

            close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close_btn.rejected.connect(dlg.accept)

            layout = QVBoxLayout(dlg)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(canvas, 1)
            layout.addWidget(note)
            layout.addWidget(close_btn)

            dlg.exec()

        except Exception as exc:
            logger.exception("UMAP rendering failed")
            self._log_area.append(f"⚠  UMAP rendering failed: {exc}")
            self._umap_btn.setEnabled(True)

    @Slot(str)
    def _on_umap_failed(self, traceback_str: str) -> None:
        """Handle UMAP worker failure."""
        self._umap_btn.setEnabled(True)
        self._log_area.append("⚠  UMAP generation failed:")
        self._log_area.append(traceback_str)

    # ------------------------------------------------------------------
    # Full Model QC Export
    # ------------------------------------------------------------------

    def _generate_full_qc(self) -> None:
        """Build QCExportConfig from UI settings and run the QC pipeline."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        result = self._syllable_service.load_existing_result()
        if not result or not result.success:
            QMessageBox.warning(self, "No Results", "Run syllable discovery first.")
            return

        # Build export config from UI controls
        formats: list[str] = []
        if self._qc_format_png.isChecked():
            formats.append("png")
        if self._qc_format_svg.isChecked():
            formats.append("svg")
        if self._qc_format_pdf.isChecked():
            formats.append("pdf")
        if not formats:
            formats = ["png"]

        exp_cfg = QCExportConfig(
            export_formats=formats,
            dpi=int(self._qc_dpi_combo.currentData()),
            dark_theme=bool(self._qc_theme_combo.currentData()),
            include_labels=self._qc_include_labels.isChecked(),
            plot_density=self._qc_density.isChecked(),
            plot_per_syllable=self._qc_per_syllable.isChecked(),
            plot_compactness=self._qc_compactness.isChecked(),
            plot_transition_graph=self._qc_transition.isChecked(),
            plot_dashboard=self._qc_dashboard.isChecked(),
        )

        umap_cfg = UMAPConfig(
            subsample_strategy=str(self._qc_subsample_combo.currentData()),
            max_frames=self._qc_max_frames_spin.value(),
        )

        model_name = self._model_name_input.text() or "moseq_model_v1"

        self._qc_btn.setEnabled(False)
        self._qc_cancel_btn.setEnabled(True)
        self._qc_cancel_flag[0] = False
        self._log_area.append(
            f"─── Starting Full Model QC Export (model={model_name}, "
            f"dpi={exp_cfg.dpi}, formats={formats}) ───"
        )
        self.progress_update_requested.emit(0, 0, "Starting QC pipeline...")

        def _run():
            def _prog(msg: str) -> None:
                self.progress_update_requested.emit(0, 0, msg)
                logger.info("QC: %s", msg)

            return self._qc_service.run_full_qc(
                model_name=model_name,
                umap_config=umap_cfg,
                export_config=exp_cfg,
                progress_callback=_prog,
                cancel_flag=self._qc_cancel_flag,
            )

        worker = TaskWorker(_run)
        worker.signals.finished.connect(self.qc_complete)
        worker.signals.failed.connect(self.qc_failed)
        self._pool.start(worker)

    def _cancel_qc(self) -> None:
        self._qc_cancel_flag[0] = True
        self._qc_cancel_btn.setEnabled(False)

    def _open_qc_settings(self) -> None:
        """Open a modal dialog with all QC export settings."""
        dlg = QDialog(self)
        dlg.setWindowTitle("QC Export Settings")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)

        # Add the persistent settings panel into the dialog temporarily.
        # After exec() we re-parent it back to self so it isn't destroyed
        # when the dialog goes out of scope.
        self._qc_settings_panel.show()
        layout.addWidget(self._qc_settings_panel)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.accept)
        layout.addWidget(btns)

        dlg.exec()

        # Re-parent back before the dialog is destroyed.
        self._qc_settings_panel.setParent(self)
        self._qc_settings_panel.hide()

    @Slot(object)
    def _on_qc_complete(self, qc_result: QCResult) -> None:
        """Show QC summary in the Model QC tab."""
        self._qc_btn.setEnabled(True)
        self._qc_cancel_btn.setEnabled(False)

        self._clear_layout(self._qc_tab_layout)
        self._results_tabs.setCurrentIndex(2)  # switch to Model QC tab

        if not qc_result.success:
            warn_text = "\n".join(qc_result.warnings) or "QC did not complete."
            self._qc_tab_layout.addWidget(QLabel(f"⚠  QC incomplete:\n{warn_text}"))
            self._log_area.append("⚠  QC export did not complete successfully.")
            return

        # Success banner
        n_figs = len(qc_result.exported_paths)
        out_dir = str(qc_result.output_dir)
        banner = QLabel(
            f"<b>QC export complete</b> — {n_figs} figure(s) saved.<br>"
            f"Output folder: <span style='color:#80CBC4;'>{out_dir}</span>"
        )
        banner.setWordWrap(True)
        banner.setTextFormat(Qt.TextFormat.RichText)
        banner.setStyleSheet("color: #4CAF50; font-size: 11px; padding: 4px;")
        self._qc_tab_layout.addWidget(banner)

        # "Open folder" button
        open_btn = QPushButton(f"📂  Open QC folder")
        open_btn.setToolTip(out_dir)
        open_btn.clicked.connect(lambda: self._open_folder(qc_result.output_dir))
        self._qc_tab_layout.addWidget(open_btn)

        # ── Over-split assessment ─────────────────────────────────────────
        assessment = qc_result.oversplit_assessment
        if assessment:
            severity = assessment.get("severity", "none")
            _sev_styles = {
                "none":   ("color: #4CAF50;", "background:#1B5E20; color:#C8E6C9;"),
                "low":    ("color: #FFA726;", "background:#E65100; color:#FFE0B2;"),
                "medium": ("color: #FF7043;", "background:#BF360C; color:#FFCCBC;"),
                "high":   ("color: #F44336;", "background:#B71C1C; color:#FFCDD2;"),
            }
            label_style, badge_style = _sev_styles.get(severity, _sev_styles["none"])
            score = assessment.get("score", 0.0)

            header = QLabel(
                f"<b>Over-split Assessment: "
                f"<span style='{badge_style}; padding:2px 7px; border-radius:3px;'>"
                f"&nbsp;{severity.upper()}&nbsp;</span></b>"
                f"&nbsp;&nbsp;score: {score:.2f}"
            )
            header.setWordWrap(True)
            header.setTextFormat(Qt.TextFormat.RichText)
            header.setStyleSheet(f"{label_style} font-size: 12px; padding: 6px 4px 2px 4px;")
            self._qc_tab_layout.addWidget(header)

            explanation = assessment.get("explanation", "")
            if explanation:
                exp_label = QLabel(explanation)
                exp_label.setWordWrap(True)
                exp_label.setStyleSheet(
                    "color: #CFD8DC; font-size: 11px; padding: 2px 4px 4px 12px;"
                )
                self._qc_tab_layout.addWidget(exp_label)

            indicators = assessment.get("indicators", [])
            if indicators:
                ind_html = "<br>".join(f"&#x2022;&nbsp;{ind}" for ind in indicators)
                ind_label = QLabel(f"<b>Specific indicators:</b><br>{ind_html}")
                ind_label.setWordWrap(True)
                ind_label.setTextFormat(Qt.TextFormat.RichText)
                ind_label.setStyleSheet(
                    "color: #90A4AE; font-size: 10px; padding: 2px 4px 8px 12px;"
                )
                self._qc_tab_layout.addWidget(ind_label)

        # Warnings from the report
        if qc_result.warnings:
            warn_label = QLabel(
                "<b>QC Warnings / Heuristics:</b><br>"
                + "<br>".join(f"• {w}" for w in qc_result.warnings)
            )
            warn_label.setWordWrap(True)
            warn_label.setTextFormat(Qt.TextFormat.RichText)
            warn_label.setStyleSheet("color: #FFA726; font-size: 11px; padding: 4px;")
            self._qc_tab_layout.addWidget(warn_label)

        # Metrics mini-table (top 15 rows)
        if qc_result.metrics:
            metrics_label = QLabel("<b>Per-syllable metrics (first 15):</b>")
            metrics_label.setStyleSheet("color: #B0BEC5; font-size: 11px; padding-top: 6px;")
            self._qc_tab_layout.addWidget(metrics_label)

            try:
                import matplotlib.figure as mfig
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                from abel.services.umap_plotting import get_theme, plot_compactness_chart

                theme = get_theme(self._qc_theme_combo.currentData())
                limited = [m.to_dict() for m in qc_result.metrics[:15]]
                fig = plot_compactness_chart(limited, theme, dpi=100)
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(220)
                self._qc_tab_layout.addWidget(canvas)
            except Exception as exc:
                self._qc_tab_layout.addWidget(QLabel(f"Preview unavailable: {exc}"))

        self._qc_tab_layout.addStretch()

        # Log update
        self._log_area.append(
            f"QC export complete: {n_figs} figure(s) → {out_dir}"
        )
        self._status_label.setText("Model QC export complete ✓")
        self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    @Slot(str)
    def _on_qc_failed(self, traceback_str: str) -> None:
        """Handle unhandled exception from the QC worker."""
        self._qc_btn.setEnabled(True)
        self._qc_cancel_btn.setEnabled(False)
        self._log_area.append("⚠  QC export failed:")
        self._log_area.append(traceback_str)
        self._status_label.setText("QC export failed")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    @staticmethod
    def _open_folder(path: Path | None) -> None:
        """Open a folder in the system file explorer."""
        import subprocess  # noqa: PLC0415
        if path and path.exists():
            try:
                import os
                os.startfile(str(path))  # type: ignore[attr-defined]
            except AttributeError:
                subprocess.Popen(["xdg-open", str(path)])

    # ------------------------------------------------------------------
    # Representative clip extraction
    # ------------------------------------------------------------------

    def _extract_representative_clips(self) -> None:
        """Build SyllableClipConfig from UI settings and run the clip extraction pipeline."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        result = self._syllable_service.load_existing_result()
        if not result or not result.success:
            QMessageBox.warning(self, "No Results", "Run syllable discovery first.")
            return

        preset = self._clip_preset_combo.currentData()
        model_name = self._model_name_input.text() or "moseq_model_v1"

        clip_cfg = SyllableClipConfig(
            n_clips_per_syllable=self._clip_n_spin.value(),
            min_syllable_frames=self._clip_min_bout_spin.value(),
            clip_frames=self._clip_frames_spin.value(),
            model_name=model_name,
            preset=preset,
        )

        self._clip_btn.setEnabled(False)
        self._clip_cancel_btn.setEnabled(True)
        self._clip_cancel_flag[0] = False
        self._log_area.append(
            f"─── Starting representative clip extraction "
            f"(model={model_name}, {clip_cfg.n_clips_per_syllable} clips/syllable) ───"
        )
        self.progress_update_requested.emit(0, 0, "Starting clip extraction...")

        discovery_result_snapshot = result

        def _run():
            def _prog(msg: str) -> None:
                self.progress_update_requested.emit(0, 0, msg)
                logger.info("ClipExtract: %s", msg)

            return self._clip_service.run(
                discovery_result=discovery_result_snapshot,
                config=clip_cfg,
                progress_callback=_prog,
                cancel_flag=self._clip_cancel_flag,
            )

        worker = TaskWorker(_run)
        worker.signals.finished.connect(self.clips_ready)
        worker.signals.failed.connect(self.clips_failed)
        self._pool.start(worker)

    def _cancel_clip_extraction(self) -> None:
        self._clip_cancel_flag[0] = True
        self._clip_cancel_btn.setEnabled(False)

    @Slot(object)
    def _on_clips_complete(self, clip_result: SyllableClipResult) -> None:
        """Show clip extraction summary in the Syllable Clips tab."""
        self._clip_btn.setEnabled(True)
        self._clip_cancel_btn.setEnabled(False)

        self._clear_layout(self._clips_tab_layout)
        self._results_tabs.setCurrentIndex(3)  # switch to Syllable Clips tab

        if not clip_result.success or clip_result.total_clips == 0:
            warn_text = "\n".join(clip_result.warnings) or "Clip extraction did not complete."
            self._clips_tab_layout.addWidget(QLabel(f"\u26a0  Extraction incomplete:\n{warn_text}"))
            self._log_area.append("\u26a0  Clip extraction did not complete successfully.")
            return

        # Success banner
        out_dir = str(clip_result.output_dir)
        banner = QLabel(
            f"<b>Clip extraction complete</b> \u2014 {clip_result.total_clips} clip(s) extracted "
            f"across {len(clip_result.per_syllable_clips)} syllable(s).<br>"
            f"Output folder: <span style='color:#80CBC4;'>{out_dir}</span>"
        )
        banner.setWordWrap(True)
        banner.setTextFormat(Qt.TextFormat.RichText)
        banner.setStyleSheet("color: #4CAF50; font-size: 11px; padding: 4px;")
        self._clips_tab_layout.addWidget(banner)

        # "Open folder" button
        open_btn = QPushButton("\ud83d\udcc2  Open clips folder")
        open_btn.setToolTip(out_dir)
        open_btn.clicked.connect(lambda: self._open_folder(clip_result.output_dir))
        self._clips_tab_layout.addWidget(open_btn)

        # Warnings
        if clip_result.warnings:
            warn_label = QLabel(
                "<b>Warnings:</b><br>"
                + "<br>".join(f"\u2022 {w}" for w in clip_result.warnings[:20])
            )
            warn_label.setWordWrap(True)
            warn_label.setTextFormat(Qt.TextFormat.RichText)
            warn_label.setStyleSheet("color: #FFA726; font-size: 11px; padding: 4px;")
            self._clips_tab_layout.addWidget(warn_label)

        # Per-syllable summary table
        summary_label = QLabel("<b>Per-syllable clip counts:</b>")
        summary_label.setStyleSheet("color: #B0BEC5; font-size: 11px; padding-top: 6px;")
        self._clips_tab_layout.addWidget(summary_label)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 2, 4, 2)
        scroll_layout.setSpacing(2)

        sorted_syllables = sorted(
            clip_result.per_syllable_clips.keys(),
            key=lambda sid: -len(clip_result.per_syllable_clips[sid]),
        )
        for syl_id in sorted_syllables:
            clips = clip_result.per_syllable_clips[syl_id]
            total_cands = clip_result.per_syllable_candidates.get(syl_id, 0)
            row_label = QLabel(
                f"  Syllable {syl_id:03d} \u2014 {len(clips)} clip(s) "
                f"(from {total_cands} candidate window(s))"
            )
            row_label.setStyleSheet("color: #CFD8DC; font-size: 10px;")
            scroll_layout.addWidget(row_label)

        scroll_content.setLayout(scroll_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        scroll.setMaximumHeight(260)
        self._clips_tab_layout.addWidget(scroll)
        self._clips_tab_layout.addStretch()

        self._log_area.append(
            f"Clip extraction complete: {clip_result.total_clips} clip(s) \u2192 {out_dir}"
        )
        self._status_label.setText("Clip extraction complete \u2713")
        self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    @Slot(str)
    def _on_clips_failed(self, traceback_str: str) -> None:
        """Handle unhandled exception from the clip extraction worker."""
        self._clip_btn.setEnabled(True)
        self._clip_cancel_btn.setEnabled(False)
        self._log_area.append("\u26a0  Clip extraction failed:")
        self._log_area.append(traceback_str)
        self._status_label.setText("Clip extraction failed")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")

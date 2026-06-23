"""Motif Discovery tab — unsupervised clustering of pose-feature windows.

Takes the kinematic feature matrices produced by the Pose Features tab and
discovers recurring movement motifs via K-Means or UMAP + HDBSCAN.

Pipeline position:
    Data Import → Behavior Definitions → Seed Examples → Pose Features
    → **Motif Discovery** ← here
    → Candidate Generation → Clip Extraction → Review
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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

from abel.models.schemas import MotifDiscoveryPreset
from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.motif_service import MotifDiscoveryResult, MotifDiscoveryService
from abel.services.pose_features_service import PoseFeaturesService
from abel.services.seed_service import SeedService
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger("abel")


class MotifDiscoveryTab(QWidget):
    """Configure and run unsupervised motif discovery on extracted pose features."""

    progress_update_requested = Signal(int, int, str)

    def __init__(
        self,
        motif_service: MotifDiscoveryService,
        pose_features_service: PoseFeaturesService,
        import_service: ImportService,
        seed_service: SeedService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = motif_service
        self._pose_features = pose_features_service
        self._imports = import_service
        self._seed_service = seed_service
        self._behavior_service = behavior_service
        self._project_root: Path | None = None
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._current_preset: MotifDiscoveryPreset | None = None
        self.progress_update_requested.connect(self._on_progress_update)

        # ── No-project placeholder ──────────────────────────────────────
        self._no_project = QLabel(
            "Open or create a project, import sessions, and extract pose features "
            "before running motif discovery."
        )
        self._no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_project.setWordWrap(True)
        self._no_project.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        # ── Seed filter group ───────────────────────────────────────────
        self._behavior_combo = QComboBox()
        self._behavior_combo.currentIndexChanged.connect(self._on_behavior_changed)
        refresh_seed_btn = QPushButton("Refresh")
        refresh_seed_btn.setToolTip("Reload behaviours and seed examples")
        refresh_seed_btn.clicked.connect(self._refresh_seed_clicked)
        behavior_row = QHBoxLayout()
        behavior_row.addWidget(self._behavior_combo, 1)
        behavior_row.addWidget(refresh_seed_btn)
        
        self._seed_filter_chk = QCheckBox(
            "Filter windows to seed examples (recommended)"
        )
        self._seed_filter_chk.setChecked(False)
        self._seed_filter_chk.setToolTip(
            "When checked, only feature windows overlapping your labelled seed examples\n"
            "are used for clustering.  This focuses motif discovery on the kinematic\n"
            "signature of the selected behaviour rather than the entire recording."
        )
        self._seed_filter_chk.toggled.connect(self._update_seed_coverage_label)
        self._seed_coverage_label = QLabel("No seeds loaded.")
        self._seed_coverage_label.setWordWrap(True)
        self._seed_coverage_label.setStyleSheet("color: #78909C; font-size: 11px;")

        seed_form = QFormLayout()
        seed_form.addRow("Behaviour:", behavior_row)
        seed_form.addRow("", self._seed_filter_chk)
        seed_form.addRow(self._seed_coverage_label)

        seed_box = QGroupBox("Seed Filter")
        seed_box.setLayout(seed_form)

        # ── Session table ───────────────────────────────────────────────
        self._session_table = QTableWidget(0, 3)
        self._session_table.setHorizontalHeaderLabels(["", "Session", "Windows"])
        self._session_table.setColumnWidth(0, 30)
        self._session_table.setColumnWidth(1, 160)
        self._session_table.setColumnWidth(2, 90)
        self._session_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._session_table.verticalHeader().setVisible(False)

        sel_all_btn = QPushButton("Select All")
        sel_none_btn = QPushButton("Select None")
        refresh_sessions_btn = QPushButton("Refresh")
        refresh_sessions_btn.setToolTip("Reload sessions and feature extraction status")
        sel_all_btn.clicked.connect(self._select_all)
        sel_none_btn.clicked.connect(self._select_none)
        refresh_sessions_btn.clicked.connect(self._refresh_sessions_clicked)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Sessions with features:"))
        sel_row.addStretch()
        sel_row.addWidget(refresh_sessions_btn)
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
        recommend_btn = QPushButton("Recommend Settings")
        recommend_btn.setToolTip(
            "Auto-fill algorithm and parameters based on selected sessions, "
            "feature-window count, and seed coverage."
        )
        recommend_btn.clicked.connect(self._recommend_settings)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self._preset_combo, 1)
        preset_row.addWidget(recommend_btn)
        preset_row.addWidget(save_preset_btn)

        # ── Algorithm parameter form ────────────────────────────────────
        algo_box = QGroupBox("Algorithm")
        algo_form = QFormLayout(algo_box)

        self._p_algo = QComboBox()
        self._p_algo.addItem("K-Means", userData="kmeans")
        self._p_algo.addItem("K-Means + UMAP", userData="kmeans_umap")
        self._p_algo.addItem("HDBSCAN + UMAP", userData="hdbscan")
        self._p_algo.currentIndexChanged.connect(self._on_algo_changed)
        algo_form.addRow("Algorithm:", self._p_algo)

        # K-Means params
        self._p_clusters = QSpinBox()
        self._p_clusters.setRange(2, 200)
        self._p_clusters.setValue(10)
        self._p_clusters.setToolTip("Number of clusters for K-Means")
        self._p_clusters_row_label = QLabel("N clusters:")
        algo_form.addRow(self._p_clusters_row_label, self._p_clusters)

        # UMAP params
        self._umap_label = QLabel("─── UMAP ───────────────")
        self._umap_label.setStyleSheet("color: #546E7A; font-size: 10px;")
        algo_form.addRow(self._umap_label)

        self._p_umap_n_comp = QSpinBox()
        self._p_umap_n_comp.setRange(2, 50)
        self._p_umap_n_comp.setValue(10)
        self._p_umap_n_comp.setToolTip("Number of UMAP output dimensions fed to the clusterer")
        self._p_umap_n_comp_label = QLabel("N components:")
        algo_form.addRow(self._p_umap_n_comp_label, self._p_umap_n_comp)

        self._p_umap_n_nbrs = QSpinBox()
        self._p_umap_n_nbrs.setRange(2, 100)
        self._p_umap_n_nbrs.setValue(15)
        self._p_umap_n_nbrs.setToolTip("Controls UMAP local vs global structure balance")
        self._p_umap_n_nbrs_label = QLabel("N neighbors:")
        algo_form.addRow(self._p_umap_n_nbrs_label, self._p_umap_n_nbrs)

        self._p_umap_min_dist = QDoubleSpinBox()
        self._p_umap_min_dist.setRange(0.0, 1.0)
        self._p_umap_min_dist.setSingleStep(0.05)
        self._p_umap_min_dist.setValue(0.1)
        self._p_umap_min_dist.setToolTip("UMAP minimum distance (lower = tighter embedding)")
        self._p_umap_min_dist_label = QLabel("Min dist:")
        algo_form.addRow(self._p_umap_min_dist_label, self._p_umap_min_dist)

        # HDBSCAN params
        self._hdbscan_label = QLabel("─── HDBSCAN ─────────────")
        self._hdbscan_label.setStyleSheet("color: #546E7A; font-size: 10px;")
        algo_form.addRow(self._hdbscan_label)

        self._p_hdb_min_cluster = QSpinBox()
        self._p_hdb_min_cluster.setRange(5, 5000)
        self._p_hdb_min_cluster.setValue(50)
        self._p_hdb_min_cluster.setToolTip("Minimum number of windows to form a cluster")
        self._p_hdb_min_cluster_label = QLabel("Min cluster size:")
        algo_form.addRow(self._p_hdb_min_cluster_label, self._p_hdb_min_cluster)

        self._p_hdb_min_samples = QSpinBox()
        self._p_hdb_min_samples.setRange(1, 500)
        self._p_hdb_min_samples.setValue(5)
        self._p_hdb_min_samples.setToolTip("Controls HDBSCAN conservativeness (higher = fewer clusters)")
        self._p_hdb_min_samples_label = QLabel("Min samples:")
        algo_form.addRow(self._p_hdb_min_samples_label, self._p_hdb_min_samples)

        # Common params
        self._p_seed_label = QLabel("─── General ─────────────")
        self._p_seed_label.setStyleSheet("color: #546E7A; font-size: 10px;")
        algo_form.addRow(self._p_seed_label)

        self._p_seed = QSpinBox()
        self._p_seed.setRange(0, 99999)
        self._p_seed.setValue(42)
        self._p_seed.setToolTip("Random seed for reproducibility")
        algo_form.addRow("Random seed:", self._p_seed)

        # ── Left panel ──────────────────────────────────────────────────
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(seed_box)
        left_layout.addWidget(session_box)
        left_layout.addLayout(preset_row)
        left_layout.addWidget(algo_box)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)

        # ── Right panel ─────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        info_label = QLabel(
            "ℹ  Motif discovery clusters pose-feature windows — no video is decoded.\n"
            "Results are saved to derived/motifs/ and used by Candidate Generation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "background: #0D2B3E; color: #4FC3F7; border: 1px solid #0288D1; "
            "border-radius: 4px; padding: 8px; font-size: 11px; font-weight: 600;"
        )

        self._model_status = QLabel("No motif model for this project.")
        self._model_status.setWordWrap(True)
        self._model_status.setStyleSheet("color: #78909C; font-size: 11px; padding: 2px 0;")

        self._run_btn = QPushButton("▶  Run Motif Discovery")
        self._clear_btn = QPushButton("Clear Existing")
        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setEnabled(False)
        run_row = QHBoxLayout()
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._clear_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch()
        self._run_btn.clicked.connect(self._run)
        self._clear_btn.clicked.connect(self._clear_existing)
        self._cancel_btn.clicked.connect(self._cancel)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")
        self._progress.setMaximum(6)
        self._progress.setValue(0)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setPlaceholderText("Discovery log will appear here…")

        # Results table
        self._result_table = QTableWidget(0, 4)
        self._result_table.setHorizontalHeaderLabels(
            ["Motif", "Windows", "% of Total", "Sessions"]
        )
        self._result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.horizontalHeader().setStretchLastSection(True)

        # Visual readout for interpreting discovery quality and behaviour relevance.
        readout_box = QGroupBox("Discovery Readout")
        readout_layout = QVBoxLayout(readout_box)

        self._readout_summary = QLabel(
            "Run motif discovery to see how much of the recording was covered and what this means."
        )
        self._readout_summary.setWordWrap(True)
        self._readout_summary.setStyleSheet("color: #B0BEC5; font-size: 11px;")

        self._bar_cluster_coverage = QProgressBar()
        self._bar_cluster_coverage.setRange(0, 100)
        self._bar_cluster_coverage.setFormat("Clustered windows: —")

        self._bar_assigned_coverage = QProgressBar()
        self._bar_assigned_coverage.setRange(0, 100)
        self._bar_assigned_coverage.setFormat("Assigned to motifs: —")

        self._bar_noise_rate = QProgressBar()
        self._bar_noise_rate.setRange(0, 100)
        self._bar_noise_rate.setFormat("Noise rate: —")

        self._readout_seed_note = QLabel("")
        self._readout_seed_note.setWordWrap(True)
        self._readout_seed_note.setStyleSheet("color: #90A4AE; font-size: 11px;")

        readout_layout.addWidget(self._readout_summary)
        readout_layout.addWidget(self._bar_cluster_coverage)
        readout_layout.addWidget(self._bar_assigned_coverage)
        readout_layout.addWidget(self._bar_noise_rate)
        readout_layout.addWidget(self._readout_seed_note)

        right_layout.addWidget(info_label)
        right_layout.addWidget(self._model_status)
        right_layout.addLayout(run_row)
        right_layout.addWidget(self._progress)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self._log)
        right_layout.addWidget(readout_box)
        right_layout.addWidget(QLabel("Discovered Motifs:"))
        right_layout.addWidget(self._result_table, 1)

        # ── Main splitter ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([420, 460])
        self._splitter = splitter

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._no_project)
        root.addWidget(splitter)
        splitter.hide()

        # Apply initial visibility based on default algorithm
        self._on_algo_changed(0)

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._service.set_project(project_root)
        self._pose_features.set_project(project_root)
        self._no_project.hide()
        self._splitter.show()
        self._behavior_service.set_project(project_root)
        self._seed_service.set_project(project_root)
        self._refresh_presets()
        self._refresh_behavior_combo()
        self._refresh_sessions()
        self._refresh_model_status()

    def _refresh_presets(self) -> None:
        presets = self._service.load_project_presets()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for p in presets:
            self._preset_combo.addItem(p.name, userData=p)
        self._preset_combo.blockSignals(False)
        if presets:
            self._on_preset_changed(0)

    def _refresh_behavior_combo(self) -> None:
        """Rebuild behavior dropdown from BehaviorService, annotated with seed counts."""
        self._behavior_combo.blockSignals(True)
        self._behavior_combo.clear()
        self._behavior_combo.addItem("(all behaviors / no filter)", userData=None)
        counts = self._seed_service.count_by_behavior()
        for b in self._behavior_service.behaviors:
            c = counts.get(b.behavior_id, 0)
            label = f"{b.name}  [{c} seed{'s' if c != 1 else ''}]"
            self._behavior_combo.addItem(label, userData=b.behavior_id)
        self._behavior_combo.blockSignals(False)
        self._update_seed_coverage_label()

    def _on_behavior_changed(self, _idx: int) -> None:
        self._update_seed_coverage_label()

    def _update_seed_coverage_label(self) -> None:
        """Show how many seed windows are available for the current selection."""
        if not self._project_root:
            self._seed_coverage_label.setText("No project loaded.")
            return

        behavior_id: str | None = self._behavior_combo.currentData()
        if behavior_id:
            seeds = self._seed_service.seeds_for_behavior(behavior_id)
        else:
            seeds = self._seed_service.seeds
        positive_seeds = [s for s in seeds if s.label_type == "positive"]

        if not positive_seeds:
            self._seed_coverage_label.setText(
                "⚠  No positive seed examples found for this behaviour.  "
                "Add seeds in the Seed Examples tab first."
            )
            self._seed_coverage_label.setStyleSheet("color: #FF8A65; font-size: 11px;")
            return

        # Estimate how many feature windows overlap the seeds
        if self._seed_filter_chk.isChecked():
            summaries = {s.session_id: s for s in self._pose_features.load_all_summaries()}
            seed_sessions = {s.session_id for s in positive_seeds}
            covered = [sid for sid in seed_sessions if sid in summaries]
            txt = (
                f"✓  {len(positive_seeds)} positive seed(s) across "
                f"{len(seed_sessions)} session(s)  "
                f"({len(covered)} with extracted features)"
            )
            color = "#4FC3F7" if covered else "#FF8A65"
        else:
            txt = (
                f"{len(positive_seeds)} positive seed(s) available — "
                "seed filter is disabled, full recording will be clustered."
            )
            color = "#78909C"
        self._seed_coverage_label.setText(txt)
        self._seed_coverage_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _subject_by_session(self) -> dict[str, str]:
        """Resolve session_id → subject label using the import manifest."""
        if not self._project_root:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, str] = {}
        for session in manifest.linked_sessions:
            subject = (session.subject_id or "").strip()
            if not subject:
                video = video_by_id.get(session.video_asset_id)
                subject = (video.subject_id or "").strip() if video else ""
            out[session.session_id] = subject or session.session_id
        return out

    def _refresh_sessions(self) -> None:
        """Populate session table from sessions that have extracted features."""
        self._session_table.setRowCount(0)
        if not self._project_root:
            return

        summaries = {s.session_id: s for s in self._pose_features.load_all_summaries()}
        if not summaries:
            return

        subject_map = self._subject_by_session()
        for sid, summary in summaries.items():
            row = self._session_table.rowCount()
            self._session_table.insertRow(row)

            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked)
            chk.setData(Qt.ItemDataRole.UserRole, sid)
            self._session_table.setItem(row, 0, chk)
            display_name = subject_map.get(sid) or sid
            self._session_table.setItem(row, 1, QTableWidgetItem(display_name))
            self._session_table.setItem(row, 2, QTableWidgetItem(str(summary.n_windows)))

    def _refresh_sessions_clicked(self) -> None:
        """Manual refresh of sessions after re-importing or regenerating features."""
        if not self._project_root:
            return
        self._refresh_sessions()
        self._append_log("Sessions refreshed.")

    def _refresh_seed_clicked(self) -> None:
        """Manual refresh of behaviors and seeds after changes in other tabs."""
        if not self._project_root:
            return
        self._refresh_behavior_combo()
        self._append_log("Behaviours and seed examples refreshed.")

    def _refresh_model_status(self) -> None:
        model = self._service.load_model()
        if model:
            created = model.created_at.strftime("%Y-%m-%d %H:%M") if model.created_at else "?"
            n_motifs = model.parameters.get("n_motifs", "?")
            n_clustered = model.parameters.get("n_windows_clustered", model.parameters.get("n_windows_total", "?"))
            n_total = model.parameters.get("n_windows_total", "?")
            behavior_id = model.parameters.get("behavior_id")
            seed_filtered = model.parameters.get("seed_filtered", False)
            if behavior_id:
                defn = self._behavior_service.get(behavior_id)
                bname = defn.name if defn else behavior_id
                seed_hint = f"  |  seeded: {bname}"
            else:
                seed_hint = "  |  seeded" if seed_filtered else ""
            self._model_status.setText(
                f"Last model: {model.name}  |  {n_motifs} motifs  |  "
                f"{n_clustered} / {n_total} windows{seed_hint}  |  {created}"
            )
            self._model_status.setStyleSheet("color: #4FC3F7; font-size: 11px; padding: 2px 0;")
            self._result_table.setRowCount(0)
            for entry in model.parameters.get("cluster_summary", []):
                self._add_result_row(
                    entry.get("motif_id", "?"),
                    entry.get("count", 0),
                    n_clustered if isinstance(n_clustered, int) else 0,
                    entry.get("sessions", []),
                )
            self._update_readout(
                n_windows_total=int(model.parameters.get("n_windows_total", 0) or 0),
                n_windows_clustered=int(model.parameters.get("n_windows_clustered", 0) or 0),
                n_windows_assigned=int(model.parameters.get("n_windows_assigned", 0) or 0),
                noise_count=int(model.parameters.get("noise_count", 0) or 0),
                n_motifs=int(model.parameters.get("n_motifs", 0) or 0),
                seed_filtered=bool(model.parameters.get("seed_filtered", False)),
                behavior_id=model.parameters.get("behavior_id"),
            )
        else:
            self._model_status.setText("No motif model for this project — run discovery to create one.")
            self._model_status.setStyleSheet("color: #78909C; font-size: 11px; padding: 2px 0;")
            self._update_readout(0, 0, 0, 0, 0, False, None)

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------

    def _on_preset_changed(self, idx: int) -> None:
        preset: MotifDiscoveryPreset | None = self._preset_combo.itemData(idx)
        if not preset:
            return
        self._current_preset = preset

        # Algorithm combo
        algo_key = preset.algorithm
        if preset.use_umap and algo_key == "kmeans":
            algo_key = "kmeans_umap"
        for i in range(self._p_algo.count()):
            if self._p_algo.itemData(i) == algo_key:
                self._p_algo.setCurrentIndex(i)
                break

        self._p_clusters.setValue(preset.n_clusters)
        self._p_umap_n_comp.setValue(preset.umap_n_components)
        self._p_umap_n_nbrs.setValue(preset.umap_n_neighbors)
        self._p_umap_min_dist.setValue(preset.umap_min_dist)
        self._p_hdb_min_cluster.setValue(preset.hdbscan_min_cluster_size)
        self._p_hdb_min_samples.setValue(preset.hdbscan_min_samples)
        self._p_seed.setValue(preset.random_state)

    def _on_algo_changed(self, _idx: int) -> None:
        algo = self._p_algo.currentData()
        show_kmeans = algo in ("kmeans", "kmeans_umap")
        show_umap = algo in ("kmeans_umap", "hdbscan")
        show_hdbscan = algo == "hdbscan"

        self._p_clusters_row_label.setVisible(show_kmeans)
        self._p_clusters.setVisible(show_kmeans)
        self._umap_label.setVisible(show_umap)
        self._p_umap_n_comp_label.setVisible(show_umap)
        self._p_umap_n_comp.setVisible(show_umap)
        self._p_umap_n_nbrs_label.setVisible(show_umap)
        self._p_umap_n_nbrs.setVisible(show_umap)
        self._p_umap_min_dist_label.setVisible(show_umap)
        self._p_umap_min_dist.setVisible(show_umap)
        self._hdbscan_label.setVisible(show_hdbscan)
        self._p_hdb_min_cluster_label.setVisible(show_hdbscan)
        self._p_hdb_min_cluster.setVisible(show_hdbscan)
        self._p_hdb_min_samples_label.setVisible(show_hdbscan)
        self._p_hdb_min_samples.setVisible(show_hdbscan)

    def _current_params_as_preset(self) -> MotifDiscoveryPreset:
        algo_key = self._p_algo.currentData()  # "kmeans" | "kmeans_umap" | "hdbscan"
        use_umap = algo_key in ("kmeans_umap", "hdbscan")
        algorithm = "hdbscan" if algo_key == "hdbscan" else "kmeans"
        return MotifDiscoveryPreset(
            preset_id=getattr(self._current_preset, "preset_id", "custom"),
            name=getattr(self._current_preset, "name", "Custom"),
            algorithm=algorithm,
            n_clusters=self._p_clusters.value(),
            use_umap=use_umap,
            umap_n_components=self._p_umap_n_comp.value(),
            umap_n_neighbors=self._p_umap_n_nbrs.value(),
            umap_min_dist=self._p_umap_min_dist.value(),
            hdbscan_min_cluster_size=self._p_hdb_min_cluster.value(),
            hdbscan_min_samples=self._p_hdb_min_samples.value(),
            random_state=self._p_seed.value(),
        )

    def _save_preset(self) -> None:
        preset = self._current_params_as_preset()
        self._service.save_project_preset(preset)
        self._append_log(f"Preset '{preset.name}' saved.")

    def _recommend_settings(self) -> None:
        """Auto-select an analysis strategy and parameter defaults for this project state."""
        session_ids = self._selected_session_ids()
        if not session_ids:
            fallback_ids: list[str] = []
            for row in range(self._session_table.rowCount()):
                item = self._session_table.item(row, 0)
                if not item:
                    continue
                sid = item.data(Qt.ItemDataRole.UserRole)
                if sid:
                    fallback_ids.append(str(sid))
            session_ids = fallback_ids

        if not session_ids:
            QMessageBox.information(
                self,
                "No Sessions",
                "No sessions with extracted features were found. Run Pose Features first.",
            )
            return

        summaries = {s.session_id: s for s in self._pose_features.load_all_summaries()}
        total_windows = 0
        for sid in session_ids:
            summary = summaries.get(sid)
            if summary is not None:
                total_windows += int(summary.n_windows)

        selected_behavior_id: str | None = self._behavior_combo.currentData()
        positive_for_selected = self._positive_seed_count(selected_behavior_id)
        positive_all = len([s for s in self._seed_service.seeds if s.label_type == "positive"])

        # Heuristic choice:
        # - Small/sparse data: K-Means (stable, simple)
        # - Medium data: K-Means + UMAP
        # - Large/seed-rich data: HDBSCAN + UMAP
        if total_windows >= 8000 and positive_all >= 30:
            algo = "hdbscan"
            n_clusters = 15
            use_seed_filter = positive_for_selected > 0 or positive_all > 0
            umap_n_components = 10
            umap_n_neighbors = 20
            umap_min_dist = 0.05
            hdb_min_cluster = max(40, min(120, total_windows // 100))
            hdb_min_samples = 8
            rationale = "large dataset with substantial seed coverage"
        elif total_windows >= 2500:
            algo = "kmeans_umap"
            n_clusters = 15 if total_windows < 6000 else 20
            use_seed_filter = positive_for_selected > 0
            umap_n_components = 10
            umap_n_neighbors = 15
            umap_min_dist = 0.1
            hdb_min_cluster = 50
            hdb_min_samples = 5
            rationale = "medium-sized dataset"
        else:
            algo = "kmeans"
            n_clusters = 10 if total_windows < 1200 else 12
            use_seed_filter = positive_for_selected > 0
            umap_n_components = 10
            umap_n_neighbors = 15
            umap_min_dist = 0.1
            hdb_min_cluster = 50
            hdb_min_samples = 5
            rationale = "smaller dataset / fewer windows"

        self._seed_filter_chk.setChecked(use_seed_filter)

        for i in range(self._p_algo.count()):
            if self._p_algo.itemData(i) == algo:
                self._p_algo.setCurrentIndex(i)
                break

        self._p_clusters.setValue(n_clusters)
        self._p_umap_n_comp.setValue(umap_n_components)
        self._p_umap_n_nbrs.setValue(umap_n_neighbors)
        self._p_umap_min_dist.setValue(umap_min_dist)
        self._p_hdb_min_cluster.setValue(hdb_min_cluster)
        self._p_hdb_min_samples.setValue(hdb_min_samples)
        self._on_algo_changed(self._p_algo.currentIndex())

        self._append_log(
            "Recommended settings applied: "
            f"algo={self._p_algo.currentText()}, sessions={len(session_ids)}, "
            f"windows≈{total_windows}, positive_seeds(selected={positive_for_selected}, all={positive_all}), "
            f"seed_filter={'on' if use_seed_filter else 'off'} ({rationale})."
        )

    def _positive_seed_count(self, behavior_id: str | None) -> int:
        if behavior_id:
            return len([s for s in self._seed_service.seeds_for_behavior(behavior_id) if s.label_type == "positive"])
        return len([s for s in self._seed_service.seeds if s.label_type == "positive"])

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
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        session_ids = self._selected_session_ids()
        if not session_ids:
            if self._session_table.rowCount() == 0:
                QMessageBox.warning(
                    self,
                    "No Features",
                    "No sessions with extracted features found.\n"
                    "Run Pose Feature extraction first.",
                )
            else:
                QMessageBox.warning(self, "No Sessions", "Select at least one session.")
            return

        preset = self._current_params_as_preset()

        algo_display = self._p_algo.currentText()
        self._append_log(
            f"Starting motif discovery: {len(session_ids)} session(s), "
            f"algorithm '{algo_display}'."
        )

        behavior_id: str | None = self._behavior_combo.currentData()
        use_seed_filter = self._seed_filter_chk.isChecked()
        behavior_targets: list[tuple[str, str]] = []
        session_id_set = set(session_ids)

        if behavior_id is None:
            behavior_targets = [
                (str(b.behavior_id), str(b.name))
                for b in self._behavior_service.behaviors
                if (b.behavior_id or "").strip()
            ]
            if not behavior_targets:
                QMessageBox.warning(self, "No Behaviors", "Define at least one behavior first.")
                return

            if not use_seed_filter:
                # Separate per-behaviour analyses require behaviour-specific window filters.
                # Enforce seed filtering to avoid repeated pooled clustering runs.
                use_seed_filter = True
                self._seed_filter_chk.setChecked(True)
                self._append_log(
                    "All behaviors selected: enabling seed filter so each behavior is clustered separately."
                )

        if use_seed_filter:
            if behavior_id:
                seeds = [
                    s
                    for s in self._seed_service.seeds_for_behavior(behavior_id)
                    if s.label_type == "positive"
                    and s.session_id in session_id_set
                ]
                if not seeds:
                    QMessageBox.warning(
                        self,
                        "No Seeds",
                        "Seed filter is enabled but no positive seed examples were found in the selected sessions.\n\n"
                        "Either add seeds in the Seed Examples tab, select a behaviour with\n"
                        "seeds, or uncheck 'Filter windows to seed examples'.",
                    )
                    return
                behavior_label = self._behavior_combo.currentText().split("[")[0].strip()
                self._append_log(
                    f"  Seed filter ON — {len(seeds)} positive seed(s), "
                    f"behaviour: {behavior_label}"
                )

            if behavior_targets:
                seeded_count = sum(
                    1
                    for target_id, _ in behavior_targets
                    if any(
                        s.label_type == "positive"
                        and s.session_id in session_id_set
                        for s in self._seed_service.seeds_for_behavior(target_id)
                    )
                )
                if seeded_count == 0:
                    QMessageBox.warning(
                        self,
                        "No Seeds",
                        "All behaviors is selected, but no positive seed examples exist for the selected sessions.",
                    )
                    return
                self._append_log(
                    f"  All behaviors: {seeded_count}/{len(behavior_targets)} behavior(s) have positive seeds in selected sessions."
                )
        else:
            self._append_log("  Seed filter OFF — clustering full recording windows.")

        self._cancel_flag[0] = False
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._result_table.setRowCount(0)
        self._progress.setValue(0)
        self._progress.setFormat("Starting…")

        worker = TaskWorker(
            self._run_discovery_task,
            session_ids,
            preset,
            use_seed_filter,
            behavior_id,
            behavior_targets,
        )
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_error)
        self._pool.start(worker)

    def _run_discovery_task(
        self,
        session_ids: list[str],
        preset: MotifDiscoveryPreset,
        use_seed_filter: bool,
        behavior_id: str | None,
        behavior_targets: list[tuple[str, str]] | None = None,
    ) -> MotifDiscoveryResult:
        behavior_targets = behavior_targets or []

        if behavior_targets:
            aggregate = MotifDiscoveryResult(session_ids=list(session_ids), behavior_id=None)
            aggregate.seed_filtered = bool(use_seed_filter)
            aggregate.cluster_summary = []
            first_model = None
            total_steps = max(1, len(behavior_targets) * 7)

            for idx, (target_id, target_name) in enumerate(behavior_targets):
                if self._cancel_flag[0]:
                    break

                if use_seed_filter:
                    seeds = [
                        s
                        for s in self._seed_service.seeds_for_behavior(target_id)
                        if s.label_type == "positive"
                        and s.session_id in set(session_ids)
                    ]
                    if not seeds:
                        aggregate.warnings.append(
                            f"[{target_name}] skipped: no positive seeds found in selected sessions."
                        )
                        continue
                else:
                    seeds = None

                def _seq_prog(step: int, _total: int, *, _idx: int = idx, _name: str = target_name) -> None:
                    labels = [
                        "Loading features…",
                        "Filtering to seeds…",
                        "Normalising…",
                        "UMAP…",
                        "Clustering…",
                        "Building assignments…",
                        "Done",
                    ]
                    fmt = labels[min(step, len(labels) - 1)]
                    global_step = min(total_steps, (_idx * 7) + step + 1)
                    self.progress_update_requested.emit(
                        global_step,
                        total_steps,
                        f"[{_idx + 1}/{len(behavior_targets)}] {_name}: {fmt}",
                    )

                run = self._service.run_discovery(
                    session_ids=session_ids,
                    preset=preset,
                    seeds=seeds,
                    behavior_id=target_id,
                    progress_callback=_seq_prog,
                    cancel_flag=self._cancel_flag,
                )

                aggregate.warnings.extend(run.warnings)
                if not run.success:
                    continue

                if first_model is None:
                    first_model = run.model

                behavior_tag = "".join(ch.lower() for ch in target_name if ch.isalnum())[:8] or "behavior"
                if len(behavior_targets) > 1:
                    behavior_tag = f"{behavior_tag}_{idx + 1:02d}"

                remapped_assignments = []
                for a in run.assignments:
                    remapped_motif = a.motif_id if a.motif_id == "noise" else f"{behavior_tag}:{a.motif_id}"
                    remapped_assignments.append(
                        a.model_copy(
                            update={
                                "assignment_id": f"{behavior_tag}_{a.assignment_id}",
                                "motif_id": remapped_motif,
                                "behavior_id": target_id,
                            }
                        )
                    )

                aggregate.assignments.extend(remapped_assignments)
                aggregate.n_windows_total += run.n_windows_total
                aggregate.n_windows_clustered += run.n_windows_clustered
                aggregate.n_windows_assigned += run.n_windows_assigned
                aggregate.noise_count += run.noise_count

                for entry in run.cluster_summary:
                    motif_id = str(entry.get("motif_id", "?"))
                    if motif_id != "noise":
                        motif_id = f"{behavior_tag}:{motif_id}"
                    aggregate.cluster_summary.append(
                        {
                            **entry,
                            "motif_id": motif_id,
                            "behavior_id": target_id,
                            "behavior_name": target_name,
                        }
                    )

            aggregate.n_motifs = len({a.motif_id for a in aggregate.assignments if a.motif_id != "noise"})
            aggregate.success = bool(aggregate.assignments)
            if first_model is not None:
                aggregate.model = first_model.model_copy(
                    update={
                        "name": "all_behaviors_sequential",
                        "parameters": {
                            **first_model.parameters,
                            "behavior_id": None,
                            "seed_filtered": aggregate.seed_filtered,
                            "n_windows_total": aggregate.n_windows_total,
                            "n_windows_clustered": aggregate.n_windows_clustered,
                            "n_windows_assigned": aggregate.n_windows_assigned,
                            "n_motifs": aggregate.n_motifs,
                            "noise_count": aggregate.noise_count,
                            "session_ids": list(session_ids),
                            "cluster_summary": aggregate.cluster_summary,
                            "per_behavior": [
                                {
                                    "behavior_id": bid,
                                    "behavior_name": bname,
                                }
                                for bid, bname in behavior_targets
                            ],
                        },
                    }
                )
            return aggregate

        if use_seed_filter:
            seeds = [s for s in self._seed_service.seeds_for_behavior(behavior_id or "") if s.label_type == "positive"]
        else:
            seeds = None

        def _single_prog(step: int, total: int) -> None:
            labels = [
                "Loading features…",
                "Filtering to seeds…",
                "Normalising…",
                "UMAP…",
                "Clustering…",
                "Building assignments…",
                "Done",
            ]
            fmt = labels[min(step, len(labels) - 1)]
            # Relay progress changes to the UI thread to avoid QWidget access
            # from worker threads (can crash with QBackingStore/QPainter errors).
            self.progress_update_requested.emit(step, total, fmt)

        return self._service.run_discovery(
            session_ids=session_ids,
            preset=preset,
            seeds=seeds,
            behavior_id=behavior_id,
            progress_callback=_single_prog,
            cancel_flag=self._cancel_flag,
        )

    def _on_finished(self, result: MotifDiscoveryResult) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

        for w in result.warnings:
            self._append_log(f"  ⚠ {w}")

        if not result.success:
            self._progress.setFormat("Failed — see log")
            self._append_log("Motif discovery did not complete successfully.")
            return

        # Save to disk
        self._service.save_results(result)

        # Update progress
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat(
            f"Done — {result.n_motifs} motifs from {result.n_windows_clustered} "
            f"clustered windows ({result.noise_count} noise)"
        )

        # Populate results table
        self._result_table.setRowCount(0)
        for entry in result.cluster_summary:
            self._add_result_row(
                entry["motif_id"],
                entry["count"],
                result.n_windows_clustered,
                entry["sessions"],
            )

        self._append_log(
            f"Discovery complete: {result.n_motifs} motifs from "
            f"{result.n_windows_clustered} clustered windows "
            f"({result.n_windows_total} total in recording).  "
            f"Noise: {result.noise_count}."
        )
        self._update_readout(
            n_windows_total=result.n_windows_total,
            n_windows_clustered=result.n_windows_clustered,
            n_windows_assigned=result.n_windows_assigned,
            noise_count=result.noise_count,
            n_motifs=result.n_motifs,
            seed_filtered=result.seed_filtered,
            behavior_id=result.behavior_id,
        )

        # Refresh model status on next event cycle to avoid painter conflict
        QTimer.singleShot(0, self._refresh_model_status)

    def _on_error(self, traceback_text: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setFormat("Error")
        self._append_log("Motif discovery failed:")
        self._append_log(traceback_text[:800])
        logger.error("Motif discovery error:\n%s", traceback_text)

    @Slot(int, int, str)
    def _on_progress_update(self, step: int, total: int, fmt: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._progress.setFormat(fmt)

    def _cancel(self) -> None:
        self._cancel_flag[0] = True
        self._append_log("Cancellation requested…")

    def _clear_existing(self) -> None:
        if not self._project_root:
            return
        answer = QMessageBox.question(
            self,
            "Clear Existing Motifs",
            "Delete the saved motif model and assignments for this project?\n\n"
            "This removes derived/motifs/motif_model.json and assignments.json.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed = self._service.clear_results()
        self._result_table.setRowCount(0)
        self._refresh_model_status()
        if removed > 0:
            self._append_log(f"Cleared existing motif results ({removed} file(s)).")
        else:
            self._append_log("No saved motif results were found to clear.")

    # ------------------------------------------------------------------
    # Results table helper
    # ------------------------------------------------------------------

    def _add_result_row(
        self,
        motif_id: str,
        count: int,
        total: int,
        sessions: list[str],
    ) -> None:
        pct = f"{100.0 * count / total:.1f}%" if total > 0 else "—"
        row = self._result_table.rowCount()
        self._result_table.insertRow(row)
        self._result_table.setItem(row, 0, QTableWidgetItem(motif_id))
        self._result_table.setItem(row, 1, QTableWidgetItem(str(count)))
        self._result_table.setItem(row, 2, QTableWidgetItem(pct))
        subject_map = self._subject_by_session()
        display_names = [subject_map.get(sid) or sid for sid in sessions]
        session_str = ", ".join(display_names[:3]) + ("…" if len(display_names) > 3 else "")
        self._result_table.setItem(row, 3, QTableWidgetItem(session_str))

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    def _update_readout(
        self,
        n_windows_total: int,
        n_windows_clustered: int,
        n_windows_assigned: int,
        noise_count: int,
        n_motifs: int,
        seed_filtered: bool,
        behavior_id: str | None,
    ) -> None:
        """Populate visual summary so users can interpret discovery quality quickly."""
        if n_windows_total <= 0:
            self._readout_summary.setText(
                "Run motif discovery to see how much of the recording was covered and what this means."
            )
            self._bar_cluster_coverage.setValue(0)
            self._bar_assigned_coverage.setValue(0)
            self._bar_noise_rate.setValue(0)
            self._bar_cluster_coverage.setFormat("Clustered windows: —")
            self._bar_assigned_coverage.setFormat("Assigned to motifs: —")
            self._bar_noise_rate.setFormat("Noise rate: —")
            self._readout_seed_note.setText("")
            return

        clustered_pct = int(round(100.0 * n_windows_clustered / max(n_windows_total, 1)))
        assigned_pct = int(round(100.0 * n_windows_assigned / max(n_windows_total, 1)))
        noise_pct = int(round(100.0 * noise_count / max(n_windows_clustered, 1)))

        self._bar_cluster_coverage.setValue(clustered_pct)
        self._bar_assigned_coverage.setValue(assigned_pct)
        self._bar_noise_rate.setValue(noise_pct)

        self._bar_cluster_coverage.setFormat(
            f"Clustered windows: {n_windows_clustered}/{n_windows_total} ({clustered_pct}%)"
        )
        self._bar_assigned_coverage.setFormat(
            f"Assigned to motifs: {n_windows_assigned}/{n_windows_total} ({assigned_pct}%)"
        )
        self._bar_noise_rate.setFormat(
            f"Noise rate: {noise_count}/{max(n_windows_clustered, 1)} ({noise_pct}%)"
        )

        self._readout_summary.setText(
            "Motifs summarize recurring movement patterns. "
            f"This run found {n_motifs} motif(s), and {n_windows_assigned} windows were confidently "
            "mapped to those motifs."
        )

        if seed_filtered:
            if behavior_id:
                self._readout_seed_note.setText(
                    "Seed filter was ON: clustering focused on windows that overlap positive seed examples "
                    f"for behaviour '{behavior_id}'."
                )
            else:
                self._readout_seed_note.setText(
                    "Seed filter was ON: clustering focused on windows that overlap positive seed examples."
                )
        else:
            self._readout_seed_note.setText(
                "Seed filter was OFF: motifs reflect all selected recording windows, not one behaviour." 
            )

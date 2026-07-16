"""Candidate Generation tab - rank motif windows for clip extraction."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
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
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.candidate_service import (
    CandidateGenerationConfig,
    CandidateGenerationResult,
    CandidateGenerationService,
)
from abel.services.motif_service import MotifDiscoveryService
from abel.services.pose_features_service import PoseFeaturesService
from abel.services.seed_service import SeedService
from abel.workers.task_worker import TaskWorker
from abel.utils.error_text import format_task_error

logger = logging.getLogger("abel")


class CandidateGenerationTab(QWidget):
    """Ranks motif-assigned windows and stores selected candidates."""

    progress_update_requested = Signal(int, int, str)

    def __init__(
        self,
        candidate_service: CandidateGenerationService,
        motif_service: MotifDiscoveryService,
        pose_features_service: PoseFeaturesService,
        seed_service: SeedService,
        behavior_service: BehaviorService,
        import_service: ImportService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = candidate_service
        self._motif_service = motif_service
        self._pose_features = pose_features_service
        self._seed_service = seed_service
        self._behavior_service = behavior_service
        self._import_service = import_service or ImportService()
        self._project_root: Path | None = None
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self.progress_update_requested.connect(self._on_progress_update)

        self._no_project = QLabel(
            "Open a project and run Motif Discovery before generating candidate windows."
        )
        self._no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_project.setWordWrap(True)
        self._no_project.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")

        self._session_table = QTableWidget(0, 3)
        self._session_table.setHorizontalHeaderLabels(["", "Subject", "Feature windows"])
        self._session_table.setColumnWidth(0, 30)
        self._session_table.setColumnWidth(1, 180)
        self._session_table.setColumnWidth(2, 130)
        self._session_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._session_table.verticalHeader().setVisible(False)

        sel_all_btn = QPushButton("Select All")
        sel_none_btn = QPushButton("Select None")
        refresh_btn = QPushButton("Refresh")
        sel_all_btn.clicked.connect(self._select_all)
        sel_none_btn.clicked.connect(self._select_none)
        refresh_btn.clicked.connect(self._refresh_clicked)

        sess_row = QHBoxLayout()
        sess_row.addWidget(QLabel("Subjects:"))
        sess_row.addStretch()
        sess_row.addWidget(refresh_btn)
        sess_row.addWidget(sel_all_btn)
        sess_row.addWidget(sel_none_btn)

        session_box = QGroupBox("Subject Selection")
        session_layout = QVBoxLayout(session_box)
        session_layout.addLayout(sess_row)
        session_layout.addWidget(self._session_table)

        self._behavior_combo = QComboBox()
        self._seed_weight_chk = QCheckBox("Prioritize using positive seed overlap")
        self._seed_weight_chk.setChecked(False)
        self._include_noise_chk = QCheckBox("Include motif noise windows")
        self._include_noise_chk.setChecked(False)

        self._top_k = QSpinBox()
        self._top_k.setRange(1, 5000)
        self._top_k.setValue(300)

        self._min_score = QDoubleSpinBox()
        self._min_score.setRange(0.0, 1.0)
        self._min_score.setSingleStep(0.05)
        self._min_score.setValue(0.0)

        self._max_overlap = QDoubleSpinBox()
        self._max_overlap.setRange(0.0, 1.0)
        self._max_overlap.setSingleStep(0.05)
        self._max_overlap.setValue(0.6)

        self._motif_weight = QDoubleSpinBox()
        self._motif_weight.setRange(0.0, 1.0)
        self._motif_weight.setSingleStep(0.05)
        self._motif_weight.setValue(0.7)

        self._seed_weight = QDoubleSpinBox()
        self._seed_weight.setRange(0.0, 1.0)
        self._seed_weight.setSingleStep(0.05)
        self._seed_weight.setValue(0.3)

        param_box = QGroupBox("Scoring Parameters")
        form = QFormLayout(param_box)
        form.addRow("Behavior target:", self._behavior_combo)
        form.addRow("", self._seed_weight_chk)
        form.addRow("", self._include_noise_chk)
        form.addRow("Top K windows:", self._top_k)
        form.addRow("Minimum score:", self._min_score)
        form.addRow("Max overlap ratio:", self._max_overlap)
        form.addRow("Motif weight:", self._motif_weight)
        form.addRow("Seed weight:", self._seed_weight)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(session_box)
        left_layout.addWidget(param_box)
        left_layout.addStretch()

        info = QLabel(
            "Ranked candidates are saved to derived/review_tables/candidate_windows.json and "
            "used by Clip Extraction."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "background: #0D2B3E; color: #4FC3F7; border: 1px solid #0288D1; "
            "border-radius: 4px; padding: 8px; font-size: 11px; font-weight: 600;"
        )

        self._status = QLabel("No candidate list generated yet.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #78909C; font-size: 11px; padding: 2px 0;")

        self._run_btn = QPushButton("Generate Candidates")
        self._clear_btn = QPushButton("Clear Existing")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._clear_btn.clicked.connect(self._clear_existing)
        self._cancel_btn.clicked.connect(self._cancel)

        run_row = QHBoxLayout()
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._clear_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch()

        self._progress = QProgressBar()
        self._progress.setMaximum(4)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)

        self._result_table = QTableWidget(0, 6)
        self._result_table.setHorizontalHeaderLabels(
            ["Subject", "Behavior", "#", "Start", "End", "Score"]
        )
        self._result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.horizontalHeader().setStretchLastSection(False)
        self._result_table.setColumnWidth(0, 130)
        self._result_table.setColumnWidth(1, 150)
        self._result_table.setColumnWidth(2, 36)
        self._result_table.setColumnWidth(3, 68)
        self._result_table.setColumnWidth(4, 68)
        self._result_table.setColumnWidth(5, 64)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(info)
        right_layout.addWidget(self._status)
        right_layout.addLayout(run_row)
        right_layout.addWidget(self._progress)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self._log)
        right_layout.addWidget(QLabel("Top candidates:"))
        right_layout.addWidget(self._result_table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([430, 470])
        self._splitter = splitter

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._no_project)
        root.addWidget(splitter)
        splitter.hide()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._service.set_project(project_root)
        self._motif_service.set_project(project_root)
        self._pose_features.set_project(project_root)
        self._seed_service.set_project(project_root)
        self._behavior_service.set_project(project_root)

        self._no_project.hide()
        self._splitter.show()
        self._refresh_behavior_combo()
        self._refresh_sessions()
        self._refresh_status()

    def _refresh_behavior_combo(self) -> None:
        self._behavior_combo.clear()
        self._behavior_combo.addItem("(all behaviors)", userData=None)
        for b in self._behavior_service.behaviors:
            self._behavior_combo.addItem(b.name, userData=b.behavior_id)

    def _refresh_sessions(self) -> None:
        self._session_table.setRowCount(0)
        summaries = {s.session_id: s for s in self._pose_features.load_all_summaries()}
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

    def _refresh_status(self) -> None:
        rows = self._service.load_candidates()
        if not rows:
            self._status.setText("No candidate list generated yet.")
            self._status.setStyleSheet("color: #78909C; font-size: 11px; padding: 2px 0;")
            self._result_table.setRowCount(0)
            return

        self._status.setText(f"Last run: {len(rows)} candidate windows available.")
        self._status.setStyleSheet("color: #4FC3F7; font-size: 11px; padding: 2px 0;")
        self._populate_result_table(rows)

    def _refresh_clicked(self) -> None:
        self._refresh_behavior_combo()
        self._refresh_sessions()
        self._refresh_status()
        self._append_log("Refreshed sessions, behaviors, and existing candidate list.")

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

    def _selected_sessions(self) -> list[str]:
        ids: list[str] = []
        for row in range(self._session_table.rowCount()):
            item = self._session_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _run(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if not self._motif_service.has_results():
            QMessageBox.warning(
                self,
                "No Motif Model",
                "Run Motif Discovery first. Candidate Generation requires motif assignments.",
            )
            return

        session_ids = self._selected_sessions()
        if not session_ids:
            QMessageBox.warning(self, "No Sessions", "Select at least one session.")
            return
        session_id_set = set(session_ids)

        behavior_id = self._behavior_combo.currentData()
        use_seed_weight = self._seed_weight_chk.isChecked()
        behavior_targets: list[tuple[str, str]] = []

        if behavior_id is None:
            behavior_targets = [
                (str(b.behavior_id), str(b.name))
                for b in self._behavior_service.behaviors
                if (b.behavior_id or "").strip()
            ]
            if not behavior_targets:
                QMessageBox.warning(self, "No Behaviors", "Define at least one behavior first.")
                return

        if use_seed_weight:
            if behavior_id:
                seeds = self._seed_service.seeds_for_behavior(behavior_id)
            else:
                seeds = self._seed_service.seeds
            seeds = [
                s for s in seeds
                if s.label_type == "positive" and s.session_id in session_id_set
            ]
        else:
            seeds = None

        cfg = CandidateGenerationConfig(
            session_ids=session_ids,
            behavior_id=behavior_id,
            top_k=self._top_k.value(),
            include_noise=self._include_noise_chk.isChecked(),
            min_total_score=self._min_score.value(),
            max_overlap_ratio=self._max_overlap.value(),
            motif_weight=self._motif_weight.value(),
            seed_weight=self._seed_weight.value() if use_seed_weight else 0.0,
        )

        self._cancel_flag[0] = False
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._progress.setFormat("Starting...")
        self._result_table.setRowCount(0)
        self._append_log(
            f"Generating candidates for {len(session_ids)} session(s), top_k={cfg.top_k}."
        )
        if behavior_targets:
            self._append_log(
                f"All behaviors selected: running {len(behavior_targets)} separate candidate analyses in sequence."
            )

        worker = TaskWorker(self._run_task, cfg, seeds, behavior_targets)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_error)
        self._pool.start(worker)

    def _run_task(
        self,
        config: CandidateGenerationConfig,
        seeds: list | None,
        behavior_targets: list[tuple[str, str]] | None = None,
    ) -> CandidateGenerationResult:
        behavior_targets = behavior_targets or []

        if behavior_targets:
            aggregate = CandidateGenerationResult(
                session_ids=list(config.session_ids),
                behavior_id=None,
            )
            total_steps = max(1, len(behavior_targets) * 5)

            for idx, (target_behavior_id, target_name) in enumerate(behavior_targets):
                if self._cancel_flag[0]:
                    break

                if config.seed_weight > 0.0:
                    target_seeds = [
                        s
                        for s in self._seed_service.seeds_for_behavior(target_behavior_id)
                        if s.label_type == "positive"
                        and s.session_id in set(config.session_ids)
                    ]
                else:
                    target_seeds = None

                target_cfg = CandidateGenerationConfig(
                    session_ids=list(config.session_ids),
                    behavior_id=target_behavior_id,
                    top_k=config.top_k,
                    include_noise=config.include_noise,
                    min_total_score=config.min_total_score,
                    max_overlap_ratio=config.max_overlap_ratio,
                    motif_weight=config.motif_weight,
                    seed_weight=config.seed_weight,
                )

                def _seq_prog(step: int, _total: int, *, _idx: int = idx, _name: str = target_name) -> None:
                    labels = [
                        "Loading assignments...",
                        "Filtering windows...",
                        "Building motif priors...",
                        "Scoring and ranking...",
                        "Done",
                    ]
                    fmt = labels[min(step, len(labels) - 1)]
                    global_step = min(total_steps, (_idx * 5) + step + 1)
                    self.progress_update_requested.emit(
                        global_step,
                        total_steps,
                        f"[{_idx + 1}/{len(behavior_targets)}] {_name}: {fmt}",
                    )

                run = self._service.generate_candidates(
                    config=target_cfg,
                    seeds=target_seeds,
                    progress_callback=_seq_prog,
                    cancel_flag=self._cancel_flag,
                )

                aggregate.n_assignments_loaded += run.n_assignments_loaded
                aggregate.n_assignments_considered += run.n_assignments_considered
                aggregate.n_candidates_scored += run.n_candidates_scored
                aggregate.n_candidates_selected += run.n_candidates_selected
                aggregate.candidates.extend(run.candidates)
                aggregate.warnings.extend(run.warnings)

            aggregate.candidates.sort(
                key=lambda c: (c.total_score, c.seed_similarity_score, c.motif_score),
                reverse=True,
            )
            aggregate.n_candidates_selected = len(aggregate.candidates)
            aggregate.success = aggregate.n_candidates_selected > 0
            return aggregate

        def _single_prog(step: int, total: int) -> None:
            labels = [
                "Loading assignments...",
                "Filtering windows...",
                "Building motif priors...",
                "Scoring and ranking...",
                "Done",
            ]
            fmt = labels[min(step, len(labels) - 1)]
            self.progress_update_requested.emit(step, total, fmt)

        return self._service.generate_candidates(
            config=config,
            seeds=seeds,
            progress_callback=_single_prog,
            cancel_flag=self._cancel_flag,
        )

    def _on_finished(self, result: CandidateGenerationResult) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

        for warning in result.warnings:
            self._append_log(f"Warning: {warning}")

        if not result.success:
            self._progress.setFormat("Failed - see log")
            self._append_log("Candidate generation failed.")
            return

        self._service.save_candidates(result)
        self._progress.setMaximum(4)
        self._progress.setValue(4)
        self._progress.setFormat(f"Done - selected {result.n_candidates_selected} candidates")
        self._populate_result_table(result.candidates)

        self._status.setText(
            f"Last run: selected {result.n_candidates_selected} of "
            f"{result.n_candidates_scored} scored windows."
        )
        self._status.setStyleSheet("color: #4FC3F7; font-size: 11px; padding: 2px 0;")
        self._append_log(
            f"Saved {result.n_candidates_selected} candidates "
            f"({result.n_assignments_considered} motif windows considered)."
        )

    def _on_error(self, traceback_text: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setFormat("Error")
        self._append_log("Candidate generation failed:")
        self._append_log(format_task_error(traceback_text))
        logger.error("Candidate generation error:\n%s", traceback_text)

    @Slot(int, int, str)
    def _on_progress_update(self, step: int, total: int, fmt: str) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._progress.setFormat(fmt)

    def _cancel(self) -> None:
        self._cancel_flag[0] = True
        self._append_log("Cancellation requested...")

    def _clear_existing(self) -> None:
        if not self._project_root:
            return
        answer = QMessageBox.question(
            self,
            "Clear Existing Candidates",
            "Delete the saved candidate list for this project?\n\n"
            "This removes derived/review_tables/candidate_windows.json.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed = self._service.clear_candidates()
        self._refresh_status()
        if removed:
            self._append_log("Cleared existing candidates.")
        else:
            self._append_log("No saved candidates were found to clear.")

    def _subject_by_session(self) -> dict[str, str]:
        """Resolve session_id → subject label using the import manifest."""
        if not self._project_root:
            return {}
        manifest = self._import_service.load_manifest(self._project_root)
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

    def _populate_result_table(self, candidates: list) -> None:
        """Fill the result table with subject, behavior name, per-session occurrence#, frames, score."""
        self._result_table.setRowCount(0)
        if not candidates:
            return

        subject_map = self._subject_by_session()
        behavior_name_map = {b.behavior_id: b.name for b in self._behavior_service.behaviors}

        # Sort by behavior → subject → start_frame so occurrence # is in temporal order.
        def _sort_key(c):
            bname = (behavior_name_map.get(c.behavior_id or "", c.behavior_id or "") or "").lower()
            subj = (subject_map.get(c.session_id, c.session_id) or "").lower()
            return (bname, subj, int(c.start_frame))

        sorted_cands = sorted(candidates, key=_sort_key)

        # Occurrence counter keyed by (session_id, behavior_id).
        occurrence: dict[tuple[str, str], int] = {}

        for cand in sorted_cands:
            bid = (cand.behavior_id or "").strip()
            key = (cand.session_id, bid)
            occurrence[key] = occurrence.get(key, 0) + 1
            occ_num = occurrence[key]

            subject = subject_map.get(cand.session_id, cand.session_id) or cand.session_id or ""
            bname = behavior_name_map.get(bid, bid) if bid else "—"

            row = self._result_table.rowCount()
            self._result_table.insertRow(row)
            self._result_table.setItem(row, 0, QTableWidgetItem(subject))
            self._result_table.setItem(row, 1, QTableWidgetItem(bname))
            occ_item = QTableWidgetItem(str(occ_num))
            occ_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._result_table.setItem(row, 2, occ_item)
            self._result_table.setItem(row, 3, QTableWidgetItem(str(cand.start_frame)))
            self._result_table.setItem(row, 4, QTableWidgetItem(str(cand.end_frame)))
            score_item = QTableWidgetItem(f"{cand.total_score:.3f}")
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._result_table.setItem(row, 5, score_item)

    def _append_log(self, message: str) -> None:
        self._log.append(message)

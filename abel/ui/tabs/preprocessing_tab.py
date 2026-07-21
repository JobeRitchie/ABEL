"""Clip extraction tab — decode clips only for selected candidates."""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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

from abel.models.schemas import CandidateWindow, ClipManifest, PreprocessingPreset
from abel.services.candidate_service import CandidateGenerationService
from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.preprocessing_service import ClipExtractionConfig, ClipExtractionService
from abel.storage.file_store import read_yaml, write_yaml
from abel.workers.task_worker import TaskWorker
from abel.utils.error_text import format_task_error

if TYPE_CHECKING:
    import numpy as np


class ClipExtractionTab(QWidget):
    """Extract clips for generated candidate windows."""

    _ALL_SUBJECTS_KEY = "__all_subjects__"
    _UNASSIGNED_BEHAVIOR_FILTER = "__unassigned_behavior__"

    extraction_progress_requested = Signal(int, int, str)

    def __init__(
        self,
        clip_extraction_service: ClipExtractionService,
        candidate_service: CandidateGenerationService,
        import_service: ImportService,
        behavior_service: BehaviorService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = clip_extraction_service
        self._candidate_service = candidate_service
        self._imports = import_service
        self._behaviors = behavior_service
        self._pose_processing = PoseProcessingService()
        self._project_root: Path | None = None
        self._loading_ui_settings: bool = False
        self._pool = QThreadPool.globalInstance()
        self._cancel_flag: list[bool] = [False]
        self._candidates_by_session: dict[str, list[CandidateWindow]] = {}
        self._session_order: list[str] = []
        self._extraction_started_at: float | None = None
        self._waiting_for_worker_finish = False
        self._centroid_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._external_candidates: list[CandidateWindow] = []
        self._external_source_label: str = ""
        self.extraction_progress_requested.connect(self._on_extraction_progress)

        self._session_combo = QComboBox()
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._behavior_combo = QComboBox()
        self._behavior_combo.currentIndexChanged.connect(self._on_session_changed)
        self._preset_combo = QComboBox()

        self._top_n = QSpinBox()
        self._top_n.setRange(0, 2000)
        self._top_n.setValue(5)

        self._bottom_n = QSpinBox()
        self._bottom_n.setRange(0, 2000)
        self._bottom_n.setValue(0)

        self._median_n = QSpinBox()
        self._median_n.setRange(0, 2000)
        self._median_n.setValue(0)

        self._before_sec = QDoubleSpinBox()
        self._before_sec.setRange(0.0, 10.0)
        self._before_sec.setSingleStep(0.1)
        self._before_sec.setDecimals(2)
        self._before_sec.setValue(0.0)

        self._after_sec = QDoubleSpinBox()
        self._after_sec.setRange(0.0, 10.0)
        self._after_sec.setSingleStep(0.1)
        self._after_sec.setDecimals(2)
        self._after_sec.setValue(0.0)

        self._crop_area_percent = QDoubleSpinBox()
        self._crop_area_percent.setRange(50.0, 1000.0)
        self._crop_area_percent.setSingleStep(5.0)
        self._crop_area_percent.setDecimals(0)
        self._crop_area_percent.setValue(125.0)
        self._crop_area_percent.setSuffix(" %")
        self._crop_area_percent.setToolTip(
            "Scales extracted crop area relative to preset baseline. "
            "100% = baseline, 125% = 25% larger area."
        )

        self._status = QLabel("No candidate windows loaded.")
        self._status.setWordWrap(True)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(["Subject", "Behavior", "#", "Start", "End", "Score", "Source", "Clip"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        # Let the "Source" column absorb extra width so the table fills the
        # panel instead of leaving a dead margin on the right; the rest keep
        # their natural sizes.
        _header = self._table.horizontalHeader()
        _header.setStretchLastSection(False)
        self._table.setColumnWidth(0, 130)
        self._table.setColumnWidth(1, 150)
        self._table.setColumnWidth(2, 36)
        self._table.setColumnWidth(3, 68)
        self._table.setColumnWidth(4, 68)
        self._table.setColumnWidth(5, 64)
        self._table.setColumnWidth(6, 120)
        self._table.setColumnWidth(7, 40)
        _header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)

        refresh_btn = QPushButton("Refresh Sessions")
        refresh_btn.clicked.connect(self._refresh)
        self._run_btn = QPushButton("Extract Clips")
        self._run_btn.clicked.connect(self._run)
        self._clear_btn = QPushButton("Clear Clips")
        self._clear_btn.clicked.connect(self._clear_clips)
        self._clear_btn.setToolTip(
            "Delete the rendered video clip files for the selected subject to free "
            "disk space.\n\nDoes NOT change your review labels or the candidate "
            "queue — clips can be re-extracted at any time."
        )
        self._clear_candidates_btn = QPushButton("Clear Candidates")
        self._clear_candidates_btn.clicked.connect(self._clear_candidates)
        self._clear_candidates_btn.setToolTip(
            "Remove pending (unreviewed) windows from the review queue shown here.\n\n"
            "Windows you've already reviewed are kept, your review labels are never "
            "changed, and no extracted clip files are deleted."
        )
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)

        self._filter_sources_btn = QPushButton("Filter Sources…")
        self._filter_sources_btn.setToolTip(
            "Choose which candidate source types to include "
            "(e.g. uncertainty, hard_negative, confound_boundary, exploration)."
        )
        self._filter_sources_btn.clicked.connect(self._open_source_filter_dialog)
        # All source types enabled by default; populated on first refresh.
        self._source_filter_enabled: dict[str, bool] = {}

        # Hide-reviewed filter: when on, candidates that already carry a
        # reviewer label are dropped from the table so the queue only shows
        # windows still needing review.  The reviewed-ID set is cached per
        # refresh (it reads reviewer_labels.parquet) and reused across combo
        # changes, which fire _on_session_changed frequently.
        self._hide_reviewed_chk = QCheckBox("Hide reviewed clips")
        self._hide_reviewed_chk.setChecked(False)
        self._hide_reviewed_chk.setToolTip(
            "Hide candidates you've already reviewed (any accept / reject / "
            "relabel decision). Your review labels and clip files are untouched."
        )
        self._hide_reviewed_chk.toggled.connect(self._on_session_changed)
        self._reviewed_ids: set[str] = set()

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._refresh_btn = refresh_btn

        params = QGroupBox("Extraction Settings")
        form = QFormLayout(params)
        form.addRow("Subject:", self._session_combo)
        form.addRow("Behavior target:", self._behavior_combo)
        form.addRow("Preset:", self._preset_combo)
        form.addRow("Top candidates:", self._top_n)
        form.addRow("Bottom candidates:", self._bottom_n)
        form.addRow("Median candidates:", self._median_n)
        form.addRow("Crop area:", self._crop_area_percent)
        form.addRow("Before (sec):", self._before_sec)
        form.addRow("After (sec):", self._after_sec)

        row = QHBoxLayout()
        row.addWidget(refresh_btn)
        row.addWidget(self._run_btn)
        row.addWidget(self._filter_sources_btn)
        row.addWidget(self._hide_reviewed_chk)
        row.addWidget(self._clear_btn)
        row.addWidget(self._clear_candidates_btn)
        row.addWidget(self._cancel_btn)
        row.addStretch()

        # Candidate table (top) and run log (bottom) share a vertical splitter
        # so the user can rebalance them; the table gets the bulk of the space.
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        log_layout.addWidget(QLabel("Log:"))
        log_layout.addWidget(self._log, 1)

        body_split = QSplitter(Qt.Orientation.Vertical)
        body_split.setChildrenCollapsible(True)
        body_split.setHandleWidth(8)
        body_split.addWidget(self._table)
        body_split.addWidget(log_widget)
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 0)
        body_split.setSizes([520, 160])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(params)
        layout.addLayout(row)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addWidget(body_split, 1)

        self._bind_project_setting_persistence()

    def set_project(self, project_root: Path) -> None:
        self._reset_runtime_state_for_project_switch()
        self._project_root = project_root
        # Defer I/O to avoid blocking the tab switch.
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        self._service.set_project(project_root)
        self._candidate_service.set_project(project_root)
        self._behaviors.set_project(project_root)
        self._refresh()
        self._load_ui_settings_from_project()

    def _reset_runtime_state_for_project_switch(self) -> None:
        self._cancel_flag[0] = False
        self._waiting_for_worker_finish = False
        self._external_candidates = []
        self._external_source_label = ""
        self._set_busy_state(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._table.setRowCount(0)
        self._log.clear()

    def load_external_candidates(self, candidates: list[CandidateWindow], source_label: str = "External selection", clear_source: str | None = None) -> None:
        """Inject ad-hoc candidates from another tab into this extraction view.

        If *clear_source* is given, any previously loaded candidates whose
        ``source`` field equals that value are removed first (both from
        in-memory and from the persisted JSON).  This lets callers do a clean
        replace rather than an accumulating upsert.
        """
        if clear_source:
            self._external_candidates = [
                c for c in self._external_candidates
                if (c.source or "") != clear_source
            ]
            try:
                self._candidate_service.remove_external_candidates_by_source(clear_source)
            except Exception:
                pass
        by_id: dict[str, CandidateWindow] = {str(c.window_id): c for c in self._external_candidates}
        for cand in candidates:
            by_id[str(cand.window_id)] = cand
        self._external_candidates = list(by_id.values())
        self._external_source_label = str(source_label or "External selection").strip()
        persisted_added = 0
        try:
            persisted_added = int(self._candidate_service.upsert_external_window_candidates(candidates))
        except Exception:
            persisted_added = 0
        self._append_log(
            f"Added {len(candidates)} external candidate(s) from {self._external_source_label}. "
            f"External queue now has {len(self._external_candidates)} item(s); "
            f"persisted new={persisted_added}."
        )
        self._refresh()

    def _combined_candidates(self) -> list[CandidateWindow]:
        rows = list(self._candidate_service.load_candidates())
        if not self._external_candidates:
            return rows
        by_id: dict[str, CandidateWindow] = {str(c.window_id): c for c in rows}
        for cand in self._external_candidates:
            by_id[str(cand.window_id)] = cand
        return list(by_id.values())

    def _reset_ui_settings_to_defaults(self) -> None:
        self._top_n.setValue(5)
        self._bottom_n.setValue(0)
        self._median_n.setValue(0)
        self._crop_area_percent.setValue(125.0)
        self._before_sec.setValue(0.0)
        self._after_sec.setValue(0.0)
        self._hide_reviewed_chk.setChecked(False)

    def _bind_project_setting_persistence(self) -> None:
        self._session_combo.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._behavior_combo.currentIndexChanged.connect(lambda _i: self._persist_ui_settings_to_project())
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._top_n.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._bottom_n.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._median_n.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._crop_area_percent.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._before_sec.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._after_sec.valueChanged.connect(lambda _v: self._persist_ui_settings_to_project())
        self._hide_reviewed_chk.toggled.connect(lambda _v: self._persist_ui_settings_to_project())

    def _on_preset_changed(self, _index: int) -> None:
        preset = self._preset_combo.currentData()
        if preset is not None and not self._loading_ui_settings:
            area_scale = float(getattr(preset, "crop_area_scale", 1.25) or 1.25)
            self._crop_area_percent.setValue(max(50.0, min(1000.0, area_scale * 100.0)))
        self._persist_ui_settings_to_project()

    def _ui_settings_payload(self) -> dict[str, Any]:
        preset = self._preset_combo.currentData()
        preset_id = getattr(preset, "preset_id", "") if preset is not None else ""
        return {
            "session": str(self._session_combo.currentData() or ""),
            "behavior": str(self._behavior_combo.currentData() or ""),
            "preset_id": str(preset_id or ""),
            "top_candidates": int(self._top_n.value()),
            "bottom_candidates": int(self._bottom_n.value()),
            "median_candidates": int(self._median_n.value()),
            "crop_area_percent": float(self._crop_area_percent.value()),
            "before_sec": float(self._before_sec.value()),
            "after_sec": float(self._after_sec.value()),
            "hide_reviewed": bool(self._hide_reviewed_chk.isChecked()),
        }

    def _persist_ui_settings_to_project(self) -> None:
        if self._project_root is None or self._loading_ui_settings:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        raw["clip_extraction_ui"] = self._ui_settings_payload()
        write_yaml(path, raw)

    def _load_ui_settings_from_project(self) -> None:
        if self._project_root is None:
            return
        path = self._project_root / "project.yaml"
        raw = read_yaml(path, {})
        ui = dict(raw.get("clip_extraction_ui") or {})

        self._loading_ui_settings = True
        try:
            self._reset_ui_settings_to_defaults()
            session = str(ui.get("session") or "").strip()
            if session:
                idx = self._session_combo.findData(session)
                if idx >= 0:
                    self._session_combo.setCurrentIndex(idx)

            behavior = str(ui.get("behavior") or "").strip()
            idx_behavior = self._behavior_combo.findData(behavior if behavior else None)
            if idx_behavior >= 0:
                self._behavior_combo.setCurrentIndex(idx_behavior)

            preset_id = str(ui.get("preset_id") or "").strip()
            if preset_id:
                for i in range(self._preset_combo.count()):
                    data = self._preset_combo.itemData(i)
                    if getattr(data, "preset_id", "") == preset_id:
                        self._preset_combo.setCurrentIndex(i)
                        break

            self._top_n.setValue(int(ui.get("top_candidates", 5)))
            self._bottom_n.setValue(int(ui.get("bottom_candidates", 0)))
            self._median_n.setValue(int(ui.get("median_candidates", 0)))
            self._crop_area_percent.setValue(float(ui.get("crop_area_percent", 125.0)))
            self._before_sec.setValue(float(ui.get("before_sec", 0.0)))
            self._after_sec.setValue(float(ui.get("after_sec", 0.0)))
            self._hide_reviewed_chk.setChecked(bool(ui.get("hide_reviewed", False)))
        finally:
            self._loading_ui_settings = False

    def showEvent(self, event) -> None:
        """Refresh when the tab becomes visible to pick up newly generated candidates."""
        super().showEvent(event)
        if self._project_root:
            self._refresh()

    def _refresh(self) -> None:
        self._set_busy_state(False)

        # Remember selections so they can be restored after the combo rebuild.
        prev_session = self._session_combo.currentData()
        prev_behavior = self._behavior_combo.currentData()
        prev_preset = self._preset_combo.currentData()
        prev_preset_id = getattr(prev_preset, "preset_id", "") if prev_preset is not None else ""

        # Block signals during rebuild to avoid spurious _on_session_changed calls
        # (which would clear the table mid-rebuild, making clips appear to vanish)
        # and spurious _on_preset_changed calls (which would reset the crop area
        # to the preset baseline and stomp the user's chosen value).
        self._session_combo.blockSignals(True)
        self._behavior_combo.blockSignals(True)
        self._preset_combo.blockSignals(True)

        self._session_combo.clear()
        self._behavior_combo.clear()
        self._preset_combo.clear()

        self._behavior_combo.addItem("(all behaviors)", userData=None)
        self._behavior_combo.addItem("(unassigned / random windows)", userData=self._UNASSIGNED_BEHAVIOR_FILTER)
        for behavior in self._behaviors.behaviors:
            self._behavior_combo.addItem(behavior.name, userData=behavior.behavior_id)

        for p in self._service.load_project_presets():
            self._preset_combo.addItem(p.name, userData=p)

        # Restore previously selected preset by id (the objects are freshly loaded,
        # so identity-based findData would not match).
        if prev_preset_id:
            for i in range(self._preset_combo.count()):
                data = self._preset_combo.itemData(i)
                if getattr(data, "preset_id", "") == prev_preset_id:
                    self._preset_combo.setCurrentIndex(i)
                    break

        rows = self._combined_candidates()
        # Cache the reviewed-segment set once per refresh so the "Hide reviewed
        # clips" filter doesn't re-read reviewer_labels.parquet on every combo
        # change (_on_session_changed fires often).
        try:
            self._reviewed_ids = self._candidate_service.reviewed_segment_ids()
        except Exception:
            self._reviewed_ids = set()
        self._candidates_by_session = {}
        for r in rows:
            self._candidates_by_session.setdefault(r.session_id, []).append(r)
        for sid in self._candidates_by_session:
            self._candidates_by_session[sid].sort(key=lambda c: c.total_score, reverse=True)

        # Build session list from import manifest first, then include any candidate-only sessions.
        session_ids: list[str] = []
        if self._project_root:
            manifest = self._imports.load_manifest(self._project_root)
            if manifest:
                session_ids.extend([s.session_id for s in manifest.linked_sessions])
        for sid in self._candidates_by_session:
            if sid not in session_ids:
                session_ids.append(sid)

        self._session_order = sorted(session_ids)
        subject_map = self._subject_by_session()
        total_candidates = sum(len(self._candidates_by_session.get(sid, [])) for sid in self._session_order)
        self._session_combo.addItem(
            f"All subjects ({total_candidates} candidate{'s' if total_candidates != 1 else ''})",
            userData=self._ALL_SUBJECTS_KEY,
        )
        for sid in self._session_order:
            n = len(self._candidates_by_session.get(sid, []))
            display = subject_map.get(sid) or sid
            self._session_combo.addItem(f"{display} ({n} candidate{'s' if n != 1 else ''})", userData=sid)

        # Restore previous selections (or default to first item).
        session_idx = self._session_combo.findData(prev_session) if prev_session is not None else -1
        self._session_combo.setCurrentIndex(max(0, session_idx))
        behavior_idx = self._behavior_combo.findData(prev_behavior) if prev_behavior is not None else 0
        self._behavior_combo.setCurrentIndex(max(0, behavior_idx))

        # If the restored behavior filter would hide ALL candidates (e.g. the
        # user was viewing "Dig" but a retrain for "Groom" replaced the queue),
        # fall back to "(all behaviors)" so the new candidates are visible.
        restored_bid = self._behavior_combo.currentData()
        if restored_bid and restored_bid != self._UNASSIGNED_BEHAVIOR_FILTER and rows:
            visible = any(c.behavior_id == restored_bid for c in rows)
            if not visible:
                self._behavior_combo.setCurrentIndex(0)  # "(all behaviors)"

        self._session_combo.blockSignals(False)
        self._behavior_combo.blockSignals(False)
        self._preset_combo.blockSignals(False)

        if not self._session_order:
            self._status.setText("No subjects found. Run Data Import first.")
            self._table.setRowCount(0)
            return

        sessions_with_candidates = sum(1 for sid in self._session_order if self._candidates_by_session.get(sid))
        status = f"Loaded {len(rows)} candidates across {sessions_with_candidates}/{len(self._session_order)} sessions."
        if self._external_candidates:
            status += f" Includes {len(self._external_candidates)} external candidate(s) from {self._external_source_label}."
        self._status.setText(status)
        self._sync_source_filter_button()
        self._on_session_changed(self._session_combo.currentIndex())

    def _on_session_changed(self, _idx: int) -> None:
        session_id = self._session_combo.currentData()
        if not session_id:
            self._table.setRowCount(0)
            return
        if session_id == self._ALL_SUBJECTS_KEY:
            rows: list[CandidateWindow] = []
            for sid in self._session_order:
                rows.extend(self._candidates_by_session.get(sid, []))
            rows.sort(key=lambda c: c.total_score, reverse=True)
        else:
            rows = list(self._candidates_by_session.get(session_id, []))
        behavior_id = self._behavior_combo.currentData()
        if behavior_id:
            if behavior_id == self._UNASSIGNED_BEHAVIOR_FILTER:
                rows = [c for c in rows if not str(c.behavior_id or "").strip() or str(c.behavior_id).strip() == "target_behavior"]
            else:
                rows = [c for c in rows if c.behavior_id == behavior_id]
        # Apply source type filter.
        rows = self._apply_source_filter(rows)
        # Hide already-reviewed candidates when the filter is on.
        if self._hide_reviewed_chk.isChecked() and self._reviewed_ids:
            rows = [c for c in rows if str(c.window_id).strip() not in self._reviewed_ids]
        self._populate_table(rows)

    def _apply_source_filter(self, rows: list[CandidateWindow]) -> list[CandidateWindow]:
        """Remove candidates whose source type is unchecked in the filter."""
        if not self._source_filter_enabled:
            return rows  # No filter configured yet → show all.
        # If ALL are enabled, skip filtering.
        if all(self._source_filter_enabled.values()):
            return rows
        return [
            c for c in rows
            if self._source_filter_enabled.get(
                str(c.selection_reason or c.source or "").strip() or "unknown", True
            )
        ]

    def _discover_source_types(self) -> list[str]:
        """Collect unique source types across all loaded candidates."""
        types: set[str] = set()
        for sessions in self._candidates_by_session.values():
            for c in sessions:
                label = str(c.selection_reason or c.source or "").strip()
                if label:
                    types.add(label)
        return sorted(types) or ["unknown"]

    def _open_source_filter_dialog(self) -> None:
        """Show a dialog where the user can check/uncheck candidate source types."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox  # noqa: PLC0415

        # Prune stale keys so the dict (and the button count) matches the
        # source types actually present now — otherwise the button could read
        # "5 sources" while the dialog shows only 4 checkboxes.
        self._sync_source_filter_button()
        source_types = self._discover_source_types()

        # Friendly display names for source types.
        _display = {
            "uncertainty": "Uncertainty (model is unsure)",
            "disagreement": "Disagreement (models conflict)",
            "hard_negative": "Hard negative (false positives)",
            "confound_boundary": "Confound boundary (similar behaviors)",
            "exploration": "Exploration (random sampling)",
            "diversity": "Diversity (under-represented regions)",
            "candidate": "Candidate (strong predictions)",
            "baseline": "Baseline",
            "umap_selection": "UMAP interactive selection",
            "umap_interactive_selection": "UMAP interactive selection",
            "active_learning_uncertainty": "Active learning uncertainty",
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Filter Candidate Sources")
        dlg.setMinimumWidth(340)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Include candidates from these sources:"))

        checkboxes: dict[str, QCheckBox] = {}
        for st in source_types:
            cb = QCheckBox(_display.get(st, st))
            cb.setChecked(self._source_filter_enabled.get(st, True))
            layout.addWidget(cb)
            checkboxes[st] = cb

        # Select-all / Deselect-all row.
        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        desel_all_btn = QPushButton("Deselect All")
        sel_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes.values()])
        desel_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes.values()])
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(desel_all_btn)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        for st, cb in checkboxes.items():
            self._source_filter_enabled[st] = cb.isChecked()

        self._sync_source_filter_button()

        # Re-apply to the table.
        self._on_session_changed(0)

    def _sync_source_filter_button(self) -> None:
        """Prune stale source keys and update the Filter Sources button label.

        The count on the button must match the number of checkboxes the dialog
        would show — i.e. the source types actually present among the loaded
        candidates — so we drop any keys for sources that are no longer present.
        """
        source_types = self._discover_source_types()
        self._source_filter_enabled = {
            st: self._source_filter_enabled.get(st, True) for st in source_types
        }
        n_on = sum(1 for v in self._source_filter_enabled.values() if v)
        n_total = len(self._source_filter_enabled)
        if n_on < n_total:
            self._filter_sources_btn.setText(f"Filter Sources ({n_on}/{n_total})")
        else:
            self._filter_sources_btn.setText("Filter Sources…")

    def _selected_candidates_by_session(
        self,
        session_id: str,
        behavior_id: str | None,
        source_fps: float,
    ) -> dict[str, list[CandidateWindow]]:
        plan: dict[str, list[CandidateWindow]] = {}
        all_candidates = self._apply_source_filter(self._combined_candidates())
        top_n = int(self._top_n.value())
        bottom_n = int(self._bottom_n.value())
        median_n = int(self._median_n.value())
        if session_id == self._ALL_SUBJECTS_KEY:
            by_session: dict[str, list[CandidateWindow]] = {}
            for cand in all_candidates:
                sid = str(cand.session_id)
                if sid not in self._session_order:
                    continue
                if behavior_id:
                    if behavior_id == self._UNASSIGNED_BEHAVIOR_FILTER:
                        if str(cand.behavior_id or "").strip() and str(cand.behavior_id).strip() != "target_behavior":
                            continue
                    elif cand.behavior_id != behavior_id:
                        continue
                by_session.setdefault(sid, []).append(cand)

            for sid, session_rows in by_session.items():
                if not behavior_id:
                    # All-behaviors view: apply counts per behavior
                    by_behavior: dict[str, list[CandidateWindow]] = {}
                    for cand in session_rows:
                        bid = str(cand.behavior_id or "")
                        by_behavior.setdefault(bid, []).append(cand)
                    seen_ids: set[str] = set()
                    for bid, beh_rows in by_behavior.items():
                        beh_rows.sort(key=lambda c: c.total_score, reverse=True)
                        selected = self._select_top_bottom_and_median(
                            beh_rows, top_n=top_n, bottom_n=bottom_n, median_n=median_n,
                        )
                        for cand in selected:
                            wid = str(cand.window_id)
                            if wid not in seen_ids:
                                plan.setdefault(sid, []).append(
                                    self._build_extraction_window(cand, source_fps))
                                seen_ids.add(wid)
                else:
                    session_rows.sort(key=lambda c: c.total_score, reverse=True)
                    selected = self._select_top_bottom_and_median(
                        session_rows, top_n=top_n, bottom_n=bottom_n, median_n=median_n,
                    )
                    for cand in selected:
                        plan.setdefault(sid, []).append(
                            self._build_extraction_window(cand, source_fps))

            return plan
        else:
            session_ids = [session_id]

        for sid in session_ids:
            candidates = [c for c in all_candidates if c.session_id == sid]
            if behavior_id:
                if behavior_id == self._UNASSIGNED_BEHAVIOR_FILTER:
                    candidates = [
                        c for c in candidates
                        if (not str(c.behavior_id or "").strip()) or str(c.behavior_id).strip() == "target_behavior"
                    ]
                else:
                    candidates = [c for c in candidates if c.behavior_id == behavior_id]
                candidates.sort(key=lambda c: c.total_score, reverse=True)
                selected = self._select_top_bottom_and_median(
                    candidates, top_n=top_n, bottom_n=bottom_n, median_n=median_n,
                )
                if not selected:
                    continue
                plan[sid] = [self._build_extraction_window(c, source_fps) for c in selected]
            else:
                # All-behaviors view: apply counts per behavior
                by_behavior_single: dict[str, list[CandidateWindow]] = {}
                for cand in candidates:
                    bid = str(cand.behavior_id or "")
                    by_behavior_single.setdefault(bid, []).append(cand)
                session_selected: list[CandidateWindow] = []
                seen_ids_single: set[str] = set()
                for bid, beh_rows in by_behavior_single.items():
                    beh_rows.sort(key=lambda c: c.total_score, reverse=True)
                    selected = self._select_top_bottom_and_median(
                        beh_rows, top_n=top_n, bottom_n=bottom_n, median_n=median_n,
                    )
                    for cand in selected:
                        wid = str(cand.window_id)
                        if wid not in seen_ids_single:
                            session_selected.append(cand)
                            seen_ids_single.add(wid)
                if not session_selected:
                    continue
                plan[sid] = [self._build_extraction_window(c, source_fps) for c in session_selected]
        return plan

    @staticmethod
    def _select_top_bottom_and_median(
        candidates: list[CandidateWindow],
        top_n: int,
        bottom_n: int,
        median_n: int,
    ) -> list[CandidateWindow]:
        if not candidates:
            return []

        top_count = max(0, int(top_n))
        bottom_count = max(0, int(bottom_n))
        median_count = max(0, int(median_n))
        selected: list[CandidateWindow] = []
        seen_ids: set[str] = set()

        for cand in candidates[:top_count]:
            cid = str(cand.window_id)
            if cid in seen_ids:
                continue
            selected.append(cand)
            seen_ids.add(cid)

        if bottom_count > 0:
            for cand in reversed(candidates[-bottom_count:]):
                cid = str(cand.window_id)
                if cid in seen_ids:
                    continue
                selected.append(cand)
                seen_ids.add(cid)

        if median_count > 0:
            n = len(candidates)
            center = (n - 1) // 2
            offsets: list[int] = [0]
            for step in range(1, n):
                offsets.append(step)
                offsets.append(-step)

            picked = 0
            for off in offsets:
                idx = center + off
                if idx < 0 or idx >= n:
                    continue
                cand = candidates[idx]
                cid = str(cand.window_id)
                if cid in seen_ids:
                    continue
                selected.append(cand)
                seen_ids.add(cid)
                picked += 1
                if picked >= median_count:
                    break

        return selected

    def _extract_sessions_task(
        self,
        session_plan: dict[str, list[CandidateWindow]],
        preset,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> dict[str, Any]:
        assert self._project_root is not None
        manifest = self._imports.load_manifest(self._project_root)
        if not manifest:
            raise ValueError("Import manifest not found. Run Data Import first.")

        # Remap any stale session IDs (from previous imports) to current manifest IDs.
        # This happens when the manifest was rebuilt after candidates were generated.
        current_ids = {s.session_id for s in manifest.linked_sessions}
        stale = [sid for sid in session_plan if sid not in current_ids]
        if stale:
            id_remap: dict[str, str] = {}
            for old_sid in stale:
                canonical = self._imports.resolve_session_id(self._project_root, old_sid, manifest)
                if canonical != old_sid:
                    id_remap[old_sid] = canonical
            if id_remap:
                remapped: dict[str, list] = {}
                for sid, windows in session_plan.items():
                    canonical = id_remap.get(sid, sid)
                    if canonical in remapped:
                        remapped[canonical].extend(windows)
                    else:
                        remapped[canonical] = list(windows)
                session_plan = remapped

        total = sum(len(windows) for windows in session_plan.values())
        all_clips = []
        warnings: list[str] = []
        extracted_sessions: list[str] = []

        done_counter = 0
        done_lock = threading.Lock()

        def _emit_progress(delta: int) -> None:
            nonlocal done_counter
            if delta <= 0:
                return
            with done_lock:
                done_counter += int(delta)
                done_now = done_counter
            if progress_cb:
                progress_cb(done_now, total)

        def _process_session(sid: str, windows: list[CandidateWindow]) -> tuple[str, list[Any], list[str], int]:
            local_warnings: list[str] = []
            local_clips: list[Any] = []

            if cancel_flag and cancel_flag[0]:
                local_warnings.append("Cancelled by user.")
                _emit_progress(len(windows))
                return sid, local_clips, local_warnings, 0

            video_path = self._imports.video_path_for_session(manifest, sid)
            if not video_path or not video_path.exists():
                local_warnings.append(f"{sid}: missing source video, skipping {len(windows)} clip(s).")
                _emit_progress(len(windows))
                return sid, local_clips, local_warnings, 0

            pose_cx = None
            pose_cy = None
            pose_path = self._imports.pose_path_for_session(manifest, sid)
            if pose_path and pose_path.exists():
                try:
                    pose = self._pose_processing.load(pose_path)
                    pose = self._pose_processing.clean_pose(
                        pose,
                        likelihood_threshold=0.2,
                        interpolate=True,
                        smoothing_window=5,
                    )
                    pose_cx = pose.centroid_x
                    pose_cy = pose.centroid_y
                except Exception as exc:
                    local_warnings.append(f"{sid}: could not load pose centroids ({exc}); using static center crop.")
            else:
                local_warnings.append(f"{sid}: no pose file linked; using static center crop.")

            # Per-animal colored dots + legend for multi-animal review clips.
            individual_overlays = None
            if pose_path and pose_path.exists():
                _sess = next((s for s in manifest.linked_sessions if s.session_id == sid), None)
                _imap = dict(getattr(_sess, "individual_subject_map", {}) or {}) if _sess else {}
                individual_overlays = ClipExtractionService.build_individual_overlays(
                    self._pose_processing, pose_path,
                    getattr(manifest, "smoothing_settings", None), _imap,
                )

            cfg = ClipExtractionConfig(
                video_path=video_path,
                session_id=sid,
                preset=preset,
                output_dir=self._project_root / "derived" / "clips" / sid,
                pose_centroid_x=pose_cx,
                pose_centroid_y=pose_cy,
                pixels_per_mm=self._imports.pixels_per_mm_for_session(manifest, sid),
                individual_overlays=individual_overlays,
            )
            if cfg.pixels_per_mm is None:
                local_warnings.append(
                    f"{sid}: pixels/mm not set in Data Import. "
                    "Metric distance scaling will remain in pixel units until assigned."
                )

            local_done = 0

            def _local_progress(done: int, _local_total: int) -> None:
                nonlocal local_done
                delta = int(done) - int(local_done)
                if delta > 0:
                    local_done = int(done)
                    _emit_progress(delta)

            result = self._service.extract_selected_clips(
                windows,
                cfg,
                progress_callback=_local_progress,
                cancel_flag=cancel_flag,
            )

            # Ensure full accounting even when extractor exits early or does not report every step.
            if local_done < len(windows):
                _emit_progress(len(windows) - local_done)

            local_clips.extend(result.clips)
            for warning in result.warnings:
                local_warnings.append(f"{sid}: {warning}")
            return sid, local_clips, local_warnings, len(result.clips)

        items = list(session_plan.items())
        if not items:
            return {
                "session_ids": [],
                "clips": [],
                "warnings": [],
                "total_windows": total,
            }

        requested_workers = 0
        env_workers = os.environ.get("ABEL_CLIP_EXTRACT_WORKERS", "").strip()
        if env_workers:
            try:
                requested_workers = max(1, int(env_workers))
            except Exception:
                requested_workers = 0
        cpu_cap = max(1, (os.cpu_count() or 1) - 1)
        max_workers = min(len(items), requested_workers or cpu_cap)

        if max_workers <= 1:
            for sid, windows in items:
                if cancel_flag and cancel_flag[0]:
                    warnings.append("Cancelled by user.")
                    break
                sess_id, sess_clips, sess_warnings, n_clips = _process_session(sid, windows)
                if n_clips > 0:
                    extracted_sessions.append(sess_id)
                all_clips.extend(sess_clips)
                warnings.extend(sess_warnings)
        else:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_session, sid, windows) for sid, windows in items]
                for fut in cf.as_completed(futures):
                    if cancel_flag and cancel_flag[0]:
                        warnings.append("Cancelled by user.")
                    sess_id, sess_clips, sess_warnings, n_clips = fut.result()
                    if n_clips > 0:
                        extracted_sessions.append(sess_id)
                    all_clips.extend(sess_clips)
                    warnings.extend(sess_warnings)

        return {
            "session_ids": extracted_sessions,
            "clips": all_clips,
            "warnings": warnings,
            "total_windows": total,
        }

    def _subject_by_session(self) -> dict[str, str]:
        """Resolve session_id → subject label using the import manifest.

        Session IDs not present in the current manifest (e.g. from a previous
        import that was rebuilt) are resolved via the session registry so that
        subjects are displayed correctly and clip extraction can proceed.
        """
        if not self._project_root:
            return {}
        manifest = self._imports.load_manifest(self._project_root)
        if manifest is None:
            return {}
        video_by_id = {v.asset_id: v for v in manifest.videos}
        out: dict[str, str] = {}
        for session in manifest.linked_sessions:
            subject = (session.subject_id or "").strip()
            video = video_by_id.get(session.video_asset_id)
            if not subject:
                subject = (video.subject_id or "").strip() if video else ""
            if not subject and video:
                subject = Path(video.source_path).stem.strip()
            out[session.session_id] = subject or session.session_id
        # For candidate session IDs not in the current manifest, fall back to registry.
        all_candidate_ids = set(self._candidates_by_session.keys())
        unmapped = all_candidate_ids - set(out.keys())
        if unmapped:
            registry = self._imports.load_registry(self._project_root)
            for old_sid in unmapped:
                entry = registry.get(old_sid)
                if entry and entry.get("subject_id"):
                    out[old_sid] = entry["subject_id"]
        return out

    def _populate_table(self, rows: list[CandidateWindow]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        if not rows:
            self._table.setSortingEnabled(True)
            return

        subject_map = self._subject_by_session()
        behavior_name_map = {b.behavior_id: b.name for b in self._behaviors.behaviors}

        # Occurrence counter within this session view: sorted by behavior then start_frame.
        sorted_rows = sorted(rows, key=lambda c: (c.behavior_id or "", int(c.start_frame)))
        counters: dict[str, int] = {}
        occurrence: dict[str, int] = {}
        for c in sorted_rows:
            key = c.behavior_id or ""
            counters[key] = counters.get(key, 0) + 1
            occurrence[c.window_id] = counters[key]

        # Color map for candidate source types.
        _SOURCE_COLORS: dict[str, str] = {
            "hard_negative": "#E65100",
            "confound_boundary": "#E65100",
            "uncertainty": "#1565C0",
            "disagreement": "#6A1B9A",
            "exploration": "#2E7D32",
            "diversity": "#00838F",
            "candidate": "#546E7A",
            "baseline": "#546E7A",
        }

        # Pre-build a lookup of existing clip files so the status column
        # reflects actual clip extraction, not just directory existence.
        # Key: (session_id, start_frame, end_frame) → True
        import re as _clip_re
        _clip_frame_pattern = _clip_re.compile(r"_(\d+)_(\d+)(?:_[0-9a-f]+)?\.mp4$")
        _existing_clips: set[tuple[str, int, int]] = set()
        _clip_sessions_with_files: set[str] = set()
        if self._project_root:
            clips_root = self._project_root / "derived" / "clips"
            if clips_root.exists():
                try:
                    for session_dir in clips_root.iterdir():
                        if not session_dir.is_dir():
                            continue
                        sid = session_dir.name
                        for clip_file in session_dir.iterdir():
                            if not clip_file.suffix == ".mp4":
                                continue
                            _clip_sessions_with_files.add(sid)
                            m = _clip_frame_pattern.search(clip_file.name)
                            if m:
                                _existing_clips.add((sid, int(m.group(1)), int(m.group(2))))
                except OSError:
                    pass

        # Batch-insert rows: set row count once then fill cells.
        self._table.setRowCount(len(rows))

        for row_idx, c in enumerate(rows):
            subject = subject_map.get(c.session_id, c.session_id) or c.session_id
            bname = behavior_name_map.get(c.behavior_id or "", c.behavior_id or "") if c.behavior_id else "—"

            # Check if a clip file matching this candidate's frame range exists.
            clip_available = (
                bool(c.clip_path)
                or (c.session_id, int(c.start_frame), int(c.end_frame)) in _existing_clips
            )

            occ_item = QTableWidgetItem()
            occ_item.setData(Qt.ItemDataRole.DisplayRole, occurrence.get(c.window_id, 0))
            occ_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            score_item = QTableWidgetItem()
            score_item.setData(Qt.ItemDataRole.DisplayRole, round(c.total_score, 3))
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            start_item = QTableWidgetItem()
            start_item.setData(Qt.ItemDataRole.DisplayRole, int(c.start_frame))

            end_item = QTableWidgetItem()
            end_item.setData(Qt.ItemDataRole.DisplayRole, int(c.end_frame))

            self._table.setItem(row_idx, 0, QTableWidgetItem(subject))
            self._table.setItem(row_idx, 1, QTableWidgetItem(bname))
            self._table.setItem(row_idx, 2, occ_item)
            self._table.setItem(row_idx, 3, start_item)
            self._table.setItem(row_idx, 4, end_item)
            self._table.setItem(row_idx, 5, score_item)

            source_label = str(c.selection_reason or c.source or "").strip()
            source_item = QTableWidgetItem(source_label if source_label else "—")
            source_item.setToolTip(source_label)
            color_hex = _SOURCE_COLORS.get(source_label)
            if color_hex:
                source_item.setForeground(QColor(color_hex))
            self._table.setItem(row_idx, 6, source_item)

            # Use a lightweight text item instead of a full QPushButton widget.
            # Clip playback is only needed in the Review tab — here we just
            # show availability status.
            status_text = "✓" if clip_available else "—"
            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setToolTip("Clip extracted" if clip_available else "Not extracted")
            self._table.setItem(row_idx, 7, status_item)

        self._table.setSortingEnabled(True)

    def _run(self) -> None:
        self._append_log("Extract Clips clicked.")
        try:
            if not self._project_root:
                self._append_log("Extraction not started: no project loaded.")
                QMessageBox.warning(self, "No Project", "Open a project first.")
                return

            try:
                self._persist_ui_settings_to_project()
            except Exception as exc:
                self._append_log(f"Warning: failed to save clip-extraction UI settings ({exc}).")

            if not self._service.can_decode_video():
                QMessageBox.warning(
                    self,
                    "OpenCV Not Available",
                    "Video decoding is not available in this Python environment.\n\n"
                    "Install preprocessing dependencies from the Dependencies tab, then retry clip extraction.",
                )
                self._append_log("Extraction blocked: OpenCV video decoding is unavailable.")
                return

            session_id = self._session_combo.currentData()
            preset = self._preset_combo.currentData()
            if not session_id or not preset:
                self._append_log("Extraction not started: missing subject or preset selection.")
                QMessageBox.warning(self, "Missing inputs", "Select subject and preset.")
                return

            runtime_crop_area_scale = float(self._crop_area_percent.value()) / 100.0
            runtime_preset = PreprocessingPreset.model_validate(
                preset.model_dump(mode="python") | {"crop_area_scale": runtime_crop_area_scale}
            )

            subject_map = self._subject_by_session()
            subject_label = "all subjects" if session_id == self._ALL_SUBJECTS_KEY else (subject_map.get(session_id, session_id) or session_id)

            manifest = self._imports.load_manifest(self._project_root)
            if not manifest:
                self._append_log("Extraction not started: import manifest not found.")
                QMessageBox.warning(self, "No import manifest", "Run Data Import first.")
                return

            source_fps = self._source_fps()
            behavior_id = self._behavior_combo.currentData()
            session_plan = self._selected_candidates_by_session(
                session_id=str(session_id),
                behavior_id=behavior_id,
                source_fps=source_fps,
            )
            total_selected = sum(len(rows) for rows in session_plan.values())

            if total_selected <= 0 and behavior_id not in (None, self._UNASSIGNED_BEHAVIOR_FILTER):
                fallback_plan = self._selected_candidates_by_session(
                    session_id=str(session_id),
                    behavior_id=self._UNASSIGNED_BEHAVIOR_FILTER,
                    source_fps=source_fps,
                )
                fallback_total = sum(len(rows) for rows in fallback_plan.values())
                if fallback_total > 0:
                    session_plan = fallback_plan
                    total_selected = fallback_total
                    idx = self._behavior_combo.findData(self._UNASSIGNED_BEHAVIOR_FILTER)
                    if idx >= 0:
                        self._behavior_combo.setCurrentIndex(idx)
                    self._append_log(
                        "No candidates matched the selected behavior filter; "
                        "falling back to unassigned/random windows."
                    )

            if total_selected <= 0 and behavior_id is not None:
                fallback_plan = self._selected_candidates_by_session(
                    session_id=str(session_id),
                    behavior_id=None,
                    source_fps=source_fps,
                )
                fallback_total = sum(len(rows) for rows in fallback_plan.values())
                if fallback_total > 0:
                    session_plan = fallback_plan
                    total_selected = fallback_total
                    idx = self._behavior_combo.findData(None)
                    if idx >= 0:
                        self._behavior_combo.setCurrentIndex(idx)
                    self._append_log(
                        "No candidates matched the current behavior filter; "
                        "falling back to all behaviors."
                    )

            if total_selected <= 0:
                QMessageBox.warning(self, "No candidates", "No candidates available for the selected settings.")
                self._append_log("Extraction not started: no candidates matched the current filters.")
                return

            self._cancel_flag[0] = False
            self._set_busy_state(True)
            self._waiting_for_worker_finish = True
            self._extraction_started_at = time.monotonic()
            self._progress.setRange(0, total_selected)
            self._progress.setValue(0)
            self._progress.setFormat(f"Starting... 0/{total_selected}")
            self._status.setText(f"Extracting clips for subject {subject_label}...")
            self._append_log(
                f"Extracting {total_selected} clip(s) for {subject_label} "
                f"(top={int(self._top_n.value())}, bottom={int(self._bottom_n.value())}, "
                f"median={int(self._median_n.value())}, "
                f"crop_area={self._crop_area_percent.value():.0f}%, "
                f"context: -{self._before_sec.value():.2f}s / +{self._after_sec.value():.2f}s @ {source_fps:.2f} fps)..."
            )
            requested_workers = os.environ.get("ABEL_CLIP_EXTRACT_WORKERS", "").strip() or "auto"
            self._append_log(f"Clip extraction workers: {requested_workers} (env ABEL_CLIP_EXTRACT_WORKERS).")

            def _prog(done: int, total: int) -> None:
                elapsed = 0.0
                if self._extraction_started_at is not None:
                    elapsed = max(0.0, time.monotonic() - self._extraction_started_at)
                if done <= 0 or elapsed <= 0:
                    eta_txt = "estimating..."
                else:
                    rate = done / elapsed
                    eta_sec = int(max(0.0, (total - done) / max(rate, 1e-6)))
                    eta_txt = self._format_eta(eta_sec)
                self.extraction_progress_requested.emit(done, total, eta_txt)

            worker = TaskWorker(
                self._extract_sessions_task,
                session_plan,
                runtime_preset,
                _prog,
                self._cancel_flag,
            )
            worker.signals.finished.connect(self._on_finished)
            worker.signals.failed.connect(self._on_failed)
            self._pool.start(worker)
        except Exception:
            self._append_log("Clip extraction failed before worker start:")
            self._append_log(traceback.format_exc()[:1000])
            self._set_busy_state(False)
            self._waiting_for_worker_finish = False

    def _on_finished(self, payload: dict[str, Any]) -> None:
        self._waiting_for_worker_finish = False
        self._set_busy_state(False)
        clips = payload.get("clips", [])
        warnings = [str(w) for w in payload.get("warnings", [])]
        total_windows = int(payload.get("total_windows", 0))
        session_ids = [str(s) for s in payload.get("session_ids", [])]

        if clips:
            manifest = ClipManifest(
                session_ids=session_ids,
                preset_name=self._preset_combo.currentText(),
                clips=clips,
                total_windows=total_windows,
                opencv_available=all(c.processed_clip_path for c in clips),
                warnings=warnings,
            )
            self._service.save_manifest(manifest)
            self._append_log(f"Clip extraction complete: {len(clips)} clip(s).")
            if len(clips) < total_windows:
                self._append_log(
                    f"Note: extracted {len(clips)}/{total_windows} selected clips. "
                    "See warnings for skipped windows."
                )
            self._progress.setValue(self._progress.maximum())
            self._progress.setFormat(
                f"Done - {len(clips)}/{total_windows} extracted"
            )
            self._status.setText(
                f"Clip extraction complete: {len(clips)} extracted across {len(session_ids)} session(s)."
            )
            self._refresh()
        else:
            self._append_log(
                f"Clip extraction failed: 0/{total_windows} playable clips created."
            )
            self._progress.setFormat("Failed - see log")
            self._status.setText("Clip extraction failed.")
        for w in warnings:
            self._append_log(f"Warning: {w}")
        if warnings:
            self._append_log(f"Finished with {len(warnings)} warning(s).")

    def _on_failed(self, traceback_text: str) -> None:
        self._waiting_for_worker_finish = False
        self._set_busy_state(False)
        self._append_log("Clip extraction failed:")
        self._append_log(format_task_error(traceback_text))
        self._progress.setFormat("Error")
        self._status.setText("Clip extraction crashed.")

    @Slot(int, int, str)
    def _on_extraction_progress(self, done: int, total: int, eta_text: str) -> None:
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)
        self._progress.setFormat(f"{done}/{total} clips  |  ETA {eta_text}")
        self._status.setText(
            f"Extracting clips... {done}/{total} complete (ETA {eta_text})."
        )
        if done > 0 and (done == total or done % 10 == 0):
            self._append_log(f"Progress: {done}/{total} clips complete (ETA {eta_text}).")

        # Safety net: if progress reaches total but finished signal never arrives,
        # recover UI state and refresh from disk so play buttons can be enabled.
        if total > 0 and done >= total and self._waiting_for_worker_finish:
            self._status.setText("Finalizing extraction...")
            QTimer.singleShot(3000, self._recover_if_worker_stuck)

    def _cancel(self) -> None:
        self._cancel_flag[0] = True
        self._cancel_btn.setEnabled(False)
        self._append_log("Cancellation requested...")

    def _clear_clips(self) -> None:
        if not self._project_root:
            return
        session_id = self._session_combo.currentData()
        if not session_id:
            QMessageBox.warning(self, "No Subject", "Select a subject first.")
            return
        subject_map = self._subject_by_session()
        clear_all = str(session_id) == self._ALL_SUBJECTS_KEY
        subject_label = "all subjects" if clear_all else (subject_map.get(session_id, session_id) or session_id)

        answer = QMessageBox.question(
            self,
            "Clear Extracted Clips",
            "Delete the rendered video clip files for the selected subject?\n\n"
            "This only removes the clip files on disk (re-extractable). Your review "
            "labels and the candidate queue are not affected.\n\n"
            f"Subject: {subject_label}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        removed_files = self._service.clear_extracted_clips(None if clear_all else str(session_id))
        cleared_links = self._candidate_service.clear_clip_paths(None if clear_all else str(session_id))
        self._append_log(
            f"Cleared clips for {subject_label}: removed {removed_files} file(s), "
            f"cleared {cleared_links} candidate clip link(s)."
        )
        self._status.setText(
            f"Cleared clips for {subject_label} ({removed_files} file(s) removed)."
        )
        self._refresh()

    def _clear_candidates(self) -> None:
        if not self._project_root:
            return

        answer = QMessageBox.question(
            self,
            "Clear Candidate Queue",
            "Remove pending (unreviewed) windows from the review queue?\n\n"
            "This clears the current clip-window queue shown in this tab. Windows "
            "you have already reviewed are KEPT, your review labels are not "
            "changed, and no extracted clip files are deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Preserve reviewed windows in both the persisted queue and the
        # in-memory external candidates loaded from other tabs.
        summary = self._candidate_service.clear_candidate_queue(preserve_reviewed=True)
        reviewed = self._candidate_service.reviewed_segment_ids()
        before_ext = len(self._external_candidates)
        self._external_candidates = [
            c for c in self._external_candidates
            if str(c.window_id).strip() in reviewed
        ]
        external_removed = before_ext - len(self._external_candidates)
        if not self._external_candidates:
            self._external_source_label = ""

        removed = int(summary.get("removed", 0)) + external_removed
        kept = int(summary.get("kept_reviewed", 0)) + len(self._external_candidates)

        if removed > 0:
            self._append_log(
                f"Cleared candidate queue: removed {removed} pending window(s); "
                f"kept {kept} reviewed window(s). Review labels and extracted clips untouched."
            )
            self._status.setText(
                f"Removed {removed} pending candidate(s); kept {kept} reviewed."
            )
        else:
            self._append_log("Clear candidates: no pending (unreviewed) windows to remove.")
            self._status.setText("No pending candidates to clear.")
        self._refresh()

    def _set_busy_state(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy)
        self._clear_btn.setEnabled(not busy)
        self._clear_candidates_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._refresh_btn.setEnabled(not busy)
        self._session_combo.setEnabled(not busy)
        self._preset_combo.setEnabled(not busy)
        self._top_n.setEnabled(not busy)
        self._bottom_n.setEnabled(not busy)
        self._median_n.setEnabled(not busy)
        self._crop_area_percent.setEnabled(not busy)
        self._before_sec.setEnabled(not busy)
        self._after_sec.setEnabled(not busy)

    def _build_extraction_window(self, cand: CandidateWindow, source_fps: float) -> CandidateWindow:
        """Return a copy expanded by user-defined static before/after time."""
        before_frames = int(round(max(0.0, self._before_sec.value()) * max(source_fps, 1.0)))
        after_frames = int(round(max(0.0, self._after_sec.value()) * max(source_fps, 1.0)))
        try:
            raw_start = int(cand.start_frame)
        except Exception:
            raw_start = 0
        try:
            raw_end = int(cand.end_frame)
        except Exception:
            raw_end = raw_start
        start = max(0, raw_start - before_frames)
        end = max(start, raw_end + after_frames)
        return cand.model_copy(update={"start_frame": int(start), "end_frame": int(end)})

    def _source_fps(self) -> float:
        if not self._project_root:
            return 30.0
        data = read_yaml(self._project_root / "project.yaml", {})
        try:
            return float(data.get("default_fps", 30.0))
        except Exception:
            return 30.0

    def _recover_if_worker_stuck(self) -> None:
        """Recover from rare case where worker finished signal does not reach the UI."""
        if not self._waiting_for_worker_finish:
            return
        self._waiting_for_worker_finish = False
        self._set_busy_state(False)
        self._append_log(
            "Recovery: extraction progress reached 100% but completion signal was delayed/missed; "
            "refreshing clip list from disk."
        )
        self._status.setText("Recovered from delayed completion; refreshed clip list.")
        self._refresh()

    @staticmethod
    def _format_eta(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        mins, secs = divmod(seconds, 60)
        if mins < 60:
            return f"{mins}m {secs:02d}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins:02d}m"

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    def _play_clip(self, clip_path: str) -> None:
        path = Path(clip_path)
        if not path.exists():
            QMessageBox.warning(self, "Missing Clip", f"Clip file not found:\n{clip_path}")
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        if ok:
            return
        try:
            os.startfile(str(path.resolve()))
        except Exception:
            self._append_log(f"Failed to open clip: {clip_path}")

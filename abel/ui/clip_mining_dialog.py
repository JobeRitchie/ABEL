"""Targeted Clip Mining dialog.

Lets a reviewer hunt for clips by *meaning* rather than by model score: set one
or more interpretable criteria (nose past the edge > 10 mm, tail near the closed
arm, centroid speed in a range, …) and pull every matching clip into the review
queue.  The **Essence Extractor** goes the other way — highlight a few exemplar
clips and it fills those criteria from the exemplars' overlapping value ranges.

Scope is the *full* feature-extraction segment pool for the project — every
window scored during feature extraction, not just the clips already loaded into
the review queue.  Scoring is deferred: nothing is computed on open.  The user
builds criteria and clicks **Find matches**, which scores the pool in a
background worker (progress bar shown) and caches the table; later criteria edits
re-filter the cached table instantly.  Essence extraction scores only the handful
of selected exemplar clips, so it works without a full pool scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QDoubleValidator

from abel.services.clip_metrics_service import (
    ClipMetricsService,
    ClipRef,
    Criterion,
    MetricDef,
    METRIC_BY_ID,
)
from abel.workers.task_worker import TaskWorker

def _fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


class _CriterionRow(QWidget):
    """One editable criterion: enable · metric · min · max · live scope range."""

    changed = Signal()
    removed = Signal(object)

    def __init__(
        self,
        stats: "pd.DataFrame | None",
        metric_defs: list[MetricDef],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stats = stats
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.enable = QPushButton("✓")
        self.enable.setCheckable(True)
        self.enable.setChecked(True)
        self.enable.setFixedWidth(28)
        self.enable.setToolTip("Include this criterion when matching")
        self.enable.toggled.connect(lambda _c: (self._sync_enabled(), self.changed.emit()))
        layout.addWidget(self.enable)

        self.metric = QComboBox()
        self.metric.setMinimumWidth(230)
        for m in metric_defs:
            self.metric.addItem(f"{m.group} — {m.label}", userData=m.id)
            self.metric.setItemData(self.metric.count() - 1, m.description, Qt.ItemDataRole.ToolTipRole)
        self.metric.currentIndexChanged.connect(self._on_metric_changed)
        layout.addWidget(self.metric)

        layout.addWidget(QLabel("≥"))
        self.low = QLineEdit()
        self.low.setPlaceholderText("any")
        self.low.setFixedWidth(64)
        self.low.setValidator(QDoubleValidator())
        self.low.textChanged.connect(lambda _t: self.changed.emit())
        layout.addWidget(self.low)

        layout.addWidget(QLabel("≤"))
        self.high = QLineEdit()
        self.high.setPlaceholderText("any")
        self.high.setFixedWidth(64)
        self.high.setValidator(QDoubleValidator())
        self.high.textChanged.connect(lambda _t: self.changed.emit())
        layout.addWidget(self.high)

        self.unit = QLabel("")
        self.unit.setFixedWidth(40)
        self.unit.setStyleSheet("color: #78909C;")
        layout.addWidget(self.unit)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color: #90A4AE; font-size: 11px;")
        self.hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.hint)

        self.remove_btn = QPushButton("✕ Remove")
        self.remove_btn.setMinimumWidth(80)
        self.remove_btn.setToolTip("Remove this criterion")
        self.remove_btn.setStyleSheet(
            "QPushButton { color: #C62828; font-weight: 600; }"
            "QPushButton:hover { background-color: #FFEBEE; }"
        )
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.remove_btn)

        self._on_metric_changed()

    def _sync_enabled(self) -> None:
        on = self.enable.isChecked()
        self.enable.setText("✓" if on else "○")
        for w in (self.metric, self.low, self.high):
            w.setEnabled(on)

    def _on_metric_changed(self) -> None:
        mid = self.metric.currentData()
        m = METRIC_BY_ID.get(mid)
        self.unit.setText(m.unit if m else "")
        if self._stats is not None and mid in self._stats.columns:
            col = pd.to_numeric(self._stats[mid], errors="coerce")
            col = col[np.isfinite(col)]
            if len(col):
                self.hint.setText(
                    f"scope: min {_fmt(float(col.min()))} · "
                    f"med {_fmt(float(col.median()))} · "
                    f"max {_fmt(float(col.max()))}"
                )
            else:
                self.hint.setText("scope: no data")
        self.changed.emit()

    def disarm(self) -> None:
        """Disconnect every child-widget signal ahead of teardown.

        Destroying the metric combo emits ``currentIndexChanged(-1)``, which would
        otherwise re-enter ``_on_metric_changed`` on the half-destroyed row (it
        reads sibling widgets and re-emits ``changed``). Left connected, that
        re-entrancy during reparent is what turns removal into a native access
        violation. Disconnecting first makes destruction inert.
        """
        for sig in (
            self.enable.toggled,
            self.metric.currentIndexChanged,
            self.low.textChanged,
            self.high.textChanged,
            self.remove_btn.clicked,
            self.changed,
            self.removed,
        ):
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass

    def set_metric(self, metric_id: str) -> None:
        idx = self.metric.findData(metric_id)
        if idx >= 0:
            self.metric.setCurrentIndex(idx)

    def set_range(self, low: float | None, high: float | None) -> None:
        self.low.setText("" if low is None else _fmt(low))
        self.high.setText("" if high is None else _fmt(high))

    def to_criterion(self) -> Criterion:
        def _num(t: str) -> float | None:
            t = t.strip().replace(",", ".")
            try:
                return float(t)
            except ValueError:
                return None
        return Criterion(
            metric_id=str(self.metric.currentData()),
            low=_num(self.low.text()),
            high=_num(self.high.text()),
            enabled=self.enable.isChecked(),
        )


class ClipMiningDialog(QDialog):
    """Interactive criteria builder + essence extractor over a clip set."""

    _progress_sig = Signal(int, int)

    def __init__(
        self,
        project_root: Path,
        exemplar_provider: Callable[[], list[ClipRef]],
        scope_label: str,
        on_apply: Callable[[list[ClipRef], dict], None],
        parent: QWidget | None = None,
        scope_sessions: set[str] | None = None,
        on_flag_queue: Callable[[list[Criterion], bool], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Targeted Clip Mining")
        self.setMinimumSize(720, 560)
        self._project_root = project_root
        self._exemplar_provider = exemplar_provider
        self._on_apply = on_apply
        self._on_flag_queue = on_flag_queue
        self._scope_sessions = scope_sessions
        self._metrics = ClipMetricsService()
        self._metrics.set_project(project_root)
        # Project-aware metric list: one ROI block per configured target zone.
        self._metric_defs = self._metrics.available_metrics()
        # Search index for the "Add feature" bar: display string → metric id, plus
        # a lowercased blob (group + label + description) for substring matching.
        self._feature_display: dict[str, str] = {}
        self._feature_index: list[tuple[str, str]] = []  # (metric_id, searchable)
        for m in self._metric_defs:
            self._feature_display[f"{m.group} — {m.label}"] = m.id
            self._feature_index.append(
                (m.id, f"{m.group} {m.label} {m.description}".lower())
            )
        # Full pool is loaded lazily on the first Find-matches; keep the map so
        # matched window_ids can be turned back into ClipRefs for extraction.
        self._clip_by_id: dict[str, ClipRef] = {}
        self._df: pd.DataFrame | None = None
        self._rows: list[_CriterionRow] = []
        self._last_matches: list[str] = []
        self._last_scores: dict = {}
        self._scored = False
        self._mining = False
        self._essence_busy = False
        self._suspend_updates = False

        root = QVBoxLayout(self)
        root.setSpacing(10)

        intro = QLabel(
            "Find clips by what the animal is doing. Add criteria and click Find matches "
            "to search every scored segment, or select exemplar clips in the review list "
            "and let Extract essence fill the ranges."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #607D8B;")
        root.addWidget(intro)

        scope = QLabel(f"Scope: {scope_label}")
        scope.setStyleSheet("font-weight: 600;")
        root.addWidget(scope)

        # Criteria area (scrollable).
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        root.addWidget(scroll, 1)

        # Criteria controls row: search-to-add feature bar + match mode.
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Add feature:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search features (e.g. speed, zone, distance)…")
        self._search.setClearButtonEnabled(True)
        self._search.setMinimumWidth(240)
        completer = QCompleter(list(self._feature_display.keys()), self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.activated.connect(self._add_searched_feature)
        self._search.setCompleter(completer)
        self._search.returnPressed.connect(self._add_searched_feature)
        ctrl.addWidget(self._search, 1)
        add_btn = QPushButton("+ Add")
        add_btn.setToolTip("Add the searched feature as a new criterion")
        add_btn.clicked.connect(self._add_searched_feature)
        ctrl.addWidget(add_btn)
        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Match:"))
        self._match_all = QRadioButton("All (AND)")
        self._match_any = QRadioButton("Any (OR)")
        self._match_all.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._match_all)
        grp.addButton(self._match_any)
        self._match_all.toggled.connect(lambda _c: self._update_count())
        ctrl.addWidget(self._match_all)
        ctrl.addWidget(self._match_any)
        ctrl.addStretch(1)
        root.addLayout(ctrl)

        # Essence row.
        ess = QHBoxLayout()
        self._essence_btn = QPushButton("⤢ Extract essence from selected clips")
        self._essence_btn.setToolTip(
            "Find the features the selected review clips most have in common and add the\n"
            "top few as criteria automatically — you don't pick the features, it does.\n"
            "This window stays open — change your selection behind it, then click again."
        )
        self._essence_btn.clicked.connect(self._extract_essence)
        ess.addWidget(self._essence_btn)
        ess.addWidget(QLabel("range:"))
        self._essence_method = QComboBox()
        self._essence_method.addItem("min–max (tight)", "minmax")
        self._essence_method.addItem("10–90th pct (robust)", "p10p90")
        ess.addWidget(self._essence_method)
        ess.addStretch(1)
        root.addLayout(ess)

        # Flag-in-queue row: audit the *current review filter* against these
        # ranges and highlight the clips that fall outside them.  Only shown when
        # the host wired a handler (i.e. opened from the review tab).
        if self._on_flag_queue is not None:
            flag_row = QHBoxLayout()
            self._flag_queue_btn = QPushButton("🚩 Flag failing clips in review queue")
            self._flag_queue_btn.setToolTip(
                "Check the clips currently shown in the review list against these ranges\n"
                "and highlight the ones that fall OUTSIDE them (fail the essence test).\n"
                "Use after Extract essence to catch reviewed clips that no longer meet a\n"
                "tightened operational definition. Acts on the current review filter, not the pool."
            )
            self._flag_queue_btn.setStyleSheet(
                "background-color: #C62828; color: white; font-weight: 600; padding: 6px 12px;"
            )
            self._flag_queue_btn.clicked.connect(self._flag_queue)
            flag_row.addWidget(self._flag_queue_btn)
            flag_row.addStretch(1)
            root.addLayout(flag_row)

        # Mine trigger + progress.
        mine_row = QHBoxLayout()
        self._mine_btn = QPushButton("🔎 Find matches")
        self._mine_btn.setToolTip(
            "Score every feature-extraction segment for the current criteria and "
            "count the matches. Scoring runs once; later edits re-filter instantly."
        )
        self._mine_btn.setStyleSheet(
            "background-color: #455A64; color: white; font-weight: 600; padding: 6px 12px;"
        )
        self._mine_btn.clicked.connect(self._mine)
        mine_row.addWidget(self._mine_btn)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Scoring segments… %p%")
        self._progress.setVisible(False)
        mine_row.addWidget(self._progress, 1)
        root.addLayout(mine_row)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("font-weight: 600; color: #00695C;")
        root.addWidget(self._count_label)

        # Action buttons.
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Max clips to load:"))
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(1, 1_000_000)
        self._cap_spin.setValue(500)
        self._cap_spin.setSingleStep(100)
        self._cap_spin.setToolTip(
            "Cap on how many matched segments get extracted and loaded, so loose\n"
            "criteria can't decode thousands of clips by accident. Only the\n"
            "highest-scoring matches up to this many are loaded."
        )
        self._cap_spin.valueChanged.connect(lambda _v: self._update_count())
        actions.addWidget(self._cap_spin)
        actions.addStretch(1)
        self._apply_btn = QPushButton("Load matches into review queue")
        self._apply_btn.setStyleSheet(
            "background-color: #00796B; color: white; font-weight: 600; padding: 6px 12px;"
        )
        self._apply_btn.clicked.connect(self._apply)
        self._apply_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        actions.addWidget(self._apply_btn)
        actions.addWidget(close_btn)
        root.addLayout(actions)

        # Marshal worker-thread progress onto the UI thread (connected once).
        self._progress_sig.connect(self._on_compute_progress)
        # Restore the project's saved criteria, or seed one row to start from — no
        # scoring happens until Find matches is clicked.
        self._restore_or_seed()
        self.refresh_exemplar_count()
        self._update_count()

    # -- metric computation (deferred until the user mines) ------------------

    def _mine(self) -> None:
        """Score the full pool (first time) then count matches for the criteria."""
        if self._mining:
            return
        if self._scored:
            # Metrics already cached — criteria edits just re-filter.
            self._update_count()
            return
        from PySide6.QtCore import QThreadPool

        if not self._clip_by_id:
            clips = self._metrics.load_segment_pool(self._scope_sessions)
            self._clip_by_id = {c.window_id: c for c in clips}
        clips = list(self._clip_by_id.values())
        if not clips:
            self._count_label.setText(
                "No scored segments found — run Feature Extraction first."
            )
            return

        self._mining = True
        self._mine_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(clips))
        self._progress.setValue(0)
        self._progress.setFormat("Scoring segments… %p%")
        worker = TaskWorker(self._metrics.compute, clips, progress_callback=self._emit_progress)
        worker.signals.finished.connect(self._on_metrics_ready)
        worker.signals.failed.connect(self._on_metrics_failed)
        QThreadPool.globalInstance().start(worker)

    def _emit_progress(self, done: int, total: int) -> None:
        # Runs in the worker thread; the queued signal marshals to the UI thread.
        # Guarded because the dialog may be closed/deleted mid-scoring — emitting
        # on a deleted QObject raises RuntimeError we simply swallow.
        try:
            self._progress_sig.emit(int(done), int(total))
        except RuntimeError:
            pass

    def _on_compute_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(done)

    def _on_metrics_ready(self, df: pd.DataFrame) -> None:
        self._df = df
        self._scored = True
        self._mining = False
        self._mine_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat(f"Scored {len(df)} segment(s)")
        # Populate the live scope hints on every existing row.
        for r in self._rows:
            r._stats = df
            r._on_metric_changed()
        self._update_count()

    def _on_metrics_failed(self, tb: str) -> None:
        self._mining = False
        self._mine_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Failed to score segments")
        self._count_label.setText("Metric computation failed — see log.")

    def refresh_exemplar_count(self) -> None:
        """Update the essence button to reflect the live review-list selection.

        Called on open and whenever the main window's selection changes, so the
        modeless dialog always acts on the clips currently highlighted behind it.
        """
        n = 0
        try:
            n = len(self._exemplar_provider() or [])
        except Exception:
            n = 0
        self._essence_btn.setText(
            f"⤢ Extract essence from {n} selected clip(s)" if n
            else "⤢ Extract essence from selected clips"
        )
        self._essence_btn.setEnabled(n > 0 and not self._essence_busy)

    # -- feature search / persistence ----------------------------------------

    def _resolve_feature(self, text: str) -> str | None:
        """Map a search string to a metric id (exact display, else substring)."""
        text = (text or "").strip()
        if not text:
            return None
        mid = self._feature_display.get(text)
        if mid is not None:
            return mid
        needle = text.lower()
        for metric_id, blob in self._feature_index:
            if needle in blob:
                return metric_id
        return None

    def _add_searched_feature(self, text: object = None) -> None:
        """Add the searched feature as a criterion row (Enter / Add / completer pick)."""
        query = (text if isinstance(text, str) else self._search.text() or "").strip()
        if not query:
            # Empty box (e.g. the returnPressed that trails a completer pick) — no-op.
            return
        mid = self._resolve_feature(query)
        if mid is None:
            self._count_label.setText(
                "No feature matches that search — try 'speed', 'zone', 'distance', 'body'…"
            )
            return
        self._add_row(mid)
        self._search.clear()

    def _restore_or_seed(self) -> None:
        """Rebuild rows from the project's saved criteria, or seed a starter row."""
        criteria, match_all = self._metrics.load_criteria()
        if match_all is not None:
            (self._match_all if match_all else self._match_any).setChecked(True)
        self._suspend_updates = True
        try:
            if criteria:
                for c in criteria:
                    row = self._add_row(c.metric_id)
                    row.set_range(c.low, c.high)
                    row.enable.setChecked(bool(c.enabled))
            else:
                self._add_row("centroid_in_roi_frac")
        finally:
            self._suspend_updates = False

    def _persist_criteria(self) -> None:
        """Save the current criteria + match mode to the project (best-effort)."""
        try:
            self._metrics.save_criteria(
                self._current_criteria(), self._match_all.isChecked()
            )
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._persist_criteria()
        super().closeEvent(event)

    def reject(self) -> None:  # noqa: D102 (Qt override)
        self._persist_criteria()
        super().reject()

    # -- criteria rows -------------------------------------------------------

    def _add_row(self, metric_id: str | None = None) -> _CriterionRow:
        row = _CriterionRow(self._df, self._metric_defs)
        if metric_id:
            row.set_metric(metric_id)
        row.changed.connect(self._on_row_changed)
        row.removed.connect(self._request_remove_row)
        # Insert before the trailing stretch.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows.append(row)
        if not self._suspend_updates:
            self._update_count()
        return row

    def _on_row_changed(self) -> None:
        if self._suspend_updates:
            return
        self._update_count()

    def _request_remove_row(self, row: _CriterionRow) -> None:
        """Defer row teardown out of the remove button's own click emission.

        Belt-and-braces alongside the disarm in :meth:`_remove_row`: running the
        teardown on the next event-loop tick means the button click has fully
        unwound before the row is destroyed.
        """
        QTimer.singleShot(0, lambda: self._remove_row(row))

    def _remove_row(self, row: _CriterionRow) -> None:
        if row not in self._rows:
            return  # already removed (e.g. a double-click before the deferred tick)
        self._rows.remove(row)
        # Neutralise the row's signals so its imminent destruction can't re-enter
        # any slot on the half-torn-down widget.
        row.disarm()
        row.blockSignals(True)
        # The remove button still holds keyboard focus; destroying the row would
        # force a native focus transfer mid-teardown — the access violation the
        # crash log pinned to setParent(None). Park focus on a stable widget, then
        # drop the row from the layout WITHOUT an explicit reparent-to-None and let
        # deleteLater finish the teardown on the next tick (the codebase idiom).
        self._mine_btn.setFocus()
        self._rows_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        if not self._suspend_updates:
            self._update_count()

    def _clear_rows(self) -> None:
        was_suspended = self._suspend_updates
        self._suspend_updates = True
        try:
            for row in list(self._rows):
                self._remove_row(row)
        finally:
            self._suspend_updates = was_suspended

    # -- essence -------------------------------------------------------------

    def _collect_exemplars(self) -> list[ClipRef]:
        try:
            return list(self._exemplar_provider() or [])
        except Exception:
            return []

    def _extract_essence(self) -> None:
        """Discover the features the selected clips most share and add the top few.

        Works straight off the highlighted clips — it scores just those exemplars,
        with no full-pool scan required.  If the pool happens to already be scored
        (after a Find matches), that baseline is reused to also favour features that
        are discriminative; otherwise ranking falls back to a scale-free measure.
        """
        if self._essence_busy or self._mining:
            return
        exemplars = self._collect_exemplars()
        if not exemplars:
            self._count_label.setText(
                "Select one or more clips in the review list first, then Extract essence."
            )
            return
        self._run_similarity_essence(exemplars)

    def _run_similarity_essence(self, exemplars: list[ClipRef]) -> None:
        """Score the exemplars, find the top shared features, and add them as criteria."""
        method = str(self._essence_method.currentData())
        self._essence_busy = True
        self._essence_btn.setEnabled(False)
        try:
            ex_df = self._metrics.compute(exemplars)
        except Exception:
            ex_df = None
        finally:
            self._essence_busy = False
        self.refresh_exemplar_count()
        if ex_df is None or ex_df.empty:
            self._count_label.setText("Could not compute metrics for the selected clips.")
            return
        crits = self._metrics.extract_similar_essence(ex_df, self._df, k=3, method=method)
        if not crits:
            self._count_label.setText(
                "Couldn't find shared features — pick two or more clips that are alike."
            )
            return
        # Replace the criteria with the discovered top shared features.
        self._suspend_updates = True
        try:
            self._clear_rows()
            for c in crits:
                row = self._add_row(c.metric_id)
                row.set_range(c.low, c.high)
        finally:
            self._suspend_updates = False
        labels = ", ".join(
            METRIC_BY_ID[c.metric_id].label if c.metric_id in METRIC_BY_ID else c.metric_id
            for c in crits
        )
        self._count_label.setText(
            f"Added {len(crits)} shared feature(s) from {len(ex_df)} clip(s): {labels}. "
            "Click Find matches to search the pool."
        )
        self._update_count()

    # -- mining --------------------------------------------------------------

    def _current_criteria(self) -> list[Criterion]:
        return [r.to_criterion() for r in self._rows]

    def _update_count(self) -> None:
        active = [
            c for c in self._current_criteria()
            if c.enabled and (c.low is not None or c.high is not None)
        ]
        if self._df is None:
            # Not scored yet — nothing to count against.
            self._last_matches = []
            self._last_scores = {}
            self._apply_btn.setEnabled(False)
            if not self._mining:
                self._count_label.setText(
                    "Click Find matches to score the pool and count matches."
                    if active else "Add one or more criteria, then Find matches."
                )
            return
        res = self._metrics.mine(
            self._df, self._current_criteria(), match_all=self._match_all.isChecked()
        )
        self._last_matches = res.matched_ids
        self._last_scores = res.scores
        if not active:
            self._count_label.setText(f"No active criteria — {res.n_evaluated} segment(s) in scope.")
            self._apply_btn.setEnabled(False)
        else:
            n_match = len(res.matched_ids)
            cap = int(self._cap_spin.value())
            capped = f" — will load top {cap} by score" if n_match > cap else ""
            self._count_label.setText(
                f"{n_match} of {res.n_evaluated} segment(s) match "
                f"{len(active)} criteria.{capped}"
            )
            self._apply_btn.setEnabled(bool(res.matched_ids))

    def _flag_queue(self) -> None:
        """Hand the current essence ranges to the host to audit the review queue."""
        if self._on_flag_queue is None:
            return
        criteria = [
            c
            for c in self._current_criteria()
            if c.enabled and (c.low is not None or c.high is not None)
        ]
        if not criteria:
            self._count_label.setText(
                "Add or extract at least one criterion range before flagging the queue."
            )
            return
        self._on_flag_queue(criteria, self._match_all.isChecked())

    def _apply(self) -> None:
        if not self._last_matches:
            return
        # Load only the highest-scoring matches up to the cap, so loose criteria
        # can't trigger extraction of thousands of clips at once.
        cap = int(self._cap_spin.value())
        ordered = sorted(
            self._last_matches,
            key=lambda w: self._last_scores.get(w, 0.0),
            reverse=True,
        )[:cap]
        refs = [self._clip_by_id[w] for w in ordered if w in self._clip_by_id]
        if not refs:
            return
        n = len(refs)
        # Persist the working criteria so this feature set survives a reload.
        self._persist_criteria()
        self._on_apply(refs, dict(self._last_scores))
        # Stay open (modeless) so criteria can be refined and re-applied while the
        # mined queue updates behind the window.
        self._count_label.setText(f"Loading {n} matched clip(s) into the review queue…")

"""Targeted Clip Mining dialog.

Lets a reviewer hunt for clips by *meaning* rather than by model score: set one
or more interpretable criteria (nose past the edge > 10 mm, tail near the closed
arm, centroid speed in a range, …) and pull every matching clip into the review
queue.  The **Essence Extractor** goes the other way, highlight a few exemplar
clips and it fills those criteria from the exemplars' overlapping value ranges.
The **Auto Hunter** tab covers the case where neither is possible yet, a fresh
project with nothing reviewed to point at, by offering hand-written presets
(see :mod:`abel.services.hunter_presets`) that describe a whole family of
behavior in features every project has, and writing them into the same editable
criteria rows.

Scope is the *full* feature-extraction segment pool for the project, every
window scored during feature extraction, not just the clips already loaded into
the review queue.  Scoring is deferred: nothing is computed on open.  The user
builds criteria and clicks **Find matches**, which scores the pool in a
background worker (progress bar shown) and caches the table; later criteria edits
re-filter the cached table instantly.  Essence extraction scores only the handful
of selected exemplar clips, so it works without a full pool scan.

Hand-set criteria stay on the interpretable pose/ROI metrics, those are the ones
a human can reason about and set a number for.  The **Essence Extractor** also
ranges over the project's *extracted* per-window features (the classifier's own
oscillation / rotation / jerk / context columns), because it picks its own
features and that is where behaviors like a wet-dog-shake actually separate.
Whatever it picks is shown as an editable row with a humanised name, so the
definition stays inspectable.  Those features are precomputed, so an essence
built on them needs no pose file, and mining them reads the feature table
directly instead of re-scoring every segment from pose.

What a batch contains is shaped by two defaults, both toggleable: clips that
already carry a review decision are left out (checked live, so a clip judged
behind this modeless window drops out of the batch about to load), and the
capped batch is filled by taking turns between *subjects* rather than by raw
score, otherwise the one animal whose recording scores highest can fill it
alone, and the mined sample says nothing about the rest of the cohort.

A third, opt-in toggle aims the batch at *coverage gaps*: it keeps only matches
from subjects with fewer than k reviewed examples of a chosen behavior (see
:mod:`abel.services.behavior_coverage_service`), so the hunt goes to the mice
leave-one-mouse-out CV cannot score yet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor, QDoubleValidator

from abel.services.behavior_coverage_service import (
    DEFAULT_ABSENT_AFTER,
    DEFAULT_MIN_POSITIVES,
    BehaviorCoverageService,
    CoverageReport,
)
from abel.services.behavior_service import NO_BEHAVIOR_ID, BehaviorService
from abel.services.clip_metrics_service import (
    ClipMetricsService,
    ClipRef,
    Criterion,
    MetricDef,
    is_rich_metric,
    metric_def_for,
    metric_label,
    rich_metric_def,
    select_even_by_group,
)
from abel.services.hunter_presets import (
    BREADTH_LEVELS,
    PRESETS,
    fit_preset,
    preset_by_id,
)
from abel.services.label_needs_service import MEANINGFUL_GAIN, LabelNeedsConfig, analyze_label_needs
from abel.utils.cancellation import CANCEL_MARKER
from abel.workers.task_worker import TaskWorker

logger = logging.getLogger(__name__)


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
        # Measured, not hard-coded: extracted-feature labels are long, and every
        # width here must survive Windows display scaling.
        em = self.fontMetrics().horizontalAdvance("M")

        self.enable = QPushButton("✓")
        self.enable.setCheckable(True)
        self.enable.setChecked(True)
        self.enable.setFixedWidth(2 * em)
        self.enable.setToolTip("Include this criterion when matching")
        self.enable.toggled.connect(lambda _c: (self._sync_enabled(), self.changed.emit()))
        layout.addWidget(self.enable)

        self.metric = QComboBox()
        self.metric.setMinimumWidth(22 * em)
        self.metric.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for m in metric_defs:
            self.metric.addItem(f"{m.group}: {m.label}", userData=m.id)
            self.metric.setItemData(self.metric.count() - 1, m.description, Qt.ItemDataRole.ToolTipRole)
        self.metric.currentIndexChanged.connect(self._on_metric_changed)
        layout.addWidget(self.metric)

        layout.addWidget(QLabel("≥"))
        self.low = QLineEdit()
        self.low.setPlaceholderText("any")
        self.low.setFixedWidth(6 * em)
        self.low.setValidator(QDoubleValidator())
        self.low.textChanged.connect(lambda _t: self.changed.emit())
        layout.addWidget(self.low)

        layout.addWidget(QLabel("≤"))
        self.high = QLineEdit()
        self.high.setPlaceholderText("any")
        self.high.setFixedWidth(6 * em)
        self.high.setValidator(QDoubleValidator())
        self.high.textChanged.connect(lambda _t: self.changed.emit())
        layout.addWidget(self.high)

        self.unit = QLabel("")
        self.unit.setFixedWidth(4 * em)
        self.unit.setStyleSheet("color: #78909C;")
        layout.addWidget(self.unit)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color: #90A4AE; font-size: 11px;")
        self.hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.hint)

        self.remove_btn = QPushButton("✕ Remove")
        self.remove_btn.setMinimumWidth(8 * em)
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
        m = metric_def_for(mid)
        self.unit.setText(m.unit if m else "")
        # Extracted features are machine-named; their description is the only
        # place the raw column name is visible, so keep it one hover away.
        self.metric.setToolTip(m.description if m else "")
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
    _needs_progress_sig = Signal(str, int, int)

    def __init__(
        self,
        project_root: Path,
        exemplar_provider: Callable[[], list[ClipRef]],
        scope_label: str,
        on_apply: Callable[[list[ClipRef], dict], None],
        parent: QWidget | None = None,
        scope_sessions: set[str] | None = None,
        on_flag_queue: Callable[[list[Criterion], bool], None] | None = None,
        reviewed_provider: Callable[[], set[str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Targeted Clip Mining")
        self.setMinimumSize(720, 560)
        self._project_root = project_root
        self._exemplar_provider = exemplar_provider
        self._on_apply = on_apply
        self._on_flag_queue = on_flag_queue
        self._reviewed_provider = reviewed_provider
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
            self._feature_display[f"{m.group}, {m.label}"] = m.id
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
        # Contrastive essence state: a background sample the exemplars are compared
        # against, plus the graded exemplar-likeness ranker built from the last
        # extraction (orders matches so "load top N" gets the best first).
        self._bg_df: pd.DataFrame | None = None
        self._bg_key: frozenset | None = None  # sessions the cached background covers
        self._bg_ids: list[str] = []           # windows behind the cached background
        self._essence_scorer = None
        self._rank_scores: pd.Series | None = None
        # Ranked mode can only be armed once a ranker has something to grade, and
        # both essence and the Auto Hunter presets ask for it *before* the pool is
        # scored. Remember the intent and honor it the moment ranking is possible,
        # so a preset hunt doesn't silently fall back to the hard criteria box.
        self._prefer_ranked = False
        # What the current ranker was built from, for the match-count wording:
        # a preset hunt isn't ranking by similarity to clips the user picked.
        self._rank_source = "similarity to your clips"
        # Extracted-feature ids the essence has committed to (criteria or ranker),
        # so their columns can be joined onto the scored pool before mining.
        self._rich_needed: list[str] = []
        # Matches dropped from the last count because they already carry a review
        # decision: reported in the count label so the shrink is never silent.
        self._skipped_reviewed = 0
        # Coverage-gap hunt: when on, matches are kept only from the sessions of
        # subjects still short of examples of the chosen behavior (None = off).
        self._coverage = BehaviorCoverageService(project_root)
        self._coverage_report: CoverageReport | None = None
        self._gap_sessions: set[str] | None = None
        self._skipped_gap = 0
        self._gap_behavior_picked = False  # the user chose one; stop auto-defaulting

        root = QVBoxLayout(self)
        root.setSpacing(10)

        intro = QLabel(
            "Find clips by what the animal is doing. Build the definition yourself "
            "under Criteria, learn it from exemplar clips with Extract essence, or "
            "start from a ready-made hunt under Auto Hunter."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8FA6B4;")
        root.addWidget(intro)

        scope = QLabel(f"Scope: {scope_label}")
        scope.setStyleSheet("font-weight: 600;")
        root.addWidget(scope)

        # Two ways in, one pipeline out: both tabs write the same criteria rows and
        # ranker, and share the batch options / Find matches / Load row below them.
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        criteria_tab = QWidget()
        crit = QVBoxLayout(criteria_tab)
        crit.setSpacing(10)
        self._tabs.addTab(criteria_tab, "Criteria")

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
        crit.addWidget(scroll, 1)

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
        self._match_ranked = QRadioButton("Ranked")
        self._match_ranked.setToolTip(
            "Rank every segment by how exemplar-like it is and load the best ones,\n"
            "instead of keeping only those inside the criteria ranges. The criteria\n"
            "below still show you what distinguishes the behavior, they just stop\n"
            "acting as a hard cut-off, which on real projects threw away most of the\n"
            "behavior's true instances. Available once you extract an essence."
        )
        self._match_all.setChecked(True)
        self._match_ranked.setEnabled(False)  # needs a fitted essence ranker
        grp = QButtonGroup(self)
        grp.addButton(self._match_all)
        grp.addButton(self._match_any)
        grp.addButton(self._match_ranked)
        self._match_all.toggled.connect(lambda _c: self._update_count())
        self._match_ranked.toggled.connect(lambda _c: self._update_count())
        # ``clicked`` fires only for a real user choice, so picking a hard filter
        # by hand permanently overrides the "prefer ranked" intent an essence or a
        # preset left behind: otherwise the next re-grade would snap it back.
        for btn in (self._match_all, self._match_any):
            btn.clicked.connect(self._on_filter_mode_chosen)
        self._match_ranked.clicked.connect(lambda: setattr(self, "_prefer_ranked", True))
        ctrl.addWidget(self._match_all)
        ctrl.addWidget(self._match_any)
        ctrl.addWidget(self._match_ranked)
        ctrl.addStretch(1)
        crit.addLayout(ctrl)

        # Essence row.
        ess = QHBoxLayout()
        self._essence_btn = QPushButton("⤢ Extract Essence from Selected Clips")
        self._essence_btn.setToolTip(
            "Find what makes the selected review clips *different from the rest of the\n"
            "pool* and add the most distinguishing features as criteria automatically,\n"
            "you don't pick the features, it does. It searches the zone/motion metrics\n"
            "above AND this project's extracted per-window features (the same ones the\n"
            "classifier learns from, including rhythm, rotation and jerk), so it can lock\n"
            "onto things no hand-set criterion can express. Matches are then ranked by\n"
            "how exemplar-like they are, so loading the top N gets the best first.\n"
            "This window stays open: change your selection behind it, then click again."
        )
        self._essence_btn.clicked.connect(self._extract_essence)
        ess.addWidget(self._essence_btn)
        ess.addWidget(QLabel("top features:"))
        self._essence_topk = QSpinBox()
        self._essence_topk.setRange(1, 20)
        self._essence_topk.setValue(5)
        self._essence_topk.setToolTip(
            "How many of the most *distinguishing* features to add as criteria,\n"
            "the ones that best separate the selected clips from the rest of the\n"
            "pool. More features means a tighter, more specific definition."
        )
        ess.addWidget(self._essence_topk)
        ess.addWidget(QLabel("breadth:"))
        self._essence_breadth = QComboBox()
        # Recall target for the contrastive box: how much of the selected clips the
        # criteria must keep. Lower = tighter/more precise, higher = broader.
        self._essence_breadth.addItem("Balanced", 0.80)
        self._essence_breadth.addItem("Broad (more recall)", 0.90)
        self._essence_breadth.addItem("Tight (more precise)", 0.65)
        self._essence_breadth.setToolTip(
            "How much of your selected clips the extracted criteria must keep.\n"
            "Broad keeps almost all of them (surfaces more, looser definition);\n"
            "Tight drops outlier clips for a sharper, more specific definition\n"
            "that matches fewer of the pool. Re-run Extract essence after changing."
        )
        ess.addWidget(self._essence_breadth)
        ess.addStretch(1)
        crit.addLayout(ess)

        # Flag-in-queue row: audit the *current review filter* against these
        # ranges and highlight the clips that fall outside them.  Only shown when
        # the host wired a handler (i.e. opened from the review tab).
        if self._on_flag_queue is not None:
            flag_row = QHBoxLayout()
            self._flag_queue_btn = QPushButton("Flag Failing Clips in Review Queue")
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
            crit.addLayout(flag_row)

        self._tabs.addTab(self._build_auto_hunter_tab(), "Auto Hunter")
        self._tabs.addTab(self._build_label_needs_tab(), "Label Needs")

        # What comes back: skip work already done, and spread it over the animals.
        opts = QHBoxLayout()
        self._skip_reviewed_chk = QCheckBox("Skip clips I've already reviewed")
        self._skip_reviewed_chk.setChecked(True)
        self._skip_reviewed_chk.setToolTip(
            "Leave out segments that already have a review decision, so mining only\n"
            "ever hands you clips you haven't judged yet. Untick to re-surface\n"
            "reviewed clips (e.g. to audit them against a tightened definition)."
        )
        self._skip_reviewed_chk.toggled.connect(lambda _c: self._update_count())
        opts.addWidget(self._skip_reviewed_chk)
        opts.addSpacing(16)
        self._balance_subjects_chk = QCheckBox("Spread evenly across subjects")
        self._balance_subjects_chk.setChecked(True)
        self._balance_subjects_chk.setToolTip(
            "Fill the batch by taking turns between subjects, each animal's\n"
            "best-scoring matches first: instead of loading whichever animal happens\n"
            "to score highest. Subjects with fewer matches give their slots back to\n"
            "the others, so you never get fewer clips than the cap allows."
        )
        opts.addWidget(self._balance_subjects_chk)
        opts.addStretch(1)
        root.addLayout(opts)
        self._build_gap_row(root)

        # Mine trigger + progress.
        mine_row = QHBoxLayout()
        self._mine_btn = QPushButton("Find Matches")
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
            "criteria can't decode thousands of clips by accident. The batch is\n"
            "filled with the highest-scoring matches, taken in turn from each\n"
            "subject while 'Spread evenly across subjects' is ticked."
        )
        self._cap_spin.valueChanged.connect(lambda _v: self._update_count())
        actions.addWidget(self._cap_spin)
        actions.addStretch(1)
        self._apply_btn = QPushButton("Load Matches into Review Queue")
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
        self._needs_progress_sig.connect(self._on_needs_progress)
        # Restore the project's saved criteria, or seed one row to start from, no
        # scoring happens until Find matches is clicked.
        self._restore_or_seed()
        self.refresh_exemplar_count()
        self._update_count()

    # -- coverage gaps -------------------------------------------------------

    def _build_gap_row(self, root: QVBoxLayout) -> None:
        """Toggle that aims the batch at subjects lacking examples of a behavior."""
        gap = QHBoxLayout()
        self._gap_chk = QCheckBox("Only subjects lacking examples of")
        self._gap_chk.setToolTip(
            "Keep only matches from subjects that have fewer than this many reviewed\n"
            "examples of the chosen behavior, so the batch goes to the animals the\n"
            "model has not seen do it yet. Counted from your own review labels, the\n"
            "same way Leave-one-mouse-out counts mice.\n"
            f"A subject with no examples whose last {DEFAULT_ABSENT_AFTER} gap-mined clips were\n"
            "all reviewed without showing the behavior is treated as likely absent\n"
            "and skipped, since some animals never do it."
        )
        gap.addWidget(self._gap_chk)
        self._gap_behavior = QComboBox()
        self._gap_behavior.setToolTip("The behavior to find more subjects for.")
        self._behavior_names: dict[str, str] = {}
        try:
            beh = BehaviorService()
            beh.set_project(self._project_root)
            for b in beh.behaviors:
                bid = str(b.behavior_id)
                if bid == NO_BEHAVIOR_ID:
                    continue
                self._behavior_names[bid] = str(b.name)
                self._gap_behavior.addItem(str(b.name), bid)
        except Exception:
            logger.debug("Clip mining: behaviors unavailable", exc_info=True)
        gap.addWidget(self._gap_behavior)
        gap.addWidget(QLabel("with fewer than"))
        self._gap_min = QSpinBox()
        self._gap_min.setRange(1, 1000)
        self._gap_min.setValue(DEFAULT_MIN_POSITIVES)
        self._gap_min.setToolTip(
            "A subject counts as covered once it has this many reviewed examples."
        )
        gap.addWidget(self._gap_min)
        gap.addWidget(QLabel("examples"))
        gap.addStretch(1)
        root.addLayout(gap)
        self._gap_label = QLabel("")
        self._gap_label.setWordWrap(True)
        self._gap_label.setStyleSheet("color: #8FA6B4;")
        self._gap_label.setVisible(False)
        root.addWidget(self._gap_label)

        if self._gap_behavior.count() == 0:
            self._gap_chk.setEnabled(False)
            self._gap_chk.setToolTip("Define behaviors in this project first.")
        self._gap_chk.toggled.connect(self._on_gap_toggled)
        self._gap_behavior.activated.connect(self._on_gap_behavior_picked)
        self._gap_behavior.currentIndexChanged.connect(lambda _i: self._update_count())
        self._gap_min.valueChanged.connect(lambda _v: self._update_count())

    def _on_gap_toggled(self, on: bool) -> None:
        if on and not self._gap_behavior_picked:
            # Default to what the highlighted exemplars are labeled: the usual
            # "find more of these, from other animals" case.
            try:
                exemplars = self._exemplar_provider() or []
            except Exception:
                exemplars = []
            bid = self._coverage.dominant_behavior([c.window_id for c in exemplars])
            idx = self._gap_behavior.findData(bid) if bid else -1
            if idx >= 0:
                self._gap_behavior.setCurrentIndex(idx)
        self._update_count()

    def _on_gap_behavior_picked(self, _index: int) -> None:
        self._gap_behavior_picked = True

    def _gap_behavior_id(self) -> str | None:
        bid = self._gap_behavior.currentData()
        return str(bid) if bid else None

    def _refresh_coverage(self) -> None:
        """Re-read coverage (live: labels change behind this modeless window)."""
        bid = self._gap_behavior_id()
        if not self._gap_chk.isChecked() or bid is None:
            self._coverage_report = None
            self._gap_sessions = None
            self._gap_label.setVisible(False)
            return
        rep = self._coverage.report(bid, int(self._gap_min.value()), DEFAULT_ABSENT_AFTER)
        self._coverage_report = rep
        self._gap_sessions = rep.gap_sessions()
        name = self._behavior_names.get(bid, bid)
        k = rep.min_positives
        gaps = rep.gap_subjects()
        absent = rep.likely_absent()
        if not rep.subjects:
            text = "No subjects found in this project's sessions."
        elif not gaps:
            text = f"Every subject has at least {k} examples of {name}: no gaps to fill."
        else:
            text = (
                f"{name}: {len(gaps)} of {len(rep.subjects)} subjects have fewer "
                f"than {k} examples."
            )
        if absent:
            text += (
                f" {len(absent)} skipped as likely absent (no examples, and "
                f"{DEFAULT_ABSENT_AFTER}+ gap-mined clips reviewed without one)."
            )
        self._gap_label.setText(text)
        lines = []
        for subj in sorted(rep.subjects, key=lambda x: (rep.subjects[x].positives, x)):
            if subj not in gaps and subj not in absent:
                continue
            cov = rep.subjects[subj]
            line = f"{subj}: {cov.positives} example(s)"
            if cov.screened:
                line += f", {cov.screened} gap clip(s) reviewed"
            if subj in absent:
                line += " (likely absent)"
            lines.append(line)
        self._gap_label.setToolTip("\n".join(lines))
        self._gap_label.setVisible(True)

    def _gap_note(self) -> str:
        """Count-label suffix saying the batch is limited to gap subjects."""
        if self._gap_sessions is None or self._coverage_report is None:
            return ""
        n = len(self._coverage_report.gap_subjects())
        left_out = (
            f" ({self._skipped_gap} match(es) from covered subjects left out)"
            if self._skipped_gap else ""
        )
        return f" Limited to {n} subject(s) lacking examples{left_out}."

    # -- auto hunter (preset hunts) ------------------------------------------

    def _build_auto_hunter_tab(self) -> QWidget:
        """Ready-made hunts for projects with nothing reviewed to learn from.

        Essence needs exemplars; a preset is what you use before you have any.
        Picking one writes its ranker and its (editable) ranges straight into the
        Criteria tab, so a preset is a *starting point* the user can inspect and
        change rather than a black box.
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(10)

        blurb = QLabel(
            "Start a hunt without any reviewed clips to learn from. Each preset is "
            "a ready-made essence over the features ABEL already extracted, tuned "
            "on this project's own spread: pick one, then click Find matches."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #8FA6B4;")
        lay.addWidget(blurb)

        body = QHBoxLayout()
        self._preset_list = QListWidget()
        # Measured, not hard-coded: the list must hold the longest preset name at
        # whatever display scaling Windows is running.
        em = self.fontMetrics().horizontalAdvance("M")
        self._preset_list.setMinimumWidth(16 * em)
        self._preset_list.setMaximumWidth(26 * em)
        for p in PRESETS:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p.id)
            item.setToolTip(p.tagline)
            self._preset_list.addItem(item)
        self._preset_list.currentItemChanged.connect(
            lambda _cur, _prev: self._on_preset_selected()
        )
        body.addWidget(self._preset_list)

        self._preset_detail = QTextBrowser()
        self._preset_detail.setOpenExternalLinks(False)
        body.addWidget(self._preset_detail, 1)
        lay.addLayout(body, 1)

        arm = QHBoxLayout()
        self._preset_btn = QPushButton("▶ Use This Preset")
        self._preset_btn.setToolTip(
            "Load this preset's ranker and its (editable) ranges into the Criteria\n"
            "tab, then click Find matches. Nothing is hidden, every feature it uses\n"
            "appears as a criterion row you can adjust or delete."
        )
        self._preset_btn.setStyleSheet(
            "background-color: #4527A0; color: white; font-weight: 600; padding: 6px 12px;"
        )
        self._preset_btn.clicked.connect(self._arm_preset)
        arm.addWidget(self._preset_btn)
        arm.addSpacing(16)
        arm.addWidget(QLabel("breadth:"))
        self._preset_breadth = QComboBox()
        for label, q in BREADTH_LEVELS:
            self._preset_breadth.addItem(label, q)
        self._preset_breadth.setToolTip(
            "How wide the preset draws its ranges, as a share of this project's\n"
            "segments. Broad describes the behavior loosely; Tight draws a sharper\n"
            "definition. The hunt itself ranks rather than filters, so this mostly\n"
            "changes what the criteria rows say: unless you switch to All/Any."
        )
        arm.addWidget(self._preset_breadth)
        arm.addStretch(1)
        lay.addLayout(arm)

        self._preset_status = QLabel("")
        self._preset_status.setWordWrap(True)
        self._preset_status.setStyleSheet("color: #00695C; font-weight: 600;")
        lay.addWidget(self._preset_status)

        if PRESETS:
            self._preset_list.setCurrentRow(0)
        return tab

    # -- label needs --------------------------------------------------------

    _NEEDS_COLUMNS = (
        "Behavior", "Verdict", "PR-AUC", "Pos", "Sessions",
        "+Pos", "+Spread", "+Hard neg", "+No Beh",
        "What to label next",
    )
    _NEEDS_TIPS = {
        "PR-AUC": "Held-out PR-AUC with all current labels (sessions split into folds).",
        "Pos": "Labeled positive windows.",
        "Sessions": "Sessions with at least one positive, of all labeled sessions.",
        "+Pos": "How much PR-AUC the second half of the positives adds.\n"
                            "Still large = the model is still learning from more examples.",
        "+Spread": "Half the positives drawn across many sessions minus half drawn\n"
                                  "from a few whole sessions. Large = examples from new animals help.",
        "+Hard neg": "PR-AUC lost when the look-alike behaviors' labels are removed.\n"
                            "Large = correcting look-alike clips is worth your time.",
        "+No Beh": "How much PR-AUC the second half of the No Behavior negatives adds.",
    }

    def _build_label_needs_tab(self) -> QWidget:
        """Per-behavior ablation: which kind of new label would improve each model."""
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)
        blurb = QLabel(
            "Find out what each model needs before you spend review time. For every "
            "chosen behavior, models are retrained on held-out session folds with part "
            "of one kind of label removed (positives, positives from fewer animals, "
            "look-alike negatives, No Behavior negatives). The accuracy each removal "
            "costs is what that kind of label is worth. Gains under "
            f"{MEANINGFUL_GAIN:.2f} are treated as run-to-run noise. Nothing in the project is changed."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #8FA6B4;")
        lay.addWidget(blurb)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Behaviors to test:"))
        self._needs_list = QListWidget()
        em = self.fontMetrics().horizontalAdvance("M")
        self._needs_list.setMinimumWidth(12 * em)
        self._needs_list.setMaximumWidth(16 * em)
        try:
            beh = BehaviorService()
            beh.set_project(self._project_root)
            for b in beh.behaviors:
                if str(b.behavior_id) == NO_BEHAVIOR_ID:
                    continue
                item = QListWidgetItem(str(b.name))
                item.setData(Qt.ItemDataRole.UserRole, str(b.behavior_id))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self._needs_list.addItem(item)
        except Exception:
            logger.debug("Label needs: behaviors unavailable", exc_info=True)
        left.addWidget(self._needs_list, 1)
        sel = QHBoxLayout()
        for text, state in (("All", Qt.CheckState.Checked), ("None", Qt.CheckState.Unchecked)):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _c=False, st=state: self._set_needs_checks(st))
            sel.addWidget(btn)
        left.addLayout(sel)
        body.addLayout(left)

        right = QVBoxLayout()
        self._needs_table = QTableWidget(0, len(self._NEEDS_COLUMNS))
        self._needs_table.setHorizontalHeaderLabels(list(self._NEEDS_COLUMNS))
        for i, name in enumerate(self._NEEDS_COLUMNS):
            tip = self._NEEDS_TIPS.get(name)
            if tip:
                self._needs_table.horizontalHeaderItem(i).setToolTip(tip)
        hdr = self._needs_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(len(self._NEEDS_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self._needs_table.setWordWrap(True)
        # Wrapped action text: row heights must follow the final column widths.
        hdr.sectionResized.connect(lambda *_: self._needs_table.resizeRowsToContents())
        self._needs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._needs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._needs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        right.addWidget(self._needs_table, 1)
        body.addLayout(right, 1)
        lay.addLayout(body, 1)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Folds:"))
        self._needs_folds = QSpinBox()
        self._needs_folds.setRange(3, 10)
        self._needs_folds.setValue(5)
        self._needs_folds.setToolTip("Session-grouped folds. Every held-out fold contains whole sessions.")
        opts.addWidget(self._needs_folds)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Repeats:"))
        self._needs_repeats = QSpinBox()
        self._needs_repeats.setRange(1, 5)
        self._needs_repeats.setValue(2)
        self._needs_repeats.setToolTip(
            "Random draws per removal. More repeats = less noise, proportionally longer run."
        )
        opts.addWidget(self._needs_repeats)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Trees:"))
        self._needs_trees = QSpinBox()
        self._needs_trees.setRange(50, 500)
        self._needs_trees.setSingleStep(50)
        self._needs_trees.setValue(150)
        self._needs_trees.setToolTip("XGBoost trees per test model. Fewer = faster, slightly noisier.")
        opts.addWidget(self._needs_trees)
        opts.addStretch(1)
        lay.addLayout(opts)

        run = QHBoxLayout()
        self._needs_run_btn = QPushButton("Run Label Needs Test")
        self._needs_run_btn.setStyleSheet(
            "background-color: #4527A0; color: white; font-weight: 600; padding: 6px 12px;"
        )
        self._needs_run_btn.clicked.connect(self._run_label_needs)
        run.addWidget(self._needs_run_btn)
        self._needs_cancel_btn = QPushButton("Cancel")
        self._needs_cancel_btn.setEnabled(False)
        self._needs_cancel_btn.clicked.connect(self._cancel_label_needs)
        run.addWidget(self._needs_cancel_btn)
        self._needs_progress = QProgressBar()
        self._needs_progress.setVisible(False)
        run.addWidget(self._needs_progress, 1)
        self._needs_gap_btn = QPushButton("Hunt Coverage Gaps for Selected")
        self._needs_gap_btn.setToolTip(
            "Turn on 'Only subjects lacking examples of' for the selected behavior and go\n"
            "to Criteria, so the next Find Matches only returns animals without examples."
        )
        self._needs_gap_btn.setEnabled(False)
        self._needs_gap_btn.clicked.connect(self._needs_hunt_gaps)
        run.addWidget(self._needs_gap_btn)
        lay.addLayout(run)
        self._needs_table.itemSelectionChanged.connect(
            lambda: self._needs_gap_btn.setEnabled(bool(self._needs_table.selectedItems()))
        )

        self._needs_status = QLabel("")
        self._needs_status.setWordWrap(True)
        self._needs_status.setStyleSheet("color: #00695C; font-weight: 600;")
        lay.addWidget(self._needs_status)
        self._needs_cancel: list[bool] = [False]
        self._needs_busy = False
        return tab

    def _set_needs_checks(self, state) -> None:
        for i in range(self._needs_list.count()):
            self._needs_list.item(i).setCheckState(state)

    def _run_label_needs(self) -> None:
        from PySide6.QtCore import QThreadPool

        if self._needs_busy:
            return
        chosen = []
        for i in range(self._needs_list.count()):
            item = self._needs_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append((str(item.data(Qt.ItemDataRole.UserRole)), item.text()))
        if not chosen:
            self._needs_status.setText("Tick at least one behavior to test.")
            return
        cfg = LabelNeedsConfig(
            folds=int(self._needs_folds.value()),
            repeats=int(self._needs_repeats.value()),
            n_estimators=int(self._needs_trees.value()),
        )
        # Confusers are named among every behavior, not just the tested ones.
        all_behaviors = [
            (str(self._needs_list.item(i).data(Qt.ItemDataRole.UserRole)), self._needs_list.item(i).text())
            for i in range(self._needs_list.count())
        ]
        self._needs_busy = True
        self._needs_cancel = [False]
        self._needs_run_btn.setEnabled(False)
        self._needs_cancel_btn.setEnabled(True)
        self._needs_progress.setVisible(True)
        self._needs_progress.setRange(0, 0)
        self._needs_status.setText(f"Testing {len(chosen)} behaviors…")
        self._needs_started = time.monotonic()
        worker = TaskWorker(self._label_needs_job, chosen, all_behaviors, cfg, self._needs_cancel)
        worker.signals.finished.connect(self._on_label_needs_ready)
        worker.signals.failed.connect(self._on_label_needs_failed)
        QThreadPool.globalInstance().start(worker)

    def _label_needs_job(self, chosen, all_behaviors, cfg, cancel_flag) -> list:
        """Worker-thread half: no widget access in here."""
        names = dict(all_behaviors)
        tested = [(bid, names.get(bid, name)) for bid, name in chosen]
        # Pass every behavior so confusers can be named; only the chosen ones run.
        return analyze_label_needs(
            self._project_root, tested, cfg,
            progress_cb=lambda name, done, total: self._needs_progress_sig.emit(name, done, total),
            cancel_flag=cancel_flag, behavior_names=names,
        )

    def _on_needs_progress(self, name: str, done: int, total: int) -> None:
        self._needs_progress.setRange(0, total)
        self._needs_progress.setValue(done)
        eta = ""
        if done >= 3:
            left = (time.monotonic() - self._needs_started) / done * (total - done)
            eta = f", about {max(1, round(left / 60))} min left"
        self._needs_progress.setFormat(f"{name}: %p%{eta}")

    def _cancel_label_needs(self) -> None:
        self._needs_cancel[0] = True
        self._needs_status.setText("Stopping after the current model…")

    def _label_needs_done(self) -> None:
        self._needs_busy = False
        self._needs_run_btn.setEnabled(True)
        self._needs_cancel_btn.setEnabled(False)
        self._needs_progress.setVisible(False)

    def _on_label_needs_failed(self, tb: str) -> None:
        self._label_needs_done()
        if CANCEL_MARKER in tb:
            self._needs_status.setText("Label needs test cancelled.")
            return
        logger.error("Label needs test failed:\n%s", tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        self._needs_status.setText(f"Label needs test failed: {last}")

    def _on_label_needs_ready(self, results: list) -> None:
        self._label_needs_done()

        def fmt(v) -> str:
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"

        colors = {"Label more": "#EF6C00", "Flat": "#C62828", "Saturated": "#2E7D32", "Too few": "#6D4C41"}
        self._needs_table.setRowCount(len(results))
        for r, n in enumerate(results):
            cells = [
                n.behavior_name, n.verdict,
                "" if n.prauc_full is None else f"{n.prauc_full:.3f}",
                str(n.positives), f"{n.sessions_with_positives}/{n.sessions_total}",
                fmt(n.gain_positives), fmt(n.gain_spread), fmt(n.gain_hard), fmt(n.gain_no_behavior),
                "\n".join(n.actions),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, n.behavior_id)
                if c == 1 and n.verdict in colors:
                    item.setForeground(QColor(colors[n.verdict]))
                self._needs_table.setItem(r, c, item)
        self._needs_table.resizeRowsToContents()
        # The stretched action column settles after layout, and resizing it emits no signal.
        QTimer.singleShot(0, self._needs_table.resizeRowsToContents)
        saved = self._save_label_needs(results)
        self._needs_status.setText(
            f"Done: {len(results)} behaviors." + (f" Saved to {saved}." if saved else "")
        )

    def _save_label_needs(self, results: list) -> str | None:
        from datetime import datetime

        try:
            out_dir = Path(self._project_root) / "derived" / "label_needs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"label_needs_{datetime.now():%Y%m%d_%H%M%S}.csv"
            pd.DataFrame([n.as_row() for n in results]).to_csv(path, index=False)
            return str(path.relative_to(self._project_root))
        except Exception:
            logger.warning("Label needs: could not save results", exc_info=True)
            return None

    def _needs_hunt_gaps(self) -> None:
        rows = self._needs_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self._needs_table.item(rows[0].row(), 0)
        bid = str(item.data(Qt.ItemDataRole.UserRole) or "")
        idx = self._gap_behavior.findData(bid)
        if idx < 0 or not self._gap_chk.isEnabled():
            self._needs_status.setText(f"{item.text()} is not available for coverage-gap hunting.")
            return
        self._gap_behavior.setCurrentIndex(idx)
        self._gap_behavior_picked = True
        self._gap_chk.setChecked(True)
        self._tabs.setCurrentIndex(0)
        self._count_label.setText(
            f"Coverage-gap hunt set to {item.text()}. Add criteria, extract an essence or pick "
            "an Auto Hunter preset, then Find Matches."
        )

    def _on_filter_mode_chosen(self) -> None:
        self._prefer_ranked = False

    def _selected_preset(self):
        item = self._preset_list.currentItem()
        return preset_by_id(str(item.data(Qt.ItemDataRole.UserRole))) if item else None

    def _on_preset_selected(self) -> None:
        p = self._selected_preset()
        if p is None:
            self._preset_detail.setPlainText("")
            return
        rows = "".join(
            f"<li>{f.note or f.columns[0]}</li>" for f in p.features if f.note
        )
        body = "".join(
            f"<p>{chunk}</p>" for chunk in p.description.split("\n\n") if chunk.strip()
        )
        self._preset_detail.setHtml(
            f"<h3 style='margin-bottom:2px;'>{p.name}</h3>"
            f"<p style='color:#8FA6B4;margin-top:0;'><i>{p.tagline}</i></p>"
            f"{body}"
            f"<p><b>Needs:</b> {p.requires}</p>"
            f"<p><b>What it measures:</b></p><ul>{rows}</ul>"
        )
        self._preset_status.setText("")

    def _arm_preset(self) -> None:
        """Install the selected preset's ranker and ranges into the Criteria tab."""
        if self._mining or self._essence_busy:
            self._preset_status.setText(
                "Wait for the current scoring pass to finish, then pick a preset."
            )
            return
        preset = self._selected_preset()
        if preset is None:
            return
        breadth = float(self._preset_breadth.currentData())
        try:
            fit = fit_preset(self._metrics, preset, breadth)
        except Exception as exc:  # pragma: no cover - unreadable/corrupt table
            self._preset_status.setText(f"Could not arm this preset: {exc}")
            return
        if not fit.usable():
            self._preset_status.setText(fit.note)
            return

        self._essence_scorer = fit.scorer
        self._rank_scores = None
        self._rank_source = f"the “{preset.name}” preset"
        self._register_metrics(fit.used)
        self._rich_needed = [m for m in fit.used if is_rich_metric(m)]
        self._suspend_updates = True
        try:
            self._clear_rows()
            for c in fit.criteria:
                row = self._add_row(c.metric_id)
                row.set_range(c.low, c.high)
        finally:
            self._suspend_updates = False
        # A preset is a ranker first: the ranges describe the family, and gating on
        # them would drop most of its real instances (the same measurement that put
        # essence into Ranked mode).
        self._prefer_ranked = True
        self._ensure_rich_columns()
        self._refresh_rank_scores()
        self._update_count()
        self._preset_status.setText(
            fit.note
            + (
                " Ranked by preset similarity: click Find matches."
                if self._df is None
                else " Ranked by preset similarity; see the match count below."
            )
        )

    # -- metric computation (deferred until the user mines) ------------------

    def _feature_only_criteria(self) -> bool:
        """True when nothing in play needs pose.

        Every active bound and every ranker feature is an extracted feature, so the
        pool can be read straight from the feature table instead of recomputing
        pose metrics for all 40-odd thousand segments, the usual state right after
        Extract essence, and the reason mining no longer needs the pose drive.
        """
        ids = [
            c.metric_id for c in self._current_criteria()
            if c.enabled and (c.low is not None or c.high is not None)
        ]
        ids += list(getattr(self._essence_scorer, "active_feature_ids", []) or [])
        return bool(ids) and all(is_rich_metric(m) for m in ids)

    def _needs_pose_scoring(self) -> bool:
        """True when an active criterion has no column in the cached pool table.

        Happens when the pool was read from the feature table (no pose pass) and
        the user then adds a hand-set geometry criterion, Find matches must do the
        real scoring pass rather than silently ignoring the new bound.
        """
        if self._df is None:
            return True
        return any(
            c.metric_id not in self._df.columns and not is_rich_metric(c.metric_id)
            for c in self._current_criteria()
            if c.enabled and (c.low is not None or c.high is not None)
        )

    def _mine(self) -> None:
        """Score the full pool (first time) then count matches for the criteria."""
        # Essence runs in a worker that shares the pool/background caches, so the
        # two jobs never overlap.
        if self._mining or self._essence_busy:
            return
        if self._scored and not self._needs_pose_scoring():
            # Metrics already cached: criteria edits just re-filter.
            self._update_count()
            return
        from PySide6.QtCore import QThreadPool

        if not self._clip_by_id:
            clips = self._metrics.load_segment_pool(self._scope_sessions)
            self._clip_by_id = {c.window_id: c for c in clips}
        clips = list(self._clip_by_id.values())
        if not clips:
            self._count_label.setText(
                "No scored segments found: run Feature Extraction first."
            )
            return

        # Pure extracted-feature criteria (the usual case straight after Extract
        # essence) are already computed per window, so the pool is read from the
        # feature table: no pose pass, and no dependency on the pose drive.
        if self._feature_only_criteria():
            self._start_feature_pool_scan(clips)
            return

        # Scoring re-reads each session's raw pose; if it's gone every clip
        # scores all-NaN and no criterion can match. Warn, and only abort if
        # *every* session is unreadable (a partial pool can still be mined).
        missing = self._metrics.unresolved_pose_clips(clips)
        if missing:
            self._warn_missing_pose(missing, "in this project")
            if len(missing) >= len({c.session_id for c in clips}):
                self._count_label.setText(
                    "Couldn't read pose for any session: see the message above. "
                    "Extract essence still works: it can range over this project's "
                    "extracted features, which need no pose file."
                )
                return

        self._mining = True
        self._mine_btn.setEnabled(False)
        self.refresh_exemplar_count()
        self._progress.setVisible(True)
        self._progress.setRange(0, len(clips))
        self._progress.setValue(0)
        self._progress.setFormat("Scoring segments… %p%")
        worker = TaskWorker(self._metrics.compute, clips, progress_callback=self._emit_progress)
        worker.signals.finished.connect(self._on_metrics_ready)
        worker.signals.failed.connect(self._on_metrics_failed)
        QThreadPool.globalInstance().start(worker)

    def _start_feature_pool_scan(self, clips: list[ClipRef]) -> None:
        """Read the pool's extracted-feature values for the current criteria."""
        from PySide6.QtCore import QThreadPool

        want = self._rich_needed + [c.metric_id for c in self._current_criteria()]
        cols = [m for m in dict.fromkeys(want) if is_rich_metric(m)]
        self._mining = True
        self._mine_btn.setEnabled(False)
        self.refresh_exemplar_count()
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # a table read, not a per-clip scan
        self._progress.setFormat("Reading extracted features…")
        worker = TaskWorker(
            self._feature_pool_job, [c.window_id for c in clips], cols
        )
        worker.signals.finished.connect(self._on_metrics_ready)
        worker.signals.failed.connect(self._on_metrics_failed)
        QThreadPool.globalInstance().start(worker)

    def _feature_pool_job(self, window_ids: list[str], cols: list[str]) -> pd.DataFrame:
        """Worker-thread read of the feature table for the pool (no widget access)."""
        df = self._metrics.load_rich_features(metric_ids=cols, segment_ids=set(window_ids))
        return df.reindex([str(w) for w in window_ids])

    def _emit_progress(self, done: int, total: int) -> None:
        # Runs in the worker thread; the queued signal marshals to the UI thread.
        # Guarded because the dialog may be closed/deleted mid-scoring, emitting
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
        self.refresh_exemplar_count()
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat(f"Scored {len(df)} segment(s)")
        # Join on any extracted-feature columns the current essence needs, so the
        # scope hints and the match count cover them too.
        self._ensure_rich_columns()
        # Populate the live scope hints on every existing row.
        for r in self._rows:
            r._stats = self._df
            r._on_metric_changed()
        # Now that the pool is scored, grade it with the last essence ranker (if
        # any) so matches load most-exemplar-like first.
        self._refresh_rank_scores()
        self._update_count()

    def _on_metrics_failed(self, tb: str) -> None:
        self._mining = False
        self._mine_btn.setEnabled(True)
        self.refresh_exemplar_count()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Failed to score segments")
        self._count_label.setText("Metric computation failed: see log.")

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
            else "⤢ Extract Essence from Selected Clips"
        )
        self._essence_btn.setEnabled(n > 0 and not self._essence_busy and not self._mining)

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
            # Empty box (e.g. the returnPressed that trails a completer pick), no-op.
            return
        mid = self._resolve_feature(query)
        if mid is None:
            self._count_label.setText(
                "No feature matches that search: try 'speed', 'zone', 'distance', 'body'…"
            )
            return
        self._add_row(mid)
        self._search.clear()

    def _register_metrics(self, metric_ids: list[str]) -> None:
        """Make extracted-feature ids displayable as criteria rows.

        The project registry only knows the interpretable metrics; essence may
        commit to any extracted feature, so a display definition (humanised label,
        feature family, raw column name in the tooltip) is minted on demand and
        appended to the row list.  It also joins the search index, so a feature the
        essence has surfaced once can be re-added by name afterwards.
        """
        known = {m.id for m in self._metric_defs}
        for mid in metric_ids:
            if mid in known or not is_rich_metric(mid):
                continue
            d = rich_metric_def(mid)
            self._metric_defs.append(d)
            known.add(mid)
            self._feature_display[f"{d.group}, {d.label}"] = d.id
            self._feature_index.append(
                (d.id, f"{d.group} {d.label} {d.description}".lower())
            )

    def _ensure_rich_columns(self) -> None:
        """Join any extracted-feature columns the criteria/ranker need onto the pool.

        The scored pool table holds the interpretable metrics only; essence
        criteria can name extracted features, whose values are read straight from
        the project's feature table for the rows already scored.  Cheap and
        idempotent, once a column is present the lookup is skipped, so this is
        safe to call from the live criteria-count path.
        """
        if self._df is None or self._df.empty:
            return
        want = self._rich_needed + [c.metric_id for c in self._current_criteria()]
        need = [
            m for m in dict.fromkeys(want)
            if is_rich_metric(m) and m not in self._df.columns
        ]
        if not need:
            return
        try:
            self._df = self._metrics.attach_rich_columns(self._df, need)
        except Exception:
            return  # mine() simply ignores criteria whose column is absent
        for r in self._rows:
            r._stats = self._df
        self._refresh_rank_scores()

    def _refresh_rank_scores(self) -> None:
        """Grade the scored pool with the current essence ranker (best-first order)."""
        self._rank_scores = None
        if self._essence_scorer is not None and self._df is not None:
            try:
                self._rank_scores = self._essence_scorer.score(self._df)
            except Exception:
                self._rank_scores = None
        # Ranked mode has nothing to order by without a graded score; don't offer
        # it, and don't strand the user in it if the ranker went away.
        ok = self._rank_scores is not None
        self._match_ranked.setEnabled(ok)
        if not ok and self._match_ranked.isChecked():
            self._match_all.setChecked(True)
        elif ok and self._prefer_ranked and not self._match_ranked.isChecked():
            self._match_ranked.setChecked(True)

    def _restore_or_seed(self) -> None:
        """Rebuild rows from the project's saved criteria, or seed a starter row."""
        criteria, match_all = self._metrics.load_criteria()
        if match_all is not None:
            (self._match_all if match_all else self._match_any).setChecked(True)
        # A saved essence may reference extracted features, register them (and
        # remember them for the pool join) before any row tries to show one.
        self._register_metrics([c.metric_id for c in criteria])
        self._rich_needed = [c.metric_id for c in criteria if is_rich_metric(c.metric_id)]
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
            # Ranked is a *loading* mode, not a criteria mode, the saved ranges are
            # a conjunction either way, so it persists as AND rather than as OR.
            self._metrics.save_criteria(
                self._current_criteria(), not self._match_any.isChecked()
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
        # Neutralize the row's signals so its imminent destruction can't re-enter
        # any slot on the half-torn-down widget.
        row.disarm()
        row.blockSignals(True)
        # The remove button still holds keyboard focus; destroying the row would
        # force a native focus transfer mid-teardown: the access violation the
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

    # -- missing pose ---------------------------------------------------------

    def _warn_missing_pose(self, missing: dict[str, str | None], scope: str) -> None:
        """Explain that the raw pose behind these clips can't be read.

        Without it every metric comes back NaN, which otherwise masquerades as
        "no shared features", so we say so plainly instead of failing silently.
        """
        n = len(missing)
        paths = [p for p in missing.values() if p]
        detail = (
            f"The pose tracking for {n} session{'s' if n != 1 else ''} {scope} "
            "could not be read: the file may have moved or its drive may be "
            "disconnected.\n\n"
            "Re-link those sessions (or reconnect the drive) so their pose can "
            "be loaded, then try again."
        )
        if paths:
            sample = "\n".join(f"  • {p}" for p in paths[:5])
            if len(paths) > 5:
                sample += f"\n  … and {len(paths) - 5} more"
            detail += "\n\nMissing files:\n" + sample
        QMessageBox.warning(self, "Pose data not found", detail)

    # -- essence -------------------------------------------------------------

    def _collect_exemplars(self) -> list[ClipRef]:
        try:
            return list(self._exemplar_provider() or [])
        except Exception:
            return []

    # How many background segments to contrast the exemplars against.
    _BG_SAMPLE = 1000

    def _essence_background(
        self, exemplars: list[ClipRef]
    ) -> tuple["pd.DataFrame | None", list[str]]:
        """``(clip-metric table, window ids)`` the exemplars are contrasted against.

        Prefers the already-scored pool (representative and free).  Before any
        Find matches, it samples other segments from the *same recordings* as the
        exemplars: their pose is already loaded to score the exemplars, so this
        stays quick, and "how do these clips differ from the rest of these
        recordings" is a valid, session-controlled contrast.  The sample is cached
        so repeated extractions don't rescore it.

        The ids are returned alongside the table because the extracted-feature
        half of the contrast is looked up by window id, and it stays usable even
        when the pose-derived table comes back empty (unreadable pose).
        """
        if self._df is not None and not self._df.empty:
            bg = self._df
            if len(bg) > self._BG_SAMPLE:
                bg = bg.sample(self._BG_SAMPLE, random_state=0)
            # Only the pose-derived half: the extracted-feature columns are
            # re-read by window id (the pool table may carry a few of them
            # already, and they must not be contributed twice).
            keep = [c for c in bg.columns if not is_rich_metric(c)]
            ids = [str(w) for w in bg.index]
            return (bg[keep] if keep else None), ids
        exclude = {c.window_id for c in exemplars}
        ex_sessions = frozenset(c.session_id for c in exemplars)
        # Reuse the cached sample only when it covers the *current* selection's
        # sessions: otherwise a changed selection would be contrasted against a
        # stale background and essence would look like it "didn't update".
        if self._bg_key == ex_sessions and self._bg_ids:
            return self._bg_df, self._bg_ids
        if not self._clip_by_id:
            clips = self._metrics.load_segment_pool(self._scope_sessions)
            self._clip_by_id = {c.window_id: c for c in clips}
        pool = [
            c for wid, c in self._clip_by_id.items()
            if wid not in exclude and c.session_id in ex_sessions
        ]
        if not pool:
            self._bg_df, self._bg_key, self._bg_ids = None, None, []
            return None, []
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pool), size=min(self._BG_SAMPLE, len(pool)), replace=False)
        sample = [pool[i] for i in idx]
        try:
            # Serial (max_workers=1): the exemplar sessions are few, so loading
            # each session's pose once in-process beats per-worker reloads.
            self._bg_df = self._metrics.compute(sample, max_workers=1)
        except Exception:
            self._bg_df = None
        self._bg_key = ex_sessions
        self._bg_ids = [c.window_id for c in sample]
        return self._bg_df, self._bg_ids

    def _extract_essence(self) -> None:
        """Discover what makes the selected clips different and add the top features.

        Works straight off the highlighted clips, no full-pool scan required.  The
        search ranges over BOTH metric spaces (see :meth:`_essence_job`), and runs
        in a background worker because reading the extracted-feature table and the
        greedy criteria search together take a second or two.
        """
        if self._essence_busy:
            return
        if self._mining:
            # Essence shares the pool/background caches with the scoring worker.
            # Say so: a click that silently does nothing leaves the previous
            # essence on screen, which reads as "it didn't update".
            self._count_label.setText(
                "Find matches is still scoring the pool: Extract essence will be "
                "available again when it finishes."
            )
            return
        from PySide6.QtCore import QThreadPool

        exemplars = self._collect_exemplars()
        if not exemplars:
            self._count_label.setText(
                "Select one or more clips in the review list first, then Extract essence."
            )
            return
        self._essence_busy = True
        self._essence_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate: the search isn't countable
        self._progress.setFormat("Extracting essence…")
        k = int(self._essence_topk.value())
        recall_target = float(self._essence_breadth.currentData())
        worker = TaskWorker(self._essence_job, exemplars, k, recall_target)
        worker.signals.finished.connect(self._on_essence_ready)
        worker.signals.failed.connect(self._on_essence_failed)
        QThreadPool.globalInstance().start(worker)

    def _essence_job(
        self, exemplars: list[ClipRef], k: int, recall_target: float
    ) -> dict:
        """Worker-thread half of Extract essence: no widget access in here.

        Builds the contrast over both metric spaces at once (interpretable clip
        metrics *and* the project's extracted per-window features), so zone
        geometry and the oscillation/angular family compete on merit for the
        criteria slots.  The pose-derived half is skipped for clips whose pose file
        can't be read, with the extracted features present that is a note, not a
        failure, because they are precomputed and need no pose drive.
        """
        missing = self._metrics.unresolved_pose_clips(exemplars)
        posed = [c for c in exemplars if c.session_id not in missing]
        # Score the exemplars and the background independently: a background
        # failure must not discard the (good) exemplar metrics and abort essence.
        ex_df = None
        if posed:
            try:
                ex_df = self._metrics.compute(posed)
            except Exception:
                ex_df = None
        try:
            bg_df, bg_ids = self._essence_background(exemplars)
        except Exception:
            bg_df, bg_ids = None, []
        frames = self._metrics.essence_frames(
            [c.window_id for c in exemplars], bg_ids, ex_df, bg_df
        )
        notes = list(frames.notes)
        if missing and "pose metrics" not in frames.sources:
            notes.append(
                f"Pose unreadable for {len(missing)} session(s), so the zone/geometry "
                "metrics were left out of this essence."
            )
        if not frames.usable():
            return {"crits": [], "scorer": None, "frames": frames, "notes": notes,
                    "n_ex": 0, "missing": missing}
        crits = self._metrics.extract_similar_essence(
            frames.exemplars, frames.background, k=k, recall_target=recall_target
        )
        try:
            scorer = self._metrics.build_essence_scorer(
                frames.exemplars, frames.background
            )
        except Exception:
            scorer = None
        return {"crits": crits, "scorer": scorer, "frames": frames, "notes": notes,
                "n_ex": int(len(frames.exemplars)), "missing": missing}

    def _on_essence_failed(self, tb: str) -> None:
        self._essence_busy = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Essence extraction failed")
        self.refresh_exemplar_count()
        self._count_label.setText("Essence extraction failed: see log.")

    def _on_essence_ready(self, out: dict) -> None:
        """UI-thread half: install the discovered criteria and report what was used."""
        self._essence_busy = False
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setVisible(self._scored)
        if self._scored:
            self._progress.setFormat(f"Scored {len(self._df)} segment(s)")
        self.refresh_exemplar_count()
        frames = out["frames"]
        note = " ".join(out["notes"])
        if not frames.usable():
            missing = out["missing"]
            if missing:
                self._warn_missing_pose(missing, "in your selection")
            self._count_label.setText(
                ("Could not read features for the selected clips, the criteria "
                 "below are unchanged from before. " + note).strip()
            )
            return
        crits = out["crits"]
        self._essence_scorer = out["scorer"]
        self._rank_source = "similarity to your clips"
        self._rank_scores = None  # regraded below, once the criteria are installed
        # The ranker, not the criteria box, is what essence delivers, a fit with no
        # displayable ranges is still a usable hunt, so only bail when neither exists.
        if not crits and self._essence_scorer is None:
            self._count_label.setText(
                "Couldn't find distinguishing features: pick two or more clips "
                "that are alike (the criteria below are unchanged). " + note
            )
            return
        # Extracted features aren't in the project's metric registry, so register
        # the chosen ones before any row tries to display them.
        chosen = [c.metric_id for c in crits]
        # Only the ranker's non-zero-coefficient features have to be materialised
        # on the pool; the rest were offered to the fit and dropped by it.
        ranker = list(getattr(self._essence_scorer, "active_feature_ids", []) or [])
        self._register_metrics(chosen + ranker)
        self._rich_needed = [m for m in dict.fromkeys(chosen + ranker) if is_rich_metric(m)]
        # Replace the criteria with the discovered distinguishing features.
        self._suspend_updates = True
        try:
            self._clear_rows()
            for c in crits:
                row = self._add_row(c.metric_id)
                row.set_range(c.low, c.high)
        finally:
            self._suspend_updates = False
        labels = ", ".join(metric_label(c.metric_id) for c in crits)
        # Grade the already-scored pool with the new ranker (joining on any
        # feature columns it needs) so matches load most-exemplar-like first.
        self._ensure_rich_columns()
        self._refresh_rank_scores()
        # Essence hunts by ranking, not by the box: the ranges are shown so the user
        # can see what defines the behavior, but gating on them was measured to
        # discard most of its real instances, so switch to Ranked automatically.
        # (The user can still pick All/Any to get the hard filter back.)
        self._prefer_ranked = True
        if self._match_ranked.isEnabled():
            self._match_ranked.setChecked(True)
        # Refresh match count / apply state against the new criteria first. When the
        # pool is already scored that live count IS the "it updated" feedback; when
        # it isn't, keep the informative feature list instead of the generic prompt.
        self._update_count()
        if self._df is None:
            found = (
                f"Added {len(crits)} distinguishing feature(s) from {out['n_ex']} "
                f"clip(s): {labels}."
                if crits else
                f"Learned what sets your {out['n_ex']} clip(s) apart."
            )
            self._count_label.setText(
                f"{found} Click Find matches to search the pool "
                f"(ranked by similarity). {note}".rstrip()
            )
        elif note:
            self._count_label.setText(self._count_label.text() + " " + note)

    # -- mining --------------------------------------------------------------

    def _current_criteria(self) -> list[Criterion]:
        return [r.to_criterion() for r in self._rows]

    def _reviewed_ids(self) -> set[str]:
        """Windows that already carry a review decision (read live, each count).

        The dialog is modeless, so the user can review clips behind it, reading
        the host's decision set on every count means a clip judged five seconds ago
        stops being offered without reopening the window.
        """
        if self._reviewed_provider is None:
            return set()
        try:
            return {str(w) for w in (self._reviewed_provider() or set())}
        except Exception:
            return set()

    def _select_for_load(self, matched: list[str], cap: int) -> list[str]:
        """The clips a Load would actually take: best-first, capped, subject-spread."""
        ordered = sorted(
            matched, key=lambda w: self._last_scores.get(w, 0.0), reverse=True
        )
        if not self._balance_subjects_chk.isChecked():
            return ordered[:cap]

        def _subject(wid: str) -> str:
            ref = self._clip_by_id.get(wid)
            return (
                self._metrics.subject_for_session(ref.session_id) if ref is not None
                else ""
            )

        return select_even_by_group(ordered, _subject, cap)

    def _update_count(self) -> None:
        active = [
            c for c in self._current_criteria()
            if c.enabled and (c.low is not None or c.high is not None)
        ]
        # Before the scored check, so the gap summary shows ahead of Find matches.
        self._refresh_coverage()
        if self._df is None:
            # Not scored yet: nothing to count against.
            self._last_matches = []
            self._last_scores = {}
            self._apply_btn.setEnabled(False)
            if not self._mining:
                self._count_label.setText(
                    "Click Find matches to score the pool and count matches."
                    if active else "Add one or more criteria, then Find matches."
                )
            return
        self._ensure_rich_columns()
        ranked = self._match_ranked.isChecked() and self._rank_scores is not None
        res = self._metrics.mine(
            self._df, self._current_criteria(), match_all=self._match_all.isChecked(),
            rank_scores=self._rank_scores, rank_only=ranked,
        )
        matched = res.matched_ids
        self._skipped_reviewed = 0
        if self._skip_reviewed_chk.isChecked():
            reviewed = self._reviewed_ids()
            if reviewed:
                kept = [w for w in matched if w not in reviewed]
                self._skipped_reviewed = len(matched) - len(kept)
                matched = kept
        self._skipped_gap = 0
        if self._gap_sessions is not None:
            gap_sessions = self._gap_sessions
            kept = [
                w for w in matched
                if (ref := self._clip_by_id.get(w)) is not None
                and str(ref.session_id) in gap_sessions
            ]
            self._skipped_gap = len(matched) - len(kept)
            matched = kept
        self._last_matches = matched
        self._last_scores = res.scores
        if ranked:
            cap = int(self._cap_spin.value())
            selected = self._select_for_load(matched, cap)
            skipped = (
                f" {self._skipped_reviewed} already-reviewed clip(s) skipped."
                if self._skipped_reviewed else ""
            )
            self._count_label.setText(
                f"Ranked {len(matched)} of {res.n_evaluated} segment(s) by "
                f"{self._rank_source}, will load the top {len(selected)}."
                f"{skipped} The {len(active)} criteria below describe the behavior "
                "but aren't filtering." + self._gap_note()
            )
            self._apply_btn.setEnabled(bool(matched))
        elif not active:
            self._count_label.setText(
                f"No active criteria: {res.n_evaluated} segment(s) in scope." + self._gap_note()
            )
            self._apply_btn.setEnabled(False)
        else:
            n_match = len(matched)
            cap = int(self._cap_spin.value())
            skipped = (
                f" {self._skipped_reviewed} already-reviewed match(es) skipped."
                if self._skipped_reviewed else ""
            )
            load = ""
            if n_match > cap:
                selected = self._select_for_load(matched, cap)
                n_subj = len({
                    self._metrics.subject_for_session(ref.session_id)
                    for ref in (self._clip_by_id.get(w) for w in selected)
                    if ref is not None
                })
                load = (
                    f", will load {len(selected)}"
                    + (
                        f" spread over {n_subj} subject(s)"
                        if self._balance_subjects_chk.isChecked() and n_subj > 1
                        else " by score"
                    )
                )
            self._count_label.setText(
                f"{n_match} of {res.n_evaluated} segment(s) match "
                f"{len(active)} criteria.{skipped}{load}" + self._gap_note()
            )
            self._apply_btn.setEnabled(bool(matched))

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
        # Auditing an existing review queue *is* a hard range test (that is the
        # question being asked of it), so Ranked flags as AND.
        self._on_flag_queue(criteria, not self._match_any.isChecked())

    def _apply(self) -> None:
        # Re-count first: the dialog is modeless, so clips may have been reviewed
        # behind it since the last count and must drop out of this batch.
        self._update_count()
        if not self._last_matches:
            return
        # Load only the highest-scoring matches up to the cap, so loose criteria
        # can't trigger extraction of thousands of clips at once, taking turns
        # between subjects when asked, so one animal can't fill the whole batch.
        cap = int(self._cap_spin.value())
        ordered = self._select_for_load(self._last_matches, cap)
        refs = [self._clip_by_id[w] for w in ordered if w in self._clip_by_id]
        if not refs:
            return
        n = len(refs)
        # Persist the working criteria so this feature set survives a reload.
        self._persist_criteria()
        # A gap hunt logs what it loaded, so reviewing those clips counts as
        # screening the subject (and enough misses mark it likely absent).
        bid = self._gap_behavior_id()
        if self._gap_sessions is not None and bid is not None:
            self._coverage.register_windows({r.window_id: r.session_id for r in refs})
            try:
                self._coverage.record_mined(bid, [r.window_id for r in refs])
            except Exception:
                logger.warning("Clip mining: could not log gap-mined clips", exc_info=True)
        self._on_apply(refs, dict(self._last_scores))
        # Stay open (modeless) so criteria can be refined and re-applied while the
        # mined queue updates behind the window.
        self._count_label.setText(f"Loading {n} matched clip(s) into the review queue…")

"""Social Interaction subtab of Behavior Analytics (multi-animal projects).

Two views over the per-frame ``social_*`` features, both computed by
:mod:`abel.services.social_analysis_service`:

* **Summary**: per (subject, session) dyadic metrics: mean inter-animal
  distance, time in contact, contact bouts, net approach, advance/yield
  balance, and orientation.
* **Dominance (HMM)**: a Gaussian HMM fit over continuous social + movement
  features (pooled across the cohort so states are comparable), and a
  displacement dominance index per subject.  A chart panel shows the
  hierarchy per session and over time, the displacement counts, group
  steepness, and the latent states (profiles, switching, dwell times,
  occupancy by rank, and a per-session ethogram).

The HMM fit runs on a worker thread and is saved under ``derived/analysis`` so
reopening the project restores it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.services.social_analysis_service import (
    DEFAULT_BIN_S,
    DEFAULT_MOVE_THRESH_BL,
    SocialAnalysisService,
)
from abel.ui.mpl_theme import style_navigation_toolbar
from abel.workers.task_worker import TaskWorker

if TYPE_CHECKING:
    from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab

logger = logging.getLogger("abel")

# Qualitative colors for latent states (index = state id).
_STATE_COLORS = [
    "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939",
]
_NO_PARTNER_COLOR = "#d9d9d9"
_FALLBACK_GROUP_COLORS = [
    "#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#a855f7",
    "#06b6d4", "#f43f5e", "#6366f1", "#84cc16", "#ec4899",
]

_VIEWS: list[tuple[str, str]] = [
    ("dominance", "Dominance index by group"),
    ("pct_dominant", "% of sessions with a significant dominant"),
    ("dominance_list", "Dominance index per session (list)"),
    ("over_time", "Dominance over time"),
    ("won_lost", "Displacements won vs lost"),
    ("compare", "Compare groups"),
    ("profiles", "State profiles"),
    ("switching", "State switching"),
    ("occ_rank", "State occupancy: dominant vs subordinate"),
    ("dwell", "State dwell times"),
    ("ethogram", "State ethogram (one session)"),
]


# Per-dyad metrics offered by the "Compare groups" view: (key, label, axis label).
# State-occupancy metrics are appended per fit (one per latent state).
_COMPARE_METRICS: list[tuple[str, str, str]] = [
    ("steepness", "Dominance index", "Dominance index of the dominant animal (0 = even, 1 = one-sided)"),
    ("track0_di", "track_0 dominance index (signed)",
     "track_0 dominance index (+1 = track_0 always wins, -1 = track_1 always wins)"),
    ("n_disp", "Displacements per dyad", "Displacements (won + lost)"),
    ("disp_rate", "Displacement rate", "Displacements per min of interaction"),
    ("inter_min", "Interaction time", "Interaction-state time (min)"),
]


_NO_HIERARCHY = "No significant hierarchy"


def _short_feature(name: str) -> str:
    s = name
    for pre in ("social_",):
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in ("_nearest_norm", "_nearest"):
        if s.endswith(suf):
            s = s[: -len(suf)] + (" (BL)" if suf == "_nearest_norm" else "")
            break
    return s.replace("dist_centroid_to_centroid", "distance").replace("_", " ")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class _Progress(QObject):
    """Carries worker-thread progress text to the UI thread (queued signal)."""

    message = Signal(str)


class SocialInteractionWidget(QWidget):
    """Social-interaction analytics subtab with a dominance HMM chart panel."""

    _SUMMARY_COLS = [
        ("Subject", "animal_id"),
        ("Session", "session_label"),
        ("Group", "group"),
        ("Frames", "n_frames"),
        ("Mean dist (norm)", "mean_distance_norm"),
        ("Contact %", "contact_fraction"),
        ("Contact bouts", "n_contact_bouts"),
        ("Mean bout (s)", "mean_contact_bout_s"),
        ("Mean approach", "mean_approach_velocity"),
        ("Advance %", "advance_fraction"),
        ("Heading align", "mean_heading_alignment"),
    ]

    _DOM_COLS = [
        ("Session", "session_label"),
        ("Subject", "animal_id"),
        ("Group", "group"),
        ("Rank", "dominance_rank"),
        ("Dominance index", "dominance_score"),
        ("Won", "displacements_won"),
        ("Lost", "displacements_lost"),
        ("p (binomial)", "displacement_p"),
        ("Advance %", "advance_fraction"),
        ("Yield %", "yield_fraction"),
        ("Interaction (s)", "interaction_time_s"),
        ("Method", "method"),
        ("Dominant?", "is_dominant"),
    ]

    _PCT_KEYS = ("contact_fraction", "advance_fraction", "yield_fraction")

    def __init__(self, host: "BehaviorAnalyticsTab") -> None:
        super().__init__()
        self._host = host
        self._frames_cache: pd.DataFrame | None = None
        self._summary_rows: list[dict[str, Any]] = []
        self._res: dict[str, Any] = {}
        self._worker: TaskWorker | None = None
        self._svc = SocialAnalysisService()
        # ── Controls ──────────────────────────────────────────────────────
        self._status = QLabel("Open a multi-animal project, then click Compute.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#90a4ae;font-size:11px;")
        # Bound to the label (a main-thread QObject) so the worker's emits are
        # queued onto the UI thread; a bare lambda would run in the worker.
        self._progress = _Progress()
        self._progress.message.connect(self._status.setText)

        self._compute_btn = QPushButton("Compute Social Metrics")
        self._compute_btn.setToolTip(
            "Load the per-frame social features and summarize the dyadic "
            "relationship for every subject / session."
        )
        self._compute_btn.clicked.connect(self._compute_summary)

        self._n_states_spin = QSpinBox()
        self._n_states_spin.setRange(2, 8)
        self._n_states_spin.setValue(4)
        self._n_states_spin.setToolTip(
            "Number of latent interaction states. States are numbered by\n"
            "proximity: S0 is the closest mode."
        )
        self._restarts_spin = QSpinBox()
        self._restarts_spin.setRange(1, 20)
        self._restarts_spin.setValue(10)
        self._restarts_spin.setToolTip(
            "Independent fits from different random starts; the best-scoring\n"
            "one is kept. Most starts land in a worse solution, so use 10 or more."
        )
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.05, 5.0)
        self._thresh_spin.setDecimals(2)
        self._thresh_spin.setSingleStep(0.1)
        self._thresh_spin.setValue(DEFAULT_MOVE_THRESH_BL)
        self._thresh_spin.setSuffix(" BL/s")
        self._thresh_spin.setToolTip(
            "A displacement needs one animal moving toward the other and the\n"
            "other moving away, both faster than this (body lengths per second)."
        )
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setRange(10.0, 3600.0)
        self._bin_spin.setDecimals(0)
        self._bin_spin.setSingleStep(30.0)
        self._bin_spin.setValue(DEFAULT_BIN_S)
        self._bin_spin.setSuffix(" s")
        self._bin_spin.setToolTip("Time-bin width for the 'Dominance over time' chart.")

        self._hmm_btn = QPushButton("Run Dominance HMM")
        self._hmm_btn.setToolTip(
            "Fit a Gaussian HMM over social + movement features (pooled across\n"
            "the cohort) and rank subjects by displacement dominance."
        )
        self._hmm_btn.clicked.connect(self._run_dominance_hmm)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self._compute_btn)
        ctrl.addSpacing(12)
        for label, w in (
            ("States:", self._n_states_spin),
            ("Restarts:", self._restarts_spin),
            ("Move ≥", self._thresh_spin),
            ("Bin:", self._bin_spin),
        ):
            ctrl.addWidget(QLabel(label))
            ctrl.addWidget(w)
        ctrl.addWidget(self._hmm_btn)
        ctrl.addStretch(1)

        # ── Summary table ─────────────────────────────────────────────────
        self._summary_table = self._make_table([c[0] for c in self._SUMMARY_COLS])

        # ── Dominance: chart panel over table + stats text ───────────────
        self._view_combo = QComboBox()
        for key, label in _VIEWS:
            self._view_combo.addItem(label, userData=key)
        self._view_combo.currentIndexChanged.connect(lambda _=None: self._on_view_changed())
        self._session_combo = QComboBox()
        self._session_combo.setToolTip("Session shown in the ethogram.")
        self._session_combo.currentIndexChanged.connect(lambda _=None: self._render())
        self._metric_lbl = QLabel("Metric:")
        self._metric_combo = QComboBox()
        self._metric_combo.setToolTip("Per-dyad measure compared across groups.")
        for key, label, _axis in _COMPARE_METRICS:
            self._metric_combo.addItem(label, userData=key)
        self._metric_combo.currentIndexChanged.connect(lambda _=None: self._render())

        # Same per-factor (combine) / (split) / level dropdowns as the Graphs tab,
        # kept local so this panel's grouping does not change the other tabs.
        self._facet_controls: dict[str, str] = {}
        self._facet: Any = None
        try:
            from abel.ui.tabs.behavior_analytics_tab import _FacetControls

            self._facet = _FacetControls("Group by:")
            self._facet.setToolTip(
                "For each factor choose (combine), (split), or a specific level.\n"
                "Split two or more factors to compare their interaction."
            )
            self._facet.changed.connect(self._on_facets_changed)
        except Exception:
            logger.debug("Facet controls unavailable for the social panel", exc_info=True)

        self._export_fig_btn = QPushButton("Export Figure…")
        self._export_fig_btn.clicked.connect(self._export_figure)
        self._prism_btn = QPushButton("Copy for Prism…")
        self._prism_btn.setToolTip(
            "The group chart's values (one per session) as Prism-ready columns."
        )
        self._prism_btn.clicked.connect(self._copy_for_prism)
        self._export_csv_btn = QPushButton("Export Results…")
        self._export_csv_btn.setToolTip(
            "Write every dominance table (per subject, per time bin, per\n"
            "displacement event, state profiles, occupancy, switching) as CSVs."
        )
        self._export_csv_btn.clicked.connect(self._export_results)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self._view_combo)
        self._session_lbl = QLabel("Session:")
        view_row.addWidget(self._session_lbl)
        view_row.addWidget(self._session_combo)
        view_row.addWidget(self._metric_lbl)
        view_row.addWidget(self._metric_combo)
        if self._facet is not None:
            view_row.addSpacing(12)
            view_row.addWidget(self._facet)
        view_row.addStretch(1)
        view_row.addWidget(self._prism_btn)
        view_row.addWidget(self._export_fig_btn)
        view_row.addWidget(self._export_csv_btn)

        chart_holder = QWidget()
        chart_v = QVBoxLayout(chart_holder)
        chart_v.setContentsMargins(0, 0, 0, 0)
        chart_v.addLayout(view_row)
        self._fig: Any = None
        self._canvas: Any = None
        try:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(8, 4.5))
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            toolbar = NavigationToolbar2QT(self._canvas, chart_holder)
            style_navigation_toolbar(toolbar)
            chart_v.addWidget(toolbar)
            chart_v.addWidget(self._canvas, 1)
        except Exception:
            logger.exception("matplotlib unavailable for the social dominance charts")
            chart_v.addWidget(QLabel("Charts need matplotlib."), 1)

        self._dom_table = self._make_table([c[0] for c in self._DOM_COLS])
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setPlaceholderText(
            "Run the dominance HMM to see the fit, the latent states, the "
            "hierarchy per session and the checks behind it."
        )
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self._dom_table)
        bottom.addWidget(self._stats_text)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)

        dom_split = QSplitter(Qt.Orientation.Vertical)
        dom_split.addWidget(chart_holder)
        dom_split.addWidget(bottom)
        dom_split.setStretchFactor(0, 3)
        dom_split.setStretchFactor(1, 2)

        self._inner = QTabWidget()
        self._inner.addTab(self._summary_table, "Summary")
        self._inner.addTab(dom_split, "Dominance (HMM)")

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.addLayout(ctrl)
        root.addWidget(self._status)
        root.addWidget(self._inner, 1)

        self._on_view_changed()
        self._set_result_widgets_enabled(False)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def _project_root(self) -> Path | None:
        return getattr(self._host, "_project_root", None)

    def _session_label(self, sid: str) -> str:
        labels = getattr(self._host, "_session_label_by_session", {}) or {}
        return str(labels.get(sid, sid))

    def _facet_groups(self) -> dict[str, str] | None:
        """session_label -> series label for this panel's facet selection.

        ``None`` when the host has no factors (or no facet support); every
        session then falls back to the host's own group map.
        """
        fn = getattr(self._host, "_session_groups_for_controls", None)
        if not callable(fn) or not getattr(self._host, "_factor_definitions", None):
            return None
        try:
            return dict(fn(self._facet_controls))
        except Exception:
            logger.debug("Could not resolve facet groups", exc_info=True)
            return None

    def _group_of(self, sid: str) -> str:
        # The host keys groups by session *label*, not by session id.
        facet = self._facet_groups()
        groups = facet if facet is not None else (getattr(self._host, "_session_groups", {}) or {})
        return str(groups.get(self._session_label(sid), groups.get(sid, "")) or "")

    def _session_included(self, sid: str) -> bool:
        """False when the facet selection filters this session out."""
        facet = self._facet_groups()
        return facet is None or self._session_label(sid) in facet or sid in facet

    def _order_groups(self, groups: Any) -> list[str]:
        """Groups in the user's saved level order (as in the Graphs tab)."""
        avail = sorted({str(g) for g in groups})
        fn = getattr(self._host, "_ordered_group_list", None)
        split_fn = getattr(self._host, "_split_factors_for_controls", None)
        if callable(fn) and self._facet_groups() is not None:
            try:
                split = split_fn(self._facet_controls) if callable(split_fn) else None
                ordered = [g for g in fn(avail, split) if g in avail]
                return ordered + [g for g in avail if g not in ordered]
            except Exception:
                logger.debug("Could not order groups", exc_info=True)
        return avail

    def on_groups_updated(self) -> None:
        """Host hook: factor definitions or session levels changed."""
        self._refresh_facets()
        if self._res:
            self._refresh_result_views()

    def _refresh_facets(self) -> None:
        if self._facet is None:
            return
        factors = list(getattr(self._host, "_factor_definitions", []) or [])
        levels_fn = getattr(self._host, "_levels_by_factor", None)
        levels = levels_fn() if callable(levels_fn) else {}
        self._facet.blockSignals(True)
        self._facet.rebuild(factors, levels)
        if not self._facet_controls and factors:
            default_fn = getattr(self._host, "_default_facet_controls", None)
            if callable(default_fn):
                self._facet_controls = dict(default_fn())
        self._facet.set_state(self._facet_controls)
        self._facet.blockSignals(False)

    def _on_facets_changed(self) -> None:
        if self._facet is not None:
            self._facet_controls = self._facet.state()
        if self._res:
            self._refresh_result_views()

    def _group_map(self, session_ids: Any) -> dict[str, str]:
        return {str(s): self._group_of(str(s)) for s in session_ids}

    def _group_color(self, group: str, index: int) -> str:
        fn = getattr(self._host, "_group_color", None)
        if callable(fn):
            try:
                return str(fn(group, index))
            except Exception:
                pass
        return _FALLBACK_GROUP_COLORS[index % len(_FALLBACK_GROUP_COLORS)]

    def _set_result_widgets_enabled(self, on: bool) -> None:
        for w in (self._view_combo, self._export_fig_btn, self._export_csv_btn, self._prism_btn):
            w.setEnabled(on)
        self._session_combo.setEnabled(on and self._view_key() == "ethogram")

    def _view_key(self) -> str:
        return str(self._view_combo.currentData() or "dominance")

    def _on_view_changed(self) -> None:
        eth = self._view_key() == "ethogram"
        cmp_view = self._view_key() == "compare"
        self._prism_btn.setVisible(self._view_key() in ("compare", "dominance", "pct_dominant"))
        self._metric_lbl.setVisible(cmp_view)
        self._metric_combo.setVisible(cmp_view)
        self._session_lbl.setVisible(eth)
        self._session_combo.setVisible(eth)
        self._session_combo.setEnabled(eth and bool(self._res))
        self._render()

    def _settings(self) -> dict[str, Any]:
        return {
            "n_states": int(self._n_states_spin.value()),
            "n_iter": 200,
            "n_restarts": int(self._restarts_spin.value()),
            "move_thresh_bl": float(self._thresh_spin.value()),
            "min_event_s": 0.1,
            "bin_s": float(self._bin_spin.value()),
        }

    def on_project_reloaded(self) -> None:
        """Clear cached results and restore a saved fit for the new project."""
        self._frames_cache = None
        self._summary_rows = []
        self._res = {}
        self._summary_table.setRowCount(0)
        self._dom_table.setRowCount(0)
        self._stats_text.clear()
        self._session_combo.clear()
        self._status.setText("Open a multi-animal project, then click Compute.")
        self._set_result_widgets_enabled(False)
        self._facet_controls = {}
        self._refresh_facets()
        root = self._project_root()
        if root is not None:
            res, fp = self._svc.load_result(root)
            settings = res.get("settings") if res else None
            if res and settings:
                # Restore even a stale fit (like the behavior-state HMM) so the
                # user is never forced to refit just to reopen the project.
                self._apply_settings(settings)
                self._show_result(res, restored=True)
                if fp != self._svc.input_fingerprint(root, settings):
                    self._status.setText(
                        self._status.text()
                        + " The pose features changed since this fit; click Run "
                        "Dominance HMM to refresh it."
                    )
                else:
                    sids = {str(r.get("session_id")) for r in res.get("dominance") or []}
                    if self._prechop_map(sorted(sids)) != settings.get("prechop_frames", {}):
                        self._status.setText(
                            self._status.text()
                            + " The prechop changed since this fit; click Run "
                            "Dominance HMM to refresh it."
                        )
        self._render()

    def _apply_settings(self, s: dict[str, Any]) -> None:
        self._n_states_spin.setValue(int(s.get("n_states", 4)))
        self._restarts_spin.setValue(int(s.get("n_restarts", 10)))
        self._thresh_spin.setValue(float(s.get("move_thresh_bl", DEFAULT_MOVE_THRESH_BL)))
        self._bin_spin.setValue(float(s.get("bin_s", DEFAULT_BIN_S)))

    def _prechop_map(self, session_ids: Any) -> dict[str, int]:
        """Per-session prechop from the Analytics tab (frames before it are dropped)."""
        fn = getattr(self._host, "_analysis_prechop_for_session", None)
        if not callable(fn):
            return {}
        out: dict[str, int] = {}
        for sid in session_ids:
            try:
                out[str(sid)] = int(fn(str(sid)))
            except Exception:
                continue
        return self._svc.clean_prechop(out)

    def _load_frames(self) -> pd.DataFrame | None:
        if self._frames_cache is not None:
            return self._frames_cache
        root = self._project_root()
        if root is None:
            return None
        df = self._svc.load_social_frames(root)
        self._frames_cache = df
        return df

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else ""
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "-"
            return f"{value:.3f}"
        return str(value)

    def _fill_table(self, table: QTableWidget, cols, rows: list[dict]) -> None:
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (_label, key) in enumerate(cols):
                val = row.get(key, "")
                is_num = isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)
                if key in self._PCT_KEYS and is_num and np.isfinite(val):
                    text = f"{val * 100:.1f}%"
                elif key == "displacement_p" and is_num:
                    text = _fmt_p(float(val))
                elif key in ("displacements_won", "displacements_lost") and is_num and np.isfinite(val):
                    text = f"{int(val)}"
                else:
                    text = self._fmt(val)
                item = QTableWidgetItem(text)
                if is_num:
                    item.setData(Qt.ItemDataRole.UserRole, float(val) if np.isfinite(val) else float("nan"))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()

    def _current_dominance(self) -> list[dict[str, Any]]:
        """Dominance rows with the host's *current* labels and groups."""
        out = []
        for r in self._res.get("dominance", []) or []:
            rr = dict(r)
            sid = str(rr["session_id"])
            if not self._session_included(sid):
                continue
            rr["session_label"] = self._session_label(sid)
            rr["group"] = self._group_of(sid)
            out.append(rr)
        return out

    # ── Actions ─────────────────────────────────────────────────────────────

    def _compute_summary(self) -> None:
        df = self._load_frames()
        if df is None:
            self._status.setText(
                "No social features found. This tab needs a multi-animal project "
                "with 'Interaction features' extracted in the Pose Features tab."
            )
            return
        fps = self._host._project_fps()
        rows = self._svc.compute_social_summary(
            df, fps, self._group_map(df["session_id"].unique())
        )
        for r in rows:
            r["session_label"] = self._session_label(str(r["session_id"]))
        self._summary_rows = rows
        self._fill_table(self._summary_table, self._SUMMARY_COLS, rows)
        n_subj = len({(r["animal_id"], r["session_id"]) for r in rows})
        self._status.setText(
            f"Summarized {n_subj} subject/session record(s) from "
            f"{len(df):,} frames. Run the dominance HMM for latent-state ranking."
        )

    def _run_dominance_hmm(self) -> None:
        if self._worker is not None:
            return
        df = self._load_frames()
        if df is None:
            self._status.setText(
                "No social features found. Extract 'Interaction features' first."
            )
            return
        root = self._project_root()
        fps = self._host._project_fps()
        settings = self._settings()
        group_map = self._group_map(df["session_id"].unique())
        prechop = self._prechop_map(df["session_id"].unique())
        svc = self._svc
        progress = self._progress

        def _job() -> dict[str, Any]:
            res = svc.fit_dominance_hmm(
                df, fps=fps,
                n_states=settings["n_states"],
                n_iter=settings["n_iter"],
                n_restarts=settings["n_restarts"],
                move_thresh_bl=settings["move_thresh_bl"],
                min_event_s=settings["min_event_s"],
                bin_s=settings["bin_s"],
                prechop_frames=prechop,
                group_map=group_map,
                progress_cb=progress.message.emit,
            )
            if not res.get("error") and root is not None:
                svc.save_result(root, res, svc.input_fingerprint(root, res["settings"]))
            return res

        self._hmm_btn.setEnabled(False)
        self._hmm_btn.setText("Fitting…")
        self._status.setText("Fitting dominance HMM… (about a minute per restart on large cohorts)")
        worker = TaskWorker(_job)
        worker.signals.finished.connect(self._on_hmm_done)
        worker.signals.failed.connect(self._on_hmm_failed)
        self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _hmm_finished(self) -> None:
        self._worker = None
        self._hmm_btn.setEnabled(True)
        self._hmm_btn.setText("Run Dominance HMM")

    def _on_hmm_failed(self, tb: str) -> None:
        self._hmm_finished()
        logger.error("Dominance HMM failed:\n%s", tb)
        self._status.setText("Dominance HMM failed; see the log for details.")

    def _on_hmm_done(self, res: dict[str, Any]) -> None:
        self._hmm_finished()
        if res.get("error"):
            self._status.setText(str(res["error"]))
            return
        self._show_result(res, restored=False)

    def _sorted_dominance(self) -> list[dict[str, Any]]:
        rows = self._current_dominance()
        order = self._order_groups(r["group"] for r in rows)
        rows.sort(key=lambda r: (order.index(r["group"]), r["session_label"], r["dominance_rank"]))
        return rows

    def _refresh_result_views(self) -> None:
        """Refill the table, stats and chart for the current grouping."""
        rows = self._sorted_dominance()
        self._fill_table(self._dom_table, self._DOM_COLS, rows)
        self._stats_text.setPlainText(self._format_stats(self._res, rows))
        self._render()

    def _show_result(self, res: dict[str, Any], *, restored: bool) -> None:
        self._res = res
        self._refresh_facets()
        self._rebuild_metric_combo(res)
        rows = self._sorted_dominance()
        self._fill_table(self._dom_table, self._DOM_COLS, rows)
        self._stats_text.setPlainText(self._format_stats(res, rows))

        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        sids = sorted({str(k[1]) for k in res.get("state_seqs", {})}, key=self._session_label)
        for sid in sids:
            self._session_combo.addItem(self._session_label(sid), userData=sid)
        self._session_combo.blockSignals(False)

        self._set_result_widgets_enabled(True)
        n_dom = sum(1 for r in rows if r.get("is_dominant"))
        fi = res.get("fit_info", {})
        prefix = "Restored the saved" if restored else "Fit a"
        conv = "" if fi.get("converged", True) else " (did not converge; raise restarts or check the data)"
        self._status.setText(
            f"{prefix} {res['n_states']}-state HMM over {len(res['feature_cols'])} features"
            f"{conv}; dominant subject identified in {n_dom} session(s)."
        )
        idc = res.get("identity_check", {})
        p = float(idc.get("p", float("nan")))
        if np.isfinite(p) and p < 0.05:
            self._status.setText(
                self._status.text()
                + f" Warning: one track id ranks first in most sessions (p = {_fmt_p(p)}); "
                "see the checks below."
            )
        self._inner.setCurrentIndex(1)
        self._render()

    # ── Stats text ──────────────────────────────────────────────────────────

    def _format_stats(self, res: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        n = int(res.get("n_states", 0))
        fi = res.get("fit_info", {})
        st = res.get("settings", {})
        labels = res.get("state_labels", {})
        occ_all = self._pooled_occupancy(res)
        dwell = res.get("dwell", {})
        lines = ["FIT"]
        lines.append(
            f"  {n} states, {fi.get('n_frames_fit', 0):,} frames in "
            f"{fi.get('n_sequences', 0)} unbroken stretches."
        )
        lls = ", ".join(f"{v:,.0f}" for v in fi.get("restart_log_likelihoods", []))
        lines.append(f"  Restart log-likelihoods: {lls or '-'} (best kept).")
        lines.append(
            f"  Converged: {'yes' if fi.get('converged') else 'no'} after "
            f"{fi.get('n_iter_run', 0)} iterations."
        )
        np_frac = list((res.get("no_partner_fraction") or {}).values())
        if np_frac:
            lines.append(
                f"  Partner not tracked in {100 * float(np.mean(np_frac)):.1f}% of frames "
                "(left out of the fit, state -1)."
            )
        lines.append("")
        lines.append("STATES (numbered by proximity; S0 = closest)")
        for s in range(n):
            d = dwell.get(s, {})
            lines.append(
                f"  {labels.get(s, f'S{s}')}: {100 * occ_all[s]:.1f}% of tracked frames, "
                f"median dwell {d.get('median_s', float('nan')):.2f} s"
            )
        lines.append("")

        paired = [r for r in rows if r.get("method") == "paired"]
        dom = [r for r in paired if r.get("is_dominant")]
        n_sess = len({r["session_id"] for r in paired})
        sig = [r for r in dom if float(r.get("displacement_p", np.nan)) < 0.05]
        lines.append("DOMINANCE")
        if n_sess:
            di = [float(r["dominance_score"]) for r in dom]
            lines.append(
                f"  {n_sess} dyad session(s); a dominant animal in {len(dom)}, "
                f"significant (binomial p < 0.05) in {len(sig)}."
            )
            if di:
                lines.append(
                    f"  Dominance index of the dominant animal: mean {np.mean(di):.2f}, "
                    f"range {min(di):.2f} to {max(di):.2f}."
                )
        unpaired = {r["session_id"] for r in rows if r.get("method") == "unpaired"}
        if unpaired:
            lines.append(
                f"  {len(unpaired)} session(s) with more than two animals use the "
                "unpaired score (advance % minus yield %)."
            )
        idc = res.get("identity_check", {})
        if idc.get("n_sessions"):
            counts = ", ".join(f"{k} {v}" for k, v in idc.get("counts", {}).items())
            p = float(idc.get("p", float("nan")))
            flag = "  <- check track identity" if np.isfinite(p) and p < 0.05 else ""
            lines.append(
                f"  Ranked first by track id: {counts} (binomial p = {_fmt_p(p)}).{flag}"
            )
        gs = self._svc.group_steepness(rows)
        if len(gs["summary"]) >= 2 or (len(gs["summary"]) == 1 and "" not in gs["summary"]):
            lines.append("  Hierarchy steepness (|dominance index|) by group:")
            for g in self._order_groups(gs["summary"]):
                s = gs["summary"][g]
                lines.append(
                    f"    {g or '(no group)'}: {s['mean']:.2f} ± {s['sem']:.2f} SEM (n = {s['n']})"
                )
            if gs["test"]:
                lines.append(f"    {gs['test']}: p = {_fmt_p(gs['p'])}")
        lines.append("")
        lines.append("DEFINITIONS")
        lines.append(
            f"  Displacement: in an interaction state, one animal moves toward the other "
            f"and the other moves away, both faster than {st.get('move_thresh_bl', 0.5):.2f} "
            f"body lengths/s, for at least {st.get('min_event_s', 0.1):.2f} s."
        )
        lines.append(
            "  Dominance index = (won - lost) / (won + lost): +1 always displaces, "
            "-1 always displaced. p is a two-sided binomial test of won vs lost."
        )
        lines.append(
            "  Interaction states: states whose mean distance is at or below the "
            f"median across states ({', '.join(f'S{s}' for s in res.get('interaction_states', []))})."
        )
        lines.append(
            "  Features are clipped to their 0.5-99.5 percentiles before the fit so "
            "tracking jumps do not form a state of their own."
        )
        return "\n".join(lines)

    @staticmethod
    def _pooled_occupancy(res: dict[str, Any]) -> np.ndarray:
        n = int(res.get("n_states", 0))
        counts = np.zeros(n, dtype=float)
        for seq in (res.get("state_seqs") or {}).values():
            ok = seq >= 0
            counts += np.bincount(seq[ok].astype(np.int64), minlength=n)[:n]
        total = counts.sum()
        return counts / total if total > 0 else counts

    # ── Rendering ───────────────────────────────────────────────────────────

    def _render(self) -> None:
        fig = self._fig
        if fig is None:
            return
        fig.clear()
        bg = "#ffffff"
        try:
            bg = str(self._host._graph_settings.get("fig_bg", "#ffffff"))
        except Exception:
            pass
        fig.set_facecolor(bg)
        if not self._res:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5, 0.5, "Run the dominance HMM to see charts.",
                ha="center", va="center", color="#607d8b", transform=ax.transAxes,
            )
            self._canvas.draw_idle()
            return
        key = self._view_key()
        draw = {
            "dominance": self._draw_dominance_by_group,
            "pct_dominant": self._draw_pct_dominant,
            "dominance_list": self._draw_dominance,
            "over_time": self._draw_over_time,
            "won_lost": self._draw_won_lost,
            "compare": self._draw_compare,
            "profiles": self._draw_profiles,
            "switching": self._draw_switching,
            "occ_rank": self._draw_occ_rank,
            "dwell": self._draw_dwell,
            "ethogram": self._draw_ethogram,
        }.get(key)
        try:
            if draw is not None:
                draw(fig)
            try:
                fig.tight_layout()
            except Exception:
                pass
        except Exception:
            logger.exception("Could not draw the social dominance view %s", key)
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "Could not draw this view; see the log.",
                    ha="center", va="center", transform=ax.transAxes)
        self._canvas.draw_idle()

    @staticmethod
    def _tidy(ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _empty(self, fig: Any, msg: str) -> None:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="#607d8b",
                transform=ax.transAxes, wrap=True)

    def _groups_in(self, rows: list[dict[str, Any]]) -> list[str]:
        return self._order_groups(str(r.get("group", "")) for r in rows)

    def _draw_dominance(self, fig: Any) -> None:
        rows = [r for r in self._current_dominance() if r.get("is_dominant")]
        if not rows:
            self._empty(fig, "No session has a dominant animal yet.")
            return
        groups = self._groups_in(rows)
        rows.sort(key=lambda r: (groups.index(r["group"]), -float(r["dominance_score"])))
        ax = fig.add_subplot(111)
        y = np.arange(len(rows))[::-1]
        colors = [self._group_color(r["group"], groups.index(r["group"])) for r in rows]
        vals = [float(r["dominance_score"]) for r in rows]
        ax.barh(y, vals, color=colors, height=0.7)
        small = len(rows) > 24
        fs = 6 if small else 8
        for yi, r, v in zip(y, rows, vals):
            won, lost = r.get("displacements_won"), r.get("displacements_lost")
            tag = r["animal_id"]
            if r.get("method") == "paired" and np.isfinite(won):
                tag += f"  {int(won)}:{int(lost)}"
            if float(r.get("displacement_p", np.nan)) < 0.05:
                tag += " *"
            ax.text(v + 0.01, yi, tag, va="center", fontsize=fs)
        ax.set_yticks(y)
        ax.set_yticklabels([r["session_label"] for r in rows], fontsize=fs)
        ax.set_xlim(0, 1.25 if any(r.get("method") == "paired" for r in rows) else None)
        ax.set_xlabel("Dominance index of the dominant animal (0 = even, 1 = one-sided)")
        ax.set_title("Who dominates each session (won:lost displacements, * p < 0.05)")
        self._group_legend(ax, groups)
        self._tidy(ax)

    def _group_legend(self, ax: Any, groups: list[str]) -> None:
        if groups in ([], [""]):
            return
        from matplotlib.patches import Patch
        handles = [
            Patch(color=self._group_color(g, i), label=g or "(no group)")
            for i, g in enumerate(groups)
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")

    def _draw_over_time(self, fig: Any) -> None:
        bins = pd.DataFrame(self._res.get("dominance_bins") or [])
        dom = [r for r in self._current_dominance() if r.get("is_dominant") and r.get("method") == "paired"]
        if bins.empty or not dom:
            self._empty(fig, "No dyad sessions with displacement events to bin.")
            return
        top = {str(r["session_id"]): (str(r["animal_id"]), r["group"]) for r in dom}
        bins = bins[bins.apply(lambda b: top.get(str(b["session_id"]), ("",))[0] == str(b["animal_id"]), axis=1)]
        if bins.empty:
            self._empty(fig, "No binned events for the dominant animals.")
            return
        bin_s = float(self._res.get("settings", {}).get("bin_s", DEFAULT_BIN_S))
        bins = bins.assign(
            t_min=(bins["bin_start_s"] + bin_s / 2.0) / 60.0,
            group=bins["session_id"].map(lambda s: top[str(s)][1]),
        )
        groups = self._order_groups(bins["group"].unique())
        ax = fig.add_subplot(111)
        for sid, b in bins.groupby("session_id"):
            gi = groups.index(top[str(sid)][1])
            ax.plot(b["t_min"], b["dominance_index"], color=self._group_color(groups[gi], gi),
                    alpha=0.25, lw=0.8)
        for gi, g in enumerate(groups):
            gb = bins[bins["group"] == g].groupby("t_min")["dominance_index"]
            m = gb.mean()
            sem = gb.std(ddof=1) / np.sqrt(gb.count())
            c = self._group_color(g, gi)
            ax.plot(m.index, m.values, color=c, lw=2.2, marker="o", ms=4,
                    label=f"{g or 'All sessions'} (mean ± SEM)")
            ax.fill_between(m.index, (m - sem).values, (m + sem).values, color=c, alpha=0.18, lw=0)
        ax.axhline(0, color="#9e9e9e", lw=0.8, ls="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Dominance index in bin")
        ax.set_title(
            f"Is the hierarchy stable? {bin_s:.0f} s bins; positive = the session's "
            "overall dominant animal also wins that bin"
        )
        ax.legend(fontsize=8, frameon=False)
        self._tidy(ax)

    def _draw_won_lost(self, fig: Any) -> None:
        rows = [r for r in self._current_dominance() if r.get("method") == "paired"]
        if not rows:
            self._empty(fig, "Displacement counts need dyad sessions (exactly two animals).")
            return
        groups = self._groups_in(rows)
        ax = fig.add_subplot(111)
        hi = 1.0
        for gi, g in enumerate(groups):
            c = self._group_color(g, gi)
            for dominant, marker in ((True, "^"), (False, "o")):
                sel = [r for r in rows if r["group"] == g and bool(r.get("is_dominant")) == dominant]
                if not sel:
                    continue
                x = [float(r["displacements_lost"]) for r in sel]
                y = [float(r["displacements_won"]) for r in sel]
                hi = max(hi, *x, *y)
                label = f"{g + ': ' if g else ''}{'dominant' if dominant else 'subordinate'}"
                ax.scatter(x, y, c=c, marker=marker, s=36, alpha=0.8,
                           edgecolors="white", linewidths=0.5, label=label)
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color="#9e9e9e", lw=0.8, ls="--")
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_xlabel("Displacements lost")
        ax.set_ylabel("Displacements won")
        ax.set_title("Each point is one animal in one session; above the line = wins more")
        ax.legend(fontsize=8, frameon=False)
        self._tidy(ax)

    def _rebuild_metric_combo(self, res: dict[str, Any]) -> None:
        prev = self._metric_combo.currentData()
        labels = res.get("state_labels", {})
        self._metric_combo.blockSignals(True)
        self._metric_combo.clear()
        for key, label, _axis in _COMPARE_METRICS:
            self._metric_combo.addItem(label, userData=key)
        for s in range(int(res.get("n_states", 0))):
            self._metric_combo.addItem(f"Time in {labels.get(s, f'S{s}')}", userData=f"occ_{s}")
        idx = self._metric_combo.findData(prev)
        self._metric_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._metric_combo.blockSignals(False)

    def dyad_metrics(self) -> pd.DataFrame:
        """One row per dyad session kept by the facets, with its group and metrics."""
        res = self._res
        rows = [r for r in self._current_dominance() if r.get("method") == "paired"]
        occ = res.get("occupancy") or {}
        n = int(res.get("n_states", 0))
        out = []
        for sid in sorted({str(r["session_id"]) for r in rows}):
            pair = [r for r in rows if str(r["session_id"]) == sid]
            top = next((r for r in pair if r.get("is_dominant")), None)
            won = sum(float(r.get("displacements_won", 0) or 0) for r in pair)
            first = min(pair, key=lambda r: str(r["animal_id"]))
            sig = top is not None and float(top.get("displacement_p", 1.0)) < 0.05
            inter_s = float(np.mean([float(r.get("interaction_time_s", 0.0)) for r in pair]))
            row: dict[str, Any] = {
                "session_id": sid,
                "session_label": self._session_label(sid),
                "group": pair[0]["group"],
                # A tied dyad has no dominant animal: steepness 0.
                "steepness": abs(float(top["dominance_score"])) if top else 0.0,
                "track0_di": float(first.get("dominance_score", np.nan)),
                "outcome": f"{top['animal_id']} dominant" if sig else _NO_HIERARCHY,
                "n_disp": won,
                "disp_rate": won / (inter_s / 60.0) if inter_s > 0 else np.nan,
                "inter_min": inter_s / 60.0,
            }
            fr = [np.asarray(occ[(str(r["animal_id"]), sid)], dtype=float)
                  for r in pair if (str(r["animal_id"]), sid) in occ]
            for s in range(n):
                vals = [f[s] for f in fr if s < f.size]
                row[f"occ_{s}"] = 100.0 * float(np.mean(vals)) if vals else np.nan
            out.append(row)
        return pd.DataFrame(out)

    def _draw_pct_dominant(self, fig: Any) -> None:
        t = self.outcome_table()
        if t.empty:
            self._empty(fig, "No dyad sessions to compare.")
            return
        n = t.sum(axis=1)
        pct = t.div(n.where(n > 0), axis=0) * 100.0
        colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b"]
        ax = fig.add_subplot(111)
        x = np.arange(len(t.index))
        bottom = np.zeros(len(t.index))
        for i, col in enumerate(t.columns):
            c = "#cfd8dc" if col == _NO_HIERARCHY else colors[i % len(colors)]
            ax.bar(x, pct[col].to_numpy(), width=0.6, bottom=bottom, color=c, label=col,
                   edgecolor="white", zorder=2)
            for xi, (b, h, k) in enumerate(zip(bottom, pct[col], t[col])):
                if k:
                    ax.text(xi, b + h / 2, f"{k}", ha="center", va="center", fontsize=9)
            bottom += pct[col].fillna(0).to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n = {k})" for g, k in zip(t.index, n)])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of sessions (numbers = session counts)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
        title = "Sessions with a significant dominant animal (binomial p < 0.05)"
        if len(t.index) > 1:
            parts = []
            test, p = self._svc.compare_proportions(t.to_numpy())
            if test:
                parts.append(f"outcome {test} p = {_fmt_p(p)}")
            any_sig = np.column_stack([n - t[_NO_HIERARCHY], t[_NO_HIERARCHY]])
            test2, p2 = self._svc.compare_proportions(any_sig)
            if test2 and t.shape[1] > 2:
                parts.append(f"any dominant {test2} p = {_fmt_p(p2)}")
            if parts:
                title += "\n" + "; ".join(parts)
        ax.set_title(title)
        self._tidy(ax)
        fig.subplots_adjust(right=0.78)

    def _draw_dominance_by_group(self, fig: Any) -> None:
        self._draw_compare(fig, key="steepness")

    def _current_metric_key(self) -> str:
        if self._view_key() == "dominance":
            return "steepness"
        return str(self._metric_combo.currentData() or "steepness")

    def outcome_table(self) -> pd.DataFrame:
        """Session counts per group (rows) and outcome (columns)."""
        df = self.dyad_metrics()
        if df.empty:
            return pd.DataFrame()
        outcomes = sorted(o for o in df["outcome"].unique() if o != _NO_HIERARCHY)
        outcomes.append(_NO_HIERARCHY)
        t = pd.crosstab(df["group"], df["outcome"]).reindex(
            index=self._order_groups(df["group"]), columns=outcomes, fill_value=0)
        t.index = [g or "All sessions" for g in t.index]
        t.index.name = "Group"
        return t

    def prism_table(self) -> tuple[pd.DataFrame, str]:
        """Current group metric as a Prism column table: one column per group,
        one row per session (dyad), shorter columns padded with blanks. The %
        view gives a contingency table instead (groups x outcome counts)."""
        if self._view_key() == "pct_dominant":
            return self.outcome_table().reset_index(), "outcome"
        key = self._current_metric_key()
        df = self.dyad_metrics()
        if df.empty or key not in df:
            return pd.DataFrame(), key
        df = df[np.isfinite(df[key].astype(float))]
        cols = {}
        for g in self._order_groups(df["group"]):
            cols[g or "All sessions"] = df.loc[df["group"] == g, key].reset_index(drop=True)
        return pd.DataFrame(cols), key

    def _copy_for_prism(self) -> None:
        from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QPlainTextEdit

        wide, key = self.prism_table()
        if wide.empty:
            QMessageBox.information(self, "Copy for Prism", "Run the dominance HMM first.")
            return
        label = "Dominance outcome" if key == "outcome" else next(
            (l for k, l, _a in _COMPARE_METRICS if k == key), self._metric_combo.currentText())
        tidy = self.dyad_metrics()[["group", "session_label", key]].rename(columns={key: label})
        wide_txt = wide.to_csv(sep="\t", index=False, na_rep="")
        tidy_txt = tidy.to_csv(sep="\t", index=False, na_rep="")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Copy for Prism: {label}")
        v = QVBoxLayout(dlg)
        contingency = key == "outcome"
        v.addWidget(QLabel(
            ("Prism: New Contingency table, then paste. One row per group, one column per "
             "outcome (session counts).\n" if contingency else
             "Prism: New Column table, then paste. One column per group, one value per session.\n")
            + "The 'by session' version keeps the session names for Excel or R."
        ))
        fmt = QComboBox()
        fmt.addItem("Prism contingency (group x outcome counts)" if contingency
                    else "Prism columns (one per group)", wide_txt)
        fmt.addItem("By session (group, session, value)", tidy_txt)
        v.addWidget(fmt)
        box = QPlainTextEdit(wide_txt)
        box.setReadOnly(True)
        v.addWidget(box, 1)
        fmt.currentIndexChanged.connect(lambda _i: box.setPlainText(str(fmt.currentData())))
        btns = QDialogButtonBox()
        copy_btn = btns.addButton("Copy to Clipboard", QDialogButtonBox.ButtonRole.ActionRole)
        save_btn = btns.addButton("Save CSV…", QDialogButtonBox.ButtonRole.ActionRole)
        btns.addButton(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)

        def _copy() -> None:
            QApplication.clipboard().setText(str(fmt.currentData()))
            copy_btn.setText("Copied")

        def _save() -> None:
            path, _ = QFileDialog.getSaveFileName(dlg, "Save CSV", f"{key}.csv", "CSV (*.csv)")
            if path:
                # ASCII with a BOM so Excel and Prism never show mojibake.
                txt = str(fmt.currentData()).replace("\t", ",").replace("\u00d7", "x")
                Path(path).write_text(txt.encode("ascii", "replace").decode(), encoding="utf-8-sig")

        copy_btn.clicked.connect(_copy)
        save_btn.clicked.connect(_save)
        v.addWidget(btns)
        dlg.resize(560, 420)
        dlg.exec()

    def _draw_compare(self, fig: Any, key: str | None = None) -> None:
        df = self.dyad_metrics()
        key = key or str(self._metric_combo.currentData() or "steepness")
        if df.empty or key not in df:
            self._empty(fig, "No dyad sessions to compare.")
            return
        label = next((l for k, l, _a in _COMPARE_METRICS if k == key), self._metric_combo.currentText())
        axis = next(
            (a for k, _l, a in _COMPARE_METRICS if k == key), f"{label} (% of tracked frames)"
        )
        df = df[np.isfinite(df[key].astype(float))]
        groups = self._order_groups(df["group"])
        by_group = {g: df.loc[df["group"] == g, key].to_numpy(dtype=float) for g in groups}
        ax = fig.add_subplot(111)
        rng = np.random.default_rng(0)
        for gi, g in enumerate(groups):
            v = by_group[g]
            c = self._group_color(g, gi)
            m = float(v.mean())
            sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
            ax.bar(gi, m, width=0.6, color=c, alpha=0.35, zorder=2)
            ax.scatter(gi + rng.uniform(-0.12, 0.12, v.size), v, color=c, s=30, alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.errorbar(gi, m, yerr=sem, color="#263238", capsize=6, lw=1.2, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g or 'All sessions'}\n(n = {by_group[g].size})" for g in groups])
        if key == "steepness":
            ax.set_ylim(0, 1.05)
        elif key == "track0_di":
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="#90a4ae", lw=0.8, zorder=1)
        ax.set_ylabel(axis)
        test, p = self._svc.compare_groups([by_group[g] for g in groups])
        title = f"{label} by group (mean ± SEM, one point per session)"
        if test:
            title += f"; {test} p = {_fmt_p(p)}"
        ax.set_title(title)
        self._tidy(ax)

    def _draw_profiles(self, fig: Any) -> None:
        res = self._res
        z = np.asarray(res.get("z_means"), dtype=float)
        feats = list(res.get("feature_cols", []))
        n = int(res.get("n_states", 0))
        if z.size == 0 or not feats:
            self._empty(fig, "No state profiles.")
            return
        occ = self._pooled_occupancy(res)
        labels = res.get("state_labels", {})
        prof = res.get("state_profiles", {})
        ax = fig.add_subplot(111)
        lim = float(np.nanmax(np.abs(z))) or 1.0
        im = ax.imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(n):
            for j, f in enumerate(feats):
                v = prof.get(i, {}).get(f, np.nan)
                if np.isfinite(v):
                    txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:.2g}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if abs(z[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([_short_feature(f) for f in feats], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{labels.get(i, f'S{i}')}  {100 * occ[i]:.0f}%" for i in range(n)], fontsize=8
        )
        for tick, i in zip(ax.get_yticklabels(), range(n)):
            tick.set_color(_STATE_COLORS[i % len(_STATE_COLORS)])
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Standardized mean (z)")
        ax.set_title("What each state looks like (cell text = mean in raw units; % = share of frames)")

    def _draw_switching(self, fig: Any) -> None:
        res = self._res
        m = np.asarray(res.get("switch_matrix"), dtype=float)
        n = int(res.get("n_states", 0))
        if m.size == 0:
            self._empty(fig, "No state switches.")
            return
        labels = res.get("state_labels", {})
        dwell = res.get("dwell", {})
        ax = fig.add_subplot(111)
        shown = m.copy()
        np.fill_diagonal(shown, np.nan)
        im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                if i == j:
                    d = dwell.get(i, {}).get("median_s", np.nan)
                    ax.text(j, i, f"dwell\n{d:.2f} s", ha="center", va="center",
                            fontsize=7, color="#757575")
                elif np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if m[i, j] > 0.6 else "black")
        names = [labels.get(i, f"S{i}") for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Next state")
        ax.set_ylabel("From state")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("P(next state | leaving)")
        ax.set_title("Where each state goes when it ends (diagonal = median dwell)")

    def _draw_occ_rank(self, fig: Any) -> None:
        res = self._res
        n = int(res.get("n_states", 0))
        occ = res.get("occupancy", {})
        rows = [r for r in self._current_dominance() if r.get("method") == "paired"]
        by_sess: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            k = (str(r["animal_id"]), str(r["session_id"]))
            if k not in occ:
                continue
            role = "dom" if r.get("is_dominant") else "sub"
            by_sess.setdefault(str(r["session_id"]), {})[role] = occ[k]
        pairs = [(d["dom"], d["sub"]) for d in by_sess.values() if "dom" in d and "sub" in d]
        if not pairs:
            self._empty(fig, "Needs dyad sessions with a dominant animal.")
            return
        dom = np.asarray([p[0] for p in pairs], dtype=float) * 100
        sub = np.asarray([p[1] for p in pairs], dtype=float) * 100
        ax = fig.add_subplot(111)
        x = np.arange(n)
        w = 0.36
        ax.bar(x - w / 2, dom.mean(axis=0), w, color="#37474f", label="Dominant", alpha=0.85)
        ax.bar(x + w / 2, sub.mean(axis=0), w, color="#b0bec5", label="Subordinate", alpha=0.95)
        for i in range(len(pairs)):
            for s in range(n):
                ax.plot([s - w / 2, s + w / 2], [dom[i, s], sub[i, s]], color="#78909c",
                        lw=0.5, alpha=0.4, marker="o", ms=2)
        top = float(max(dom.max(), sub.max()))
        ax.set_ylim(0, top * 1.12)
        try:
            from scipy.stats import wilcoxon
            for s in range(n):
                diff = dom[:, s] - sub[:, s]
                if len(pairs) >= 6 and np.any(diff != 0):
                    p = float(wilcoxon(dom[:, s], sub[:, s]).pvalue)
                    ax.text(s, top * 1.04, f"p = {_fmt_p(p)}", ha="center", fontsize=7)
        except Exception:
            pass
        labels = res.get("state_labels", {})
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(s, f"S{s}") for s in range(n)], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("% of tracked frames")
        ax.set_title(f"Do dominant and subordinate animals spend time differently? "
                     f"(n = {len(pairs)} dyads; Wilcoxon signed-rank)")
        ax.legend(fontsize=8, frameon=False)
        self._tidy(ax)

    def _draw_dwell(self, fig: Any) -> None:
        res = self._res
        n = int(res.get("n_states", 0))
        dwell = res.get("dwell", {})
        data = [np.asarray(dwell.get(s, {}).get("samples", []), dtype=float) for s in range(n)]
        if not any(d.size for d in data):
            self._empty(fig, "No dwell times.")
            return
        ax = fig.add_subplot(111)
        bp = ax.boxplot([d if d.size else np.array([np.nan]) for d in data],
                        showfliers=False, patch_artist=True, widths=0.55)
        for s, box in enumerate(bp["boxes"]):
            box.set_facecolor(_STATE_COLORS[s % len(_STATE_COLORS)])
            box.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("#263238")
        ax.set_yscale("log")
        labels = res.get("state_labels", {})
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(
            [f"{labels.get(s, f'S{s}')}\nn = {dwell.get(s, {}).get('n', 0):,}" for s in range(n)],
            rotation=0, fontsize=7,
        )
        ax.set_ylabel("Dwell time (s, log scale)")
        ax.set_title("How long each state lasts once entered")
        self._tidy(ax)

    def _draw_ethogram(self, fig: Any) -> None:
        res = self._res
        sid = str(self._session_combo.currentData() or "")
        if not sid:
            self._empty(fig, "Pick a session.")
            return
        fps = float(res.get("fps", 30.0)) or 30.0
        n = int(res.get("n_states", 0))
        labels = res.get("state_labels", {})
        seqs = res.get("state_seqs", {})
        frames = res.get("frames", {})
        animals = sorted(a for (a, s) in seqs if s == sid)
        if not animals:
            self._empty(fig, "No states for this session.")
            return
        dom = {str(r["animal_id"]): r for r in self._current_dominance() if str(r["session_id"]) == sid}
        ax = fig.add_subplot(111)
        for row, aid in enumerate(animals):
            seq = np.asarray(seqs[(aid, sid)])
            fr = np.asarray(frames.get((aid, sid), np.arange(seq.size)), dtype=float)
            t = fr / fps / 60.0
            if seq.size == 0:
                continue
            step = (1.0 / fps) / 60.0
            change = np.flatnonzero(np.diff(seq) != 0) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [seq.size]))
            for s in range(-1, n):
                sel = seq[starts] == s
                if not sel.any():
                    continue
                spans = [(t[a], t[b - 1] - t[a] + step) for a, b in zip(starts[sel], ends[sel])]
                color = _NO_PARTNER_COLOR if s < 0 else _STATE_COLORS[s % len(_STATE_COLORS)]
                ax.broken_barh(spans, (row - 0.35, 0.7), facecolors=color, linewidth=0)
        ev = [e for e in (res.get("displacement_events") or []) if str(e["session_id"]) == sid]
        for e in ev:
            if e["winner"] in animals:
                row = animals.index(e["winner"])
                ax.plot(e["start_s"] / 60.0, row + 0.45, marker="v", ms=4,
                        color="#212121", alpha=0.7, lw=0)
        ytl = []
        for aid in animals:
            r = dom.get(aid, {})
            extra = ""
            if r:
                extra = f"\nrank {r.get('dominance_rank', '-')}"
                if np.isfinite(float(r.get("displacements_won", np.nan))):
                    extra += f", won {int(r['displacements_won'])}"
            ytl.append(f"{aid}{extra}")
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels(ytl, fontsize=8)
        ax.set_ylim(-0.6, len(animals) - 0.3)
        ax.set_xlabel("Time (min)")
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(color=_STATE_COLORS[s % len(_STATE_COLORS)], label=labels.get(s, f"S{s}"))
                   for s in range(n)]
        handles.append(Patch(color=_NO_PARTNER_COLOR, label="partner not tracked"))
        handles.append(Line2D([], [], marker="v", color="#212121", lw=0, label="displaces partner"))
        ax.legend(handles=handles, fontsize=7, frameon=False, ncol=min(4, len(handles)),
                  loc="upper center", bbox_to_anchor=(0.5, -0.14))
        ax.set_title(f"{self._session_label(sid)}: latent state over time")
        self._tidy(ax)

    # ── Export ──────────────────────────────────────────────────────────────

    def _export_figure(self) -> None:
        if self._fig is None or not self._res:
            QMessageBox.information(self, "Export", "No figure to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)",
        )
        if not path:
            return
        try:
            dpi = int(self._host._graph_settings.get("dpi", 150))
        except Exception:
            dpi = 150
        try:
            self._fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=self._fig.get_facecolor())
            self._status.setText(f"Exported figure to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def result_tables(self) -> dict[str, pd.DataFrame]:
        """Every dominance table as a DataFrame, keyed by CSV file stem."""
        res = self._res
        if not res:
            return {}
        n = int(res.get("n_states", 0))
        labels = res.get("state_labels", {})
        out: dict[str, pd.DataFrame] = {}
        dom = self._current_dominance()
        out["social_dominance"] = pd.DataFrame(dom)
        out["social_dyad_metrics"] = self.dyad_metrics()

        bins = pd.DataFrame(res.get("dominance_bins") or [])
        if not bins.empty:
            bins["session_label"] = bins["session_id"].map(lambda s: self._session_label(str(s)))
            bins["group"] = bins["session_id"].map(lambda s: self._group_of(str(s)))
        out["social_dominance_bins"] = bins

        ev = pd.DataFrame(res.get("displacement_events") or [])
        if not ev.empty:
            ev["session_label"] = ev["session_id"].map(lambda s: self._session_label(str(s)))
            ev["group"] = ev["session_id"].map(lambda s: self._group_of(str(s)))
        out["social_displacement_events"] = ev

        occ_all = self._pooled_occupancy(res)
        dwell = res.get("dwell", {})
        z = np.asarray(res.get("z_means"), dtype=float)
        feats = list(res.get("feature_cols", []))
        prof_rows = []
        for s in range(n):
            row: dict[str, Any] = {
                "state": s,
                "label": labels.get(s, f"S{s}"),
                "interaction": s in res.get("interaction_states", []),
                "occupancy": float(occ_all[s]),
                "n_bouts": dwell.get(s, {}).get("n", 0),
                "mean_dwell_s": dwell.get(s, {}).get("mean_s", np.nan),
                "median_dwell_s": dwell.get(s, {}).get("median_s", np.nan),
            }
            for j, f in enumerate(feats):
                row[f"mean_{f}"] = res.get("state_profiles", {}).get(s, {}).get(f, np.nan)
                if z.ndim == 2 and j < z.shape[1]:
                    row[f"z_{f}"] = float(z[s, j])
            prof_rows.append(row)
        out["social_state_profiles"] = pd.DataFrame(prof_rows)

        rank = {(str(r["animal_id"]), str(r["session_id"])): r for r in dom}
        occ_rows = []
        for (aid, sid), fracs in (res.get("occupancy") or {}).items():
            r = rank.get((aid, sid), {})
            row = {
                "animal_id": aid,
                "session_id": sid,
                "session_label": self._session_label(sid),
                "group": self._group_of(sid),
                "dominance_rank": r.get("dominance_rank", np.nan),
                "is_dominant": r.get("is_dominant", False),
                "no_partner_fraction": (res.get("no_partner_fraction") or {}).get((aid, sid), np.nan),
            }
            for s, v in enumerate(fracs):
                row[labels.get(s, f"S{s}")] = v
            occ_rows.append(row)
        out["social_state_occupancy"] = pd.DataFrame(occ_rows)

        sw = np.asarray(res.get("switch_matrix"), dtype=float)
        if sw.size:
            names = [labels.get(s, f"S{s}") for s in range(n)]
            out["social_state_switching"] = pd.DataFrame(sw, index=names, columns=names).rename_axis("from_state").reset_index()

        if self._summary_rows:
            out["social_summary"] = pd.DataFrame(self._summary_rows)
        return out

    def _export_results(self) -> None:
        tables = self.result_tables()
        if not tables:
            QMessageBox.information(self, "Export", "Run the dominance HMM first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export Dominance Results To")
        if not folder:
            return
        written = []
        try:
            for stem, df in tables.items():
                path = Path(folder) / f"{stem}.csv"
                # BOM so Excel opens the file as UTF-8.
                df.to_csv(path, index=False, encoding="utf-8-sig")
                written.append(path.name)
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", str(exc))
            return
        self._status.setText(f"Wrote {len(written)} CSV file(s) to {folder}.")

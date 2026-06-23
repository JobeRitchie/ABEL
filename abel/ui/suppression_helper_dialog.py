"""Interactive suppression-matrix helper with live synthetic-data preview.

Lets the user specify which behaviors act as suppressors for which other
behaviors (a per-pair weight matrix), see a live preview on synthetic
waveforms that include overlap periods, and observe the effect of
probability-temperature scaling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Lazy matplotlib import (same pattern as temporal_refinement_tab)
# ---------------------------------------------------------------------------
FigureCanvas = None
NavigationToolbar = None
Figure = None
_MPL_OK: bool | None = None


def _ensure_mpl() -> bool:
    global FigureCanvas, NavigationToolbar, Figure, _MPL_OK  # noqa: PLW0603
    if _MPL_OK is not None:
        return _MPL_OK
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt import NavigationToolbar2QT
        from matplotlib.figure import Figure as _Fig

        FigureCanvas = FigureCanvasQTAgg
        NavigationToolbar = NavigationToolbar2QT
        Figure = _Fig
        _MPL_OK = True
    except Exception:
        _MPL_OK = False
    return _MPL_OK


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_BEHAVIOR_COLORS = [
    "#4A90E2",  # blue
    "#7ED321",  # green
    "#9013FE",  # purple
    "#F5A623",  # orange
    "#50E3C2",  # teal
    "#D0021B",  # red
    "#BD10E0",  # magenta
    "#8B572A",  # brown
]


def _generate_synthetic_traces(
    n_behaviors: int,
    n_frames: int = 600,
    fps: float = 30.0,
) -> dict[str, np.ndarray]:
    """Return {name: probability_trace} with deliberate overlap periods.

    Each trace is a smoothed waveform in [0, 1] built from overlapping
    Gaussian bumps.  Placement ensures both isolated bouts and regions
    where two or more behaviors overlap.
    """
    rng = np.random.RandomState(42)
    t = np.arange(n_frames, dtype=np.float64)
    traces: dict[str, np.ndarray] = {}

    # Create a deterministic but varied set of Gaussian bumps per behavior.
    # Strategy: distribute bout centers across the timeline, then force
    # some centers to coincide across behaviors to create overlap.
    sigma_base = n_frames / (n_behaviors * 3.0)  # width of bumps

    all_centers: list[list[float]] = []
    for i in range(n_behaviors):
        # 3-5 bumps per behavior spread across the timeline
        n_bumps = 3 + (i % 3)
        phase_offset = (i / n_behaviors) * n_frames * 0.15
        centers = np.linspace(
            phase_offset + sigma_base,
            n_frames - sigma_base - phase_offset * 0.5,
            n_bumps,
        )
        # Add small jitter
        centers = centers + rng.randn(n_bumps) * sigma_base * 0.3
        all_centers.append(centers.tolist())

    # Force overlap: for every pair of consecutive behaviors, share 1-2 bump
    # centers so their peaks coincide.
    for i in range(n_behaviors - 1):
        shared_idx = rng.randint(0, len(all_centers[i]))
        shared_center = all_centers[i][shared_idx]
        # Nudge second behavior to have a bump near the same spot
        all_centers[i + 1].append(shared_center + rng.randn() * sigma_base * 0.15)

    for i in range(n_behaviors):
        trace = np.zeros(n_frames, dtype=np.float64)
        for center in all_centers[i]:
            sigma = sigma_base * (0.6 + rng.rand() * 0.8)
            amplitude = 0.55 + rng.rand() * 0.40  # 0.55-0.95
            bump = amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
            trace = np.maximum(trace, bump)  # envelope (no additive blow-up)
        # Add mild noise
        trace += rng.rand(n_frames) * 0.04
        trace = np.clip(trace, 0.0, 1.0)
        traces[f"Behavior {i + 1}"] = trace

    return traces


# ---------------------------------------------------------------------------
# Suppression math (mirrors the service-side algorithm)
# ---------------------------------------------------------------------------

def _apply_temperature(
    traces: dict[str, np.ndarray],
    temperature: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if abs(temperature - 1.0) < 1e-6:
        return {k: v.copy() for k, v in traces.items()}
    for bid, raw in traces.items():
        clipped = np.clip(raw, 1e-9, 1.0 - 1e-9)
        logit = np.log(clipped / (1.0 - clipped))
        scaled = logit / temperature
        out[bid] = 1.0 / (1.0 + np.exp(-scaled))
    return out


def _apply_suppression(
    traces: dict[str, np.ndarray],
    suppression_matrix: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Apply per-pair subtractive suppression identical to the service."""
    bids = list(traces.keys())
    out: dict[str, np.ndarray] = {}
    for bid in bids:
        suppressed = traces[bid].copy()
        row = suppression_matrix.get(bid, {})
        for other_bid in bids:
            if other_bid == bid:
                continue
            w = float(row.get(other_bid, 0.0))
            if w > 0:
                suppressed = suppressed - w * traces[other_bid]
        out[bid] = np.clip(suppressed, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class SuppressionHelperDialog(QDialog):
    """Modal dialog for configuring per-behavior suppression weights.

    Parameters
    ----------
    behavior_names : list[str]
        Display names of the active (non-excluded) behaviors.
    behavior_ids : list[str]
        Corresponding IDs (used as keys in the persisted matrix).
    initial_matrix : dict[str, dict[str, float]]
        Pre-existing per-pair weights (may be empty).
    initial_temperature : float
        Current probability-temperature value.
    global_inhibition : float
        Current global inhibition weight (used as a fill preset).
    parent : QWidget | None
    """

    def __init__(
        self,
        behavior_names: list[str],
        behavior_ids: list[str],
        initial_matrix: dict[str, dict[str, float]] | None = None,
        initial_temperature: float = 1.0,
        global_inhibition: float = 0.20,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Suppression Helper — Behavior Inhibition Matrix")
        self.resize(960, 720)

        self._names = list(behavior_names)
        self._ids = list(behavior_ids)
        n = len(self._names)
        self._n = n
        self._initial_matrix = dict(initial_matrix or {})
        self._global_inhibition = global_inhibition

        # ── Debounce timer for live preview ───────────────────────
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)  # ms
        self._redraw_timer.timeout.connect(self._redraw_preview)

        # ── Top: explanation ──────────────────────────────────────
        explanation = QLabel(
            "Configure which behaviors suppress which others during temporal "
            "refinement.  Rows are the <b>suppressed</b> behavior; columns are "
            "the <b>suppressor</b>.  A weight of 0.20 in cell (Groom, Rear) "
            "means <i>Rear suppresses Groom by 0.20 × P(Rear)</i>.\n\n"
            "The preview below shows synthetic probability waveforms before and "
            "after suppression + temperature scaling."
        )
        explanation.setWordWrap(True)

        # ── Matrix table ──────────────────────────────────────────
        matrix_group = QGroupBox("Suppression Matrix")
        matrix_layout = QVBoxLayout(matrix_group)

        self._table = QTableWidget(n, n)
        self._table.setHorizontalHeaderLabels(self._names)
        self._table.setVerticalHeaderLabels(self._names)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setMinimumHeight(max(120, 36 * n + 30))
        self._spinboxes: list[list[QDoubleSpinBox | None]] = []

        for r in range(n):
            row_widgets: list[QDoubleSpinBox | None] = []
            for c in range(n):
                if r == c:
                    # Diagonal — cannot suppress yourself
                    item = QTableWidgetItem("—")
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(r, c, item)
                    row_widgets.append(None)
                else:
                    sb = QDoubleSpinBox()
                    sb.setRange(0.0, 0.50)
                    sb.setSingleStep(0.01)
                    sb.setDecimals(2)
                    # Load from initial_matrix:  matrix[suppressed][suppressor]
                    # Row = suppressed (self._ids[r]), Col = suppressor (self._ids[c])
                    w = float(
                        self._initial_matrix.get(self._ids[r], {}).get(
                            self._ids[c], 0.0
                        )
                    )
                    sb.setValue(w)
                    sb.valueChanged.connect(self._schedule_redraw)
                    self._table.setCellWidget(r, c, sb)
                    row_widgets.append(sb)
            self._spinboxes.append(row_widgets)

        matrix_layout.addWidget(self._table)

        # Preset buttons
        preset_row = QHBoxLayout()
        fill_uniform_btn = QPushButton("Fill Uniform…")
        fill_uniform_btn.setToolTip(
            "Set all off-diagonal cells to the current global inhibition weight."
        )
        fill_uniform_btn.clicked.connect(self._fill_uniform)

        fill_symmetric_btn = QPushButton("Make Symmetric")
        fill_symmetric_btn.setToolTip(
            "Copy the upper triangle into the lower triangle so A→B == B→A."
        )
        fill_symmetric_btn.clicked.connect(self._make_symmetric)

        clear_btn = QPushButton("Clear All")
        clear_btn.setToolTip("Set every cell to 0 (no suppression).")
        clear_btn.clicked.connect(self._clear_all)

        preset_row.addWidget(fill_uniform_btn)
        preset_row.addWidget(fill_symmetric_btn)
        preset_row.addWidget(clear_btn)
        preset_row.addStretch(1)
        matrix_layout.addLayout(preset_row)

        # ── Temperature control ───────────────────────────────────
        temp_group = QGroupBox("Probability Temperature Preview")
        temp_layout = QFormLayout(temp_group)
        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.1, 10.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setDecimals(2)
        self._temperature_spin.setValue(initial_temperature)
        self._temperature_spin.valueChanged.connect(self._schedule_redraw)
        temp_layout.addRow("Temperature:", self._temperature_spin)
        temp_hint = QLabel(
            "<1.0 sharpens (more confident); >1.0 smooths (more conservative). "
            "Preview only — the value on the main panel is what gets saved."
        )
        temp_hint.setWordWrap(True)
        temp_layout.addRow(temp_hint)

        # ── Matplotlib preview ────────────────────────────────────
        preview_group = QGroupBox("Live Preview — Synthetic Waveforms")
        preview_layout = QVBoxLayout(preview_group)

        if _ensure_mpl():
            self._fig = Figure(figsize=(9.0, 5.0), tight_layout=True)
            self._canvas = FigureCanvas(self._fig)
            self._toolbar = NavigationToolbar(self._canvas, self)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            preview_layout.addWidget(self._toolbar)
            preview_layout.addWidget(self._canvas, 1)
            self._has_plot = True
        else:
            self._has_plot = False
            preview_layout.addWidget(
                QLabel("matplotlib is not available — preview disabled.")
            )

        # ── Dialog buttons ────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # ── Main layout ──────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.addWidget(explanation)
        layout.addWidget(matrix_group)
        layout.addWidget(temp_group)
        layout.addWidget(preview_group, 1)
        layout.addWidget(buttons)

        # Generate synthetic data once and keep it fixed.
        self._synthetic = _generate_synthetic_traces(
            max(n, 2), n_frames=600, fps=30.0
        )
        # Rename keys to match behavior names
        renamed: dict[str, np.ndarray] = {}
        synth_keys = list(self._synthetic.keys())
        for i, name in enumerate(self._names):
            renamed[name] = self._synthetic[synth_keys[i]]
        self._synthetic = renamed

        # Initial draw
        if self._has_plot:
            self._redraw_preview()

    # ------------------------------------------------------------------
    # Preset helpers
    # ------------------------------------------------------------------

    def _fill_uniform(self) -> None:
        w = self._global_inhibition
        for r in range(self._n):
            for c in range(self._n):
                sb = self._spinboxes[r][c]
                if sb is not None:
                    sb.setValue(w)

    def _make_symmetric(self) -> None:
        for r in range(self._n):
            for c in range(r + 1, self._n):
                upper = self._spinboxes[r][c]
                lower = self._spinboxes[c][r]
                if upper is not None and lower is not None:
                    lower.setValue(upper.value())

    def _clear_all(self) -> None:
        for r in range(self._n):
            for c in range(self._n):
                sb = self._spinboxes[r][c]
                if sb is not None:
                    sb.setValue(0.0)

    # ------------------------------------------------------------------
    # Extract result
    # ------------------------------------------------------------------

    def suppression_matrix(self) -> dict[str, dict[str, float]]:
        """Return the user-configured matrix as {suppressed_id: {suppressor_id: weight}}."""
        matrix: dict[str, dict[str, float]] = {}
        for r in range(self._n):
            row: dict[str, float] = {}
            for c in range(self._n):
                sb = self._spinboxes[r][c]
                if sb is not None:
                    w = round(sb.value(), 4)
                    if w > 0:
                        row[self._ids[c]] = w
            if row:
                matrix[self._ids[r]] = row
        return matrix

    def temperature(self) -> float:
        return float(self._temperature_spin.value())

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _schedule_redraw(self) -> None:
        self._redraw_timer.start()

    def _read_matrix_by_name(self) -> dict[str, dict[str, float]]:
        """Read the table into a name-keyed matrix (for preview math)."""
        matrix: dict[str, dict[str, float]] = {}
        for r in range(self._n):
            row: dict[str, float] = {}
            for c in range(self._n):
                sb = self._spinboxes[r][c]
                if sb is not None:
                    w = round(sb.value(), 4)
                    if w > 0:
                        row[self._names[c]] = w
            if row:
                matrix[self._names[r]] = row
        return matrix

    def _redraw_preview(self) -> None:
        if not self._has_plot:
            return

        temp = self._temperature_spin.value()
        name_matrix = self._read_matrix_by_name()

        # Apply temperature then suppression to synthetic data.
        after_temp = _apply_temperature(self._synthetic, temp)
        after_all = _apply_suppression(after_temp, name_matrix)

        n_frames = len(next(iter(self._synthetic.values())))
        x = np.arange(n_frames) / 30.0  # seconds

        self._fig.clear()

        # --- Top subplot: raw synthetic traces ---
        ax_raw = self._fig.add_subplot(2, 1, 1)
        for i, name in enumerate(self._names):
            color = _BEHAVIOR_COLORS[i % len(_BEHAVIOR_COLORS)]
            ax_raw.plot(x, self._synthetic[name], color=color, alpha=0.85, label=name)
        ax_raw.set_ylabel("P(behavior)")
        ax_raw.set_title("Raw Probability Traces (synthetic)", fontsize=10)
        ax_raw.set_ylim(-0.05, 1.05)
        ax_raw.legend(loc="upper right", fontsize=7, ncol=min(self._n, 4))
        ax_raw.grid(True, alpha=0.25)

        # Shade overlap regions
        self._shade_overlaps(ax_raw, self._synthetic, x)

        # --- Bottom subplot: after temperature + suppression ---
        ax_post = self._fig.add_subplot(2, 1, 2)
        for i, name in enumerate(self._names):
            color = _BEHAVIOR_COLORS[i % len(_BEHAVIOR_COLORS)]
            ax_post.plot(x, after_all[name], color=color, alpha=0.85, label=name)
        ax_post.set_xlabel("Time (s)")
        ax_post.set_ylabel("P(behavior)")
        title_parts = ["After Suppression"]
        if abs(temp - 1.0) > 0.01:
            title_parts.append(f"+ Temperature {temp:.2f}")
        ax_post.set_title(" ".join(title_parts), fontsize=10)
        ax_post.set_ylim(-0.05, 1.05)
        ax_post.legend(loc="upper right", fontsize=7, ncol=min(self._n, 4))
        ax_post.grid(True, alpha=0.25)

        self._shade_overlaps(ax_post, after_all, x)

        self._canvas.draw_idle()

    @staticmethod
    def _shade_overlaps(
        ax: Any,
        traces: dict[str, np.ndarray],
        x: np.ndarray,
        threshold: float = 0.30,
    ) -> None:
        """Lightly shade time regions where ≥2 behaviors exceed *threshold*."""
        arrays = list(traces.values())
        if len(arrays) < 2:
            return
        above = np.array([arr > threshold for arr in arrays])  # (n_behaviors, n_frames)
        overlap = above.sum(axis=0) >= 2
        # Find contiguous overlap blocks
        diff = np.diff(overlap.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if overlap[0]:
            starts = np.concatenate([[0], starts])
        if overlap[-1]:
            ends = np.concatenate([ends, [len(overlap)]])
        for s, e in zip(starts, ends):
            ax.axvspan(x[s], x[min(e, len(x) - 1)], color="#FF6B6B", alpha=0.10)

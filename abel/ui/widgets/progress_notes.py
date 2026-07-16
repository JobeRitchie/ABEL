"""Compact, self-populating "telemetry" panel for long pipeline runs.

Sits between the progress bar and the run log on the Active Learning tab and
surfaces the most useful facts from the live progress stream — current phase,
which behavior of how many, elapsed / ETA / projected finish, the last model
metric — in a compact, lightly sci-fi readout. It is driven entirely by the
same ``(value, maximum, log_line, status)`` updates that feed the progress bar,
so it needs no extra plumbing: call :meth:`update_from` on each progress event,
:meth:`set_running` when a run starts/stops.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# ── Palette (muted sci-fi: deep navy, cyan/green accents) ──────────────────
_BG = "#070C12"
_BORDER = "#16323C"
_DIM = "#4C6472"          # key labels / inactive
_ACCENT = "#39D0D8"       # cyan — headings, primary values
_GREEN = "#7CFF9E"        # ticker / good metrics
_AMBER = "#FFC24B"        # ETA
_TEXT = "#CFE7EE"         # general values

_ELAPSED_RE = re.compile(r"elapsed\s+([^|]+?)\s*(?:\||$)")
_ETA_RE = re.compile(r"\bETA\s+([^|]+?)\s*(?:\||$)")
_FINISH_RE = re.compile(r"finish ~\s*([0-9:]+)")
# Only ever read the behavior counter from an explicit "behavior N/M" marker so
# unrelated fractions (e.g. "ensemble model 3/3") can never be mistaken for it.
_BEH_RE = re.compile(r"behaviou?r\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_F1_RE = re.compile(r"\bF1=([0-9.]+)")
_PRAUC_RE = re.compile(r"PR-AUC=([0-9.]+)")


class ProgressNotesPanel(QFrame):
    """A compact readout of live run telemetry, populated from progress events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("progressNotes")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"#progressNotes {{ background: {_BG}; border: 1px solid {_BORDER};"
            "  border-radius: 6px; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 7, 10, 7)
        root.setSpacing(5)

        # Header: pulse dot + title + right-aligned finish time.
        head = QHBoxLayout()
        head.setSpacing(7)
        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        title = QLabel("ACTIVE-LEARNING TELEMETRY")
        title.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; font-weight: 700;"
            "letter-spacing: 2px; font-family: 'Consolas','Courier New',monospace;"
        )
        self._finish = QLabel("")
        self._finish.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._finish.setStyleSheet(
            f"color: {_DIM}; font-size: 10px; letter-spacing: 1px;"
            "font-family: 'Consolas','Courier New',monospace;"
        )
        head.addWidget(self._dot)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self._finish)
        root.addLayout(head)

        # Readout grid: two rows of key/value cells.
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)
        self._val_phase = self._add_cell(grid, 0, 0, "PHASE", _TEXT, stretch=2)
        self._val_behavior = self._add_cell(grid, 0, 1, "BEHAVIOR", _TEXT)
        self._val_progress = self._add_cell(grid, 0, 2, "PROGRESS", _ACCENT)
        self._val_elapsed = self._add_cell(grid, 1, 0, "ELAPSED", _TEXT)
        self._val_eta = self._add_cell(grid, 1, 1, "ETA (remaining)", _AMBER)
        self._val_metric = self._add_cell(grid, 1, 2, "LAST MODEL", _GREEN)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        root.addLayout(grid)

        # Ticker: newest interesting event, monospace.
        self._ticker = QLabel("› standby — awaiting run…")
        self._ticker.setStyleSheet(
            f"color: {_DIM}; font-size: 11px;"
            "font-family: 'Consolas','Courier New',monospace;"
        )
        self._ticker.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        root.addWidget(self._ticker)

        # Pulse timer (only runs while active — cheap, single 700 ms tick).
        self._pulse_on = False
        self._pulse = QTimer(self)
        self._pulse.setInterval(700)
        self._pulse.timeout.connect(self._toggle_pulse)
        self._running = False

    # ------------------------------------------------------------------
    def _add_cell(
        self, grid: QGridLayout, row: int, col: int, key: str, value_color: str, *, stretch: int = 1
    ) -> QLabel:
        box = QVBoxLayout()
        box.setSpacing(0)
        k = QLabel(key)
        k.setStyleSheet(
            f"color: {_DIM}; font-size: 8px; font-weight: 700; letter-spacing: 1px;"
            "font-family: 'Consolas','Courier New',monospace;"
        )
        v = QLabel("—")
        v.setStyleSheet(
            f"color: {value_color}; font-size: 12px; font-weight: 600;"
            "font-family: 'Consolas','Courier New',monospace;"
        )
        v.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        box.addWidget(k)
        box.addWidget(v)
        holder = QWidget()
        holder.setLayout(box)
        grid.addWidget(holder, row, col)
        return v

    # ------------------------------------------------------------------
    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        color = _ACCENT if self._pulse_on else _BORDER
        self._dot.setStyleSheet(f"color: {color}; font-size: 11px;")

    def set_running(self, running: bool) -> None:
        """Enter/leave the live-run visual state."""
        self._running = running
        if running:
            if not self._pulse.isActive():
                self._pulse.start()
            self._dot.setStyleSheet(f"color: {_ACCENT}; font-size: 11px;")
        else:
            self._pulse.stop()
            self._dot.setStyleSheet(f"color: {_DIM}; font-size: 11px;")

    def reset(self, message: str = "standby — awaiting run…") -> None:
        for lbl in (
            self._val_phase, self._val_behavior, self._val_progress,
            self._val_elapsed, self._val_eta, self._val_metric,
        ):
            lbl.setText("—")
        self._finish.setText("")
        self._ticker.setStyleSheet(
            f"color: {_DIM}; font-size: 11px;"
            "font-family: 'Consolas','Courier New',monospace;"
        )
        self._ticker.setText(f"› {message}")

    # ------------------------------------------------------------------
    def update_from(
        self, value: int, maximum: int, log_line: str, status: str, step_scale: int = 1
    ) -> None:
        """Refresh all readouts from one progress event. Never raises."""
        try:
            self._update_from(value, maximum, log_line, status, step_scale)
        except Exception:
            pass  # telemetry is cosmetic — never let it break a run

    def _update_from(
        self, value: int, maximum: int, log_line: str, status: str, step_scale: int
    ) -> None:
        line = log_line or ""
        # Everything up to the timing tail is the "event" text.
        event = line.split(" | ", 1)[0].strip().strip("━").strip()

        # Phase — the (broad) status drives this; fall back to the event text.
        phase = (status or event or "").strip().rstrip("…")
        if phase:
            self._val_phase.setText(_clip(phase, 46))

        # Behavior N/M — only from an explicit "behavior N/M" marker.
        m = _BEH_RE.search(status or "") or _BEH_RE.search(line)
        if m:
            self._val_behavior.setText(f"{m.group(1)} / {m.group(2)}")

        # Progress percent from the (global) bar value.
        if maximum > 0:
            pct = max(0.0, min(100.0, 100.0 * value / maximum))
            self._val_progress.setText(f"{pct:.0f}%")

        # Timing fields parsed from the appended tail.
        me = _ELAPSED_RE.search(line)
        if me:
            self._val_elapsed.setText(me.group(1).strip())
        mo = _ETA_RE.search(line)
        if mo:
            eta_txt = mo.group(1).strip()
            if eta_txt.lower().startswith("calculating"):
                # Not enough timing data yet — show a filler and defer the clock.
                self._val_eta.setText("Calculating…")
                self._finish.setText("done ~ estimating…")
            else:
                self._val_eta.setText(eta_txt)
        mf = _FINISH_RE.search(line)
        if mf:
            self._finish.setText(f"done ~ {mf.group(1)}")

        # Last model metric, if this event reported one.
        mf1 = _F1_RE.search(line)
        mpr = _PRAUC_RE.search(line)
        if mf1 or mpr:
            parts = []
            if mf1:
                parts.append(f"F1 {mf1.group(1)}")
            if mpr:
                parts.append(f"PR-AUC {mpr.group(1)}")
            self._val_metric.setText(" · ".join(parts))

        # Ticker — newest event, brightened while running.
        if event:
            self._ticker.setStyleSheet(
                f"color: {_GREEN if self._running else _DIM}; font-size: 11px;"
                "font-family: 'Consolas','Courier New',monospace;"
            )
            self._ticker.setText(f"› {_clip(event, 92)}")


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"

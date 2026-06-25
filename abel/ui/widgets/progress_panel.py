"""A polished, stage-aware progress + ETA panel.

Renders a :class:`~abel.utils.run_timeline.TimelineSnapshot`: an overall bar,
elapsed / remaining / finish-clock readouts, and a per-stage checklist with live
per-stage timings and an animated marker on the active stage.  The panel is a
pure view — it owns no timing logic — so it can be reused by any long-running
tab (Features prep, Active Learning pipeline, …).

Typical use::

    self.progress_panel = ProgressPanel()
    self.progress_panel.set_stages([("preprocess", "Extract features"), ...])
    # on each progress event, from the GUI thread:
    self.progress_panel.update_snapshot(timeline.snapshot())
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from abel.utils.run_timeline import StageView, TimelineSnapshot, format_duration

# Frames for the animated "running" marker (braille spinner).
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_STATE_GLYPH = {
    "pending": "○",
    "running": "",      # filled in live by the spinner
    "done": "✓",
    "skipped": "⏭",
}
_STATE_COLOR = {
    "pending": "#546E7A",
    "running": "#90CAF9",
    "done": "#66BB6A",
    "skipped": "#78909C",
}

_PROGRESS_RESOLUTION = 1000  # sub-percent granularity for a smooth bar


class _StageRow(QWidget):
    """One stage line: state glyph • label • timing."""

    def __init__(self, key: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(8)

        self._glyph = QLabel("○")
        self._glyph.setFixedWidth(16)
        self._glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(label)
        self._label.setStyleSheet("font-size: 12px; color: #B0BEC5;")

        self._timing = QLabel("")
        self._timing.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._timing.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; color: #78909C;"
        )

        row.addWidget(self._glyph)
        row.addWidget(self._label, 1)
        row.addWidget(self._timing)

    def update_view(self, view: StageView, spinner_frame: str) -> None:
        color = _STATE_COLOR.get(view.state, "#B0BEC5")
        glyph = spinner_frame if view.state == "running" else _STATE_GLYPH.get(view.state, "○")
        self._glyph.setText(glyph)
        self._glyph.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 700;")

        label = view.label
        if view.total_units > 1 and view.state in ("running", "done"):
            label = f"{view.label}  ({view.done_units}/{view.total_units})"
        weight = "700" if view.state == "running" else "400"
        self._label.setText(label)
        self._label.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: {weight};")

        if view.state == "done":
            self._timing.setText(format_duration(view.elapsed_seconds or 0.0))
        elif view.state == "running":
            el = format_duration(view.elapsed_seconds or 0.0)
            est = format_duration(view.estimate_seconds)
            self._timing.setText(f"{el} / ~{est}")
        elif view.state == "skipped":
            self._timing.setText("skipped")
        else:
            self._timing.setText(f"~{format_duration(view.estimate_seconds)}")


class ProgressPanel(QWidget):
    """Reusable overall-progress + per-stage + ETA panel."""

    def __init__(self, title: str = "Progress", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, _StageRow] = {}
        self._spinner_idx = 0
        self._last_snapshot: TimelineSnapshot | None = None
        self._provider = None  # optional () -> TimelineSnapshot for live ticking

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("progressCard")
        card.setStyleSheet(
            "#progressCard { background: #0A1929; border: 1px solid #1E3A5F;"
            " border-radius: 6px; }"
        )
        outer.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        self._title = QLabel(title)
        self._title.setStyleSheet(
            "font-size: 13px; font-weight: 800; color: #90CAF9;"
        )
        layout.addWidget(self._title)

        self._bar = QProgressBar()
        self._bar.setRange(0, _PROGRESS_RESOLUTION)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setFixedHeight(20)
        self._bar.setStyleSheet(
            "QProgressBar { background: #0D2137; border: 1px solid #1E3A5F;"
            " border-radius: 4px; color: #E0E0E0; font-size: 11px; text-align: center; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #1976D2, stop:1 #42A5F5); border-radius: 3px; }"
        )
        layout.addWidget(self._bar)

        # Elapsed / Remaining / Finish clock readouts.
        stats = QGridLayout()
        stats.setHorizontalSpacing(18)
        stats.setVerticalSpacing(0)
        self._elapsed = self._make_stat(stats, 0, "Elapsed")
        self._remaining = self._make_stat(stats, 1, "Remaining")
        self._eta = self._make_stat(stats, 2, "Finish ~")
        layout.addLayout(stats)

        self._stage_box = QVBoxLayout()
        self._stage_box.setSpacing(1)
        layout.addLayout(self._stage_box)

        # Repaint the spinner / live clocks even between progress events.
        self._tick = QTimer(self)
        self._tick.setInterval(120)
        self._tick.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_snapshot_provider(self, provider) -> None:
        """Supply a ``() -> TimelineSnapshot`` so the panel can refresh the live
        elapsed/remaining/ETA readouts on its own timer between explicit updates.
        """
        self._provider = provider

    def set_stages(self, stages: list[tuple[str, str]]) -> None:
        """Define the stage rows as ``[(key, label), …]``."""
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        for key, label in stages:
            row = _StageRow(key, label)
            self._rows[key] = row
            self._stage_box.addWidget(row)

    def update_snapshot(self, snapshot: TimelineSnapshot) -> None:
        """Refresh the whole panel from a timeline snapshot."""
        self._last_snapshot = snapshot
        if not self._tick.isActive():
            self._tick.start()
        self._render(advance_spinner=False)

    def reset(self) -> None:
        self._last_snapshot = None
        self._bar.setValue(0)
        self._elapsed.setText("—")
        self._remaining.setText("—")
        self._eta.setText("—")
        for row in self._rows.values():
            row.update_view(
                StageView(row.key, "", "pending", 0, 1, None, 0.0), ""
            )

    def stop(self) -> None:
        """Freeze the panel (stops the spinner timer)."""
        self._tick.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_stat(self, grid: QGridLayout, col: int, caption: str) -> QLabel:
        cap = QLabel(caption)
        cap.setStyleSheet("font-size: 10px; color: #607D8B;")
        val = QLabel("—")
        val.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 13px;"
            " font-weight: 700; color: #CFD8DC;"
        )
        grid.addWidget(cap, 0, col)
        grid.addWidget(val, 1, col)
        return val

    def _on_tick(self) -> None:
        if self._provider is not None:
            try:
                snap = self._provider()
                if snap is not None:
                    self._last_snapshot = snap
            except Exception:
                pass
        self._render(advance_spinner=True)

    def _render(self, *, advance_spinner: bool) -> None:
        snap = self._last_snapshot
        if snap is None:
            return
        if advance_spinner:
            self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
        spinner = _SPINNER[self._spinner_idx]

        self._bar.setValue(int(round(snap.fraction * _PROGRESS_RESOLUTION)))
        self._elapsed.setText(format_duration(snap.elapsed_seconds))
        self._remaining.setText(format_duration(snap.remaining_seconds))

        from datetime import datetime, timedelta
        finish = datetime.now() + timedelta(seconds=snap.remaining_seconds)
        self._eta.setText(finish.strftime("%H:%M:%S"))

        any_running = False
        for view in snap.stages:
            row = self._rows.get(view.key)
            if row is None:
                continue
            row.update_view(view, spinner)
            if view.state == "running":
                any_running = True

        # Once nothing is running, stop animating to save cycles.
        if not any_running and self._tick.isActive() and snap.remaining_seconds <= 0.0:
            self._tick.stop()

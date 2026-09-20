"""Visual dialog for assigning identities and correcting identity swaps.

Multi-animal pose trackers label animals generically (``track_0``, ``Mouse1``…)
and the tracker's identity assignment can *swap* mid-video.  This dialog shows
the session's video with each tracked animal's pose overlaid in a distinct
color and lets the user:

* **Name the animals**: click an animal (or type in the panel) to give its
  track a real identity (``green`` / ``black``).  The names become each
  animal's ``animal_id`` everywhere downstream.
* **Review & correct swaps**: walk the frames flagged by the detector, compare
  the current frame with the one before it, and mark a frame where two animals'
  identities flipped.  A correction exchanges the two tracks from that frame
  onward; the overlay, the "showing track" column and the remaining-swap count
  all update live so the fix can be confirmed rather than assumed.

After acceptance, :attr:`result_map` holds ``{individual: identity}`` and
:attr:`result_corrections` holds ``[{"frame", "a", "b"}, …]``.  :attr:`nav_delta`
is ``+1``/``-1`` when the user asked for the next/previous session.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.services.pose_processing_service import PoseProcessingService
from abel.utils.individual_colors import color_for as _color_for  # shared palette


def _fmt_time(frame: int, fps: float) -> str:
    """``12:34.5`` for a frame index (falls back to frames when fps is unknown)."""
    if not fps or fps <= 0:
        return f"frame {int(frame)}"
    secs = float(frame) / float(fps)
    return f"{int(secs // 60):d}:{secs % 60:04.1f}"


class _ClickableFrame(QLabel):
    """Displays a pixmap and emits image-space coords on click."""

    clicked = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scale = 1.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMinimumSize(360, 240)
        self.setStyleSheet("background: #0a0e18;")

    def set_scale(self, scale: float) -> None:
        self._scale = scale if scale > 0 else 1.0

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        pos = event.position()
        self.clicked.emit(pos.x() / self._scale, pos.y() / self._scale)
        super().mousePressEvent(event)


class AnimalIdentityDialog(QDialog):
    """Assign identities and correct identity swaps for one session.

    Parameters
    ----------
    session_label : str
    multi : MultiAnimalPoseData
        Cleaned per-individual poses (raw identities; corrections are applied
        here for preview only and returned for the caller to persist).
    frame_provider : Callable[[int], np.ndarray | None]
        Returns the session video's BGR frame for a given index (or None).
    n_frames : int
    default_frame : int
        Frame to show first (typically the video midpoint).
    fps : float
        Used only to show times and to size the second-level step buttons.
    current_map : dict[str, str]
    current_corrections : list[dict]
    session_index, session_count : int | None
        0-based position among the project's multi-animal sessions.  Given both,
        the dialog grows Previous/Next Session buttons so the user can walk the
        whole set without reselecting rows in the import table.
    """

    _MAX_W = 720
    _PREV_W = 240

    def __init__(
        self,
        session_label,
        multi,
        frame_provider=None,
        n_frames=0,
        default_frame=0,
        current_map=None,
        current_corrections=None,
        fps=30.0,
        video_path=None,
        session_index=None,
        session_count=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_index = None if session_index is None else int(session_index)
        self._session_count = None if session_count is None else int(session_count)
        pos = ""
        if self._session_index is not None and self._session_count:
            pos = f"  ({self._session_index + 1} of {self._session_count})"
        self.setWindowTitle(f"Animal Identities & Swap Correction: {session_label}{pos}")

        self._multi = multi
        self._individuals = list(multi.individuals)
        self._provider = frame_provider
        self._fps = float(fps or 30.0)
        self._n = max(1, int(n_frames or multi.n_frames))
        self._corrections = [dict(c) for c in (current_corrections or [])]
        self._current_map = dict(current_map or {})
        self._video_path = video_path
        self._frame_idx = min(max(0, int(default_frame)), self._n - 1)
        self.result_map: dict[str, str] = {}
        self.result_corrections: list[dict] = []
        # +1/-1 when the user left via Next/Previous Session; 0 on OK or Cancel.
        self.nav_delta = 0

        # Suspected swaps in the *original* tracking: the review list, fixed so
        # that marking a correction doesn't make entries jump around underneath
        # the user.  ``_remaining`` re-runs the detector on the corrected tracks
        # and is what tells them whether a fix actually landed.
        self._events = PoseProcessingService.analyze_identity_swaps(multi).get("events", [])
        self._remaining: list[dict] = []

        root = QVBoxLayout(self)
        root.addWidget(self._build_help())

        body = QHBoxLayout()
        body.addLayout(self._build_viewer(), 1)
        body.addWidget(self._build_side_panel(), 0)
        root.addLayout(body)
        root.addLayout(self._build_buttons())

        self._refresh_corrections_list()
        self._refresh_events_list()
        self._render()
        self.adjustSize()

    # ── Construction ──────────────────────────────────────────────────
    def _build_help(self) -> QLabel:
        info = QLabel(
            "Each track is drawn in its own color. <b>Name</b> each one on the right "
            "(click an animal in the frame to jump to its name box). If the colors "
            "trade places partway through the video, scrub to the frame where it "
            "happens and press <b>Mark Swap at This Frame</b>, everything from that "
            "frame on is exchanged, and the small “previous frame” view lets you check "
            "the handover."
        )
        info.setWordWrap(True)
        return info

    def _build_viewer(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self._frame = _ClickableFrame()
        self._frame.clicked.connect(self._on_frame_clicked)
        col.addWidget(self._frame)

        # Timeline
        nav = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._n - 1)
        self._slider.setValue(self._frame_idx)
        self._slider.valueChanged.connect(self._on_frame_changed)
        nav.addWidget(self._slider, 1)
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, self._n - 1)
        self._frame_spin.setValue(self._frame_idx)
        self._frame_spin.setPrefix("frame ")
        self._frame_spin.valueChanged.connect(self._on_frame_changed)
        nav.addWidget(self._frame_spin)
        self._time_lbl = QLabel(_fmt_time(self._frame_idx, self._fps))
        nav.addWidget(self._time_lbl)
        col.addLayout(nav)

        # Step buttons: frames for the handover, seconds for getting around.
        steps = QHBoxLayout()
        f1s = max(1, int(round(self._fps)))
        for text, delta, tip in (
            ("−10s", -10 * f1s, "Back 10 seconds"),
            ("−1s", -f1s, "Back 1 second"),
            ("−1", -1, "Back one frame"),
            ("+1", +1, "Forward one frame"),
            ("+1s", +f1s, "Forward 1 second"),
            ("+10s", +10 * f1s, "Forward 10 seconds"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, d=delta: self._step(d))
            steps.addWidget(btn)
        steps.addStretch()
        self._prev_swap_btn = QPushButton("◀ Prev Swap")
        self._next_swap_btn = QPushButton("Next Swap ▶")
        self._prev_swap_btn.clicked.connect(lambda: self._jump_swap(-1))
        self._next_swap_btn.clicked.connect(lambda: self._jump_swap(+1))
        for b in (self._prev_swap_btn, self._next_swap_btn):
            b.setEnabled(bool(self._events))
            steps.addWidget(b)
        col.addLayout(steps)

        # Previous-frame comparison + who-is-who at this frame.
        cmp_row = QHBoxLayout()
        prev_box = QVBoxLayout()
        self._prev_caption = QLabel("previous frame")
        self._prev_caption.setStyleSheet("color: #90A4AE;")
        self._prev_frame = QLabel()
        self._prev_frame.setFixedWidth(self._PREV_W)
        self._prev_frame.setStyleSheet("background: #0a0e18;")
        self._prev_frame.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        prev_box.addWidget(self._prev_caption)
        prev_box.addWidget(self._prev_frame)
        prev_box.addStretch()
        cmp_row.addLayout(prev_box)
        self._frame_note = QLabel()
        self._frame_note.setWordWrap(True)
        self._frame_note.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        cmp_row.addWidget(self._frame_note, 1)
        col.addLayout(cmp_row)
        return col

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        # Width in character widths so the panel keeps its proportions under
        # Windows display scaling instead of clipping the swap descriptions.
        panel.setMinimumWidth(self.fontMetrics().averageCharWidth() * 46)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)

        # ── Identities ────────────────────────────────────────────────
        ident_box = QGroupBox("Identities")
        ident_box.setToolTip(
            "The name becomes this animal's id in every feature row, label and "
            "export for this session."
        )
        grid = QGridLayout(ident_box)
        grid.addWidget(QLabel("<b>Name</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Showing at this frame</b>"), 0, 2)
        self._edits: dict[str, QLineEdit] = {}
        self._source_lbls: dict[str, QLabel] = {}
        cur = dict(self._current_map)
        for idx, ind in enumerate(self._individuals):
            r, g, b = _color_for(idx)
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid #222;")
            grid.addWidget(swatch, idx + 1, 0)
            edit = QLineEdit(str(cur.get(ind, ind)))
            edit.setPlaceholderText("identity (e.g. green)")
            edit.textChanged.connect(self._render)
            self._edits[ind] = edit
            grid.addWidget(edit, idx + 1, 1)
            src = QLabel(ind)
            src.setStyleSheet("color: #90A4AE;")
            self._source_lbls[ind] = src
            grid.addWidget(src, idx + 1, 2)
        side.addWidget(ident_box)

        # ── Suspected swaps ───────────────────────────────────────────
        swap_box = QGroupBox("Suspected swaps")
        swap_layout = QVBoxLayout(swap_box)
        self._swap_banner = QLabel()
        self._swap_banner.setWordWrap(True)
        swap_layout.addWidget(self._swap_banner)
        self._events_list = QListWidget()
        self._events_list.setWordWrap(True)
        self._events_list.setMinimumHeight(110)
        self._events_list.itemActivated.connect(self._jump_to_event)
        self._events_list.currentItemChanged.connect(
            lambda cur, _prev: self._jump_to_event(cur)
        )
        swap_layout.addWidget(self._events_list)
        side.addWidget(swap_box)

        # ── Corrections ───────────────────────────────────────────────
        corr_box = QGroupBox("Corrections")
        corr_layout = QVBoxLayout(corr_box)
        pair_row = QHBoxLayout()
        if len(self._individuals) > 2:
            self._swap_a = QComboBox()
            self._swap_a.addItems(self._individuals)
            self._swap_b = QComboBox()
            self._swap_b.addItems(self._individuals)
            self._swap_b.setCurrentIndex(1)
            pair_row.addWidget(QLabel("Swap"))
            pair_row.addWidget(self._swap_a)
            pair_row.addWidget(QLabel("↔"))
            pair_row.addWidget(self._swap_b)
        else:
            self._swap_a = self._swap_b = None
        mark_btn = QPushButton("Mark Swap at This Frame")
        mark_btn.setToolTip(
            "Exchange the two animals' tracks from this frame to the end of the "
            "session. Pressing it again on the same frame undoes it."
        )
        mark_btn.clicked.connect(self._mark_swap)
        pair_row.addWidget(mark_btn)
        corr_layout.addLayout(pair_row)

        self._auto_btn = QPushButton("Auto-Detect Swaps from Appearance")
        self._auto_btn.setToolTip(
            "When the animals differ in coat color (a black and a white mouse), "
            "the pixels at each track's centroid say which is which frame by frame. "
            "This reads the video and proposes the swap corrections that put the "
            "whole session on one consistent identity. It says so instead of "
            "guessing when the animals look alike."
        )
        self._auto_btn.setEnabled(self._video_path is not None)
        self._auto_btn.clicked.connect(self._auto_detect)
        corr_layout.addWidget(self._auto_btn)

        self._corr_list = QListWidget()
        self._corr_list.setMaximumHeight(80)
        self._corr_list.itemActivated.connect(self._jump_to_correction)
        corr_layout.addWidget(self._corr_list)
        remove_btn = QPushButton("Remove Selected Correction")
        remove_btn.clicked.connect(self._remove_correction)
        corr_layout.addWidget(remove_btn)
        side.addWidget(corr_box)
        side.addStretch()
        return panel

    def _build_buttons(self) -> QHBoxLayout:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        # Session navigation: saves this session's identities and reopens the
        # dialog on the neighboring multi-animal session, so mapping a whole
        # project is one pass instead of a select-open-close cycle per session.
        row = QHBoxLayout()
        if self._session_index is not None and (self._session_count or 0) > 1:
            self._prev_session_btn = QPushButton("◀ Previous Session")
            self._next_session_btn = QPushButton("Next Session ▶")
            self._prev_session_btn.setToolTip(
                "Save these identities and open the previous multi-animal session (Alt+Left)."
            )
            self._next_session_btn.setToolTip(
                "Save these identities and open the next multi-animal session (Alt+Right)."
            )
            self._prev_session_btn.setShortcut(QKeySequence("Alt+Left"))
            self._next_session_btn.setShortcut(QKeySequence("Alt+Right"))
            self._prev_session_btn.setEnabled(self._session_index > 0)
            self._next_session_btn.setEnabled(self._session_index < self._session_count - 1)
            self._prev_session_btn.clicked.connect(lambda: self._go_session(-1))
            self._next_session_btn.clicked.connect(lambda: self._go_session(+1))
            row.addWidget(self._prev_session_btn)
            row.addWidget(self._next_session_btn)
        else:
            self._prev_session_btn = self._next_session_btn = None
        row.addStretch()
        row.addWidget(buttons)
        return row

    # ── Preview helpers ───────────────────────────────────────────────
    def _perm_at(self, frame: int) -> dict[str, str]:
        """Effective identity→source mapping after corrections up to ``frame``."""
        perm = {o: o for o in self._individuals}
        for c in sorted(self._corrections, key=lambda c: int(c["frame"])):
            if int(c["frame"]) <= frame:
                a, b = c["a"], c["b"]
                if a in perm and b in perm:
                    perm[a], perm[b] = perm[b], perm[a]
        return perm

    def _points_at(self, source_ind: str, frame: int):
        pose = self._multi.per_individual.get(source_ind)
        if pose is None or frame < 0 or frame >= pose.n_frames:
            return [], (float("nan"), float("nan"))
        xs, ys = pose.x.iloc[frame], pose.y.iloc[frame]
        pts = [(float(xs[bp]), float(ys[bp])) for bp in pose.body_parts]
        cen = (float(pose.centroid_x[frame]), float(pose.centroid_y[frame]))
        return pts, cen

    def _current_centroids(self) -> dict[str, tuple[float, float]]:
        perm = self._perm_at(self._frame_idx)
        return {o: self._points_at(perm[o], self._frame_idx)[1] for o in self._individuals}

    def _label_for(self, ind: str) -> str:
        edit = self._edits.get(ind)
        return (edit.text().strip() if edit else "") or ind

    # ── Rendering ─────────────────────────────────────────────────────
    def _render(self) -> None:
        self._time_lbl.setText(_fmt_time(self._frame_idx, self._fps))
        self._render_main()
        self._render_prev()
        self._render_note()

    def _paint_overlay(self, pix: QPixmap, frame: int, scale: float, with_text: bool) -> None:
        perm = self._perm_at(frame)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for idx, ind in enumerate(self._individuals):
            r, g, b = _color_for(idx)
            col = QColor(r, g, b)
            pts, cen = self._points_at(perm[ind], frame)
            painter.setPen(QPen(col, 2))
            for (x, y) in pts:
                if np.isfinite(x) and np.isfinite(y):
                    painter.drawEllipse(int(x * scale) - 3, int(y * scale) - 3, 6, 6)
            if np.isfinite(cen[0]) and np.isfinite(cen[1]):
                cx, cy = int(cen[0] * scale), int(cen[1] * scale)
                painter.setPen(QPen(col, 3))
                painter.drawEllipse(cx - 11, cy - 11, 22, 22)
                if with_text:
                    painter.drawText(cx + 13, cy + 4, self._label_for(ind))
        painter.end()

    def _pixmap_for(self, frame: int, max_w: int, with_text: bool):
        bgr = self._provider(frame) if self._provider else None
        if bgr is None:
            return None
        arr = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scale = min(1.0, max_w / float(w)) if w else 1.0
        if scale < 1.0:
            pix = pix.scaled(
                int(w * scale), int(h * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._paint_overlay(pix, frame, scale, with_text)
        return pix, scale

    def _render_main(self) -> None:
        got = self._pixmap_for(self._frame_idx, self._MAX_W, True)
        if got is None:
            self._frame.setText(
                "(no video frame available: identities can still be named on the right)"
            )
            self._frame.setStyleSheet("color: #90A4AE; padding: 24px; background: #0a0e18;")
            self._frame.set_scale(1.0)
            return
        pix, scale = got
        self._frame.set_scale(scale)
        self._frame.setPixmap(pix)
        self._frame.setFixedSize(pix.size())

    def _render_prev(self) -> None:
        prev_idx = max(0, self._frame_idx - 1)
        self._prev_caption.setText(
            f"frame {prev_idx}, the frame before this one; a swap shows up as the "
            "colors trading animals between these two views."
        )
        self._prev_caption.setWordWrap(True)
        got = self._pixmap_for(prev_idx, self._PREV_W, False)
        if got is None:
            self._prev_frame.clear()
            return
        pix, _scale = got
        self._prev_frame.setPixmap(pix)
        self._prev_frame.setFixedSize(pix.size())

    def _render_note(self) -> None:
        """Spell out which source track each identity is showing right now."""
        perm = self._perm_at(self._frame_idx)
        swapped = [o for o in self._individuals if perm[o] != o]
        for ind in self._individuals:
            src = perm[ind]
            lbl = self._source_lbls[ind]
            lbl.setText(src if src == ind else f"{src}  (corrected)")
            lbl.setStyleSheet("color: #FFB74D;" if src != ind else "color: #90A4AE;")

        cents = self._current_centroids()
        gap = ""
        vals = [c for c in cents.values() if np.isfinite(c[0]) and np.isfinite(c[1])]
        if len(vals) == 2:
            d = float(np.hypot(vals[0][0] - vals[1][0], vals[0][1] - vals[1][1]))
            gap = f"Animals are {d:.0f} px apart. "
            if d < 30:
                gap += (
                    "They overlap here, so the tracker's assignment is a guess, "
                    "step to a frame where they separate before judging a swap. "
                )
        state = (
            f"Corrections active at this frame: {', '.join(f'{o}←{perm[o]}' for o in swapped)}."
            if swapped else "No correction applies at this frame: tracks are as the tracker left them."
        )
        self._frame_note.setText(gap + state)

    # ── Navigation ────────────────────────────────────────────────────
    def _on_frame_changed(self, value: int) -> None:
        value = min(max(0, int(value)), self._n - 1)
        if value == self._frame_idx:
            return
        self._frame_idx = value
        for w in (self._slider, self._frame_spin):
            w.blockSignals(True)
            w.setValue(value)
            w.blockSignals(False)
        self._render()

    def _step(self, delta: int) -> None:
        self._on_frame_changed(self._frame_idx + int(delta))

    def _event_frames(self) -> list[int]:
        return sorted({int(e["frame"]) for e in self._events})

    def _jump_swap(self, direction: int) -> None:
        frames = self._event_frames()
        if not frames:
            return
        cur = self._frame_idx
        if direction > 0:
            nxt = next((f for f in frames if f > cur), frames[-1])
        else:
            nxt = next((f for f in reversed(frames) if f < cur), frames[0])
        self._on_frame_changed(nxt)

    def _jump_to_event(self, item: "QListWidgetItem | None") -> None:
        if item is None:
            return
        frame = item.data(Qt.ItemDataRole.UserRole)
        if frame is not None:
            self._on_frame_changed(int(frame))

    def _jump_to_correction(self, item: "QListWidgetItem | None") -> None:
        if item is None:
            return
        frame = item.data(Qt.ItemDataRole.UserRole)
        if frame is not None:
            self._on_frame_changed(int(frame))

    # ── Interaction ───────────────────────────────────────────────────
    def _on_frame_clicked(self, ix, iy) -> None:
        centroids = self._current_centroids()
        nearest, best = None, None
        for ind, cen in centroids.items():
            if not (np.isfinite(cen[0]) and np.isfinite(cen[1])):
                continue
            d = (cen[0] - ix) ** 2 + (cen[1] - iy) ** 2
            if best is None or d < best:
                best, nearest = d, ind
        if nearest is not None:
            self._edits[nearest].setFocus()
            self._edits[nearest].selectAll()

    def _mark_swap(self) -> None:
        if len(self._individuals) < 2:
            return
        if self._swap_a is not None and self._swap_b is not None:
            a, b = self._swap_a.currentText(), self._swap_b.currentText()
        else:
            a, b = self._individuals[0], self._individuals[1]
        if a == b:
            return
        frame = self._frame_idx
        existing = next(
            (c for c in self._corrections
             if int(c["frame"]) == frame and {c["a"], c["b"]} == {a, b}),
            None,
        )
        if existing:  # toggle off
            self._corrections.remove(existing)
        else:
            self._corrections.append({"frame": frame, "a": a, "b": b})
        self._refresh_corrections_list()
        self._refresh_events_list()
        self._render()

    def _auto_detect(self) -> None:
        """Propose corrections from coat appearance, for the user to accept."""
        if self._video_path is None:
            return
        from abel.services.appearance_identity_service import (  # noqa: PLC0415
            analyze_appearance_identity,
        )

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = analyze_appearance_identity(self._video_path, self._multi)
        except Exception as exc:  # pragma: no cover - decode failures
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Appearance check failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        if not result.usable:
            QMessageBox.information(self, "Animals look alike", result.message)
            return

        proposed = [dict(c) for c in result.corrections]
        same = sorted((int(c["frame"]), c["a"], c["b"]) for c in proposed) == sorted(
            (int(c["frame"]), c["a"], c["b"]) for c in self._corrections
        )
        if same:
            QMessageBox.information(self, "Already correct", "Your corrections match " + result.message)
            return
        answer = QMessageBox.question(
            self,
            "Apply detected swaps?",
            f"{result.message}\n\nReplace the {len(self._corrections)} correction(s) "
            f"on this session with the {len(proposed)} detected one(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._corrections = proposed
        self._refresh_corrections_list()
        self._refresh_events_list()
        self._render()
        if proposed:
            self._on_frame_changed(int(proposed[0]["frame"]))

    def _refresh_corrections_list(self) -> None:
        self._corr_list.clear()
        for c in sorted(self._corrections, key=lambda c: int(c["frame"])):
            f = int(c["frame"])
            item = QListWidgetItem(
                f"from frame {f} ({_fmt_time(f, self._fps)}):  {c['a']} ↔ {c['b']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._corr_list.addItem(item)

    def _corrected_multi(self):
        if not self._corrections:
            return self._multi
        return PoseProcessingService.apply_identity_corrections(self._multi, self._corrections)

    def _refresh_events_list(self) -> None:
        """Re-list the detector's events and re-check them against the corrections.

        The list itself stays anchored to the original tracking; the banner
        reports how many suspected swaps survive the user's corrections, which is
        the only honest answer to "did my fix work?".
        """
        self._remaining = PoseProcessingService.analyze_identity_swaps(
            self._corrected_multi()
        ).get("events", [])
        remaining_frames = {int(e["frame"]) for e in self._remaining}
        corrected_frames = {int(c["frame"]) for c in self._corrections}

        self._events_list.blockSignals(True)
        self._events_list.clear()
        for ev in self._events:
            f = int(ev["frame"])
            near = any(abs(f - cf) <= 2 for cf in corrected_frames)
            mark = "✔" if (near and f not in remaining_frames) else ("•" if near else "-")
            notes = [f"{ev['a']} ↔ {ev['b']}"]
            if ev["n_flips"] > 1:
                notes.append(f"{ev['n_flips']} flips{'' if ev['net'] else ', canceling out'}")
            if ev.get("overlapping"):
                notes.append("animals overlapping: check the video")
            if ev.get("unreliable"):
                notes.append("keypoints scattered: pose lost, probably not a swap")
            item = QListWidgetItem(
                f"{mark}  frame {f}  ({_fmt_time(f, self._fps)})\n     " + ", ".join(notes)
            )
            item.setData(Qt.ItemDataRole.UserRole, f)
            if ev.get("overlapping") or ev.get("unreliable"):
                item.setForeground(QColor("#FFB74D"))
            self._events_list.addItem(item)
        self._events_list.blockSignals(False)

        for b in (self._prev_swap_btn, self._next_swap_btn):
            b.setEnabled(bool(self._events))
        if not self._events:
            self._swap_banner.setText(
                "The automatic check found no identity swaps. Spot-check a few frames "
                "anyway: it only sees swaps where the animals jump."
            )
            self._swap_banner.setStyleSheet("")
            return
        net = sum(1 for e in self._events if e["net"])
        text = (
            f"{len(self._events)} suspected swap(s) in the original tracking "
            f"({net} leave the identities exchanged). "
            f"{len(self._remaining)} still suspected after your corrections."
        )
        self._swap_banner.setText(("⚠  " if self._remaining else "✔  ") + text)
        self._swap_banner.setStyleSheet(
            "color: #FFB74D; background: #2E2A0A; border: 1px solid #6D5A00;"
            " border-radius: 4px; padding: 6px;"
            if self._remaining else
            "color: #A5D6A7; background: #16301A; border: 1px solid #2E5E36;"
            " border-radius: 4px; padding: 6px;"
        )

    def _remove_correction(self) -> None:
        row = self._corr_list.currentRow()
        ordered = sorted(self._corrections, key=lambda c: int(c["frame"]))
        if 0 <= row < len(ordered):
            self._corrections.remove(ordered[row])
            self._refresh_corrections_list()
            self._refresh_events_list()
            self._render()

    # ── Results ───────────────────────────────────────────────────────
    def _commit_results(self) -> None:
        self.result_map = {
            ind: (edit.text().strip() or ind) for ind, edit in self._edits.items()
        }
        self.result_corrections = [
            {"frame": int(c["frame"]), "a": c["a"], "b": c["b"]} for c in self._corrections
        ]

    def _on_accept(self) -> None:
        self._commit_results()
        self.nav_delta = 0
        self.accept()

    def _go_session(self, delta: int) -> None:
        """Save this session and ask the caller to open a neighboring one."""
        self._commit_results()
        self.nav_delta = int(delta)
        self.accept()

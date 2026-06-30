"""Visual dialog for assigning identities and correcting identity swaps.

Multi-animal DeepLabCut exports label animals generically (``Mouse1``,
``Mouse2``…) and the tracker's identity assignment can *swap* mid-video.  This
dialog shows the session's video with each tracked animal's pose overlaid in a
distinct colour and lets the user:

* **Assign identities** — click an animal (or use the fields) to name it
  (``green`` / ``black``).  The names become each animal's ``animal_id``.
* **Review & correct swaps** — step through frames (or jump between suspected
  swap frames flagged by the detector), and mark a frame where two animals'
  identities flipped.  A correction exchanges the two tracks from that frame
  onward; the overlay updates live so the user can confirm the fix.

After acceptance, :attr:`result_map` holds ``{individual: identity}`` and
:attr:`result_corrections` holds ``[{"frame", "a", "b"}, …]``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_PALETTE = [
    (255, 80, 80), (80, 160, 255), (80, 220, 120),
    (240, 200, 60), (210, 110, 240), (90, 230, 230),
]


def _color_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


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
        Cleaned per-individual poses (raw identities; corrections applied here
        only for preview).
    frame_provider : Callable[[int], np.ndarray | None]
        Returns the session video's BGR frame for a given index (or None).
    n_frames : int
    default_frame : int
        Frame to show first (typically the video midpoint).
    swap_frames : list[int]
        Suspected swap frame indices from ``detect_identity_swaps``.
    current_map : dict[str, str]
    current_corrections : list[dict]
    """

    _MAX_W = 760

    def __init__(
        self,
        session_label,
        multi,
        frame_provider=None,
        n_frames=0,
        default_frame=0,
        swap_frames=None,
        current_map=None,
        current_corrections=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Animal Identities & Swap Correction — {session_label}")
        self._multi = multi
        self._individuals = list(multi.individuals)
        self._provider = frame_provider
        self._n = max(1, int(n_frames or multi.n_frames))
        self._swap_frames = sorted(int(f) for f in (swap_frames or []))
        self._corrections = [dict(c) for c in (current_corrections or [])]
        self._frame_idx = min(max(0, int(default_frame)), self._n - 1)
        self.result_map: dict[str, str] = {}
        self.result_corrections: list[dict] = []

        layout = QVBoxLayout(self)
        info = QLabel(
            "Each tracked animal is outlined in a colour below. Click an animal to set "
            "its identity. Use the frame controls to review suspected identity swaps; "
            "if two animals' colours have flipped, press “Mark swap at this frame”."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        swap_note = (
            f"{len(self._swap_frames)} suspected identity swap(s) detected."
            if self._swap_frames else "No identity swaps detected by the automatic check."
        )
        self._swap_banner = QLabel(("⚠  " + swap_note) if self._swap_frames else swap_note)
        self._swap_banner.setWordWrap(True)
        if self._swap_frames:
            self._swap_banner.setStyleSheet(
                "color: #FFB74D; background: #2E2A0A; border: 1px solid #6D5A00;"
                " border-radius: 4px; padding: 6px;"
            )
        layout.addWidget(self._swap_banner)

        self._frame = _ClickableFrame()
        self._frame.clicked.connect(self._on_frame_clicked)
        layout.addWidget(self._frame)

        # ── Frame navigation ──────────────────────────────────────────
        nav = QHBoxLayout()
        nav.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, self._n - 1)
        self._frame_spin.setValue(self._frame_idx)
        self._frame_spin.valueChanged.connect(self._on_frame_changed)
        nav.addWidget(self._frame_spin)
        prev_swap = QPushButton("◀ Prev swap")
        next_swap = QPushButton("Next swap ▶")
        prev_swap.clicked.connect(lambda: self._jump_swap(-1))
        next_swap.clicked.connect(lambda: self._jump_swap(+1))
        prev_swap.setEnabled(bool(self._swap_frames))
        next_swap.setEnabled(bool(self._swap_frames))
        nav.addWidget(prev_swap)
        nav.addWidget(next_swap)
        nav.addStretch()
        layout.addLayout(nav)

        # ── Swap-correction controls ──────────────────────────────────
        swap_row = QHBoxLayout()
        if len(self._individuals) > 2:
            self._swap_a = QComboBox(); self._swap_a.addItems(self._individuals)
            self._swap_b = QComboBox(); self._swap_b.addItems(self._individuals)
            self._swap_b.setCurrentIndex(1)
            swap_row.addWidget(QLabel("Swap"))
            swap_row.addWidget(self._swap_a)
            swap_row.addWidget(QLabel("↔"))
            swap_row.addWidget(self._swap_b)
        else:
            self._swap_a = self._swap_b = None
        mark_btn = QPushButton("Mark swap at this frame")
        mark_btn.setToolTip("Exchange the two animals' tracks from this frame onward.")
        mark_btn.clicked.connect(self._mark_swap)
        swap_row.addWidget(mark_btn)
        swap_row.addStretch()
        layout.addLayout(swap_row)

        self._corr_list = QListWidget()
        self._corr_list.setMaximumHeight(90)
        layout.addWidget(QLabel("Applied corrections (select + Remove to undo):"))
        layout.addWidget(self._corr_list)
        remove_btn = QPushButton("Remove selected correction")
        remove_btn.clicked.connect(self._remove_correction)
        layout.addWidget(remove_btn)

        # ── Identity name fields ──────────────────────────────────────
        self._edits: dict[str, QLineEdit] = {}
        cur = dict(current_map or {})
        for idx, ind in enumerate(self._individuals):
            row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 2, 0, 2)
            r, g, b = _color_for(idx)
            swatch = QLabel(); swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid #222;")
            hl.addWidget(swatch)
            hl.addWidget(QLabel(f"{ind}:"))
            edit = QLineEdit(str(cur.get(ind, ind)))
            edit.setPlaceholderText("identity (e.g. green)")
            hl.addWidget(edit, 1)
            self._edits[ind] = edit
            layout.addWidget(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_corrections_list()
        self._render()
        self.adjustSize()

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
        out = {}
        for o in self._individuals:
            _pts, cen = self._points_at(perm[o], self._frame_idx)
            out[o] = cen
        return out

    # ── Rendering ─────────────────────────────────────────────────────
    def _render(self) -> None:
        bgr = self._provider(self._frame_idx) if self._provider else None
        if bgr is None:
            self._frame.setText("(no video frame available — identities can still be named below)")
            self._frame.setStyleSheet("color: #90A4AE; padding: 24px; background: #0a0e18;")
            self._frame.set_scale(1.0)
            return
        arr = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scale = min(1.0, self._MAX_W / float(w)) if w else 1.0
        if scale < 1.0:
            pix = pix.scaled(int(w * scale), int(h * scale),
                             Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self._frame.set_scale(scale)

        perm = self._perm_at(self._frame_idx)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for idx, ind in enumerate(self._individuals):
            r, g, b = _color_for(idx)
            col = QColor(r, g, b)
            pts, cen = self._points_at(perm[ind], self._frame_idx)
            painter.setPen(QPen(col, 2))
            for (x, y) in pts:
                if np.isfinite(x) and np.isfinite(y):
                    painter.drawEllipse(int(x * scale) - 3, int(y * scale) - 3, 6, 6)
            if np.isfinite(cen[0]) and np.isfinite(cen[1]):
                cx, cy = int(cen[0] * scale), int(cen[1] * scale)
                painter.setPen(QPen(col, 3))
                painter.drawEllipse(cx - 11, cy - 11, 22, 22)
                label = self._edits[ind].text().strip() if ind in self._edits else ind
                painter.drawText(cx + 13, cy + 4, label or ind)
        painter.end()
        self._frame.setPixmap(pix)
        self._frame.setFixedSize(pix.size())

    # ── Navigation ────────────────────────────────────────────────────
    def _on_frame_changed(self, value: int) -> None:
        self._frame_idx = int(value)
        self._render()

    def _jump_swap(self, direction: int) -> None:
        if not self._swap_frames:
            return
        cur = self._frame_idx
        if direction > 0:
            nxt = next((f for f in self._swap_frames if f > cur), self._swap_frames[-1])
        else:
            nxt = next((f for f in reversed(self._swap_frames) if f < cur), self._swap_frames[0])
        self._frame_spin.setValue(nxt)

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
        self._render()

    def _refresh_corrections_list(self) -> None:
        self._corr_list.clear()
        for c in sorted(self._corrections, key=lambda c: int(c["frame"])):
            self._corr_list.addItem(f"frame {int(c['frame'])}:  {c['a']} ↔ {c['b']}")

    def _remove_correction(self) -> None:
        row = self._corr_list.currentRow()
        ordered = sorted(self._corrections, key=lambda c: int(c["frame"]))
        if 0 <= row < len(ordered):
            self._corrections.remove(ordered[row])
            self._refresh_corrections_list()
            self._render()

    def _on_accept(self) -> None:
        self.result_map = {
            ind: (edit.text().strip() or ind) for ind, edit in self._edits.items()
        }
        self.result_corrections = [
            {"frame": int(c["frame"]), "a": c["a"], "b": c["b"]} for c in self._corrections
        ]
        self.accept()

"""Pop-out behavior "soundboard" for rapid clip review.

Two modes:

* **Single-animal** (no animals supplied): one button per behavior; clicking
  labels the current clip through the review tab's normal path.  Identical to
  pressing the hotkey.

* **Multi-animal** (animals supplied via :meth:`set_animals`): the whole-window
  clip may contain several animals, so a label must say *which* animal(s) it is
  about.  You pick a behavior, then designate the animal (solo) or the two
  animals (social) using the colored animal buttons.  Directed social behaviors
  ask for the *actor* then the *recipient*; mutual ones just ask for the pair.
  Each structured label ``(behavior, focal, partner)`` is emitted to the review
  tab and shown in a per-clip list.

Because this is a separate top-level window, the review tab's ``WindowShortcut``
hotkeys don't fire while it's focused, so navigation keys are forwarded back to
the review-tab actions here.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


_SOLO_BTN_QSS = (
    "QPushButton{background:#1E2A36;color:#ECEFF1;border:1px solid #33475B;"
    "border-radius:8px;padding:6px 10px;font-weight:600;}"
    "QPushButton:hover{background:#26374A;border-color:#4A6377;}"
)
_SOCIAL_BTN_QSS = (
    "QPushButton{background:#241E36;color:#ECEFF1;border:1px solid #7E57C2;"
    "border-radius:8px;padding:6px 10px;font-weight:600;}"
    "QPushButton:hover{background:#2E2547;border-color:#9575CD;}"
)
_WINDOW_QSS = (
    "#hint{color:#8A97A3;font-size:11px;}"
    "#status{color:#E3F2FD;font-weight:600;font-size:12px;}"
    "#section{color:#90A4AE;font-weight:700;font-size:11px;}"
    "#divider{color:#2A3A47;}"
    "#chip{background:#16212B;border:1px solid #2A3A47;border-radius:6px;}"
    "#chipRemove{background:transparent;border:none;color:#EF9A9A;font-weight:700;}"
    "#chipRemove:hover{color:#EF5350;}"
    "#commit{background:#2E7D32;color:#FFFFFF;border:none;border-radius:8px;padding:9px;font-weight:700;}"
    "#commit:hover{background:#388E3C;}"
    "#commit:disabled{background:#33475B;color:#78909C;}"
)


class BehaviorSoundboard(QWidget):
    """Non-modal window: behavior buttons + per-animal designation + key pass-through."""

    def __init__(self, parent=None, columns: int = 4) -> None:
        super().__init__(parent)
        self.setWindowTitle("Behavior Soundboard")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._columns = max(1, int(columns))

        # Callbacks
        self._on_behavior: Callable[[str], None] = lambda _bid: None          # single-animal path
        self._on_structured: Callable[[str, str, "str | None"], None] = lambda *_: None
        self._on_commit: Callable[[list], None] = lambda _labels: None         # persist clip labels
        self._nav: dict[str, Callable[[], None]] = {}

        # State
        self._behaviors: list[tuple] = []          # (bid, name, key, is_social, directionality)
        self._animals: list[tuple] = []            # (animal_id, name, (r,g,b))
        self._key_to_behavior: dict[str, str] = {}
        self._selected_animal: str | None = None
        self._pending_social: dict | None = None   # {bid, directionality, picked:[...]}
        self._clip_labels: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        self._hint = QLabel()
        self._hint.setObjectName("hint")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._on_top_chk = QCheckBox("Keep window on top")
        self._on_top_chk.toggled.connect(self._toggle_on_top)
        root.addWidget(self._on_top_chk)

        # Animal selector row (multi-animal only)
        self._animal_bar = QWidget()
        self._animal_bar_layout = QHBoxLayout(self._animal_bar)
        self._animal_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._animal_bar_layout.setSpacing(6)
        self._animal_btns: list[QPushButton] = []
        root.addWidget(self._animal_bar)

        # Status line (guides multi-step social designation)
        self._status = QLabel("")
        self._status.setObjectName("status")
        self._status.setMinimumHeight(16)
        root.addWidget(self._status)

        # Behavior button grid — compact buttons, top-left aligned (stretch
        # absorbers keep them at natural size instead of filling the window).
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._grid_host)
        root.addWidget(self._scroll, 3)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(divider)

        # Per-clip structured label list (given real vertical space).
        header = QLabel("LABELS ON THIS CLIP")
        header.setObjectName("section")
        root.addWidget(header)
        self._labels_host = QWidget()
        self._labels_layout = QVBoxLayout(self._labels_host)
        self._labels_layout.setContentsMargins(0, 0, 0, 0)
        self._labels_layout.setSpacing(4)
        lbl_scroll = QScrollArea()
        lbl_scroll.setWidgetResizable(True)
        lbl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        lbl_scroll.setWidget(self._labels_host)
        lbl_scroll.setMinimumHeight(150)
        root.addWidget(lbl_scroll, 2)

        # Commit: persist the clip's collected labels via the review tab.
        self._commit_btn = QPushButton("✓ Commit labels for this clip")
        self._commit_btn.setObjectName("commit")
        self._commit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._commit_btn.setMinimumHeight(38)
        self._commit_btn.clicked.connect(self._commit)
        root.addWidget(self._commit_btn)

        self.setStyleSheet(_WINDOW_QSS)
        self._refresh_labels_list()
        self.resize(640, 640)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure(
        self,
        behaviors: "list[tuple]",
        on_behavior: Callable[[str], None],
        nav: "dict[str, Callable[[], None]]",
        on_structured: "Callable[[str, str, str | None], None] | None" = None,
        on_commit: "Callable[[list], None] | None" = None,
    ) -> None:
        """``behaviors``: list of ``(behavior_id, name, key, is_social, directionality)``.

        ``on_behavior(bid)`` is the single-animal labeling path (used when no
        animals are set). ``on_structured(behavior_id, focal, partner)`` is the
        multi-animal path (partner is ``None`` for solo behaviors).
        ``on_commit(labels)`` persists the clip's collected structured labels,
        where ``labels`` is a list of ``{behavior_id, focal_animal_id,
        partner_animal_id}`` dicts.
        """
        self._behaviors = list(behaviors)
        self._on_behavior = on_behavior
        self._nav = dict(nav or {})
        if on_structured is not None:
            self._on_structured = on_structured
        if on_commit is not None:
            self._on_commit = on_commit
        self._key_to_behavior = {
            str(key).lower(): bid
            for (bid, _n, key, *_rest) in self._behaviors if key
        }
        self._rebuild_behavior_grid()
        self._reset_designation()

    def set_animals(self, animals: "list[tuple]") -> None:
        """``animals``: list of ``(animal_id, display_name, (r,g,b))``.

        Empty/None -> single-animal mode (no selector, legacy behavior).
        """
        self._animals = list(animals or [])
        self._selected_animal = self._animals[0][0] if len(self._animals) == 1 else None
        self._rebuild_animal_bar()
        self._reset_designation()
        self.set_clip_labels([])

    def set_clip_labels(self, labels: "list[dict]") -> None:
        """Replace the displayed per-clip label list (for loading an existing clip)."""
        self._clip_labels = list(labels or [])
        self._refresh_labels_list()

    def load_labels(self, payload: "list[dict]") -> None:
        """Repopulate the chip list from a stored commit payload for editing.

        ``payload`` is a list of ``{behavior_id, focal_animal_id,
        partner_animal_id}`` (as persisted by the review tab). Display strings
        are rebuilt from the current behavior/animal tables; nothing is
        re-emitted (this is a load, not a new label). Call *after* set_animals.
        """
        rebuilt: list[dict] = []
        for lab in payload or []:
            bid = lab.get("behavior_id")
            focal = lab.get("focal_animal_id")
            partner = lab.get("partner_animal_id")
            if not bid or not focal:
                continue
            b = self._behavior(bid)
            social = bool(b and b[3])
            direction = (b[4] if b else "none")
            if social and partner is not None:
                arrow = "→" if direction == "directed" else "⇄"
                display = f"{b[1]}: {self._animal_name(focal)} {arrow} {self._animal_name(partner)}"
            else:
                display = f"{(b[1] if b else bid)}: {self._animal_name(focal)}"
            rebuilt.append({
                "behavior_id": bid, "focal_animal_id": focal,
                "partner_animal_id": partner, "display": display,
            })
        self._clip_labels = rebuilt
        self._refresh_labels_list()

    # ------------------------------------------------------------------
    # Building UI
    # ------------------------------------------------------------------
    def _multi(self) -> bool:
        return len(self._animals) > 0

    def _rebuild_animal_bar(self) -> None:
        while self._animal_bar_layout.count():
            it = self._animal_bar_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._animal_btns = []
        self._animal_bar.setVisible(self._multi())
        if not self._multi():
            return
        self._animal_bar_layout.addWidget(QLabel("Animal:"))
        for animal_id, name, rgb in self._animals:
            r, g, b = rgb
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setMinimumHeight(34)
            btn.setStyleSheet(
                f"QPushButton {{ border:2px solid rgb({r},{g},{b}); border-radius:4px; padding:2px 8px; }}"
                f"QPushButton:checked {{ background-color: rgb({r},{g},{b}); color:#111; font-weight:700; }}"
            )
            btn.clicked.connect(lambda _c=False, a=animal_id: self._on_animal_clicked(a))
            self._animal_bar_layout.addWidget(btn)
            self._animal_btns.append(btn)
        self._animal_bar_layout.addStretch(1)
        self._sync_animal_buttons()

    def _rebuild_behavior_grid(self) -> None:
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not self._behaviors:
            self._grid.addWidget(QLabel("No behaviors defined for this project."), 0, 0)
            return
        n_rows = 0
        for i, (bid, name, key, is_social, direction) in enumerate(self._behaviors):
            tag = ""
            if is_social:
                tag = " →" if direction == "directed" else " ⇄"
            text = f"{name}{tag}" + (f"   ({key})" if key else "")
            btn = QPushButton(text)
            btn.setMinimumSize(132, 42)
            btn.setMaximumHeight(46)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_SOCIAL_BTN_QSS if is_social else _SOLO_BTN_QSS)
            if is_social:
                verb = "actor → recipient" if direction == "directed" else "the two animals"
                btn.setToolTip(f"Social behavior — click it, then designate {verb}.")
            btn.clicked.connect(lambda _c=False, b=bid: self._on_behavior_clicked(b))
            row, col = i // self._columns, i % self._columns
            self._grid.addWidget(btn, row, col)
            n_rows = row
        # Absorb extra space so buttons stay compact at top-left rather than stretching.
        self._grid.setColumnStretch(self._columns, 1)
        self._grid.setRowStretch(n_rows + 1, 1)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _behavior(self, bid: str) -> "tuple | None":
        return next((b for b in self._behaviors if b[0] == bid), None)

    def _animal_name(self, animal_id: str) -> str:
        return next((n for (aid, n, _c) in self._animals if aid == animal_id), animal_id)

    def _on_behavior_clicked(self, bid: str) -> None:
        # Single-animal project -> legacy path.
        if not self._multi():
            self._on_behavior(bid)
            return
        b = self._behavior(bid)
        is_social = bool(b and b[3])
        direction = (b[4] if b else "none")
        if is_social:
            self._pending_social = {"bid": bid, "direction": direction, "picked": []}
            first = "actor" if direction == "directed" else "first animal"
            self._status.setText(f"“{b[1]}”: click the {first}.")
            return
        # Solo behavior -> needs exactly one selected animal.
        if self._selected_animal is None:
            self._status.setText("Select an animal (top), then click the behavior.")
            return
        self._add_label(bid, self._selected_animal, None)

    def _on_animal_clicked(self, animal_id: str) -> None:
        if self._pending_social is not None:
            ps = self._pending_social
            if animal_id in ps["picked"]:
                return  # can't pick the same animal twice
            ps["picked"].append(animal_id)
            b = self._behavior(ps["bid"])
            if len(ps["picked"]) == 1:
                nxt = "recipient" if ps["direction"] == "directed" else "second animal"
                self._status.setText(f"“{b[1]}”: click the {nxt}.")
                self._sync_animal_buttons(highlight=ps["picked"])
                return
            focal, partner = ps["picked"][0], ps["picked"][1]
            self._add_label(ps["bid"], focal, partner)
            self._pending_social = None
            self._status.setText("")
            self._sync_animal_buttons()
            return
        # Normal single-select of the active animal (for solo labels).
        self._selected_animal = animal_id
        self._sync_animal_buttons()

    def _sync_animal_buttons(self, highlight: "list[str] | None" = None) -> None:
        hi = set(highlight or ([] if self._pending_social else ([self._selected_animal] if self._selected_animal else [])))
        for btn, (aid, _n, _c) in zip(self._animal_btns, self._animals):
            btn.setChecked(aid in hi)

    def _add_label(self, bid: str, focal: str, partner: "str | None") -> None:
        b = self._behavior(bid)
        social = bool(b and b[3])
        direction = (b[4] if b else "none")
        if social and partner is not None:
            arrow = "→" if direction == "directed" else "⇄"
            display = f"{b[1]}: {self._animal_name(focal)} {arrow} {self._animal_name(partner)}"
        else:
            display = f"{(b[1] if b else bid)}: {self._animal_name(focal)}"
        self._clip_labels.append({
            "behavior_id": bid, "focal_animal_id": focal,
            "partner_animal_id": partner, "display": display,
        })
        self._refresh_labels_list()
        self._on_structured(bid, focal, partner)

    def _remove_label(self, idx: int) -> None:
        if 0 <= idx < len(self._clip_labels):
            self._clip_labels.pop(idx)
            self._refresh_labels_list()

    def _commit(self) -> None:
        """Persist the clip's collected labels through the review tab."""
        if not self._clip_labels:
            self._status.setText("No labels to commit — pick a behavior first.")
            return
        payload = [
            {
                "behavior_id": lab["behavior_id"],
                "focal_animal_id": lab["focal_animal_id"],
                "partner_animal_id": lab.get("partner_animal_id"),
            }
            for lab in self._clip_labels
        ]
        n = len(payload)
        # Clear the current clip's chips *before* invoking the callback: the
        # commit handler may auto-advance and repopulate this soundboard with the
        # next clip's committed labels, and clearing afterwards would wipe them.
        self.set_clip_labels([])
        self._reset_designation()
        self._on_commit(payload)
        self._status.setText(f"Committed {n} label{'s' if n != 1 else ''}.")

    def _refresh_labels_list(self) -> None:
        while self._labels_layout.count():
            it = self._labels_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not self._clip_labels:
            empty = QLabel("No labels yet — pick a behavior above.")
            empty.setStyleSheet("color:#607D8B; font-style:italic;")
            self._labels_layout.addWidget(empty)
            self._labels_layout.addStretch(1)
            return
        for i, lab in enumerate(self._clip_labels):
            chip = QWidget()
            chip.setObjectName("chip")
            hl = QHBoxLayout(chip)
            hl.setContentsMargins(10, 5, 6, 5)
            hl.setSpacing(8)
            text = QLabel(lab["display"])
            text.setStyleSheet("color:#ECEFF1;")
            hl.addWidget(text, 1)
            rm = QPushButton("✕")
            rm.setObjectName("chipRemove")
            rm.setFixedSize(22, 22)
            rm.setToolTip("Remove this label")
            rm.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            rm.clicked.connect(lambda _c=False, idx=i: self._remove_label(idx))
            hl.addWidget(rm)
            self._labels_layout.addWidget(chip)
        self._labels_layout.addStretch(1)

    def _reset_designation(self) -> None:
        self._pending_social = None
        if self._multi():
            self._hint.setText(
                "Pick a behavior, then the animal(s). Solo: click a behavior for the "
                "selected animal. Social: click the behavior, then the two animals "
                "(actor then recipient for directed). Arrow keys still move clips."
            )
            self._status.setText("")
        else:
            self._hint.setText(
                "Click a behavior to label the current clip. Arrow keys move clips; "
                "Space plays/pauses; Enter saves."
            )
        self._sync_animal_buttons()

    # ------------------------------------------------------------------
    def _toggle_on_top(self, on: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(on))
        self.show()

    def _call(self, action: str) -> None:
        fn = self._nav.get(action)
        if fn is not None:
            fn()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_Escape and self._pending_social is not None:
            self._pending_social = None
            self._status.setText("Cancelled.")
            self._sync_animal_buttons()
            return
        if ctrl and key == Qt.Key.Key_A:
            self._call("accept_all"); return
        if ctrl and key == Qt.Key.Key_R:
            self._call("reject_all"); return
        if key == Qt.Key.Key_Left:
            self._call("frame_back" if shift else "prev"); return
        if key == Qt.Key.Key_Right:
            self._call("frame_fwd" if shift else "next"); return
        if key == Qt.Key.Key_Up:
            self._call("prev"); return
        if key == Qt.Key.Key_Down:
            self._call("next"); return
        if key == Qt.Key.Key_Space:
            self._call("play"); return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._call("save"); return

        text = (event.text() or "").lower().strip()
        if text and text in self._key_to_behavior:
            self._on_behavior_clicked(self._key_to_behavior[text]); return

        super().keyPressEvent(event)

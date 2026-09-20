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

from abel.services.behavior_service import NO_BEHAVIOR_ID, behavior_label


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
_NONE_BTN_QSS = (
    "QPushButton{background:#263238;color:#CFD8DC;border:1px solid #455A64;"
    "border-radius:8px;padding:8px 10px;font-weight:700;}"
    "QPushButton:hover{background:#31424B;border-color:#607D8B;}"
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
    "#clear{background:#3A2429;color:#FFCDD2;border:1px solid #8C4A52;border-radius:8px;padding:7px;font-weight:600;}"
    "#clear:hover{background:#4A2C32;border-color:#B71C1C;}"
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
        self._on_clear: Callable[[], int] = lambda: 0                          # erase saved clip labels
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

        # Behavior button grid: compact buttons, top-left aligned (stretch
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

        # Universal negative, two scopes: the whole clip in one click, or just
        # the selected subject (one animal can be idle while another rears).
        none_row = QWidget()
        none_layout = QHBoxLayout(none_row)
        none_layout.setContentsMargins(0, 0, 0, 0)
        none_layout.setSpacing(8)
        self._none_btn = QPushButton()
        self._none_btn.setObjectName("noneAll")
        self._none_btn.setStyleSheet(_NONE_BTN_QSS)
        self._none_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._none_btn.setMinimumHeight(40)
        self._none_btn.clicked.connect(self._label_all_no_behavior)
        none_layout.addWidget(self._none_btn, 1)
        self._none_one_btn = QPushButton()
        self._none_one_btn.setObjectName("noneOne")
        self._none_one_btn.setStyleSheet(_NONE_BTN_QSS)
        self._none_one_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._none_one_btn.setMinimumHeight(40)
        self._none_one_btn.clicked.connect(self._label_selected_no_behavior)
        none_layout.addWidget(self._none_one_btn, 1)
        root.addWidget(none_row)

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
        self._commit_btn = QPushButton("✓ Commit Labels for This Clip")
        self._commit_btn.setObjectName("commit")
        self._commit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._commit_btn.setMinimumHeight(38)
        self._commit_btn.clicked.connect(self._commit)
        root.addWidget(self._commit_btn)

        # Clear: drop the staged chips *and* whatever was already saved for this
        # clip, so a mislabeled clip can be put back to unreviewed without
        # leaving the soundboard.
        self._clear_btn = QPushButton("✕ Clear Labels for This Clip")
        self._clear_btn.setObjectName("clear")
        self._clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._clear_btn.setMinimumHeight(32)
        self._clear_btn.setToolTip(
            "Remove every label staged or already saved for this clip and put it "
            "back in the queue as unreviewed. The clip file is kept."
        )
        self._clear_btn.clicked.connect(self._clear_clip_labels)
        root.addWidget(self._clear_btn)

        self.setStyleSheet(_WINDOW_QSS)
        self._sync_none_button()
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
        on_clear: "Callable[[], int] | None" = None,
    ) -> None:
        """``behaviors``: list of ``(behavior_id, name, key, is_social, directionality)``.

        ``on_behavior(bid)`` is the single-animal labeling path (used when no
        animals are set). ``on_structured(behavior_id, focal, partner)`` is the
        multi-animal path (partner is ``None`` for solo behaviors).
        ``on_commit(labels)`` persists the clip's collected structured labels,
        where ``labels`` is a list of ``{behavior_id, focal_animal_id,
        partner_animal_id}`` dicts. ``on_clear()`` erases the clip's saved
        labels and returns how many stored rows it removed.
        """
        self._behaviors = list(behaviors)
        self._on_behavior = on_behavior
        self._nav = dict(nav or {})
        if on_structured is not None:
            self._on_structured = on_structured
        if on_commit is not None:
            self._on_commit = on_commit
        if on_clear is not None:
            self._on_clear = on_clear
        self._key_to_behavior = {
            str(key).lower(): bid
            for (bid, _n, key, *_rest) in self._behaviors if key
        }
        self._rebuild_behavior_grid()
        self._sync_none_button()
        self._reset_designation()

    def set_animals(self, animals: "list[tuple]") -> None:
        """``animals``: list of ``(animal_id, display_name, (r,g,b))``.

        Empty/None -> single-animal mode (no selector, legacy behavior).
        """
        self._animals = list(animals or [])
        self._selected_animal = self._animals[0][0] if len(self._animals) == 1 else None
        self._rebuild_animal_bar()
        self._sync_none_button()
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
                display = f"{(b[1] if b else behavior_label(bid))}: {self._animal_name(focal)}"
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
        """Lay the behavior buttons out in two labeled blocks: solo, then social.

        ``no_behavior`` is left out of the grid, it gets its own all-subjects
        button below, since the universal negative applies to the whole clip
        rather than to one designated animal.
        """
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        entries = [b for b in self._behaviors if b[0] != NO_BEHAVIOR_ID]
        if not entries:
            self._grid.addWidget(QLabel("No behaviors defined for this project."), 0, 0)
            return
        solo = [b for b in entries if not b[3]]
        social = [b for b in entries if b[3]]
        # Headers only earn their space when there is actually a split to show
        # (single-animal projects have no social behaviors at all).
        show_headers = bool(solo and social)
        row = 0
        for title, group in (("SOLO", solo), ("SOCIAL", social)):
            if not group:
                continue
            if show_headers:
                header = QLabel(f"{title}  ({len(group)})")
                header.setObjectName("section")
                self._grid.addWidget(header, row, 0, 1, self._columns)
                row += 1
            for i, entry in enumerate(group):
                self._grid.addWidget(
                    self._make_behavior_button(entry), row + i // self._columns, i % self._columns
                )
            row += (len(group) + self._columns - 1) // self._columns
        # Absorb extra space so buttons stay compact at top-left rather than stretching.
        self._grid.setColumnStretch(self._columns, 1)
        self._grid.setRowStretch(row, 1)

    def _make_behavior_button(self, entry: tuple) -> QPushButton:
        bid, name, key, is_social, direction = entry
        tag = ""
        if is_social:
            tag = " →" if direction == "directed" else " ⇄"
        btn = QPushButton(f"{name}{tag}" + (f"   ({key})" if key else ""))
        btn.setMinimumSize(132, 42)
        btn.setMaximumHeight(46)
        btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setStyleSheet(_SOCIAL_BTN_QSS if is_social else _SOLO_BTN_QSS)
        if is_social:
            verb = "actor → recipient" if direction == "directed" else "the two animals"
            btn.setToolTip(f"Social behavior, click it, then designate {verb}.")
        btn.clicked.connect(lambda _c=False, b=bid: self._on_behavior_clicked(b))
        return btn

    def _no_behavior_name(self) -> str:
        b = self._behavior(NO_BEHAVIOR_ID)
        return str(b[1]) if b else behavior_label(NO_BEHAVIOR_ID)

    def _sync_none_button(self) -> None:
        name = self._no_behavior_name()
        key = next((k for (bid, _n, k, *_r) in self._behaviors if bid == NO_BEHAVIOR_ID and k), "")
        suffix = f"   ({key})" if key else ""
        self._none_one_btn.setVisible(self._multi())
        if self._multi():
            self._none_btn.setText(f"⌀ {name}, all subjects{suffix}")
            self._none_btn.setToolTip(
                "Label every animal in this clip as the universal negative and commit "
                "it. Replaces any labels staged for this clip."
            )
            who = self._animal_name(self._selected_animal) if self._selected_animal else "selected subject"
            self._none_one_btn.setText(f"⌀ {name}, {who}")
            self._none_one_btn.setEnabled(self._selected_animal is not None)
            self._none_one_btn.setToolTip(
                "Stage the universal negative for the selected animal only, leaving "
                "the other animals' labels alone (one can be idle while another is "
                "behaving). Commit when the clip is done."
            )
        else:
            self._none_btn.setText(f"⌀ {name}{suffix}")
            self._none_btn.setToolTip("Label this clip as the universal negative.")

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _behavior(self, bid: str) -> "tuple | None":
        return next((b for b in self._behaviors if b[0] == bid), None)

    def _animal_name(self, animal_id: str) -> str:
        return next((n for (aid, n, _c) in self._animals if aid == animal_id), animal_id)

    def _on_behavior_clicked(self, bid: str) -> None:
        # The universal negative is clip-wide, never per-animal (this also
        # routes its keyboard shortcut to the all-subjects button).
        if bid == NO_BEHAVIOR_ID:
            self._label_all_no_behavior()
            return
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
        self._sync_none_button()

    def _add_label(self, bid: str, focal: str, partner: "str | None") -> None:
        b = self._behavior(bid)
        social = bool(b and b[3])
        direction = (b[4] if b else "none")
        if social and partner is not None:
            arrow = "→" if direction == "directed" else "⇄"
            display = f"{b[1]}: {self._animal_name(focal)} {arrow} {self._animal_name(partner)}"
        else:
            display = f"{(b[1] if b else behavior_label(bid))}: {self._animal_name(focal)}"
        self._clip_labels.append({
            "behavior_id": bid, "focal_animal_id": focal,
            "partner_animal_id": partner, "display": display,
        })
        self._refresh_labels_list()
        self._on_structured(bid, focal, partner)

    def _label_all_no_behavior(self) -> None:
        """Label every subject in the clip as ``no_behavior`` and commit it.

        The universal negative contradicts any positive label, so this replaces
        whatever was staged for the clip rather than adding to it. In a
        single-animal project there is nothing to designate, so it falls back to
        the review tab's normal no-behavior path.
        """
        if not self._multi():
            self._on_behavior(NO_BEHAVIOR_ID)
            return
        self._pending_social = None
        self._sync_animal_buttons()
        name = self._no_behavior_name()
        self._clip_labels = [
            {
                "behavior_id": NO_BEHAVIOR_ID, "focal_animal_id": aid,
                "partner_animal_id": None, "display": f"{name}: {animal_name}",
            }
            for (aid, animal_name, _rgb) in self._animals
        ]
        self._refresh_labels_list()
        for lab in self._clip_labels:
            self._on_structured(NO_BEHAVIOR_ID, lab["focal_animal_id"], None)
        self._commit()

    def _label_selected_no_behavior(self) -> None:
        """Stage the universal negative for the selected animal only.

        The clip-wide button is wrong when only *some* subjects are idle, so
        this one behaves like any other solo behavior: it applies to the
        designated animal and waits for the commit. Because the negative
        contradicts that animal's positives, its existing chips are dropped,
        the other animals' labels are untouched.
        """
        if not self._multi():
            self._on_behavior(NO_BEHAVIOR_ID)
            return
        if self._selected_animal is None:
            self._status.setText("Select an animal (top), then click the ⌀ button.")
            return
        self._pending_social = None
        focal = self._selected_animal
        self._clip_labels = [
            lab for lab in self._clip_labels
            if lab.get("focal_animal_id") != focal and lab.get("partner_animal_id") != focal
        ]
        self._sync_animal_buttons()
        self._add_label(NO_BEHAVIOR_ID, focal, None)

    def _clear_clip_labels(self) -> None:
        """Discard staged chips and ask the review tab to erase saved labels."""
        self._pending_social = None
        staged = len(self._clip_labels)
        self.set_clip_labels([])
        self._reset_designation()
        try:
            removed = int(self._on_clear() or 0)
        except Exception:  # a clearing failure must not wedge the window
            removed = 0
        if removed:
            self._status.setText(f"Cleared {removed} saved label{'s' if removed != 1 else ''}.")
        elif staged:
            self._status.setText(f"Discarded {staged} staged label{'s' if staged != 1 else ''}.")
        else:
            self._status.setText("Nothing to clear on this clip.")

    def _remove_label(self, idx: int) -> None:
        if 0 <= idx < len(self._clip_labels):
            self._clip_labels.pop(idx)
            self._refresh_labels_list()

    def _commit(self) -> None:
        """Persist the clip's collected labels through the review tab."""
        if not self._clip_labels:
            self._status.setText("No labels to commit: pick a behavior first.")
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
            empty = QLabel("No labels yet: pick a behavior above.")
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
                "(actor then recipient for directed). Empty clip: the ⌀ all-subjects "
                "button labels everyone at once; the ⌀ per-subject button marks just "
                "the selected animal idle. Arrow keys still move clips."
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
            self._status.setText("Canceled.")
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

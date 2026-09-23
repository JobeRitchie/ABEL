"""Soundboard keyboard: arrows keep working after a commit, Enter commits."""

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from abel.ui.behavior_soundboard import BehaviorSoundboard


BEHAVIORS = [("rearing", "Rearing", "r", False, "none")]
ANIMALS = [("sub:track_0", "black", (200, 0, 0)), ("sub:track_1", "green", (0, 200, 0))]


def _board(calls, committed, animals=ANIMALS) -> BehaviorSoundboard:
    app = QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    nav = {k: (lambda k=k: calls.append(k)) for k in ("next", "prev", "save")}
    sb.configure(BEHAVIORS, lambda _b: None, nav, on_commit=committed.append)
    sb.set_animals(animals)
    sb.show()
    sb.activateWindow()
    app.processEvents()
    return sb


def _key(sb, key):
    QTest.keyClick(QApplication.focusWidget() or sb, key)
    QApplication.processEvents()


def test_no_child_can_steal_keyboard_focus():
    sb = _board([], [])
    focusable = [
        w for w in sb.findChildren(QWidget)
        if w.focusPolicy() != Qt.FocusPolicy.NoFocus
    ]
    assert focusable == []
    sb.close()


def test_arrows_move_clips_after_commit():
    calls, committed = [], []
    sb = _board(calls, committed)
    sb._on_animal_clicked("sub:track_0")
    sb._on_behavior_clicked("rearing")
    QTest.mouseClick(sb._commit_btn, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert len(committed) == 1
    _key(sb, Qt.Key.Key_Down)
    _key(sb, Qt.Key.Key_Up)
    assert calls == ["next", "prev"]
    sb.close()


def test_enter_commits_in_multi_animal_mode():
    calls, committed = [], []
    sb = _board(calls, committed)
    sb._on_animal_clicked("sub:track_0")
    sb._on_behavior_clicked("rearing")
    _key(sb, Qt.Key.Key_Return)
    assert len(committed) == 1
    assert "save" not in calls
    sb.close()


def test_enter_saves_in_single_animal_mode():
    calls, committed = [], []
    sb = _board(calls, committed, animals=[])
    _key(sb, Qt.Key.Key_Return)
    assert calls == ["save"]
    assert committed == []
    sb.close()


SOCIAL = [
    ("chase", "Chase", "c", True, "directed"),
    ("flee", "Flee", "f", True, "directed"),
]


def _social_board(committed, animals=ANIMALS):
    app = QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    sb.configure(SOCIAL, lambda _b: None, {}, on_commit=committed.append)
    sb.set_animals(animals)
    sb.show()
    app.processEvents()
    return sb


def _stage_chase_flee(sb):
    sb._on_behavior_clicked("chase")
    sb._on_animal_clicked("sub:track_1")
    sb._on_animal_clicked("sub:track_0")
    sb._on_behavior_clicked("flee")
    sb._on_animal_clicked("sub:track_0")
    sb._on_animal_clicked("sub:track_1")


def test_repeat_stages_last_combo_on_next_clip():
    committed = []
    sb = _social_board(committed)
    assert not sb._repeat_btn.isEnabled()
    _stage_chase_flee(sb)
    sb._commit()
    # Next clip from another session: different ids, same track names.
    sb.set_animals([("other:track_0", "track_0", (1, 1, 1)), ("other:track_1", "track_1", (2, 2, 2))])
    assert sb._repeat_btn.isEnabled()
    _key(sb, Qt.Key.Key_Period)
    _key(sb, Qt.Key.Key_Period)  # a second press does not duplicate
    _key(sb, Qt.Key.Key_Return)
    assert len(committed) == 2
    got = {(p["behavior_id"], p["focal_animal_id"], p["partner_animal_id"]) for p in committed[1]}
    assert got == {("chase", "other:track_1", "other:track_0"), ("flee", "other:track_0", "other:track_1")}
    assert len(committed[1]) == 2
    sb.close()


def test_repeat_falls_back_to_roster_position_and_ignores_no_behavior_commits():
    committed = []
    sb = _social_board(committed)
    _stage_chase_flee(sb)
    sb._commit()
    sb._label_all_no_behavior()  # must not overwrite the remembered combo
    sb.set_animals([("m:a", "Mouse A", (1, 1, 1)), ("m:b", "Mouse B", (2, 2, 2))])
    sb._apply_last_combo()
    got = {(lab["behavior_id"], lab["focal_animal_id"], lab["partner_animal_id"]) for lab in sb._clip_labels}
    assert got == {("chase", "m:b", "m:a"), ("flee", "m:a", "m:b")}
    sb.close()

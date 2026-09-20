"""Soundboard: one-click universal negative for every subject, and grouped layout."""

from PySide6.QtWidgets import QApplication

from abel.services.behavior_service import NO_BEHAVIOR_ID
from abel.ui.behavior_soundboard import BehaviorSoundboard


BEHAVIORS = [
    ("rearing", "Rearing", "r", False, "none"),
    ("grooming", "Grooming", "g", False, "none"),
    ("sniffing", "Sniffing", "s", True, "mutual"),
    ("fighting", "Fighting", "f", True, "directed"),
    (NO_BEHAVIOR_ID, "No Behavior", "n", False, "none"),
]
ANIMALS = [("sub:track_0", "black", (200, 0, 0)), ("sub:track_1", "green", (0, 200, 0))]


def _board(**kw) -> BehaviorSoundboard:
    QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    sb.configure(BEHAVIORS, kw.pop("on_behavior", lambda _b: None), {}, **kw)
    return sb


def _grid_texts(sb):
    return [
        sb._grid.itemAt(i).widget().text()
        for i in range(sb._grid.count())
        if sb._grid.itemAt(i).widget() is not None
    ]


def test_no_behavior_labels_every_subject_and_commits():
    committed = []
    sb = _board(on_commit=committed.append)
    sb.set_animals(ANIMALS)
    sb._label_all_no_behavior()
    assert len(committed) == 1
    assert {lab["focal_animal_id"] for lab in committed[0]} == {"sub:track_0", "sub:track_1"}
    assert {lab["behavior_id"] for lab in committed[0]} == {NO_BEHAVIOR_ID}
    assert all(lab["partner_animal_id"] is None for lab in committed[0])


def test_no_behavior_replaces_staged_positive_labels():
    committed = []
    sb = _board(on_commit=committed.append)
    sb.set_animals(ANIMALS)
    sb._on_animal_clicked("sub:track_0")
    sb._on_behavior_clicked("rearing")
    assert len(sb._clip_labels) == 1
    sb._on_behavior_clicked(NO_BEHAVIOR_ID)  # also the path the 'n' shortcut takes
    assert [lab["behavior_id"] for lab in committed[0]] == [NO_BEHAVIOR_ID] * 2


def test_single_animal_no_behavior_uses_legacy_path():
    seen = []
    sb = _board(on_behavior=seen.append, on_commit=lambda _p: seen.append("commit"))
    sb.set_animals([])
    sb._on_behavior_clicked(NO_BEHAVIOR_ID)
    assert seen == [NO_BEHAVIOR_ID]  # no structured commit in single-animal mode


def test_grid_groups_solo_and_social_and_omits_no_behavior():
    sb = _board()
    sb.set_animals(ANIMALS)
    texts = _grid_texts(sb)
    assert not any("No Behavior" in t for t in texts)  # lives on its own button
    solo_hdr, social_hdr = texts.index("SOLO  (2)"), texts.index("SOCIAL  (2)")
    assert solo_hdr < texts.index("Rearing   (r)") < social_hdr
    assert social_hdr < texts.index("Sniffing \u21c4   (s)")


def test_solo_only_project_gets_no_group_headers():
    QApplication.instance() or QApplication([])
    sb = BehaviorSoundboard()
    sb.configure(
        [("rearing", "Rearing", "r", False, "none"), (NO_BEHAVIOR_ID, "No Behavior", "n", False, "none")],
        lambda _b: None, {},
    )
    assert _grid_texts(sb) == ["Rearing   (r)"]


def test_none_button_text_tracks_mode():
    sb = _board()
    sb.set_animals(ANIMALS)
    assert "all subjects" in sb._none_btn.text()
    sb.set_animals([])
    assert "all subjects" not in sb._none_btn.text()
    assert "No Behavior" in sb._none_btn.text()


def test_per_subject_no_behavior_stages_only_that_animal():
    committed = []
    sb = _board(on_commit=committed.append)
    sb.set_animals(ANIMALS)
    sb._on_animal_clicked("sub:track_1")
    sb._on_behavior_clicked("rearing")            # green is rearing
    sb._on_animal_clicked("sub:track_0")
    sb._label_selected_no_behavior()              # black is doing nothing
    assert not committed                          # per-subject stages, never auto-commits
    assert [(l["behavior_id"], l["focal_animal_id"]) for l in sb._clip_labels] == [
        ("rearing", "sub:track_1"), (NO_BEHAVIOR_ID, "sub:track_0"),
    ]


def test_per_subject_no_behavior_replaces_that_animals_labels_only():
    sb = _board()
    sb.set_animals(ANIMALS)
    sb._on_animal_clicked("sub:track_1")
    sb._on_behavior_clicked("rearing")
    sb._on_behavior_clicked("sniffing")           # social: green -> black
    sb._on_animal_clicked("sub:track_1")
    sb._on_animal_clicked("sub:track_0")
    assert len(sb._clip_labels) == 2
    sb._on_animal_clicked("sub:track_0")
    sb._label_selected_no_behavior()
    # the social label involved black too, so it goes; green's solo rearing stays
    assert [(l["behavior_id"], l["focal_animal_id"]) for l in sb._clip_labels] == [
        ("rearing", "sub:track_1"), (NO_BEHAVIOR_ID, "sub:track_0"),
    ]


def test_per_subject_button_tracks_selection_and_mode():
    sb = _board()
    sb.set_animals(ANIMALS)
    assert sb._none_one_btn.isVisibleTo(sb) and not sb._none_one_btn.isEnabled()
    sb._on_animal_clicked("sub:track_0")
    assert sb._none_one_btn.isEnabled() and "black" in sb._none_one_btn.text()
    sb.set_animals([])
    assert not sb._none_one_btn.isVisibleTo(sb)

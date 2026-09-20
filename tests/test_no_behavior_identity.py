"""The ``No Behavior`` label is a reserved identity, not an editable behavior.

A user renamed the built-in "No Behavior" to "Freezing" and added a new behavior
called "No Behavior".  Because ``no_behavior`` is the universal negative class,
training collapses alternate labels onto it, temporal refinement skips it in the
competition, exports drop it, the renamed behavior was silently read as
"nothing happened", which is how it surfaced in temporal review as a behavior
that was really freezing.

These tests pin both halves of the fix: the guard that makes the identity
unforgeable, and the repair that untangles a project that already did it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

from abel.models.schemas import BehaviorDefinition
from abel.services.behavior_service import (
    NO_BEHAVIOR_ID,
    BehaviorService,
    ReservedBehaviorError,
)
from abel.services.no_behavior_repair import NoBehaviorRepair, detect_no_behavior_conflict
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml


# ---------------------------------------------------------------------------
# Guard: the reserved identity cannot be claimed or given away
# ---------------------------------------------------------------------------


def _service(tmp_path: Path) -> BehaviorService:
    svc = BehaviorService()
    svc.set_project(tmp_path)
    return svc


def test_builtin_no_behavior_cannot_be_renamed(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    builtin = svc.get(NO_BEHAVIOR_ID)
    assert builtin is not None

    with pytest.raises(ReservedBehaviorError):
        svc.update(NO_BEHAVIOR_ID, builtin.model_copy(update={"name": "Freezing"}))

    assert svc.get(NO_BEHAVIOR_ID).name == "No Behavior"


def test_builtin_no_behavior_stays_cosmetically_editable(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    builtin = svc.get(NO_BEHAVIOR_ID)

    assert svc.update(
        NO_BEHAVIOR_ID,
        builtin.model_copy(update={"color": "#123456", "keyboard_shortcut": "0"}),
    )
    reloaded = _service(tmp_path).get(NO_BEHAVIOR_ID)
    assert reloaded.color == "#123456"
    assert reloaded.keyboard_shortcut == "0"
    assert reloaded.name == "No Behavior"


@pytest.mark.parametrize("name", ["No Behavior", "no_behavior", "no behavior", "NoBehavior"])
def test_second_no_behavior_cannot_be_added(tmp_path: Path, name: str) -> None:
    svc = _service(tmp_path)
    with pytest.raises(ReservedBehaviorError):
        svc.add(BehaviorDefinition(behavior_id=str(uuid.uuid4()), name=name, short_name="nb"))
    assert [b.behavior_id for b in svc.behaviors].count(NO_BEHAVIOR_ID) == 1


def test_existing_behaviour_cannot_be_renamed_into_the_negative(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    rear = svc.add(BehaviorDefinition(behavior_id=str(uuid.uuid4()), name="Rear", short_name="rear"))
    with pytest.raises(ReservedBehaviorError):
        svc.update(rear.behavior_id, rear.model_copy(update={"name": "No Behavior"}))
    assert svc.get(rear.behavior_id).name == "Rear"


def test_import_skips_a_reserved_definition(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    src = tmp_path / "defs.yaml"
    write_yaml(src, {"behaviors": [
        {"behavior_id": NO_BEHAVIOR_ID, "name": "No Behavior", "short_name": "none"},
        {"behavior_id": str(uuid.uuid4()), "name": "Rear", "short_name": "rear"},
    ]})
    assert svc.import_definitions(src) == 1
    assert [b.behavior_id for b in svc.behaviors].count(NO_BEHAVIOR_ID) == 1
    assert sum(1 for b in svc.behaviors if b.name == "No Behavior") == 1


# ---------------------------------------------------------------------------
# Repair: untangle a project that already made the swap
# ---------------------------------------------------------------------------

FREEZING_LABELS = ["seg_a_s1_0_59", "seg_a_s1_60_119"]
SYSTEM_NEGATIVE = "seg_feedback_s1_300_359"


def _broken_project(tmp_path: Path) -> tuple[Path, str]:
    """A project where ``no_behavior`` is bound to "Freezing" and a new
    "No Behavior" behavior holds the real negatives."""
    root = tmp_path / "proj"
    negative_id = str(uuid.uuid4())
    rear_id = str(uuid.uuid4())

    write_yaml(root / "config" / "behavior_definitions.yaml", {"behaviors": [
        {"behavior_id": NO_BEHAVIOR_ID, "name": "Freezing", "short_name": "freeze"},
        {"behavior_id": negative_id, "name": "No Behavior", "short_name": "none"},
        {"behavior_id": rear_id, "name": "Rear", "short_name": "rear"},
    ]})

    labels = pd.DataFrame([
        {"segment_id": FREEZING_LABELS[0], "review_label": NO_BEHAVIOR_ID,
         "reviewer_id": "jane", "confidence": 1.0, "notes": ""},
        {"segment_id": FREEZING_LABELS[1], "review_label": f"{NO_BEHAVIOR_ID}|{rear_id}",
         "reviewer_id": "jane", "confidence": 1.0, "notes": ""},
        {"segment_id": "seg_a_s1_120_179", "review_label": negative_id,
         "reviewer_id": "jane", "confidence": 1.0, "notes": ""},
        # Written by the app itself as a hard negative for the Rear model.
        {"segment_id": SYSTEM_NEGATIVE, "review_label": NO_BEHAVIOR_ID,
         "reviewer_id": "temporal_feedback", "confidence": 1.0,
         "notes": f"temporal:no_behavior:{rear_id}"},
    ])
    (root / "derived" / "review_labels").mkdir(parents=True, exist_ok=True)
    labels.to_parquet(root / "derived" / "review_labels" / "reviewer_labels.parquet", index=False)

    write_json(root / "derived" / "review_tables" / "review_decisions.json", {"decisions": [
        {"decision_id": "d1", "clip_id": FREEZING_LABELS[0], "reviewer": "jane",
         "old_status": "", "new_status": "", "decision": "accept",
         "behavior_label": NO_BEHAVIOR_ID},
        {"decision_id": "d2", "clip_id": SYSTEM_NEGATIVE, "reviewer": "temporal_feedback",
         "old_status": "", "new_status": "", "decision": "reject",
         "behavior_label": rear_id},
    ]})
    write_json(root / "derived" / "review_tables" / "candidate_windows.json", {"candidates": [
        {"window_id": "w1", "behavior_id": NO_BEHAVIOR_ID},
        {"window_id": "w2", "behavior_id": negative_id},
    ]})
    write_json(root / "config" / "seeds.json", {"seeds": [
        {"seed_id": "s1", "behavior_id": NO_BEHAVIOR_ID},
    ]})
    write_json(root / "config" / "temporal_review_settings.json", {
        "__all__": {"onset_threshold": 0.5},
        NO_BEHAVIOR_ID: {"onset_threshold": 0.8},
    })
    write_json(root / "config" / "temporal_refinement_settings.json", {
        "__all__": {"suppression_matrix": {NO_BEHAVIOR_ID: {rear_id: 0.2}}},
    })
    write_json(root / "derived" / "review_labels" / "soundboard_labels.json", {"windows": {
        FREEZING_LABELS[0]: [{"behavior_id": NO_BEHAVIOR_ID, "focal_animal_id": "a"}],
    }})

    model_dir = root / "derived" / "models" / "behavior_model_Freezing"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / "run_settings.json", {"target_behavior": NO_BEHAVIOR_ID})
    (model_dir / "model_state.pkl").write_bytes(b"x")
    keep_dir = root / "derived" / "models" / "behavior_model_Rear"
    keep_dir.mkdir(parents=True, exist_ok=True)
    write_json(keep_dir / "run_settings.json", {"target_behavior": rear_id})

    tr_dir = root / "derived" / "temporal_refinement" / NO_BEHAVIOR_ID
    tr_dir.mkdir(parents=True, exist_ok=True)
    write_json(tr_dir / "latest.json", {"behavior": NO_BEHAVIOR_ID})

    return root, negative_id


def test_detects_the_repurposed_label(tmp_path: Path) -> None:
    root, negative_id = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)

    conflict = svc.no_behavior_conflict()
    assert conflict is not None
    assert conflict.repurposed_name == "Freezing"
    assert conflict.replacement_id == negative_id


def test_healthy_project_reports_no_conflict(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.add(BehaviorDefinition(behavior_id=str(uuid.uuid4()), name="Rear", short_name="rear"))
    assert svc.no_behavior_conflict() is None
    assert detect_no_behavior_conflict(svc.behaviors) is None


def test_plan_changes_nothing_on_disk(tmp_path: Path) -> None:
    root, _ = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)
    before = read_yaml(root / "config" / "behavior_definitions.yaml", {})

    report = NoBehaviorRepair(root, svc.behaviors).plan()

    assert report.dry_run
    assert report.counts["labels"] >= 3
    assert read_yaml(root / "config" / "behavior_definitions.yaml", {}) == before
    assert (root / "derived" / "models" / "behavior_model_Freezing").exists()


def test_repair_swaps_the_identities_and_carries_the_work(tmp_path: Path) -> None:
    root, negative_id = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)

    report = NoBehaviorRepair(root, svc.behaviors).apply()
    new_id = report.new_behavior_id

    # Definitions: Freezing owns a fresh id, the user's negative owns the reserved one.
    svc.set_project(root)
    assert svc.no_behavior_conflict() is None
    assert svc.get(new_id).name == "Freezing"
    assert svc.get(NO_BEHAVIOR_ID).name == "No Behavior"
    assert svc.get(negative_id) is None

    # Reviewer labels follow the behavior, including inside a pipe-joined label…
    labels = pd.read_parquet(root / "derived" / "review_labels" / "reviewer_labels.parquet")
    by_seg = dict(zip(labels["segment_id"], labels["review_label"]))
    assert by_seg[FREEZING_LABELS[0]] == new_id
    assert by_seg[FREEZING_LABELS[1]].split("|")[0] == new_id
    assert by_seg["seg_a_s1_120_179"] == NO_BEHAVIOR_ID
    # …but a hard negative the app wrote itself stays a negative.
    assert by_seg[SYSTEM_NEGATIVE] == NO_BEHAVIOR_ID

    decisions = read_json(root / "derived" / "review_tables" / "review_decisions.json", {})
    by_clip = {d["clip_id"]: d["behavior_label"] for d in decisions["decisions"]}
    assert by_clip[FREEZING_LABELS[0]] == new_id

    candidates = read_json(root / "derived" / "review_tables" / "candidate_windows.json", {})
    ids = {c["window_id"]: c["behavior_id"] for c in candidates["candidates"]}
    assert ids["w1"] == new_id
    assert ids["w2"] == NO_BEHAVIOR_ID

    seeds = read_json(root / "config" / "seeds.json", {})
    assert seeds["seeds"][0]["behavior_id"] == new_id

    review_settings = read_json(root / "config" / "temporal_review_settings.json", {})
    assert new_id in review_settings and NO_BEHAVIOR_ID not in review_settings

    refinement = read_json(root / "config" / "temporal_refinement_settings.json", {})
    assert new_id in refinement["__all__"]["suppression_matrix"]

    soundboard = read_json(root / "derived" / "review_labels" / "soundboard_labels.json", {})
    assert soundboard["windows"][FREEZING_LABELS[0]][0]["behavior_id"] == new_id


def test_repair_retires_contaminated_models_into_a_backup(tmp_path: Path) -> None:
    root, _ = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)

    report = NoBehaviorRepair(root, svc.behaviors).apply()

    assert "behavior_model_Freezing" in report.retired_models
    assert not (root / "derived" / "models" / "behavior_model_Freezing").exists()
    # An unrelated behavior's model is untouched.
    assert (root / "derived" / "models" / "behavior_model_Rear").exists()
    # The stale per-behavior refinement artifacts are retired too.
    assert not (root / "derived" / "temporal_refinement" / NO_BEHAVIOR_ID).exists()

    backup = Path(report.backup_dir)
    assert (backup / "config" / "behavior_definitions.yaml").exists()
    assert (backup / "derived" / "review_labels" / "reviewer_labels.parquet").exists()
    assert (backup / "derived" / "models" / "behavior_model_Freezing" / "model_state.pkl").exists()


def test_repair_creates_a_negative_label_when_the_user_made_none(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    write_yaml(root / "config" / "behavior_definitions.yaml", {"behaviors": [
        {"behavior_id": NO_BEHAVIOR_ID, "name": "Freezing", "short_name": "freeze"},
    ]})
    svc = BehaviorService()
    svc.set_project(root)

    report = NoBehaviorRepair(root, svc.behaviors).apply()
    assert report.promoted_id is None

    svc.set_project(root)
    assert svc.get(NO_BEHAVIOR_ID).name == "No Behavior"
    assert svc.get(report.new_behavior_id).name == "Freezing"


def test_repair_is_idempotent(tmp_path: Path) -> None:
    root, _ = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)
    NoBehaviorRepair(root, svc.behaviors).apply()

    svc.set_project(root)
    with pytest.raises(ValueError):
        NoBehaviorRepair(root, svc.behaviors).apply()


def test_duplicate_no_behavior_merges_into_the_builtin(tmp_path: Path) -> None:
    """The other half of the mistake: the built-in kept its name, but a second
    behavior named "No Behavior" was added anyway."""
    root = tmp_path / "proj"
    dup_id = str(uuid.uuid4())
    write_yaml(root / "config" / "behavior_definitions.yaml", {"behaviors": [
        {"behavior_id": NO_BEHAVIOR_ID, "name": "No Behavior", "short_name": "none"},
        {"behavior_id": dup_id, "name": "No Behavior", "short_name": "nb2"},
    ]})
    write_json(root / "config" / "seeds.json", {"seeds": [{"seed_id": "s1", "behavior_id": dup_id}]})

    svc = BehaviorService()
    svc.set_project(root)
    conflict = svc.no_behavior_conflict()
    assert conflict is not None
    assert conflict.repurposed_name is None
    assert conflict.replacement_id == dup_id

    report = NoBehaviorRepair(root, svc.behaviors).apply()
    assert report.merged_duplicate
    assert report.new_behavior_id is None

    svc.set_project(root)
    assert svc.no_behavior_conflict() is None
    assert len(svc.behaviors) == 1
    assert svc.get(NO_BEHAVIOR_ID).name == "No Behavior"
    assert read_json(root / "config" / "seeds.json", {})["seeds"][0]["behavior_id"] == NO_BEHAVIOR_ID


# ---------------------------------------------------------------------------
# Behavior tab: the reserved row is visibly locked, the conflict is surfaced
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.behavior_tab import BehaviorTab  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _select_behavior(tab: BehaviorTab, behavior_id: str) -> None:
    from PySide6.QtCore import Qt

    for row in range(tab._table.rowCount()):
        item = tab._table.item(row, 2)
        if item and item.data(Qt.ItemDataRole.UserRole) == behavior_id:
            tab._table.selectRow(row)
            return
    raise AssertionError(f"{behavior_id} not in the table")


def test_tab_locks_the_reserved_row(_app, tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "config").mkdir(parents=True)
    svc = BehaviorService()
    tab = BehaviorTab(svc)
    tab.set_project(root)
    rear = svc.add(BehaviorDefinition(behavior_id=str(uuid.uuid4()), name="Rear", short_name="rear"))
    tab.refresh()

    _select_behavior(tab, rear.behavior_id)
    assert tab._f_name.isEnabled() and tab._delete_btn.isEnabled()
    assert not tab._reserved_hint.isVisibleTo(tab)

    _select_behavior(tab, NO_BEHAVIOR_ID)
    assert not tab._f_name.isEnabled()
    assert not tab._f_short.isEnabled()
    assert not tab._delete_btn.isEnabled()
    # Cosmetic edits stay available.
    assert tab._f_color_btn.isEnabled()
    assert tab._f_description.isEnabled()
    assert tab._reserved_hint.isVisibleTo(tab)
    assert not tab._conflict_bar.isVisibleTo(tab)


def test_tab_surfaces_the_conflict_banner(_app, tmp_path: Path) -> None:
    root, _ = _broken_project(tmp_path)
    tab = BehaviorTab(BehaviorService())
    tab.set_project(root)

    assert tab._conflict_bar.isVisibleTo(tab)
    assert "Freezing" in tab._conflict_label.text()


def test_resaving_an_unchanged_definition_never_raises(tmp_path: Path) -> None:
    """Dialogs that re-save every definition (e.g. the analytics color picker)
    must keep working in a project that already has the conflict."""
    root, negative_id = _broken_project(tmp_path)
    svc = BehaviorService()
    svc.set_project(root)

    for b in svc.behaviors:
        assert svc.update(b.behavior_id, b.model_copy(update={"color": "#010203"}))

    svc.set_project(root)
    assert svc.get(NO_BEHAVIOR_ID).name == "Freezing"
    assert svc.get(negative_id).color == "#010203"
    # The conflict is still reported: the repair, not the guard, is what fixes it.
    assert svc.no_behavior_conflict() is not None

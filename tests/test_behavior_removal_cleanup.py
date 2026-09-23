"""Deleting a behavior must purge its orphaned trained model + eval output.

Downstream tools (unified UMAP, analytics, apply-models) discover behaviors by
scanning ``derived/models`` for ``behavior_model_*`` directories. Before this
fix, ``BehaviorService.delete`` only edited the definitions YAML, leaving the
trained model on disk, so a removed behavior (e.g. "Wall Rear") kept showing
up in the active-learning UMAP. These tests pin the cleanup behavior.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from abel.models.schemas import BehaviorDefinition
from abel.services.behavior_service import BehaviorService
from abel.storage.file_store import read_json, write_json


def _add(svc: BehaviorService, name: str) -> str:
    b = svc.add(BehaviorDefinition(behavior_id=str(uuid.uuid4()), name=name, short_name=name))
    return b.behavior_id


def _make_model(root: Path, name: str, target_behavior: str) -> Path:
    md = root / "derived" / "models" / f"behavior_model_{name}"
    md.mkdir(parents=True, exist_ok=True)
    write_json(md / "run_settings.json", {"target_behavior": target_behavior})
    (md / "model_state.pkl").write_bytes(b"x")
    (md / "segment_predictions.parquet").write_bytes(b"x")
    eval_dir = root / "derived" / "evaluation" / "by_model" / f"behavior_model_{name}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text("{}")
    return md


def test_delete_purges_orphaned_model_and_eval(tmp_path: Path) -> None:
    svc = BehaviorService()
    svc.set_project(tmp_path)
    keep_id = _add(svc, "Rear")
    drop_id = _add(svc, "Wall Rear")

    keep_md = _make_model(tmp_path, "Rear", keep_id)
    drop_md = _make_model(tmp_path, "Wall_Rear", drop_id)

    assert svc.delete(drop_id) is True

    # Orphaned model + its per-model evaluation output are gone.
    assert not drop_md.exists()
    assert not (tmp_path / "derived" / "evaluation" / "by_model" / "behavior_model_Wall_Rear").exists()
    # The surviving behavior's model is untouched.
    assert keep_md.exists()
    assert (tmp_path / "derived" / "evaluation" / "by_model" / "behavior_model_Rear").exists()


def test_delete_matches_by_target_id_not_folder_name(tmp_path: Path) -> None:
    """Model folders may be custom-named; match on run_settings target id."""
    svc = BehaviorService()
    svc.set_project(tmp_path)
    drop_id = _add(svc, "Wall Rear")

    # Folder name bears no resemblance to the behavior name.
    md = _make_model(tmp_path, "custom_experiment_42", drop_id)
    assert md.exists()

    assert svc.delete(drop_id) is True
    assert not md.exists()


def test_purge_is_noop_without_matching_model(tmp_path: Path) -> None:
    svc = BehaviorService()
    svc.set_project(tmp_path)
    keep_id = _add(svc, "Rear")
    other_id = _add(svc, "Groom")

    keep_md = _make_model(tmp_path, "Rear", keep_id)

    assert svc.delete(other_id) is True  # deleting Groom (no model) is fine
    assert keep_md.exists()  # Rear's model untouched


def test_delete_purges_candidate_decision_and_label_references(tmp_path: Path) -> None:
    """A deleted behavior must vanish from the review queue, decisions, and
    reviewer labels so it no longer appears in the clip-review dropdown or in
    training data."""
    svc = BehaviorService()
    svc.set_project(tmp_path)
    keep_id = _add(svc, "Head Dip")
    drop_id = _add(svc, "Stretch Attend")

    rt = tmp_path / "derived" / "review_tables"
    rt.mkdir(parents=True, exist_ok=True)

    write_json(
        rt / "external_window_candidates.json",
        {"candidates": [
            {"window_id": "w1", "session_id": "s1", "behavior_id": keep_id},
            {"window_id": "w2", "session_id": "s1", "behavior_id": drop_id},
            {"window_id": "w3", "session_id": "s1", "behavior_id": f"{keep_id}|{drop_id}"},
        ]},
    )
    write_json(
        rt / "review_decisions.json",
        {"decisions": [
            {"clip_id": "c1", "behavior_label": keep_id},
            {"clip_id": "c2", "behavior_label": drop_id},
        ]},
    )

    lbl_path = tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet"
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"segment_id": ["seg1", "seg2", "seg3"],
         "review_label": [keep_id, drop_id, f"{drop_id}|{keep_id}"]}
    ).to_parquet(lbl_path, index=False)

    assert svc.delete(drop_id) is True

    # Candidate queue: pure-drop row removed, multi-label row stripped to keep_id.
    cands = read_json(rt / "external_window_candidates.json", {})["candidates"]
    ids = {c["window_id"]: c["behavior_id"] for c in cands}
    assert ids == {"w1": keep_id, "w3": keep_id}

    # Decisions: the drop-only decision is gone.
    decs = read_json(rt / "review_decisions.json", {})["decisions"]
    assert [d["clip_id"] for d in decs] == ["c1"]

    # Reviewer labels: drop-only row removed, multi-label stripped.
    df = pd.read_parquet(lbl_path)
    got = dict(zip(df["segment_id"], df["review_label"]))
    assert got == {"seg1": keep_id, "seg3": keep_id}


def test_delete_purges_soundboard_training_set_settings_and_bouts(tmp_path: Path) -> None:
    svc = BehaviorService()
    svc.set_project(tmp_path)
    keep_id = _add(svc, "Attack")
    drop_id = _add(svc, "Submit")

    sb = tmp_path / "derived" / "review_labels" / "soundboard_labels.json"
    write_json(sb, {"windows": {
        "w1": [{"behavior_id": drop_id, "focal_animal_id": "track_0"}],
        "w2": [
            {"behavior_id": drop_id, "focal_animal_id": "track_0"},
            {"behavior_id": keep_id, "focal_animal_id": "track_1"},
        ],
    }})
    ts = tmp_path / "derived" / "training_sets" / "training_set.parquet"
    ts.parent.mkdir(parents=True)
    pd.DataFrame({
        "segment_id": ["a", "b", "c"],
        "label": [drop_id, f"{keep_id}|{drop_id}", keep_id],
    }).to_parquet(ts, index=False)
    trs = tmp_path / "config" / "temporal_review_settings.json"
    write_json(trs, {"by_behavior": {drop_id: {"onset_threshold": 0.2}, keep_id: {"onset_threshold": 0.3}}})
    ref = tmp_path / "config" / "temporal_refinement_settings.json"
    write_json(ref, {"by_behavior": {"target_behavior": {"selected_behavior_models": {
        drop_id: "behavior_model_Submit", keep_id: "behavior_model_Attack"}}}})
    cands = tmp_path / "derived" / "review_tables" / "candidate_segments.json"
    write_json(cands, {"candidates": [{"behavior_id": keep_id, "behavior_ids": [keep_id, drop_id]}]})
    bouts = tmp_path / "derived" / "behavior_bouts" / f"{drop_id}_bouts.parquet"
    bouts.parent.mkdir(parents=True)
    bouts.write_bytes(b"x")

    assert svc.delete(drop_id) is True

    windows = read_json(sb)["windows"]
    assert "w1" not in windows
    assert windows["w2"] == [{"behavior_id": keep_id, "focal_animal_id": "track_1"}]
    df = pd.read_parquet(ts)
    assert df["segment_id"].tolist() == ["b", "c"]
    assert df["label"].tolist() == [keep_id, keep_id]
    assert list(read_json(trs)["by_behavior"]) == [keep_id]
    assert read_json(ref)["by_behavior"]["target_behavior"]["selected_behavior_models"] == {
        keep_id: "behavior_model_Attack"}
    assert read_json(cands)["candidates"][0]["behavior_ids"] == [keep_id]
    assert not bouts.exists()

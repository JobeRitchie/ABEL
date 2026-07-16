"""Tests for cross-project model import (apply another project's models here)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from abel.services.model_refinement_service import (
    AUTO_CREATE_BEHAVIOR,
    SKIP_BEHAVIOR,
    ModelRefinementService,
)


FEATS = [f"feat_{i}" for i in range(8)]


def _write_behaviors(root: Path, behaviors: list[dict]) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "behavior_definitions.yaml").write_text(
        yaml.dump({"behaviors": behaviors}), encoding="utf-8",
    )


def _make_host(tmp_path: Path, feat_cols: list[str] = FEATS) -> Path:
    """Host project with a training set (defines its feature schema)."""
    root = tmp_path / "host"
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "host-freeze", "name": "Freeze"},
    ])
    ts_dir = root / "derived" / "training_sets"
    ts_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"segment_id": ["h1", "h2"], "label": ["host-freeze", "no_behavior"],
                       "label_source": "reviewer", "session_id": ["s", "s"]})
    for c in feat_cols:
        df[c] = [0.1, 0.2]
    df.to_parquet(ts_dir / "training_set.parquet", index=False)
    return root


def _make_model_dir(root: Path, behavior_id: str, behavior_name: str,
                    feat_cols: list[str] = FEATS) -> str:
    """A real (tiny) trained model dir in the expected on-disk shape."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(feat_cols)))
    y = (x[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(x, y)

    dir_name = f"behavior_model_{behavior_name}"
    md = root / "derived" / "models" / dir_name
    md.mkdir(parents=True, exist_ok=True)
    with open(md / "model_state.pkl", "wb") as f:
        pickle.dump({"model": clf, "feature_cols": list(feat_cols)}, f)
    (md / "run_settings.json").write_text(
        json.dumps({"model_version": dir_name, "target_behavior": behavior_id}),
        encoding="utf-8",
    )
    (md / "model_card.yaml").write_text(yaml.safe_dump({
        "model_version": dir_name,
        "labels": [behavior_id, "no_behavior"],
        "feature_columns": list(feat_cols),
    }), encoding="utf-8")
    # Stale source-scored prediction that must be dropped on import.
    pd.DataFrame({"segment_id": ["src1"], "prediction_prob": [0.9]}).to_parquet(
        md / "segment_predictions.parquet", index=False,
    )
    return dir_name


def _make_source(tmp_path: Path, name: str = "src") -> Path:
    root = tmp_path / name
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-freeze", "name": "Freeze"},     # matches host Freeze
        {"behavior_id": "src-explore", "name": "Explore"},   # no host match
    ])
    _make_model_dir(root, "src-freeze", "Freeze")
    _make_model_dir(root, "src-explore", "Explore")
    return root


def test_list_source_models_excludes_no_behavior(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    # add a no_behavior model that must be excluded
    _make_model_dir(src, "no_behavior", "No_Behavior")
    models = ModelRefinementService().list_source_models(src)
    names = {m.behavior_name for m in models}
    assert names == {"Freeze", "Explore"}
    freeze = next(m for m in models if m.behavior_name == "Freeze")
    assert freeze.feature_count == len(FEATS)


def test_preview_reports_coverage_and_behavior_match(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    src = _make_source(tmp_path)
    pv = ModelRefinementService().preview_model_import(host, src)
    assert pv.host_feature_count == len(FEATS)
    by_name = {i.model.behavior_name: i for i in pv.items}
    # Freeze: full coverage, matches host Freeze by name.
    assert by_name["Freeze"].compatible
    assert by_name["Freeze"].coverage == 1.0
    assert by_name["Freeze"].host_behavior_id == "host-freeze"
    # Explore: compatible features, but no host behaviour → unmatched.
    assert by_name["Explore"].compatible
    assert not by_name["Explore"].behavior_matched
    assert any(i.model.behavior_name == "Explore" for i in pv.unmatched_behaviors)


def test_preview_blocks_incompatible_features(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = tmp_path / "bad"
    _write_behaviors(src, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "b-freeze", "name": "Freeze"},
    ])
    # Model trained on features the host doesn't have.
    _make_model_dir(src, "b-freeze", "Freeze", feat_cols=[f"other_{i}" for i in range(8)])
    pv = ModelRefinementService().preview_model_import(host, src)
    item = pv.items[0]
    assert not item.compatible
    assert item.coverage == 0.0


def test_import_maps_to_existing_behavior(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    src = _make_source(tmp_path, name="CAB")
    svc = ModelRefinementService()
    res = svc.import_models(
        host, src, model_dirs=["behavior_model_Freeze"],
        behavior_decisions={"src-freeze": "host-freeze"},
    )
    assert res["status"] == "success", res
    assert len(res["imported"]) == 1
    new_dir = res["imported"][0]["model_dir"]
    # Namespaced so it can't clobber the host's own model dir.
    assert new_dir == "behavior_model_Freeze__CAB"
    md = host / "derived" / "models" / new_dir
    assert (md / "model_state.pkl").exists()
    # target_behavior rewritten to the host id; stale predictions dropped.
    rs = json.loads((md / "run_settings.json").read_text())
    assert rs["target_behavior"] == "host-freeze"
    assert rs["model_version"] == new_dir
    assert not (md / "segment_predictions.parquet").exists()
    card = yaml.safe_load((md / "model_card.yaml").read_text())
    assert "host-freeze" in card["labels"] and "src-freeze" not in card["labels"]
    # Manifest records it and removal cleans it up.
    imports = svc.list_model_imports(host)
    assert imports and imports[0]["tag"] == "CAB"


def test_import_auto_creates_missing_behavior(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    src = _make_source(tmp_path, name="CAB")
    svc = ModelRefinementService()
    res = svc.import_models(
        host, src, model_dirs=["behavior_model_Explore"],
        behavior_decisions={"src-explore": AUTO_CREATE_BEHAVIOR},
    )
    assert res["status"] == "success", res
    # The source's Explore behaviour definition is now in the host project.
    host_behaviors = yaml.safe_load(
        (host / "config" / "behavior_definitions.yaml").read_text()
    )["behaviors"]
    ids = {b["behavior_id"] for b in host_behaviors}
    assert "src-explore" in ids
    # The imported model targets that (preserved) behaviour id.
    rs = json.loads(
        (host / "derived" / "models" / res["imported"][0]["model_dir"] / "run_settings.json").read_text()
    )
    assert rs["target_behavior"] == "src-explore"


def test_import_skip_decision_is_respected(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    src = _make_source(tmp_path, name="CAB")
    svc = ModelRefinementService()
    res = svc.import_models(
        host, src, model_dirs=["behavior_model_Explore"],
        behavior_decisions={"src-explore": SKIP_BEHAVIOR},
    )
    assert res["status"] == "error"  # nothing imported
    assert not res["imported"]
    assert any(s["reason"] == "behaviour skipped" for s in res["skipped"])


def test_remove_model_import_deletes_dirs(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    src = _make_source(tmp_path, name="CAB")
    svc = ModelRefinementService()
    svc.import_models(host, src, model_dirs=["behavior_model_Freeze"],
                      behavior_decisions={"src-freeze": "host-freeze"})
    new_dir = host / "derived" / "models" / "behavior_model_Freeze__CAB"
    assert new_dir.exists()
    out = svc.remove_model_import(host, "CAB")
    assert out["removed_models"] == 1
    assert not new_dir.exists()
    assert svc.list_model_imports(host) == []


# ---------------------------------------------------------------------------
# Legacy pairwise-distance ordering
#
# Distance is symmetric, so dist_A_to_B and dist_B_to_A are the same
# measurement. Older extractor builds ordered the pair by keypoint position in
# the pose file; the current one canonicalises the ordering. A model trained
# before that change asks for the opposite spelling to the one a new project
# emits — which is a naming difference, not missing data, and must not block
# the import.
# ---------------------------------------------------------------------------

_PAIR_FEATS = [
    "dist_nose_to_left_ear_mean",       # model spelling (legacy ordering)
    "dist_nose_to_left_ear_norm_std",
]
_HOST_PAIR_FEATS = [
    "dist_left_ear_to_nose_mean",       # host spelling (canonical ordering)
    "dist_left_ear_to_nose_norm_std",
]


def test_swapped_distance_pair_is_not_counted_as_missing(tmp_path: Path) -> None:
    """A model naming a distance the other way round still imports."""
    host = _make_host(tmp_path, feat_cols=FEATS + _HOST_PAIR_FEATS)
    src = tmp_path / "src"
    _write_behaviors(src, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-freeze", "name": "Freeze"},
    ])
    _make_model_dir(src, "src-freeze", "Freeze", feat_cols=FEATS + _PAIR_FEATS)

    pv = ModelRefinementService().preview_model_import(host, src)
    item = next(i for i in pv.items if i.model.behavior_name == "Freeze")

    # The host has the same distances, just spelled the other way round.
    assert item.missing_features == 0
    assert item.coverage == 1.0
    assert item.compatible
    assert sorted(item.legacy_pair_columns) == sorted(_PAIR_FEATS)


def test_import_keeps_feature_cols_intact_and_records_pair_order(tmp_path: Path) -> None:
    """Import never drops the swapped columns; it records why they were allowed."""
    host = _make_host(tmp_path, feat_cols=FEATS + _HOST_PAIR_FEATS)
    src = tmp_path / "src"
    _write_behaviors(src, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-freeze", "name": "Freeze"},
    ])
    _make_model_dir(src, "src-freeze", "Freeze", feat_cols=FEATS + _PAIR_FEATS)

    svc = ModelRefinementService()
    out = svc.import_models(
        host, src, ["behavior_model_Freeze"],
        behavior_decisions={"src-freeze": "host-freeze"},
    )
    assert out["status"] == "success"

    md = host / "derived" / "models" / out["imported"][0]["model_dir"]
    with open(md / "model_state.pkl", "rb") as f:
        payload = pickle.load(f)

    # The classifier's input width is fixed, so the columns survive verbatim —
    # scoring re-derives the mapping, it does not need them rewritten.
    assert set(_PAIR_FEATS).issubset(payload["feature_cols"])

    card = yaml.safe_load((md / "model_card.yaml").read_text(encoding="utf-8"))
    assert sorted(card["legacy_pair_order_columns"]) == sorted(_PAIR_FEATS)


def test_genuinely_absent_features_still_block_import(tmp_path: Path) -> None:
    """The pair-order allowance must not mask a real feature gap."""
    host = _make_host(tmp_path, feat_cols=FEATS)
    src = tmp_path / "src"
    _write_behaviors(src, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-freeze", "name": "Freeze"},
    ])
    # Host has neither ordering of these, and lacks a whole video-feature family.
    absent = [f"flow_mag_paw_L_{s}" for s in ("mean", "std", "max", "energy")]
    _make_model_dir(src, "src-freeze", "Freeze", feat_cols=FEATS + _PAIR_FEATS + absent)

    pv = ModelRefinementService().preview_model_import(host, src)
    item = next(i for i in pv.items if i.model.behavior_name == "Freeze")

    assert not item.compatible
    assert set(absent).issubset(item.missing_columns)
    # The swapped pair has no host counterpart either, so it is a real gap too.
    assert set(_PAIR_FEATS).issubset(item.missing_columns)
    assert item.legacy_pair_columns == []

"""Tests for baseline import — seed a feature-extracted-but-untrained project
with another project's clips + labeled feature rows + trained models.

Covers the key scenario the feature targets: a host that has extracted features
(``segment_features.parquet``) but has NOT run active learning (no
``training_set.parquet``, no behaviours, no models), and the partial-overlap edge
case where the host already has some — but not all — of the source's behaviours.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from abel.services.model_refinement_service import (
    AUTO_CREATE_BEHAVIOR,
    SKIP_BEHAVIOR,
    ModelRefinementService,
)

FEATS = [f"feat_{i}" for i in range(12)]


def _write_behaviors(root: Path, behaviors: list[dict]) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "behavior_definitions.yaml").write_text(
        yaml.dump({"behaviors": behaviors}), encoding="utf-8",
    )


def _feature_frame(seg_ids: list[str], sessions: list[str]) -> pd.DataFrame:
    data: dict = {
        "segment_id": seg_ids,
        "session_id": sessions,
        "animal_id": ["a"] * len(seg_ids),
        "start_frame": list(range(len(seg_ids))),
        "end_frame": [i + 14 for i in range(len(seg_ids))],
    }
    for i, c in enumerate(FEATS):
        data[c] = [float(i + j) for j in range(len(seg_ids))]
    return pd.DataFrame(data)


def _make_model_dir(root: Path, behavior_id: str, behavior_name: str) -> str:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(FEATS)))
    y = (x[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(x, y)

    dir_name = f"behavior_model_{behavior_name}"
    md = root / "derived" / "models" / dir_name
    md.mkdir(parents=True, exist_ok=True)
    with open(md / "model_state.pkl", "wb") as f:
        pickle.dump({"model": clf, "feature_cols": list(FEATS)}, f)
    (md / "run_settings.json").write_text(
        json.dumps({"model_version": dir_name, "target_behavior": behavior_id}),
        encoding="utf-8",
    )
    (md / "model_card.yaml").write_text(yaml.safe_dump({
        "model_version": dir_name,
        "labels": [behavior_id, "no_behavior"],
        "feature_columns": list(FEATS),
    }), encoding="utf-8")
    return dir_name


def _make_source(tmp_path: Path, name: str = "DONOR") -> Path:
    """Source with segment_features + reviewer labels + two trained models."""
    root = tmp_path / name
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-walk", "name": "Walk"},
        {"behavior_id": "src-rear", "name": "Rear"},
    ])
    rep = root / "derived" / "representations"
    rep.mkdir(parents=True, exist_ok=True)
    seg_ids = ["s1", "s2", "s3", "s4", "s5"]
    _feature_frame(seg_ids, ["ss1"] * 5).to_parquet(
        rep / "segment_features.parquet", index=False,
    )
    lab_dir = root / "derived" / "review_labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "segment_id": seg_ids,
        "review_label": ["src-walk", "src-walk", "src-rear", "src-rear", "no_behavior"],
        "confidence": [1.0] * 5,
        "timestamp": ["2026-01-01T00:00:0" + str(i) for i in range(5)],
    }).to_parquet(lab_dir / "reviewer_labels.parquet", index=False)
    _make_model_dir(root, "src-walk", "Walk")
    _make_model_dir(root, "src-rear", "Rear")
    return root


def _make_untrained_host(tmp_path: Path, behaviors: list[dict] | None = None) -> Path:
    """Host with extracted features only — no training set, no models."""
    root = tmp_path / "host"
    _write_behaviors(root, behaviors or [{"behavior_id": "no_behavior", "name": "No Behavior"}])
    rep = root / "derived" / "representations"
    rep.mkdir(parents=True, exist_ok=True)
    _feature_frame(["h1", "h2", "h3"], ["hs1"] * 3).to_parquet(
        rep / "segment_features.parquet", index=False,
    )
    return root


def test_preview_baseline_detects_new_project(tmp_path: Path) -> None:
    host = _make_untrained_host(tmp_path)
    src = _make_source(tmp_path)
    pv = ModelRefinementService().preview_baseline(host, src)

    assert pv.host_is_new is True
    assert pv.schema_ok is True
    assert pv.host_feature_count == len(FEATS)
    by_name = {r.source_name: r for r in pv.rows}
    assert set(by_name) == {"Walk", "Rear"}  # no_behavior excluded
    assert by_name["Walk"].example_count == 2 and by_name["Rear"].example_count == 2
    assert by_name["Walk"].has_model and by_name["Walk"].model_compatible
    # Nothing matches an existing host behaviour → all "new".
    assert all(r.status == "new" for r in pv.rows)
    assert pv.model_count == 2


def test_import_baseline_seeds_training_set_and_models(tmp_path: Path) -> None:
    host = _make_untrained_host(tmp_path)
    src = _make_source(tmp_path)
    svc = ModelRefinementService()
    res = svc.import_baseline(host, src, behavior_decisions={
        "src-walk": AUTO_CREATE_BEHAVIOR,
        "src-rear": AUTO_CREATE_BEHAVIOR,
    })
    assert res["status"] == "success", res

    # Training set created from extracted features + imported labeled rows.
    ts_path = host / "derived" / "training_sets" / "training_set.parquet"
    assert ts_path.exists()
    ts = pd.read_parquet(ts_path)
    assert "label" in ts.columns
    labels = set(ts["label"].astype(str))
    assert {"src-walk", "src-rear", "no_behavior"} <= labels
    # Behaviours auto-created in the host.
    host_ids = {
        b["behavior_id"]
        for b in yaml.safe_load((host / "config" / "behavior_definitions.yaml").read_text())["behaviors"]
    }
    assert {"src-walk", "src-rear"} <= host_ids
    # Both models copied (namespaced) and listed in the manifest.
    assert len(res["imported_models"]) == 2
    assert svc.list_model_imports(host)
    assert (host / "derived" / "models" / "behavior_model_Walk__DONOR").exists()


def test_import_baseline_partial_overlap_and_skip(tmp_path: Path) -> None:
    # Host already defines "Walk" (host-walk) but not "Rear".
    host = _make_untrained_host(tmp_path, behaviors=[
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "host-walk", "name": "Walk"},
    ])
    src = _make_source(tmp_path)
    svc = ModelRefinementService()

    pv = svc.preview_baseline(host, src)
    assert pv.host_is_new is False  # has a real behaviour already
    by_name = {r.source_name: r for r in pv.rows}
    assert by_name["Walk"].status == "matched"
    assert by_name["Walk"].matched_host_id == "host-walk"
    assert by_name["Rear"].status == "new"

    # Map Walk onto the existing host behaviour; skip Rear entirely.
    res = svc.import_baseline(host, src, behavior_decisions={
        "src-walk": "host-walk",
        "src-rear": SKIP_BEHAVIOR,
    })
    assert res["status"] == "success", res

    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    labels = set(ts["label"].astype(str))
    assert "host-walk" in labels      # Walk examples mapped onto existing behaviour
    assert "src-rear" not in labels   # Rear skipped — no examples
    # Only Walk's model imported; Rear skipped.
    imported_targets = {m["behavior_id"] for m in res["imported_models"]}
    assert imported_targets == {"host-walk"}
    assert not (host / "derived" / "models" / "behavior_model_Rear__DONOR").exists()

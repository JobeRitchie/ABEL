"""Enriched reviewed windows in multi-animal sessions belong to one animal.

The frame table holds one row per (frame, animal). The enrichment path used to
summarize a reviewed off-grid window over *both* animals' frames and stamp the
first animal's id on it, so ``seg_track_0_…`` and ``seg_track_1_…`` got the same
blended feature row. In a dyad the actor's label (Chase) and the partner's
(Flee) then sat on identical features. The scoring step also wrote those rows
back into segment_features.parquet, where they shadowed any later fix, and the
label remap could snap a label onto the other animal's window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.services.import_service import ImportService
from abel.storage.file_store import read_json
from abel.ui.tabs.active_learning_tab import ActiveLearningTab

SID = "session_dyad0000"


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    n = 200
    frames = pd.concat([
        pd.DataFrame({
            "frame": np.arange(n), "session_id": SID, "animal_id": "track_0",
            "speed": np.linspace(0.0, 4.0, n),
        }),
        pd.DataFrame({
            "frame": np.arange(n), "session_id": SID, "animal_id": "track_1",
            "speed": np.sin(np.arange(n) / 5.0) * 3.0,
        }),
    ], ignore_index=True)
    pose_dir = tmp_path / "derived" / "pose_features" / "sessions"
    pose_dir.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(pose_dir / f"{SID}.parquet", index=False)

    labels = tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet"
    labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"segment_id": f"seg_track_0_{SID}_101_114", "review_label": "chase"},
        {"segment_id": f"seg_track_1_{SID}_101_114", "review_label": "flee"},
    ]).to_parquet(labels, index=False)
    return tmp_path


def _tab(project: Path) -> ActiveLearningTab:
    tab = ActiveLearningTab.__new__(ActiveLearningTab)
    tab._project_root = project
    tab._imports = ImportService()
    return tab


def _grid() -> pd.DataFrame:
    rows = []
    for animal in ("track_0", "track_1"):
        for start in (96, 102):
            rows.append({
                "segment_id": f"seg_{animal}_{SID}_{start}_{start + 13}",
                "session_id": SID, "animal_id": animal,
                "start_frame": start, "end_frame": start + 13,
                "speed_mean": 0.0, "speed_std": 0.0,
            })
    return pd.DataFrame(rows)


def _enriched(out: pd.DataFrame) -> pd.DataFrame:
    return out[out["start_frame"] == 101].set_index("segment_id")


def test_each_track_is_summarized_from_its_own_frames(project) -> None:
    out = _enriched(_tab(project)._enrich_segment_df_for_reviewed_labels(_grid()))

    t0 = out.loc[f"seg_track_0_{SID}_101_114"]
    t1 = out.loc[f"seg_track_1_{SID}_101_114"]
    assert t0["animal_id"] == "track_0"
    assert t1["animal_id"] == "track_1"
    assert not np.isclose(t0["speed_mean"], t1["speed_mean"])

    meta = read_json(project / "derived" / "representations" / "enriched_segments_meta.json", {})
    assert meta.get("version", 0) >= 2


def test_outdated_blended_cache_is_recomputed(project) -> None:
    rep = project / "derived" / "representations"
    rep.mkdir(parents=True, exist_ok=True)
    # A version-1 cache: both tracks carry the same blended row, track_0's id.
    blended = pd.DataFrame([
        {"segment_id": f"seg_track_{i}_{SID}_101_114", "session_id": SID,
         "animal_id": "track_0", "start_frame": 101, "end_frame": 114,
         "speed_mean": 9.0, "speed_std": 9.0}
        for i in (0, 1)
    ])
    blended.to_parquet(rep / "enriched_segments.parquet", index=False)

    out = _enriched(_tab(project)._enrich_segment_df_for_reviewed_labels(_grid()))

    assert out.loc[f"seg_track_1_{SID}_101_114", "animal_id"] == "track_1"
    assert not (out["speed_mean"] == 9.0).any()


def test_rows_leaked_into_the_feature_table_are_replaced(project) -> None:
    """A blended copy persisted into segment_features must not shadow the cache."""
    tab = _tab(project)
    tab._enrich_segment_df_for_reviewed_labels(_grid())  # writes a correct v2 cache
    leaked = pd.DataFrame([
        {"segment_id": f"seg_track_{i}_{SID}_101_114", "session_id": SID,
         "animal_id": "track_0", "start_frame": 101, "end_frame": 114,
         "speed_mean": 9.0, "speed_std": 9.0}
        for i in (0, 1)
    ])
    grid_with_leak = pd.concat([_grid(), leaked], ignore_index=True)

    out = _enriched(tab._enrich_segment_df_for_reviewed_labels(grid_with_leak))

    assert len(out) == 2
    assert not (out["speed_mean"] == 9.0).any()
    assert out.loc[f"seg_track_1_{SID}_101_114", "animal_id"] == "track_1"
    assert tab._enriched_segment_ids() == set(out.index)


def test_remap_keeps_each_label_on_its_own_animal(project) -> None:
    tab = _tab(project)
    labels = pd.DataFrame([
        {"segment_id": f"seg_track_1_{SID}_101_114", "review_label": "flee"},
    ])

    out = tab._remap_review_labels_to_current_windows(labels, _grid())

    assert len(out) == 1
    assert out["segment_id"].iloc[0].startswith("seg_track_1_")


def test_segment_id_animal() -> None:
    f = ActiveLearningTab._segment_id_animal
    assert f(f"seg_track_1_{SID}_1_14") == "track_1"
    assert f(f"rand_track_0_{SID}_1_14") == "track_0"
    assert f(f"{SID}_1_14") is None

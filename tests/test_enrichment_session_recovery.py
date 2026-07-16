"""On-the-fly enrichment must not recompute the same segments on every retrain.

The retrain pipeline enriches reviewed segments once per behaviour (for training)
and again for inference. Segments that yield no feature row left no trace in the
enrichment cache, so they were re-attempted on every pass — and a session with no
per-session pose file re-read the whole frame-pose store each time, only to find
it held nothing. Labels recorded under a since-replaced session id hit exactly
that path: they were both the slowest and the ones silently lost from training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset
from abel.services.behavior_representation_service import BehaviorRepresentationService
from abel.services.import_service import ImportService
from abel.storage.file_store import read_json
from abel.ui.tabs.active_learning_tab import ActiveLearningTab

CURRENT = "session_new00000"
STALE = "session_old00000"
GHOST = "session_ghost00"


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A project whose recording was re-imported: labels exist under both ids."""
    service = ImportService()
    video = VideoAsset(asset_id="vid_1", source_path=str(tmp_path / "raw" / "videos" / "CBMRE04.mp4"))
    pose = PoseAsset(asset_id="pose_1", source_path=str(tmp_path / "raw" / "pose" / "CBMRE04.csv"), format="csv")
    manifest = ImportManifest(
        videos=[video],
        poses=[pose],
        linked_sessions=[LinkedSession(session_id=STALE, video_asset_id="vid_1", pose_asset_id="pose_1")],
    )
    service.save_manifest(tmp_path, manifest)  # registry records the old id …
    manifest.linked_sessions[0].session_id = CURRENT
    service.save_manifest(tmp_path, manifest)  # … and now the current one

    frames = pd.DataFrame(
        {
            "frame": np.arange(200),
            "session_id": CURRENT,
            "animal_id": "ind0",
            "speed": np.linspace(0.0, 4.0, 200),
        }
    )
    pose_dir = tmp_path / "derived" / "pose_features" / "sessions"
    pose_dir.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(pose_dir / f"{CURRENT}.parquet", index=False)

    labels_path = tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"segment_id": f"bout_abcd_{STALE}_100_129", "review_label": "Rear"},
            {"segment_id": f"{GHOST}_100_129", "review_label": "Rear"},
        ]
    ).to_parquet(labels_path, index=False)
    return tmp_path


def _tab(project: Path) -> ActiveLearningTab:
    tab = ActiveLearningTab.__new__(ActiveLearningTab)
    tab._project_root = project
    tab._imports = ImportService()
    return tab


def _segment_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": f"{CURRENT}_0_29",
                "session_id": CURRENT,
                "animal_id": "ind0",
                "start_frame": 0,
                "end_frame": 29,
                "speed_mean": 0.1,
                "speed_std": 0.01,
            }
        ]
    )


def test_stale_session_labels_are_recovered_and_never_recomputed(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original = BehaviorRepresentationService._segment_summary

    def counting_summary(window_df, feature_cols, segment_id):
        calls.append(str(segment_id))
        return original(window_df, feature_cols, segment_id)

    monkeypatch.setattr(
        BehaviorRepresentationService, "_segment_summary", staticmethod(counting_summary)
    )

    tab = _tab(project)
    enriched = tab._enrich_segment_df_for_reviewed_labels(_segment_df())

    stale_id = f"bout_abcd_{STALE}_100_129"
    assert stale_id in set(enriched["segment_id"]), (
        "a label recorded under the recording's previous session id must still be "
        "featurised from the session that now owns those frames"
    )
    assert calls == [stale_id]
    assert float(enriched.loc[enriched["segment_id"] == stale_id, "speed_mean"].iloc[0]) != 0.0

    # The ghost session has no pose data anywhere and never will from this cache.
    skipped = read_json(project / "derived" / "representations" / "enriched_segments_skipped.json", {})
    assert skipped["segment_ids"] == [f"{GHOST}_100_129"]

    # Second pass (the inference half of the same retrain, and every retrain after):
    # everything is either cached or known-unfeaturisable, so nothing is recomputed.
    again = tab._enrich_segment_df_for_reviewed_labels(_segment_df())

    assert calls == [stale_id], "enrichment recomputed segments it had already resolved"
    assert stale_id in set(again["segment_id"])


def test_sessions_without_pose_data_do_not_reread_the_frame_pose_store(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ghost sessions are dropped from the work list, not probed one full read at a time."""
    monolith = project / "derived" / "pose_features" / "frame_pose.parquet"
    frames = pd.read_parquet(project / "derived" / "pose_features" / "sessions" / f"{CURRENT}.parquet")
    frames.to_parquet(monolith, index=False)

    # Three more reviewed sessions that the project holds no pose data for at all.
    labels_path = project / "derived" / "review_labels" / "reviewer_labels.parquet"
    labels = pd.read_parquet(labels_path)
    extra = pd.DataFrame(
        [{"segment_id": f"session_ghost0{i}_100_129", "review_label": "Rear"} for i in range(3)]
    )
    pd.concat([labels, extra], ignore_index=True).to_parquet(labels_path, index=False)

    full_reads: list[Path] = []
    real_read = pd.read_parquet

    def spy(path, *args, **kwargs):
        if kwargs.get("columns") is None:  # a column-projected read is cheap; a full read is not
            full_reads.append(Path(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)

    tab = _tab(project)
    tab._enrich_segment_df_for_reviewed_labels(_segment_df())

    assert [p for p in full_reads if p == monolith] == [], (
        "sessions with no pose data must be skipped up front, not discovered by "
        "reading the whole frame-pose store once per session"
    )

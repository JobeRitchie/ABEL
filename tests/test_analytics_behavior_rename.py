"""Renaming a behavior must reach the analytics graphs.

Two independent caches held the old name.  The disk analytics cache stores the
behavior *name* on every summary row and bout row, but its fingerprint tracks
only the data files -- a rename touches ``behavior_definitions.yaml`` alone, so
the stale rows were replayed verbatim.  Separately, merged external projects
were matched to host behaviors by name only, so a host rename orphaned them.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab
from abel.services.project_merge_service import ProjectMergeService, ProjectMergeEntry


BID = "11111111-1111-1111-1111-111111111111"
OLD_NAME = "Rear"
NEW_NAME = "Vertical Exploration"


def _write_cache(project_root: Path) -> None:
    cache_dir = project_root / "derived" / "analytics_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": 2,
        "fingerprint": "fp",
        "summary_rows": [
            {"session_id": "s1", "subject": "M01", "behavior_id": BID,
             "behavior": OLD_NAME, "n_bouts": 3.0, "time_spent_s": 7.0},
            # Pseudo-behavior rows carry ids the behavior list never contains.
            {"session_id": "s1", "subject": "M01", "behavior_id": "__distance__",
             "behavior": "Distance Traveled", "n_bouts": 0.0, "distance_cm": 12.0},
        ],
    }
    (cache_dir / "analytics_cache.json").write_text(json.dumps(meta), encoding="utf-8")
    pd.DataFrame([
        {"session_id": "s1", "start_frame": 0, "end_frame": 10,
         "behavior_id": BID, "behavior": OLD_NAME},
    ]).to_parquet(cache_dir / f"bouts_{BehaviorAnalyticsTab._safe_name(BID)}.parquet",
                  index=False)


def test_cached_rows_are_relabelled_after_a_rename(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    host = SimpleNamespace(
        _analytics_cache_dir=lambda root: root / "derived" / "analytics_cache",
        _safe_name=BehaviorAnalyticsTab._safe_name,
    )
    behavior_list = [SimpleNamespace(behavior_id=BID, name=NEW_NAME)]

    result = BehaviorAnalyticsTab._try_load_analytics_cache(
        host, tmp_path, "fp", behavior_list,
    )

    assert result is not None, "matching fingerprint must still hit the cache"
    assert result["summary_rows"][0]["behavior"] == NEW_NAME
    # Pseudo-behaviors are not in the definition list and keep their label.
    assert result["summary_rows"][1]["behavior"] == "Distance Traveled"
    assert result["raw_bouts"][BID]["behavior"].tolist() == [NEW_NAME]


def _write_external_project(root: Path, name: str) -> None:
    """A minimal external project: behaviors, one session, one cached row."""
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "behavior_definitions.yaml").write_text(
        "behaviors:\n"
        f"  - behavior_id: {BID}\n"
        f"    name: {name}\n",
        encoding="utf-8",
    )
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "import_manifest.json").write_text(
        json.dumps({"linked_sessions": [{"session_id": "s1", "subject_id": "M01"}]}),
        encoding="utf-8",
    )
    cache_dir = derived / "analytics_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "analytics_cache.json").write_text(json.dumps({
        "version": 2,
        "fingerprint": "fp",
        "summary_rows": [
            {"session_id": "s1", "subject": "M01", "behavior_id": BID,
             "behavior": name, "n_bouts": 3.0, "time_spent_s": 7.0},
        ],
    }), encoding="utf-8")


def test_merged_project_matches_on_behavior_id_not_name(tmp_path: Path) -> None:
    """The external project still says "Rear"; the host has been renamed.

    Matching on name alone dropped the behavior into an ``ext::<bid>``
    namespace, so the merged rows stopped joining the host's own.
    """
    external = tmp_path / "external"
    _write_external_project(external, OLD_NAME)
    host_map = {BID: NEW_NAME}
    name_to_host = ProjectMergeService._build_name_to_host_bid(host_map)
    assert name_to_host.get(OLD_NAME.lower()) is None, "name match must have broken"

    summary, raw_bouts, *_rest = ProjectMergeService()._load_one(
        ProjectMergeEntry(project_root=external, tag="ext"),
        host_map, name_to_host, 30.0, use_cached=True,
    )

    assert [r["behavior_id"] for r in summary] == [BID]
    assert [r["behavior"] for r in summary] == [NEW_NAME]

import re
from pathlib import Path

import pytest

from abel.models.schemas import ImportNameSettings
from abel.services.import_service import ImportService
from abel.utils.filename_pattern import generate_capture_regex, snap_selection


def _select(stem: str, text: str) -> tuple[int, int]:
    start = stem.index(text)
    return start, start + len(text)


def _extract(pattern: str, name: str) -> str | None:
    m = re.search(pattern, Path(name).stem)
    return m.group(1) if m else None


@pytest.mark.parametrize(
    ("stem", "selected", "others"),
    [
        # whole leading field; pose suffixes after the field don't matter
        ("COA301", "COA301", {"COA302.tracked.sleap.h5": "COA302", "M7b_x.mp4": "M7b"}),
        # second field
        ("COA301_Day2_TMT", "Day2", {"COA315-Day10-TMT.avi": "Day10", "X1_Base_TMTDLC_resnet.h5": "Base"}),
        # multi-field selection
        ("2026_09_14_mouse3_EPM", "2026_09_14", {"2025_01_02_m9_OFT.mp4": "2025_01_02"}),
        # sub-field selections split at letter/digit/camelCase runs
        ("DG01BehavioralCamera0", "DG01", {"DG12BehavioralCamera1DLC_resnet50.csv": "DG12"}),
        ("DG01BehavioralCamera0", "Camera0", {"DG12BehavioralCamera1DLC_resnet50.csv": "Camera1"}),
        ("cage4mouse12_test", "12", {"cage10mouse3_test.mp4": "3"}),
    ],
)
def test_generated_regex_generalizes(stem: str, selected: str, others: dict[str, str]) -> None:
    start, end = _select(stem, selected)
    pattern = generate_capture_regex(stem, start, end)
    assert _extract(pattern, stem + ".mp4") == selected
    for name, expected in others.items():
        assert _extract(pattern, name) == expected, (pattern, name)


def test_selection_snaps_to_whole_runs_and_trims_separators() -> None:
    stem = "COA301_Day2"
    assert snap_selection(stem, 1, 5) == (0, 6)  # "OA30" -> "COA301"
    assert snap_selection(stem, 6, 11) == (7, 11)  # "_Day2" -> "Day2"
    with pytest.raises(ValueError):
        snap_selection(stem, 6, 7)  # only a separator


def test_generated_patterns_drive_subject_session_pairing() -> None:
    # Pose names carry the parts in a different order, so only the parsed
    # subject + session can pair them.
    video_stem = "COA301_Day2"
    video_settings = ImportNameSettings(
        subject_regex=generate_capture_regex(video_stem, *_select(video_stem, "COA301")),
        session_regex=generate_capture_regex(video_stem, *_select(video_stem, "Day2")),
    )
    service = ImportService()
    manifest = service.build_manifest(
        [Path("COA301_Day2.mp4"), Path("COA302_Day2.mp4")],
        [Path("COA302_Day2_cam.h5"), Path("COA301_Day2_cam.h5")],
        subject_name_settings=video_settings,
    )
    pairs = {
        Path(next(v for v in manifest.videos if v.asset_id == s.video_asset_id).source_path).name:
        Path(next(p for p in manifest.poses if p.asset_id == s.pose_asset_id).source_path).name
        for s in manifest.linked_sessions
    }
    assert pairs == {"COA301_Day2.mp4": "COA301_Day2_cam.h5", "COA302_Day2.mp4": "COA302_Day2_cam.h5"}

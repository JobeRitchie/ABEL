"""The Subject-override list must reach every imported session.

The bug: a project whose subject names never got parsed out of the filenames
carries ``subject_id = None`` on every linked session.  The list required both
a subject id *and* a session id, so it silently dropped all 60 rows and
"Subject override" showed an empty box — with no way to give those sessions the
per-session zones they exist to hold.  Feature extraction does not drop them:
``_build_prep_jobs`` falls back to the session id, keying the ROI
``session::session``, so the list falls back the same way and the key it writes
is the key extraction reads.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset  # noqa: E402
from abel.services.import_service import ImportService  # noqa: E402
from abel.services.roi_service import ROIService, clear_roi_config_cache  # noqa: E402
from abel.ui.tabs.roi_definition_tab import ROIDefinitionTab  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _project(tmp_path, subject_ids):
    """Write a project whose manifest has one session per entry of *subject_ids*."""
    videos, poses, sessions = [], [], []
    for i, subj in enumerate(subject_ids):
        v, p = tmp_path / f"v{i}.mp4", tmp_path / f"v{i}.csv"
        v.write_bytes(b""), p.write_text("x")
        videos.append(VideoAsset(asset_id=f"v{i}", source_path=str(v)))
        poses.append(PoseAsset(asset_id=f"p{i}", source_path=str(p), format="dlc"))
        sessions.append(LinkedSession(session_id=f"session_{i}", video_asset_id=f"v{i}",
                                      pose_asset_id=f"p{i}", subject_id=subj))
    manifest = ImportManifest(videos=videos, poses=poses, linked_sessions=sessions)
    ImportService().save_manifest(tmp_path, manifest)
    return tmp_path


def _subject_tab(qapp, root):
    clear_roi_config_cache()
    tab = ROIDefinitionTab(ROIService(), ImportService())
    tab.set_project(root)
    idx = next(i for i in range(tab._scope.count())
               if tab._scope.itemData(i) == "subject")
    tab._scope.setCurrentIndex(idx)
    return tab


def _keys(tab):
    return [tab._subject_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(tab._subject_list.count())]


def test_named_subjects_are_listed_with_composite_keys(qapp, tmp_path):
    tab = _subject_tab(qapp, _project(tmp_path, ["M1", "M2"]))
    assert _keys(tab) == ["M1::session_0", "M2::session_1"]
    assert "2" in tab._subject_counter.text()


def test_sessions_without_a_parsed_subject_still_appear(qapp, tmp_path):
    """The regression: every row dropped, so the list came up empty."""
    tab = _subject_tab(qapp, _project(tmp_path, [None, None, None]))
    assert _keys(tab) == [
        "session_0::session_0", "session_1::session_1", "session_2::session_2",
    ]


def test_unnamed_subject_label_does_not_repeat_the_session_id(qapp, tmp_path):
    tab = _subject_tab(qapp, _project(tmp_path, [None]))
    text = tab._subject_list.item(0).text()
    assert text.count("session_0") == 1


def test_key_written_for_an_unnamed_subject_is_the_key_extraction_reads(qapp, tmp_path):
    """``_build_prep_jobs`` keys these ``session_id::session_id``."""
    root = _project(tmp_path, [None])
    tab = _subject_tab(qapp, root)
    tab._subject_list.setCurrentRow(0)
    tab._load_target_zones([{"x": 5, "y": 6, "w": 7, "h": 8}])
    tab._save_subject_quiet(tab._current_subject_id())

    clear_roi_config_cache()
    resolved = ROIService().resolve_target_rois(root, "session_0::session_0")
    assert resolved[0] == {"x": 5, "y": 6, "w": 7, "h": 8}


def test_empty_project_says_why_instead_of_showing_nothing(qapp, tmp_path):
    tab = _subject_tab(qapp, _project(tmp_path, []))
    assert tab._subject_list.count() == 1
    hint = tab._subject_list.item(0)
    assert hint.data(Qt.ItemDataRole.UserRole) is None
    assert hint.flags() == Qt.ItemFlag.NoItemFlags
    assert "import" in hint.text().lower()
    assert tab._subject_counter.text().startswith("0 / 0")

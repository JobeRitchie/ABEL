"""Session-type override resolution, editing, and the type-filter selector."""

from __future__ import annotations

import pytest

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset
from abel.services.import_service import ImportService


def _manifest(video_session_id: str | None = None) -> ImportManifest:
    return ImportManifest(
        videos=[
            VideoAsset(
                asset_id="v1",
                source_path="DG01_TestingDay2.avi",
                subject_id="DG01",
                session_id=video_session_id,
            )
        ],
        poses=[
            PoseAsset(
                asset_id="p1",
                source_path="DG01_TestingDay2DLC.csv",
                format="csv",
                subject_id="DG01",
            )
        ],
        linked_sessions=[
            LinkedSession(
                session_id="session_1",
                video_asset_id="v1",
                pose_asset_id="p1",
                subject_id="DG01",
            )
        ],
    )


def test_effective_session_type_prefers_explicit_override() -> None:
    service = ImportService()
    manifest = _manifest(video_session_id="TestingDay2")
    service.update_session_type(manifest, "session_1", "Validation")

    assert manifest.linked_sessions[0].session_type == "Validation"
    assert service.effective_session_type(manifest, manifest.linked_sessions[0]) == "Validation"


def test_effective_session_type_falls_back_to_regex_value() -> None:
    service = ImportService()
    manifest = _manifest(video_session_id="TestingDay2")

    # No override → regex-derived VideoAsset.session_id.
    assert service.effective_session_type(manifest, manifest.linked_sessions[0]) == "TestingDay2"


def test_effective_session_type_falls_back_to_filename_strip() -> None:
    service = ImportService()
    manifest = _manifest(video_session_id=None)

    # No override and no regex value → strip the subject prefix off the stem.
    assert service.effective_session_type(manifest, manifest.linked_sessions[0]) == "TestingDay2"


def test_update_session_type_clears_on_empty_string() -> None:
    service = ImportService()
    manifest = _manifest(video_session_id="TestingDay2")
    service.update_session_type(manifest, "session_1", "Validation")
    service.update_session_type(manifest, "session_1", "   ")

    assert manifest.linked_sessions[0].session_type is None
    # Cleared override falls back to the regex-derived value again.
    assert service.effective_session_type(manifest, manifest.linked_sessions[0]) == "TestingDay2"


def test_session_option_label_includes_type() -> None:
    pytest.importorskip("PySide6.QtWidgets")
    from abel.ui.widgets.session_selection_dialog import SessionOption

    opt = SessionOption(session_id="session_1", subject="DG01", session_type="Validation")
    label = opt.label()
    assert "session_1" in label
    assert "subject: DG01" in label
    assert "type: Validation" in label


def test_session_type_matches_filter() -> None:
    pytest.importorskip("PySide6.QtWidgets")
    from abel.ui.widgets.session_selection_dialog import (
        _ALL_TYPES,
        _UNTYPED,
        session_type_matches,
    )

    # All-types matches everything, including untyped.
    assert session_type_matches(_ALL_TYPES, "Validation")
    assert session_type_matches(_ALL_TYPES, "")
    # Exact type match only.
    assert session_type_matches("Validation", "Validation")
    assert not session_type_matches("Validation", "TestingDay2")
    # Untyped bucket matches only sessions with no type.
    assert session_type_matches(_UNTYPED, "")
    assert not session_type_matches(_UNTYPED, "Validation")


def test_override_survives_regex_reapply() -> None:
    from abel.models.schemas import ImportNameSettings

    service = ImportService()
    manifest = _manifest(video_session_id="TestingDay2")
    service.update_session_type(manifest, "session_1", "Validation")

    service.apply_subject_name_settings(manifest, ImportNameSettings())

    # Reapplying parsing settings rewrites VideoAsset.session_id but must not
    # touch the explicit per-session override.
    assert manifest.linked_sessions[0].session_type == "Validation"
    assert service.effective_session_type(manifest, manifest.linked_sessions[0]) == "Validation"

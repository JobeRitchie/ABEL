"""Session navigation in the multi-animal identity dialog.

Mapping identities used to require selecting each session in the import table
first; the dialog now carries Previous/Next Session buttons that save the
current names and hand a step back to the caller.
"""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.animal_identity_dialog import AnimalIdentityDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _multi(n=8):
    """A tiny two-animal pose set the dialog can render and analyze."""
    from abel.services.pose_processing_service import MultiAnimalPoseData, PoseData

    def pose(offset):
        x = pd.DataFrame({"nose": [float(i + offset) for i in range(n)]})
        y = pd.DataFrame({"nose": [2.0] * n})
        return PoseData(
            body_parts=["nose"], x=x, y=y,
            likelihood=pd.DataFrame({"nose": [1.0] * n}),
            centroid_x=x["nose"].to_numpy(dtype=float),
            centroid_y=y["nose"].to_numpy(dtype=float),
            n_frames=n,
        )

    per = {"Mouse1": pose(0.0), "Mouse2": pose(50.0)}
    return MultiAnimalPoseData(individuals=list(per), per_individual=per, n_frames=n)


def _dialog(index, count):
    return AnimalIdentityDialog(
        session_label="s1",
        multi=_multi(),
        frame_provider=lambda _i: None,
        n_frames=8,
        session_index=index,
        session_count=count,
    )


def test_nav_buttons_absent_for_a_single_session(_app):
    dlg = _dialog(0, 1)
    assert dlg._next_session_btn is None
    assert dlg._prev_session_btn is None


def test_nav_buttons_clamp_at_the_ends(_app):
    first = _dialog(0, 3)
    assert not first._prev_session_btn.isEnabled()
    assert first._next_session_btn.isEnabled()

    last = _dialog(2, 3)
    assert last._prev_session_btn.isEnabled()
    assert not last._next_session_btn.isEnabled()
    assert "3 of 3" in last.windowTitle()


def test_next_session_saves_names_and_reports_the_step(_app):
    dlg = _dialog(0, 3)
    dlg._edits["Mouse1"].setText("green")
    dlg._next_session_btn.click()
    assert dlg.nav_delta == 1
    assert dlg.result_map["Mouse1"] == "green"
    assert dlg.result() == int(dlg.DialogCode.Accepted)


def test_ok_reports_no_step(_app):
    dlg = _dialog(1, 3)
    dlg._on_accept()
    assert dlg.nav_delta == 0

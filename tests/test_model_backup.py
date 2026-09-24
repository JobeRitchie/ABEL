"""A retrain that does not finish must never cost the user their old model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from abel.models.schemas import ReviewerLabelRecord
from abel.services.model_backup import (
    backup_root,
    protect_model_dir,
    recover_interrupted_models,
)
from abel.services.review_service import ReviewService
from abel.utils import cancellation


def _make_model(project: Path, name: str = "behavior_model_Rear", tag: str = "old") -> Path:
    d = project / "derived" / "models" / name
    d.mkdir(parents=True)
    (d / "model_state.pkl").write_text(tag)
    (d / "metrics.json").write_text(json.dumps({"tag": tag}))
    return d


def _tag(model_dir: Path) -> str:
    return (model_dir / "model_state.pkl").read_text()


def _no_backups_left(project: Path) -> bool:
    root = backup_root(project)
    return not root.exists() or not any(root.iterdir())


@pytest.mark.parametrize("start_fresh", [True, False])
def test_finished_retrain_keeps_new_model_and_drops_backup(tmp_path, start_fresh):
    d = _make_model(tmp_path)
    with protect_model_dir(d, start_fresh=start_fresh):
        d.mkdir(exist_ok=True)
        (d / "model_state.pkl").write_text("new")
    assert _tag(d) == "new"
    assert (d / "metrics.json").exists() is (not start_fresh)
    assert _no_backups_left(tmp_path)


@pytest.mark.parametrize("start_fresh", [True, False])
def test_failed_or_cancelled_retrain_restores_old_model(tmp_path, start_fresh):
    d = _make_model(tmp_path)
    with pytest.raises(cancellation.OperationCancelled):
        with protect_model_dir(d, start_fresh=start_fresh):
            d.mkdir(exist_ok=True)
            (d / "model_state.pkl").write_text("half")
            raise cancellation.OperationCancelled()
    assert _tag(d) == "old"
    assert json.loads((d / "metrics.json").read_text()) == {"tag": "old"}
    assert _no_backups_left(tmp_path)


def test_retrain_that_produced_nothing_restores_old_model(tmp_path):
    d = _make_model(tmp_path)
    with protect_model_dir(d, start_fresh=True) as guard:
        guard.restore_previous()
    assert _tag(d) == "old"


def test_nested_guard_defers_to_enclosing_run(tmp_path):
    d = _make_model(tmp_path)
    with protect_model_dir(d, start_fresh=True):
        with protect_model_dir(d) as inner:
            inner.restore_previous()  # no-op: the outer run decides
            d.mkdir(exist_ok=True)
            (d / "model_state.pkl").write_text("new")
    assert _tag(d) == "new"
    assert _no_backups_left(tmp_path)


def test_new_model_name_needs_no_backup(tmp_path):
    d = tmp_path / "derived" / "models" / "behavior_model_New"
    with protect_model_dir(d):
        d.mkdir(parents=True)
        (d / "model_state.pkl").write_text("new")
    assert _tag(d) == "new"
    assert _no_backups_left(tmp_path)


@pytest.mark.parametrize("start_fresh", [True, False])
def test_app_killed_mid_retrain_recovers_old_model_on_open(tmp_path, start_fresh):
    """The reported bug: ABEL closed during Retrain-all left no model."""
    d = _make_model(tmp_path)
    script = textwrap.dedent(f"""
        import os
        from pathlib import Path
        from abel.services.model_backup import protect_model_dir
        d = Path({str(d)!r})
        with protect_model_dir(d, start_fresh={start_fresh}):
            d.mkdir(exist_ok=True)
            (d / "model_state.pkl").write_text("half")
            os._exit(1)  # process killed: no cleanup runs
    """)
    subprocess.run([sys.executable, "-c", script], check=False, cwd=Path(__file__).resolve().parents[1])
    assert _tag(d) == "half"  # the broken state a killed run leaves behind

    assert recover_interrupted_models(tmp_path) == ["behavior_model_Rear"]
    assert _tag(d) == "old"
    assert _no_backups_left(tmp_path)
    # A second open finds nothing to do.
    assert recover_interrupted_models(tmp_path) == []


def test_backup_held_by_another_running_abel_is_left_alone(tmp_path):
    d = _make_model(tmp_path)
    root = backup_root(tmp_path)
    root.mkdir(parents=True)
    (root / d.name).mkdir()
    (root / d.name / "model_state.pkl").write_text("older")
    # The parent process (pytest's launcher) is alive and is not us.
    (root / f"{d.name}.owner.json").write_text(json.dumps({"pid": os.getppid()}))
    assert recover_interrupted_models(tmp_path) == []
    assert _tag(d) == "old"
    with pytest.raises(RuntimeError, match="another ABEL window"):
        with protect_model_dir(d):
            pass


def test_backups_are_not_listed_as_models(tmp_path):
    d = _make_model(tmp_path)
    with protect_model_dir(d, start_fresh=True):
        models = [p.name for p in (tmp_path / "derived" / "models").iterdir()]
        assert models == []  # old model parked outside derived/models


def test_unreadable_label_file_is_never_overwritten(tmp_path):
    svc = ReviewService()
    svc.set_project(tmp_path)
    path = svc._segment_label_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a parquet file")
    with pytest.raises(RuntimeError, match="Nothing was changed"):
        svc.replace_segment_labels([_label("seg_new")])
    with pytest.raises(RuntimeError, match="Nothing was changed"):
        svc.append_segment_label(_label("seg_new"))
    assert path.read_bytes() == b"not a parquet file"


def test_labels_still_append_normally(tmp_path):
    svc = ReviewService()
    svc.set_project(tmp_path)
    svc.append_segment_label(_label("a"))
    svc.replace_segment_labels([_label("b"), _label("a")])
    df = pd.read_parquet(svc._segment_label_path())
    assert sorted(df["segment_id"]) == ["a", "b"]
    assert not list(svc._segment_label_path().parent.glob("*.tmp"))


def _label(segment_id: str) -> ReviewerLabelRecord:
    return ReviewerLabelRecord(segment_id=segment_id, review_label="Rear", reviewer_id="tester")


def test_app_shutdown_cancels_every_job():
    flag = [False]
    try:
        assert not cancellation.is_cancelled(flag)
        cancellation.request_shutdown()
        assert cancellation.is_cancelled(flag)
        assert cancellation.is_cancelled(None)
        with pytest.raises(cancellation.OperationCancelled):
            cancellation.checkpoint()
    finally:
        cancellation._shutdown.clear()

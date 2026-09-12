"""Raw-data availability checks — the guard against silent degradation.

The bug these cover: a project whose videos/pose live on an unmounted drive opens
fine, then every downstream stage that recomputes from raw data produces empty or
NaN output, which reads as a real (negative) result. See
:mod:`abel.services.raw_data_availability`.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset
from abel.services import raw_data_availability as rda
from abel.services.raw_data_availability import (
    KIND_POSE,
    KIND_VIDEO,
    check_manifest_raw_data,
    check_project_raw_data,
)

# Drive letters these tests treat as "not mounted".  They are answered here rather
# than by the OS: on the dev machine J: is a mapped UNC share, and stat'ing it while
# it is unreachable blocked this file for 338 s.
_FAKE_UNMOUNTED = {"H:\\", "J:\\"}


@pytest.fixture(autouse=True)
def _fake_drives(monkeypatch):
    real = rda._exists
    monkeypatch.setattr(
        rda, "_exists",
        lambda p: False if Path(p).anchor in _FAKE_UNMOUNTED else real(p))


def _manifest(entries) -> ImportManifest:
    """Build a manifest from (session_id, video_path, pose_path) triples.

    A ``None`` path means "asset record missing entirely" (never imported), which
    is a different user problem than "path recorded but file gone".
    """
    videos, poses, sessions = [], [], []
    for i, (sid, vpath, ppath) in enumerate(entries):
        vid = f"v{i}" if vpath is not None else ""
        pid = f"p{i}" if ppath is not None else ""
        if vpath is not None:
            videos.append(VideoAsset(asset_id=vid, source_path=str(vpath)))
        if ppath is not None:
            poses.append(PoseAsset(asset_id=pid, source_path=str(ppath), format="dlc"))
        sessions.append(LinkedSession(session_id=sid, video_asset_id=vid,
                                      pose_asset_id=pid, subject_id=f"S{i}"))
    return ImportManifest(videos=videos, poses=poses, linked_sessions=sessions)


def test_all_present_reports_ok(tmp_path):
    v, p = tmp_path / "a.mp4", tmp_path / "a.csv"
    v.write_bytes(b""), p.write_text("x")
    rep = check_manifest_raw_data(_manifest([("s0", v, p)]), tmp_path)
    assert rep.ok and not rep.missing
    assert rep.n_sessions == 1 and rep.n_checked == 2
    assert "reachable" in rep.summary()


def test_unmounted_drive_flags_every_asset_and_names_the_drive(tmp_path):
    """The real HomeCage failure: an H: drive that is simply not mounted."""
    entries = [(f"s{i}", Path(rf"H:\blinded\{i}.mp4"), Path(rf"H:\blinded\{i}.csv"))
               for i in range(3)]
    rep = check_manifest_raw_data(_manifest(entries), tmp_path)
    assert not rep.ok
    assert len(rep.missing) == 6
    assert rep.kinds == {KIND_VIDEO, KIND_POSE}
    # Grouping by volume is what makes the message actionable.
    assert rep.drives() == ["H:"]
    assert len(rep.affected_sessions()) == 3
    assert "H:" in rep.summary()


def test_local_copy_satisfies_a_missing_source_path(tmp_path):
    """A project-local copy is what ABEL actually opens, so it must count."""
    local_v, local_p = tmp_path / "a.mp4", tmp_path / "a.csv"
    local_v.write_bytes(b""), local_p.write_text("x")
    m = _manifest([("s0", Path(r"H:\gone.mp4"), Path(r"H:\gone.csv"))])
    m.videos[0].local_path = str(local_v)
    m.poses[0].local_path = str(local_p)
    assert check_manifest_raw_data(m, tmp_path).ok


def test_missing_pose_only_is_reported_without_touching_video(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"")
    rep = check_manifest_raw_data(
        _manifest([("s0", v, Path(r"H:\gone.csv"))]), tmp_path)
    assert rep.kinds == {KIND_POSE}
    assert len(rep.missing_by_kind(KIND_VIDEO)) == 0
    assert rep.missing_by_kind(KIND_POSE)[0].session_id == "s0"


def test_kinds_filter_limits_what_is_checked(tmp_path):
    """A pose-only consumer (essence mining) should not be blocked on video."""
    entries = [("s0", Path(r"H:\gone.mp4"), Path(r"H:\gone.csv"))]
    rep = check_manifest_raw_data(_manifest(entries), tmp_path, kinds=(KIND_POSE,))
    assert rep.kinds == {KIND_POSE} and rep.n_checked == 1


def test_session_ids_filter_narrows_the_check(tmp_path):
    entries = [(f"s{i}", Path(rf"H:\{i}.mp4"), Path(rf"H:\{i}.csv")) for i in range(3)]
    rep = check_manifest_raw_data(_manifest(entries), tmp_path, session_ids=["s1"])
    assert rep.n_sessions == 1
    assert rep.affected_sessions() == ["s1"]


def test_session_without_asset_record_is_unlinked_not_missing(tmp_path):
    """Never-imported is a different fix than drive-not-mounted; keep them apart."""
    rep = check_manifest_raw_data(_manifest([("s0", None, None)]), tmp_path)
    assert not rep.ok
    assert rep.unlinked_sessions == ["s0"]
    assert not rep.missing


def test_signature_is_stable_per_problem_and_changes_with_it(tmp_path):
    """Drives the once-per-distinct-problem dialog cadence."""
    a = _manifest([("s0", Path(r"H:\a.mp4"), Path(r"H:\a.csv"))])
    b = _manifest([("s0", Path(r"J:\a.mp4"), Path(r"J:\a.csv"))])
    sig_a1 = check_manifest_raw_data(a, tmp_path).signature()
    sig_a2 = check_manifest_raw_data(a, tmp_path).signature()
    sig_b = check_manifest_raw_data(b, tmp_path).signature()
    assert sig_a1 == sig_a2          # same problem → warn once
    assert sig_a1 != sig_b           # different drive → warn again


def test_hung_network_volume_is_probed_once_not_stat_per_file(tmp_path, monkeypatch):
    """A mapped-but-unreachable share blocks each stat for minutes; the check must
    give up on the volume after one short probe and never stat its files."""
    release = threading.Event()
    file_stats: list[Path] = []

    def fake_exists(p):
        p = Path(p)
        if p.anchor == "K:\\":
            if p == Path(p.anchor):
                release.wait(30)      # the volume root hangs like an SMB timeout
                return False
            file_stats.append(p)
        return p.exists()

    monkeypatch.setattr(rda, "_exists", fake_exists)
    monkeypatch.setattr(rda, "VOLUME_PROBE_TIMEOUT_S", 0.2)
    v, p = tmp_path / "a.mp4", tmp_path / "a.csv"
    v.write_bytes(b""), p.write_text("x")
    entries = [("s0", v, p)] + [(f"s{i}", Path(rf"K:\{i}.mp4"), Path(rf"K:\{i}.csv"))
                                for i in range(1, 4)]
    try:
        t0 = time.monotonic()
        rep = check_manifest_raw_data(_manifest(entries), tmp_path)
        elapsed = time.monotonic() - t0
        # A second check while the first probe is still stuck reuses its verdict.
        t1 = time.monotonic()
        rep2 = check_manifest_raw_data(_manifest(entries), tmp_path)
        elapsed2 = time.monotonic() - t1
    finally:
        release.set()
    assert elapsed < 2.0 and elapsed2 < 0.1
    assert not file_stats
    assert len(rep.missing) == 6 and rep.drives() == ["K:"]
    assert rep.signature() == rep2.signature()
    rda._stuck_probes.clear()


def test_project_with_no_manifest_is_ok_not_a_warning(tmp_path):
    """A brand-new project has nothing to be missing; warning there is noise."""
    assert check_project_raw_data(tmp_path).ok


def test_check_project_reads_the_manifest_on_disk(tmp_path):
    m = _manifest([("s0", Path(r"H:\gone.mp4"), Path(r"H:\gone.csv"))])
    out = tmp_path / "derived" / "review_tables"
    out.mkdir(parents=True)
    (out / "import_manifest.json").write_text(
        json.dumps(m.model_dump(mode="json")), encoding="utf-8")
    rep = check_project_raw_data(tmp_path)
    assert not rep.ok and rep.drives() == ["H:"]

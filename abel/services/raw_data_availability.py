"""Is the project's raw data actually reachable right now?

ABEL projects reference their source videos and pose files by path.  Those paths
routinely point at removable or network storage (``H:\\``, ``J:\\``, a UNC share),
so a project that worked yesterday can open today with every raw asset missing,
the drive simply is not mounted.  Nothing about the project is corrupt; it is
just unreadable.

The failure mode this guards against is *silence*.  Downstream stages degrade
quietly when the raw data is gone: clip metrics come back all-NaN, essence mining
falls back to random, clip extraction writes empty crops, a validation arm
disables itself.  The user sees a finished run with a plausible-looking figure and
no indication that a whole input was missing.

So availability is checked once, centrally, and reported *up front*, see
:func:`check_project_raw_data`.  The UI layer turns the report into a dialog
(:mod:`abel.ui.raw_data_warning`); headless callers can read the same report.

Checks are existence-only (``Path.exists``), never a read, so a 47-session project
resolves in milliseconds.  A drive letter that is not mapped at all fails fast on
its own, but a *mapped* network share that is unreachable (VPN down, ``J:`` →
``\\\\ad.unc.edu\\...``) blocks every ``stat`` for the full SMB timeout, measured
~158 s per path, uncached.  So each volume is probed once per check with a short
timeout (:func:`_hung_volumes`), and files on a hung volume are reported missing
without being stat'ed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from abel.models.schemas import ImportManifest

KIND_VIDEO = "video"
KIND_POSE = "pose"

KIND_LABELS = {
    KIND_VIDEO: "video",
    KIND_POSE: "pose",
}

# What breaks when each asset kind is unreachable, shown in the warning so the
# user can judge whether to proceed or go mount the drive.
KIND_IMPACT = {
    KIND_VIDEO: ("clip extraction, review clips, video features, crops and any "
                 "overlay or playback"),
    KIND_POSE: ("pose feature extraction, clip mining / Essence Miner, and "
                "anything that recomputes from raw pose"),
}


@dataclass
class MissingAsset:
    """One unreachable raw file, with enough context to find it again."""

    session_id: str
    kind: str                 # KIND_VIDEO | KIND_POSE
    path: Path
    subject_id: str = ""

    @property
    def drive(self) -> str:
        """Drive root / UNC share of the missing path ("" when relative).

        Missing files cluster by *volume*, not by session, one unmounted drive
        explains 47 missing files, so this is what the summary groups on.
        """
        try:
            anchor = self.path.anchor
        except Exception:
            return ""
        return anchor.rstrip("\\/") if anchor else ""


@dataclass
class RawDataReport:
    """Availability of every raw asset a project's sessions reference."""

    project_root: Path
    n_sessions: int = 0
    n_checked: int = 0
    missing: list[MissingAsset] = field(default_factory=list)
    # Sessions whose manifest entry has no asset record at all (never imported),
    # kept separate from "path recorded but file gone", different user action.
    unlinked_sessions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unlinked_sessions

    @property
    def kinds(self) -> set[str]:
        return {m.kind for m in self.missing}

    def missing_by_kind(self, kind: str) -> list[MissingAsset]:
        return [m for m in self.missing if m.kind == kind]

    def affected_sessions(self) -> list[str]:
        seen: list[str] = []
        for m in self.missing:
            if m.session_id not in seen:
                seen.append(m.session_id)
        return seen

    def drives(self) -> list[str]:
        """Distinct volumes the missing files live on, most-affected first."""
        counts: dict[str, int] = {}
        for m in self.missing:
            d = m.drive
            if d:
                counts[d] = counts.get(d, 0) + 1
        return sorted(counts, key=lambda d: -counts[d])

    def signature(self) -> str:
        """Stable identity of *this* problem, for once-per-problem warnings.

        Re-warning on every tab switch is noise; staying silent after the user
        mounts a different drive (a genuinely new problem) is worse.  Keying on
        the missing set gives the dialog the right cadence: once per distinct
        problem, again when the problem changes.
        """
        parts = sorted(f"{m.kind}:{m.path}" for m in self.missing)
        parts += sorted(f"unlinked:{s}" for s in self.unlinked_sessions)
        return f"{self.project_root}|" + "|".join(parts)

    def summary(self) -> str:
        """One-line plain-text summary (log lines, status bars, headless runs)."""
        if self.ok:
            return f"All {self.n_checked} raw files reachable."
        bits: list[str] = []
        for kind in (KIND_VIDEO, KIND_POSE):
            n = len(self.missing_by_kind(kind))
            if n:
                bits.append(f"{n} {KIND_LABELS[kind]} file(s)")
        if self.unlinked_sessions:
            bits.append(f"{len(self.unlinked_sessions)} session(s) with no linked asset")
        where = ""
        drv = self.drives()
        if drv:
            where = f" on {', '.join(drv)}"
        return (f"{', '.join(bits)} unreachable{where} "
                f"({len(self.affected_sessions())} of {self.n_sessions} sessions).")


def check_manifest_raw_data(
    manifest: ImportManifest,
    project_root: Path,
    *,
    kinds: tuple[str, ...] = (KIND_VIDEO, KIND_POSE),
    session_ids: list[str] | None = None,
) -> RawDataReport:
    """Existence-check the raw assets referenced by ``manifest``.

    Mirrors :meth:`ImportService.video_path_for_session` /
    ``pose_path_for_session`` resolution order, local project copy first, then
    the original source path, so an asset ABEL *can* open is never reported
    missing.  ``session_ids`` narrows the check to the sessions a caller actually
    needs (a single-session preview does not care about the other 46).
    """
    report = RawDataReport(project_root=Path(project_root))
    videos = {v.asset_id: v for v in manifest.videos}
    poses = {p.asset_id: p for p in manifest.poses}

    sessions = list(manifest.linked_sessions)
    if session_ids is not None:
        wanted = {str(s) for s in session_ids}
        sessions = [s for s in sessions if str(s.session_id) in wanted]
    report.n_sessions = len(sessions)

    hung = _hung_volumes(
        Path(c).anchor
        for sess in sessions
        for table, asset_id in ((videos, sess.video_asset_id), (poses, sess.pose_asset_id))
        if asset_id and asset_id in table
        for c in (table[asset_id].local_path, table[asset_id].source_path) if c
    )

    for sess in sessions:
        for kind, table, asset_id in (
            (KIND_VIDEO, videos, sess.video_asset_id),
            (KIND_POSE, poses, sess.pose_asset_id),
        ):
            if kind not in kinds:
                continue
            asset = table.get(asset_id) if asset_id else None
            if asset is None:
                if sess.session_id not in report.unlinked_sessions:
                    report.unlinked_sessions.append(str(sess.session_id))
                continue
            report.n_checked += 1
            candidates = [c for c in (asset.local_path, asset.source_path) if c]
            if any(Path(c).anchor not in hung and _exists(Path(c)) for c in candidates):
                continue
            report.missing.append(MissingAsset(
                session_id=str(sess.session_id), kind=kind,
                path=Path(candidates[-1]) if candidates else Path(""),
                subject_id=str(sess.subject_id or ""),
            ))
    return report


def check_project_raw_data(
    project_root: Path,
    *,
    kinds: tuple[str, ...] = (KIND_VIDEO, KIND_POSE),
    session_ids: list[str] | None = None,
) -> RawDataReport:
    """Load the project's import manifest and check its raw assets.

    Returns an empty (``ok``) report when there is no manifest yet, a brand-new
    project has nothing to be missing, and warning there would be noise.
    """
    root = Path(project_root)
    from abel.services.import_service import ImportService  # local: avoids a cycle

    manifest = ImportService().load_manifest(root)
    if manifest is None:
        return RawDataReport(project_root=root)
    return check_manifest_raw_data(manifest, root, kinds=kinds, session_ids=session_ids)


# How long a volume root may take to answer before it is treated as unreachable.
# A reachable share answers in milliseconds; an unreachable mapped one takes minutes.
VOLUME_PROBE_TIMEOUT_S = 2.0

# Probes still stuck in a network timeout, by anchor.  A later check reuses the
# verdict instead of stacking another blocked thread on the same volume.
_stuck_probes: dict[str, threading.Thread] = {}
_stuck_lock = threading.Lock()


def _hung_volumes(anchors, timeout: float | None = None) -> set[str]:
    """Anchors (drive roots / UNC shares) whose root does not answer in ``timeout``.

    Only a *hang* counts.  A root that answers, present or not, is left to the
    per-file check, so a share whose root is merely unlistable (permissions) is
    never mis-reported as missing.
    """
    timeout = VOLUME_PROBE_TIMEOUT_S if timeout is None else timeout
    probes: dict[str, threading.Thread] = {}
    hung: set[str] = set()
    for anchor in {a for a in anchors if a}:
        with _stuck_lock:
            stuck = _stuck_probes.get(anchor)
            if stuck is not None and stuck.is_alive():
                hung.add(anchor)
                continue
            _stuck_probes.pop(anchor, None)
        th = threading.Thread(target=_exists, args=(Path(anchor),), daemon=True,
                              name=f"abel-volume-probe {anchor}")
        th.start()
        probes[anchor] = th
    # One shared deadline: probing three dead shares costs one timeout, not three.
    deadline = time.monotonic() + timeout
    for anchor, th in probes.items():
        th.join(max(0.0, deadline - time.monotonic()))
        if th.is_alive():
            hung.add(anchor)
            with _stuck_lock:
                _stuck_probes[anchor] = th
    return hung


def _exists(path: Path) -> bool:
    """``Path.exists`` that treats an unreachable volume as "missing", not an error.

    An unmounted network share raises OSError rather than returning False on
    Windows; for our purposes both mean the same thing.
    """
    try:
        return path.exists()
    except OSError:
        return False

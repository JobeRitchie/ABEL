"""Is the project's raw data actually reachable right now?

ABEL projects reference their source videos and pose files by path.  Those paths
routinely point at removable or network storage (``H:\\``, ``J:\\``, a UNC share),
so a project that worked yesterday can open today with every raw asset missing —
the drive simply is not mounted.  Nothing about the project is corrupt; it is
just unreadable.

The failure mode this guards against is *silence*.  Downstream stages degrade
quietly when the raw data is gone: clip metrics come back all-NaN, essence mining
falls back to random, clip extraction writes empty crops, a validation arm
disables itself.  The user sees a finished run with a plausible-looking figure and
no indication that a whole input was missing.

So availability is checked once, centrally, and reported *up front* — see
:func:`check_project_raw_data`.  The UI layer turns the report into a dialog
(:mod:`abel.ui.raw_data_warning`); headless callers can read the same report.

Checks are existence-only (``Path.exists``), never a read, so a 47-session project
resolves in milliseconds and an unmounted drive fails fast instead of blocking on
a network timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from abel.models.schemas import ImportManifest

KIND_VIDEO = "video"
KIND_POSE = "pose"

KIND_LABELS = {
    KIND_VIDEO: "video",
    KIND_POSE: "pose",
}

# What breaks when each asset kind is unreachable — shown in the warning so the
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

        Missing files cluster by *volume*, not by session — one unmounted drive
        explains 47 missing files — so this is what the summary groups on.
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
    # kept separate from "path recorded but file gone" — different user action.
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
    ``pose_path_for_session`` resolution order — local project copy first, then
    the original source path — so an asset ABEL *can* open is never reported
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
            if any(_exists(Path(c)) for c in candidates):
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

    Returns an empty (``ok``) report when there is no manifest yet — a brand-new
    project has nothing to be missing, and warning there would be noise.
    """
    root = Path(project_root)
    from abel.services.import_service import ImportService  # local: avoids a cycle

    manifest = ImportService().load_manifest(root)
    if manifest is None:
        return RawDataReport(project_root=root)
    return check_manifest_raw_data(manifest, root, kinds=kinds, session_ids=session_ids)


def _exists(path: Path) -> bool:
    """``Path.exists`` that treats an unreachable volume as "missing", not an error.

    An unmounted network share raises OSError rather than returning False on
    Windows; for our purposes both mean the same thing.
    """
    try:
        return path.exists()
    except OSError:
        return False

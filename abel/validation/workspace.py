"""Persistent workspace for the external validation suite.

A validation *session* is the setup a run was made from: which projects were
loaded, which behaviors were checked/unchecked, and every display rename the
user applied to a project or a behavior.  None of that lives in any ABEL
project — it is cross-project, it is the thing a reviewer asks about six months
later ("what exactly went into figure 3?"), and until now it evaporated when the
window closed.

Everything lives under one home root (``~/ABEL Validation`` by default,
overridable with ``ABEL_VALIDATION_HOME``), with each run filed inside the
session it came from::

    ~/ABEL Validation/
      sessions/
        manuscript-main/
          session.json      ← projects, checked behaviors, renames, holdout settings
          SETUP.md          ← the same thing, readable
          runs/
            run_2026-07-21_101500/   ← ResultsStore output (cells.parquet, report.html, …)
              session.json           ← the setup as it was for THAT run

The per-run copy matters: the session file keeps changing as the user adds
projects, but a finished run's setup must never change under it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from abel.storage.file_store import write_json
from abel.validation.datamodel import ProjectRef

HOME_ENV = "ABEL_VALIDATION_HOME"
DEFAULT_HOME_NAME = "ABEL Validation"
SESSION_FILE = "session.json"
RUNS_DIRNAME = "runs"


def workspace_root() -> Path:
    """The validation home directory (``~/ABEL Validation`` unless overridden)."""
    env = str(os.environ.get(HOME_ENV, "")).strip()
    return Path(env) if env else Path.home() / DEFAULT_HOME_NAME


def sessions_root() -> Path:
    return workspace_root() / "sessions"


def slugify(name: str) -> str:
    """Folder-safe stem for a session name ('Manuscript — main' → 'manuscript-main')."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return slug or "session"


# ── the captured setup ──────────────────────────────────────────────────────


@dataclass
class BehaviorEntry:
    """One behavior as the user left it: checked or not, renamed or not."""

    behavior_id: str
    disk_name: str          # name in the project's behavior_definitions.yaml
    display_name: str       # what the suite reports it as (== disk_name if unrenamed)
    checked: bool = False

    @property
    def renamed(self) -> bool:
        return self.display_name != self.disk_name


@dataclass
class ProjectEntry:
    """One loaded project, keyed by the path it was loaded from."""

    project_id: str         # display name; also the key every figure groups by
    source_name: str        # the project's own name on disk
    root: str
    behaviors: list[BehaviorEntry] = field(default_factory=list)

    @property
    def renamed(self) -> bool:
        return bool(self.source_name) and self.project_id != self.source_name

    @property
    def checked_ids(self) -> list[str]:
        return [b.behavior_id for b in self.behaviors if b.checked]


@dataclass
class SessionRecord:
    """A named, reloadable validation setup."""

    name: str
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""
    projects: list[ProjectEntry] = field(default_factory=list)
    holdout: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.name)

    # ── capture ──
    @classmethod
    def capture(
        cls,
        name: str,
        projects: "dict[str, ProjectRef] | list[ProjectRef]",
        selected: dict[str, set[str]] | dict[str, list[str]],
        holdout: dict[str, Any] | None = None,
        notes: str = "",
        created_at: str = "",
        keep_entries: "list[ProjectEntry] | None" = None,
    ) -> "SessionRecord":
        """Snapshot the current Projects-tab state into a record.

        ``keep_entries`` carries forward projects that could not be loaded this
        time (an unmounted drive, typically).  Without it, re-saving a session
        that was reloaded while a drive was offline would quietly erase those
        projects from the record — losing the setup, not just the access.
        """
        refs = list(projects.values()) if isinstance(projects, dict) else list(projects)
        entries: list[ProjectEntry] = []
        for proj in refs:
            chosen = set(selected.get(proj.project_id, ()) or ())
            behaviors = [
                BehaviorEntry(
                    behavior_id=str(bid),
                    disk_name=proj.behavior_disk_name(bid),
                    display_name=proj.behavior_label(bid),
                    checked=str(bid) in chosen,
                )
                for bid in proj.behavior_names
                if str(bid) != "no_behavior"
            ]
            entries.append(ProjectEntry(
                project_id=proj.project_id,
                source_name=proj.original_name,
                root=str(proj.root),
                behaviors=behaviors,
            ))
        live_roots = {e.root for e in entries}
        entries += [e for e in (keep_entries or []) if e.root not in live_roots]
        now = datetime.now().isoformat(timespec="seconds")
        return cls(
            name=str(name).strip() or "session",
            created_at=created_at or now,
            updated_at=now,
            notes=notes,
            projects=entries,
            holdout=dict(holdout or {}),
        )

    # ── serialization ──
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Derived flags are for the reader of the JSON, not for round-tripping.
        for pe, src in zip(d["projects"], self.projects):
            pe["renamed"] = src.renamed
            for be, bsrc in zip(pe["behaviors"], src.behaviors):
                be["renamed"] = bsrc.renamed
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionRecord":
        projects: list[ProjectEntry] = []
        for p in raw.get("projects", []) or []:
            behaviors = [
                BehaviorEntry(
                    behavior_id=str(b.get("behavior_id", "")),
                    disk_name=str(b.get("disk_name", "")),
                    display_name=str(b.get("display_name", "") or b.get("disk_name", "")),
                    checked=bool(b.get("checked", False)),
                )
                for b in (p.get("behaviors", []) or [])
                if str(b.get("behavior_id", "")).strip()
            ]
            projects.append(ProjectEntry(
                project_id=str(p.get("project_id", "")),
                source_name=str(p.get("source_name", "") or p.get("project_id", "")),
                root=str(p.get("root", "")),
                behaviors=behaviors,
            ))
        return cls(
            name=str(raw.get("name", "session")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            notes=str(raw.get("notes", "")),
            projects=projects,
            holdout=dict(raw.get("holdout", {}) or {}),
        )

    # ── restore ──
    def restore(self) -> "RestoreResult":
        """Rebuild live :class:`ProjectRef` objects + the checked-behavior map.

        Projects are re-read from disk (so behaviors added since the save show
        up) and the saved renames are re-applied on top.  Anything that has
        since moved or disappeared is reported rather than silently dropped, so
        a reloaded setup can never quietly differ from the saved one.
        """
        refs: dict[str, ProjectRef] = {}
        selected: dict[str, set[str]] = {}
        missing_projects: list[str] = []
        missing_behaviors: list[str] = []
        unavailable: list[ProjectEntry] = []

        def _unavailable(entry: ProjectEntry, why: str) -> None:
            unavailable.append(entry)
            missing_projects.append(f"{entry.project_id} — {why} ({entry.root})")

        for entry in self.projects:
            root = Path(entry.root)
            if not root.exists():
                _unavailable(entry, "path not reachable")
                continue
            try:
                proj = ProjectRef.load(root)
            except Exception:  # noqa: BLE001 — an unreadable project is a report, not a crash
                _unavailable(entry, "could not be read")
                continue
            if not proj.is_valid():
                # Reachable but no training set — the same bar Add Project(s) applies.
                _unavailable(entry, "no derived/training_sets/training_set.parquet")
                continue

            if entry.project_id != proj.project_id:
                proj.rename(entry.project_id)
            chosen: set[str] = set()
            for beh in entry.behaviors:
                if beh.behavior_id not in proj.behavior_names:
                    missing_behaviors.append(f"{entry.project_id}: {beh.display_name}")
                    continue
                if beh.renamed:
                    proj.set_behavior_alias(beh.behavior_id, beh.display_name)
                if beh.checked:
                    chosen.add(beh.behavior_id)
            if proj.project_id in refs:
                # project_id is the key every figure groups by — two projects under
                # one id would merge their results into a single bar.
                _unavailable(entry, f"name '{proj.project_id}' collides with another project")
                continue
            refs[proj.project_id] = proj
            selected[proj.project_id] = chosen

        return RestoreResult(refs, selected, missing_projects, missing_behaviors, unavailable)

    # ── human-readable mirror ──
    def to_markdown(self) -> str:
        lines = [f"# Validation session — {self.name}", ""]
        lines.append(f"- Created: {self.created_at}")
        lines.append(f"- Last saved: {self.updated_at}")
        if self.holdout:
            lines.append("- Held-out settings: "
                         + ", ".join(f"{k}={v}" for k, v in sorted(self.holdout.items())))
        if self.notes:
            lines += ["", self.notes]
        for entry in self.projects:
            title = entry.project_id
            if entry.renamed:
                title += f"  (renamed from '{entry.source_name}')"
            lines += ["", f"## {title}", f"`{entry.root}`", ""]
            checked = [b for b in entry.behaviors if b.checked]
            unchecked = [b for b in entry.behaviors if not b.checked]
            lines.append(f"{len(checked)}/{len(entry.behaviors)} behaviors included.")
            lines.append("")
            for label, group in (("Included", checked), ("Excluded", unchecked)):
                if not group:
                    continue
                lines.append(f"**{label}**")
                for b in group:
                    suffix = f"  (renamed from '{b.disk_name}')" if b.renamed else ""
                    lines.append(f"- {b.display_name}{suffix}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class RestoreResult:
    projects: dict[str, ProjectRef]
    selected: dict[str, set[str]]
    missing_projects: list[str] = field(default_factory=list)
    missing_behaviors: list[str] = field(default_factory=list)
    # Saved entries that could not be loaded — pass back into SessionRecord.capture
    # as ``keep_entries`` so re-saving does not erase them.
    unavailable: list[ProjectEntry] = field(default_factory=list)


@dataclass
class SessionInfo:
    """Listing entry — enough to populate a picker without loading everything."""

    name: str
    slug: str
    path: Path
    updated_at: str = ""
    n_projects: int = 0
    n_behaviors: int = 0
    n_runs: int = 0


# ── the store ───────────────────────────────────────────────────────────────


class SessionStore:
    """Named validation setups on disk, each owning its own ``runs/`` folder."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else sessions_root()

    # ── paths ──
    def session_dir(self, name: str) -> Path:
        return self.root / slugify(name)

    def session_path(self, name: str) -> Path:
        return self.session_dir(name) / SESSION_FILE

    def runs_dir(self, name: str) -> Path:
        """The output root a run launched from this session writes into."""
        d = self.session_dir(name) / RUNS_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exists(self, name: str) -> bool:
        return self.session_path(name).exists()

    # ── read / write ──
    def save(self, session: SessionRecord) -> Path:
        session.updated_at = datetime.now().isoformat(timespec="seconds")
        d = self.session_dir(session.name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / SESSION_FILE
        write_json(path, session.to_dict())
        (d / "SETUP.md").write_text(session.to_markdown(), encoding="utf-8")
        return path

    def load(self, name: str) -> SessionRecord:
        path = self.session_path(name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SessionRecord.from_dict(raw)

    def list_sessions(self) -> list[SessionInfo]:
        """Every saved session, most recently saved first."""
        out: list[SessionInfo] = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob(f"*/{SESSION_FILE}")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a corrupt session must not hide the rest
                continue
            projects = raw.get("projects", []) or []
            n_beh = sum(
                1 for p in projects for b in (p.get("behaviors", []) or [])
                if b.get("checked")
            )
            runs = path.parent / RUNS_DIRNAME
            out.append(SessionInfo(
                name=str(raw.get("name", path.parent.name)),
                slug=path.parent.name,
                path=path,
                updated_at=str(raw.get("updated_at", "")),
                n_projects=len(projects),
                n_behaviors=n_beh,
                n_runs=len(list(runs.glob("run_*"))) if runs.exists() else 0,
            ))
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def delete(self, name: str) -> None:
        """Remove the session file only — never its runs (those are results)."""
        self.session_path(name).unlink(missing_ok=True)
        (self.session_dir(name) / "SETUP.md").unlink(missing_ok=True)

    # ── run bookkeeping ──
    def attach_to_run(self, session: SessionRecord, run_dir: str | Path) -> Path | None:
        """Freeze the setup inside a finished run directory.

        The session file keeps mutating as the user works; a run's own copy is
        what makes its numbers attributable months later.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            return None
        path = run_dir / SESSION_FILE
        write_json(path, session.to_dict())
        (run_dir / "SETUP.md").write_text(session.to_markdown(), encoding="utf-8")
        return path

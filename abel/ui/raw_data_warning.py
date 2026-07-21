"""The "your raw data is unreachable" warning dialog, shared by every tab.

:mod:`abel.services.raw_data_availability` decides *whether* raw video/pose files
are reachable; this module decides *how the user hears about it*.  It exists as
one shared widget rather than a per-tab message box so the wording, the
once-per-problem cadence, and the "which drive?" diagnosis are identical
everywhere — a user who sees it on the Features tab should see the same thing on
Validation.

Cadence is the whole design problem.  Warning on every tab switch trains people to
dismiss it; warning once per session hides a genuinely new problem when a
different drive drops.  :class:`RawDataWarningPresenter` therefore keys on the
report's :meth:`~abel.services.raw_data_availability.RawDataReport.signature` —
the missing set itself — so the dialog appears once per *distinct* problem and
again the moment that problem changes.

DPI note: the dialog sizes from the font metrics, not fixed pixels, so it does not
clip under Windows display scaling.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.services.raw_data_availability import (
    KIND_IMPACT,
    KIND_LABELS,
    KIND_POSE,
    KIND_VIDEO,
    RawDataReport,
    check_project_raw_data,
)

# How many individual paths to list before collapsing into "+N more".  Enough to
# recognise the pattern, few enough that the dialog stays readable.
_MAX_LISTED = 8


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def format_report_html(report: RawDataReport) -> str:
    """Render a report as the dialog's body: what's missing, where, what breaks."""
    parts: list[str] = []
    n_sess = len(report.affected_sessions())
    parts.append(
        f"<p><b>{n_sess} of {report.n_sessions} session(s)</b> reference raw files "
        f"that ABEL cannot read right now.</p>")

    drives = report.drives()
    if drives:
        parts.append(
            "<p>The missing files are on "
            + ", ".join(f"<b>{_esc(d)}</b>" for d in drives)
            + ". If that is a removable or network drive, connecting it and "
              "reopening the project is usually the whole fix.</p>")

    for kind in (KIND_VIDEO, KIND_POSE):
        missing = report.missing_by_kind(kind)
        if not missing:
            continue
        parts.append(
            f"<p><b>{len(missing)} {KIND_LABELS[kind]} file(s) missing.</b><br>"
            f"<span style='color:#a33;'>Affects: {KIND_IMPACT[kind]}.</span></p>")
        parts.append("<ul style='margin-top:2px;'>")
        for m in missing[:_MAX_LISTED]:
            who = f" <i>({_esc(m.subject_id)})</i>" if m.subject_id else ""
            parts.append(f"<li><code>{_esc(m.path)}</code>{who}</li>")
        if len(missing) > _MAX_LISTED:
            parts.append(f"<li>… and {len(missing) - _MAX_LISTED} more</li>")
        parts.append("</ul>")

    if report.unlinked_sessions:
        parts.append(
            f"<p><b>{len(report.unlinked_sessions)} session(s)</b> have no linked "
            f"video/pose asset at all. These were never fully imported — re-run "
            f"the pairing step on the Data Import tab.</p>")

    parts.append(
        "<p style='color:#666;'>You can keep working: steps that read only "
        "already-extracted features still run. Steps that recompute from raw "
        "video or pose will be skipped or produce empty results — which is easy "
        "to mistake for a real finding, so check this before interpreting "
        "output.</p>")
    return "".join(parts)


class RawDataWarningDialog(QDialog):
    """Modal warning listing unreachable raw files and what they block."""

    def __init__(self, report: RawDataReport, parent: QWidget | None = None,
                 *, allow_mute: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Raw data not available")
        self.setModal(True)
        self._report = report

        layout = QVBoxLayout(self)
        headline = QLabel(report.summary())
        headline.setWordWrap(True)
        headline.setStyleSheet("font-weight: 600;")
        layout.addWidget(headline)

        body = QTextBrowser()
        body.setOpenExternalLinks(False)
        body.setHtml(format_report_html(report))
        layout.addWidget(body, 1)

        self._mute = QCheckBox("Don't warn me again for this project in this session")
        if allow_mute:
            layout.addWidget(self._mute)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        # Size from font metrics so the dialog scales with the user's DPI /
        # font-size settings instead of clipping at 125%+ scaling.
        em = self.fontMetrics().horizontalAdvance("M")
        self.resize(em * 46, em * 34)

    def muted(self) -> bool:
        return bool(self._mute.isChecked())


class RawDataWarningPresenter:
    """Owns *when* the warning shows, so callers only say "check this project".

    One instance per main window.  Tabs and the project loader call
    :meth:`check` freely — on every tab switch if they like — and the presenter
    suppresses repeats of a problem the user has already seen, and everything for
    a project the user has muted.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        self._parent = parent
        self._seen: set[str] = set()          # report signatures already shown
        self._muted_projects: set[str] = set()

    def reset(self, project_root: Path | None = None) -> None:
        """Forget history — call when a project is opened or its assets change.

        Reopening a project is the user's way of saying "I fixed it"; the warning
        must be able to fire again (or stay silent) based on fresh evidence.
        """
        self._seen.clear()
        if project_root is not None:
            self._muted_projects.discard(str(project_root))

    def check(self, project_root: Path | None, *, force: bool = False,
              **kwargs) -> RawDataReport | None:
        """Check ``project_root`` and warn if anything is unreachable.

        Returns the report (``None`` when there is no project). ``force`` shows
        the dialog even for an already-seen problem — for the explicit
        "Check raw data" action, where silence would read as a broken button.
        Extra kwargs pass through to
        :func:`~abel.services.raw_data_availability.check_project_raw_data`
        (e.g. ``kinds=``, ``session_ids=``).
        """
        if project_root is None:
            return None
        try:
            report = check_project_raw_data(Path(project_root), **kwargs)
        except Exception:
            # Availability checking must never be the thing that breaks a tab.
            return None
        if report.ok:
            return report
        if not force:
            if str(project_root) in self._muted_projects:
                return report
            sig = report.signature()
            if sig in self._seen:
                return report
            self._seen.add(sig)
        self._show(report, project_root)
        return report

    def _show(self, report: RawDataReport, project_root: Path) -> None:
        dlg = RawDataWarningDialog(report, self._parent)
        dlg.exec()
        if dlg.muted():
            self._muted_projects.add(str(project_root))


def confirm_run_with_missing_raw_data(
    parent: QWidget | None,
    project_roots: list[Path],
    *,
    what: str = "this run",
    **kwargs,
) -> bool:
    """Gate a long, expensive run on the user seeing what raw data is missing.

    Returns True to proceed.  Unlike the passive tab warning this is a *decision*:
    a multi-hour validation run whose pose files are unreachable produces figures
    with silently-dropped arms, and the user must be the one who chooses to accept
    that.  Checks every project in the run and reports them together, since a
    per-project cascade of dialogs is what makes people click through blindly.
    """
    reports = []
    for root in project_roots:
        try:
            rep = check_project_raw_data(Path(root), **kwargs)
        except Exception:
            continue
        if not rep.ok:
            reports.append(rep)
    if not reports:
        return True

    from PySide6.QtWidgets import QMessageBox  # local: keeps import cost off startup

    lines = "".join(
        f"<li><b>{_esc(Path(r.project_root).name)}</b> — {_esc(r.summary())}</li>"
        for r in reports)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Raw data not available")
    box.setText(f"Some raw data for {_esc(what)} cannot be read.")
    box.setInformativeText(
        f"<ul>{lines}</ul>"
        "<p>Analyses that recompute from raw video or pose will be <b>skipped or "
        "disabled</b>. The run will still finish and still produce figures — those "
        "figures will just be missing whole arms, with no marking on the figure "
        "itself.</p>"
        "<p>Connect the drive and re-open the projects to fix this, or continue "
        "with the reduced run.</p>")
    box.setDetailedText("\n\n".join(
        f"{Path(r.project_root)}\n" + "\n".join(
            f"  [{m.kind}] {m.path}" for m in r.missing)
        for r in reports))
    box.setStandardButtons(QMessageBox.StandardButton.Cancel
                           | QMessageBox.StandardButton.Ok)
    box.button(QMessageBox.StandardButton.Ok).setText("Run anyway")
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return box.exec() == QMessageBox.StandardButton.Ok


def warn_if_raw_data_missing(
    parent: QWidget | None, project_root: Path | None, **kwargs
) -> RawDataReport | None:
    """One-shot check + warn, for callers with no presenter to hand.

    Always shows the dialog when something is missing (no de-duplication), so use
    it for deliberate, user-initiated actions — pressing Run on a long pipeline —
    rather than anything that fires repeatedly.
    """
    if project_root is None:
        return None
    try:
        report = check_project_raw_data(Path(project_root), **kwargs)
    except Exception:
        return None
    if not report.ok:
        RawDataWarningDialog(report, parent, allow_mute=False).exec()
    return report

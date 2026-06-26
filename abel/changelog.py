"""ABEL version history, shown in the Info → Version History tab.

Keep newest first. When you bump ``abel.__version__`` for a release, add an
entry here and update ``VERSION_DATE`` to that release's date.
"""

from __future__ import annotations

# Date of the current ``abel.__version__`` release.
VERSION_DATE = "June 26, 2026"

# (version, date, [bullet lines]) — newest first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    ("0.5.1", "June 26, 2026", [
        "Data Import: new \"Rename Body Parts\" tool to give keypoints new names "
        "of your choosing that propagate to all downstream processing (feature "
        "extraction, context features, trained models). Renames now correctly "
        "invalidate cached features so re-extraction rebuilds under the new "
        "names, and no longer raise a spurious \"keypoints don't match the "
        "project scheme\" warning.",
        "Direct Use: the source model's feature settings — \"Include video "
        "features\" and the per-feature/robustness toggles — now carry over to "
        "the new project, so re-running models keeps the intended video/context "
        "features instead of silently dropping them.",
        "Direct Use: added an adjustable zoom for ROI drawing that persists as "
        "you step through subjects, and tidied the tab into collapsible steps.",
        "Feature extraction: pose, context, and representation inputs are now "
        "pre-built and cached during feature extraction (and reused by Active "
        "Learning) with a clearer progress timeline.",
        "Housekeeping: ignore stray spreadsheet files so they don't clutter the "
        "project.",
    ]),
    ("0.5.0", "June 23, 2026", [
        "First public release of ABEL — Active-learning Behavior Estimation and "
        "Labeling: a local-first desktop app for reproducible, human-in-the-loop "
        "behavior modeling from DLC-tracked rodent videos. Includes the full "
        "pipeline — data import, behavior definition, pose + context feature "
        "extraction, ROI editing, active-learning training and review, temporal "
        "refinement, cross-project model/example reuse, behavior analytics, and "
        "export.",
    ]),
]


def format_changelog() -> str:
    """Render the changelog as a plain-text block for display."""
    lines: list[str] = []
    for version, date, bullets in CHANGELOG:
        header = f"Version {version}   •   {date}"
        lines.append(header)
        lines.append("─" * 72)
        for b in bullets:
            lines.append(f"  • {b}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

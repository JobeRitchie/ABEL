"""ABEL version history, shown in the Info → Version History tab.

Keep newest first. When you bump ``abel.__version__`` for a release, add an
entry here and update ``VERSION_DATE`` to that release's date.
"""

from __future__ import annotations

# Date of the current ``abel.__version__`` release.
VERSION_DATE = "June 30, 2026"

# (version, date, [bullet lines]) — newest first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    ("0.6.2", "June 30, 2026", [
        "Behavior Grid: the montage now fills all 25 cells when enough bouts "
        "exist. It still places the most confident detections first, but "
        "backfills the remaining cells with the next-strongest bouts instead of "
        "leaving them blank (previously only the top ~40% by probability were "
        "shown, so behaviors with fewer strong detections produced a partly "
        "empty grid).",
        "Behavior Grid: added a “Dot size” control to scale the overlaid "
        "pose-tracking keypoints up or down (0.3×–5×), persisted per project.",
        "Behavior Grid: raised the crop multiplier limit from 3× to 8× for "
        "wider zoom-out (the crop is still capped at the full source frame).",
    ]),
    ("0.6.1", "June 30, 2026", [
        "Baseline import: when a source project's trained models can't be "
        "imported because this project is missing some of the feature columns "
        "they were trained on, the Import Baseline dialog now explains why. A "
        "new “Diagnose models” helper groups the missing columns into families "
        "(e.g. video/optical-flow context, inter-keypoint distances, "
        "oscillation) and gives ordered fixes — most often enabling “Include "
        "video features” and re-extracting so the host produces the same "
        "columns. This clarifies the previously confusing case where the "
        "feature-schema coverage read 100% but models still showed a lower "
        "percentage and were silently skipped.",
        "Remove Sessions now also deletes each removed session's syllable "
        "assignments (derived/syllables/<session>_syllables.npz), which were "
        "previously left orphaned on disk.",
        "Maintenance: corrected the default subject-name extraction tests to "
        "match real separator-delimited filenames, and removed obsolete tests "
        "for the retired temporal-refinement training internals (temporal "
        "refinement is now inference-only).",
    ]),
    ("0.6.0", "June 30, 2026", [
        "Multi-animal tracking: ABEL now loads multi-animal DLC pose files "
        "(CSV or H5), keeping one cleaned track per tracked individual. Data "
        "Import gains a visual Identity Map dialog to assign each tracked "
        "individual to a real subject and to correct identity swaps frame-by-"
        "frame, and those corrections are applied on load so all downstream "
        "features see identity-consistent tracks.",
        "Social features and behaviors: multi-animal projects can extract "
        "inter-animal interaction (social_*) features, and social-behavior "
        "fields now appear in the Behavior tab only when the project tracks "
        "more than one animal. Enabling or disabling social features rebuilds "
        "the pose feature cache; solo single-animal projects are unaffected.",
        "Baseline import (Model Refinement): import another project as a "
        "baseline — a detection summary previews each source behavior with its "
        "importable example count, model coverage, feature-schema coverage, and "
        "whether it matches an existing host behavior. Importing brings over "
        "labeled examples and models, auto-creating matched or new host "
        "behaviors so the merged training set resolves to defined behaviors. "
        "Preview and import run off the UI thread.",
        "Behavior Analytics: ROI zones and inter-keypoint distance measures are "
        "now exposed as synthetic \"pseudo-behavior\" rows alongside scored "
        "behaviors, and short on/off runs are debounced so brief flicker no "
        "longer fragments the analytics.",
        "Smoothing preview now overlays body-part dots and a centroid trail on "
        "the video frame so the effect of smoothing/interpolation settings is "
        "visible directly on the animal.",
        "Distance-feature canonicalization (extends 0.5.2): symmetric pairwise "
        "distance columns (dist_a_to_b / dist_b_to_a) are merged onto a single "
        "canonical sorted name before any statistics are computed, so mixed-"
        "order pose exports no longer leave half-populated \"dead\" distance "
        "columns. The representation cache signature now includes a parquet "
        "footer-statistics digest, and a cache-version bump rebuilds segment "
        "features under the corrected schema.",
        "Added a \"clear feature caches\" option that deletes all generated "
        "feature artefacts so the next run rebuilds every stage from the source "
        "pose/video — the nuclear option when stale caches are suspected.",
        "Validation: positive example bouts are now spread across subjects "
        "using the import manifest's session→subject mapping, giving more "
        "representative validation cells.",
    ]),
    ("0.5.3", "June 26, 2026", [
        "Fixed the Features tab not remembering settings across a project "
        "reload: restoring presets during project load fired change handlers "
        "that overwrote the project's saved settings (most visibly flipping the "
        "\"Include video features\" checkbox back off) before they were read. "
        "Settings writes are now suspended while a project loads.",
    ]),
    ("0.5.2", "June 26, 2026", [
        "Cross-project feature compatibility: pairwise inter-keypoint distance "
        "features (dist_A_to_B) are now named in a canonical, order-independent "
        "way, so two projects with the same keypoints listed in a different DLC "
        "column order produce identical feature columns. This fixes Direct Use "
        "model reuse failing because the projects had mismatched feature sets.",
        "Feature extraction now rebuilds cached pose features when keypoint "
        "renames are applied after a first extraction (previously the rename was "
        "silently ignored and old body-part names persisted), and rebuilds "
        "context features when the ROI configuration changes (previously stale "
        "ROIs were reused). A pose feature-format version forces a one-time "
        "rebuild so existing projects adopt the compatible schema.",
    ]),
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

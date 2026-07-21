"""ABEL version history, shown in the Info → Version History tab.

Keep newest first. When you bump ``abel.__version__`` for a release, add an
entry here and update ``VERSION_DATE`` to that release's date.
"""

from __future__ import annotations

# Date of the current ``abel.__version__`` release.
VERSION_DATE = "July 21, 2026"

# (version, date, [bullet lines]) — newest first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    ("0.9.0", "July 21, 2026", [
        "External Validation & Meta-Analysis Suite now ships with ABEL: the "
        "cross-project suite (learning curves, ablation, behavior "
        "discrimination, generalization / human agreement, active learning vs. "
        "random, behaviorscape, video-feature value, throughput) and its "
        "launcher are part of the application rather than a local-only tool, "
        "with its test suite included.",
        "Validation sessions: a run's setup — which projects were loaded, which "
        "behaviors were checked or unchecked, and every project/behavior rename "
        "applied on top — is now saved and reloadable, and each run is filed "
        "inside the session it came from with a frozen copy of that setup. "
        "Everything lives in one workspace folder (default “ABEL Validation” in "
        "your user folder, or set ABEL_VALIDATION_HOME), so results are no "
        "longer scattered next to whichever project happened to be first. "
        "Reloading reports anything that moved, lost its training set, or would "
        "collide by name, and a project on an unmounted drive is kept in the "
        "record rather than quietly erased from it.",
        "Rare-behavior discovery analysis: a new validation arm measures "
        "whether ABEL's clip hunting (essence mining, active learning, UMAP "
        "neighbourhoods) actually finds a rare behavior faster than random or "
        "whole-video review, with a cheap up-front rarity + evidence check so a "
        "hunt is never launched on too few confirmed positives, plus "
        "cross-validated enrichment, effort-to-quality curves and rarity "
        "scaling.",
        "Feature-role clustering: a new analysis clusters behaviors by which "
        "feature modality they actually rely on (pose, kinematics, context, "
        "video), reporting each cluster's over-pose ΔF1 so “what kind of "
        "measurement does this behavior need?” has an answer backed by the "
        "ablation numbers.",
        "Prism-ready and meta-summary exports: every validation run now writes "
        "GraphPad-shaped pivots (prism/) and consolidated summary tables "
        "(summary/) automatically, so figures no longer need hand-reformatting "
        "of the tidy results table.",
        "Assay-scoped behaviors in validation: behaviors with the same name in "
        "different assays (EPM “Rear” vs. OFT “Rear”) are never pooled, and the "
        "behaviorscape PERMANOVA is reported as descriptive rather than "
        "inferential where pooling would have been required.",
        "Project and behavior renaming in the validation suite: rename either "
        "for reporting and the new name flows into every figure, table and "
        "export, while lookups on disk keep using the original name — so "
        "matching names across projects merges them in the generalization "
        "figure without touching any project.",
        "Essence Extractor over the shipped feature space: exemplar-driven clip "
        "mining can now range over the same ~1100 extracted features the "
        "classifier consumes, not just the ~30 interpretable clip metrics, with "
        "a contrastive search that picks the features separating exemplars from "
        "the background pool, human-readable criteria labels, and degenerate "
        "features (constant across the fixed window) excluded by construction.",
        "Raw-data availability warning: a project whose videos or pose files "
        "live on an unmounted or unreachable drive now says so when you open a "
        "tab that needs them, and before a long validation run starts, instead "
        "of silently degrading into empty results — once per distinct problem, "
        "app-wide.",
        "Calibration leakage fix: probability calibration can now be fit on a "
        "dedicated split the model was never trained on and that the run is not "
        "scored on. Where a caller supplies that split, calibration is skipped "
        "rather than quietly falling back to the scored split.",
        "Review tab behavior filter no longer polluted by UMAP display labels: "
        "coordinates carrying display names (“A + B” multi-labels, cluster "
        "names, short-name codes) are mapped back to real behavior ids, and "
        "anything that doesn't resolve to one defined behavior lands as "
        "unassigned instead of creating a duplicate filter entry.",
        "Clip-mining and Review fixes: the source-filter button count now "
        "matches the sources actually present, mined clips that are already "
        "reviewed are surfaced when reviewed rows are shown, “no matches” is "
        "distinguished from “all matches hidden by the filter”, and edge-case "
        "candidates selected in Active Learning now land in the Clips tab "
        "visibly instead of looking like a no-op.",
    ]),
    ("0.8.0", "July 16, 2026", [
        "Methods tab: a new project-independent tab documenting ABEL's "
        "statistical procedures for users and reviewers — a References subtab "
        "(the peer-reviewed source justifying each procedure, with links) and a "
        "Formulas subtab (the raw formula ABEL evaluates, each tied to its code), "
        "rendered from a single methods-content source of truth.",
        "Targeted Clip Mining: a new dialog turns each candidate window into "
        "interpretable, physically-meaningful metrics (time in a zone, distance "
        "to a zone, centroid speed, distance travelled, body elongation, …), then "
        "lets you mine every clip whose metrics satisfy user-defined criteria — "
        "and an Essence Extractor that infers those criteria automatically from a "
        "handful of exemplar clips.",
        "Deployment-accurate model evaluation: a new refined-evaluation engine "
        "grades trained models on the bouts the product actually ships (smooth → "
        "threshold → merge close bouts → drop short bouts, using each behavior's "
        "Temporal Review settings) instead of the raw prob ≥ 0.5 cut, reports "
        "held-out refined metrics, and guards against evaluation leakage from the "
        "deploy model.",
        "Automatic temporal-refinement settings: ABEL can now search a grid of "
        "(onset threshold, min bout duration, merge gap) against held-out "
        "probabilities and reviewer labels and suggest the combination that "
        "maximizes event-level bout F1 — the same number the Temporal Review tab "
        "and Validation report judge, using shared bout-matching primitives.",
        "Calibrated, cross-run ETAs: a per-project timing profile records the "
        "wall-clock cost of each pipeline phase (Preparing, Training, Scoring, "
        "Evaluating, Benchmarking, …) observed in ANY run — single or batch, "
        "retrain / pipeline / run-model — so a later run of any kind seeds a "
        "calibrated estimate, and batch runs show a whole-run ETA rather than "
        "summed per-phase guesses.",
        "Session types and smarter session selection: sessions now carry an "
        "editable Session Type; a shared 'Choose Sessions' dialog (Active "
        "Learning and Temporal Refinement) filters by type with a 'Check all of "
        "type' button; and ABEL detects duplicate imported sessions and repairs "
        "stale session references.",
        "Removal cascade: deleting sessions or behaviors now prunes every derived "
        "artifact that referenced them — per-session parquet/JSON caches, review "
        "work, and trained-model label references — so inference and analytics "
        "never read orphaned data.",
        "Imported-model label-map consistency: shared target-class resolution "
        "(tolerant of punctuation and case) plus an import-time remap and a repair "
        "pass keep an imported model pointed at the correct target behavior even "
        "when its stored behavior ids differ from the host project's.",
        "Regenerate Missing Clips: Active Learning persists every ranked window "
        "but extracts clips for only a subset; a new Review-tab action "
        "regenerates the missing clips on demand.",
        "Validation platform expansion: additional publication-grade checks "
        "(leave-one-subject-out, held-out leakage guards, model-vs-human "
        "agreement reporting CCC and bias, class discrimination, feature-bucket "
        "coverage), a consolidated suite report, GUI panels, and Prism/CSV export.",
        "Faster XGBoost inference: predictions now run on the CPU via DMatrix so a "
        "GPU-fit booster no longer copies the whole feature matrix host→device on "
        "every call, alongside broader fusion-inference performance work.",
        "Advanced ROI features: freehand polygon ROIs, easier num-animals "
        "editing, and additional per-ROI geometric features computed per tracked "
        "individual.",
        "Spatial analytics — clean backgrounds and manuscript-consistent figures: "
        "Density Analysis and the Spatial Heatmap now build the background by "
        "temporal-median compositing across video frames, so the moving animal is "
        "removed instead of leaving ghost blobs. The Spatial Heatmap caches this "
        "plate with a 'Regenerate BG' button so it isn't rebuilt every render. "
        "The Group Comparison map gains an 'averaging radius' control that pools "
        "each group's density over a wider area, merging small opposite-sign "
        "specks into the surrounding trend. Both tabs gained shared "
        "contrast/brightness/sharpness/blur and custom-image controls and can "
        "reuse each other's exact background, so figures match without re-tuning. "
        "The Spatial Heatmap now draws from the same filtered temporal-review "
        "bouts as the rest of Analytics, fixing behaviors that plotted nothing.",
        "Robustness and internals: input preflight validation fails a "
        "preprocessing run fast with an actionable message; worker errors surface "
        "the exception message first instead of truncating it away; atomic "
        "parquet writes; structured multi-animal label persistence (bulk upsert, "
        "per-window structured labels); active-learning sample weighting with a "
        "GPU-fit fallback; inline UMAP; and substantially expanded test coverage.",
    ]),
    ("0.7.0", "July 1, 2026", [
        "SLEAP import: ABEL now imports SLEAP prediction files (.slp) directly. "
        "Data Import detects a SLEAP pose file, offers to convert it to ABEL's "
        "DLC-style format, and wires the converted per-individual tracks through "
        "the same pipeline used for native DeepLabCut output.",
        "Multi-animal behavior soundboard: the pop-out soundboard now supports "
        "structured multi-animal labeling — pick a behavior, then designate "
        "which animal (solo) or which two animals (social: actor → recipient for "
        "directed interactions, or the pair for mutual ones). A new “Commit” "
        "button persists the clip's collected labels; each label is shown as a "
        "removable chip until committed.",
        "“A mouse is a mouse” pooling: multi-animal labels are keyed to each "
        "animal's own segment with an identity-agnostic behavior label, so "
        "instances pool across animals at training time. Directed social "
        "behaviors label only the actor; mutual behaviors label both animals; "
        "and multiple behaviors on one animal in one window merge into a "
        "co-occurring label instead of being dropped as ambiguous.",
        "Clip identity overlays: multi-animal review clips draw a colored dot "
        "per tracked animal with a legend (using the same palette as the "
        "Identity Map) so reviewers can tell which individual is which. Clips "
        "are now centered on the average centroid over the whole clip, removing "
        "the jitter that occurred when the per-frame centroid was unstable.",
        "Social analytics: new per-dyad interaction summaries (inter-animal "
        "distance, contact time and bouts, approach/advance balance, "
        "orientation) plus a cohort-pooled dominance HMM with a "
        "spatial-displacement dominance score and per-session ranking, surfaced "
        "in the Behavior Analytics tab.",
        "Per-individual context features: in multi-animal projects, context "
        "(ROI/video) features are now computed per tracked individual so each "
        "animal's segments carry their own identity-consistent features.",
        "Freehand polygon ROIs plus easier num-animals editing; the ROI tab "
        "uses the full multi-animal session name for a single shared arena ROI.",
        "Fixes: committed multi-animal soundboard labels now use the resolved "
        "animal id, so they correctly join to their segment features at training "
        "time (previously the raw track id was used and every soundboard label "
        "silently missed the join). Pose/video filename matching no longer "
        "truncates names containing a dotted “mp4/avi/mov/mkv” letter sequence "
        "mid-word. The social advance-fraction metric is no longer diluted by "
        "undetected frames.",
    ]),
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

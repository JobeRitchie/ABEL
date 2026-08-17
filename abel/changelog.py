"""ABEL version history, shown in the Info → Version History tab.

Keep newest first. When you bump ``abel.__version__`` for a release, add an
entry here and update ``VERSION_DATE`` to that release's date.
"""

from __future__ import annotations

# Date of the current ``abel.__version__`` release.
VERSION_DATE = "August 16, 2026"

# (version, date, [bullet lines]) — newest first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    ("0.11.0", "August 16, 2026", [
        "Behaviors that look different but move alike are now separable: every "
        "segment carries a 512-dimensional R3D-18 appearance embedding alongside its "
        "pose and kinematic features. These are ordinary numeric feature columns, so "
        "training, the ablation harness and the validation suite all see them without "
        "special-casing, and they are on by default wherever pixel features are "
        "available. This replaces the post-hoc fusion path, which measured as inert "
        "and has been removed — appearance information now enters before training "
        "rather than being blended onto finished predictions.",
        "Dense temporal traces use the appearance features they were trained on. "
        "Dense inference rebuilds its windows from the frame-level table, which never "
        "contained the segment-level R3D columns, so all 512 were silently filled with "
        "zeros — every trace in the app was scored by a model missing the features it "
        "relied on. Behaviors that pose alone cannot express collapsed outright (EPM "
        "Stretch Attend produced literally no detections), while pose-redundant ones "
        "over-detected, because the appearance features had been suppressing their "
        "false positives. Embeddings are now computed on an anchor grid aligned to the "
        "training stride — so the existing cache is reused rather than re-decoded — and "
        "interpolated onto each window. Direct Use shares this path and was affected "
        "identically. Any dense trace or exported bout set produced since appearance "
        "features were introduced should be regenerated.",
        "A model can no longer learn where its training rows came from. Symmetric "
        "pairwise distances have two possible spellings (``dist_a_to_b`` / "
        "``dist_b_to_a``), and the training set kept whichever one each row arrived "
        "with — so rows imported from another project and rows labeled natively ended "
        "up in different columns, each half-populated with complementary blanks. The "
        "pattern of missing values then encoded provenance, which a tree model will "
        "happily split on; since deployment data all looks like one stratum, a behavior "
        "whose positives lived in the other stratum simply stopped firing. In fear "
        "conditioning this took Explore/Walk from one detection across 426,411 windows "
        "to 25,446, with held-out F1 essentially unchanged (0.973 → 0.971) — the score "
        "had been measured entirely inside the stratum deployment never visits. "
        "Distances are now canonicalised both when the training set is written and when "
        "it is loaded, so existing projects repair themselves on the next retrain.",
        "Keypoint-MoSeq syllable discovery, UMAP QC/export, and post-hoc fusion "
        "inference have been removed. The syllable subsystem had become unreachable "
        "from the UI and no longer imported; the fusion path was measured and found to "
        "add nothing over training on appearance features directly. Roughly 4,700 lines "
        "of dead code, one tab, and their dependency requirements are gone, which also "
        "shortens first-run setup.",
        "Targeted clip mining ranks by a learned essence rather than a box of "
        "thresholds. Given exemplar clips, an L1-penalised logistic fit selects a sparse "
        "subset from up to 300 candidate features jointly — against hard negatives drawn "
        "from the background — instead of intersecting per-feature ranges, which failed "
        "whenever the defining combination was not axis-aligned. The displayed criteria "
        "still show the interpretable box; a stateless separation-ranked fallback covers "
        "projects with too few exemplars to fit.",
        "Held-out data no longer leaks back in through temporal-review feedback. "
        "When sessions were unticked in the training selector, any of them that had "
        "been reviewed in the Temporal Review tab still contributed hard-negative and "
        "hard-positive rows, pulled from the segment cache by session name with no "
        "scope check — so the very mice being held out were reintroduced, tuned by the "
        "reviewer's own corrections on those mice. Feedback intervals are now filtered "
        "to the selected sessions before any relabeling or injection runs, and the log "
        "records how many out-of-scope sessions were dropped. Any model trained with a "
        "restricted session set and reviewed feedback should be retrained.",
        "Leave-one-mouse-out CV reports what a manuscript can publish. Alongside the "
        "existing macro numbers it now returns target-class precision, recall and F1 "
        "and pooled PR-AUC — at 3-8 % prevalence a macro average pairs the target's F1 "
        "with a ~0.97 “not this behavior” F1 and puts a ~0.50 floor under a detector "
        "that never fires. The interval to publish is a subject-level bootstrap CI "
        "(2000 replicates resampling mice with replacement, rescoring the pooled "
        "held-out rows), because the animal is the unit of analysis. The per-fold SEM "
        "is retained for the existing figures but explicitly marked invalid as an "
        "error bar — LOSO folds share (N−2)/(N−1) of their training data, and Bengio & "
        "Grandvalet (2004) proved no unbiased estimator of CV variance exists.",
        "Leave-one-mouse-out CV lets you choose the cohort. A picker lists every mouse "
        "with its labeled-window and session counts; an unchecked mouse is dropped from "
        "the analysis entirely — never held out and never trained on — so a broken "
        "camera, a mis-tracked session or an off-protocol subject can be removed "
        "without contaminating any fold's training pool. A restricted run says so on "
        "the figure, in the results panel and in the CSV, which now also carries the "
        "included and excluded subject lists.",
        "Leave-one-mouse-out CV accounts for the folds it could not score. Skipped "
        "folds are recorded with their reason (no target positives in the held-out "
        "mouse, empty pool, no finite scores), the scored/skipped counts travel with "
        "the result, and a run that loses more than a quarter of its folds is flagged — "
        "a mean over surviving folds only is survivorship-biased. A per-subject table "
        "of TP/FP/FN/TN and target-class F1 ships with every run.",
        "Competition / Direct-Use temporal refinement measures each behavior instead of "
        "“any behavior”. In that mode the generic probability column holds the "
        "per-frame maximum across behaviors, so thresholding it emitted a bout wherever "
        "anything was happening — inflating bout counts and collapsing latency toward "
        "zero. Postprocessing now splits the trace into its per-behavior probability "
        "columns and writes one bout set per real behavior, keyed separately so the "
        "Temporal Review session list is unaffected.",
        "Adaptive model complexity is off by default. Validation across 43 behaviors "
        "found no benefit from auto-tuning n_estimators/max_depth (mean ΔF1 −0.005, "
        "improving 7 of 43), so runs are now reproducible at the stated hyperparameters "
        "rather than at whatever the data talked the tuner into.",
        "Spatial heatmaps and density analysis pool mice recorded at different "
        "resolutions correctly. Pose coordinates live in the pixel space of the video "
        "each subject was tracked on, so raw pooling squeezed lower-resolution subjects "
        "into the top-left corner and let higher-resolution subjects overflow the arena. "
        "Every session is now rescaled onto the reference frame (the resolution the "
        "background plate and axes are drawn in), probing the video file when the "
        "manifest has no dimensions. The arena is also drawn at its true aspect ratio "
        "rather than stretched to fill the axes box, which had been making wide, narrow "
        "enclosures look nearly square.",
        "Learning curves report error rates as percentages of the held-out set, with "
        "confidence intervals. Raw confusion counts are dominated by behaviors with "
        "larger held-out sets and by the shifting behavior mix as the clip budget grows; "
        "the per-behavior percentages average fairly. The across-behavior average curve "
        "now also reaches the Prism export — previously the Prism tables carried every "
        "behavior except the pooled line users actually plot — and figures can be "
        "re-rendered from the saved CSVs without retraining.",
        "Rare-behavior discovery gains the pooled, across-model exports the manuscript "
        "asks for: the discovery curve averaged over every project that hunted a rare "
        "behavior, and the human scoring time each strategy costs — minutes to reach a "
        "quality target, minutes to find N confirmed examples including the whole-video "
        "scan arm, and minutes of review to reach the learning-curve knee for a strong "
        "model. Clip length is each project's own median rather than a global constant, "
        "and every table carries N so an arm that reached the target in two projects is "
        "not read as fast.",
    ]),
    ("0.10.0", "July 23, 2026", [
        "Pooled discrimination landscape: the whole run's pairwise-discrimination "
        "result now reads off one two-panel figure instead of a matrix per project "
        "per feature family. Left, every behavior pair of every assay is placed by "
        "how much error pose alone leaves against the share of that error the best "
        "feature family removes, coloured by which modality does the work — so the "
        "pairs pose already solves, the pairs a modality rescues, and the pairs that "
        "are still unsolved separate into visible groups. Right, a volcano over every "
        "pair × family sets effect size against reproducibility across seeds, with "
        "both a p<0.05 and a Benjamini-Hochberg 5% FDR line, since a full run tests "
        "40-100 combinations. The per-project matrices remain behind it as per-assay "
        "detail.",
        "Held-out confusion counts as a first-class output: TP / FN / FP per assay "
        "and behavior are now promoted to their own table, figure, Prism export and "
        "plain-language finding. F1 and PR-AUC are the comparable numbers, but counts "
        "are what a reviewer can check against their own scoring — with the counts "
        "taken from the headline cells only (never summed across seeds, which would "
        "advertise an n the study never had) and the “these are windows, not bouts” "
        "caveat carried alongside them.",
        "Behavior rarity and prevalence reporting: validation now measures how rare "
        "each behavior actually is, preferring dense temporal-refinement probability "
        "traces, then bouts, then confirmed labels, and reports which source it used. "
        "Per-session zero-inflation is reported separately from prevalence where the "
        "behavior does occur, so a behavior absent from most sessions is no longer "
        "averaged into looking merely uncommon.",
        "Rare-behavior discovery expanded: effort-to-quality curves, clips-to-target "
        "effort tables, rarity-scaling curves and per-seed replicate Prism exports for "
        "every arm, plus label-coverage reporting against the full window pool (the "
        "behavior grid unioned with the enrichment cache) so a poor join is visible "
        "rather than silently shrinking the analysis.",
        "Refinement evaluability gate: temporal refinement is no longer scored on "
        "held-out sets where the labels are too sparse to score it. Segments are split "
        "into runs of genuinely observed frames, and a set that cannot support the "
        "measurement says so instead of returning bout metrics whose false positives "
        "are interpolation gaps and whose false negatives are label fragmentation.",
        "Validation suite interface rebuilt around its results: every analysis tab is "
        "now settings and a plain-language explanation on the left, figures on the "
        "right, with a scrollable thumbnail grid that fills the pane and re-flows on "
        "resize, and an “Open Data Folder” button that goes straight to the figures, "
        "CSVs and intermediates that tab produced. Forms survive narrow columns and "
        "Windows display scaling rather than clipping their controls.",
        "Prism-ready exports for the scatter analyses: the discrimination landscape "
        "and volcano ship as Multiple-variables tables (one row per point, carrying "
        "its x, y, family, assay, size and significance), and the raw per-seed ROC-AUCs "
        "ship as replicate subcolumns so the paired test can be re-run in Prism rather "
        "than taken on trust. Active-learning, effort-to-quality and rarity curves "
        "likewise export per-seed replicates rather than pre-averaged means.",
        "Paired significance testing unified and corrected: the ablation, video-value "
        "and discrimination analyses now share one paired t-test. Its zero-variance "
        "guard is a tolerance rather than an exact comparison — three identical "
        "differences leave ~1e-18 of floating-point dust, which previously slipped "
        "past the guard and produced p ≈ 1e-33 out of no variance at all.",
        "Prism exports no longer zero a real p-value: the rounding rule that collapses "
        "sub-1e-9 float dust to a clean zero is right for a confidence-interval "
        "half-width and wrong for a p of 9e-10, which was exporting as “0” — a value a "
        "p-value can never take.",
        "Discrimination matrix cells no longer conflate three meanings: a behavior "
        "pair skipped for too few clips is now hatched like any other untrained pair "
        "instead of rendering in the same grey as “already solved by the baseline” — a "
        "reading that is impossible for a pair that was never scored.",
    ]),
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

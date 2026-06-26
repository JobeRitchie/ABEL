==============================================================================
 ABEL - Active-learning Behavior Estimation and Labeling
==============================================================================

 Version 0.5.3 | Python 3.10+ | UNC Non-Commercial License

ABEL is a local-first desktop application for reproducible, human-in-the-loop
behavior modeling from DLC-tracked videos. It is built around a pose-first
active-learning workflow, with optional video-derived features for
context-sensitive behaviors. Source data and all derived artifacts stay in your
project folder - nothing is uploaded anywhere.


------------------------------------------------------------------------------
 INSTALLATION
------------------------------------------------------------------------------

ABEL requires Python 3.10 or newer.

Option A - one-click launcher (Windows)
---------------------------------------
Double-click run_abel.bat. It creates a virtual environment, installs the app
in editable mode, runs a PySide6 self-test, and launches the GUI. Re-run it any
time to update.

Option B - manual install
-------------------------
    python -m venv .venv
    .venv\Scripts\activate        (Windows)
    source .venv/bin/activate     (macOS / Linux)
    pip install -e .
    abel

Optional dependency groups
--------------------------
Heavy or task-specific dependencies are opt-in and can also be managed from the
in-app Dependencies tab:

    pip install -e ".[preprocessing]"   video features: opencv, scipy, imageio
    pip install -e ".[gpu]"             torch (GPU fusion backend)
    pip install -e ".[benchmarks]"      xgboost
    pip install -e ".[syllables]"       keypoint-moseq, umap-learn, hdbscan
    pip install -e ".[dev]"             pytest
    pip install -e ".[all]"             everything above


------------------------------------------------------------------------------
 CORE WORKFLOW
------------------------------------------------------------------------------

 1. Import videos + pose files and auto-link sessions.
 2. Define behaviors and add seed examples.
 3. Configure pose features (window duration, stride, smoothing) and optionally
    enable video-derived features (optical flow, motion).
 4. Run Active Learning:
      - framewise pose feature extraction
          (derived/pose_features/frame_pose.parquet)
      - optional framewise context extraction
          (derived/context_features/frame_context.parquet)
          - only when video features are enabled
      - frame and segment representations
          (derived/representations/*.parquet)
      - supervised training with context-padded label propagation
          (derived/models/<model_version>/)
      - uncertainty scoring and candidate ranking
          (derived/review_tables/candidate_segments.json)
 5. Extract clips for selected segments.
 6. Review and relabel segments.
 7. Retrain and rerank candidates.
 8. Validate models with the Validation tab (blind quiz + reliability metrics).
 9. Export labels, reports, and bout-level assay outputs.


------------------------------------------------------------------------------
 APPLYING MODELS TO NEW PROJECTS
------------------------------------------------------------------------------

 - Direct Use: replay a trained workflow on new videos/pose files without
   retraining. A workflow snapshot captures every behavior's model (full
   multi-behavior competition), window/stride, temporal-refinement thresholds
   and bout settings, and whether video/context features were used. The tab
   walks through source project -> input data -> pixel/mm calibration ->
   keypoint mapping -> run.
 - Keypoint mapping: when new DLC files name keypoints differently
   (e.g. back_mid vs center_body), map them onto the names the model expects so
   derived features line up. Auto-suggested, saved per source project, and also
   surfaced as a warning + remap in Data Import.
 - Transfer Feedback (Direct Use subtab): after refreshing analytics, scores
   how well the model transferred per subject and across the population, with
   red flags (near-zero detections, population outliers, stuck-high/lost-low
   confidence runs, behavior-profile divergence) and a per-subject deep-dive.
 - Model Refinement: import labeled examples from other projects sharing the
   same keypoint scheme, merge into the training set, and retrain. Incompatible
   schemas are detected and blocked.
 - Run Models (Active Learning): score a chosen subset of behaviors with their
   existing trained models, no retraining.


------------------------------------------------------------------------------
 SCIENTIFIC / ENGINEERING HIGHLIGHTS
------------------------------------------------------------------------------

 - Multi-behavior supervised modeling (target behavior selectable per
   active-learning run)
 - Pose-only or pose+video feature modes selectable at project setup
 - Overlap-aware negative learning rule for behavior interactions
 - Uncertainty components: entropy, ensemble variance, density outlier,
   optional margin term
 - Optional selective fusion for uncertain segments (3D CNN backend when
   available with robust fallback)
 - Group-aware splitting by subject/session
 - Interactive ROI definition with drag-to-draw for both Target Zone and
   Subject Crop, plus Copy to All Subjects
 - Reproducibility manifests including app version, git hash, model version,
   feature version, config hash, timestamp


------------------------------------------------------------------------------
 PROJECT ARTIFACTS
------------------------------------------------------------------------------

A project folder collects everything ABEL produces:

 - Features:          derived/pose_features/, derived/context_features/,
                      derived/representations/
 - Models:            derived/models/<model_version>/
 - Review labels:     derived/review_labels/reviewer_labels.parquet
 - Candidate segments:derived/review_tables/candidate_segments.json
 - Evaluation:        derived/evaluation/ (model_metrics.json, PR_curve.png,
                      confusion_matrix.png, manifests)
 - Bouts:             derived/behavior_bouts/<behavior_id>_bouts.parquet


------------------------------------------------------------------------------
 BENCHMARKS
------------------------------------------------------------------------------

An ablation benchmark suite is included:

    abel-benchmark          (or: python -m abel.benchmark)

On Windows you can also double-click run_benchmark.bat (run run_abel.bat first
to set up the environment).


------------------------------------------------------------------------------
 TESTS
------------------------------------------------------------------------------

    pip install -e ".[dev]"
    pytest


------------------------------------------------------------------------------
 REPOSITORY LAYOUT
------------------------------------------------------------------------------

    abel/            Application package (UI, services, models, workers,
                     benchmark, ...)
    docs/            Architecture and structure notes
    scripts/         Developer utilities
    tests/           Test suite
    pyproject.toml   Packaging, entry points, optional dependency groups
    run_abel.bat     One-click Windows launcher


------------------------------------------------------------------------------
 NOTES
------------------------------------------------------------------------------

 - ABEL is local-first: source data and derived artifacts stay in the project
   folder.
 - Heavy dependencies are managed explicitly from the in-app Dependencies tab.


------------------------------------------------------------------------------
 LICENSE
------------------------------------------------------------------------------

Copyright (C) 2026 The University of North Carolina at Chapel Hill.
UNC Software ABEL (UNC Ref No 26-0187). All rights reserved.

ABEL is released for NON-COMMERCIAL USE ONLY under the UNC copyright and
permission notice - see the LICENSE file for full terms. Use, copying, and
redistribution (with or without modification) are permitted for non-commercial
purposes provided the copyright notice and conditions are retained. Any party
desiring a license to use the Software for commercial purposes must contact
the UNC Office of Technology Commercialization at 919-966-3929.

# ABEL

**ABEL — Active-learning Behavior Estimation and Labeling**

Version 0.12.1 · Python ≥ 3.10 · UNC Non-Commercial License

ABEL is a local-first desktop application for human-in-the-loop behavior modeling
from DLC-tracked videos. You label a few examples, it ranks the frames worth
labeling next, and you iterate until the model is good enough to run on new data.
Pose features are the default; video-derived features are optional for behaviors
that need visual context. Source data and derived artifacts stay in your project
folder.

---

## Installation

ABEL requires **Python 3.10 or newer**.

### Option A — one-click launcher (Windows)

Double-click **`run_abel.bat`**. It creates a virtual environment, installs the
app in editable mode, runs a PySide6 self-test, and launches the GUI. Re-run it
any time to update.

### Option B — manual install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e .
abel
```

### Optional dependency groups

Heavy or task-specific dependencies are opt-in and can also be managed from the
in-app **Dependencies** tab:

```bash
pip install -e ".[preprocessing]"   # video features: opencv, scipy, imageio, imageio-ffmpeg
pip install -e ".[gpu]"             # torch (GPU backend for R3D appearance features)
pip install -e ".[benchmarks]"      # xgboost
pip install -e ".[clustering]"      # scikit-learn, umap-learn, hdbscan: UMAP + motif clustering
pip install -e ".[dev]"             # pytest
pip install -e ".[all]"             # everything above
```

R3D appearance features download Kinetics-pretrained weights from
`download.pytorch.org` the first time they run, then cache them locally. That is
the only network access ABEL performs — no data leaves the machine.

---

## Core workflow

1. Import videos + pose files and auto-link sessions.
2. Define behaviors and add seed examples.
3. Configure pose features (window duration, stride, smoothing) and optionally
   enable video-derived features (optical flow, motion).
4. Run Active Learning: feature extraction, frame and segment representations,
   supervised training with context-padded label propagation, then uncertainty
   scoring and candidate ranking. (Outputs are listed under *Project artifacts*.)
5. Extract clips for selected segments.
6. Review and relabel segments.
7. Retrain and rerank candidates.
8. Validate models with the interactive Validation tab (blind quiz, reliability metrics, suggestions).
9. Export labels, reports, and bout-level assay outputs.

---

## Applying models to new projects

- **Direct Use** — replay a trained workflow on new videos without retraining. A
  *workflow snapshot* captures every behavior's model, window/stride,
  temporal-refinement thresholds, bout settings, and whether video features were
  used. Steps: source project → input data → pixel/mm calibration → keypoint
  mapping → run.
- **Keypoint mapping** — maps differently-named DLC keypoints (`back_mid` vs
  `center_body`) onto the names the model expects. Auto-filled and saved per
  source project; also available in **Data Import**.
- **Transfer Feedback** — scores how well the model transferred, per subject and
  across the population, worst-first, flagging near-zero detections, population
  outliers, stuck-high / lost-low confidence, and profile divergence.
- **Model Refinement** — import labeled examples from other projects and retrain.
  Keypoint names are reconciled automatically; incompatible schemas are blocked.
  Each source reports feature-value shift, pixel/mm calibration, DLC pose-model
  match, and extraction settings against the target project. Imports appear in
  Review as source-tagged entries and can be removed at any time.
- **Run Models** (Active Learning) — score a subset of behaviors with their
  existing models, no retraining.

---

## Modeling details

- Multi-behavior supervised modeling; target behavior selectable per run
- Pose-only or pose+video feature modes, selected at project setup
- Overlap-aware negative learning rule for behavior interactions
- Uncertainty components: entropy, ensemble variance, density outlier, optional margin
- R3D-18 appearance features (512-d per segment, per-animal centered crops)
  computed before training, toggleable on the **Features** tab
- Group-aware splitting by subject/session
- Reproducibility manifests recording app version, git hash, model version,
  feature version, config hash, and timestamp

The in-app **Methods** tab lists the formulas ABEL evaluates, each tagged with the
function that implements it, and the sources behind them.

---

## External validation suite

A separate meta-analysis suite (`abel/validation/`) compares models across
projects: leave-one-subject-out validation with subject-level bootstrap CIs,
learning curves, cross-project discrimination, rare-behavior discovery, and
Prism-ready exports. Launch it with `run_validation.bat`, or:

```bash
python -m abel.validation
```

Runs and saved setups are stored outside the repository, in the validation
workspace you choose on first launch.

---

## Project artifacts

A project folder collects everything ABEL produces:

- Features: `derived/pose_features/`, `derived/context_features/`, `derived/representations/`
- Models: `derived/models/<model_version>/`
- Review labels: `derived/review_labels/reviewer_labels.parquet`
- Candidate segments: `derived/review_tables/candidate_segments.json`
- Evaluation: `derived/evaluation/` (`model_metrics.json`, `PR_curve.png`, `confusion_matrix.png`, manifests)
- Bouts: `derived/behavior_bouts/<behavior_id>_bouts.parquet`
- Validation: `derived/validation/` (assembled tests, per-reviewer answers, extracted quiz clips)

---

## Benchmarks

An ablation benchmark suite is included:

```bash
abel-benchmark          # or: python -m abel.benchmark
```

On Windows you can also double-click `run_benchmark.bat` (run `run_abel.bat`
first to set up the environment).

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Repository layout

```text
abel/            Application package (UI, services, models, workers, benchmark, validation, …)
scripts/         Developer utilities
tests/           Test suite
pyproject.toml   Packaging, entry points, optional dependency groups
run_abel.bat     One-click Windows launcher
run_validation.bat   External validation suite launcher
```

---

## License

Copyright (C) 2026 The University of North Carolina at Chapel Hill. UNC Software
ABEL (UNC Ref No 26-0187). All rights reserved.

ABEL is released for **non-commercial use only** under the UNC copyright and
permission notice — see [LICENSE](LICENSE) for full terms. Use, copying, and
redistribution (with or without modification) are permitted for non-commercial
purposes provided the copyright notice and conditions are retained. Any party
desiring a license to use the Software for commercial purposes must contact the
UNC Office of Technology Commercialization at 919-966-3929.

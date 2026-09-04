# ABEL

**ABEL — Active-learning Behavior Estimation and Labeling**

Version 0.14.0 · Python ≥ 3.10 · UNC academic-use license (UNC Ref. No. 26-0187)

ABEL is a graphical user interface (GUI)-based, no-coding required platform for
human-in-the-loop annotation and training of predictive models for behavior
analysis. It adapts the frame-level active-learning approach introduced by
A-SOiD to a clip-level pipeline with a built-in annotation platform.
Temporally- and contextually-aware feature extraction adds pose, video, and
region of interest (ROI)-context features that pose alone does not provide,
which yields models with high precision on held-out subjects and improved
differentiation of like behaviors. ABEL takes the user from raw pose-tracked
data to refined exports in a single pipeline.

Alongside the supervised loop, ABEL includes a suite of data analysis,
visualization, and validation tools for group-based analyses, graphing,
statistics, and export of bout start and end frames for use with aligning fiber
photometry data.

---

## Citation

Ritchie JL, George BE, Roland AV, Krieman CG, Bender BN, Eberle MR, Der WC,
Kooyman LS, Lawes AM, Scott RT, O'Buckley TK, McLean MMR, Gallagher CJ, Besheer
J, Kash TL. *ABEL: an active-learning behavior estimation and labeling
platform.* bioRxiv 2026.08.30.748115.
[doi:10.64898/2026.08.30.748115](https://doi.org/10.64898/2026.08.30.748115)

The version used for the analyses reported there is v0.11.0. Across eight assays
and 45 behaviors, model training required 19.5 hours of human annotation in
total, with the reviewer scoring ~8% of available video. Models achieved a mean
precision-recall area under the curve (PR-AUC) of 0.90 (SD 0.09, range
0.60–0.99) and a mean Cohen's kappa of 0.79, with no detected association between
performance and behavior prevalence (r = 0.20).

---

## Hardware

ABEL is designed to be run locally on a Windows operating system. All computation
and analysis reported in the preprint was performed on an ASUS laptop with an
Intel Core i9-13980HX CPU, 64 GB physical memory, and an NVIDIA 4070 laptop GPU.
CPU fallback is available but will require considerably longer time for
computation.

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
pip install -e ".[gpu]"             # torch (CUDA backend for R3D-18 appearance features)
pip install -e ".[benchmarks]"      # xgboost
pip install -e ".[clustering]"      # scikit-learn, umap-learn, hdbscan: UMAP + motif clustering
pip install -e ".[dev]"             # pytest
pip install -e ".[all]"             # everything above
```

R3D-18 appearance features download Kinetics-pretrained weights from
`download.pytorch.org` on first use and cache them locally. That is the only
network access ABEL performs; video, pose, and derived data stay in the project
folder.

---

## Input data

- **Video** in .mp4 or .avi format.
- **Keypoint tracking** from either DeepLabCut or SLEAP, in .csv or .h5 format.
- **ROI zones** defined within ABEL's interactive ROI tool. Any number of zones
  can be defined for context-level feature extraction, in a session-specific or
  project-wide manner, and time in ROI is provided as a basic endpoint.

---

## Pipeline

**Segmentation and feature extraction.** Raw data are segmented into user-defined
clip lengths, depending on the length of the behaviors of interest — the reported
models used single fixed-scale windowing of 0.5 s with a stride of either 0.1 or
0.5 s. Pose data are filtered to a minimum confidence threshold of 0.2, gaps of
10 frames or less are filled using linear interpolation, then a 3-frame rolling
average is applied to remove jitter. Features are extracted across video, pose,
and context/ROI domains. Feature count scales with the number of body parts
tracked and the ROI count (1952–2374 across the reported projects, ~2080 on
average, of which 512 are R3D-18 appearance embeddings).

**Seeding.** Three modes: dispersed pseudorandom labeling, UMAP Interactive
Selection, and Essence Extractor. Pseudorandom clip generation provides clips
dispersed across subjects and sessions through seeded random number generation,
with an optional per-session cap.

**Annotation.** Clips are annotated in ABEL's built-in annotation suite.
Experienced users annotate in as little as 1–5 seconds per clip; timestamps from
the review queue across projects show a median of 1 s per clip decision (43,431
decisions). Scored clips can be re-reviewed at any point.

**Classifier.** One binary XGBoost model per behavior (one-vs-rest) with
Platt-scaled calibration, trained on CUDA with a CPU fallback. Base
hyperparameters: `tree_method "hist"`, `max_depth 6`, learning rate 0.1,
`n_estimators 300`, `subsample 0.8`, `colsample_bytree 0.8`, `reg_alpha 0.1`,
`reg_lambda 1.0`, `min_child_weight 5`, `random_state 42`. Adaptive complexity is
disabled by default; when enabled it scales `n_estimators` and `max_depth` by the
ratio of positive clips to features. R3D-18 appearance embeddings enter the
XGBoost feature matrix as ordinary numeric features. Positive augmentation is on
by default: Gaussian jitter at σ = 0.05 of each feature's standard deviation, 10%
feature dropout, and 3 synthetic copies per positive clip.

**Active learning.** Candidates are scored by a single blended priority. Each
unlabeled clip receives a rank score combining its behavior-aware uncertainty,
predicted probability for the target behavior (weight 0.5), reviewer feedback
(0.2), and exclusivity uncertainty (0.2). Clips previously reviewed as negatives
that score above 0.6 receive an additional +0.3 hard-negative bonus, surfacing
most confident errors. No automatic stopping criterion is set. Batch size is
defaulted to 20 but is adjusted per project.

**Temporal refinement.** Once satisfactory models have been generated, they are
applied to a newly segmented dataset with a high degree of clip overlap (default
0.1 s stride). Per-behavior frame-level confidence traces are generated by
finding the per-frame average of the densely overlapping clip-level predictions.
Mutual inhibition can then be applied to specific behavior pairs or with a global
setting (default 0.20) to aid in differentiating confounding behaviors. Next,
per-behavior minimum confidence and bout length thresholds are set to establish
positive bouts for subsequent analysis. Default settings: onset threshold 0.50,
minimum bout duration 6 frames, merge gap 3 frames, smoothing window 5, and a
1.5 s warmup (excludes unreliable frames at start of estimation). Tools for
flagging false positives and false negatives are available for further
refinement.

Clip-level metrics reported before dense temporal refinement should be taken as a
conservative lower bound; the bouts that ABEL exports are further refined by
smoothing, thresholding, and bout duration filtering.

---

## Clip discovery tools

Two tools surface positive clips during early model refinement and for
low-abundance behaviors.

**Essence Extractor** (within ABEL's Targeted Clip Mining dialog) surveys a
subset of user selected clips for shared features which differ from the total
clip population. It first surfaces the features that separate the exemplar clips
from the unlabeled population using a rank-based statistical equivalent to
|AUC − 0.5|. Features falling below 0.10 are discarded, and remaining candidate
features are filtered to a candidate set (default: 5 criterion features, 8
ranking features); the displayed criterion features summarize what separates the
exemplar clips. Clip ranking is performed by a sparse linear model — features are
ranked by |AUC − 0.5|, the top 300 are retained, then an L1-penalized logistic
regression (liblinear, C = 0.1, class balanced, on median-imputed and
standardized features) is fit to discriminate the exemplar clips from a
background sample of 1500 unlabeled clips.

**UMAP Interactive Selection** allows for manual selection based on proximity in
a reduced-dimension feature space. The UMAP is fit on labeled and predicted
points with default settings of 2 components, an adaptive neighborhood size of
max(5, min(25, √n)), a minimum distance of 0.15, and a default to principal
component analysis when a UMAP cannot be fit.

The relative value of selection tool choice depends on clip budget. At a budget
of 25 clips, Essence Extractor surfaced 1.55x more positives than active-learning
and 1.89x more than UMAP Interactive Selection, making it the most effective tool
during early model refinement. At a budget of 300 clips, active-learning
overtakes Essence Extractor (1.36x) and UMAP Interactive Selection (1.71x). Both
targeted strategies extracted ~5–6x more positives than random selection at every
tested clip budget. These tools are meant to function as supplements to the
already efficient active-learning approach, not replacements for it.

---

## Unsupervised tools

Supervised approaches cannot detect novel behaviors not anticipated by the user.
ABEL therefore provides built-in tools for unsupervised analysis in parallel to
the supervised workflow — an unsupervised UMAP Selection interface, and
transition-probability and motif-discovery analytics — which surface
unanticipated candidates that can then be named and analyzed.

---

## Applying models to new projects

- **Direct Use** — replay a trained workflow on new videos without retraining. A
  *workflow snapshot* captures every behavior's model, window/stride,
  temporal-refinement thresholds, bout settings, and whether video features were
  used. Steps: source project → input data → pixel/mm calibration → keypoint
  mapping → run.
- **Keypoint mapping** — maps differently-named keypoints (`back_mid` vs
  `center_body`) onto the names the model expects. Auto-filled and saved per
  source project; also available in **Data Import**.
- **Transfer Feedback** — scores how well the model transferred, per subject and
  across the population, worst-first, flagging near-zero detections, population
  outliers, stuck-high / lost-low confidence, and profile divergence.
- **Model Refinement** — import labeled examples from other projects and retrain.
  Keypoint names are reconciled automatically; incompatible schemas are blocked.
  Each source reports feature-value shift, pixel/mm calibration, pose-model match,
  and extraction settings against the target project. Imports appear in Review as
  source-tagged entries and can be removed at any time.
- **Run Models** (Active Learning) — score a subset of behaviors with their
  existing models, no retraining.

---

## Photometry alignment

Bout start and end frames are exported for alignment with fiber photometry
recordings. Processing and alignment on the photometry side is handled by TRACY
(https://github.com/JobeRitchie/TRACY-Photometry-Processing-Suite), which reads
ABEL's bout frame exports directly. Per-session `<Subject>ABELposition.csv` files
can also be written for import into TRACY.

---

## Validation

The in-app **Validation** tab supports blind quizzes against held-out clips,
reliability metrics, and leave-one-subject-out evaluation.

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

## Methods reporting

The in-app **Methods** tab lists the formulas ABEL evaluates, each tagged with the
function that implements it, and the sources behind them. Its **Write-up Helper**
subtab drafts a methods section from the open project's own settings and results
— model metrics, behaviors, window sizes, bout thresholds, HMM settings — with a
[FILL IN: ...] placeholder wherever ABEL cannot know the answer. The draft is a
guide to edit and verify, not text to paste into a manuscript.

Runs record reproducibility manifests with app version, git hash, model version,
feature version, config hash, and timestamp.

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

ABEL is distributed under a University of North Carolina academic-use license and
is free for academic or non-profit use — see [LICENSE](LICENSE) for full terms.
Use, copying, and redistribution (with or without modification) are permitted for
non-commercial purposes provided the copyright notice and conditions are
retained. Any party desiring a license to use the Software for commercial
purposes must contact the UNC Office of Technology Commercialization at
919-966-3929.

"""Publication-grade Methods documentation: references and formulas.

This module is the single source of truth behind the **Methods** tab. Keeping the
content as structured data (rather than inline HTML in the widget) makes it testable
and lets the same content be exported later. Tests assert that every reference URL is
well formed and that every formula's ``source`` resolves to a real importable attribute.

Two public tables:

* :data:`REFERENCES`, the sources behind the non-trivial procedures ABEL performs,
  plus the pose formats it consumes (DeepLabCut, SLEAP), grouped by topic. Most back
  a specific formula; a few are cited for the input format or library ABEL uses.
* :data:`FORMULAS`, the raw formulas ABEL evaluates, each tagged with the source
  function so a reviewer can trace equation → code.

Add a formula only when the code actually implements it: a formula here is a claim
about what ABEL computes, and a reviewer will read it as one.

Formulas render as HTML + Unicode (the app has no LaTeX engine); helper glyphs live
in this module so the markup stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """A citation and the ABEL procedure(s) it justifies."""

    key: str
    authors: str
    year: str
    title: str
    venue: str
    url: str  # DOI or stable URL (must be non-empty)
    used_for: str  # which ABEL procedure this reference backs


@dataclass(frozen=True)
class Formula:
    """A raw formula ABEL evaluates, tied to its implementing function."""

    name: str
    category: str
    formula_html: str  # HTML + Unicode
    description: str
    source: str  # ABEL module.function that implements it
    refs: tuple[str, ...] = field(default_factory=tuple)  # Reference.key values


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
# Grouped by topic; group order preserved in the rendered output.
REFERENCES: list[Reference] = [
    # ---- Cross-validation / LOMO ----
    Reference(
        "pedregosa2011", "Pedregosa, F., Varoquaux, G., Gramfort, A., et al.", "2011",
        "Scikit-learn: Machine Learning in Python",
        "Journal of Machine Learning Research, 12, 2825–2830",
        "https://jmlr.org/papers/v12/pedregosa11a.html",
        "All classification metrics, cross-validation splitters (LeaveOneGroupOut / "
        "GroupKFold), calibration, and estimators.",
    ),
    Reference(
        "saeb2017", "Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P.", "2017",
        "The need to approximate the use-case in clinical machine learning",
        "GigaScience, 6(5), 1–9",
        "https://doi.org/10.1093/gigascience/gix019",
        "Justifies subject-wise (leave-one-mouse-out) cross-validation over "
        "record-wise CV, which overestimates accuracy when subjects contribute "
        "many correlated samples.",
    ),
    Reference(
        "varoquaux2018", "Varoquaux, G.", "2018",
        "Cross-validation failure: Small sample sizes lead to large error bars",
        "NeuroImage, 180, 68–77",
        "https://doi.org/10.1016/j.neuroimage.2017.06.061",
        "Reporting per-fold variability (mean ± SEM) and cautioning that small "
        "subject counts yield wide error bars.",
    ),
    # ---- Evaluation metrics ----
    Reference(
        "saito2015", "Saito, T., & Rehmsmeier, M.", "2015",
        "The precision-recall plot is more informative than the ROC plot when "
        "evaluating binary classifiers on imbalanced datasets",
        "PLOS ONE, 10(3), e0118432",
        "https://doi.org/10.1371/journal.pone.0118432",
        "Choosing PR-AUC (average precision) as the primary metric for imbalanced "
        "behavior data instead of ROC-AUC.",
    ),
    Reference(
        "chicco2020", "Chicco, D., & Jurman, G.", "2020",
        "The advantages of the Matthews correlation coefficient (MCC) over F1 score "
        "and accuracy in binary classification evaluation",
        "BMC Genomics, 21, 6",
        "https://doi.org/10.1186/s12864-019-6413-7",
        "Rationale for why confusion-matrix-balanced metrics (and reporting TP/FP/FN "
        "counts, not accuracy alone) matter under class imbalance.",
    ),
    # ---- Rater agreement ----
    Reference(
        "cohen1960", "Cohen, J.", "1960",
        "A coefficient of agreement for nominal scales",
        "Educational and Psychological Measurement, 20(1), 37–46",
        "https://doi.org/10.1177/001316446002000104",
        "Cohen's κ for two-rater (user-vs-reference) agreement in the validation quiz.",
    ),
    Reference(
        "fleiss1971", "Fleiss, J. L.", "1971",
        "Measuring nominal scale agreement among many raters",
        "Psychological Bulletin, 76(5), 378–382",
        "https://doi.org/10.1037/h0031619",
        "Fleiss' κ for agreement among more than two raters.",
    ),
    Reference(
        "landis1977", "Landis, J. R., & Koch, G. G.", "1977",
        "The measurement of observer agreement for categorical data",
        "Biometrics, 33(1), 159–174",
        "https://doi.org/10.2307/2529310",
        "Interpretation benchmarks for κ agreement magnitudes.",
    ),
    Reference(
        "benjamini1995", "Benjamini, Y., & Hochberg, Y.", "1995",
        "Controlling the false discovery rate: A practical and powerful approach to "
        "multiple testing",
        "Journal of the Royal Statistical Society: Series B, 57(1), 289–300",
        "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
        "Benjamini–Hochberg FDR correction for transition/motif significance tests.",
    ),
    Reference(
        "good2005", "Good, P.", "2005",
        "Permutation, Parametric, and Bootstrap Tests of Hypotheses (3rd ed.)",
        "Springer, New York",
        "https://doi.org/10.1007/b138696",
        "Label-shuffle permutation tests for behavioral-motif differences.",
    ),
    # ---- Machine-learning platforms & models ----
    Reference(
        "nilsson2020", "Nilsson, S. R. O., Goodwin, N. L., Choong, J. J., et al.", "2020",
        "Simple Behavioral Analysis (SimBA) - an open source toolkit for computer "
        "classification of complex social behaviors in experimental animals",
        "bioRxiv, 2020.04.19.049452",
        "https://doi.org/10.1101/2020.04.19.049452",
        "Original SimBA description; source of the pose-feature vocabulary ABEL "
        "extends (inter-keypoint distances, body-length normalization, rolling-window "
        "statistics) and of its social / dyadic feature conventions.",
    ),
    Reference(
        "mathis2018", "Mathis, A., Mamidanna, P., Cury, K. M., et al.", "2018",
        "DeepLabCut: markerless pose estimation of user-defined body parts with deep "
        "learning",
        "Nature Neuroscience, 21, 1281-1289",
        "https://doi.org/10.1038/s41593-018-0209-y",
        "Source of the pose tracking ABEL consumes; DeepLabCut CSV/H5 is ABEL's native "
        "pose input format.",
    ),
    Reference(
        "pereira2022", "Pereira, T. D., Tabris, N., Matsliah, A., et al.", "2022",
        "SLEAP: A deep learning system for multi-animal pose tracking",
        "Nature Methods, 19, 486-495",
        "https://doi.org/10.1038/s41592-022-01426-1",
        "Multi-animal pose tracking; SLEAP predictions are converted to the "
        "DeepLabCut layout on import.",
    ),
    Reference(
        "tran2018", "Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M.", "2018",
        "A closer look at spatiotemporal convolutions for action recognition",
        "Proceedings of CVPR 2018, 6450-6459",
        "https://doi.org/10.1109/CVPR.2018.00675",
        "The 3D ResNet-18 architecture behind ABEL's video appearance features "
        "(torchvision r3d_18).",
    ),
    Reference(
        "kay2017", "Kay, W., Carreira, J., Simonyan, K., et al.", "2017",
        "The Kinetics human action video dataset",
        "arXiv:1705.06950",
        "https://arxiv.org/abs/1705.06950",
        "Pretraining corpus for the R3D-18 backbone whose penultimate activations ABEL "
        "uses as transferable video features.",
    ),
    Reference(
        "chen2016", "Chen, T., & Guestrin, C.", "2016",
        "XGBoost: A scalable tree boosting system",
        "Proceedings of KDD 2016, 785-794",
        "https://doi.org/10.1145/2939672.2939785",
        "Gradient-boosted-tree classifier used for the segment and center-frame models.",
    ),
    Reference(
        "ke2017", "Ke, G., Meng, Q., Finley, T., et al.", "2017",
        "LightGBM: A highly efficient gradient boosting decision tree",
        "Advances in Neural Information Processing Systems, 30, 3146-3154",
        "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
        "Alternative gradient-boosting backend (scikit-learn histogram gradient "
        "boosting is the always-available default).",
    ),
    # ---- Temporal model / calibration / active learning ----
    Reference(
        "rabiner1989", "Rabiner, L. R.", "1989",
        "A tutorial on hidden Markov models and selected applications in speech recognition",
        "Proceedings of the IEEE, 77(2), 257–286",
        "https://doi.org/10.1109/5.18626",
        "The Gaussian hidden Markov model (via hmmlearn) used for dominance-state "
        "estimation in social analysis.",
    ),
    Reference(
        "platt1999", "Platt, J. C.", "1999",
        "Probabilistic outputs for support vector machines and comparisons to "
        "regularized likelihood methods",
        "Advances in Large Margin Classifiers, 61–74 (MIT Press)",
        "https://www.researchgate.net/publication/2594015",
        "Platt (sigmoid) probability calibration of classifier scores.",
    ),
    Reference(
        "zadrozny2002", "Zadrozny, B., & Elkan, C.", "2002",
        "Transforming classifier scores into accurate multiclass probability estimates",
        "Proceedings of KDD 2002, 694–699",
        "https://doi.org/10.1145/775047.775151",
        "Isotonic-regression probability calibration.",
    ),
    Reference(
        "settles2009", "Settles, B.", "2009",
        "Active Learning Literature Survey",
        "University of Wisconsin–Madison, Computer Sciences Technical Report 1648",
        "https://minds.wisconsin.edu/handle/1793/60660",
        "Uncertainty-sampling acquisition (entropy, margin, ensemble disagreement, "
        "density) for active-learning candidate ranking.",
    ),
    Reference(
        "shannon1948", "Shannon, C. E.", "1948",
        "A mathematical theory of communication",
        "Bell System Technical Journal, 27(3), 379–423",
        "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x",
        "Shannon entropy as an uncertainty measure and for ROI occupancy features.",
    ),
    Reference(
        "mcinnes2018", "McInnes, L., Healy, J., & Melville, J.", "2018",
        "UMAP: Uniform Manifold Approximation and Projection for dimension reduction",
        "arXiv:1802.03426",
        "https://arxiv.org/abs/1802.03426",
        "UMAP embedding used in the analytics / behavior-space visualizations.",
    ),
]


# ---------------------------------------------------------------------------
# Formulas  (HTML + Unicode; no LaTeX engine in the app)
# ---------------------------------------------------------------------------
FORMULAS: list[Formula] = [
    # ---- Classification metrics ----
    Formula(
        "Macro-F1", "Classification metrics",
        "F1<sub>macro</sub> = (1/K) · Σ<sub>k</sub> F1<sub>k</sub>",
        "Unweighted mean of per-class F1 across the K classes (used by the trainer "
        "and refined scorer).",
        "abel.temporal_refinement.refined_eval._macro_prf",
        ("pedregosa2011",),
    ),
    Formula(
        "Average Precision (PR-AUC)", "Classification metrics",
        "AP = Σ<sub>n</sub> (R<sub>n</sub> − R<sub>n−1</sub>) · P<sub>n</sub>",
        "Area under the precision–recall curve as a step-wise sum over thresholds; "
        "the primary threshold-free metric for imbalanced behavior data.",
        "abel.services.evaluation_service.EvaluationService.segment_metrics",
        ("saito2015", "pedregosa2011"),
    ),
    Formula(
        "Matthews correlation coefficient", "Classification metrics",
        "MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
        "Chance-corrected correlation between prediction and truth. A high value "
        "requires doing well on both classes at once, so it cannot be gamed by the "
        "majority class; MCC ≤ 0 is ABEL's degeneracy gate for a held-out result.",
        "abel.validation.metrics.matthews_corrcoef",
        ("chicco2020", "pedregosa2011"),
    ),
    Formula(
        "Cohen's κ", "Rater agreement",
        "κ = (p<sub>o</sub> − p<sub>e</sub>) / (1 − p<sub>e</sub>)",
        "Chance-corrected agreement between two raters; p<sub>o</sub> observed, "
        "p<sub>e</sub> expected-by-chance agreement.",
        "abel.services.validation_service.ValidationService.compute_metrics",
        ("cohen1960", "landis1977"),
    ),
    Formula(
        "Fleiss' κ", "Rater agreement",
        "κ = (P&#772; − P&#772;<sub>e</sub>) / (1 − P&#772;<sub>e</sub>)",
        "Agreement among more than two raters; P&#772; mean per-item agreement, "
        "P&#772;<sub>e</sub> = Σ<sub>j</sub> p<sub>j</sub>² expected agreement.",
        "abel.services.validation_service.ValidationService._kappa",
        ("fleiss1971",),
    ),
    # ---- Cross-validation ----
    Formula(
        "Leave-one-mouse-out CV", "Cross-validation",
        "for each subject s:  train on {all mice ≠ s},  test on {mouse s}",
        "Leave-One-Group-Out cross-validation: the subject-grouped special case of "
        "k-fold with k = number of mice: so no mouse appears in both train and test.",
        "abel.validation.loso.leave_one_subject_out",
        ("saeb2017", "pedregosa2011", "varoquaux2018"),
    ),
    # ---- Temporal refinement ----
    Formula(
        "Moving-average smoothing", "Temporal refinement",
        "p&#771;<sub>t</sub> = (1/w) Σ<sub>i=t−w/2</sub><sup>t+w/2</sup> p<sub>i</sub>",
        "Box-filter smoothing of the per-window probability trace over a window w.",
        "abel.temporal_refinement.bout_postprocess.smooth_probabilities",
    ),
    Formula(
        "Temporal IoU (bout matching)", "Temporal refinement",
        "IoU(A,B) = |A ∩ B| / |A ∪ B|",
        "Overlap between a predicted bout and a labeled bout; a match requires "
        "IoU ≥ 0.2 for event-level TP/FP/FN.",
        "abel.temporal_refinement.refined_eval._bout_iou",
    ),
    Formula(
        "Mutual-inhibition competition", "Temporal refinement",
        "p&#771;<sub>i</sub> = clip( p<sub>i</sub> − λ · Σ<sub>j≠i</sub> p<sub>j</sub>, 0, 1 );&nbsp; "
        "label = argmax<sub>i</sub> p&#771;<sub>i</sub>",
        "Each behavior's probability is suppressed by the summed activity of the "
        "others (weight λ), then the winning behavior per frame is the argmax.",
        "abel.temporal_refinement.temporal_refinement_service",
    ),
    # ---- Feature engineering ----
    Formula(
        "Finite-difference kinematics", "Feature engineering",
        "v<sub>t</sub> = (x<sub>t</sub> − x<sub>t−1</sub>) · fps;&nbsp; "
        "speed = &radic;(v<sub>x</sub>² + v<sub>y</sub>²);&nbsp; "
        "a = Δv · fps;&nbsp; jerk = Δa · fps",
        "Velocity, speed, acceleration and jerk from per-frame keypoint differences "
        "scaled to units per second.",
        "abel.services.pose_processing_service.PoseProcessingService.compute_frame_pose_features",
    ),
    Formula(
        "Joint angle (three keypoints)", "Feature engineering",
        "θ = arccos( (BA · BC) / (|BA| · |BC|) )",
        "Interior angle at joint B from the dot product of the two limb vectors "
        "(e.g. spine flexion).",
        "abel.services.pose_processing_service",
    ),
    Formula(
        "Body-length self-normalization", "Feature engineering",
        "d&#770;<sub>ij</sub> = d<sub>ij</sub> / L<sub>body</sub>,&nbsp; "
        "d<sub>ij</sub> = &radic;((x<sub>i</sub>−x<sub>j</sub>)² + (y<sub>i</sub>−y<sub>j</sub>)²)",
        "Inter-keypoint distances divided by nose-to-tail body length, making "
        "features scale-invariant across animals and cameras.",
        "abel.services.pose_processing_service",
        ("nilsson2020",),
    ),
    Formula(
        "Grouped z-score", "Feature engineering",
        "z = (x − μ<sub>g</sub>) / σ<sub>g</sub>",
        "Per-group standardization (σ→1 for constant/singleton groups) of segment "
        "features before modeling.",
        "abel.services.behavior_representation_service",
    ),
    Formula(
        "Dominant frequency (FFT)", "Feature engineering",
        "f* = argmax<sub>f&gt;0</sub> |FFT(x)(f)|",
        "Peak non-DC frequency of a keypoint signal for rhythmic behaviors "
        "(e.g. scratching, digging).",
        "abel.services.pose_processing_service",
    ),
    # ---- Learned models ----
    Formula(
        "Gradient-boosted trees", "Learned models",
        "F<sub>M</sub>(x) = &Sigma;<sub>m=1</sub><sup>M</sup> &eta; · f<sub>m</sub>(x);&nbsp; "
        "obj = &Sigma;<sub>i</sub> l(y<sub>i</sub>, F(x<sub>i</sub>)) + "
        "&Sigma;<sub>m</sub> &Omega;(f<sub>m</sub>)",
        "The segment and center-frame classifiers: an additive ensemble of M regression "
        "trees fit stage-wise to the loss gradient, with shrinkage &eta; and a complexity "
        "penalty &Omega;. Backends are XGBoost, LightGBM, or scikit-learn histogram "
        "gradient boosting.",
        "abel.services.active_learning_trainer_service",
        ("chen2016", "ke2017", "pedregosa2011"),
    ),
    Formula(
        "R3D-18 video embedding", "Learned models",
        "z = GAP( f<sub>&theta;</sub>(C&times;T&times;H&times;W clip) ) &isin; ℝ<sup>512</sup>",
        "Appearance features: a Kinetics-pretrained 3D ResNet-18 is run over the cropped "
        "clip around each window and its global-average-pooled penultimate activations "
        "(512 dims) are concatenated with the pose features before training.",
        "abel.services.r3d_feature_service",
        ("tran2018", "kay2017"),
    ),
    # ---- Statistical tests ----
    Formula(
        "Benjamini–Hochberg FDR", "Statistical tests",
        "q<sub>(k)</sub> = min<sub>k′≥k</sub> [ p<sub>(k′)</sub> · m / k′ ]",
        "False-discovery-rate adjusted p-values for motif/transition tests.",
        "abel.ui.tabs.behavior_analytics_tab._BehaviorMotifWidget._fdr_bh_adjust",
        ("benjamini1995",),
    ),
    Formula(
        "Permutation p-value", "Statistical tests",
        "p = ( #{ |Δ<sub>perm</sub>| ≥ |Δ<sub>obs</sub>| } ) / N<sub>perm</sub>",
        "Label-shuffle null distribution (default N=1000) for motif differences.",
        "abel.services.behavioral_motif_service",
        ("good2005",),
    ),
    # ---- Active learning ----
    Formula(
        "Shannon entropy (uncertainty)", "Active learning",
        "H(p) = − Σ<sub>k</sub> p<sub>k</sub> · ln p<sub>k</sub>",
        "Predictive uncertainty of a segment's class distribution (in nats).",
        "abel.services.uncertainty_service.UncertaintyScoringService.entropy",
        ("shannon1948", "settles2009"),
    ),
    Formula(
        "Classification margin", "Active learning",
        "margin = |p<sub>(1)</sub> − p<sub>(2)</sub>|",
        "Gap between the top-two class probabilities; small margins flag ambiguous "
        "segments (least-confidence = 1 − margin).",
        "abel.services.uncertainty_service.UncertaintyScoringService.classification_margin",
        ("settles2009",),
    ),
    Formula(
        "Composite acquisition score", "Active learning",
        "raw = w<sub>H</sub>·H + w<sub>var</sub>·Var + w<sub>ρ</sub>·ρ + "
        "w<sub>m</sub>·(1−margin);&nbsp; score = (raw − min) / (max − min)",
        "Min-max–normalized weighted blend of entropy, ensemble variance, k-NN "
        "density outlier score, and margin for ranking review candidates.",
        "abel.services.uncertainty_service",
        ("settles2009",),
    ),
    # ---- Calibration & HMM ----
    Formula(
        "Platt (sigmoid) calibration", "Calibration & models",
        "P(y=1 | s) = 1 / (1 + exp(A·s + B))",
        "Logistic mapping of a classifier score s to a calibrated probability; "
        "A, B fit on the validation split (isotonic regression is the alternative).",
        "abel.services.active_learning_trainer_service",
        ("platt1999", "zadrozny2002"),
    ),
    Formula(
        "Gaussian HMM log-likelihood", "Calibration & models",
        "log P(O | λ) via forward algorithm;&nbsp; "
        "occupancy<sub>k</sub> = (#frames in state k) / T",
        "Diagonal-covariance Gaussian HMM (EM-fit) over standardized social features; "
        "state occupancy summarizes dominance dynamics.",
        "abel.services.social_analysis_service",
        ("rabiner1989",),
    ),
    Formula(
        "Motif HMM state occupancy", "Calibration & models",
        "Viterbi:&nbsp; occ<sub>k</sub> = |{t : z*<sub>t</sub> = k}| / T,&nbsp; "
        "z* = argmax<sub>z</sub> P(z | O, λ)"
        "<br>Posterior:&nbsp; occ<sub>k</sub> = (1/T) Σ<sub>t</sub> "
        "γ<sub>t</sub>(k),&nbsp; γ<sub>t</sub>(k) = P(z<sub>t</sub> = k | O, λ)",
        "Fractional occupancy of each hidden state of the categorical bout-sequence "
        "HMM, where t indexes bouts rather than frames. The Viterbi form assigns "
        "every bout to one state; the posterior (forward-backward) form reports "
        "expected occupancy and is preferred when a state's defining behavior is "
        "rare, because the maximum-likelihood path will not pay the transition cost "
        "twice to enter a state for one or two bouts and so reports exactly zero for "
        "subjects that did perform the behavior.",
        "abel.services.behavioral_motif_service",
        ("rabiner1989",),
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_WRAP_OPEN = (
    "<div style='font-family: Segoe UI, sans-serif; color:#CFD8DC; font-size:13px;'>"
)
_WRAP_CLOSE = "</div>"
_H2 = "font-size:15px; font-weight:800; color:#90CAF9; margin-top:16px;"


def _ordered_groups(items: list, key: str) -> list[str]:
    """Distinct group labels in first-seen order."""
    seen: list[str] = []
    for it in items:
        g = getattr(it, key)
        if g not in seen:
            seen.append(g)
    return seen


def _reference_group(ref: Reference) -> str:
    """Map a reference to its display section (mirrors the authored ordering)."""
    return {
        "nilsson2020": "Machine-learning platforms & models",
        "mathis2018": "Machine-learning platforms & models",
        "pereira2022": "Machine-learning platforms & models",
        "tran2018": "Machine-learning platforms & models",
        "kay2017": "Machine-learning platforms & models",
        "chen2016": "Machine-learning platforms & models",
        "ke2017": "Machine-learning platforms & models",
        "pedregosa2011": "Cross-validation & study design",
        "saeb2017": "Cross-validation & study design",
        "varoquaux2018": "Cross-validation & study design",
        "saito2015": "Evaluation metrics",
        "chicco2020": "Evaluation metrics",
        "cohen1960": "Rater agreement",
        "fleiss1971": "Rater agreement",
        "landis1977": "Rater agreement",
        "benjamini1995": "Statistical tests",
        "good2005": "Statistical tests",
        "rabiner1989": "Models, calibration & active learning",
        "platt1999": "Models, calibration & active learning",
        "zadrozny2002": "Models, calibration & active learning",
        "settles2009": "Models, calibration & active learning",
        "shannon1948": "Models, calibration & active learning",
        "mcinnes2018": "Models, calibration & active learning",
    }.get(ref.key, "Other")


_REFERENCE_SECTIONS = [
    "Machine-learning platforms & models",
    "Cross-validation & study design",
    "Evaluation metrics",
    "Rater agreement",
    "Statistical tests",
    "Models, calibration & active learning",
]


def render_references_html() -> str:
    """Build the References subtab HTML (sectioned, with clickable links)."""
    parts = [
        _WRAP_OPEN,
        "<p style='color:#90A4AE;'>Sources for the non-trivial procedures ABEL "
        "performs and for the machine-learning platforms and models it builds on. "
        "Textbook statistics (t-tests, ANOVA, precision/recall) are used but not "
        "cited. Each entry notes which ABEL analysis it supports; links open the DOI "
        "or archival page.</p>",
    ]
    for section in _REFERENCE_SECTIONS:
        refs = [r for r in REFERENCES if _reference_group(r) == section]
        if not refs:
            continue
        parts.append(f"<div style='{_H2}'>{escape(section)}</div>")
        parts.append("<ul style='margin-top:4px;'>")
        for r in refs:
            parts.append(
                "<li style='margin-bottom:8px;'>"
                f"{escape(r.authors)} ({escape(r.year)}). "
                f"<a href='{escape(r.url, quote=True)}' style='color:#80D8FF;'>"
                f"{escape(r.title)}</a>. "
                f"<i>{escape(r.venue)}</i>."
                f"<br><span style='color:#90A4AE;'>Used for: {escape(r.used_for)}</span>"
                "</li>"
            )
        parts.append("</ul>")
    parts.append(_WRAP_CLOSE)
    return "".join(parts)


def render_formulas_html() -> str:
    """Build the Formulas subtab HTML (grouped by category)."""
    parts = [
        _WRAP_OPEN,
        "<p style='color:#90A4AE;'>The non-obvious formulas ABEL evaluates across its "
        "primary analyses; standard textbook definitions are omitted. Each is tagged "
        "with the implementing function so equations can be traced directly to the "
        "code.</p>",
    ]
    by_key = {r.key: r for r in REFERENCES}
    for category in _ordered_groups(FORMULAS, "category"):
        parts.append(f"<div style='{_H2}'>{escape(category)}</div>")
        for f in [f for f in FORMULAS if f.category == category]:
            cite = ""
            if f.refs:
                names = []
                for k in f.refs:
                    ref = by_key.get(k)
                    if ref is not None:
                        first = ref.authors.split(",")[0]
                        names.append(f"{first} {ref.year}")
                if names:
                    cite = (
                        "<span style='color:#78909C;'>, "
                        + escape("; ".join(names))
                        + "</span>"
                    )
            parts.append(
                "<div style='margin:6px 0 12px 0;'>"
                f"<div style='font-weight:700; color:#ECEFF1;'>{escape(f.name)}{cite}</div>"
                f"<div style='font-family: Consolas, monospace; color:#B2FF59; "
                f"margin:3px 0; font-size:14px;'>{f.formula_html}</div>"
                f"<div style='color:#B0BEC5;'>{escape(f.description)}</div>"
                f"<div style='color:#607D8B; font-size:11px;'>Source: "
                f"<code>{escape(f.source)}</code></div>"
                "</div>"
            )
    parts.append(_WRAP_CLOSE)
    return "".join(parts)

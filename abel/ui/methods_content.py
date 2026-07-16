"""Publication-grade Methods documentation: references and formulas.

This module is the single source of truth behind the **Methods** tab. Keeping the
content as structured data (rather than inline HTML in the widget) makes it testable
— tests assert every reference has a resolvable link and every formula names the
ABEL function that implements it — and lets the same content be exported later.

Two public tables:

* :data:`REFERENCES` — the peer-reviewed / canonical sources that justify each
  statistical procedure ABEL performs, grouped by topic.
* :data:`FORMULAS` — the raw formulas ABEL evaluates, each tagged with the source
  function so a reviewer can trace equation → code.

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
    Reference(
        "stone1974", "Stone, M.", "1974",
        "Cross-validatory choice and assessment of statistical predictions",
        "Journal of the Royal Statistical Society: Series B, 36(2), 111–147",
        "https://doi.org/10.1111/j.2517-6161.1974.tb00994.x",
        "Foundational definition of cross-validation.",
    ),
    Reference(
        "kohavi1995", "Kohavi, R.", "1995",
        "A study of cross-validation and bootstrap for accuracy estimation and model selection",
        "Proceedings of IJCAI 1995, 1137–1143",
        "https://dl.acm.org/doi/10.5555/1643031.1643047",
        "Empirical basis for k-fold cross-validation as a generalization estimator.",
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
        "powers2011", "Powers, D. M. W.", "2011",
        "Evaluation: From precision, recall and F-measure to ROC, informedness, "
        "markedness and correlation",
        "Journal of Machine Learning Technologies, 2(1), 37–63",
        "https://arxiv.org/abs/2010.16061",
        "Definitions of precision, recall, and F-measure.",
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
    # ---- Statistical tests (Behavior Analytics) ----
    Reference(
        "student1908", "Student (Gosset, W. S.)", "1908",
        "The probable error of a mean",
        "Biometrika, 6(1), 1–25",
        "https://doi.org/10.1093/biomet/6.1.1",
        "The t-test for two-group comparisons.",
    ),
    Reference(
        "welch1947", "Welch, B. L.", "1947",
        "The generalization of 'Student's' problem when several different population "
        "variances are involved",
        "Biometrika, 34(1–2), 28–35",
        "https://doi.org/10.1093/biomet/34.1-2.28",
        "Welch's unequal-variance t-test used for two-group analytics comparisons.",
    ),
    Reference(
        "fisher1925", "Fisher, R. A.", "1925",
        "Statistical Methods for Research Workers",
        "Oliver and Boyd, Edinburgh (reprinted in S. Kotz & N. L. Johnson, Eds., "
        "Breakthroughs in Statistics, Springer, 1992)",
        "https://doi.org/10.1007/978-1-4612-4380-9_6",
        "One-way and two-way analysis of variance (ANOVA).",
    ),
    Reference(
        "wilcoxon1945", "Wilcoxon, F.", "1945",
        "Individual comparisons by ranking methods",
        "Biometrics Bulletin, 1(6), 80–83",
        "https://doi.org/10.2307/3001968",
        "The Wilcoxon signed-rank test for paired feature-ablation F1 differences.",
    ),
    Reference(
        "sidak1967", "Šidák, Z.", "1967",
        "Rectangular confidence regions for the means of multivariate normal distributions",
        "Journal of the American Statistical Association, 62(318), 626–633",
        "https://doi.org/10.1080/01621459.1967.10482935",
        "Šidák correction for post-hoc pairwise comparisons.",
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
        "niculescu2005", "Niculescu-Mizil, A., & Caruana, R.", "2005",
        "Predicting good probabilities with supervised learning",
        "Proceedings of ICML 2005, 625–632",
        "https://doi.org/10.1145/1102351.1102430",
        "Reliability-curve assessment of calibration quality.",
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
        "Precision", "Classification metrics",
        "Precision = TP / (TP + FP)",
        "Fraction of predicted-positive windows that are truly positive.",
        "abel.services.evaluation_service.segment_metrics",
        ("powers2011", "pedregosa2011"),
    ),
    Formula(
        "Recall (Sensitivity)", "Classification metrics",
        "Recall = TP / (TP + FN)",
        "Fraction of true-positive windows the model recovers.",
        "abel.services.evaluation_service.segment_metrics",
        ("powers2011", "pedregosa2011"),
    ),
    Formula(
        "F1 score", "Classification metrics",
        "F1 = 2 · (Precision · Recall) / (Precision + Recall)",
        "Harmonic mean of precision and recall; robust default under imbalance.",
        "abel.services.evaluation_service.segment_metrics",
        ("powers2011",),
    ),
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
        "abel.services.evaluation_service.segment_metrics",
        ("saito2015", "pedregosa2011"),
    ),
    Formula(
        "Cohen's κ", "Rater agreement",
        "κ = (p<sub>o</sub> − p<sub>e</sub>) / (1 − p<sub>e</sub>)",
        "Chance-corrected agreement between two raters; p<sub>o</sub> observed, "
        "p<sub>e</sub> expected-by-chance agreement.",
        "abel.services.validation_service.compute_metrics",
        ("cohen1960", "landis1977"),
    ),
    Formula(
        "Fleiss' κ", "Rater agreement",
        "κ = (P&#772; − P&#772;<sub>e</sub>) / (1 − P&#772;<sub>e</sub>)",
        "Agreement among more than two raters; P&#772; mean per-item agreement, "
        "P&#772;<sub>e</sub> = Σ<sub>j</sub> p<sub>j</sub>² expected agreement.",
        "abel.services.validation_service._fleiss_kappa",
        ("fleiss1971",),
    ),
    # ---- Cross-validation ----
    Formula(
        "Leave-one-mouse-out CV", "Cross-validation",
        "for each subject s:  train on {all mice ≠ s},  test on {mouse s}",
        "Leave-One-Group-Out cross-validation — the subject-grouped special case of "
        "k-fold with k = number of mice — so no mouse appears in both train and test.",
        "abel.validation.loso.leave_one_subject_out",
        ("saeb2017", "pedregosa2011", "kohavi1995"),
    ),
    Formula(
        "Mean ± SEM across folds", "Cross-validation",
        "x&#772; = (1/n) Σ x<sub>i</sub>&nbsp;&nbsp;&nbsp;SEM = s / &radic;n,&nbsp; "
        "s = &radic;[ Σ(x<sub>i</sub> − x&#772;)² / (n−1) ]",
        "Per-fold metric summarized across the n held-out subjects (each mouse one "
        "observation); s is the sample standard deviation.",
        "abel.validation.loso._mean_std_sem",
        ("varoquaux2018",),
    ),
    Formula(
        "95% confidence interval", "Cross-validation",
        "CI<sub>95</sub> = x&#772; ± 1.96 · SEM",
        "Normal-approximation interval used for analytics error bars.",
        "abel.ui.tabs.behavior_analytics_tab",
        ("varoquaux2018",),
    ),
    # ---- Temporal refinement ----
    Formula(
        "Moving-average smoothing", "Temporal refinement",
        "p&#771;<sub>t</sub> = (1/w) Σ<sub>i=t−w/2</sub><sup>t+w/2</sup> p<sub>i</sub>",
        "Box-filter smoothing of the per-window probability trace over a window w.",
        "abel.temporal_refinement.bout_postprocess.moving_average",
    ),
    Formula(
        "Hysteresis (Schmitt) thresholding", "Temporal refinement",
        "onset when p ≥ θ<sub>on</sub>; offset when p &lt; θ<sub>off</sub>, "
        "θ<sub>off</sub> = 0.7 · θ<sub>on</sub>",
        "Two-threshold gating so a bout opens on a strong frame and only closes when "
        "confidence drops well below, preventing flicker.",
        "abel.temporal_refinement.bout_postprocess.hysteresis_threshold",
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
        "abel.services.pose_processing_service.compute_frame_pose_features",
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
    # ---- Statistical tests ----
    Formula(
        "Welch's t-test", "Statistical tests",
        "t = (x&#772;<sub>1</sub> − x&#772;<sub>2</sub>) / "
        "&radic;(s<sub>1</sub>²/n<sub>1</sub> + s<sub>2</sub>²/n<sub>2</sub>)",
        "Unequal-variance two-group comparison in Behavior Analytics.",
        "abel.ui.tabs.behavior_analytics_tab",
        ("welch1947", "student1908"),
    ),
    Formula(
        "One-way ANOVA F", "Statistical tests",
        "F = MS<sub>between</sub> / MS<sub>within</sub>",
        "Ratio of between-group to within-group mean squares across ≥3 groups.",
        "abel.ui.tabs.behavior_analytics_tab",
        ("fisher1925",),
    ),
    Formula(
        "Šidák correction", "Statistical tests",
        "p<sub>adj</sub> = 1 − (1 − p)<sup>m</sup>,&nbsp; m = C(k, 2)",
        "Family-wise error control across m post-hoc pairwise comparisons.",
        "abel.ui.tabs.behavior_analytics_tab",
        ("sidak1967",),
    ),
    Formula(
        "Benjamini–Hochberg FDR", "Statistical tests",
        "q<sub>(k)</sub> = min<sub>k′≥k</sub> [ p<sub>(k′)</sub> · m / k′ ]",
        "False-discovery-rate adjusted p-values for motif/transition tests.",
        "abel.ui.tabs.behavior_analytics_tab._fdr_bh_adjust",
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
        "abel.services.uncertainty_service.entropy",
        ("shannon1948", "settles2009"),
    ),
    Formula(
        "Classification margin", "Active learning",
        "margin = |p<sub>(1)</sub> − p<sub>(2)</sub>|",
        "Gap between the top-two class probabilities; small margins flag ambiguous "
        "segments (least-confidence = 1 − margin).",
        "abel.services.uncertainty_service.classification_margin",
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
        ("platt1999", "zadrozny2002", "niculescu2005"),
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
        "pedregosa2011": "Cross-validation & study design",
        "saeb2017": "Cross-validation & study design",
        "varoquaux2018": "Cross-validation & study design",
        "stone1974": "Cross-validation & study design",
        "kohavi1995": "Cross-validation & study design",
        "saito2015": "Evaluation metrics",
        "powers2011": "Evaluation metrics",
        "chicco2020": "Evaluation metrics",
        "cohen1960": "Rater agreement",
        "fleiss1971": "Rater agreement",
        "landis1977": "Rater agreement",
        "student1908": "Statistical tests",
        "welch1947": "Statistical tests",
        "fisher1925": "Statistical tests",
        "wilcoxon1945": "Statistical tests",
        "sidak1967": "Statistical tests",
        "benjamini1995": "Statistical tests",
        "good2005": "Statistical tests",
        "rabiner1989": "Models, calibration & active learning",
        "platt1999": "Models, calibration & active learning",
        "zadrozny2002": "Models, calibration & active learning",
        "niculescu2005": "Models, calibration & active learning",
        "settles2009": "Models, calibration & active learning",
        "shannon1948": "Models, calibration & active learning",
        "mcinnes2018": "Models, calibration & active learning",
    }.get(ref.key, "Other")


_REFERENCE_SECTIONS = [
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
        "<p style='color:#90A4AE;'>Peer-reviewed and canonical sources for every "
        "statistical procedure ABEL performs. Each entry notes which ABEL analysis it "
        "supports. Links open the DOI or archival page.</p>",
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
        "<p style='color:#90A4AE;'>The raw formulas ABEL evaluates across its primary "
        "analyses. Each is tagged with the implementing function so equations can be "
        "traced directly to the code.</p>",
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
                        "<span style='color:#78909C;'> — "
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

"""ABEL Validation / Meta-Analysis Platform.

A rebuilt-from-scratch validation suite that reuses ABEL's *real* shipped
training primitive (``ActiveLearningTrainerService.train_and_evaluate``) to
produce publication-grade analyses across many projects:

1. Optimal-clips learning curves (data efficiency)
2. Cross-project meta-analysis
3. Feature/pipeline ablation impact
4. Generalization / human-agreement validation

Ground truth = the project's own reviewed/accepted clips on held-out
subjects/sessions (no separate "gold" dataset).  See ``holdout`` for the
high-confidence held-out evaluation set.
"""

from abel.validation.datamodel import (
    CellResult,
    ConfigEvalResult,
    ProjectRef,
    RunManifest,
)

__all__ = ["CellResult", "ConfigEvalResult", "ProjectRef", "RunManifest"]

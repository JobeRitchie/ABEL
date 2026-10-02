"""Analysis options for :mod:`abel.services.group_stats`.

Kept free of scipy so the Analytics tab can hold and persist the options
without importing the statistics stack at startup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

POSTHOC_LABELS: dict[str, str] = {
    "tukey": "Tukey",
    "sidak": "Šídák",
    "holm_sidak": "Holm-Šídák",
    "bonferroni": "Bonferroni",
    "dunnett": "Dunnett (each vs control)",
    "fisher": "Fisher's LSD (uncorrected)",
    "none": "None",
}


@dataclass
class StatsOptions:
    """The analysis choices, laid out like Prism's analysis dialog."""

    # "unpaired": every value is a different animal.  "paired": the same
    # subjects appear in every group, matched by subject name.
    design: str = "unpaired"
    gaussian: bool = True          # False: rank-based (nonparametric) tests
    equal_sd: bool = True          # False: Welch t-test / Welch ANOVA
    posthoc: str = "tukey"         # see POSTHOC_LABELS
    control_group: str = ""        # Dunnett control; "" means the first group
    # Geisser-Greenhouse correction for repeated measures (Prism's default).
    geisser_greenhouse: bool = True
    # Prism runs multiple comparisons regardless of the omnibus p; this
    # restores the older "only after a significant ANOVA" gate if wanted.
    posthoc_only_if_significant: bool = False
    # Two-way designs: name of the factor measured repeatedly on each subject
    # ("" = both factors between subjects).
    repeated_factor: str = ""
    # Two-way simple effects: "first" compares first-factor levels within each
    # level of the second factor, "second" the reverse, "both", or "none".
    twoway_compare: str = "first"
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "StatsOptions":
        out = cls()
        if isinstance(data, dict):
            for k, v in data.items():
                if hasattr(out, k):
                    try:
                        setattr(out, k, type(getattr(out, k))(v))
                    except (TypeError, ValueError):
                        pass
        if out.posthoc not in POSTHOC_LABELS:
            out.posthoc = "tukey"
        if out.design not in ("unpaired", "paired"):
            out.design = "unpaired"
        if out.twoway_compare not in ("first", "second", "both", "none"):
            out.twoway_compare = "first"
        return out


def methods_text(opts: StatsOptions) -> str:
    """Methods-section description of the group comparisons *opts* select."""
    paired = opts.design == "paired"
    if not opts.gaussian:
        two = "a Wilcoxon matched-pairs signed rank test" if paired else "a Mann-Whitney U test"
        many = (
            "a Friedman test" if paired else "a Kruskal-Wallis test"
        ) + " followed by Dunn's multiple comparisons"
    else:
        if paired:
            two = "a paired t-test"
            many = "a repeated-measures one-way ANOVA" + (
                " with the Geisser-Greenhouse correction" if opts.geisser_greenhouse else ""
            )
        elif opts.equal_sd:
            two, many = "an unpaired Student's t-test", "an ordinary one-way ANOVA"
        else:
            two, many = "an unpaired t-test with Welch's correction", "Welch's ANOVA"
        if opts.posthoc == "none":
            many += " without multiple comparisons"
        else:
            name = POSTHOC_LABELS.get(opts.posthoc, opts.posthoc)
            if not opts.equal_sd and not paired and opts.posthoc == "tukey":
                name = "Games-Howell"
            many += f" followed by {name} multiple comparisons"
    unit = "the same subjects matched across groups" if paired else "one value per animal"
    two_way = (
        "Designs with two factors used a two-way ANOVA with Type III sums of squares"
        + (
            f", with {opts.repeated_factor} as a repeated (within-subject) factor"
            if opts.repeated_factor else ""
        )
        + "."
    )
    return (
        f"Groups were compared in ABEL ({unit}) with {two} for two groups and {many} "
        f"for three or more. {two_way} Tests were two-tailed with alpha = {opts.alpha:g}. "
        "Animals that never performed a behavior were included with 0 bouts and 0 s, "
        "and with a latency equal to the time available; their mean bout duration is "
        "undefined and was left out. Error bars show the SEM, SD or a t-based 95% "
        "confidence interval, as labeled."
    )

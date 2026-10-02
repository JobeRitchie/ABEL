"""Statistics options and results dialog for the Analytics tab.

Shows the exact test behind every p-value on screen and lets the user change
the analysis the way Prism's dialog does: experimental design, Gaussian or
rank-based, equal SDs or Welch, and the multiple comparisons method.
"""

from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from abel.services.group_stats_options import POSTHOC_LABELS, StatsOptions


def describe_test(opts: StatsOptions, n_groups: int, two_way: bool = False) -> str:
    """Name of the test the options select for *n_groups* groups."""
    if two_way:
        if opts.repeated_factor:
            return f"Two-way ANOVA with '{opts.repeated_factor}' repeated within subject (mixed design)"
        return "Ordinary two-way ANOVA (Type III)"
    paired = opts.design == "paired"
    if n_groups == 2:
        if not opts.gaussian:
            return "Wilcoxon matched-pairs signed rank test" if paired else "Mann-Whitney test"
        if paired:
            return "Paired t test"
        return "Unpaired t test" if opts.equal_sd else "Unpaired t test with Welch's correction"
    if not opts.gaussian:
        return "Friedman test + Dunn's" if paired else "Kruskal-Wallis test + Dunn's"
    if paired:
        gg = " (Geisser-Greenhouse)" if opts.geisser_greenhouse else ""
        return f"Repeated measures one-way ANOVA{gg}"
    return "Ordinary one-way ANOVA" if opts.equal_sd else "Welch's ANOVA"


class StatsOptionsWidget(QWidget):
    """Editable :class:`StatsOptions`, laid out like Prism's analysis dialog."""

    def __init__(
        self,
        opts: StatsOptions,
        group_names: Sequence[str] = (),
        factor_names: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._factor_names = list(factor_names)

        self._design = QComboBox()
        self._design.addItem("Unpaired: a different animal in each group", "unpaired")
        self._design.addItem("Paired / repeated measures: same animals, matched by subject", "paired")
        self._design.setToolTip(
            "Paired compares each subject with itself across groups (e.g. the same\n"
            "mouse on Day 1 and Day 2). Subjects are matched by name; a subject\n"
            "missing from any group is left out of a paired analysis."
        )
        self._gaussian = QCheckBox("Assume Gaussian distribution (parametric tests)")
        self._gaussian.setToolTip(
            "Off: rank-based tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman)."
        )
        self._equal_sd = QCheckBox("Assume equal SDs (off: Welch's t test / Welch's ANOVA)")
        self._gg = QCheckBox("Geisser-Greenhouse correction for repeated measures")
        self._gg.setToolTip("Prism's default. Corrects for unequal variances of the differences (sphericity).")

        self._posthoc = QComboBox()
        for key, lab in POSTHOC_LABELS.items():
            self._posthoc.addItem(lab, key)
        self._posthoc.setToolTip(
            "Tukey: every pair (Prism's default after one-way ANOVA).\n"
            "Dunnett: each group vs the control group.\n"
            "Šídák / Holm-Šídák / Bonferroni: corrected t tests on every pair.\n"
            "Fisher's LSD: no correction.\n"
            "Rank tests always use Dunn's method; Welch's ANOVA uses Games-Howell for Tukey."
        )
        self._control = QComboBox()
        self._control.addItem("(first group)", "")
        for g in group_names:
            self._control.addItem(str(g), str(g))
        self._gate = QCheckBox("Only run comparisons when the overall test is significant")
        self._gate.setToolTip("Prism runs comparisons regardless; leave off to match Prism.")
        self._alpha = QDoubleSpinBox()
        self._alpha.setDecimals(3)
        self._alpha.setRange(0.001, 0.2)
        self._alpha.setSingleStep(0.005)

        self._repeated = QComboBox()
        self._repeated.addItem("(none: both factors between subjects)", "")
        for f in self._factor_names:
            self._repeated.addItem(f, f)
        self._compare = QComboBox()
        if len(self._factor_names) == 2:
            first, second = self._factor_names
        else:  # chosen later (Statistics dialog): name the roles, not the factors
            first, second = "primary-factor", "second-factor"
        self._compare.addItem(f"{first} levels within each {second} level", "first")
        self._compare.addItem(f"{second} levels within each {first} level", "second")
        self._compare.addItem("Both", "both")
        self._compare.addItem("No comparisons", "none")

        design_box = QGroupBox("Experimental design")
        f1 = QFormLayout(design_box)
        f1.addRow(self._design)
        f1.addRow(self._gaussian)
        f1.addRow(self._equal_sd)
        f1.addRow(self._gg)
        cmp_box = QGroupBox("Multiple comparisons")
        f2 = QFormLayout(cmp_box)
        f2.addRow("Method:", self._posthoc)
        f2.addRow("Control group:", self._control)
        f2.addRow(self._gate)
        f2.addRow("Alpha:", self._alpha)
        self._twoway_box = QGroupBox("Two-way designs (two factors split)")
        f3 = QFormLayout(self._twoway_box)
        f3.addRow("Repeated (within-subject) factor:", self._repeated)
        f3.addRow("Compare:", self._compare)
        self._twoway_box.setVisible(len(self._factor_names) >= 2)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(design_box)
        lay.addWidget(cmp_box)
        lay.addWidget(self._twoway_box)
        lay.addStretch(1)

        self.set_options(opts)
        for w in (self._design, self._posthoc, self._repeated):
            w.currentIndexChanged.connect(self._sync_enabled)
        self._gaussian.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    def set_options(self, opts: StatsOptions) -> None:
        self._design.setCurrentIndex(max(0, self._design.findData(opts.design)))
        self._gaussian.setChecked(opts.gaussian)
        self._equal_sd.setChecked(opts.equal_sd)
        self._gg.setChecked(opts.geisser_greenhouse)
        self._posthoc.setCurrentIndex(max(0, self._posthoc.findData(opts.posthoc)))
        self._control.setCurrentIndex(max(0, self._control.findData(opts.control_group)))
        self._gate.setChecked(opts.posthoc_only_if_significant)
        self._alpha.setValue(float(opts.alpha))
        self._repeated.setCurrentIndex(max(0, self._repeated.findData(opts.repeated_factor)))
        self._compare.setCurrentIndex(max(0, self._compare.findData(opts.twoway_compare)))

    def options(self) -> StatsOptions:
        return StatsOptions(
            design=str(self._design.currentData()),
            gaussian=self._gaussian.isChecked(),
            equal_sd=self._equal_sd.isChecked(),
            posthoc=str(self._posthoc.currentData()),
            control_group=str(self._control.currentData() or ""),
            geisser_greenhouse=self._gg.isChecked(),
            posthoc_only_if_significant=self._gate.isChecked(),
            repeated_factor=str(self._repeated.currentData() or ""),
            twoway_compare=str(self._compare.currentData()),
            alpha=float(self._alpha.value()),
        )

    def _sync_enabled(self) -> None:
        paired = self._design.currentData() == "paired"
        gaussian = self._gaussian.isChecked()
        self._equal_sd.setEnabled(gaussian and not paired)
        self._gg.setEnabled(gaussian and (paired or bool(self._repeated.currentData())))
        self._control.setEnabled(self._posthoc.currentData() == "dunnett")


class StatsReportDialog(QDialog):
    """Options on the left, the full report of what was run on the right.

    ``run(opts)`` recomputes and returns the report text; it is called on
    Apply so the user sees the effect of a change immediately.
    """

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        opts: StatsOptions,
        report: str,
        run: Callable[[StatsOptions], str] | None,
        group_names: Sequence[str] = (),
        factor_names: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._run = run
        self.applied: StatsOptions | None = None

        self._opts = StatsOptionsWidget(opts, group_names, factor_names, self)
        self._text = QTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._text.setPlainText(report or "(No statistics: nothing comparable is shown.)")

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._opts)
        splitter.addWidget(self._text)
        splitter.setStretchFactor(1, 1)

        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Re-run the statistics with these options.")
        apply_btn.clicked.connect(self._apply)
        apply_btn.setEnabled(run is not None)
        copy_btn = QPushButton("Copy Report")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._text.toPlainText()))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        hint = QLabel(
            "Values are one per animal (session), exactly as exported with Export Data."
        )
        hint.setWordWrap(True)
        btns = QHBoxLayout()
        btns.addWidget(hint, 1)
        btns.addWidget(apply_btn)
        btns.addWidget(copy_btn)
        btns.addWidget(close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(splitter, 1)
        lay.addLayout(btns)
        fm = self.fontMetrics()
        em = fm.horizontalAdvance("M")
        w, h = em * 120, fm.height() * 38
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            w, h = min(w, int(avail.width() * 0.9)), min(h, int(avail.height() * 0.85))
        self.resize(w, h)
        splitter.setSizes([int(w * 0.36), int(w * 0.64)])

    def _apply(self) -> None:
        if self._run is None:
            return
        opts = self._opts.options()
        self.applied = opts
        self._text.setPlainText(self._run(opts))

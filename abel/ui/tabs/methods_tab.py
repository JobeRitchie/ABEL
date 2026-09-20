"""Methods tab: references, formulas, and a methods-section write-up helper.

Documents the statistical rigor of ABEL for users and reviewers. Three subtabs:

* **References**: the peer-reviewed sources justifying each procedure, with links.
* **Formulas**: the raw formulas ABEL evaluates, each tied to its code.
* **Write-up Helper**: a draft methods section assembled from the open project's
  own settings and results.

The first two render from :mod:`abel.ui.methods_content` (the single source of
truth) and are static; only the write-up helper is project-scoped, so
``set_project`` forwards to it alone.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.ui.methods_content import render_formulas_html, render_references_html
from abel.ui.tabs.methods_writeup_tab import MethodsWriteupTab


def _doc_browser(html: str) -> QTextBrowser:
    view = QTextBrowser()
    view.setOpenExternalLinks(True)  # DOI / archival links open in the browser
    view.setStyleSheet("QTextBrowser { background: #1c2530; border: none; }")
    view.setHtml(html)
    return view


class MethodsTab(QWidget):
    """Top-level tab hosting the References, Formulas and Write-up subtabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.writeup_tab = MethodsWriteupTab()

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.addTab(_doc_browser(render_references_html()), "References")
        self._tabs.addTab(_doc_browser(render_formulas_html()), "Formulas")
        self._tabs.addTab(self.writeup_tab, "Write-up Helper")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

    def set_project(self, project_root: Path) -> None:
        """Point the write-up helper at the open project (the docs are static)."""
        self.writeup_tab.set_project(project_root)

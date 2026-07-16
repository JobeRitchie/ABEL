"""Methods tab: literature references and raw formulas for ABEL's analyses.

A static, project-independent tab documenting the statistical rigor of ABEL for
users and reviewers. Two subtabs, both rendered from :mod:`abel.ui.methods_content`
(the single source of truth):

* **References** — the peer-reviewed sources justifying each procedure, with links.
* **Formulas** — the raw formulas ABEL evaluates, each tied to its code.

Content is static, so this tab needs no ``set_project`` / refresh wiring.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.ui.methods_content import render_formulas_html, render_references_html


def _doc_browser(html: str) -> QTextBrowser:
    view = QTextBrowser()
    view.setOpenExternalLinks(True)  # DOI / archival links open in the browser
    view.setStyleSheet("QTextBrowser { background: #1c2530; border: none; }")
    view.setHtml(html)
    return view


class MethodsTab(QWidget):
    """Top-level tab hosting References and Formulas subtabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.addTab(_doc_browser(render_references_html()), "References")
        self._tabs.addTab(_doc_browser(render_formulas_html()), "Formulas")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

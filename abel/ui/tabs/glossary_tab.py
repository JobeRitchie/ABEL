"""Glossary subtab of Info: type any word from ABEL and read what it means."""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from abel.ui.glossary import GlossaryEntry, entry, search

_ROLE_TERM = Qt.ItemDataRole.UserRole


class GlossaryTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        title = QLabel("Glossary")
        title.setStyleSheet("font-size: 14px; font-weight: 800; color: #90CAF9;")
        blurb = QLabel(
            "Type any word you see in ABEL: a button, a setting, a tab or a term."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("font-size: 11px; color: #8FA6B4;")

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search, e.g. stride, Retrain, PR-AUC, merge gap")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refilter)
        self._search.returnPressed.connect(self._focus_first)

        self._count = QLabel()
        self._count.setStyleSheet("font-size: 11px; color: #78909C;")

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_current)
        em = self.fontMetrics().horizontalAdvance("M")
        self._list.setMinimumWidth(14 * em)

        self._detail = QTextBrowser()
        self._detail.setOpenLinks(False)
        self._detail.anchorClicked.connect(self._on_link)
        self._detail.setStyleSheet("QTextBrowser { background: #1c2530; border: none; }")

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._list)
        split.addWidget(self._detail)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(blurb)
        layout.addWidget(self._search)
        layout.addWidget(self._count)
        layout.addWidget(split, 1)

        self._refilter("")

    # ── search ─────────────────────────────────────────────────────────

    def _refilter(self, text: str) -> None:
        hits = search(text)
        self._list.blockSignals(True)
        self._list.clear()
        for e in hits:
            item = QListWidgetItem(e.term)
            item.setData(_ROLE_TERM, e.term)
            item.setToolTip(e.where)
            self._list.addItem(item)
        self._list.blockSignals(False)
        n = len(hits)
        self._count.setText(f"{n} match{'es' if n != 1 else ''}" if text.strip() else f"{n} terms")
        if hits:
            self._list.setCurrentRow(0)
        else:
            self._detail.setHtml(
                "<p style='color:#90A4AE;'>No match. Try a shorter or different word.</p>"
            )

    def _focus_first(self) -> None:
        if self._list.count():
            self._list.setCurrentRow(0)
            self._list.setFocus()

    def _on_current(self, item: QListWidgetItem | None, _prev=None) -> None:
        if item is None:
            return
        e = entry(item.data(_ROLE_TERM))
        if e is not None:
            self._show(e)

    def _on_link(self, url: QUrl) -> None:
        term = url.toString().removeprefix("term:")
        for row in range(self._list.count()):
            if self._list.item(row).data(_ROLE_TERM) == term:
                self._list.setCurrentRow(row)
                return
        # Not in the current filter: search for it directly.
        self._search.setText(term)

    # ── rendering ──────────────────────────────────────────────────────

    def _show(self, e: GlossaryEntry) -> None:
        parts = [
            f"<h2 style='color:#90CAF9; margin-bottom:2px;'>{escape(e.term)}</h2>",
            f"<p style='color:#78909C; margin-top:0;'>Where: {escape(e.where)}</p>",
            f"<p style='color:#ECEFF1; font-size:14px;'>{escape(e.text)}</p>",
        ]
        if e.aliases:
            parts.append(
                "<p style='color:#90A4AE;'>Also appears as: "
                + ", ".join(escape(a) for a in e.aliases)
                + "</p>"
            )
        links = [
            f"<a href='term:{escape(t)}' style='color:#64B5F6;'>{escape(t)}</a>"
            for t in e.see
            if entry(t) is not None
        ]
        if links:
            parts.append("<p style='color:#90A4AE;'>See also: " + ", ".join(links) + "</p>")
        self._detail.setHtml("".join(parts))

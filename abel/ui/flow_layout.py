"""A layout that wraps its children onto additional rows instead of squeezing them.

A ``QHBoxLayout`` full of buttons has one failure mode on a narrow window: it
shrinks every child below its ``sizeHint`` until the labels clip.  ``FlowLayout``
keeps each child at its preferred width and starts a new row when it runs out of
space, which is what a toolbar of variable-width buttons actually wants.

Standard Qt flow-layout pattern, ported to PySide6.
"""

from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto new rows as the width shrinks."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margin: int = 0,
        h_spacing: int = 6,
        v_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # -- QLayout plumbing ------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        return Qt.Orientation(0)

    # -- Height-for-width ------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    # -- Placement -------------------------------------------------------

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(),
                                  -margins.right(), -margins.bottom())
        x = effective.x()
        y = effective.y()
        row_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() and row_height > 0:
                # Does not fit on this row -- wrap.
                x = effective.x()
                y = y + row_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            row_height = max(row_height, hint.height())

        return y + row_height - rect.y() + margins.bottom()


def labelled(text: str, widget: QWidget, *, spacing: int = 5) -> QWidget:
    """Weld a caption to its control so a wrapping row never separates them."""
    from PySide6.QtWidgets import QHBoxLayout, QLabel

    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    row.addWidget(QLabel(text))
    row.addWidget(widget)
    return holder


def flow_row(widgets: list[QWidget], *, h_spacing: int = 6, v_spacing: int = 6) -> QWidget:
    """Wrap ``widgets`` in a container using a :class:`FlowLayout`.

    Returned as a widget rather than a layout so callers can drop it straight
    into an existing ``QVBoxLayout`` with ``addWidget``.
    """
    holder = QWidget()
    holder.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    layout = FlowLayout(holder, h_spacing=h_spacing, v_spacing=v_spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return holder

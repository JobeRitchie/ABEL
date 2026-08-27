"""Theme matplotlib's Qt navigation toolbar to match ABEL's dark stylesheet.

``NavigationToolbar2QT`` recolours its icons for dark themes only when
``self.palette().color(self.backgroundRole()).value() < 128``.  A Qt stylesheet
paints the widget dark but never updates its ``QPalette``, so that check fails
and the icons stay pure black on our ``#0D1B2A`` ground -- a 1.21:1 contrast
ratio, which is effectively invisible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QToolButton, QWidget

#: Icon/foreground tint. 8.4:1 against ``#0D1B2A``.
TOOLBAR_FG = "#B0C2D0"
#: Ground shared with the app stylesheet.
TOOLBAR_BG = "#0D1B2A"


def style_navigation_toolbar(
    toolbar: QWidget | None,
    *,
    fg: str = TOOLBAR_FG,
    bg: str = TOOLBAR_BG,
) -> None:
    """Recolour a matplotlib navigation toolbar's icons and palette in place.

    Safe to call with ``None`` and safe to call twice.  Works on the rendered
    pixmaps rather than matplotlib's ``_icon`` internals, so it does not depend
    on the backend's private API.
    """
    if toolbar is None:
        return

    colour = QColor(fg)

    palette = toolbar.palette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                 QPalette.ColorRole.Button):
        palette.setColor(role, QColor(bg))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(role, colour)
    toolbar.setPalette(palette)
    toolbar.setStyleSheet(
        f"QToolBar {{ background: {bg}; border: none; }}"
        f"QToolButton {{ background: transparent; border: none; padding: 3px; }}"
        f"QToolButton:hover {{ background: #163D6E; border-radius: 3px; }}"
        f"QLabel {{ color: {fg}; }}"
    )

    for button in toolbar.findChildren(QToolButton):
        icon = button.icon()
        if icon.isNull():
            continue
        sizes = icon.availableSizes()
        size = sizes[0] if sizes else button.iconSize()
        if size.isEmpty():
            continue
        pixmap = icon.pixmap(size)
        if pixmap.isNull():
            continue
        # Mask out everything that is *not* black, then flood the remaining
        # (glyph) pixels with the tint -- the same trick matplotlib uses, just
        # applied after construction where the palette is finally correct.
        mask = pixmap.createMaskFromColor(QColor("black"),
                                          Qt.MaskMode.MaskOutColor)
        pixmap.fill(colour)
        pixmap.setMask(mask)
        button.setIcon(QIcon(pixmap))

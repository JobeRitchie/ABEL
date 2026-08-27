"""Application bootstrap and dependency-safe launch."""

from __future__ import annotations

import faulthandler
import os
import sys

# Enable low-level crash traces (SIGSEGV / STATUS_ACCESS_VIOLATION) to stderr
# so they appear in launcher_last.log when a native crash occurs.
faulthandler.enable()


def run() -> int:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication, QIcon
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        print("ABEL could not load PySide6 GUI runtime.")
        print("This is often a Qt DLL/runtime conflict, not a missing package.")
        print(f"Startup failed: {exc}")
        return 1

    # Keep scaling consistent on Windows multi-monitor and non-100% DPI setups.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    from abel.ui.assets import icon_path
    from abel.ui.main_window import MainWindow

    # On Windows, declare an explicit AppUserModelID so the taskbar shows the
    # ABEL icon (not the generic python feather) and groups all windows together.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Abel.App")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(icon_path())))
    app.setStyleSheet(_STYLESHEET)
    win = MainWindow()
    win.showMaximized()
    return app.exec()


_STYLESHEET = """
/* ── Global ────────────────────────────────────────────── */
QWidget {
    background-color: #0D1B2A;
    color: #E0E8F0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    font-weight: 600;
}

/* ── Main window / frames ──────────────────────────────── */
QMainWindow, QDialog {
    background-color: #0D1B2A;
}

QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #1565C0;
}

/* ── Tab bar ───────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #1565C0;
    background: #0D1B2A;
}
QTabBar::tab {
    background: #0F2744;
    color: #90CAF9;
    font-weight: 700;
    /* 15 top-level tabs at min-width 84 needed ~1620 px, so four of them fell
       off the bar and were reachable only via the scroll chevrons. Let each tab
       size to its own label instead. */
    padding: 6px 8px;
    min-width: 0px;
    max-width: 176px;
    border: 1px solid #1565C0;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1565C0;
    color: #FFFFFF;
}
QTabBar::tab:hover:!selected {
    background: #163D6E;
}

/* ── Buttons ───────────────────────────────────────────── */
QPushButton {
    background-color: #1565C0;
    color: #FFFFFF;
    font-weight: 700;
    border: 1px solid #1E88E5;
    border-radius: 5px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #1976D2;
    border-color: #42A5F5;
}
QPushButton:pressed {
    background-color: #0D47A1;
}
QPushButton:disabled {
    background-color: #1A2A3A;
    color: #8FA6B4;
    border-color: #263238;
}

/* Destructive actions. Setting only `color:` left red text on the theme's
   blue button ground, which measured as low as 1.8:1. */
QPushButton#destructive {
    background-color: #6B1E1E;
    color: #FFD9D9;
    border: 1px solid #8E2020;
}
QPushButton#destructive:hover {
    background-color: #8E2020;
    border-color: #B33A3A;
}
QPushButton#destructive:disabled {
    background-color: #2A1616;
    color: #B08A8A;
    border-color: #3E1F1F;
}

/* Qt's default disabled palette lands around #5D5D5D, which measures 2.6:1 on
   our grounds -- a user can see something is greyed out but cannot read which
   action it is. Restate it for every widget class that carries text. */
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled,
QGroupBox:disabled, QGroupBox::title:disabled,
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QTabBar::tab:disabled, QToolButton:disabled {
    color: #8FA6B4;
}

/* ── Input fields ──────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0F2744;
    color: #E0E8F0;
    border: 1px solid #1565C0;
    border-radius: 4px;
    padding: 4px 7px;
    font-weight: 600;
    selection-background-color: #1976D2;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #42A5F5;
}
QComboBox::drop-down {
    border-left: 1px solid #1565C0;
}
QComboBox QAbstractItemView {
    background-color: #0F2744;
    color: #E0E8F0;
    selection-background-color: #1565C0;
    border: 1px solid #1565C0;
}

/* ── Labels ────────────────────────────────────────────── */
QLabel {
    font-weight: 700;
    background: transparent;
}

/* ── Lists & tables ────────────────────────────────────── */
QListWidget, QTableWidget, QTreeWidget {
    background-color: #0F2744;
    alternate-background-color: #112A4A;
    color: #E0E8F0;
    border: 1px solid #1565C0;
    gridline-color: #163D6E;
    font-weight: 600;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #1565C0;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #0D47A1;
    color: #E0E8F0;
    font-weight: 700;
    border: 1px solid #163D6E;
    padding: 4px;
}

/* ── Scrollbars ────────────────────────────────────────── */
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0D1B2A;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #1565C0;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0; width: 0;
}

/* ── Checkbox ──────────────────────────────────────────── */
QCheckBox {
    font-weight: 700;
    color: #90CAF9;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #1565C0;
    background: #0F2744;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #1565C0;
}

/* ── Status bar / separators ───────────────────────────── */
QStatusBar {
    background: #0D1B2A;
    color: #8FA6B4;
    font-weight: 600;
}

/* ── Tooltips ──────────────────────────────────────────── */
QToolTip {
    background-color: #163D6E;
    color: #E0E8F0;
    border: 1px solid #1976D2;
    font-weight: 600;
    padding: 4px;
}
"""

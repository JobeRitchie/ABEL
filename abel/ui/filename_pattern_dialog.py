"""Build the subject/session filename regexes by highlighting an example name.

Instead of writing a regular expression, the user selects the characters of one
filename that name the subject, then (optionally) the session.  The generated
patterns are previewed against every imported filename, with a live count of
the videos that would pair with a pose file.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import ImportNameSettings, PoseAsset, VideoAsset
from abel.services.import_service import ImportService
from abel.utils.filename_pattern import generate_capture_regex, snap_selection


class FilenamePatternDialog(QDialog):
    """Highlight the subject/session in a filename to generate the parsing regexes.

    ``result_settings`` holds the accepted :class:`ImportNameSettings`.
    """

    def __init__(
        self,
        import_service: ImportService,
        video_paths: list[Path],
        pose_paths: list[Path],
        settings: ImportNameSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Filename Pattern")
        self._service = import_service
        self._videos = [Path(p) for p in video_paths]
        self._poses = [Path(p) for p in pose_paths]
        self.result_settings: ImportNameSettings | None = None

        layout = QVBoxLayout(self)
        explainer = QLabel(
            "Pick an example file, highlight the part of its name that identifies the "
            "subject and click <b>Use as Subject</b>. If the name also carries the "
            "session (day, test, condition), highlight that and click <b>Use as "
            "Session</b>. The patterns are checked against every imported file below."
        )
        explainer.setWordWrap(True)
        layout.addWidget(explainer)

        self._example_combo = QComboBox()
        self._example_combo.setEditable(True)
        for path in [*self._videos, *self._poses]:
            self._example_combo.addItem(path.name)
        if not (self._videos or self._poses):
            self._example_combo.addItem("COA301_Day2.mp4")
        self._example_combo.currentTextChanged.connect(self._on_example_changed)

        self._stem_edit = QLineEdit()
        self._stem_edit.setReadOnly(True)
        font = self._stem_edit.font()
        font.setPointSizeF(font.pointSizeF() * 1.4)
        self._stem_edit.setFont(font)
        self._stem_edit.setToolTip("Drag across the characters you want to extract.")

        subject_btn = QPushButton("Use as Subject")
        session_btn = QPushButton("Use as Session")
        clear_session_btn = QPushButton("No Session in Name")
        subject_btn.clicked.connect(lambda: self._use_selection("subject"))
        session_btn.clicked.connect(lambda: self._use_selection("session"))
        clear_session_btn.clicked.connect(lambda: self._session_regex.setText(""))
        # Clicking a focusable button would clear the line edit's selection.
        for btn in (subject_btn, session_btn, clear_session_btn):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        pick_row = QHBoxLayout()
        pick_row.addWidget(subject_btn)
        pick_row.addWidget(session_btn)
        pick_row.addWidget(clear_session_btn)
        pick_row.addStretch(1)

        self._subject_regex = QLineEdit(settings.subject_regex)
        self._session_regex = QLineEdit(settings.session_regex)
        self._subject_regex.textChanged.connect(self._refresh_preview)
        self._session_regex.textChanged.connect(self._refresh_preview)

        form = QFormLayout()
        form.addRow("Example file:", self._example_combo)
        form.addRow("Name to highlight:", self._stem_edit)
        layout.addLayout(form)
        layout.addLayout(pick_row)
        regex_form = QFormLayout()
        regex_form.addRow("Subject regex:", self._subject_regex)
        regex_form.addRow("Session regex:", self._session_regex)
        layout.addLayout(regex_form)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["File", "Kind", "Subject", "Session"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply Pattern")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Size from the font rather than fixed pixels so scaling doesn't clip.
        em = self.fontMetrics().horizontalAdvance("M")
        self.resize(em * 60, em * 45)

        self._on_example_changed(self._example_combo.currentText())

    # ------------------------------------------------------------------

    def _on_example_changed(self, name: str) -> None:
        self._stem_edit.setText(Path(name.strip()).stem if name.strip() else "")
        self._refresh_preview()

    def _use_selection(self, which: str) -> None:
        stem = self._stem_edit.text()
        start = self._stem_edit.selectionStart()
        length = len(self._stem_edit.selectedText())
        if start < 0 or length == 0:
            QMessageBox.information(
                self, "Highlight Text",
                "Drag across part of the filename first, then click the button.",
            )
            return
        try:
            snapped = snap_selection(stem, start, start + length)
            pattern = generate_capture_regex(stem, start, start + length)
        except ValueError as exc:
            QMessageBox.information(self, "Highlight Text", str(exc))
            return
        self._stem_edit.setSelection(snapped[0], snapped[1] - snapped[0])
        target = self._subject_regex if which == "subject" else self._session_regex
        target.setText(pattern)

    def _settings(self) -> ImportNameSettings:
        return ImportNameSettings(
            subject_regex=self._subject_regex.text().strip() or ImportNameSettings().subject_regex,
            subject_group_index=1,
            session_regex=self._session_regex.text().strip(),
            session_group_index=1,
        )

    def _regex_error(self) -> str | None:
        for label, edit in (("Subject", self._subject_regex), ("Session", self._session_regex)):
            text = edit.text().strip()
            if not text:
                continue
            try:
                compiled = re.compile(text)
            except re.error as exc:
                return f"{label} regex is invalid: {exc}"
            if compiled.groups < 1:
                return f"{label} regex needs a capture group ( … ) around the part to extract."
        return None

    def _refresh_preview(self) -> None:
        error = self._regex_error()
        settings = self._settings()
        rows = [(p, "video") for p in self._videos] + [(p, "pose") for p in self._poses]
        self._table.setRowCount(len(rows))
        missing = QBrush(QColor("#E57373"))
        n_subject = 0
        for r, (path, kind) in enumerate(rows):
            subject = None if error else self._service.extract_subject_name(path, settings)
            session = None if error else self._service.extract_session_type(path, settings)
            n_subject += bool(subject)
            values = [
                path.name,
                kind,
                subject or "(no match)",
                session or ("(no match)" if settings.session_regex else ""),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if value == "(no match)":
                    item.setForeground(missing)
                self._table.setItem(r, c, item)

        if error:
            self._summary.setText(f"⚠ {error}")
            return
        if not rows:
            self._summary.setText("Import videos and pose files to preview the pattern on them.")
            return
        linked = self._service.auto_match(
            [
                VideoAsset(
                    asset_id=f"v{i}", source_path=str(p),
                    subject_id=self._service.extract_subject_name(p, settings),
                    session_id=self._service.extract_session_type(p, settings),
                )
                for i, p in enumerate(self._videos)
            ],
            [
                PoseAsset(
                    asset_id=f"p{i}", source_path=str(p), format=p.suffix.lower().lstrip("."),
                    subject_id=self._service.extract_subject_name(p, settings),
                    session_id=self._service.extract_session_type(p, settings),
                )
                for i, p in enumerate(self._poses)
            ],
        )
        self._summary.setText(
            f"Subject found in {n_subject} of {len(rows)} file(s). "
            f"{len(linked)} of {len(self._videos)} video(s) would pair with a pose file."
        )

    def _accept(self) -> None:
        error = self._regex_error()
        if error:
            QMessageBox.warning(self, "Invalid Pattern", error)
            return
        self.result_settings = self._settings()
        self.accept()

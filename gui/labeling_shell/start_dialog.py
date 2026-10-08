"""Startup dialog: open or create a project before the labeler is usable.

A project is required (videos, labels, frames and models live in it), so the
app asks for one up front instead of opening onto an empty window. Recent
projects are remembered per user in QSettings; the dialog only reports the
choice — MainWindow does the opening.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from .theme import mark_accent

RECENT_KEY = "recent_projects"
MAX_RECENT = 8


def recent_projects(settings: QtCore.QSettings) -> list[str]:
    """Remembered project folders that still hold a project.json, newest first."""
    value = settings.value(RECENT_KEY, []) or []
    if isinstance(value, str):  # QSettings collapses a one-item list on some backends
        value = [value]
    return [p for p in value if os.path.isfile(os.path.join(p, "project.json"))]


def remember_project(settings: QtCore.QSettings, path) -> None:
    path = os.path.abspath(str(path))
    value = settings.value(RECENT_KEY, []) or []
    if isinstance(value, str):
        value = [value]
    settings.setValue(RECENT_KEY, ([path] + [p for p in value if p != path])[:MAX_RECENT])


class StartDialog(QtWidgets.QDialog):
    """Sets `choice` to ("new", None), ("browse", None) or ("open", path)
    and accepts; Quit or closing the window rejects."""

    def __init__(self, recent: list[str], title: str = "Open a project", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.choice: tuple[str, str | None] | None = None

        heading = QtWidgets.QLabel("<b>Open a project to start labeling</b>")
        blurb = QtWidgets.QLabel(
            "A project folder keeps your videos, labels, frames and models together.")
        blurb.setWordWrap(True)

        self.new_btn = QtWidgets.QPushButton("New Project…")
        self.open_btn = QtWidgets.QPushButton("Open Project…")
        mark_accent(self.new_btn)
        self.new_btn.clicked.connect(lambda: self._choose("new"))
        self.open_btn.clicked.connect(lambda: self._choose("browse"))
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.new_btn)
        buttons.addWidget(self.open_btn)

        self.recent = QtWidgets.QListWidget()
        for path in recent:
            item = QtWidgets.QListWidgetItem(os.path.basename(path.rstrip("/\\")) or path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.recent.addItem(item)
        self.recent.itemActivated.connect(self._open_recent)
        self.recent_open_btn = QtWidgets.QPushButton("Open Selected")
        self.recent_open_btn.setEnabled(False)
        self.recent.currentItemChanged.connect(
            lambda cur, _prev: self.recent_open_btn.setEnabled(cur is not None))
        self.recent_open_btn.clicked.connect(
            lambda: self._open_recent(self.recent.currentItem()))

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(blurb)
        layout.addLayout(buttons)
        if recent:
            layout.addSpacing(8)
            layout.addWidget(QtWidgets.QLabel("<b>Recent projects</b>"))
            layout.addWidget(self.recent)
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(self.recent_open_btn)
            layout.addLayout(row)
            self.recent.setCurrentRow(0)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.quit_btn)
        layout.addLayout(footer)
        self.resize(460, 380 if recent else 160)

    def _choose(self, kind: str, path: str | None = None) -> None:
        self.choice = (kind, path)
        self.accept()

    def _open_recent(self, item) -> None:
        if item is not None:
            self._choose("open", item.data(QtCore.Qt.ItemDataRole.UserRole))

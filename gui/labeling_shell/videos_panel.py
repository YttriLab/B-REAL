"""Videos panel — the project's registered videos; click one to open it.

Qt successor to the Electron app's per-project video list. Renders the
manifest's `videos` records; a video whose file is gone is greyed and marked
missing (opening it asks where it went). The panel only requests — MainWindow
does the loading and relinking.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import mark_compact


class VideosPanel(QtWidgets.QWidget):
    videoRequested = QtCore.Signal(str)  # video id
    addRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        self.list = QtWidgets.QListWidget()
        self.list.itemClicked.connect(self._on_clicked)
        # A project holds a handful of videos; don't take rows from behaviors/bouts.
        self.list.setMaximumHeight(130)
        self.add_btn = QtWidgets.QPushButton("Add Video…")
        mark_compact(self.add_btn)
        self.add_btn.clicked.connect(self.addRequested)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Videos</b>"))
        head.addStretch(1)
        head.addWidget(self.add_btn)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self.list)

    def set_videos(self, records, current_id=None) -> None:
        self.list.clear()
        for rec in records:
            vid = rec.get("id")
            name = rec.get("name") or vid
            path = rec.get("path") or ""
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, vid)
            item.setToolTip(path)
            if not os.path.exists(path):
                item.setText(f"{name}  (missing)")
                item.setForeground(QtGui.QBrush(QtGui.QColor("#888888")))
            if vid == current_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.list.addItem(item)
            if vid == current_id:
                self.list.setCurrentItem(item)

    def _on_clicked(self, item) -> None:
        self.videoRequested.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

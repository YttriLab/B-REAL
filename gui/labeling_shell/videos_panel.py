"""Videos panel — the project's registered videos; click one to open it.

Qt successor to the Electron app's per-project video list. Renders the
manifest's `videos` records; a video whose file is gone is greyed and marked
missing (opening it asks where it went). Remove acts on the selected row, and
a right-click removes any row without opening it first. The panel only
requests — MainWindow does the loading, relinking and removing.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import mark_compact


class VideosPanel(QtWidgets.QWidget):
    videoRequested = QtCore.Signal(str)  # video id
    addRequested = QtCore.Signal()
    removeRequested = QtCore.Signal(str)  # video id
    locateRequested = QtCore.Signal(str)  # video id: pick a different file for it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        self.list = QtWidgets.QListWidget()
        self.list.itemClicked.connect(self._on_clicked)
        # A project holds a handful of videos; don't take rows from behaviors/bouts.
        self.list.setMaximumHeight(130)
        self.list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.currentItemChanged.connect(
            lambda cur, _prev: self.remove_btn.setEnabled(cur is not None))
        self.add_btn = QtWidgets.QPushButton("Add Video…")
        mark_compact(self.add_btn)
        self.add_btn.clicked.connect(self.addRequested)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        mark_compact(self.remove_btn)
        self.remove_btn.setEnabled(False)
        self.remove_btn.setToolTip("Remove the selected video from the project "
                                   "(the video file itself is not deleted).")
        self.remove_btn.clicked.connect(self._remove_current)

        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Videos</b>"))
        head.addStretch(1)
        head.addWidget(self.add_btn)
        head.addWidget(self.remove_btn)
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
        self.remove_btn.setEnabled(self.list.currentItem() is not None)

    def _on_clicked(self, item) -> None:
        self.videoRequested.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _remove_current(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.removeRequested.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        locate = menu.addAction("Locate File…")
        locate.setToolTip("Point this video at a different file, keeping its labels.")
        remove = menu.addAction("Remove from project…")
        chosen = menu.exec(self.list.viewport().mapToGlobal(pos))
        vid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if chosen is remove:
            self.removeRequested.emit(vid)
        elif chosen is locate:
            self.locateRequested.emit(vid)

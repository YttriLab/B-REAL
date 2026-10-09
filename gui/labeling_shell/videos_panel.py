"""Videos panel — the project's registered videos; click one to open it.

Qt successor to the Electron app's per-project video list. Renders the
manifest's `videos` records; a video whose file is gone is greyed and marked
missing (opening it asks where it went). Remove acts on the selected row, and
a right-click removes any row without opening it first. The panel only
requests — MainWindow does the loading, relinking and removing.

Each row can carry a status line under the name, with a colored dot (the
pipeline's "preprocessed?" here). The panel doesn't know what it means: it
draws the (text, color, tooltip) it is given.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .theme import mark_compact


STATUS_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1  # (text, color) or None


class StatusDelegate(QtWidgets.QStyledItemDelegate):
    """The name as usual, then a smaller status line with a colored dot."""

    DOT = 8

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        if index.data(STATUS_ROLE):
            size.setHeight(size.height() + QtGui.QFontMetrics(self._small(option.font)).height() + 2)
        return size

    @staticmethod
    def _small(font):
        f = QtGui.QFont(font)
        f.setPointSizeF(max(7.0, f.pointSizeF() * 0.95))
        f.setBold(False)
        return f

    def paint(self, painter, option, index):
        status = index.data(STATUS_ROLE)
        if not status:
            return super().paint(painter, option, index)
        text, color = status
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        # Where the style would put the text: right of a check box, if any.
        text_left = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget).left()
        name = opt.text
        opt.text = ""
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        rect = QtCore.QRect(opt.rect.adjusted(0, 2, -4, -2))
        rect.setLeft(max(opt.rect.left() + 6, text_left + 2))
        name_h = QtGui.QFontMetrics(opt.font).height()
        painter.save()
        painter.setFont(opt.font)
        fg = index.data(QtCore.Qt.ItemDataRole.ForegroundRole)  # e.g. a missing file's grey
        selected = bool(opt.state & QtWidgets.QStyle.StateFlag.State_Selected)
        painter.setPen(fg.color() if fg is not None else opt.palette.color(
            QtGui.QPalette.ColorRole.HighlightedText if selected else QtGui.QPalette.ColorRole.Text))
        painter.drawText(QtCore.QRect(rect.left(), rect.top(), rect.width(), name_h),
                         QtCore.Qt.AlignmentFlag.AlignVCenter,
                         QtGui.QFontMetrics(opt.font).elidedText(
                             name, QtCore.Qt.TextElideMode.ElideMiddle, rect.width()))
        small = self._small(opt.font)
        small_h = QtGui.QFontMetrics(small).height()
        y = rect.top() + name_h + 1
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(QtCore.QRectF(rect.left(), y + (small_h - self.DOT) / 2,
                                          self.DOT, self.DOT))
        painter.setFont(small)
        painter.setPen(QtGui.QColor(color))
        x = rect.left() + self.DOT + 5
        painter.drawText(QtCore.QRect(x, y, rect.right() - x, small_h),
                         QtCore.Qt.AlignmentFlag.AlignVCenter,
                         QtGui.QFontMetrics(small).elidedText(
                             text, QtCore.Qt.TextElideMode.ElideRight, rect.right() - x))
        painter.restore()


class VideosPanel(QtWidgets.QWidget):
    MAX_ROWS = 6
    videoRequested = QtCore.Signal(str)  # video id
    addRequested = QtCore.Signal()
    removeRequested = QtCore.Signal(str)  # video id
    locateRequested = QtCore.Signal(str)  # video id: pick a different file for it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        self.list = QtWidgets.QListWidget()
        self.list.setItemDelegate(StatusDelegate(self.list))
        self.list.itemClicked.connect(self._on_clicked)
        # A project holds a handful of videos: fit them (see _fit_height), but
        # never take more than MAX_ROWS rows from behaviors/bouts.
        self.list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                QtWidgets.QSizePolicy.Policy.Fixed)
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

    def set_videos(self, records, current_id=None, statuses=None) -> None:
        """statuses: {video id: (text, color, tooltip)} for the line under each name."""
        statuses = statuses or {}
        self.list.clear()
        for rec in records:
            vid = rec.get("id")
            name = rec.get("name") or vid
            path = rec.get("path") or ""
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, vid)
            item.setToolTip(path)
            status = statuses.get(vid)
            if status:
                text, color, tip = status
                item.setData(STATUS_ROLE, (text, color))
                item.setToolTip(f"{path}\n{tip}" if tip else path)
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
        self._fit_height()

    def _fit_height(self) -> None:
        rows = [self.list.sizeHintForRow(i) for i in range(min(self.list.count(), self.MAX_ROWS))]
        content = sum(rows) if rows else self.list.fontMetrics().height() * 2
        self.list.setFixedHeight(content + 2 * self.list.frameWidth() + 2)

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

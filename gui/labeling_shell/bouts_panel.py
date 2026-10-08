"""Bouts panel — table of recorded bouts with jump/reassign/edit/delete.

Qt successor to renderBouts (core/behaviors.js) + editBoutRow (core/player.js).
Clicking a time cell requests a seek; the behavior column is a combo for
reassignment; Edit opens a small time dialog; Delete removes the bout.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .bouts import LabelModel
from .no_wheel import NoWheelComboBox
from .theme import mark_compact
from .timefmt import format_time, frame_to_ms, ms_to_frame, parse_time


class BoutsPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()
    seekRequested = QtCore.Signal(int)

    def __init__(self, model: LabelModel, fps_getter, parent=None):
        super().__init__(parent)
        self.model = model
        self._fps_getter = fps_getter  # callable -> float
        self._build()
        self.refresh()

    def _build(self) -> None:
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Behavior", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellClicked.connect(self._on_cell_clicked)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Bouts</b>"))
        layout.addWidget(self.table, 1)

    def fps(self) -> float:
        return self._fps_getter() or 30.0

    # ----------------------------------------------------------------- render
    def refresh(self) -> None:
        self.table.setRowCount(0)
        for idx, bout in enumerate(self.model.bouts):
            self.table.insertRow(idx)
            self.table.setItem(idx, 0, QtWidgets.QTableWidgetItem(
                format_time(frame_to_ms(bout.start_frame, self.fps()))))
            self.table.setItem(idx, 1, QtWidgets.QTableWidgetItem(
                format_time(frame_to_ms(bout.end_frame, self.fps()))))

            # NoWheel: this combo sits in a scrolling table, and a plain
            # QComboBox reassigns the bout under the cursor as you scroll past.
            combo = NoWheelComboBox()
            current = 0
            for k, beh in enumerate(self.model.behaviors):
                combo.addItem(beh.name, beh.id)
                if beh.id == bout.behavior_id:
                    current = k
            combo.setCurrentIndex(current)
            combo.currentIndexChanged.connect(
                lambda _i, row=idx, c=combo: self._reassign(row, c.currentData()))
            self.table.setCellWidget(idx, 2, combo)

            actions = QtWidgets.QWidget()
            ah = QtWidgets.QHBoxLayout(actions)
            ah.setContentsMargins(0, 0, 0, 0)
            edit_btn = QtWidgets.QPushButton("Edit")
            del_btn = QtWidgets.QPushButton("Delete")
            mark_compact(edit_btn)
            mark_compact(del_btn)
            edit_btn.clicked.connect(lambda _=False, row=idx: self._edit(row))
            del_btn.clicked.connect(lambda _=False, row=idx: self.delete_bout(row))
            ah.addWidget(edit_btn)
            ah.addWidget(del_btn)
            self.table.setCellWidget(idx, 3, actions)

    # ---------------------------------------------------------------- actions
    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col in (0, 1) and 0 <= row < len(self.model.bouts):
            bout = self.model.bouts[row]
            self.seekRequested.emit(bout.start_frame if col == 0 else bout.end_frame)

    def _reassign(self, row: int, behavior_id) -> None:
        if 0 <= row < len(self.model.bouts) and behavior_id:
            self.model.reassign_bout(row, behavior_id)
            self.changed.emit()

    def delete_bout(self, row: int) -> None:
        if 0 <= row < len(self.model.bouts):
            self.model.remove_bout(row)
            self.refresh()
            self.changed.emit()

    def set_bout_times(self, row: int, start_ms: float, end_ms: float) -> bool:
        """Apply edited times to a bout (min/max, re-sorted). Returns success."""
        if not (0 <= row < len(self.model.bouts)):
            return False
        sf, ef = ms_to_frame(start_ms, self.fps()), ms_to_frame(end_ms, self.fps())
        if sf == ef:
            return False
        self.model.set_bout_frames(row, sf, ef)
        self.refresh()
        self.changed.emit()
        return True

    def _edit(self, row: int) -> None:
        if not (0 <= row < len(self.model.bouts)):
            return
        bout = self.model.bouts[row]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit bout")
        form = QtWidgets.QFormLayout(dlg)
        start = QtWidgets.QLineEdit(format_time(frame_to_ms(bout.start_frame, self.fps())))
        end = QtWidgets.QLineEdit(format_time(frame_to_ms(bout.end_frame, self.fps())))
        form.addRow("Start (MM:SS:mmm)", start)
        form.addRow("End (MM:SS:mmm)", end)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        sm, em = parse_time(start.text()), parse_time(end.text())
        if sm is None or em is None:
            return
        self.set_bout_times(row, sm, em)

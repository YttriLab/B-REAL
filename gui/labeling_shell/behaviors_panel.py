"""Behaviors panel — add / rename / remove / merge behaviors.

Qt successor to the behavior-list rendering in core/behaviors.js. Drives a
LabelModel; emits `changed` after any edit so the host can refresh the bouts
panel (names may have changed) and autosave.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .bouts import LabelModel
from .theme import mark_accent


class BehaviorsPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, model: LabelModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._build()
        self.refresh()

    def _build(self) -> None:
        self.add_edit = QtWidgets.QLineEdit()
        self.add_edit.setPlaceholderText("New behavior name")
        self.add_btn = QtWidgets.QPushButton("Add")
        mark_accent(self.add_btn)
        add_row = QtWidgets.QHBoxLayout()
        add_row.addWidget(self.add_edit, 1)
        add_row.addWidget(self.add_btn)

        self.list = QtWidgets.QListWidget()
        self.merge_btn = QtWidgets.QPushButton("Merge selected")
        self.merge_btn.setEnabled(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Behaviors</b>"))
        layout.addLayout(add_row)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.merge_btn)

        self.add_btn.clicked.connect(self._on_add)
        self.add_edit.returnPressed.connect(self._on_add)
        self.merge_btn.clicked.connect(self._on_merge)

    # --------------------------------------------------------------- actions
    def add_behavior(self, name: str) -> None:
        """Add by name (used by the Add button and tests)."""
        name = name.strip()
        if not name:
            return
        self.model.add_behavior(name)
        self.refresh()
        self.changed.emit()

    def _on_add(self) -> None:
        self.add_behavior(self.add_edit.text())
        self.add_edit.clear()

    def remove_behavior(self, behavior_id) -> None:
        self.model.remove_behavior(behavior_id)
        self.refresh()
        self.changed.emit()

    def _on_merge(self) -> None:
        ids = self._checked_ids()
        if len(ids) < 2:
            return
        pairs = [(self.model.behavior_by_id(i).name, i) for i in ids]
        names = [n for n, _ in pairs]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Merge behaviors", "Keep:", names, 0, False)
        if not ok:
            return
        target_id = next(i for n, i in pairs if n == choice)
        self.model.merge_behaviors(ids, target_id)
        self.refresh()
        self.changed.emit()

    # ----------------------------------------------------------------- render
    def refresh(self) -> None:
        self.list.clear()
        for b in self.model.behaviors:
            item = QtWidgets.QListWidgetItem(self.list)
            row = self._row_widget(b)
            item.setSizeHint(row.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
        self._update_merge_btn()

    def _row_widget(self, behavior) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(4, 2, 4, 2)

        check = QtWidgets.QCheckBox()
        check.setToolTip("Select to merge")
        check.setProperty("behavior_id", behavior.id)
        check.toggled.connect(self._update_merge_btn)

        badge = QtWidgets.QLabel(behavior.key or "-")
        badge.setFixedWidth(20)
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "background:#3c3f41;color:#e6e6e6;border-radius:3px;font-weight:bold;")

        name = QtWidgets.QLineEdit(behavior.name)
        name.editingFinished.connect(lambda bid=behavior.id, e=name: self._commit_rename(bid, e))

        rec = QtWidgets.QLabel("● REC" if self.model.is_recording(behavior.id) else "")
        rec.setStyleSheet("color:#ff5c5c;font-weight:bold;")

        remove = QtWidgets.QPushButton("Remove")
        remove.clicked.connect(lambda _=False, bid=behavior.id: self.remove_behavior(bid))

        h.addWidget(check)
        h.addWidget(badge)
        h.addWidget(name, 1)
        h.addWidget(rec)
        h.addWidget(remove)
        w._merge_check = check  # for _checked_ids
        return w

    def _commit_rename(self, behavior_id, edit: QtWidgets.QLineEdit) -> None:
        if self.model.rename_behavior(behavior_id, edit.text()):
            self.refresh()
            self.changed.emit()
        else:
            b = self.model.behavior_by_id(behavior_id)
            if b:
                edit.setText(b.name)

    def _checked_ids(self) -> list:
        ids = []
        for i in range(self.list.count()):
            w = self.list.itemWidget(self.list.item(i))
            if w is not None and w._merge_check.isChecked():
                ids.append(w._merge_check.property("behavior_id"))
        return ids

    def _update_merge_btn(self) -> None:
        self.merge_btn.setEnabled(len(self._checked_ids()) >= 2)

"""Checklists that toggle wherever you click on a row.

A checkable QListWidget only toggles on its small indicator box; clicking the
name does nothing, which reads as "this list is broken". toggle_on_row_click
makes the whole row the target, without double-toggling a click that Qt
already handled (one on the box itself).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def toggle_on_row_click(lw: QtWidgets.QListWidget) -> None:
    pressed = {}  # the clicked item's check state when the button went down

    def on_pressed(item):
        pressed["item"], pressed["state"] = item, item.checkState()

    def on_clicked(item):
        flags = item.flags()
        if not (flags & QtCore.Qt.ItemFlag.ItemIsUserCheckable
                and flags & QtCore.Qt.ItemFlag.ItemIsEnabled):
            return
        if pressed.get("item") is item and item.checkState() == pressed.get("state"):
            item.setCheckState(QtCore.Qt.CheckState.Unchecked
                               if item.checkState() == QtCore.Qt.CheckState.Checked
                               else QtCore.Qt.CheckState.Checked)
        pressed.clear()

    lw.itemPressed.connect(on_pressed)
    lw.itemClicked.connect(on_clicked)

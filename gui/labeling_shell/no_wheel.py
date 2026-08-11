"""Widgets that don't steal the mouse wheel.

Qt's combo boxes and spin boxes change value on wheel events even when they
don't have focus. Inside a scrolling container that is a data-loss bug rather
than a nicety: scrolling the bouts table past a behavior combo silently
reassigns whichever bout happened to be under the cursor, and scrolling the
Train tab past a spin box silently edits a hyperparameter.

These subclasses ignore the wheel unless the widget has keyboard focus, so the
event propagates to the scroll area instead. Click into a control first and the
wheel works normally on it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class _NoWheelMixin:
    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelComboBox(_NoWheelMixin, QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Without this the widget takes focus on wheel-over and then happily
        # consumes the next scroll.
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


class NoWheelSpinBox(_NoWheelMixin, QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


class NoWheelDoubleSpinBox(_NoWheelMixin, QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

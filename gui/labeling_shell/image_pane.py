"""A QLabel that displays a scaled image without resizing itself to fit it.

A plain QLabel reports its pixmap as its size hint, so the obvious
"scale the pixmap to label.size()" is a feedback loop: the pixmap sets the
label's preferred size, the layout resizes the label, and the next scale uses
the new size. Repeated stepping through preview frames made the panes visibly
oscillate and creep larger (150 -> 164 -> 152 -> 166 ... on a fixed image).

ImagePane breaks the loop by keeping the source pixmap and treating the widget
geometry as the single source of truth: the layout decides the size, the pane
scales into whatever it is given, and the pixmap never feeds back.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class ImagePane(QtWidgets.QLabel):
    def __init__(self, placeholder: str = "—", parent=None):
        super().__init__(parent)
        self._source: QtGui.QPixmap | None = None
        self._placeholder = placeholder
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # A hard floor, not a hint: with an Ignored size policy the layout would
        # otherwise be free to squash the pane to nothing.
        self.setMinimumSize(170, 170)
        # Ignored in both directions: the pixmap must never influence layout,
        # or the scale-to-fit feeds back into the size it scaled against.
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.setText(placeholder)

    def set_image(self, pixmap: QtGui.QPixmap | None) -> None:
        """Show a pixmap (or clear back to the placeholder)."""
        self._source = pixmap
        if pixmap is None or pixmap.isNull():
            self._source = None
            self.setPixmap(QtGui.QPixmap())
            self.setText(self._placeholder)
        else:
            self._rescale()

    def set_message(self, text: str) -> None:
        self._source = None
        self.setPixmap(QtGui.QPixmap())
        self.setText(text)

    def has_image(self) -> bool:
        return self._source is not None

    def _rescale(self) -> None:
        if self._source is None:
            return
        # setPixmap on a size-policy-Ignored widget cannot change our geometry,
        # so this is safe to call from resizeEvent.
        super().setPixmap(self._source.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._rescale()

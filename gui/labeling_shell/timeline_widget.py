"""Uncertainty / misclassification timeline strip.

Qt successor to renderUncertaintyOverlay (core/timeline.js): a horizontal bar
under the video where each x maps to a frame, colored green->red by the model's
per-frame uncertainty, with magenta ticks on misclassified frames and a marker
at the current frame. Click to seek.
"""

from __future__ import annotations


from PySide6 import QtCore, QtGui, QtWidgets


def uncertainty_color(u: float) -> QtGui.QColor:
    """0 -> green, 1 -> red (HSL hue 120..0), matching the JS heatmap."""
    u = max(0.0, min(1.0, u))
    hue = int((1.0 - u) * 120)  # 120=green, 0=red
    return QtGui.QColor.fromHsl(hue, 200, 110)


class UncertaintyTimeline(QtWidgets.QWidget):
    seekRequested = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_frames = 0
        self.al_state = None
        self.threshold = 0.5
        self.current = 0
        self.misclassified: set = set()
        self.setFixedHeight(46)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    # ------------------------------------------------------------- data in
    def set_data(self, al_state, num_frames: int) -> None:
        self.al_state = al_state
        self.num_frames = max(0, int(num_frames))
        self.update()

    def set_threshold(self, t: float) -> None:
        self.threshold = float(t)
        self.update()

    def set_current(self, frame: int) -> None:
        self.current = int(frame)
        self.update()

    def set_misclassified(self, frames) -> None:
        self.misclassified = set(frames)
        self.update()

    # ---------------------------------------------------- frame<->x mapping
    def x_at_frame(self, frame: int) -> int:
        if self.num_frames <= 1:
            return 0
        w = max(1, self.width() - 1)
        return int(round(frame / (self.num_frames - 1) * w))

    def frame_at_x(self, x: int) -> int:
        if self.num_frames <= 1:
            return 0
        w = max(1, self.width() - 1)
        return max(0, min(self.num_frames - 1, int(round(x / w * (self.num_frames - 1)))))

    # ------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:
        if self.num_frames > 0:
            self.seekRequested.emit(self.frame_at_x(int(event.position().x())))

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor("#202020"))
        if self.num_frames > 0 and self.al_state is not None:
            for x in range(w):
                frame = self.frame_at_x(x)
                u = self.al_state.uncertainty(frame)
                if u is not None:
                    p.setPen(uncertainty_color(u))
                    p.drawLine(x, 4, x, h - 8)
            p.setPen(QtGui.QColor("#ff00ff"))
            for f in self.misclassified:
                x = self.x_at_frame(f)
                p.drawLine(x, h - 8, x, h - 1)
        # current-frame marker
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        cx = self.x_at_frame(self.current)
        p.drawLine(cx, 0, cx, h)

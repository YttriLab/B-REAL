"""The learning-rate finder's result: the loss curve, and a rate to train with.

The backend's range test (training_service.find_lr) trains for a couple of
hundred steps while raising the learning rate exponentially and reports the
smoothed loss at each. The useful rates sit where the loss is falling fastest,
before it turns up; the suggestion is one reading of that (lr_valley.py), and
the curve is shown so it can be checked: click anywhere on it to pick another.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from labeling_shell.theme import mark_accent

SUGGESTED_COLOR = "#3fb950"
CURRENT_COLOR = "#8b949e"
PICKED_COLOR = "#e3b341"


def fmt_lr(lr) -> str:
    return "—" if lr is None else f"{lr:.1e}"


class LrCurve(QtWidgets.QWidget):
    """Smoothed loss against log learning rate, with vertical markers."""

    picked = QtCore.Signal(float)
    PAD = (48, 10, 12, 30)  # left, top, right, bottom

    def __init__(self, lrs, losses, parent=None):
        super().__init__(parent)
        pts = [(math.log10(lr), l) for lr, l in zip(lrs, losses)
               if lr > 0 and math.isfinite(l)]
        self.points = pts
        self.markers: dict[str, tuple[float, str, bool]] = {}  # name -> (lr, color, dashed)
        self.setMinimumSize(420, 220)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def set_marker(self, name: str, lr, color: str, dashed: bool = False) -> None:
        if lr:
            self.markers[name] = (float(lr), color, dashed)
        else:
            self.markers.pop(name, None)
        self.update()

    # ------------------------------------------------------------- geometry
    def _ranges(self):
        xs = [p[0] for p in self.points] or [-6.0, -1.0]
        ys = [p[1] for p in self.points] or [0.0, 1.0]
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-9:
            y0, y1 = y0 - 0.5, y1 + 0.5
        return min(xs), max(xs), y0, y1

    def _plot_rect(self) -> QtCore.QRectF:
        l, t, r, b = self.PAD
        return QtCore.QRectF(l, t, max(1, self.width() - l - r), max(1, self.height() - t - b))

    def _x(self, logx: float) -> float:
        x0, x1, _, _ = self._ranges()
        rect = self._plot_rect()
        return rect.left() + (logx - x0) / max(1e-9, x1 - x0) * rect.width()

    def _y(self, loss: float) -> float:
        _, _, y0, y1 = self._ranges()
        rect = self._plot_rect()
        return rect.bottom() - (loss - y0) / (y1 - y0) * rect.height()

    def lr_at_x(self, x: float) -> float:
        x0, x1, _, _ = self._ranges()
        rect = self._plot_rect()
        frac = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
        return 10 ** (x0 + frac * (x1 - x0))

    # --------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:
        if self.points:
            self.picked.emit(self.lr_at_x(event.position().x()))

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self._plot_rect()
        pal = self.palette()
        text = pal.color(QtGui.QPalette.ColorRole.WindowText)
        grid = QtGui.QColor(text)
        grid.setAlpha(40)
        x0, x1, y0, y1 = self._ranges()

        p.setPen(QtGui.QPen(grid, 1))
        p.drawRect(rect)
        p.setFont(QtGui.QFont(p.font().family(), 8))
        for dec in range(math.ceil(x0), math.floor(x1) + 1):  # one tick per decade
            x = self._x(dec)
            p.setPen(QtGui.QPen(grid, 1))
            p.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            p.setPen(text)
            p.drawText(QtCore.QRectF(x - 24, rect.bottom() + 4, 48, 14),
                       QtCore.Qt.AlignmentFlag.AlignHCenter, f"1e{dec}")
        p.setPen(text)
        for v in (y0, (y0 + y1) / 2, y1):
            p.drawText(QtCore.QRectF(0, self._y(v) - 7, self.PAD[0] - 6, 14),
                       QtCore.Qt.AlignmentFlag.AlignRight, f"{v:.2f}")
        p.drawText(QtCore.QRectF(rect.left(), self.height() - 14, rect.width(), 14),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, "learning rate (log)  ·  y: smoothed loss")

        if len(self.points) > 1:
            path = QtGui.QPainterPath(QtCore.QPointF(self._x(self.points[0][0]),
                                                     self._y(self.points[0][1])))
            for lx, l in self.points[1:]:
                path.lineTo(self._x(lx), self._y(l))
            p.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 1.6))
            p.drawPath(path)

        for lr, color, dashed in self.markers.values():
            lx = math.log10(lr)
            if not x0 <= lx <= x1:
                continue
            pen = QtGui.QPen(QtGui.QColor(color), 1.5)
            if dashed:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            p.setPen(pen)
            x = self._x(lx)
            p.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))


class LrFinderDialog(QtWidgets.QDialog):
    """Accepts with `chosen_lr` set to the rate to train with."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Learning-rate finder")
        self.suggested = result.get("suggested_lr")
        self.current = result.get("current_lr")
        self.chosen_lr = self.suggested

        self.curve = LrCurve(result.get("lrs") or [], result.get("losses") or [])
        self.curve.set_marker("current", self.current, CURRENT_COLOR, dashed=True)
        self.curve.set_marker("suggested", self.suggested, SUGGESTED_COLOR)
        self.curve.picked.connect(self._pick)

        def swatch(color, dashed=False):
            return f'<span style="color:{color}">{"┆" if dashed else "│"}</span>'

        if self.suggested is not None:
            what = (f"{swatch(SUGGESTED_COLOR)} <b>Suggested {fmt_lr(self.suggested)}</b>: "
                    "a little past the middle of the longest stretch where the loss "
                    "keeps falling (fastai's valley rule).")
        else:
            what = "<b>No suggestion</b>: the loss never fell over the range tested."
        lines = [what, f"{swatch(CURRENT_COLOR, True)} Current: {fmt_lr(self.current)}."]
        if result.get("lightning_lr") is not None:
            lines.append(f"Lightning's own pick, {fmt_lr(result['lightning_lr'])}, takes the "
                         "steepest slope against the linear rate, so it is drawn to noise "
                         "at the smallest rates.")
        lines.append("Good rates sit on the falling part, before the loss turns up. "
                     "Click the curve to pick another.")
        info = QtWidgets.QLabel("<br>".join(lines))
        info.setWordWrap(True)

        self.use_btn = QtWidgets.QPushButton()
        mark_accent(self.use_btn)
        self.use_btn.clicked.connect(self.accept)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        self._update_use()

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self.use_btn)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.curve, 1)
        layout.addWidget(info)
        layout.addLayout(buttons)
        self.resize(560, 440)

    def _pick(self, lr: float) -> None:
        self.chosen_lr = float(f"{lr:.1e}")  # two significant figures is all a curve supports
        self.curve.set_marker("picked", self.chosen_lr, PICKED_COLOR)
        self._update_use()

    def _update_use(self) -> None:
        self.use_btn.setEnabled(self.chosen_lr is not None)
        self.use_btn.setText(f"Use {fmt_lr(self.chosen_lr)}" if self.chosen_lr else "Use")

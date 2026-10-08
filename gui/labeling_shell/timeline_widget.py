"""Timeline strip under the video: the model's view and yours, frame by frame.

Qt successor to renderUncertaintyOverlay (core/timeline.js), with two lanes:

- **model** (top): per-frame uncertainty, green -> red, or the predicted
  behavior, in the behavior's color (``set_mode``).
- **labels** (bottom): your bouts, in their behaviors' colors — shown even
  before any inference.

Magenta ticks under them mark frames where the prediction disagrees with your
label, and a white line marks the current frame. Click to seek; hover for that
frame's prediction and label. A long video is ~half a million frames over a
few hundred pixels, so each pixel aggregates its frames: the highest
uncertainty in it, and every predicted / labeled segment at least 1 px wide
(background "other" drawn first, so short behaviors stay visible).
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .behavior_colors import OTHER
from .theme import mark_compact

MODES = ("uncertainty", "predicted")


def uncertainty_color(u: float) -> QtGui.QColor:
    """0 -> green, 1 -> red (HSL hue 120..0), matching the JS heatmap."""
    u = max(0.0, min(1.0, u))
    hue = int((1.0 - u) * 120)  # 120=green, 0=red
    return QtGui.QColor.fromHsl(hue, 200, 110)


def next_segment(runs, current: int, forward: bool = True):
    """Start of the next (or previous) segment in `runs` [(start, end, ...)],
    relative to `current`; None when there is none. Backward is like a media
    player's previous: the start of the segment you're in, or, from its first
    frame, the start of the one before."""
    starts = sorted(r[0] for r in runs)
    if forward:
        return next((s for s in starts if s > current), None)
    return next((s for s in reversed(starts) if s < current), None)


class UncertaintyTimeline(QtWidgets.QWidget):
    seekRequested = QtCore.Signal(int)

    MODEL = (3, 21)    # y range of the model lane
    LABELS = (24, 38)  # y range of the labels lane
    TICKS = (40, 47)   # misclassification ticks

    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_frames = 0
        self.al_state = None
        self.threshold = 0.5
        self.current = 0
        self.misclassified: set = set()
        self.mode = "uncertainty"
        self.bouts: list = []
        self.colors: dict = {}  # behavior / class name -> QColor
        self._unc_cache = None  # (key, per-pixel max uncertainty)
        self.setFixedHeight(50)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    # ------------------------------------------------------------- data in
    def set_data(self, al_state, num_frames: int) -> None:
        self.al_state = al_state
        self.num_frames = max(0, int(num_frames))
        self._unc_cache = None
        self.update()

    def set_labels(self, bouts, colors: dict) -> None:
        """Your bouts for the labels lane, and the name -> color map both lanes use."""
        self.bouts = list(bouts)
        self.colors = dict(colors)
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self.mode = mode
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

    def mouseMoveEvent(self, event) -> None:
        if self.num_frames <= 0:
            return
        frame = self.frame_at_x(int(event.position().x()))
        parts = [f"frame {frame}"]
        if self.al_state is not None:
            pred = self.al_state.predicted_name(frame)
            if pred is not None:
                u = self.al_state.uncertainty(frame)
                parts.append(f"predicted {pred}" + (f" (unc {u:.2f})" if u is not None else ""))
        label = next((b.behavior_name for b in self.bouts
                      if b.start_frame <= frame <= b.end_frame), None)
        parts.append(f"labeled {label}" if label else "unlabeled")
        QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), " · ".join(parts), self)

    # ------------------------------------------------------------- paint
    def _span(self, start: int, end: int) -> tuple[int, int]:
        """x and width covering frames start..end, at least 1 px."""
        x0 = self.x_at_frame(start)
        return x0, max(1, self.x_at_frame(end) - x0 + 1)

    def _max_uncertainty_per_px(self, w: int):
        """Highest uncertainty among each pixel's frames (NaN where none)."""
        key = (id(self.al_state), w, self.num_frames)
        if self._unc_cache is None or self._unc_cache[0] != key:
            unc = self.al_state.arrays()[0][:self.num_frames]
            per_px = np.full(w, np.nan, np.float32)
            if len(unc) and self.num_frames > 1:
                px = np.round(np.arange(len(unc)) / (self.num_frames - 1) * max(1, w - 1))
                firsts = np.clip(np.searchsorted(px, np.arange(w)), 0, len(unc) - 1)
                maxes = np.maximum.reduceat(np.where(np.isnan(unc), -1.0, unc), firsts)
                per_px = np.where(maxes < 0, np.nan, maxes).astype(np.float32)
            self._unc_cache = (key, per_px)
        return self._unc_cache[1]

    def _paint_runs(self, p, runs, y0, y1) -> None:
        """runs: [(start, end, name)]; "other" first so it can't hide the rest."""
        is_other = [str(n).strip().lower() == "other" for _s, _e, n in runs]
        for background in (True, False):
            for (s, e, name), other in zip(runs, is_other):
                if other == background:
                    x, wd = self._span(s, e)
                    p.fillRect(x, y0, wd, y1 - y0, self.colors.get(name) or QtGui.QColor(OTHER))

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        w = self.width()
        p.fillRect(self.rect(), QtGui.QColor("#202020"))
        if self.num_frames > 0:
            for y0, y1 in (self.MODEL, self.LABELS):
                p.fillRect(0, y0, w, y1 - y0, QtGui.QColor("#2a2a2a"))
            if self.al_state is not None and self.al_state.by_index:
                if self.mode == "predicted":
                    names = self.al_state.class_names
                    runs = [(s, e, names[c] if 0 <= c < len(names) else c)
                            for s, e, c in self.al_state.predicted_runs() if s < self.num_frames]
                    self._paint_runs(p, runs, *self.MODEL)
                else:
                    for x, u in enumerate(self._max_uncertainty_per_px(w)):
                        if not np.isnan(u):
                            p.setPen(uncertainty_color(float(u)))
                            p.drawLine(x, self.MODEL[0], x, self.MODEL[1] - 1)
            self._paint_runs(p, [(b.start_frame, b.end_frame, b.behavior_name)
                                 for b in self.bouts], *self.LABELS)
            p.setPen(QtGui.QColor("#ff00ff"))
            for f in self.misclassified:
                x = self.x_at_frame(f)
                p.drawLine(x, self.TICKS[0], x, self.TICKS[1])
        # current-frame marker
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        cx = self.x_at_frame(self.current)
        p.drawLine(cx, 0, cx, self.height())


class TimelineControls(QtWidgets.QWidget):
    """Under the timeline: what the model lane shows, jumping between segments
    of one behavior (in your labels or in the predictions), and a color key."""

    modeChanged = QtCore.Signal(str)
    jumpRequested = QtCore.Signal(str, str, bool)  # behavior name, "labels"|"predicted", forward

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("Uncertainty", "uncertainty")
        self.mode.addItem("Predicted behavior", "predicted")
        self.mode.setToolTip("What the top lane of the timeline shows; the bottom lane is your labels.")
        self.mode.currentIndexChanged.connect(lambda _i: self.modeChanged.emit(self.mode.currentData()))
        self.behavior = QtWidgets.QComboBox()
        self.behavior.setToolTip("Behavior to jump between")
        self.source = QtWidgets.QComboBox()
        self.source.addItem("in labels", "labels")
        self.source.addItem("in predictions", "predicted")
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        for b, fwd, tip in ((self.prev_btn, False, "Previous segment  ["),
                            (self.next_btn, True, "Next segment  ]")):
            mark_compact(b)
            b.setFixedWidth(36)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, f=fwd: self.request_jump(f))
        self.legend = QtWidgets.QLabel()
        self.legend.setWordWrap(True)
        self.legend.setProperty("muted", "true")

        self.behavior.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.behavior.setMinimumContentsLength(8)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(QtWidgets.QLabel("Timeline:"))
        top.addWidget(self.mode)
        top.addWidget(self.legend, 1)
        jump = QtWidgets.QHBoxLayout()
        jump.setContentsMargins(0, 0, 0, 0)
        jump.addWidget(QtWidgets.QLabel("Jump to"))
        jump.addWidget(self.behavior, 1)
        jump.addWidget(self.source)
        jump.addWidget(self.prev_btn)
        jump.addWidget(self.next_btn)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addLayout(jump)

    def set_mode(self, mode: str) -> None:
        self.mode.setCurrentIndex(max(0, self.mode.findData(mode)))

    def set_behaviors(self, names, colors: dict) -> None:
        current = self.behavior.currentText()
        self.behavior.blockSignals(True)
        self.behavior.clear()
        self.behavior.addItems(list(names))
        if current in names:
            self.behavior.setCurrentText(current)
        self.behavior.blockSignals(False)
        self.legend.setText("&nbsp;&nbsp;".join(
            f'<span style="color:{colors[n].name()}">■</span>&nbsp;{n}'
            for n in names if n in colors))

    def request_jump(self, forward: bool) -> None:
        if self.behavior.currentText():
            self.jumpRequested.emit(self.behavior.currentText(), self.source.currentData(), forward)

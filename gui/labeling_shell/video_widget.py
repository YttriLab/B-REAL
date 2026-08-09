"""Qt video display + transport, on top of the PyAV VideoReader.

`VideoView` draws one frame scaled-to-fit with letterboxing (the Qt equivalent
of the Electron canvas `object-fit: contain`). `PlayerWidget` adds transport:
play/pause via a QTimer, single-frame stepping, a scrubber, an FPS lock, and the
same keyboard shortcuts as the Electron app (Space, Left/Right).

Overlays (HUD, uncertainty timeline, bout marks) are deliberately *not* baked
into the frame pixels here -- later phases layer them as separate Qt items, the
same separation the JS app uses (DOM/z-index over the video).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .qt_image import ndarray_to_qimage
from .timefmt import format_time
from .video_reader import VideoReader


class VideoView(QtWidgets.QWidget):
    """Draws a single RGB frame, aspect-fit and centered on a black field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("black"))
        self.setPalette(pal)

    def set_frame(self, arr) -> None:
        self._image = ndarray_to_qimage(arr) if arr is not None else None
        self.update()

    @property
    def current_image(self) -> QtGui.QImage | None:
        return self._image

    def _fit_rect(self) -> QtCore.QRectF:
        img = self._image
        ww, wh = self.width(), self.height()
        if img is None or img.width() == 0 or img.height() == 0:
            return QtCore.QRectF(0, 0, ww, wh)
        scale = min(ww / img.width(), wh / img.height())
        w, h = img.width() * scale, img.height() * scale
        return QtCore.QRectF((ww - w) / 2, (wh - h) / 2, w, h)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("black"))
        if self._image is not None:
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(self._fit_rect(), self._image)


class PlayerWidget(QtWidgets.QWidget):
    """Video view + transport controls, backed by a VideoReader."""

    frameChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reader: VideoReader | None = None
        self.index = 0

        self.view = VideoView()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.info = QtWidgets.QLabel("no video loaded")
        self.info.setMinimumWidth(260)

        self.lock_chk = QtWidgets.QCheckBox("Lock FPS")
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 240.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setEnabled(False)

        controls = QtWidgets.QHBoxLayout()
        for w in (self.play_btn, self.prev_btn, self.next_btn):
            controls.addWidget(w)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.lock_chk)
        controls.addWidget(self.fps_spin)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.info)
        layout.addLayout(controls)

        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        self.slider.valueChanged.connect(self._on_slider)
        self.lock_chk.toggled.connect(self._on_lock)
        self.fps_spin.valueChanged.connect(self._on_lock)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ load
    def load(self, path: str) -> None:
        self.pause()
        if self.reader is not None:
            self.reader.close()
        self.reader = VideoReader(path)
        self.index = 0
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.reader.num_frames - 1))
        self.fps_spin.setValue(round(self.reader.detected_fps, 3))
        self._show(0)

    # -------------------------------------------------------------- transport
    def toggle_play(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    def play(self) -> None:
        if self.reader is None or self._timer.isActive():
            return
        self._timer.start(max(1, int(round(1000.0 / self.reader.fps))))
        self.play_btn.setText("Pause")

    def pause(self) -> None:
        self._timer.stop()
        self.play_btn.setText("Play")

    def _on_tick(self) -> None:
        if self.reader is None or not self._timer.isActive():
            return
        if self.index >= self.reader.num_frames - 1:
            self.pause()
            return
        self._show(self.index + 1)

    def step(self, delta: int) -> None:
        if self.reader is not None:
            self._show(self.index + delta)

    def _on_slider(self, value: int) -> None:
        if self.reader is not None and value != self.index:
            self._show(value)

    def _on_lock(self, *_) -> None:
        if self.reader is None:
            return
        locked = self.lock_chk.isChecked()
        self.fps_spin.setEnabled(locked)
        self.reader.set_fps_override(self.fps_spin.value() if locked else None)
        if self._timer.isActive():  # re-arm at the new rate
            self.play()
        self._update_info()

    # ------------------------------------------------------------------ paint
    def _show(self, index: int) -> None:
        if self.reader is None:
            return
        index = max(0, min(index, self.reader.num_frames - 1))
        self.index = index
        self.view.set_frame(self.reader.get_frame(index))
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._update_info()
        self.frameChanged.emit(index)

    def _update_info(self) -> None:
        r = self.reader
        if r is None:
            self.info.setText("no video loaded")
            return
        self.info.setText(
            f"frame {self.index} / {r.num_frames - 1}   "
            f"{format_time(r.frame_to_ms(self.index))}   "
            f"{r.fps:.3f} fps{' (locked)' if r._fps_override else ''}"
        )

    # --------------------------------------------------------------- keyboard
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space:
            self.toggle_play()
        elif key == QtCore.Qt.Key.Key_Left:
            self.step(-1)
        elif key == QtCore.Qt.Key.Key_Right:
            self.step(+1)
        else:
            super().keyPressEvent(event)

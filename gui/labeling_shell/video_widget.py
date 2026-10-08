"""Qt video display + transport, on top of the PyAV VideoReader.

`VideoView` draws one frame scaled-to-fit with letterboxing (the Qt equivalent
of the Electron canvas `object-fit: contain`). `PlayerWidget` adds transport:
play/pause via a QTimer, single-frame stepping, a scrubber, Snap Back, and the
same keyboard shortcuts as the Electron app (Space, Left/Right).

Overlays (HUD, uncertainty timeline, bout marks) are deliberately *not* baked
into the frame pixels here -- later phases layer them as separate Qt items, the
same separation the JS app uses (DOM/z-index over the video).
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .qt_image import ndarray_to_qimage
from .theme import mark_ghost
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


class _LoadCancelled(Exception):
    pass


class _ReaderLoader(QtCore.QThread):
    """Open a VideoReader off the UI thread.

    Opening scans the whole file to index its frames — seconds for a
    multi-gigabyte video, longer over a network share — which would otherwise
    freeze the window with nothing painted.
    """

    progress = QtCore.Signal(float)
    done = QtCore.Signal(object)  # VideoReader
    failed = QtCore.Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _report(self, fraction: float) -> None:
        if self._cancel:
            raise _LoadCancelled
        self.progress.emit(fraction)

    def run(self) -> None:
        try:
            reader = VideoReader(self.path, progress=self._report)
        except _LoadCancelled:
            return
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if self._cancel:
            reader.close()
            return
        self.done.emit(reader)


class LoadingOverlay(QtWidgets.QWidget):
    """Dimmed panel over the video while a file is opening."""

    def __init__(self, parent):
        super().__init__(parent)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color: white; background: transparent;")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(260)
        box = QtWidgets.QVBoxLayout(self)
        box.addStretch(1)
        box.addWidget(self.label)
        box.addWidget(self.bar, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        box.addStretch(1)
        parent.installEventFilter(self)
        self.hide()

    def start(self, name: str) -> None:
        self.label.setText(f"Loading {name}…")
        self.bar.setValue(0)
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()

    def set_progress(self, fraction: float) -> None:
        self.bar.setValue(int(fraction * 1000))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.parentWidget() and event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(obj.rect())
        return False

    def paintEvent(self, event) -> None:
        QtGui.QPainter(self).fillRect(self.rect(), QtGui.QColor(0, 0, 0, 170))


class PlayerWidget(QtWidgets.QWidget):
    """Video view + transport controls, backed by a VideoReader."""

    frameChanged = QtCore.Signal(int)
    # The player has no view of the label model, so it asks the host to work out
    # where "back" is (the end of the last bout) and seek there.
    snapBackRequested = QtCore.Signal()
    loaded = QtCore.Signal(str)           # path, once its reader is installed
    loadFailed = QtCore.Signal(str, str)  # path, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reader: VideoReader | None = None
        self.index = 0
        self._loader: _ReaderLoader | None = None
        self._loaders: set = set()  # incl. cancelled ones still winding down

        self.view = VideoView()
        self.overlay = LoadingOverlay(self.view)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.snap_btn = QtWidgets.QPushButton("Snap Back")
        self.snap_btn.setToolTip(
            "Jump to the end of the last labeled bout (or the start of the "
            "video if there are none) — where you left off labeling.")
        mark_ghost(self.snap_btn)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.info = QtWidgets.QLabel("no video loaded")
        self.info.setMinimumWidth(260)

        controls = QtWidgets.QHBoxLayout()
        for w in (self.play_btn, self.prev_btn, self.next_btn, self.snap_btn):
            controls.addWidget(w)
        controls.addWidget(self.slider, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.info)
        layout.addLayout(controls)

        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        self.snap_btn.clicked.connect(self.snapBackRequested)
        self.slider.valueChanged.connect(self._on_slider)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ load
    @property
    def loading(self) -> bool:
        return self._loader is not None

    def load(self, path: str) -> None:
        """Open `path` on this thread (blocks until indexed)."""
        self._cancel_load()
        self._install(VideoReader(path), path)

    def load_async(self, path: str) -> None:
        """Open `path` on a worker thread behind a loading overlay.

        The current video stays up (and covered) until the new one is ready.
        A later load supersedes this one. Emits `loaded` or `loadFailed`.
        """
        self.pause()
        self._cancel_load()
        loader = _ReaderLoader(path)
        loader.progress.connect(self.overlay.set_progress)
        loader.done.connect(lambda r, ld=loader: self._on_loader_done(ld, r))
        loader.failed.connect(lambda msg, ld=loader: self._on_loader_failed(ld, msg))
        loader.finished.connect(lambda ld=loader: self._loaders.discard(ld))
        self._loader = loader
        self._loaders.add(loader)
        self.overlay.start(os.path.basename(path))
        self._set_transport_enabled(False)
        loader.start()

    def unload(self) -> None:
        """Close the current video (and any load in flight); show nothing."""
        self.pause()
        self._cancel_load()
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.index = 0
        self.view.set_frame(None)
        self.slider.setEnabled(False)
        self._update_info()

    def shutdown(self) -> None:
        """Stop any background load; call before the widget goes away."""
        self._cancel_load()
        for loader in list(self._loaders):
            loader.wait()

    def _cancel_load(self) -> None:
        if self._loader is not None:
            self._loader.cancel()
            self._loader = None
            self.overlay.hide()
            self._set_transport_enabled(True)

    def _on_loader_done(self, loader, reader) -> None:
        if loader is not self._loader:  # superseded while it finished
            reader.close()
            return
        self._loader = None
        self.overlay.hide()
        self._set_transport_enabled(True)
        self._install(reader, loader.path)

    def _on_loader_failed(self, loader, msg: str) -> None:
        if loader is not self._loader:
            return
        self._loader = None
        self.overlay.hide()
        self._set_transport_enabled(True)
        self.loadFailed.emit(loader.path, msg)

    def _set_transport_enabled(self, on: bool) -> None:
        for w in (self.play_btn, self.prev_btn, self.next_btn, self.snap_btn):
            w.setEnabled(on)
        self.slider.setEnabled(on and self.reader is not None)

    def _install(self, reader: VideoReader, path: str) -> None:
        self.pause()
        if self.reader is not None:
            self.reader.close()
        self.reader = reader
        self.index = 0
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self.reader.num_frames - 1))
        self._show(0)
        self.loaded.emit(path)

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

    def set_fps_override(self, fps: float | None) -> None:
        """Force a playback/timing FPS (None restores the detected rate).

        The UI control for this was removed — the detected rate is right for
        every file PyAV reads — but the capability stays for odd sources with a
        wrong or missing header.
        """
        if self.reader is None:
            return
        self.reader.set_fps_override(fps)
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
        if self.loading:
            return
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space:
            self.toggle_play()
        elif key == QtCore.Qt.Key.Key_Left:
            self.step(-1)
        elif key == QtCore.Qt.Key.Key_Right:
            self.step(+1)
        else:
            super().keyPressEvent(event)

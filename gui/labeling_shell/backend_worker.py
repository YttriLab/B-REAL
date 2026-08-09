"""Run backend calls (including SSE streams) off the Qt UI thread.

The BackendClient is synchronous and its *_status methods block while streaming
SSE. These QThread wrappers keep the UI responsive: StreamWorker emits one
`event` per SSE dict; CallWorker runs a one-shot call and emits `done`.
"""

from __future__ import annotations

from typing import Callable, Iterator

from PySide6 import QtCore


class StreamWorker(QtCore.QThread):
    """Iterate an SSE generator on a background thread, emitting each event."""

    event = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    finished_ok = QtCore.Signal()

    def __init__(self, make_iter: Callable[[], Iterator[dict]], parent=None):
        super().__init__(parent)
        self._make_iter = make_iter
        self._stop = False

    def run(self) -> None:
        try:
            for ev in self._make_iter():
                if self._stop:
                    break
                self.event.emit(ev)
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")

    def request_stop(self) -> None:
        self._stop = True


class CallWorker(QtCore.QThread):
    """Run a one-shot backend call on a background thread."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn: Callable[[], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")

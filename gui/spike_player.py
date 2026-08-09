"""Standalone Phase 0 spike: a minimal frame-accurate video player.

    .venv-gui/Scripts/python.exe gui/spike_player.py [video_path]

With no argument it generates a synthetic index-marked sample so you can watch
frame-accurate stepping (each frame shows its own index) immediately. Use the
scrubber, Prev/Next, Space to play/pause, and Left/Right to step one frame.

This exists to eyeball smoothness/behaviour; correctness is covered headlessly by
gui/tests/test_video_reader.py and gui/tests/test_qt_display.py.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.video_widget import PlayerWidget  # noqa: E402


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frame-accurate video spike — PySide6 + PyAV")
        self.player = PlayerWidget()
        self.setCentralWidget(self.player)
        toolbar = self.addToolBar("main")
        act = toolbar.addAction("Open…")
        act.triggered.connect(self.open_file)
        self.resize(900, 640)

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)")
        if path:
            self.player.load(path)


def _sample() -> str:
    from tools.index_video import write_index_video
    path = os.path.join(tempfile.gettempdir(), "spike_sample.mp4")
    write_index_video(path, 300)
    return path


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.player.load(sys.argv[1] if len(sys.argv) > 1 else _sample())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

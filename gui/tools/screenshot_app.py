"""Render MainWindow to a PNG via QWidget.grab() — works offscreen, no display.

    QT_QPA_PLATFORM=offscreen .venv-gui/Scripts/python.exe gui/tools/screenshot_app.py [out.png]

Loads a synthetic sample video and adds a couple of behaviors/bouts so panels
aren't empty, for a representative screenshot.
"""

from __future__ import annotations

import os
import sys
import tempfile

# No QT_QPA_PLATFORM override: the offscreen plugin's font backend renders
# glyphs as tofu boxes. The native platform has real font rendering, and
# QWidget.grab() paints through Qt's own backing store — it doesn't require
# the window to actually be shown on screen (we skip .show()).

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtWidgets  # noqa: E402

from app import MainWindow  # noqa: E402
from labeling_shell.bouts import Bout  # noqa: E402
from labeling_shell.theme import apply_theme  # noqa: E402
from tools.index_video import write_index_video  # noqa: E402


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "screenshot.png"

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app)

    win = MainWindow()
    win.resize(1320, 820)

    sample = os.path.join(tempfile.gettempdir(), "screenshot_sample.mp4")
    write_index_video(sample, 300, w=640, h=480, fps=30)
    win.load_video(sample)

    a = win.behaviors_panel.model.add_behavior("groom")
    b = win.behaviors_panel.model.add_behavior("rear")
    win.behaviors_panel.model.add_behavior("walk")
    win.behaviors_panel.model.insert_bout_sorted(Bout(a.id, a.name, 10, 45))
    win.behaviors_panel.model.insert_bout_sorted(Bout(b.id, b.name, 60, 90))
    win.behaviors_panel.model.insert_bout_sorted(Bout(a.id, a.name, 150, 200))
    win._seek(75)
    win._refresh_all()
    if len(sys.argv) > 2 and sys.argv[2] == "pipeline":
        win.tabs.setCurrentWidget(win.pipeline)
        win.pipeline.setCurrentWidget(win.pipeline.train)

    win.ensurePolished()
    win.adjustSize()
    win.layout() and win.layout().activate()
    app.processEvents()
    pixmap = win.grab()
    pixmap.save(out_path)
    print(f"saved {out_path} ({pixmap.width()}x{pixmap.height()})")


if __name__ == "__main__":
    main()

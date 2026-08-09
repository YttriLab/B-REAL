"""RealtimeASOID desktop app (PySide6) — main window.

The all-Python successor to the Electron shell. Left: frame-accurate video
player. Right: behaviors + bouts panels (and, in later phases, pipeline tabs).
Keyboard: 1-0 toggle behavior bouts at the current frame, Space play/pause,
Left/Right step, F fullscreen — same shortcuts as the Electron app.

    .venv-gui/Scripts/python.exe gui/app.py [video]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from labeling_shell.behaviors_panel import BehaviorsPanel  # noqa: E402
from labeling_shell.bouts import KEY_ORDER, LabelModel  # noqa: E402
from labeling_shell.bouts_panel import BoutsPanel  # noqa: E402
from labeling_shell.csv_io import (  # noqa: E402
    generate_bouts_csv, generate_training_label_csv, parse_csv_and_import,
)
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402
from labeling_shell.video_widget import PlayerWidget  # noqa: E402

EDITABLE = (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealtimeASOID — Labeler")
        self.model = LabelModel()
        self.project: ProjectStore | None = None
        self.current_video_id: str | None = None

        self.player = PlayerWidget()
        self.player.frameChanged.connect(self._on_frame)
        self.behaviors_panel = BehaviorsPanel(self.model)
        self.bouts_panel = BoutsPanel(self.model, self._fps)
        self.behaviors_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.seekRequested.connect(self._seek)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(self.behaviors_panel, 1)
        rl.addWidget(self.bouts_panel, 2)

        split = QtWidgets.QSplitter()
        split.addWidget(self.player)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)

        self._build_menu()
        self.resize(1280, 800)
        QtWidgets.QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        m = self.menuBar()
        fm = m.addMenu("&File")
        fm.addAction("Open Video…", self.open_video)
        fm.addSeparator()
        fm.addAction("Import Labels (CSV)…", self.import_csv)
        fm.addAction("Export Bout CSV…", self.export_bout_csv)
        fm.addAction("Export Training CSV…", self.export_training_csv)
        fm.addSeparator()
        fm.addAction("New Project…", self.new_project)
        fm.addAction("Open Project…", self.open_project)

    # --------------------------------------------------------------- helpers
    def _fps(self) -> float:
        return self.player.reader.fps if self.player.reader else 30.0

    def _rebind_model(self) -> None:
        self.behaviors_panel.model = self.model
        self.bouts_panel.model = self.model

    def _refresh_all(self) -> None:
        self.behaviors_panel.refresh()
        self.bouts_panel.refresh()

    def _seek(self, frame: int) -> None:
        self.player._show(frame)

    def _on_frame(self, _idx: int) -> None:
        pass  # Phase 4 will use this to highlight the current bout / HUD

    def _on_model_changed(self) -> None:
        self.behaviors_panel.refresh()
        self.bouts_panel.refresh()
        self._autosave()

    def _autosave(self) -> None:
        if self.project and self.current_video_id:
            self.project.save_video_labels(self.current_video_id, self.model)

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    # ------------------------------------------------------------------ video
    def open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        self._autosave()  # persist the outgoing video first
        self.player.load(path)
        name = Path(path).name
        self.current_video_id = safe_id(name)
        if self.project:
            self.model = self.project.load_into_model(self.current_video_id)
            self._rebind_model()
            self.project.register_video(
                self.current_video_id, name, path, self.player.reader.num_frames)
        self._refresh_all()
        self._status(f"Loaded {name} — {self.player.reader.num_frames} frames")

    # --------------------------------------------------------------- keyboard
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            focus = QtWidgets.QApplication.focusWidget()
            if not isinstance(focus, EDITABLE):
                if self._handle_key(event.text(), event.key()):
                    return True
        return super().eventFilter(obj, event)

    def _handle_key(self, text: str, key: int) -> bool:
        if text in KEY_ORDER:
            self._toggle_behavior(text)
            return True
        if key == QtCore.Qt.Key.Key_Space:
            self.player.toggle_play()
            return True
        if key == QtCore.Qt.Key.Key_Left:
            self.player.step(-1)
            return True
        if key == QtCore.Qt.Key.Key_Right:
            self.player.step(+1)
            return True
        if key == QtCore.Qt.Key.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return True
        return False

    def _toggle_behavior(self, key: str) -> None:
        if self.player.reader is None:
            self._status("Load a video first.")
            return
        res = self.model.toggle(key, self.player.index)
        if res.action == "no_behavior":
            return
        if res.action == "started":
            self._status(f"Started {res.behavior.name} @ frame {self.player.index}")
        elif res.action == "ended":
            self._status(f"Ended {res.behavior.name} "
                         f"[{res.bout.start_frame}–{res.bout.end_frame}]")
        elif res.action == "zero_length":
            self._status("Zero-length bout skipped.")
        self._refresh_all()
        self._autosave()

    # ------------------------------------------------------------------- CSV
    def import_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import labels", "", "CSV (*.csv);;All files (*)")
        if not path:
            return
        n = parse_csv_and_import(Path(path).read_text(encoding="utf-8"), self.model, self._fps())
        self._refresh_all()
        self._autosave()
        self._status(f"Imported {n} bouts.")

    def export_bout_csv(self) -> None:
        if not self.model.bouts:
            self._status("No bouts to export.")
            return
        if self.model.find_overlapping_bouts() is not None:
            QtWidgets.QMessageBox.warning(
                self, "Overlap", "Cannot export: some bouts overlap.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export bout CSV", "bouts.csv", "CSV (*.csv)")
        if path:
            Path(path).write_text(generate_bouts_csv(self.model.bouts, self._fps()),
                                  encoding="utf-8")
            self._status(f"Saved {path}")

    def export_training_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export training CSV", "labels.csv", "CSV (*.csv)")
        if path:
            Path(path).write_text(generate_training_label_csv(self.model.bouts),
                                  encoding="utf-8")
            self._status(f"Saved {path}")

    # --------------------------------------------------------------- project
    def new_project(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not d:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Project name", "Name:")
        if not ok:
            return
        self.project = ProjectStore(d)
        self.project.create(name or Path(d).name)
        self._status(f"Project created at {d}")

    def open_project(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder")
        if not d:
            return
        store = ProjectStore(d)
        if not store.exists():
            QtWidgets.QMessageBox.warning(self, "Not a project", "No project.json found.")
            return
        self.project = store
        from labeling_shell.bouts import Behavior
        self.model = LabelModel()
        self.model.behaviors = [Behavior(**b) for b in store.load_manifest().get("behaviors", [])]
        self.current_video_id = None
        self._rebind_model()
        self._refresh_all()
        self._status(f"Opened project {store.load_manifest().get('name')}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    if len(sys.argv) > 1:
        win.load_video(sys.argv[1])
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

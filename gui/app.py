"""B-REAL desktop app (PySide6) — main window.

The all-Python successor to the Electron shell. Left: frame-accurate video
player. Right: a tabbed panel — "Label" (behaviors + bouts) plus the pipeline
tabs (Preprocess / Train / Export), backed by the FastAPI backend.

Keyboard: 1-0 toggle behavior bouts at the current frame, Space play/pause,
Left/Right step, F fullscreen — same shortcuts as the Electron app.

    pixi run -e gui python gui/app.py [video]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from labeling_shell.backend import Backend, BackendConfig  # noqa: E402
from labeling_shell.backend_worker import CallWorker  # noqa: E402
from labeling_shell.behaviors_panel import BehaviorsPanel  # noqa: E402
from labeling_shell.bouts import KEY_ORDER, Behavior, LabelModel  # noqa: E402
from labeling_shell.bouts_panel import BoutsPanel  # noqa: E402
from labeling_shell.csv_io import (  # noqa: E402
    generate_bouts_csv, generate_training_label_csv, parse_csv_and_import,
)
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402
from labeling_shell.active_learning import hud_info  # noqa: E402
from labeling_shell.theme import apply_theme  # noqa: E402
from labeling_shell.timeline_widget import UncertaintyTimeline  # noqa: E402
from labeling_shell.video_widget import PlayerWidget  # noqa: E402
from plugins.pipeline_panel import PipelinePanel  # noqa: E402

EDITABLE = (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
            QtWidgets.QPlainTextEdit)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, autostart_backend: bool = True):
        super().__init__()
        self.setWindowTitle("B-REAL — Labeler")
        self.model = LabelModel()
        self.project: ProjectStore | None = None
        self.current_video_id: str | None = None
        self._video_path: str | None = None
        self.backend: Backend | None = None
        self._bg_workers: set = set()
        self._al_state = None
        self.al_threshold = 0.5

        self.player = PlayerWidget()
        self.player.frameChanged.connect(self._on_frame)
        self.player.snapBackRequested.connect(self.snap_back)
        self.behaviors_panel = BehaviorsPanel(self.model)
        self.bouts_panel = BoutsPanel(self.model, self._fps)
        self.behaviors_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.seekRequested.connect(self._seek)
        self.pipeline = PipelinePanel(self)  # MainWindow is the pipeline ctx

        label_tab = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(label_tab)
        ll.addWidget(self.behaviors_panel, 1)
        ll.addWidget(self.bouts_panel, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(label_tab, "Label")
        self.tabs.addTab(self.pipeline, "Pipeline")

        self.timeline = UncertaintyTimeline()
        self.timeline.seekRequested.connect(self._seek)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.player, 1)
        lv.addWidget(self.timeline)

        # On-video HUD (prediction / confidence / label), overlaid on the frame.
        self.hud = QtWidgets.QLabel(self.player.view)
        self.hud.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud.hide()

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)

        self._build_menu()
        self.resize(1320, 820)
        QtWidgets.QApplication.instance().installEventFilter(self)

        # The pipeline tabs are useless without a backend and every session
        # began by opening the Backend menu, so just start one. Deferred to the
        # event loop so the window paints first rather than appearing to hang
        # while uvicorn boots; the spawn itself is on a worker thread.
        # Tests and the screenshot tool pass False — they pump events, and
        # spawning uvicorn is not something a screenshot should do.
        if autostart_backend:
            QtCore.QTimer.singleShot(0, self.connect_local)

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

        bm = m.addMenu("&Backend")
        bm.addAction("Connect Local…", self.connect_local)
        bm.addAction("Connect Remote…", self.connect_remote)
        bm.addAction("Disconnect", self.disconnect_backend)

    # --------------------------------------------------------------- helpers
    def _fps(self) -> float:
        return self.player.reader.fps if self.player.reader else 30.0

    def _rebind_model(self) -> None:
        self.behaviors_panel.model = self.model
        self.bouts_panel.model = self.model

    def _refresh_all(self) -> None:
        self.behaviors_panel.refresh()
        self.bouts_panel.refresh()
        self.pipeline.refresh()

    def _seek(self, frame: int) -> None:
        self.player._show(frame)

    def snap_back(self) -> None:
        """Jump to where labeling left off: the end of the latest bout.

        Ported from the Electron trackbar's Snap Back button — "latest" is by
        end frame, not list order, since bouts are sorted by start and a long
        early bout can outlast a later short one. No bouts means frame 0.
        """
        if self.player.reader is None:
            self._status("Load a video first.")
            return
        target = max((b.end_frame for b in self.model.bouts), default=0)
        self.player.pause()
        self._seek(target)
        self._status(f"Snapped to frame {target}.")

    def _on_frame(self, idx: int) -> None:
        self.timeline.set_current(idx)
        self._update_hud(idx)

    def _on_model_changed(self) -> None:
        self._refresh_all()
        self._recompute_misclassified()
        self._autosave()

    def _autosave(self) -> None:
        if self.project and self.current_video_id:
            self.project.save_video_labels(self.current_video_id, self.model)

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    # ---- pipeline context (duck-typed for PipelinePanel) ------------------
    def client(self):
        return self.backend.client if self.backend else None

    def video_path(self):
        return self._video_path

    def fps(self):
        return self._fps()

    def bouts(self):
        return self.model.bouts

    def behaviors(self):
        return self.model.behaviors

    def frames_per_group(self):
        return self.pipeline.train.frames_per_group.value()

    def current_frames_dir(self):
        if self.project and self.current_video_id:
            return str(self.project.frames_dir(self.current_video_id))
        return None

    def current_bg_path(self):
        if self.project and self.current_video_id:
            p = self.project.background_path(self.current_video_id)
            return str(p) if p.exists() else None
        return None

    def label_paths(self):
        if self.project and self.current_video_id:
            p = self.project.label_csv_path(self.current_video_id)
            return [str(p)] if p.exists() else []
        return []

    def frame_dirs(self):
        d = self.current_frames_dir()
        return [d] if d else []

    def backgrounds_dir(self):
        """Where a computed background should be written. None without a
        project — the backend then falls back to a folder beside the video."""
        return str(self.project.backgrounds_dir()) if self.project else None

    def checkpoint_dir(self):
        return str(self.project.checkpoint_dir()) if self.project else None

    def register_model(self, entry: dict) -> None:
        """Record a trained checkpoint in the project manifest so it travels
        with the .rlt. A no-op without an open project (checkpoints then land
        in the backend's temp dir and there is nothing to record them against)."""
        if self.project:
            self.project.add_model(entry)

    def status(self, msg):
        self._status(msg)

    # ---- active-learning ctx ----------------------------------------------
    def crop_params(self):
        return self.pipeline.preprocess.build_crop_params()

    def al_state(self):
        return self._al_state

    def current_frame(self):
        return self.player.index

    def seek(self, frame):
        self._seek(frame)

    def set_al_threshold(self, t):
        self.al_threshold = t
        self.timeline.set_threshold(t)

    def set_al_results(self, state):
        self._al_state = state
        if self.player.reader is not None:
            n = self.player.reader.num_frames
        else:
            n = (max(state.by_index) + 1) if state.by_index else 0
        self.timeline.set_data(state, n)
        self._recompute_misclassified()
        self._update_hud(self.player.index)

    def _recompute_misclassified(self) -> None:
        if self._al_state is None:
            return

        def label_at(i):
            b = self.model.bout_at_frame(i)
            return b.behavior_name if b else None

        self.timeline.set_misclassified(self._al_state.misclassified_frames(label_at))

    def _update_hud(self, frame: int) -> None:
        b = self.model.bout_at_frame(frame)
        human = b.behavior_name if b else None
        info = hud_info(self._al_state, frame, human)
        if self._al_state is None or info["predicted"] is None:
            self.hud.hide()
            return
        text = f"pred: {info['predicted']}"
        if info["confidence"] is not None:
            text += f"  ({info['confidence']:.2f})"
        if info["label"]:
            text += f"\nlabel: {info['label']}"
        color = "#ff5555" if info["mismatch"] else "#ffffff"
        self.hud.setText(text)
        self.hud.setStyleSheet(
            f"background: rgba(0,0,0,160); color: {color}; padding:4px; border-radius:4px;")
        self.hud.adjustSize()
        self.hud.move(10, 10)
        self.hud.show()

    # ------------------------------------------------------------------ video
    def open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        self._autosave()
        self.player.load(path)
        self._video_path = path
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
            QtWidgets.QMessageBox.warning(self, "Overlap", "Cannot export: some bouts overlap.")
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
        self.model = LabelModel()
        self.model.behaviors = [Behavior(**b) for b in store.load_manifest().get("behaviors", [])]
        self.current_video_id = None
        self._rebind_model()
        self._refresh_all()
        self._status(f"Opened project {store.load_manifest().get('name')}")

    # --------------------------------------------------------------- backend
    def _spawn_bg(self, fn, on_done):
        w = CallWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(self._status)
        w.finished.connect(lambda: self._bg_workers.discard(w))
        self._bg_workers.add(w)
        w.start()

    def connect_local(self) -> None:
        self.disconnect_backend()
        self.backend = Backend(BackendConfig())
        self._status("Starting local backend…")

        def done(ok: bool) -> None:
            if ok:
                self._status("Backend ready.")
            else:
                # Don't leave a dead Backend attached: the pipeline tabs check
                # ctx.client() for None to decide whether to prompt.
                self.disconnect_backend()
                self._status(
                    "Backend failed to start — run `pixi install`, or use "
                    "Backend ▸ Connect Local to retry.")

        self._spawn_bg(lambda: self.backend.start(wait=True, timeout=40), done)

    def connect_remote(self) -> None:
        host, ok = QtWidgets.QInputDialog.getText(
            self, "Connect remote backend", "Host (e.g. 192.168.1.20):")
        if not ok or not host.strip():
            return
        self.disconnect_backend()
        self.backend = Backend(BackendConfig.remote(host.strip()))
        self._status(f"Connecting to {host}…")
        self._spawn_bg(
            lambda: self.backend.start(wait=True, timeout=20),
            lambda ok: self._status("Backend reachable." if ok else "Backend not reachable."),
        )

    def disconnect_backend(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None
            self._status("Backend disconnected.")

    def closeEvent(self, event) -> None:
        self._autosave()
        self.disconnect_backend()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    if len(sys.argv) > 1:
        win.load_video(sys.argv[1])
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

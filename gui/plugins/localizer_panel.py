"""Train Localizer tab: distil SAM2 into a localizer for this rig.

The SAM2 localizer crop method ships with weights trained on one top-down
mouse rig. For anything else, this tab has SAM2 track the animal through short
clips of the project's videos and trains a fast network on the result
(backend: /api/localizer, localizer_train.py). The result goes to the
project's models/localizer/ and, one click later, into the Preprocess tab.

Clips are seeded two ways: automatically, by the current localizer's guess on
each clip's first confident frame; or by a click on the animal, which is the
way in when the current localizer doesn't work on this rig at all.
"""

from __future__ import annotations

import base64
import os

from PySide6 import QtCore, QtGui, QtWidgets

from labeling_shell.image_pane import ImagePane
from labeling_shell.no_wheel import NoWheelComboBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .pipeline_panel import _Base

SAM2_MODELS = ("tiny", "small", "base-plus", "large")
INITS = [("Current localizer (fine-tune)", "finetune"), ("ImageNet (from scratch)", "imagenet")]
PHASES = {"plan": "Seeding clips", "label": "SAM2 labelling", "negatives": "Empty-arena frames",
          "train": "Training"}


class ClickableImagePane(ImagePane):
    """An ImagePane that reports clicks in source-image fractions (0..1)."""

    clicked = QtCore.Signal(float, float)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        shown = self.pixmap()
        if self._source is None or shown is None or shown.isNull():
            return
        pos = event.position()
        x0 = (self.width() - shown.width()) / 2
        y0 = (self.height() - shown.height()) / 2
        fx = (pos.x() - x0) / shown.width()
        fy = (pos.y() - y0) / shown.height()
        if 0 <= fx <= 1 and 0 <= fy <= 1:
            self.clicked.emit(fx, fy)


def _pixmap(b64: str) -> QtGui.QPixmap:
    pix = QtGui.QPixmap()
    pix.loadFromData(base64.b64decode(b64), "JPG")
    return pix


class LocalizerTab(_Base):
    def __init__(self, ctx, preprocess_tab=None, parent=None):
        super().__init__(ctx, parent)
        self.preprocess_tab = preprocess_tab
        self._info = None
        self._task_id = None
        self._frame = None      # {"video", "frame", "width", "height"} shown for seeding
        self._click = None      # (x, y) native px of the last click on that frame
        self.seeds: list[dict] = []
        self.result = None

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        body = QtWidgets.QVBoxLayout(inner)

        intro = QtWidgets.QLabel(
            "The SAM2 localizer crop method finds the animal with a small network. The "
            "bundled one was trained on a single top-down mouse rig. To fit yours, SAM2 "
            "tracks the animal through short clips of your videos, and a fast network learns "
            "to reproduce it. SAM2 labels about 10 frames/s on a GPU.")
        intro.setWordWrap(True)
        body.addWidget(intro)
        self.sam2_label = QtWidgets.QLabel("SAM2: connect a backend to check.")
        self.sam2_label.setWordWrap(True)
        body.addWidget(self.sam2_label)

        # ---- clips --------------------------------------------------------
        clips_box = QtWidgets.QGroupBox("Clips seeded by the current localizer")
        cf = QtWidgets.QFormLayout(clips_box)
        self.videos = QtWidgets.QListWidget()
        self.videos.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.videos.setMaximumHeight(120)
        self.clips_per_video = self._spin(6, 0, 200, "Evenly spaced through each ticked video.")
        self.frames_per_clip = self._spin(500, 10, 20000, "Frames SAM2 tracks per clip.")
        self.frame_step = self._spin(2, 1, 100, "Track every Nth frame: a clip covers more "
                                                "time for the same SAM2 cost.")
        cf.addRow("Videos", self.videos)
        cf.addRow("Clips per video", self.clips_per_video)
        cf.addRow("Frames per clip", self.frames_per_clip)
        cf.addRow("Frame step", self.frame_step)
        body.addWidget(clips_box)

        # ---- click seeds ----------------------------------------------------
        seed_box = QtWidgets.QGroupBox("Clips seeded by a click")
        sv = QtWidgets.QVBoxLayout(seed_box)
        hint = QtWidgets.QLabel(
            "Where the current localizer misses the animal, seed a clip yourself: load the "
            "player's frame, click the animal, check that SAM2's green mask covers it, then add "
            "the seed. The clip starts at that frame.")
        hint.setWordWrap(True)
        sv.addWidget(hint)
        self.seed_pane = ClickableImagePane("Load a frame, then click the animal")
        self.seed_pane.setMinimumSize(320, 240)
        self.seed_pane.setStyleSheet("background:#1e1f22;border:1px solid #3c3f41;")
        self.seed_pane.clicked.connect(self._on_seed_click)
        sv.addWidget(self.seed_pane, 1)
        self.seed_info = QtWidgets.QLabel("")
        self.seed_info.setWordWrap(True)
        sv.addWidget(self.seed_info)
        row = QtWidgets.QHBoxLayout()
        self.load_frame_btn = QtWidgets.QPushButton("Load current frame")
        self.load_frame_btn.clicked.connect(self._load_frame)
        self.add_seed_btn = QtWidgets.QPushButton("Add seed")
        self.add_seed_btn.setEnabled(False)
        self.add_seed_btn.clicked.connect(self._add_seed)
        self.remove_seed_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_seed_btn.clicked.connect(self._remove_seed)
        for b in (self.load_frame_btn, self.add_seed_btn, self.remove_seed_btn):
            row.addWidget(b)
        row.addStretch(1)
        sv.addLayout(row)
        self.seed_table = QtWidgets.QTableWidget(0, 4)
        self.seed_table.setHorizontalHeaderLabels(["video", "frame", "x", "y"])
        self.seed_table.horizontalHeader().setStretchLastSection(True)
        self.seed_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.seed_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.seed_table.setMaximumHeight(130)
        sv.addWidget(self.seed_table)
        body.addWidget(seed_box)

        # ---- training --------------------------------------------------------
        train_box = QtWidgets.QGroupBox("Training")
        tf = QtWidgets.QFormLayout(train_box)
        self.sam2_model = NoWheelComboBox()
        self.sam2_model.addItems(SAM2_MODELS)
        self.sam2_model.setToolTip("SAM2 size. tiny is what the bundled localizer was "
                                   "distilled from; larger is slower and rarely needed.")
        self.negatives = QtWidgets.QListWidget()
        self.negatives.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.negatives.setMaximumHeight(90)
        self.negatives.setToolTip("Optional: videos of the empty arena. Their frames teach the "
                                  "presence head what 'no animal' looks like on this rig "
                                  "(synthetic empty frames are made either way).")
        self.init = NoWheelComboBox()
        for label, value in INITS:
            self.init.addItem(label, value)
        self.steps = self._spin(4000, 1, 500000, "Optimizer steps (batches of 64).")
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("localizer_<date>")
        self.name.setValidator(QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[A-Za-z0-9._-]*")))
        tf.addRow("SAM2 model", self.sam2_model)
        tf.addRow("Empty-arena videos", self.negatives)
        tf.addRow("Start from", self.init)
        tf.addRow("Steps", self.steps)
        tf.addRow("Name", self.name)
        body.addWidget(train_box)

        # ---- result ----------------------------------------------------------
        self.result_box = QtWidgets.QGroupBox("Result")
        rv = QtWidgets.QVBoxLayout(self.result_box)
        self.result_info = QtWidgets.QLabel("")
        self.result_info.setWordWrap(True)
        rv.addWidget(self.result_info)
        self.qc_pane = ImagePane("QC sheet: SAM2 centre (green) vs the new localizer (magenta)")
        self.qc_pane.setMinimumSize(320, 220)
        rv.addWidget(self.qc_pane, 1)
        self.use_btn = QtWidgets.QPushButton("Use for cropping")
        self.use_btn.setToolTip("Set it as the Preprocess localizer and switch the crop method "
                                "to SAM2 localizer. Re-preprocess before training a classifier.")
        self.use_btn.clicked.connect(self._use_result)
        rv.addWidget(self.use_btn)
        self.result_box.setVisible(False)
        body.addWidget(self.result_box)
        body.addStretch(1)
        outer.addWidget(scroll, 1)

        # ---- run (pinned) ------------------------------------------------------
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.phase_label = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar()
        run_row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Train localizer")
        mark_accent(self.train_btn)
        self.train_btn.clicked.connect(self._train)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.train_btn, 1)
        run_row.addWidget(self.stop_btn)
        outer.addLayout(run_row)
        outer.addWidget(self.phase_label)
        outer.addWidget(self.progress)
        outer.addWidget(self.log)

    @staticmethod
    def _spin(value, lo, hi, tip):
        w = NoWheelSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        w.setToolTip(tip)
        return w

    # ------------------------------------------------------------- refresh
    def refresh(self) -> None:
        videos = getattr(self.ctx, "project_videos", lambda: [])()
        self._fill(self.videos, videos, default=True)
        self._fill(self.negatives, videos, default=False)
        client = self.ctx.client()
        if client is not None and self._info is None:
            self._info = {}  # one request at a time
            self._spawn_call(client.localizer_info, self._show_info,
                             on_fail=lambda m: setattr(self, "_info", None))

    @staticmethod
    def _fill(widget: QtWidgets.QListWidget, videos: list, default: bool) -> None:
        """Rebuild the checklist from the project's videos, keeping ticks."""
        ticked = {widget.item(i).data(QtCore.Qt.ItemDataRole.UserRole):
                  widget.item(i).checkState() for i in range(widget.count())}
        widget.clear()
        for v in videos:
            path = v.get("path")
            if not path:
                continue
            item = QtWidgets.QListWidgetItem(v.get("name") or os.path.basename(path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            on = QtCore.Qt.CheckState.Checked if default else QtCore.Qt.CheckState.Unchecked
            item.setCheckState(ticked.get(path, on))
            widget.addItem(item)

    @staticmethod
    def _checked(widget: QtWidgets.QListWidget) -> list[str]:
        return [widget.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(widget.count())
                if widget.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def _show_info(self, info: dict) -> None:
        self._info = info
        if info.get("sam2_available"):
            self.sam2_label.setText(f"SAM2 ready ({info.get('sam2_detail')}). The first run "
                                    f"downloads the model (~150 MB for tiny).")
        else:
            self.sam2_label.setText(f"SAM2 unavailable on the backend: {info.get('sam2_detail')}")

    def current_localizer(self) -> str | None:
        """The weights the Preprocess tab crops with (None = bundled)."""
        if self.preprocess_tab is None:
            return None
        return self.preprocess_tab.loc_edit.text().strip() or None

    # -------------------------------------------------------------- seeding
    def _load_frame(self) -> None:
        client = self._client()
        video = self.ctx.video_path()
        if not client or not video:
            self.seed_info.setText("Load a video in the player first.")
            return
        frame = int(self.ctx.current_frame())
        self._click = None
        self.add_seed_btn.setEnabled(False)
        self.seed_info.setText(f"Loading frame {frame}…")

        def show(res: dict):
            self._frame = {"video": video, "frame": frame, "width": res["width"],
                           "height": res["height"]}
            self.seed_pane.set_image(_pixmap(res["image"]))
            self.seed_info.setText(f"{os.path.basename(video)} frame {frame}: click the animal.")
        self._spawn_call(lambda: client.localizer_seed_preview(video, frame), show,
                         on_fail=self.seed_info.setText)

    def _on_seed_click(self, fx: float, fy: float) -> None:
        client = self._client()
        if not client or self._frame is None:
            return
        f = dict(self._frame)
        x, y = fx * f["width"], fy * f["height"]
        model = self.sam2_model.currentText()
        self.seed_info.setText(f"Running SAM2 on the click at ({x:.0f}, {y:.0f})… "
                               f"(the first time loads the model)")

        def show(res: dict):
            if self._frame != f:  # a different frame was loaded meanwhile
                return
            self.seed_pane.set_image(_pixmap(res["image"]))
            if res.get("center"):
                self._click = (x, y)
                self.add_seed_btn.setEnabled(True)
                self.seed_info.setText(
                    f"SAM2 mask {res['area']:,} px, body centre ({res['center'][0]:.0f}, "
                    f"{res['center'][1]:.0f}). Add the seed if green covers the animal; "
                    f"otherwise click again.")
            else:
                self._click = None
                self.add_seed_btn.setEnabled(False)
                self.seed_info.setText("SAM2 found nothing there; click on the animal's body.")
        self._spawn_call(lambda: client.localizer_seed_preview(f["video"], f["frame"], x, y, model),
                         show, on_fail=self.seed_info.setText)

    def _add_seed(self) -> None:
        if self._frame is None or self._click is None:
            return
        seed = {"video_path": self._frame["video"], "frame": self._frame["frame"],
                "x": round(self._click[0], 1), "y": round(self._click[1], 1)}
        if seed not in self.seeds:
            self.seeds.append(seed)
        self._render_seeds()
        self.add_seed_btn.setEnabled(False)

    def _remove_seed(self) -> None:
        rows = sorted({i.row() for i in self.seed_table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.seeds[r]
        self._render_seeds()

    def _render_seeds(self) -> None:
        self.seed_table.setRowCount(len(self.seeds))
        for r, s in enumerate(self.seeds):
            for c, v in enumerate((os.path.basename(s["video_path"]), s["frame"], s["x"], s["y"])):
                self.seed_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))

    # ------------------------------------------------------------- training
    def build_request(self) -> dict:
        return {
            "videos": self._checked(self.videos),
            "clips_per_video": self.clips_per_video.value(),
            "seeds": list(self.seeds),
            "frames_per_clip": self.frames_per_clip.value(),
            "frame_step": self.frame_step.value(),
            "sam2_model": self.sam2_model.currentText(),
            "negative_videos": self._checked(self.negatives),
            "init": self.init.currentData(),
            "init_weights": self.current_localizer(),
            "steps": self.steps.value(),
            "output_dir": getattr(self.ctx, "localizer_dir", lambda: None)(),
            "name": self.name.text().strip() or None,
        }

    def validate(self, req: dict) -> str | None:
        if not req["output_dir"]:
            return "Open a project first: the localizer is saved into it."
        if not req["seeds"] and not (req["videos"] and req["clips_per_video"] > 0):
            return "Nothing to label: tick a video (with clips per video > 0) or add a click seed."
        return None

    def _train(self) -> None:
        client = self._client()
        if not client:
            return
        req = self.build_request()
        problem = self.validate(req)
        if problem:
            self.ctx.status(problem)
            self.phase_label.setText(problem)
            return
        self.log.clear()
        self.result_box.setVisible(False)
        self.progress.setValue(0)
        self.train_btn.setEnabled(False)
        self.phase_label.setText("Starting…")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if not self._task_id:
                self.train_btn.setEnabled(True)
                return
            self.stop_btn.setEnabled(True)
            task_id = self._task_id
            w = self._spawn_stream(lambda: client.localizer_status(task_id), self._on_event,
                                   on_done=self._on_stream_end)
            w.failed.connect(self._on_failed)

        # Re-enable Train if the request itself fails (e.g. 503: no SAM2).
        self._spawn_call(lambda: client.train_localizer(req), start, on_fail=self._on_failed)

    def _spawn_call(self, fn, on_done, on_fail=None):
        from labeling_shell.backend_worker import CallWorker
        w = CallWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(on_fail or self.ctx.status)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()

    def _on_failed(self, msg: str) -> None:
        self.ctx.status(msg)
        self.phase_label.setText(msg)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_event(self, ev: dict) -> None:
        for line in ev.get("log") or []:
            self.log.appendPlainText(line)
        total, prog = ev.get("total") or 0, ev.get("progress") or 0
        if total:
            self.progress.setMaximum(total)
            self.progress.setValue(min(prog, total))
        status = ev.get("status")
        phase = PHASES.get(ev.get("phase"), ev.get("phase") or "")
        if status == "running":
            self.phase_label.setText(f"{phase}: {prog}/{total}" if total else phase)
        elif status == "completed":
            self._show_result(ev.get("result") or {})
        elif status == "error":
            self.phase_label.setText(f"Failed: {ev.get('error')}")
            self.ctx.status(f"Train localizer failed: {ev.get('error')}")
        elif status == "cancelled":
            self.phase_label.setText("Stopped.")

    def _on_stream_end(self) -> None:
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._task_id = None

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            task_id = self._task_id
            self._spawn_call(lambda: client.stop_localizer(task_id), lambda _r: None)

    def _show_result(self, res: dict) -> None:
        self.result = res
        self.phase_label.setText("Done.")

        def fmt(m):
            if not m or m.get("median_error_px") is None:
                return "n/a"
            return f"median {m['median_error_px']:.1f} px, p95 {m['p95_error_px']:.1f} px"
        lines = [f"Saved {res.get('weights')}",
                 f"{res.get('train_frames')} training frames, {res.get('val_frames')} held out, "
                 f"{res.get('dropped_frames')} dropped (SAM2 lost the animal or the mask jumped)."]
        after = res.get("after") or {}
        if res.get("before"):
            lines.append(f"Held-out centre error vs SAM2: {fmt(res['before'])} before → "
                         f"{fmt(after)} after.")
        else:
            lines.append(f"Held-out centre error vs SAM2: {fmt(after)}.")
        if res.get("dropped_clips"):
            lines.append(f"{len(res['dropped_clips'])} clip(s) left out (see the log): check their seeds.")
        if res.get("improved") is False:
            lines.append("Training never beat the starting weights on the held-out clips, so this "
                         "file is a copy of them. Keep the current localizer, or label more clips.")
        if after.get("presence_on_empty") is not None:
            lines.append(f"Presence: {after.get('presence_on_animal', 0):.2f} on the animal, "
                         f"{after['presence_on_empty']:.2f} on empty frames.")
        self.result_info.setText("\n".join(lines))
        qc = res.get("qc_image")
        self.qc_pane.set_image(QtGui.QPixmap(qc) if qc and os.path.exists(qc) else None)
        self.result_box.setVisible(True)
        self.use_btn.setEnabled(res.get("improved") is not False)
        self.ctx.status("Localizer trained — 'Use for cropping' to switch Preprocess to it."
                        if res.get("improved") is not False else
                        "Localizer training didn't improve on the current one.")

    def _use_result(self) -> None:
        weights = (self.result or {}).get("weights")
        if not weights or self.preprocess_tab is None:
            return
        self.preprocess_tab.set_localizer(weights, select=True)
        self.ctx.status(f"Preprocess now crops with {os.path.basename(weights)}. "
                        f"Re-run preprocessing before training a classifier on it.")

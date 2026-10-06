"""Pipeline UI — Preprocess / Train / Export tabs (Phase 3).

Qt successor to the preprocess/train/export parts of plugins/pipeline.js. Reads
its inputs from a host context (`ctx`) and drives the backend through
BackendClient, streaming SSE progress on background threads (backend_worker).

`ctx` duck-type (provided by MainWindow):
    client() -> BackendClient | None      fps() -> float
    video_path() -> str | None            bouts() -> list[Bout]
    behaviors() -> list[Behavior]         frames_per_group() -> int
    label_paths() -> list[str]            frame_dirs() -> list[str]
    current_frames_dir() -> str | None    current_bg_path() -> str | None
    checkpoint_dir() -> str | None        status(msg: str) -> None
    register_model(entry: dict) -> None   (optional; records a trained checkpoint)
"""

from __future__ import annotations

import base64
import datetime
import os

from PySide6 import QtCore, QtGui, QtWidgets

from labeling_shell.backend_worker import CallWorker, StreamWorker
from labeling_shell.frame_ranges import compute_labeled_frame_ranges
from labeling_shell.image_pane import ImagePane
from labeling_shell.no_wheel import (
    NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox,
)
from labeling_shell.theme import mark_accent

from .pipeline_config import (
    BATCH_CHOICES, CROP_DEFAULTS, CROP_KEYS_BY_METHOD, CROP_METHODS,
    EXPORT_FORMATS, FRAMES_CHOICES, LOCALIZER_SMOOTHING,
    IMAGE_SIZE_CHOICES, OBJECTIVES, POOLS, PRECISIONS, TARGET_LABELS, TARGETS,
    VARIANT_NATIVE_SIZE, VARIANTS,
    build_train_config, build_tuning_request, choice_spec, latency_spec,
    parse_stages, range_spec, resolve_binarize,
)


class _Base(QtWidgets.QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._workers: set = set()

    def _client(self):
        c = self.ctx.client()
        if c is None:
            self.ctx.status("Connect a backend first (Backend menu).")
        return c

    def _spawn_call(self, fn, on_done):
        w = CallWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(self.ctx.status)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()

    def _spawn_stream(self, make_iter, on_event, on_done=None):
        w = StreamWorker(make_iter)
        w.event.connect(on_event)
        w.failed.connect(self.ctx.status)
        if on_done:
            w.finished_ok.connect(on_done)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()
        return w


class PreprocessTab(_Base):
    # Preview panes, in pipeline order: what was detected, the mask it came
    # from, and the crop that actually gets written to disk.
    PREVIEW_PANES = (
        ("original_bbox", "frame + box"),
        ("mask", "mask"),
        ("cropped", "cropped (saved)"),
    )

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        body = QtWidgets.QVBoxLayout(inner)
        form = QtWidgets.QFormLayout()

        self.video_label = QtWidgets.QLabel("—")
        form.addRow("Video", self.video_label)

        # How the animal is found. Background subtraction is the default; the
        # SAM2-distilled localizer copes with a drifting background or an
        # empty arena; a rig with no static background (moving camera, or the
        # animal filling the frame) can skip cropping and use whole frames.
        # Active Learning crops with the same settings, so they must match
        # what the model was trained on.
        self.crop_method = NoWheelComboBox()
        for label, method in CROP_METHODS:
            self.crop_method.addItem(label, method)
        self.crop_method.setToolTip(
            "Background subtraction: needs a background image.\n"
            "SAM2 localizer: a network finds the animal; needs its weights (.pt), "
            "no background.\n"
            "None: write whole frames. resize_factor still applies.")
        self.crop_method.currentIndexChanged.connect(
            lambda _i: self._on_method_changed(self.method()))
        form.addRow("Crop method", self.crop_method)

        self.loc_edit = QtWidgets.QLineEdit()
        self.loc_edit.setPlaceholderText("localizer weights (.pt)")
        self.loc_btn = QtWidgets.QPushButton("Browse…")
        self.loc_btn.clicked.connect(self._pick_localizer)
        loc_row = QtWidgets.QHBoxLayout()
        loc_row.addWidget(self.loc_edit, 1)
        loc_row.addWidget(self.loc_btn)
        self.loc_row_label = QtWidgets.QLabel("Localizer")
        form.addRow(self.loc_row_label, self._wrap(loc_row))

        self.bg_edit = QtWidgets.QLineEdit()
        self.bg_frames = NoWheelSpinBox()
        self.bg_frames.setRange(1, 1_000_000)
        self.bg_frames.setValue(500)
        self.bg_frames.setToolTip(
            "How many frames to average into the background. More is a cleaner "
            "background but a slower computation; they are sampled across the "
            "whole video, so this is not a time limit.")
        self.bg_btn = QtWidgets.QPushButton("Compute background")
        self.bg_btn.clicked.connect(self._compute_background)
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.addWidget(self.bg_edit, 1)
        bg_row.addWidget(QtWidgets.QLabel("frames"))
        bg_row.addWidget(self.bg_frames)
        bg_row.addWidget(self.bg_btn)
        self.bg_row_label = QtWidgets.QLabel("Background")
        form.addRow(self.bg_row_label, self._wrap(bg_row))

        self.crop_widgets: dict[str, QtWidgets.QWidget] = {}
        self.crop_labels: dict[str, QtWidgets.QLabel] = {}
        for key, default in CROP_DEFAULTS.items():
            if key == "difference_mode":
                w = NoWheelComboBox()
                w.addItems(["directional", "absolute"])
            elif isinstance(default, float):
                w = NoWheelDoubleSpinBox()
                w.setRange(0.0, 100000.0)
                w.setDecimals(3)
                w.setValue(default)
            else:
                w = NoWheelSpinBox()
                w.setRange(0, 100000)
                w.setValue(default)
            self.crop_widgets[key] = w
            label = QtWidgets.QLabel(key)
            self.crop_labels[key] = label
            form.addRow(label, w)
        body.addLayout(form)

        # ---- preview ------------------------------------------------------
        self.preview_box = QtWidgets.QGroupBox("Preview")
        pv = QtWidgets.QVBoxLayout(self.preview_box)

        panes = QtWidgets.QHBoxLayout()
        self.preview_labels: dict[str, ImagePane] = {}
        for key, caption in self.PREVIEW_PANES:
            col = QtWidgets.QVBoxLayout()
            img = ImagePane()
            img.setStyleSheet("background:#1e1f22;border:1px solid #3c3f41;")
            cap = QtWidgets.QLabel(caption)
            cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            col.addWidget(img, 1)
            col.addWidget(cap)
            panes.addLayout(col, 1)
            self.preview_labels[key] = img
        pv.addLayout(panes)

        self.preview_info = QtWidgets.QLabel("Preview a frame to check the crop "
                                             "before extracting thousands of them.")
        self.preview_info.setWordWrap(True)
        pv.addWidget(self.preview_info)

        prow = QtWidgets.QHBoxLayout()
        self.preview_frame_spin = NoWheelSpinBox()
        self.preview_frame_spin.setRange(0, 10_000_000)
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_here_btn = QtWidgets.QPushButton("Use current frame")
        self.preview_prev_btn = QtWidgets.QPushButton("◀")
        self.preview_next_btn = QtWidgets.QPushButton("▶")
        for b in (self.preview_prev_btn, self.preview_next_btn):
            b.setFixedWidth(32)
        self.preview_prev_btn.setToolTip("Previous frame")
        self.preview_next_btn.setToolTip(
            "Next frame — stepping carries the smoothed crop centre forward, so "
            "the temporal stabilization is visible")
        self.preview_btn.clicked.connect(lambda: self._preview(reset_center=True))
        self.preview_here_btn.clicked.connect(self._preview_current_frame)
        self.preview_prev_btn.clicked.connect(lambda: self._step_preview(-1))
        self.preview_next_btn.clicked.connect(lambda: self._step_preview(+1))
        prow.addWidget(QtWidgets.QLabel("frame"))
        prow.addWidget(self.preview_frame_spin)
        prow.addWidget(self.preview_prev_btn)
        prow.addWidget(self.preview_next_btn)
        prow.addWidget(self.preview_btn)
        prow.addWidget(self.preview_here_btn)
        prow.addStretch(1)
        pv.addLayout(prow)
        body.addWidget(self.preview_box)
        body.addStretch(1)
        outer.addWidget(scroll, 1)

        self.progress = QtWidgets.QProgressBar()
        run_btn = QtWidgets.QPushButton("Run preprocessing")
        mark_accent(run_btn)
        run_btn.clicked.connect(self._run)
        outer.addWidget(run_btn)
        outer.addWidget(self.progress)

        # Smoothed crop centre carried between previewed frames.
        self._preview_center = None
        self._on_method_changed(self.method())

    @staticmethod
    def _wrap(layout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def method(self) -> str:
        return self.crop_method.currentData()

    def _on_method_changed(self, method: str) -> None:
        """Enable only the knobs the chosen crop method reads: the background
        and detection settings are background-only, the box and its
        stabilisation are shared with the localizer, and resize_factor applies
        to every method (it still resizes whole frames)."""
        for key, w in self.crop_widgets.items():
            enabled = key in CROP_KEYS_BY_METHOD[method]
            w.setEnabled(enabled)
            self.crop_labels[key].setEnabled(enabled)
        for w in (self.bg_edit, self.bg_btn, self.bg_frames, self.bg_row_label):
            w.setEnabled(method == "background")
        for w in (self.loc_edit, self.loc_btn, self.loc_row_label):
            w.setEnabled(method == "localizer")
        self.preview_box.setEnabled(method != "none")
        if method == "none":
            self.preview_info.setText(
                "Raw mode: whole frames are written, no background subtraction. "
                "Nothing to preview.")
        elif method == "localizer" and self.crop_widgets["smoothing"].value() == 0:
            # Unsmoothed localizer centres jitter by a few px frame to frame;
            # 0.5 is the EMA the localizer pipeline was validated with.
            self.crop_widgets["smoothing"].setValue(LOCALIZER_SMOOTHING)
            self.ctx.status(f"Localizer: smoothing set to {LOCALIZER_SMOOTHING}.")

    def _pick_localizer(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Localizer weights", "", "PyTorch weights (*.pt *.pth);;All files (*)")
        if path:
            self.loc_edit.setText(path)

    def refresh(self) -> None:
        self.video_label.setText(self.ctx.video_path() or "—")
        if self.ctx.current_bg_path() and not self.bg_edit.text():
            self.bg_edit.setText(self.ctx.current_bg_path())

    def build_crop_params(self) -> dict:
        out = {}
        for key, w in self.crop_widgets.items():
            if isinstance(w, QtWidgets.QComboBox):
                out[key] = w.currentText()
            else:
                out[key] = w.value()
        out["method"] = self.method()
        if out["method"] == "localizer":
            out["localizer_path"] = self.loc_edit.text().strip() or None
        return out

    def _compute_background(self) -> None:
        client = self._client()
        video = self.ctx.video_path()
        if not client or not video:
            return
        frames = self.bg_frames.value()
        # Into the project, so it is found again on the next video switch and
        # travels with the .rlt. Without a project the backend falls back to a
        # folder beside the video.
        out_dir = getattr(self.ctx, "backgrounds_dir", lambda: None)()
        self.ctx.status(f"Computing background from {frames} frames…")
        self._spawn_call(
            lambda: client.compute_background(video, num_frames=frames,
                                              output_dir=out_dir),
            lambda res: (self.bg_edit.setText(res.get("path", "")),
                         self.ctx.status("Background computed.")),
        )

    # ----------------------------------------------------------------- preview
    def _preview_current_frame(self) -> None:
        """Preview whatever frame the player is showing."""
        self.preview_frame_spin.setValue(int(self.ctx.current_frame()))
        self._preview(reset_center=True)

    def _step_preview(self, delta: int) -> None:
        """Step one frame, keeping the smoothed centre so the stabilization
        (smoothing / deadband) is visible rather than re-seeded every time."""
        self.preview_frame_spin.setValue(
            max(0, self.preview_frame_spin.value() + delta))
        self._preview(reset_center=False)

    def _preview(self, reset_center: bool = True) -> None:
        client = self._client()
        video = self.ctx.video_path()
        if not client or not video:
            self.ctx.status("Load a video first.")
            return
        bg = None
        if self.method() == "localizer":
            if not self.loc_edit.text().strip():
                self.preview_info.setText("Choose the localizer weights (.pt) first.")
                return
        else:
            bg = self.bg_edit.text().strip() or self.ctx.current_bg_path()
            if not bg:
                self.preview_info.setText(
                    "No background yet — click Compute background first.")
                return
        if reset_center:
            self._preview_center = None

        frame = self.preview_frame_spin.value()
        # Snapshot the centre and params here rather than reading them inside
        # the lambda: that body runs on a worker thread, and a second preview
        # started meanwhile would otherwise change what this request sends.
        center = self._preview_center
        params = self.build_crop_params()
        self.preview_info.setText(f"Previewing frame {frame}…")
        self._spawn_call(
            lambda: client.preview_frame(
                video, bg, frame, crop_params=params, prev_center=center),
            self._show_preview,
        )

    def _show_preview(self, res: dict) -> None:
        for key, _caption in self.PREVIEW_PANES:
            pane = self.preview_labels[key]
            data = res.get(key)
            if not data:
                pane.set_image(None)
                continue
            pix = QtGui.QPixmap()
            if pix.loadFromData(base64.b64decode(data), "JPG"):
                # ImagePane keeps the source and scales to its own geometry, so
                # stepping frames can't ratchet the pane size up or down.
                pane.set_image(pix)
            else:
                pane.set_message("(decode failed)")

        self._preview_center = res.get("center")
        src = f"{res.get('frame_width')}×{res.get('frame_height')}"
        proc = f"{res.get('resized_width')}×{res.get('resized_height')}"
        box = self.crop_widgets["box_size"].value()
        note = f"frame {self.preview_frame_spin.value()} · source {src}"
        if proc != src:
            note += f" · processed {proc}"
        note += f" · crop {box}×{box} px"
        if res.get("presence") is not None:
            note += f" · animal present p={res['presence']:.2f}"
        self.preview_info.setText(note)

    def _run(self) -> None:
        client = self._client()
        video = self.ctx.video_path()
        out_dir = self.ctx.current_frames_dir()
        if not client or not video or not out_dir:
            self.ctx.status("Need a video and a project (for the output dir).")
            return
        ranges = compute_labeled_frame_ranges(self.ctx.bouts(), self.ctx.frames_per_group())
        self.progress.setValue(0)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            if ev.get("status") in ("completed", "error", "cancelled"):
                self.ctx.status(f"Preprocess {ev.get('status')}.")

        def start(res: dict):
            task_id = res.get("task_id")
            if task_id:
                self._spawn_stream(lambda: client.preprocess_status(task_id), on_event)

        method = self.method()
        if method == "localizer" and not self.loc_edit.text().strip():
            self.ctx.status("Choose the localizer weights (.pt) first.")
            return
        crop = method != "none"
        params = self.build_crop_params()
        self.ctx.status("Extracting frames…" if not crop else "Preprocessing…")
        self._spawn_call(
            lambda: client.run_preprocessing(
                video, out_dir, crop_params=params,
                frame_ranges=ranges,
                # Only background mode reads a background; don't send a stale one.
                bg_path=(self.bg_edit.text() or None) if method == "background" else None,
                # Only used when no bg_path is given and the backend has to
                # build one itself — same frame count either way.
                num_frames_bg=self.bg_frames.value(),
                crop=crop),
            start,
        )


class TrainTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        # The form outgrew a typical window once boundary smoothing, W&B and the
        # positive-class checklist landed, so it scrolls; the action buttons and
        # log stay pinned below it.
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        layout = QtWidgets.QVBoxLayout(inner)
        form = QtWidgets.QFormLayout()

        self.frames_per_group = NoWheelSpinBox()
        self.frames_per_group.setRange(1, 32)
        self.frames_per_group.setValue(3)
        self.variant = NoWheelComboBox()
        self.variant.addItems(VARIANTS)
        self.tsm_stages = QtWidgets.QLineEdit("5,6,7")
        self.temporal_pool = NoWheelComboBox()
        self.temporal_pool.addItems(POOLS)
        self.unidirectional = QtWidgets.QCheckBox("Unidirectional (causal)")
        self.pretrained = QtWidgets.QCheckBox("Pretrained")
        self.pretrained.setChecked(True)
        self.dropout = self._dspin(0.2, 0.0, 0.9, 0.05)
        self.lr = self._dspin(1e-4, 1e-6, 1.0, 1e-4, decimals=6)
        self.weight_decay = self._dspin(1e-3, 0.0, 1.0, 1e-4, decimals=6)
        self.batch_size = self._ispin(64, 1, 1024)
        self.image_size = self._ispin(224, 32, 800)
        self.max_epochs = self._ispin(100, 1, 100000)
        self.patience = self._ispin(20, 1, 100000)

        # Positive class: a checklist, not a single pick. Ticking two or more
        # behaviors unions them into one positive label (see resolve_binarize).
        self.binarize = QtWidgets.QListWidget()
        self.binarize.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.binarize.setMaximumHeight(110)
        self.binarize_hint = QtWidgets.QLabel()
        self.binarize_hint.setWordWrap(True)
        self.binarize.itemChanged.connect(lambda _i: self._update_binarize_hint())

        # A clip is num_frames images, so the backbone sees batch_size × frames
        # per step — the number that decides whether a big variant fits in VRAM.
        self.batch_hint = QtWidgets.QLabel()
        self.batch_hint.setWordWrap(True)
        for w in (self.batch_size, self.frames_per_group, self.image_size):
            w.valueChanged.connect(self._update_batch_hint)
        self.variant.currentTextChanged.connect(self._update_batch_hint)
        self._update_batch_hint()

        for label, w in [
            ("frames/group", self.frames_per_group), ("variant", self.variant),
            ("tsm_stages", self.tsm_stages), ("temporal_pool", self.temporal_pool),
            ("", self.unidirectional), ("", self.pretrained), ("dropout", self.dropout),
            ("lr", self.lr), ("weight_decay", self.weight_decay),
            ("batch_size", self.batch_size), ("image_size", self.image_size),
            ("", self.batch_hint),
            ("max_epochs", self.max_epochs), ("early stop patience", self.patience),
            ("positive class (binarize)", self.binarize), ("", self.binarize_hint),
        ]:
            form.addRow(label, w)
        layout.addLayout(form)

        # ---- boundary smoothing ------------------------------------------
        bs_box = QtWidgets.QGroupBox("Boundary smoothing")
        bs_form = QtWidgets.QFormLayout(bs_box)
        self.bs_enabled = QtWidgets.QCheckBox("Soften labels near bout edges")
        self.bs_enabled.setChecked(True)  # matches config_tsm.yaml
        self.bs_epsilon = self._dspin(0.3, 0.0, 1.0, 0.05, decimals=2)
        self.bs_window = self._ispin(5, 1, 20)
        bs_form.addRow(self.bs_enabled)
        bs_form.addRow("epsilon", self.bs_epsilon)
        bs_form.addRow("window", self.bs_window)
        for w in (self.bs_epsilon, self.bs_window):
            self.bs_enabled.toggled.connect(w.setEnabled)
        layout.addWidget(bs_box)

        # ---- Weights & Biases --------------------------------------------
        wb_box = QtWidgets.QGroupBox("Weights && Biases")
        wb_form = QtWidgets.QFormLayout(wb_box)
        self.wandb_enabled = QtWidgets.QCheckBox("Log this run to W&&B")
        self.wandb_project = QtWidgets.QLineEdit("b-real")
        self.wandb_run = QtWidgets.QLineEdit()
        self.wandb_run.setPlaceholderText("auto-generated if empty")
        self.wandb_key = QtWidgets.QLineEdit()
        self.wandb_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.wandb_key.setPlaceholderText("leave empty if already logged in")
        wb_form.addRow(self.wandb_enabled)
        wb_form.addRow("project", self.wandb_project)
        wb_form.addRow("run name", self.wandb_run)
        wb_form.addRow("API key", self.wandb_key)
        for w in (self.wandb_project, self.wandb_run, self.wandb_key):
            w.setEnabled(False)
            self.wandb_enabled.toggled.connect(w.setEnabled)
        layout.addWidget(wb_box)
        layout.addStretch(1)
        outer.addWidget(scroll, 1)

        btns = QtWidgets.QHBoxLayout()
        self.coverage_btn = QtWidgets.QPushButton("Check coverage")
        self.start_btn = QtWidgets.QPushButton("Start training")
        mark_accent(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.coverage_btn.clicked.connect(self._check_coverage)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        for b in (self.coverage_btn, self.start_btn, self.stop_btn):
            btns.addWidget(b)
        outer.addLayout(btns)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)  # the form is the working surface, not this
        outer.addWidget(self.log)
        self._task_id = None

    def _dspin(self, val, lo, hi, step, decimals=4):
        w = NoWheelDoubleSpinBox()
        w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(val)
        return w

    def _ispin(self, val, lo, hi):
        w = NoWheelSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    def _update_batch_hint(self) -> None:
        effective = self.batch_size.value() * self.frames_per_group.value()
        native = VARIANT_NATIVE_SIZE.get(self.variant.currentText(), 224)
        msg = f"effective batch: {effective} images/step"
        if self.image_size.value() < native:
            msg += (f" — {self.variant.currentText()} is trained at {native}px; "
                    f"at {self.image_size.value()}px you pay its full size for little gain")
        self.batch_hint.setText(msg)

    def refresh(self) -> None:
        """Rebuild the positive-class checklist, keeping whatever is still ticked."""
        checked = set(self.positive_behaviors())
        self.binarize.blockSignals(True)
        self.binarize.clear()
        for b in self.ctx.behaviors():
            item = QtWidgets.QListWidgetItem(b.name)
            item.setCheckState(QtCore.Qt.Checked if b.name in checked
                               else QtCore.Qt.Unchecked)
            self.binarize.addItem(item)
        self.binarize.blockSignals(False)
        self._update_binarize_hint()

    def positive_behaviors(self) -> list:
        return [self.binarize.item(i).text()
                for i in range(self.binarize.count())
                if self.binarize.item(i).checkState() == QtCore.Qt.Checked]

    def _update_binarize_hint(self) -> None:
        n = len(self.positive_behaviors())
        if n == 0:
            self.binarize_hint.setText(
                "nothing ticked → multi-class over every behavior")
        elif n == 1:
            self.binarize_hint.setText(
                "binary: the ticked behavior vs everything else")
        else:
            binarize, _ = resolve_binarize(self.positive_behaviors(),
                                           [b.name for b in self.ctx.behaviors()])
            self.binarize_hint.setText(
                f"binary: the {n} ticked behaviors merged into \"{binarize}\" "
                f"vs everything else")

    def build_config(self) -> dict:
        binarize, label_union = resolve_binarize(
            self.positive_behaviors(), [b.name for b in self.ctx.behaviors()])
        return build_train_config(
            self.ctx.label_paths(), self.ctx.frame_dirs(),
            frames_per_group=self.frames_per_group.value(),
            variant=self.variant.currentText(),
            tsm_stages=parse_stages(self.tsm_stages.text()),
            temporal_pool=self.temporal_pool.currentText(),
            unidirectional=self.unidirectional.isChecked(),
            pretrained=self.pretrained.isChecked(),
            dropout=self.dropout.value(), lr=self.lr.value(),
            weight_decay=self.weight_decay.value(),
            batch_size=self.batch_size.value(),
            image_size=self.image_size.value(),
            max_epochs=self.max_epochs.value(),
            early_stopping_patience=self.patience.value(),
            binarize=binarize, label_union=label_union,
            boundary_smoothing_enabled=self.bs_enabled.isChecked(),
            boundary_smoothing_epsilon=self.bs_epsilon.value(),
            boundary_smoothing_window=self.bs_window.value(),
            use_wandb=self.wandb_enabled.isChecked(),
            project_name=self.wandb_project.text().strip() or "b-real",
            run_name=self.wandb_run.text().strip() or None,
            wandb_api_key=self.wandb_key.text().strip() or None,
            checkpoint_dir=self.ctx.checkpoint_dir(),
        )

    def _check_coverage(self) -> None:
        client = self._client()
        if not client:
            return
        self._spawn_call(
            lambda: client.check_coverage(
                self.ctx.frame_dirs(), self.ctx.label_paths(),
                self.frames_per_group.value()),
            lambda res: self.log.appendPlainText(f"[coverage] {res}"),
        )

    def _start(self) -> None:
        client = self._client()
        if not client:
            return
        if not self.ctx.label_paths() or not self.ctx.frame_dirs():
            self.ctx.status("Need labels + extracted frames (preprocess first).")
            return
        config = self.build_config()
        self.log.appendPlainText("[train] starting…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        def on_event(ev: dict):
            self.log.appendPlainText(str(ev))
            if ev.get("status") == "completed" and ev.get("best_checkpoint"):
                self._register_model(config, ev)

        def on_done():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.ctx.status("Training stream ended.")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.training_status(self._task_id), on_event, on_done)

        self._spawn_call(lambda: client.start_training(config), start)

    def _register_model(self, config: dict, ev: dict) -> None:
        """Record the trained checkpoint in the project manifest.

        Stores the parts of the config needed to interpret the model later —
        binarize/label_union in particular, without which Active Learning can't
        map a behavior name into a binarized model's collapsed label space.
        """
        register = getattr(self.ctx, "register_model", None)
        if register is None:
            return
        path = ev["best_checkpoint"]
        entry = {
            "name": os.path.basename(path),
            "path": path,
            "created": datetime.date.today().isoformat(),
            "val_mcc": ev.get("best_val_mcc"),
            "val_loss": ev.get("best_val_loss"),
            "config": {
                "variant": config.get("variant"),
                "binarize": config.get("binarize"),
                "label_union": config.get("label_union"),
                "num_frames": config.get("num_frames"),
                "tsm_stages": config.get("tsm_stages"),
                "unidirectional": config.get("unidirectional"),
                "image_size": config.get("image_size"),
            },
        }
        try:
            register(entry)
            self.ctx.status(f"Recorded model in the project: {entry['name']}")
        except Exception as e:
            # A manifest write failing must not look like a training failure.
            self.log.appendPlainText(f"[train] could not record the model: {e}")

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_training(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))


class _ChoiceList(QtWidgets.QListWidget):
    """A compact multi-select over a fixed set of values, used for the sweep's
    categorical dimensions. Keeps the value alongside the label so `tsm_stages`
    can display "5,6,7" while frames_per_group holds an int."""

    def __init__(self, values, labels=None, checked=(), parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        labels = labels or [str(v) for v in values]
        for value, label in zip(values, labels):
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, value)
            item.setCheckState(
                QtCore.Qt.Checked if value in checked else QtCore.Qt.Unchecked)
            self.addItem(item)
        # Show every choice: a scrollbar inside the panel's own scroll area
        # makes the options hard to reach.
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(len(values) * 22 + 8)

    def checked_values(self) -> list:
        return [self.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.count())
                if self.item(i).checkState() == QtCore.Qt.Checked]


class _SweepRow:
    """One dimension of the search space: an enable checkbox plus its inputs.

    Holds no Qt parent of its own — the widgets are laid out by TuneTab. `spec()`
    returns the wire-format dict, or None when the row is disabled.
    """

    def __init__(self, name: str, checkbox, widgets: list, spec_fn):
        self.name = name
        self.checkbox = checkbox
        self.widgets = widgets
        self._spec_fn = spec_fn
        checkbox.toggled.connect(self._set_enabled)
        self._set_enabled(checkbox.isChecked())

    def _set_enabled(self, on: bool) -> None:
        for w in self.widgets:
            w.setEnabled(on)

    def enabled(self) -> bool:
        return self.checkbox.isChecked()

    def spec(self):
        return self._spec_fn() if self.enabled() else None


class TuneTab(_Base):
    """Optuna sweep over the Train tab's configuration.

    The baseline (data sources, splits, binarize, epochs) comes from TrainTab so
    there is exactly one config builder; this tab only describes what varies.
    """

    # Leaderboard columns
    COL_TRIAL, COL_STATE, COL_VALUE, COL_PARAMS = range(4)

    def __init__(self, ctx, train_tab=None, parent=None):
        super().__init__(ctx, parent)
        self.train_tab = train_tab
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel(
            "Sweeps the parameters ticked below. Everything else — data sources, "
            "splits, positive class, max_epochs — is taken from the Train tab."))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        vbox = QtWidgets.QVBoxLayout(inner)

        space_box = QtWidgets.QGroupBox("Search space")
        grid = QtWidgets.QGridLayout(space_box)
        self.rows: list[_SweepRow] = []
        self._grid_row = 0

        self.lr_lo, self.lr_hi, self.lr_log = self._add_range_row(
            grid, "lr", 1e-5, 1e-2, log=True, checked=True)
        self.wd_lo, self.wd_hi, self.wd_log = self._add_range_row(
            grid, "weight_decay", 1e-6, 1e-2, log=True)
        self.do_lo, self.do_hi, self.do_log = self._add_range_row(
            grid, "dropout", 0.0, 0.6, log=False)

        self.variant_list = self._add_choice_row(
            grid, "variant", VARIANTS,
            labels=[f"{v}  ({VARIANT_NATIVE_SIZE[v]}px native)" for v in VARIANTS],
            checked=("b0",))
        self.batch_list = self._add_choice_row(grid, "batch_size", BATCH_CHOICES)
        self.image_list = self._add_choice_row(grid, "image_size", IMAGE_SIZE_CHOICES)
        self.pool_list = self._add_choice_row(grid, "temporal_pool", POOLS)
        self.frames_list = self._add_choice_row(
            grid, "frames_per_group", FRAMES_CHOICES,
            hint="changes the data window — clips are re-grouped per trial")

        # tsm_stages is free text: too many useful combinations to enumerate.
        self.stages_edit = QtWidgets.QLineEdit("5,6,7;3,4,5,6,7;1,2,3,4,5,6,7")
        self._add_widget_row(
            grid, "tsm_stages", [self.stages_edit],
            lambda: choice_spec([s.strip() for s in self.stages_edit.text().split(";")
                                 if s.strip()]),
            hint="semicolon-separated stage lists")

        vbox.addWidget(space_box)

        # ---- latency constraint -------------------------------------------
        lat_box = QtWidgets.QGroupBox("Frame budget (reject slow architectures)")
        lat_form = QtWidgets.QFormLayout(lat_box)
        self.lat_enabled = QtWidgets.QCheckBox("Constrain p99 inference latency")
        self.lat_p99 = NoWheelDoubleSpinBox()
        self.lat_p99.setDecimals(2)
        self.lat_p99.setRange(0.01, 10000.0)
        self.lat_p99.setValue(33.0)  # ~30 fps
        self.lat_p99.setSuffix(" ms")
        self.lat_batch = NoWheelSpinBox()
        self.lat_batch.setRange(1, 256)
        self.lat_batch.setValue(1)
        self.lat_precision = NoWheelComboBox()
        self.lat_precision.addItems(PRECISIONS)
        self.lat_target = NoWheelComboBox()
        for t in TARGETS:
            self.lat_target.addItem(TARGET_LABELS[t], t)
        lat_form.addRow(self.lat_enabled)
        lat_form.addRow("deployment target", self.lat_target)
        lat_form.addRow("max p99", self.lat_p99)
        lat_form.addRow("batch size", self.lat_batch)
        lat_form.addRow("precision", self.lat_precision)
        lat_form.addRow(QtWidgets.QLabel(
            "Each sampled architecture is exported to the deployment format "
            "(OpenVINO for a CPU rig, TensorRT for a GPU one) and timed before "
            "it trains; anything over budget is rejected without spending a "
            "training run. Engine builds are cached per architecture. If the "
            "runtime isn't installed the sweep refuses to start, rather than "
            "reporting numbers from a format you don't deploy."))
        for w in (self.lat_p99, self.lat_batch, self.lat_precision, self.lat_target):
            w.setEnabled(False)
            self.lat_enabled.toggled.connect(w.setEnabled)
        vbox.addWidget(lat_box)

        # ---- sweep settings -----------------------------------------------
        settings = QtWidgets.QGroupBox("Sweep")
        sf = QtWidgets.QFormLayout(settings)
        self.n_trials = NoWheelSpinBox()
        self.n_trials.setRange(1, 1000)
        self.n_trials.setValue(20)
        self.objective = NoWheelComboBox()
        self.objective.addItems(OBJECTIVES)
        sf.addRow("trials", self.n_trials)
        sf.addRow("objective", self.objective)
        vbox.addWidget(settings)
        vbox.addStretch(1)
        # The controls are what the user works in; the leaderboard and log only
        # matter once a sweep is running, so give the form the bulk of the panel.
        scroll.setMinimumHeight(320)
        layout.addWidget(scroll, 4)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start sweep")
        mark_accent(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.apply_btn = QtWidgets.QPushButton("Apply best to Train tab")
        self.apply_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.apply_btn.clicked.connect(self._apply_best)
        for b in (self.start_btn, self.stop_btn, self.apply_btn):
            btns.addWidget(b)
        layout.addLayout(btns)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["trial", "state", "objective", "params"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 2)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log, 1)

        self._task_id = None
        self._rows_by_trial: dict[int, int] = {}   # trial number -> table row
        self._best_params: dict | None = None

    # ---- search-space row builders ----------------------------------------

    def _add_widget_row(self, grid, name, widgets, spec_fn, checked=False, hint=None):
        cb = QtWidgets.QCheckBox(name)
        cb.setChecked(checked)
        grid.addWidget(cb, self._grid_row, 0)
        holder = QtWidgets.QWidget()
        hb = QtWidgets.QHBoxLayout(holder)
        hb.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            hb.addWidget(w)
        if hint:
            label = QtWidgets.QLabel(hint)
            label.setWordWrap(True)
            hb.addWidget(label, 1)
        else:
            hb.addStretch(1)
        grid.addWidget(holder, self._grid_row, 1)
        self._grid_row += 1
        self.rows.append(_SweepRow(name, cb, widgets, spec_fn))
        return widgets

    def _add_range_row(self, grid, name, low, high, log=False, checked=False):
        lo = self._range_spin(low)
        hi = self._range_spin(high)
        log_cb = QtWidgets.QCheckBox("log scale")
        log_cb.setChecked(log)
        self._add_widget_row(
            grid, name, [lo, hi, log_cb],
            lambda: range_spec(lo.value(), hi.value(), log_cb.isChecked()),
            checked=checked)
        return lo, hi, log_cb

    def _add_choice_row(self, grid, name, values, labels=None, checked=(), hint=None):
        lst = _ChoiceList(values, labels=labels, checked=checked)
        self._add_widget_row(grid, name, [lst],
                             lambda: choice_spec(lst.checked_values()), hint=hint)
        return lst

    @staticmethod
    def _range_spin(value):
        w = NoWheelDoubleSpinBox()
        w.setDecimals(6)
        w.setRange(0.0, 1000.0)
        w.setSingleStep(1e-4)
        w.setValue(value)
        return w

    # ---- request assembly --------------------------------------------------

    def build_space(self) -> dict:
        """{param: spec} for every ticked row (empty rows included, so validation
        can report them)."""
        return {row.name: row.spec() for row in self.rows if row.enabled()}

    def build_latency(self):
        if not self.lat_enabled.isChecked():
            return None
        return latency_spec(self.lat_p99.value(),
                            batch_size=self.lat_batch.value(),
                            precision=self.lat_precision.currentText(),
                            target=self.lat_target.currentData())

    def build_request(self, base: dict) -> dict:
        return build_tuning_request(
            base, space=self.build_space(), n_trials=self.n_trials.value(),
            objective=self.objective.currentText(), latency=self.build_latency())

    def validate(self) -> str | None:
        """Mirror the server's 422s so the user gets an inline message."""
        space = self.build_space()
        if not space:
            return "Tick at least one hyperparameter to sweep."
        for name, spec in space.items():
            if "choices" in spec and not spec["choices"]:
                return f"'{name}' is ticked but has no choices selected."
            if "low" in spec and spec["low"] > spec["high"]:
                return f"'{name}': low is greater than high."
        return None

    def refresh(self) -> None:
        pass

    # ---- run ---------------------------------------------------------------

    def _start(self) -> None:
        client = self._client()
        if not client:
            return
        problem = self.validate()
        if problem:
            self.ctx.status(problem)
            self.log.appendPlainText(f"[tune] {problem}")
            return
        try:
            base = self._train_tab().build_config()
        except Exception as e:
            self.ctx.status(f"Could not build the baseline config: {e}")
            return
        if not base.get("label_paths") or not base.get("frame_dirs"):
            self.ctx.status("Need labels + extracted frames (preprocess first).")
            return

        self.table.setRowCount(0)
        self._rows_by_trial.clear()
        self._best_params = None
        self.apply_btn.setEnabled(False)
        self.log.appendPlainText("[tune] starting…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        def on_done():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.ctx.status("Sweep stream ended.")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.tuning_status(self._task_id), self._on_event, on_done)
            else:
                on_done()

        self._spawn_call(lambda: client.start_tuning(self.build_request(base)), start)

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_tuning(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))

    # ---- event stream ------------------------------------------------------

    def _row_for(self, trial: int) -> int:
        if trial not in self._rows_by_trial:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, self.COL_TRIAL, QtWidgets.QTableWidgetItem(str(trial)))
            for col in (self.COL_STATE, self.COL_VALUE, self.COL_PARAMS):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(""))
            self._rows_by_trial[trial] = row
        return self._rows_by_trial[trial]

    def _set(self, trial: int, col: int, text: str) -> None:
        self.table.item(self._row_for(trial), col).setText(text)

    @staticmethod
    def _fmt_params(params: dict) -> str:
        def one(v):
            return f"{v:.3g}" if isinstance(v, float) else str(v)
        return ", ".join(f"{k}={one(v)}" for k, v in (params or {}).items())

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("event")
        trial = ev.get("trial")

        if kind == "start":
            self.log.appendPlainText(
                f"[tune] {ev.get('n_trials')} trials, optimizing "
                f"{ev.get('objective')} ({ev.get('direction')})")
        elif kind == "trial_start":
            self._set(trial, self.COL_STATE, "running")
            self._set(trial, self.COL_PARAMS, self._fmt_params(ev.get("params")))
            self.log.appendPlainText(
                f"[trial {trial}] {self._fmt_params(ev.get('params'))}")
        elif kind == "latency":
            p99, budget = ev.get("p99_ms"), ev.get("budget_ms")
            if p99 is not None:
                self.log.appendPlainText(
                    f"[trial {trial}] p99 {p99:.2f} ms (budget {budget:.2f} ms) "
                    f"on {ev.get('device')} {ev.get('precision')}")
        elif kind == "latency_error":
            self.log.appendPlainText(
                f"[trial {trial}] latency check failed, constraint not applied: "
                f"{ev.get('error')}")
        elif kind == "trial_rejected":
            self._set(trial, self.COL_STATE, "rejected (slow)")
            self._set(trial, self.COL_VALUE, f"p99 {ev.get('p99_ms', 0):.1f} ms")
            self._grey(trial)
            self.log.appendPlainText(
                f"[trial {trial}] rejected: p99 {ev.get('p99_ms'):.2f} ms exceeds "
                f"{ev.get('budget_ms'):.2f} ms")
        elif kind == "epoch":
            metric = next((k for k in ev if k not in ("event", "trial", "epoch")), None)
            if metric:
                self._set(trial, self.COL_VALUE, f"{ev[metric]:.4f}")
                self._set(trial, self.COL_STATE, f"epoch {ev.get('epoch')}")
        elif kind == "trial_done":
            value = ev.get("value")
            self._set(trial, self.COL_STATE, "done")
            self._set(trial, self.COL_VALUE, "—" if value is None else f"{value:.4f}")
            self.log.appendPlainText(f"[trial {trial}] done: {value}")
        elif kind == "trial_pruned":
            self._set(trial, self.COL_STATE, "pruned")
            self._grey(trial)
        elif kind == "trial_failed":
            self._set(trial, self.COL_STATE,
                      "OOM" if ev.get("reason") == "oom" else "failed")
            self._grey(trial)
            self.log.appendPlainText(f"[trial {trial}] failed: {ev.get('error')}")
        elif kind == "done":
            best = ev.get("best_trial")
            self._best_params = ev.get("best_params") or None
            if best is not None:
                self._set(best, self.COL_STATE, "BEST")
                self.log.appendPlainText(
                    f"[tune] best trial {best}: {ev.get('best_value')} "
                    f"({self._fmt_params(self._best_params)})")
            else:
                self.log.appendPlainText("[tune] no trial produced a usable result.")
            self.apply_btn.setEnabled(self._best_params is not None)
        elif kind == "cancelled":
            self.log.appendPlainText("[tune] cancelled.")
        elif kind == "error":
            self.log.appendPlainText(f"[tune] error: {ev.get('error')}")

    def _grey(self, trial: int) -> None:
        row = self._row_for(trial)
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setForeground(QtGui.QBrush(QtGui.QColor(128, 128, 128)))

    # ---- apply -------------------------------------------------------------

    def _train_tab(self):
        """The sibling TrainTab, wired by PipelinePanel. Set explicitly rather
        than walked up the parent chain — QTabWidget reparents its pages into an
        internal stack, so the chain is an implementation detail."""
        if self.train_tab is None:
            raise RuntimeError("TuneTab has no Train tab to take its baseline from.")
        return self.train_tab

    def _apply_best(self) -> None:
        train = self.train_tab
        if train is None or not self._best_params:
            return
        p = self._best_params
        setters = {
            "lr": train.lr.setValue,
            "weight_decay": train.weight_decay.setValue,
            "dropout": train.dropout.setValue,
            "batch_size": lambda v: train.batch_size.setValue(int(v)),
            "image_size": lambda v: train.image_size.setValue(int(v)),
            "frames_per_group": lambda v: train.frames_per_group.setValue(int(v)),
            "variant": train.variant.setCurrentText,
            "temporal_pool": train.temporal_pool.setCurrentText,
            "tsm_stages": lambda v: train.tsm_stages.setText(str(v)),
        }
        applied = []
        for name, value in p.items():
            setter = setters.get(name)
            if setter:
                setter(value)
                applied.append(name)
        self.ctx.status(f"Applied to Train tab: {', '.join(applied)}")
        self.log.appendPlainText(f"[tune] applied best params: {self._fmt_params(p)}")


class ExportTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)

        bench = QtWidgets.QGroupBox("Benchmark inference (size your compute budget)")
        bf = QtWidgets.QFormLayout(bench)
        self.b_variant = NoWheelComboBox()
        self.b_variant.addItems(VARIANTS)
        self.b_frames = NoWheelSpinBox()
        self.b_frames.setRange(1, 32)
        self.b_frames.setValue(3)
        self.b_stages = QtWidgets.QLineEdit("5,6,7")
        self.b_pool = NoWheelComboBox()
        self.b_pool.addItems(POOLS)
        self.b_uni = QtWidgets.QCheckBox("Unidirectional")
        self.b_prec = NoWheelComboBox()
        self.b_prec.addItems(PRECISIONS)
        self.b_target = NoWheelComboBox()
        for t in TARGETS:
            self.b_target.addItem(TARGET_LABELS[t], t)
        self.b_size = NoWheelSpinBox()
        self.b_size.setRange(32, 800)
        self.b_size.setValue(224)
        bf.addRow("runtime", self.b_target)
        bf.addRow("variant", self.b_variant)
        bf.addRow("num_frames", self.b_frames)
        bf.addRow("tsm_stages", self.b_stages)
        bf.addRow("temporal_pool", self.b_pool)
        bf.addRow("", self.b_uni)
        bf.addRow("image_size", self.b_size)
        bf.addRow("precision", self.b_prec)
        bench_btn = QtWidgets.QPushButton("Run benchmark")
        mark_accent(bench_btn)
        bench_btn.clicked.connect(self._benchmark)
        bf.addRow(bench_btn)
        self.bench_result = QtWidgets.QLabel("—")
        self.bench_result.setWordWrap(True)
        bf.addRow(self.bench_result)
        layout.addWidget(bench)

        export = QtWidgets.QGroupBox("Export for deployment")
        ef = QtWidgets.QFormLayout(export)
        self.fmt = NoWheelComboBox()
        self.fmt.addItems(EXPORT_FORMATS)
        self.fmt.setCurrentText("onnx")
        self.e_size = NoWheelSpinBox()
        self.e_size.setRange(32, 800)
        self.e_size.setValue(224)
        self.e_prec = NoWheelComboBox()
        self.e_prec.addItems(PRECISIONS)
        # Must match how the checkpoint was trained: the load is strict=False,
        # so a mismatch produces a silently wrong model rather than an error.
        self.e_stages = QtWidgets.QLineEdit("5,6,7")
        self.ckpt = QtWidgets.QLineEdit()
        ckpt_btn = QtWidgets.QPushButton("Browse…")
        ckpt_btn.clicked.connect(self._pick_ckpt)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.ckpt, 1)
        crow.addWidget(ckpt_btn)
        self.out_dir = QtWidgets.QLineEdit()
        self.out_name = QtWidgets.QLineEdit("tsm_model")
        ef.addRow("format", self.fmt)
        ef.addRow("checkpoint", crow)
        ef.addRow("output dir", self.out_dir)
        ef.addRow("output name", self.out_name)
        ef.addRow("tsm_stages (as trained)", self.e_stages)
        ef.addRow("image_size", self.e_size)
        ef.addRow("precision", self.e_prec)
        # Only the deployment formats care about input shape / precision —
        # TorchScript is traced at the checkpoint's own shape.
        self.fmt.currentTextChanged.connect(self._sync_export_fields)
        self._sync_export_fields(self.fmt.currentText())
        export_btn = QtWidgets.QPushButton("Export")
        mark_accent(export_btn)
        export_btn.clicked.connect(self._export)
        ef.addRow(export_btn)
        layout.addWidget(export)
        layout.addStretch(1)

    def refresh(self) -> None:
        pass

    def build_benchmark_kwargs(self) -> dict:
        return {
            "variant": self.b_variant.currentText(),
            "num_frames": self.b_frames.value(),
            "tsm_stages": parse_stages(self.b_stages.text()),
            "temporal_pool": self.b_pool.currentText(),
            "unidirectional": self.b_uni.isChecked(),
            "image_size": self.b_size.value(),
            "precision": self.b_prec.currentText(),
            "target": self.b_target.currentData(),
        }

    def _benchmark(self) -> None:
        client = self._client()
        if not client:
            return
        # An engine build (TensorRT especially) is slow enough that silence
        # reads as a hang.
        self.bench_result.setText("Exporting and building the engine — this can "
                                  "take a few minutes the first time…")

        def show(res: dict):
            def ms(key):
                v = res.get(key)
                return f"{v:.2f}" if isinstance(v, (int, float)) else "?"
            self.bench_result.setText(
                f"[{res.get('runtime', '?')}] p50 {ms('ms_per_batch_p50')} ms · "
                f"p95 {ms('ms_per_batch_p95')} ms · p99 {ms('ms_per_batch_p99')} ms "
                f"({ms('ms_per_clip')} ms/clip) on {res.get('device_name', '?')} "
                f"{res.get('precision', '')}")

        self._spawn_call(
            lambda: client.benchmark_inference(**self.build_benchmark_kwargs()), show)

    def _pick_ckpt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Checkpoint", "", "Checkpoints (*.ckpt *.pt);;All files (*)")
        if path:
            self.ckpt.setText(path)

    def _sync_export_fields(self, fmt: str) -> None:
        deployment = fmt != "torchscript"
        self.e_size.setEnabled(deployment)
        self.e_prec.setEnabled(deployment)

    def _export(self) -> None:
        client = self._client()
        if not client or not self.ckpt.text() or not self.out_dir.text():
            self.ctx.status("Need a checkpoint and output dir.")
            return
        fmt = self.fmt.currentText()
        name = self.out_name.text() or "tsm_model"
        stages = parse_stages(self.e_stages.text()) or None

        if fmt == "torchscript":
            call = lambda: client.export_torchscript(  # noqa: E731
                self.ckpt.text(), self.out_dir.text(), name, tsm_stages=stages)
        else:
            call = lambda: client.export_optimized(  # noqa: E731
                self.ckpt.text(), self.out_dir.text(), name, fmt=fmt,
                tsm_stages=stages, image_size=self.e_size.value(),
                precision=self.e_prec.currentText())

        self.ctx.status(f"Exporting {fmt}…")
        self._spawn_call(
            call, lambda res: self.ctx.status(f"Exported: {res.get('output_path')}"))


class PipelinePanel(QtWidgets.QTabWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        from .active_learning_panel import ActiveLearningPanel  # lazy: avoids import cycle
        self.preprocess = PreprocessTab(ctx)
        self.train = TrainTab(ctx)
        self.tune = TuneTab(ctx, train_tab=self.train)
        self.export = ExportTab(ctx)
        self.active_learning = ActiveLearningPanel(ctx)
        self.addTab(self.preprocess, "Preprocess")
        self.addTab(self.train, "Train")
        self.addTab(self.tune, "Tune")
        self.addTab(self.export, "Export")
        self.addTab(self.active_learning, "Active Learning")

    def refresh(self) -> None:
        self.preprocess.refresh()
        self.train.refresh()
        self.tune.refresh()
        self.export.refresh()
        self.active_learning.refresh()

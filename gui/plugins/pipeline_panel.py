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
"""

from __future__ import annotations

from PySide6 import QtWidgets

from labeling_shell.backend_worker import CallWorker, StreamWorker
from labeling_shell.frame_ranges import compute_labeled_frame_ranges
from labeling_shell.theme import mark_accent

from .pipeline_config import CROP_DEFAULTS, build_train_config, parse_stages

VARIANTS = ["b0", "b1", "b2", "b3"]
POOLS = ["last", "mean", "max"]


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
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        form = QtWidgets.QFormLayout(self)

        self.video_label = QtWidgets.QLabel("—")
        form.addRow("Video", self.video_label)

        self.bg_edit = QtWidgets.QLineEdit()
        bg_btn = QtWidgets.QPushButton("Compute background")
        bg_btn.clicked.connect(self._compute_background)
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.addWidget(self.bg_edit, 1)
        bg_row.addWidget(bg_btn)
        form.addRow("Background", bg_row)

        self.crop_widgets: dict[str, QtWidgets.QWidget] = {}
        for key, default in CROP_DEFAULTS.items():
            if key == "difference_mode":
                w = QtWidgets.QComboBox()
                w.addItems(["directional", "absolute"])
            elif isinstance(default, float):
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(0.0, 100000.0)
                w.setDecimals(3)
                w.setValue(default)
            else:
                w = QtWidgets.QSpinBox()
                w.setRange(0, 100000)
                w.setValue(default)
            self.crop_widgets[key] = w
            form.addRow(key, w)

        self.progress = QtWidgets.QProgressBar()
        run_btn = QtWidgets.QPushButton("Run preprocessing")
        mark_accent(run_btn)
        run_btn.clicked.connect(self._run)
        form.addRow(run_btn)
        form.addRow(self.progress)

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
        return out

    def _compute_background(self) -> None:
        client = self._client()
        video = self.ctx.video_path()
        if not client or not video:
            return
        self.ctx.status("Computing background…")
        self._spawn_call(
            lambda: client.compute_background(video),
            lambda res: (self.bg_edit.setText(res.get("path", "")),
                         self.ctx.status("Background computed.")),
        )

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

        self.ctx.status("Preprocessing…")
        self._spawn_call(
            lambda: client.run_preprocessing(
                video, out_dir, crop_params=self.build_crop_params(),
                frame_ranges=ranges, bg_path=self.bg_edit.text() or None),
            start,
        )


class TrainTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.frames_per_group = QtWidgets.QSpinBox()
        self.frames_per_group.setRange(1, 32)
        self.frames_per_group.setValue(3)
        self.variant = QtWidgets.QComboBox()
        self.variant.addItems(VARIANTS)
        self.tsm_stages = QtWidgets.QLineEdit("5,6,7")
        self.temporal_pool = QtWidgets.QComboBox()
        self.temporal_pool.addItems(POOLS)
        self.unidirectional = QtWidgets.QCheckBox("Unidirectional (causal)")
        self.pretrained = QtWidgets.QCheckBox("Pretrained")
        self.pretrained.setChecked(True)
        self.dropout = self._dspin(0.2, 0.0, 0.9, 0.05)
        self.lr = self._dspin(1e-4, 1e-6, 1.0, 1e-4, decimals=6)
        self.weight_decay = self._dspin(1e-3, 0.0, 1.0, 1e-4, decimals=6)
        self.max_epochs = self._ispin(100, 1, 100000)
        self.patience = self._ispin(20, 1, 100000)
        self.binarize = QtWidgets.QComboBox()

        for label, w in [
            ("frames/group", self.frames_per_group), ("variant", self.variant),
            ("tsm_stages", self.tsm_stages), ("temporal_pool", self.temporal_pool),
            ("", self.unidirectional), ("", self.pretrained), ("dropout", self.dropout),
            ("lr", self.lr), ("weight_decay", self.weight_decay),
            ("max_epochs", self.max_epochs), ("early stop patience", self.patience),
            ("binarize (positive class)", self.binarize),
        ]:
            form.addRow(label, w)
        layout.addLayout(form)

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
        layout.addLayout(btns)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        self._task_id = None

    def _dspin(self, val, lo, hi, step, decimals=4):
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(val)
        return w

    def _ispin(self, val, lo, hi):
        w = QtWidgets.QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    def refresh(self) -> None:
        current = self.binarize.currentText()
        self.binarize.clear()
        self.binarize.addItem("(none)")
        for b in self.ctx.behaviors():
            self.binarize.addItem(b.name)
        i = self.binarize.findText(current)
        if i >= 0:
            self.binarize.setCurrentIndex(i)

    def build_config(self) -> dict:
        binarize = self.binarize.currentText()
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
            max_epochs=self.max_epochs.value(),
            early_stopping_patience=self.patience.value(),
            binarize=None if binarize in ("", "(none)") else binarize,
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
        self.log.appendPlainText("[train] starting…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        def on_event(ev: dict):
            self.log.appendPlainText(str(ev))

        def on_done():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.ctx.status("Training stream ended.")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.training_status(self._task_id), on_event, on_done)

        self._spawn_call(lambda: client.start_training(self.build_config()), start)

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_training(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))


class ExportTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)

        bench = QtWidgets.QGroupBox("Benchmark inference (size your compute budget)")
        bf = QtWidgets.QFormLayout(bench)
        self.b_variant = QtWidgets.QComboBox()
        self.b_variant.addItems(VARIANTS)
        self.b_frames = QtWidgets.QSpinBox()
        self.b_frames.setRange(1, 32)
        self.b_frames.setValue(3)
        self.b_stages = QtWidgets.QLineEdit("5,6,7")
        self.b_pool = QtWidgets.QComboBox()
        self.b_pool.addItems(POOLS)
        self.b_uni = QtWidgets.QCheckBox("Unidirectional")
        self.b_prec = QtWidgets.QComboBox()
        self.b_prec.addItems(["fp32", "fp16"])
        bf.addRow("variant", self.b_variant)
        bf.addRow("num_frames", self.b_frames)
        bf.addRow("tsm_stages", self.b_stages)
        bf.addRow("temporal_pool", self.b_pool)
        bf.addRow("", self.b_uni)
        bf.addRow("precision", self.b_prec)
        bench_btn = QtWidgets.QPushButton("Run benchmark")
        mark_accent(bench_btn)
        bench_btn.clicked.connect(self._benchmark)
        bf.addRow(bench_btn)
        self.bench_result = QtWidgets.QLabel("—")
        self.bench_result.setWordWrap(True)
        bf.addRow(self.bench_result)
        layout.addWidget(bench)

        export = QtWidgets.QGroupBox("Export TorchScript")
        ef = QtWidgets.QFormLayout(export)
        self.ckpt = QtWidgets.QLineEdit()
        ckpt_btn = QtWidgets.QPushButton("Browse…")
        ckpt_btn.clicked.connect(self._pick_ckpt)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.ckpt, 1)
        crow.addWidget(ckpt_btn)
        self.out_dir = QtWidgets.QLineEdit()
        self.out_name = QtWidgets.QLineEdit("tsm_model")
        ef.addRow("checkpoint", crow)
        ef.addRow("output dir", self.out_dir)
        ef.addRow("output name", self.out_name)
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
            "precision": self.b_prec.currentText(),
        }

    def _benchmark(self) -> None:
        client = self._client()
        if not client:
            return
        self.bench_result.setText("Running…")
        self._spawn_call(
            lambda: client.benchmark_inference(**self.build_benchmark_kwargs()),
            lambda res: self.bench_result.setText(
                f"{res.get('ms_per_clip', '?')} ms/clip, "
                f"{res.get('clips_per_sec', '?')} clips/s on {res.get('device_name', '?')}"),
        )

    def _pick_ckpt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Checkpoint", "", "Checkpoints (*.ckpt *.pt);;All files (*)")
        if path:
            self.ckpt.setText(path)

    def _export(self) -> None:
        client = self._client()
        if not client or not self.ckpt.text() or not self.out_dir.text():
            self.ctx.status("Need a checkpoint and output dir.")
            return
        self._spawn_call(
            lambda: client.export_torchscript(
                self.ckpt.text(), self.out_dir.text(), self.out_name.text() or "tsm_model"),
            lambda res: self.ctx.status(f"Exported: {res.get('output_path')}"),
        )


class PipelinePanel(QtWidgets.QTabWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        from .active_learning_panel import ActiveLearningPanel  # lazy: avoids import cycle
        self.preprocess = PreprocessTab(ctx)
        self.train = TrainTab(ctx)
        self.export = ExportTab(ctx)
        self.active_learning = ActiveLearningPanel(ctx)
        self.addTab(self.preprocess, "Preprocess")
        self.addTab(self.train, "Train")
        self.addTab(self.export, "Export")
        self.addTab(self.active_learning, "Active Learning")

    def refresh(self) -> None:
        self.preprocess.refresh()
        self.train.refresh()
        self.export.refresh()
        self.active_learning.refresh()

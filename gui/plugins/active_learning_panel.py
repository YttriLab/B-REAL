"""Active-learning panel (Phase 4).

Run inference on the current video, then review: the uncertainty timeline and the
on-video HUD (both owned by the main window) light up, and "Next uncertain" jumps
to the frames the model is least sure about — where new labels help most.

Results are saved into the project (predictions/<video>.json) with what made
them, and reloaded when the video is opened again.

Relabeling happens here too: a readout of the current frame's prediction vs.
your label, and one button per behavior that does exactly what its number key
does (first press starts a bout, second ends it; a new bout replaces the labels
it covers). Ctrl+Z undoes.

Additional ctx duck-type used here (on top of pipeline_panel's):
    crop_params() -> dict        current_bg_path() -> str | None
    set_al_results(state, meta)  set_al_threshold(t)
    al_state() -> AlState|None   current_frame() -> int      seek(frame)
    (optional) video_id() -> str|None    al_meta() -> dict|None
               save_predictions(video_id, results, meta)
               toggle_behavior(key)          is_recording(behavior_id) -> bool
               project_models() -> list[dict] model_entry(path) -> dict|None
"""

from __future__ import annotations

import datetime
import os

from PySide6 import QtCore, QtWidgets

from labeling_shell.active_learning import AlState
from labeling_shell.behavior_colors import behavior_colors
from labeling_shell.no_wheel import NoWheelComboBox
from labeling_shell.theme import mark_accent

from .crop_summary import differences, summarize
from .pipeline_config import DECODERS, INFERENCE_RUNTIMES
from .pipeline_panel import _Base


class ActiveLearningPanel(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self._task_id = None
        form = QtWidgets.QFormLayout(self)

        # Models trained in this project (their entries know their training crop).
        self.models = NoWheelComboBox()
        self.models.currentIndexChanged.connect(self._on_model_chosen)
        form.addRow("Project model", self.models)
        self.ckpt = QtWidgets.QLineEdit()
        self.ckpt.textChanged.connect(lambda _t: self._update_crop_info())
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._pick_ckpt)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.ckpt, 1)
        crow.addWidget(browse)
        form.addRow("Checkpoint", crow)

        # A model only works on crops like the ones it was trained on, so by
        # default inference crops the way its training frames were made.
        self.crop_source = NoWheelComboBox()
        self.crop_source.addItem("This model's training crop", "model")
        self.crop_source.addItem("The Preprocess tab's settings", "preprocess")
        self.crop_source.setToolTip(
            "Training crop: what the model's training frames were made with "
            "(recorded when it was trained in this app). Preprocess tab: whatever "
            "it shows now, e.g. to try other settings.")
        self.crop_source.currentIndexChanged.connect(lambda _i: self._update_crop_info())
        form.addRow("Crop from", self.crop_source)
        self.crop_info = QtWidgets.QLabel()
        self.crop_info.setWordWrap(True)
        self.crop_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow("Cropping", self.crop_info)

        self.decoder = NoWheelComboBox()
        for label, dec in DECODERS:
            self.decoder.addItem(label, dec)
        self.decoder.setToolTip(
            "Auto: decode on the GPU (NVDEC) when it reproduces OpenCV's frames "
            "bit-for-bit on this video — monochrome videos — otherwise OpenCV.")
        form.addRow("Decoder", self.decoder)

        self.runtime = NoWheelComboBox()
        for label, rt, prec in INFERENCE_RUNTIMES:
            self.runtime.addItem(label, (rt, prec))
        self.runtime.setToolTip(
            "TensorRT compiles the classifier for this GPU: built from the "
            "checkpoint on first use (~0.5-2 min) and cached, then reused.")
        form.addRow("Runtime", self.runtime)

        self.threshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(50)
        self.threshold.valueChanged.connect(self._on_threshold)
        self.threshold_label = QtWidgets.QLabel("0.50")
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(self.threshold, 1)
        trow.addWidget(self.threshold_label)
        form.addRow("Uncertainty ≥", trow)

        self.progress = QtWidgets.QProgressBar()
        form.addRow(self.progress)

        # Which run the predictions on screen came from, so stale ones (an
        # older checkpoint, other crop settings) are recognisable.
        self.pred_info = QtWidgets.QLabel()
        self.pred_info.setWordWrap(True)
        self.pred_info.setProperty("muted", "true")
        form.addRow("Predictions", self.pred_info)

        self.frame_info = QtWidgets.QLabel("—")
        self.frame_info.setWordWrap(True)
        self.frame_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow("This frame", self.frame_info)
        self.behavior_box = QtWidgets.QWidget()
        self.behavior_grid = QtWidgets.QGridLayout(self.behavior_box)
        self.behavior_grid.setContentsMargins(0, 0, 0, 0)
        self.behavior_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._button_spec = None
        form.addRow("Label", self.behavior_box)
        self.refresh()

        run_btn = QtWidgets.QPushButton("Run inference (current video)")
        mark_accent(run_btn)
        run_btn.clicked.connect(self._run)
        next_btn = QtWidgets.QPushButton("Next uncertain →")
        next_btn.clicked.connect(self._next_uncertain)
        form.addRow(run_btn)
        form.addRow(next_btn)

    def refresh(self) -> None:
        self._refresh_models()
        self._update_crop_info()
        self._refresh_behavior_buttons()
        if getattr(self.ctx, "al_state", lambda: None)() is None:
            self.pred_info.setText("None for this video yet.")
            self.pred_info.setToolTip("")
            return
        meta = getattr(self.ctx, "al_meta", lambda: None)() or {}
        ckpt = meta.get("checkpoint")
        if not ckpt:
            self.pred_info.setText("Saved by an earlier version of the app "
                                   "(model and settings not recorded).")
            self.pred_info.setToolTip("")
            return
        when = (meta.get("created") or "")[:16].replace("T", " ")
        crop = (meta.get("crop_params") or {}).get("method", "?")
        self.pred_info.setText(f"{os.path.basename(ckpt)} · {crop} crop · {when}")
        self.pred_info.setToolTip(ckpt)

    # ---------------------------------------------------------------- crop
    def _refresh_models(self) -> None:
        entries = [m for m in getattr(self.ctx, "project_models", lambda: [])()
                   if m.get("kind") != "torchscript" and m.get("path")]
        spec = [m["path"] for m in entries]
        if spec == getattr(self, "_models_spec", None):
            return
        self._models_spec = spec
        self.models.blockSignals(True)
        self.models.clear()
        self.models.addItem("— choose, or browse for a checkpoint —", None)
        for m in entries:
            score = f" · val MCC {m['val_mcc']:.3f}" if isinstance(m.get("val_mcc"), float) else ""
            self.models.addItem(f"{m.get('name')}{score} · {m.get('created', '')}", m["path"])
        idx = self.models.findData(self.ckpt.text())
        self.models.setCurrentIndex(max(idx, 0))
        self.models.blockSignals(False)

    def _on_model_chosen(self, _i: int) -> None:
        path = self.models.currentData()
        if path:
            self.ckpt.setText(path)

    def _effective_crop(self):
        """(crop params to use, where they came from, warnings)."""
        current = getattr(self.ctx, "crop_params", lambda: {})() or {}
        entry = getattr(self.ctx, "model_entry", lambda _p: None)(self.ckpt.text()) or {}
        trained = entry.get("crop_params")
        note = entry.get("crop_note") or "not recorded for this model"
        if self.crop_source.currentData() == "model":
            if trained:
                return trained, "from this model's training", []
            return current, "Preprocess tab", [f"Training crop {note}: using the Preprocess tab."]
        warnings = []
        if trained:
            warnings = [f"Differs from training: {d}" for d in differences(trained, current)]
        return current, "Preprocess tab", warnings

    def _update_crop_info(self) -> None:
        if not hasattr(self, "crop_info"):
            return
        params, source, warnings = self._effective_crop()
        text = f"{summarize(params)} <i>({source})</i>"
        if warnings:
            text += "".join(f'<br><span style="color:#e5a50a">{w}</span>' for w in warnings)
        self.crop_info.setText(text)

    # ------------------------------------------------------------ relabel
    def _refresh_behavior_buttons(self) -> None:
        behaviors = list(getattr(self.ctx, "behaviors", lambda: [])())
        recording = getattr(self.ctx, "is_recording", lambda _i: False)
        spec = [(b.id, b.name, b.key, recording(b.id)) for b in behaviors]
        if spec == self._button_spec:
            return
        self._button_spec = spec
        while self.behavior_grid.count():
            self.behavior_grid.takeAt(0).widget().deleteLater()
        self.behavior_buttons = {}
        if not behaviors:
            self.behavior_grid.addWidget(QtWidgets.QLabel("Add behaviors in the Label tab."), 0, 0)
            return
        colors = behavior_colors([b.name for b in behaviors])
        for i, (bid, name, key, rec) in enumerate(spec):
            btn = QtWidgets.QPushButton(f"{key}  {name}" + ("  ● recording" if rec else ""))
            btn.setToolTip(f"Recording {name}: press to end the bout on this frame" if rec
                           else f"Start a {name} bout on this frame (same as key {key})")
            border = colors[name].name()
            btn.setStyleSheet(f"border-left: 6px solid {border};"
                              + (f"background: {border}; color: #111;" if rec else ""))
            btn.clicked.connect(lambda _=False, k=key: self._toggle(k))
            self.behavior_grid.addWidget(btn, i // 2, i % 2)
            self.behavior_buttons[bid] = btn

    def _toggle(self, key: str) -> None:
        toggle = getattr(self.ctx, "toggle_behavior", None)
        if toggle:
            toggle(key)

    def show_frame(self, frame: int, info: dict) -> None:
        """Readout for the current frame: the model's call vs. your label."""
        label = info.get("label")
        labeled = f"labeled <b>{label}</b>" if label else "unlabeled"
        if info.get("predicted") is None:
            self.frame_info.setText(f"{frame} — no prediction · {labeled}")
            return
        conf = info.get("confidence")
        pred = f"predicted <b>{info['predicted']}</b>" + (f" ({conf:.2f})" if conf is not None else "")
        text = f"{frame} — {pred} · {labeled}"
        if info.get("mismatch"):
            text = f'<span style="color:#ff5c5c">{text} · disagrees</span>'
        self.frame_info.setText(text)

    # ------------------------------------------------------------- handlers
    def _pick_ckpt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Checkpoint", "", "Checkpoints (*.ckpt *.pt);;All files (*)")
        if path:
            self.ckpt.setText(path)

    def _threshold_value(self) -> float:
        return self.threshold.value() / 100.0

    def _on_threshold(self, _v: int) -> None:
        self.threshold_label.setText(f"{self._threshold_value():.2f}")
        self.ctx.set_al_threshold(self._threshold_value())

    def _run(self) -> None:
        client = self._client()
        video = self.ctx.video_path()
        if not client or not video:
            return
        # The results belong to this video even if another is open when they land.
        video_id = getattr(self.ctx, "video_id", lambda: None)()
        if not self.ckpt.text():
            self.ctx.status("Choose a checkpoint first.")
            return
        self.progress.setValue(0)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            if ev.get("message") and ev.get("status") == "running":
                self.ctx.status(ev["message"])

        def on_stream_done():
            if self._task_id:
                self._spawn_call(
                    lambda: client.inference_results(self._task_id),
                    lambda res: self._on_results(res, video_id, meta))

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.inference_status(self._task_id), on_event, on_stream_done)

        # Snapshot on the GUI thread; the lambda runs on a worker.
        crop_params, crop_source, _warnings = self._effective_crop()
        decoder = self.decoder.currentData()
        runtime, precision = self.runtime.currentData()
        meta = {"checkpoint": self.ckpt.text(), "video": video,
                "crop_params": crop_params, "crop_source": crop_source,
                "frames_per_group": self.ctx.frames_per_group(),
                "decoder": decoder, "runtime": runtime, "precision": precision,
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
        self.ctx.status("Running inference…")
        self._spawn_call(
            lambda: client.run_inference(
                self.ckpt.text(), video_path=video,
                bg_path=self.ctx.current_bg_path(),
                crop_params=crop_params,
                frames_per_group=self.ctx.frames_per_group(),
                frame_ranges=None,
                decoder=decoder, runtime=runtime, precision=precision),
            start,
        )

    def _on_results(self, results: dict, video_id=None, meta: dict | None = None) -> None:
        saver = getattr(self.ctx, "save_predictions", None)
        if saver and video_id:
            saver(video_id, results, meta or {})
        if video_id and video_id != getattr(self.ctx, "video_id", lambda: video_id)():
            self.ctx.status(f"Inference for {video_id} finished and was saved; "
                            "open that video to review it.")
            return
        state = AlState()
        state.load_results(results)
        self.ctx.set_al_results(state, meta)
        n = len(state.by_index)
        self.ctx.status(f"Inference done — {n} frames, classes {state.class_names}")

    def _next_uncertain(self) -> None:
        al = self.ctx.al_state()
        if not al:
            self.ctx.status("Run inference first.")
            return
        nxt = al.next_uncertain(self.ctx.current_frame(), self._threshold_value())
        if nxt is None:
            self.ctx.status("No more uncertain frames above threshold.")
        else:
            self.ctx.seek(nxt)
            self.ctx.status(f"Jumped to uncertain frame {nxt}")

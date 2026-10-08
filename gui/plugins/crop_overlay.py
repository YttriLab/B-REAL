"""Crop box over the video: where the TSM crop is taken on the frame on screen.

Two sources, in order:

- **recorded** (solid green): the box preprocessing actually cropped this
  frame with, from ``frames/<id>/crops.csv``. Exactly what a training sample
  saw; only frames that were extracted have one.
- **live** (dashed amber): asked of the backend (``/api/preprocess/crop-box``)
  with the Preprocess tab's current settings, the same FrameCropper. Works on
  any frame, but is only requested once the player rests (a short debounce, so
  playback and scrubbing don't flood the backend), and its smoothing restarts
  after a jump — so it can differ a little from a recorded box.

``ctx`` (MainWindow) provides: client(), video_path(), current_frames_dir(),
current_bg_path(), crop_params(), status(msg).
"""

from __future__ import annotations

import csv
import json
import os
import time

from PySide6 import QtCore

from labeling_shell.backend_worker import CallWorker

CROPS_CSV = "crops.csv"  # written by backend/services/preprocess_service.py
RECORDED_COLOR = "#3fb950"
LIVE_COLOR = "#e3b341"
LEGEND = ("Crop box: solid green = recorded by preprocessing; "
          "dashed amber = live, from the Preprocess tab's settings.")


def read_crop_boxes(path: str) -> dict[int, tuple]:
    """crops.csv -> {frame: (x, y, w, h)} in native video px; {} if absent."""
    boxes = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                boxes[int(row["frame"])] = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
    except (OSError, KeyError, ValueError):
        return {}
    return boxes


class CropBoxOverlay(QtCore.QObject):
    DEBOUNCE_MS = 100
    RECHECK_S = 2.0  # how often to look for a re-written crops.csv
    MAX_LIVE = 5000

    def __init__(self, ctx, player, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.player = player
        self.enabled = False
        self._recorded: dict = {}
        self._csv_path: str | None = None
        self._csv_mtime: float | None = None
        self._csv_checked = 0.0
        self._live: dict = {}
        self._live_key: str | None = None
        self._failed_key: str | None = None
        self._worker: CallWorker | None = None
        self._again = False
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._request_live)
        player.frameChanged.connect(self._on_frame)

    # ---------------------------------------------------------------- control
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if self.enabled:
            self.video_changed()
        else:
            self._debounce.stop()
            self.player.view.set_boxes([])

    def video_changed(self) -> None:
        """The open video (or its frames dir) changed: drop everything cached."""
        self._recorded, self._csv_path, self._csv_mtime, self._csv_checked = {}, None, None, 0.0
        self._live, self._live_key, self._failed_key = {}, None, None
        self.refresh()

    def refresh(self) -> None:
        if self.enabled and self.player.reader is not None:
            self._show(self.player.index)
        else:
            self.player.view.set_boxes([])

    # ------------------------------------------------------------------ show
    def _on_frame(self, index: int) -> None:
        if self.enabled:
            self._show(index)

    def _show(self, index: int) -> None:
        rec = self._recorded_boxes().get(index)
        if rec is not None:
            self.player.view.set_boxes(
                [{"rect": rec, "color": RECORDED_COLOR, "label": "crop"}])
            return
        key = self._current_live_key()
        box = self._live.get(index) if key is not None and key == self._live_key else None
        if box is not None:
            self.player.view.set_boxes(
                [{"rect": box, "color": LIVE_COLOR, "dashed": True, "label": "crop (live)"}])
            return
        self.player.view.set_boxes([])  # never leave the previous frame's box up
        if key is not None and key != self._failed_key:
            self._debounce.start()

    def _recorded_boxes(self) -> dict:
        frames_dir = self.ctx.current_frames_dir()
        path = os.path.join(frames_dir, CROPS_CSV) if frames_dir else None
        now = time.monotonic()
        if path != self._csv_path or now - self._csv_checked > self.RECHECK_S:
            self._csv_checked = now
            try:
                mtime = os.path.getmtime(path) if path else None
            except OSError:
                mtime = None
            if path != self._csv_path or mtime != self._csv_mtime:
                self._csv_path, self._csv_mtime = path, mtime
                self._recorded = read_crop_boxes(path) if mtime is not None else {}
        return self._recorded

    # ------------------------------------------------------------------ live
    def _live_request(self):
        """(client, video, params, bg) for a live box, or None if there can't be one."""
        client, video = self.ctx.client(), self.ctx.video_path()
        if client is None or not video:
            return None
        params = self.ctx.crop_params()
        method = params.get("method", "background")
        if method == "none":
            return None  # the "crop" is the whole frame
        bg = self.ctx.current_bg_path() if method == "background" else None
        if method == "background" and not bg:
            return None  # nothing to subtract yet (Compute background)
        return client, video, params, bg

    def _current_live_key(self) -> str | None:
        req = self._live_request()
        if req is None:
            return None
        _client, video, params, bg = req
        key = json.dumps([video, params, bg], sort_keys=True, default=str)
        if key != self._live_key:  # settings changed: old live boxes are stale
            self._live, self._live_key = {}, key
        return key

    def _request_live(self) -> None:
        if not self.enabled or self.player.reader is None:
            return
        if self._worker is not None:
            self._again = True  # ask for wherever the player is once this lands
            return
        req = self._live_request()
        if req is None:
            return
        client, video, params, bg = req
        key = self._current_live_key()
        index = self.player.index
        if index in self._live or index in self._recorded_boxes():
            return
        w = CallWorker(lambda: client.crop_box(video, index, params, bg))
        w.done.connect(lambda res, k=key, i=index: self._on_live(k, i, res))
        w.failed.connect(lambda msg, k=key: self._on_live_failed(k, msg))
        w.finished.connect(self._on_worker_finished)
        self._worker = w
        w.start()

    def _on_live(self, key, index, res) -> None:
        if key != self._live_key:
            return
        if len(self._live) >= self.MAX_LIVE:
            self._live.clear()
        self._live[index] = tuple(res["box"])
        if self.enabled and self.player.index == index:
            self._show(index)

    def _on_live_failed(self, key, msg: str) -> None:
        # Once per settings: a bad setting would otherwise fail on every frame.
        self._failed_key = key
        self.ctx.status(f"Live crop box unavailable: {msg}")

    def _on_worker_finished(self) -> None:
        self._worker = None
        if self._again:
            self._again = False
            if self.enabled and self.player.reader is not None:
                self._show(self.player.index)

    def shutdown(self) -> None:
        self._debounce.stop()
        if self._worker is not None:
            self._worker.wait(5000)

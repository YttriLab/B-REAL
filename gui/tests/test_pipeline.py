"""Phase 3: pipeline config building, panels (offscreen), and backend workers.

Live preprocess/train/export need the backend's CUDA/torch env; this validates
the config marshalling, the widgets' construction/refresh, and the threaded
worker plumbing in isolation.

    QT_QPA_PLATFORM=offscreen .venv-gui/Scripts/python.exe gui/tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.backend_worker import CallWorker, StreamWorker  # noqa: E402
from labeling_shell.bouts import LabelModel  # noqa: E402
from plugins.pipeline_config import (  # noqa: E402
    CROP_DEFAULTS, build_train_config, parse_stages,
)
from plugins.pipeline_panel import PipelinePanel, PreprocessTab, TrainTab  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeCtx:
    def __init__(self):
        self._client = None
        self.model = LabelModel()
        self.msgs = []

    def client(self):
        return self._client

    def video_path(self):
        return "v.mp4"

    def fps(self):
        return 30.0

    def bouts(self):
        return self.model.bouts

    def behaviors(self):
        return self.model.behaviors

    def frames_per_group(self):
        return 3

    def label_paths(self):
        return ["/p/labels/clip_labels.csv"]

    def frame_dirs(self):
        return ["/p/frames/clip"]

    def current_frames_dir(self):
        return "/p/frames/clip"

    def current_bg_path(self):
        return None

    def checkpoint_dir(self):
        return "/p/checkpoints"

    def status(self, m):
        self.msgs.append(m)


def test_parse_stages():
    assert parse_stages("5,6,7") == [5, 6, 7]
    assert parse_stages(" 1 , 2 ,x, 3 ") == [1, 2, 3]
    assert parse_stages("") == []


def test_build_train_config():
    cfg = build_train_config(
        ["l.csv"], ["f"], frames_per_group=5, variant="b2",
        tsm_stages=[6, 7], unidirectional=True, binarize="groom",
        checkpoint_dir="/ck")
    assert cfg["label_paths"] == ["l.csv"] and cfg["frame_dirs"] == ["f"]
    assert cfg["frames_per_group"] == 5 and cfg["num_frames"] == 5
    assert cfg["variant"] == "b2" and cfg["tsm_stages"] == [6, 7]
    assert cfg["unidirectional"] is True and cfg["binarize"] == "groom"
    assert cfg["checkpoint_dir"] == "/ck"
    # binarize omitted when falsy
    assert "binarize" not in build_train_config(["l"], ["f"], binarize=None)


def test_preprocess_build_crop_params_defaults():
    _app()
    tab = PreprocessTab(FakeCtx())
    assert tab.build_crop_params() == CROP_DEFAULTS


def test_train_build_config_from_widgets():
    _app()
    ctx = FakeCtx()
    tab = TrainTab(ctx)
    tab.variant.setCurrentText("b1")
    tab.tsm_stages.setText("5,6,7")
    tab.unidirectional.setChecked(True)
    cfg = tab.build_config()
    assert cfg["label_paths"] == ctx.label_paths()
    assert cfg["frame_dirs"] == ctx.frame_dirs()
    assert cfg["variant"] == "b1"
    assert cfg["tsm_stages"] == [5, 6, 7]
    assert cfg["unidirectional"] is True
    assert "binarize" not in cfg  # "(none)" selected


def test_pipeline_panel_refresh_populates_binarize():
    _app()
    ctx = FakeCtx()
    ctx.model.add_behavior("groom")
    ctx.model.add_behavior("rear")
    panel = PipelinePanel(ctx)
    panel.refresh()
    items = [panel.train.binarize.itemText(i) for i in range(panel.train.binarize.count())]
    assert items == ["(none)", "groom", "rear"]


def test_stream_worker_emits_events():
    _app()
    events, done = [], []
    w = StreamWorker(lambda: iter([{"a": 1}, {"a": 2}]))
    w.event.connect(events.append)
    w.finished_ok.connect(lambda: done.append(True))
    w.run()  # run synchronously (no separate thread) for the test
    assert events == [{"a": 1}, {"a": 2}]
    assert done == [True]


def test_stream_worker_reports_failure():
    _app()
    errs = []

    def boom():
        raise RuntimeError("nope")
        yield  # pragma: no cover

    w = StreamWorker(boom)
    w.failed.connect(errs.append)
    w.run()
    assert errs and "nope" in errs[0]


def test_call_worker():
    _app()
    out = []
    w = CallWorker(lambda: {"ok": True})
    w.done.connect(out.append)
    w.run()
    assert out == [{"ok": True}]


def _main() -> int:
    _app()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

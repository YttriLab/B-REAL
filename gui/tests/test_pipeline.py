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

from PySide6 import QtCore, QtWidgets  # noqa: E402

from labeling_shell.backend_worker import CallWorker, StreamWorker  # noqa: E402
from labeling_shell.bouts import LabelModel  # noqa: E402
from plugins.pipeline_config import (  # noqa: E402
    CROP_DEFAULTS, TARGETS, VARIANTS, build_train_config, build_tuning_request,
    choice_spec, latency_spec, parse_stages, range_spec,
)
from plugins.pipeline_panel import (  # noqa: E402
    ExportTab, PipelinePanel, PreprocessTab, TrainTab, TuneTab,
)


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


def test_build_train_config_carries_data_loader_knobs():
    cfg = build_train_config(["l"], ["f"], batch_size=16, image_size=380)
    assert cfg["batch_size"] == 16 and cfg["image_size"] == 380
    assert cfg["num_workers"] == 4  # default still sent


def test_build_tuning_request_shape():
    base = build_train_config(["l"], ["f"])
    req = build_tuning_request(
        base,
        space={"lr": range_spec(1e-5, 1e-2, log=True),
               "variant": choice_spec(["b0", "b4"])},
        n_trials=7, objective="val_loss",
        latency=latency_spec(25.0, precision="fp16", target="tensorrt"))
    assert req["base"] is base
    assert req["n_trials"] == 7 and req["objective_metric"] == "val_loss"
    assert req["lr"] == {"low": 1e-5, "high": 1e-2, "log": True}
    assert req["variant"] == {"choices": ["b0", "b4"]}
    assert req["latency"]["max_p99_ms"] == 25.0
    assert req["latency"]["precision"] == "fp16"
    assert req["latency"]["target"] == "tensorrt"
    # latency omitted entirely when not constrained
    assert "latency" not in build_tuning_request(base, space={"lr": range_spec(1, 2)})


def test_tune_tab_space_follows_checkboxes():
    _app()
    tab = TuneTab(FakeCtx())
    # lr is the only row ticked by default
    assert set(tab.build_space()) == {"lr"}

    variant_row = next(r for r in tab.rows if r.name == "variant")
    variant_row.checkbox.setChecked(True)
    space = tab.build_space()
    assert space["variant"]["choices"] == ["b0"]  # the default tick
    assert tab.validate() is None

    # a ticked row with nothing selected is caught before the request goes out
    for i in range(tab.variant_list.count()):
        tab.variant_list.item(i).setCheckState(QtCore.Qt.Unchecked)
    assert "variant" in (tab.validate() or "")


def test_tune_tab_requires_a_parameter():
    _app()
    tab = TuneTab(FakeCtx())
    for row in tab.rows:
        row.checkbox.setChecked(False)
    assert "at least one" in (tab.validate() or "")


def test_tune_tab_latency_spec():
    _app()
    tab = TuneTab(FakeCtx())
    assert tab.build_latency() is None
    tab.lat_enabled.setChecked(True)
    tab.lat_p99.setValue(12.5)
    tab.lat_batch.setValue(4)
    lat = tab.build_latency()
    assert lat["max_p99_ms"] == 12.5 and lat["batch_size"] == 4
    assert lat["target"] == "auto"


def test_tune_tab_deployment_target_is_selectable():
    """The budget must be measurable against the rig's runtime, which need not
    be the machine running the sweep."""
    _app()
    tab = TuneTab(FakeCtx())
    tab.lat_enabled.setChecked(True)
    tab.lat_target.setCurrentIndex(TARGETS.index("openvino"))
    assert tab.build_latency()["target"] == "openvino"
    tab.lat_target.setCurrentIndex(TARGETS.index("tensorrt"))
    assert tab.build_latency()["target"] == "tensorrt"


def test_export_tab_selects_format_and_runtime():
    _app()
    tab = ExportTab(FakeCtx())
    assert tab.build_benchmark_kwargs()["target"] == "auto"
    tab.b_target.setCurrentIndex(TARGETS.index("openvino"))
    assert tab.build_benchmark_kwargs()["target"] == "openvino"

    # shape/precision only apply to the deployment formats
    tab.fmt.setCurrentText("torchscript")
    assert not tab.e_size.isEnabled()
    tab.fmt.setCurrentText("tensorrt")
    assert tab.e_size.isEnabled() and tab.e_prec.isEnabled()


def test_tune_tab_leaderboard_tracks_events():
    _app()
    tab = TuneTab(FakeCtx())
    tab._on_event({"event": "start", "n_trials": 2, "objective": "val_mcc",
                   "direction": "maximize"})
    tab._on_event({"event": "trial_start", "trial": 0, "params": {"lr": 0.001}})
    tab._on_event({"event": "epoch", "trial": 0, "epoch": 1, "val_mcc": 0.42})
    tab._on_event({"event": "trial_done", "trial": 0, "value": 0.42,
                   "params": {"lr": 0.001}})
    tab._on_event({"event": "trial_rejected", "trial": 1, "reason": "latency",
                   "p99_ms": 91.0, "budget_ms": 33.0, "params": {"variant": "b7"}})
    assert tab.table.rowCount() == 2
    assert tab.table.item(0, TuneTab.COL_STATE).text() == "done"
    assert tab.table.item(0, TuneTab.COL_VALUE).text() == "0.4200"
    assert "rejected" in tab.table.item(1, TuneTab.COL_STATE).text()

    tab._on_event({"event": "done", "best_trial": 0, "best_value": 0.42,
                   "best_params": {"lr": 0.001}})
    assert tab.table.item(0, TuneTab.COL_STATE).text() == "BEST"
    assert tab.apply_btn.isEnabled()


def test_tune_tab_applies_best_to_train_tab():
    _app()
    ctx = FakeCtx()
    train = TrainTab(ctx)
    tab = TuneTab(ctx, train_tab=train)
    tab._on_event({"event": "done", "best_trial": 3, "best_value": 0.9,
                   "best_params": {"lr": 0.005, "variant": "b2", "batch_size": 16,
                                   "tsm_stages": "3,4,5"}})
    tab._apply_best()
    assert abs(train.lr.value() - 0.005) < 1e-9
    assert train.variant.currentText() == "b2"
    assert train.batch_size.value() == 16
    assert train.tsm_stages.text() == "3,4,5"


def test_variants_include_the_large_efficientnets():
    assert VARIANTS[:4] == ["b0", "b1", "b2", "b3"]
    assert "b7" in VARIANTS


def test_pipeline_panel_has_a_tune_tab():
    _app()
    panel = PipelinePanel(FakeCtx())
    titles = [panel.tabText(i) for i in range(panel.count())]
    assert titles == ["Preprocess", "Train", "Tune", "Export", "Active Learning"]
    assert panel.tune.train_tab is panel.train


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

"""Phase 3: pipeline config building, panels (offscreen), and backend workers.

Live preprocess/train/export need the backend's CUDA/torch env; this validates
the config marshalling, the widgets' construction/refresh, and the threaded
worker plumbing in isolation.

    QT_QPA_PLATFORM=offscreen pixi run -e gui python gui/tests/test_pipeline.py
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
    choice_spec, latency_spec, parse_stages, range_spec, resolve_binarize,
)
from plugins.pipeline_panel import (  # noqa: E402
    ExportTab, PipelinePanel, PreprocessTab, TrainTab, TuneTab,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(cond, timeout=10.0):
    import time
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    return cond()


class FakeCtx:
    def __init__(self):
        self._client = None
        self.model = LabelModel()
        self.msgs = []
        self.registered = []

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

    def current_frame(self):
        return 42

    def backgrounds_dir(self):
        return "/p/backgrounds"

    def register_model(self, entry):
        self.registered.append(entry)

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
    assert tab.build_crop_params() == {**CROP_DEFAULTS, "method": "background"}


def test_compute_background_sends_frame_count_and_project_dir():
    """The frame count is a real knob (Electron had it), and the background has
    to land in the project or it is never found again."""
    _app()
    ctx = FakeCtx()
    calls = []

    class StubClient:
        def compute_background(self, video, num_frames=500, output_dir=None):
            calls.append({"video": video, "num_frames": num_frames,
                          "output_dir": output_dir})
            return {"path": "/p/backgrounds/bg_v.png"}

    ctx._client = StubClient()
    tab = PreprocessTab(ctx)
    assert tab.bg_frames.value() == 500  # backend default, surfaced not hidden

    tab.bg_frames.setValue(120)
    tab._compute_background()

    app = QtWidgets.QApplication.instance()
    for _ in range(200):
        if calls:
            break
        app.processEvents()
        QtCore.QThread.msleep(10)
    assert calls, "compute_background was never called"
    assert calls[-1]["num_frames"] == 120
    assert calls[-1]["output_dir"] == ctx.backgrounds_dir()


def test_preprocess_writes_into_the_videos_frames_dir():
    """Frames go in frames/<id>/ itself, where training reads them; and a video
    with no bouts extracts nothing (the backend reads "no ranges" as "all")."""
    from labeling_shell.bouts import Bout
    _app()
    ctx = FakeCtx()
    calls = []

    class StubClient:
        def run_preprocessing(self, video, output_dir, **kw):
            calls.append({"video": video, "output_dir": output_dir, **kw})
            return {}

    ctx._client = StubClient()
    tab = PreprocessTab(ctx)
    tab._run()
    assert not calls and "nothing to preprocess" in ctx.msgs[-1]

    beh = ctx.model.add_behavior("groom")
    ctx.model.bouts.append(Bout(beh.id, "groom", 10, 20))
    tab._run()
    assert _wait(lambda: calls)
    assert calls[-1]["output_dir"] == "/p/frames"
    assert calls[-1]["output_name"] == "clip"
    assert calls[-1]["frame_ranges"] == [[8, 20]]


def test_preprocess_raw_mode_disables_crop_controls():
    """Raw mode writes whole frames: background and the detection knobs are
    meaningless, but resize_factor still resizes what gets written."""
    _app()
    tab = PreprocessTab(FakeCtx())
    assert tab.method() == "background"
    assert tab.crop_widgets["box_size"].isEnabled()
    assert tab.bg_edit.isEnabled()
    assert not tab.loc_edit.isEnabled()

    _set_method(tab, "none")
    assert not tab.crop_widgets["box_size"].isEnabled()
    assert not tab.bg_edit.isEnabled()
    assert not tab.preview_box.isEnabled()
    assert tab.crop_widgets["resize_factor"].isEnabled()  # still applies
    assert tab.build_crop_params()["method"] == "none"

    _set_method(tab, "background")
    assert tab.crop_widgets["box_size"].isEnabled()
    assert tab.preview_box.isEnabled()


def _set_method(tab, method):
    tab.crop_method.setCurrentIndex(tab.crop_method.findData(method))


def test_preprocess_localizer_mode():
    """The localizer needs its weights instead of a background, reads only the
    box + stabilisation knobs, and starts from the validated smoothing."""
    _app()
    tab = PreprocessTab(FakeCtx())
    _set_method(tab, "localizer")
    assert tab.loc_edit.isEnabled()
    assert not tab.bg_edit.isEnabled()
    assert tab.preview_box.isEnabled()
    for key in ("box_size", "resize_factor", "smoothing", "deadband"):
        assert tab.crop_widgets[key].isEnabled(), key
    for key in ("difference_threshold", "close_kernel", "min_contour_area"):
        assert not tab.crop_widgets[key].isEnabled(), key
    assert tab.crop_widgets["smoothing"].value() == 0.5

    tab.loc_edit.setText("/w/loc2_final.pt")
    params = tab.build_crop_params()
    assert params["method"] == "localizer"
    assert params["localizer_path"] == "/w/loc2_final.pt"
    assert params["localizer_runtime"] == "auto"  # TensorRT when available


class SettingsCtx(FakeCtx):
    """FakeCtx plus the per-project settings store MainWindow provides."""

    def __init__(self, store=None, videos=()):
        super().__init__()
        self.store = {} if store is None else store
        self.videos = list(videos)

    def setting(self, key, default=None):
        return self.store.get(key, default)

    def set_setting(self, key, value):
        self.store[key] = value

    def project_videos(self):
        return self.videos


def test_preprocess_settings_survive_a_restart():
    """Crop method and knobs are saved per project and restored on refresh:
    Active Learning crops with whatever this tab shows, so a restart that
    silently fell back to background subtraction would feed the model the
    wrong crops."""
    _app()
    store = {}
    tab = PreprocessTab(SettingsCtx(store))
    _set_method(tab, "localizer")
    tab.crop_widgets["box_size"].setValue(300)
    assert store["preprocess"]["method"] == "localizer"
    assert store["preprocess"]["params"]["box_size"] == 300

    again = PreprocessTab(SettingsCtx(store))  # a fresh app on the same project
    again.refresh()
    assert again.method() == "localizer"
    assert again.crop_widgets["box_size"].value() == 300
    assert again.crop_widgets["smoothing"].value() == 0.5
    assert store["preprocess"] == again._current_settings()  # restoring saved nothing new

    other = PreprocessTab(SettingsCtx({}))  # a project with nothing saved: defaults
    other.refresh()
    assert other.method() == "background"


def test_preprocess_starts_from_electron_era_video_settings():
    _app()
    rec = {"id": "v.mp4", "path": "v.mp4",
           "preprocessing": {"box_size": 214, "difference_threshold": 4, "bogus": 1}}
    tab = PreprocessTab(SettingsCtx({}, [rec]))
    tab.refresh()
    assert tab.method() == "background"
    assert tab.crop_widgets["box_size"].value() == 214
    assert tab.crop_widgets["difference_threshold"].value() == 4


def test_preprocess_shows_only_the_rows_the_method_uses():
    _app()
    tab = PreprocessTab(FakeCtx())
    form = tab._form
    row = lambda w: form.getWidgetPosition(w)[0]
    assert form.isRowVisible(row(tab.bg_field)) and not form.isRowVisible(row(tab.loc_field))
    _set_method(tab, "localizer")
    assert not form.isRowVisible(row(tab.bg_field)) and form.isRowVisible(row(tab.loc_field))
    assert "bundled" in tab.loc_info.text()
    tab.loc_edit.setText("/no/such/weights.pt")
    assert "Not found" in tab.loc_info.text()
    _set_method(tab, "none")
    assert not form.isRowVisible(row(tab.bg_field)) and not form.isRowVisible(row(tab.loc_field))
    assert tab.method_hint.text()


def test_preprocess_localizer_preview_sends_no_background():
    _app()
    ctx = FakeCtx()
    calls = []

    class StubClient:
        def preview_frame(self, video, bg, frame, crop_params=None, prev_center=None):
            calls.append({"bg": bg, "crop_params": crop_params})
            return {"presence": 0.9}

    ctx._client = StubClient()
    tab = PreprocessTab(ctx)
    tab.bg_edit.setText("/p/bg.png")
    _set_method(tab, "localizer")

    app = QtWidgets.QApplication.instance()

    def wait(n):
        for _ in range(200):
            if len(calls) >= n:
                break
            app.processEvents()
            QtCore.QThread.msleep(10)

    tab._preview(reset_center=True)  # no weights chosen: the bundled ones
    wait(1)
    assert calls and calls[-1]["bg"] is None
    assert calls[-1]["crop_params"]["localizer_path"] is None

    tab.loc_edit.setText("/w/loc.pt")
    tab._preview(reset_center=True)
    wait(2)
    assert calls[-1]["crop_params"]["localizer_path"] == "/w/loc.pt"


def test_preprocess_preview_renders_the_three_panes():
    _app()
    tab = PreprocessTab(FakeCtx())
    # a 2x2 red JPEG, base64 — stands in for the backend's encoded preview
    import base64
    from PySide6 import QtGui
    img = QtGui.QImage(2, 2, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor("red"))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG")
    encoded = base64.b64encode(bytes(buf.data())).decode("ascii")

    tab._show_preview({
        "original_bbox": encoded, "mask": encoded, "cropped": encoded,
        "frame_width": 640, "frame_height": 480,
        "resized_width": 320, "resized_height": 240,
        "center": [10.0, 20.0],
    })
    for key, _cap in PreprocessTab.PREVIEW_PANES:
        assert not tab.preview_labels[key].pixmap().isNull()
    # the smoothed centre is kept so the next step continues the track
    assert tab._preview_center == [10.0, 20.0]
    # a differing processed size is worth surfacing — it's what gets trained on
    assert "640×480" in tab.preview_info.text()
    assert "320×240" in tab.preview_info.text()


def test_preview_panes_do_not_resize_themselves():
    """Regression: scaling a pixmap to its own QLabel's size is a feedback loop
    (the pixmap is the label's size hint), so stepping frames made the panes
    oscillate and creep — 150 -> 164 -> 152 -> 166 on a fixed image."""
    _app()
    tab = PreprocessTab(FakeCtx())
    tab.resize(900, 850)
    tab.show()
    QtWidgets.QApplication.instance().processEvents()

    import base64
    from PySide6 import QtGui
    img = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor("red"))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG")
    payload = {k: base64.b64encode(bytes(buf.data())).decode("ascii")
               for k, _ in PreprocessTab.PREVIEW_PANES}

    # Prime once so the layout settles (the placeholder and the first image
    # legitimately differ); the bug was that it never settled after that.
    tab._show_preview(dict(payload))
    QtWidgets.QApplication.instance().processEvents()

    seen = []
    for _ in range(6):
        tab._show_preview(dict(payload))
        QtWidgets.QApplication.instance().processEvents()
        pane = tab.preview_labels["cropped"]
        seen.append((pane.width(), pane.height()))

    # the same image must render at the same size every time
    assert len(set(seen)) == 1, f"preview pane size drifted: {seen}"
    tab.hide()


def test_preprocess_preview_stepping_chains_the_center():
    _app()
    ctx = FakeCtx()

    calls = []

    class StubClient:
        def preview_frame(self, video, bg, frame, crop_params=None, prev_center=None):
            calls.append({"frame": frame, "prev_center": prev_center})
            return {}

    ctx._client = StubClient()
    tab = PreprocessTab(ctx)
    tab.bg_edit.setText("/p/bg.png")
    tab._preview_center = [1.0, 2.0]
    tab.preview_frame_spin.setValue(5)

    def wait_for(n):
        """The request goes out on a worker thread."""
        app = QtWidgets.QApplication.instance()
        for _ in range(200):
            if len(calls) >= n:
                return
            app.processEvents()
            QtCore.QThread.msleep(10)
        raise AssertionError(f"expected {n} preview call(s), got {len(calls)}")

    # stepping forwards the smoothed centre, so the stabilization is visible
    # instead of being re-seeded on every frame
    tab._step_preview(+1)
    assert tab.preview_frame_spin.value() == 6
    wait_for(1)
    assert calls[-1] == {"frame": 6, "prev_center": [1.0, 2.0]}

    # an explicit Preview starts a fresh track
    tab._preview(reset_center=True)
    assert tab._preview_center is None
    wait_for(2)
    assert calls[-1]["prev_center"] is None

    # and it can't step below the first frame
    tab.preview_frame_spin.setValue(0)
    tab._step_preview(-1)
    assert tab.preview_frame_spin.value() == 0


def test_preprocess_preview_needs_a_background():
    """Without a background there is nothing to subtract — say so instead of
    firing a request that will 500."""
    _app()
    ctx = FakeCtx()
    ctx._client = object()  # never reached
    tab = PreprocessTab(ctx)
    tab.bg_edit.clear()
    tab._preview()
    assert "Compute background" in tab.preview_info.text()


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
    assert "binarize" not in cfg  # nothing ticked


def test_pipeline_panel_refresh_populates_binarize():
    _app()
    ctx = FakeCtx()
    ctx.model.add_behavior("groom")
    ctx.model.add_behavior("rear")
    panel = PipelinePanel(ctx)
    panel.refresh()
    lst = panel.train.binarize
    items = [lst.item(i).text() for i in range(lst.count())]
    assert items == ["groom", "rear"]
    # nothing ticked by default -> multi-class
    assert panel.train.positive_behaviors() == []


def test_resolve_binarize():
    assert resolve_binarize([]) == (None, None)                    # multi-class
    assert resolve_binarize(["groom"]) == ("groom", None)          # plain binary
    # two or more merge into one synthetic positive class
    b, u = resolve_binarize(["groom", "rear"], ["groom", "rear", "walk"])
    assert b == "positive" and u == {"positive": ["groom", "rear"]}
    # ...whose name must not collide with a real behavior, or load_data would
    # merge the wrong rows
    b, u = resolve_binarize(["groom", "rear"], ["groom", "rear", "positive"])
    assert b == "positive_" and u == {"positive_": ["groom", "rear"]}


def test_train_tab_binarize_checklist_drives_the_config():
    _app()
    ctx = FakeCtx()
    for n in ("groom", "rear", "walk"):
        ctx.model.add_behavior(n)
    tab = TrainTab(ctx)
    tab.refresh()

    def tick(*names):
        for i in range(tab.binarize.count()):
            item = tab.binarize.item(i)
            item.setCheckState(QtCore.Qt.Checked if item.text() in names
                               else QtCore.Qt.Unchecked)

    assert "binarize" not in tab.build_config()

    tick("rear")
    cfg = tab.build_config()
    assert cfg["binarize"] == "rear" and "label_union" not in cfg

    tick("groom", "walk")
    cfg = tab.build_config()
    assert cfg["binarize"] == "positive"
    assert cfg["label_union"] == {"positive": ["groom", "walk"]}


def test_train_tab_refresh_keeps_ticks():
    _app()
    ctx = FakeCtx()
    ctx.model.add_behavior("groom")
    tab = TrainTab(ctx)
    tab.refresh()
    tab.binarize.item(0).setCheckState(QtCore.Qt.Checked)
    ctx.model.add_behavior("rear")
    tab.refresh()  # a new behavior must not silently clear the selection
    assert tab.positive_behaviors() == ["groom"]


def test_train_tab_boundary_smoothing_and_wandb():
    _app()
    tab = TrainTab(FakeCtx())
    cfg = tab.build_config()
    # boundary smoothing defaults on, matching config_tsm.yaml
    assert cfg["boundary_smoothing_enabled"] is True
    assert cfg["boundary_smoothing_epsilon"] == 0.3
    assert cfg["boundary_smoothing_window"] == 5
    assert cfg["use_wandb"] is False
    assert "wandb_api_key" not in cfg  # never sent when W&B is off

    tab.bs_enabled.setChecked(False)
    tab.wandb_enabled.setChecked(True)
    tab.wandb_project.setText("my-project")
    tab.wandb_run.setText("run-001")
    tab.wandb_key.setText("secret")
    cfg = tab.build_config()
    assert cfg["boundary_smoothing_enabled"] is False
    assert cfg["use_wandb"] is True
    assert cfg["project_name"] == "my-project"
    assert cfg["run_name"] == "run-001"
    assert cfg["wandb_api_key"] == "secret"


def test_train_tab_registers_the_trained_model():
    _app()
    ctx = FakeCtx()
    ctx.model.add_behavior("groom")
    tab = TrainTab(ctx)
    tab.refresh()
    tab.binarize.item(0).setCheckState(QtCore.Qt.Checked)
    config = tab.build_config()

    tab._register_model(config, {
        "status": "completed",
        "best_checkpoint": r"/p/models/checkpoints/groom-epoch=07-val_mcc=0.812.ckpt",
        "best_val_mcc": 0.812, "best_val_loss": 0.31,
    })
    assert len(ctx.registered) == 1
    entry = ctx.registered[0]
    assert entry["name"] == "groom-epoch=07-val_mcc=0.812.ckpt"
    assert entry["val_mcc"] == 0.812 and entry["val_loss"] == 0.31
    # binarize/label_union must survive: Active Learning needs them to map a
    # behavior name into the model's collapsed label space
    assert entry["config"]["binarize"] == "groom"
    assert "label_union" in entry["config"]


def _lr_result(suggested=2.5e-4):
    import math
    lrs = [10 ** (-6 + 5 * i / 99) for i in range(100)]
    losses = [0.68 - 0.1 * min(1, max(0, (math.log10(lr) + 4) / 1.3)) for lr in lrs]
    return {"status": "completed", "lr_find": True, "lrs": lrs, "losses": losses,
            "suggested_lr": suggested, "lightning_lr": 3.5e-6, "current_lr": 1e-4}


def test_find_lr_runs_the_range_test_and_offers_the_rate():
    """Find LR sends the Train tab's config with find_lr, shows the curve, and
    Use puts the rate in the lr field; nothing is registered as a model."""
    _app()
    ctx = FakeCtx()
    sent = []

    class StubClient:
        def start_training(self, config):
            sent.append(config)
            return {"task_id": "t1"}

        def training_status(self, task_id):
            yield {"status": "starting", "message": "Finding a learning rate"}
            yield _lr_result()

    ctx._client = StubClient()
    tab = TrainTab(ctx)
    tab._find_lr()
    assert _wait(lambda: getattr(tab, "lr_dialog", None) is not None)
    assert sent[0]["find_lr"] is True and sent[0]["lr"] == tab.lr.value()
    assert not ctx.registered
    dialog = tab.lr_dialog
    assert dialog.use_btn.text() == "Use 2.5e-04" and dialog.use_btn.isEnabled()
    dialog.accept()
    assert abs(tab.lr.value() - 2.5e-4) < 1e-9
    assert _wait(lambda: tab.find_lr_btn.isEnabled() and tab.start_btn.isEnabled())

    tab._start()  # plain training doesn't ask for the range test
    assert _wait(lambda: len(sent) == 2)
    assert "find_lr" not in sent[1]


def test_lr_finder_dialog_picks_from_the_curve():
    from plugins.lr_finder_dialog import LrFinderDialog
    _app()
    dialog = LrFinderDialog(_lr_result())
    dialog.resize(560, 440)
    curve = dialog.curve
    rect = curve._plot_rect()
    assert abs(curve.lr_at_x(rect.left()) - 1e-6) < 1e-12
    assert abs(curve.lr_at_x(rect.right()) - 1e-1) < 1e-9
    dialog._pick(curve.lr_at_x(rect.left() + rect.width() * 0.5))  # 10 ** -3.5
    assert dialog.chosen_lr == 3.2e-4 and dialog.use_btn.text() == "Use 3.2e-04"
    curve.grab()  # paints without error

    none = LrFinderDialog(_lr_result(suggested=None))
    assert not none.use_btn.isEnabled()


def test_preprocess_status_states():
    import tempfile
    import time
    from labeling_shell.bouts import Bout
    from plugins import preprocess_status as ps
    with tempfile.TemporaryDirectory() as d:
        bouts = [Bout("g", "groom", 10, 12)]  # needs 8..12 with frames/group 3
        assert ps.video_status([], d).state == ps.UNLABELED
        st = ps.video_status(bouts, d)
        assert (st.state, st.needed, st.present) == (ps.MISSING, 5, 0)
        for i in (8, 9, 10):
            open(os.path.join(d, f"{i}.png"), "w").close()
        open(os.path.join(d, "crops.csv"), "w").close()  # not a frame
        st = ps.video_status(bouts, d)
        assert (st.state, st.present) == (ps.PARTIAL, 3)
        assert st.method is None and "not recorded" in st.detail
        for i in (11, 12, 40):  # 40: a frame the labels no longer need
            open(os.path.join(d, f"{i}.png"), "w").close()
        _write_crop_record(d, {"method": "background"}, mixed=True)
        st = ps.video_status(bouts, d)  # the dir changed, so it is re-listed
        assert (st.state, st.present, st.method, st.mixed) == (ps.READY, 5, "background", True)
        assert st.short == "preprocessed · background subtraction" and "mixed" in st.detail
        assert ps.video_status(bouts, os.path.join(d, "nope")).state == ps.MISSING

        # A settled directory is listed once and then served from the cache.
        old = time.time() - 10
        os.utime(d, (old, old))
        ps.frames_on_disk(d)
        os.remove(os.path.join(d, "8.png"))
        os.utime(d, (old, old))  # same mtime: the cache can't tell
        assert 8 in ps.frames_on_disk(d)


def _write_crop_record(d, params, mixed=False):
    import json
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "crop_params.json"), "w") as f:
        json.dump({"crop_params": params, "mixed": mixed}, f)


BG_CROP = {"method": "background", "box_size": 214, "downsample_factor": 3,
           "resize_factor": 1.0, "smoothing": 0.6, "difference_threshold": 4}
LOC_CROP = {"method": "localizer", "box_size": 214, "resize_factor": 1.0,
            "smoothing": 0.6, "localizer_path": None}


def test_crop_summary_speaks_video_pixels():
    from plugins.crop_summary import box_px, differences, summarize
    assert box_px(BG_CROP) == 642 and box_px(LOC_CROP) == 214
    assert box_px({**LOC_CROP, "resize_factor": 0.5}) == 428
    assert box_px({"method": "none"}) is None
    assert summarize(BG_CROP).startswith("background subtraction · 642 px box (214 × downsample 3)")
    assert "bundled weights" in summarize(LOC_CROP)
    assert differences(BG_CROP, LOC_CROP) == [
        "method SAM2 localizer vs background subtraction in training"]
    assert differences(BG_CROP, {**BG_CROP, "downsample_factor": 1, "smoothing": 0.0}) == [
        "box 214 px vs 642 px in training", "smoothing 0.0 vs 0.6 in training"]
    assert differences(BG_CROP, dict(BG_CROP)) == []


def test_training_crop_from_frame_dirs():
    import tempfile
    from plugins.crop_summary import training_crop
    with tempfile.TemporaryDirectory() as d:
        a, b, c, e = (os.path.join(d, x) for x in "abce")
        _write_crop_record(a, BG_CROP)
        _write_crop_record(b, dict(BG_CROP))
        _write_crop_record(c, {**BG_CROP, "box_size": 100})
        _write_crop_record(e, BG_CROP, mixed=True)
        assert training_crop([a, b]) == (BG_CROP, "")
        assert training_crop([a, c])[0] is None and "differently" in training_crop([a, c])[1]
        assert training_crop([e])[0] is None and "mixed" in training_crop([e])[1]
        assert training_crop([a, os.path.join(d, "none")])[1].startswith("not recorded")


def test_registered_model_records_its_training_crop():
    import tempfile
    _app()
    ctx = FakeCtx()
    tab = TrainTab(ctx)
    with tempfile.TemporaryDirectory() as d:
        _write_crop_record(d, BG_CROP)
        tab._register_model({"frame_dirs": [d]}, {"best_checkpoint": "/p/m.ckpt"})
    assert ctx.registered[0]["crop_params"] == BG_CROP
    assert ctx.registered[0]["crop_note"] == ""


class AlCtx(FakeCtx):
    """FakeCtx plus what the Active Learning tab reads."""

    def __init__(self, entry=None, current=None):
        super().__init__()
        self.entry, self.current = entry, current or LOC_CROP

    def crop_params(self):
        return dict(self.current)

    def current_bg_path(self):
        return "/p/bg.png"

    def project_models(self):
        return [self.entry] if self.entry else []

    def model_entry(self, path):
        return self.entry if self.entry and path == self.entry["path"] else None

    def set_al_threshold(self, t):
        pass


def test_inference_crops_like_the_model_was_trained():
    from plugins.active_learning_panel import ActiveLearningPanel
    _app()
    entry = {"name": "m.ckpt", "path": "/p/m.ckpt", "val_mcc": 0.8, "created": "2026-10-08",
             "crop_params": BG_CROP, "crop_note": ""}
    ctx = AlCtx(entry)
    calls = []

    class Client:
        def run_inference(self, ckpt, **kw):
            calls.append(kw)
            return {}
    ctx._client = Client()
    panel = ActiveLearningPanel(ctx)
    panel.refresh()
    assert panel.models.count() == 2  # placeholder + the project's model
    panel.models.setCurrentIndex(1)
    assert panel.ckpt.text() == "/p/m.ckpt"
    params, source, warnings = panel._effective_crop()
    assert params == BG_CROP and "training" in source and not warnings
    assert "642 px box" in panel.crop_info.text()

    # trying the Preprocess tab's (localizer) settings warns how they differ
    panel.crop_source.setCurrentIndex(1)
    params, _source, warnings = panel._effective_crop()
    assert params["method"] == "localizer"
    assert warnings and "method SAM2 localizer vs background" in warnings[0]

    panel.crop_source.setCurrentIndex(0)
    panel._run()
    assert _wait(lambda: calls)
    assert calls[0]["crop_params"] == BG_CROP

    # a model with no record falls back to the Preprocess tab, and says so
    old = AlCtx({"name": "old.ckpt", "path": "/p/old.ckpt"})
    p2 = ActiveLearningPanel(old)
    p2.ckpt.setText("/p/old.ckpt")
    params, source, warnings = p2._effective_crop()
    assert params == LOC_CROP and source == "Preprocess tab"
    assert "not recorded" in warnings[0]


def test_register_model_survives_a_manifest_failure():
    """A failed manifest write must not read as a failed training run."""
    _app()
    ctx = FakeCtx()

    def boom(entry):
        raise OSError("disk full")
    ctx.register_model = boom

    tab = TrainTab(ctx)
    tab._register_model({}, {"best_checkpoint": "/p/x.ckpt"})
    assert "could not record the model" in tab.log.toPlainText()


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
    assert titles == ["Preprocess", "Train Localizer", "Train", "Tune", "Export",
                      "Active Learning"]
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

"""Train Localizer tab (offscreen): request building, click seeding, results,
and handing the trained weights to the Preprocess tab.

    QT_QPA_PLATFORM=offscreen pixi run -e gui python gui/tests/test_localizer_panel.py
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from labeling_shell.bouts import LabelModel  # noqa: E402
from labeling_shell.project import ProjectStore  # noqa: E402
from plugins.localizer_panel import ClickableImagePane, LocalizerTab  # noqa: E402
from plugins.pipeline_panel import PipelinePanel, PreprocessTab  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _jpeg_b64(w=40, h=20) -> str:
    img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor("gray"))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG")
    return base64.b64encode(bytes(buf.data())).decode()


def _wait(cond, ms=3000):
    app = QtWidgets.QApplication.instance()
    for _ in range(ms // 10):
        if cond():
            return True
        app.processEvents()
        QtCore.QThread.msleep(10)
    return cond()


class FakeCtx:
    def __init__(self, project_dir="/p"):
        self._client = None
        self.model = LabelModel()
        self.msgs = []
        self.saved_localizer = "unset"
        self.project_dir = project_dir

    def client(self):
        return self._client

    def video_path(self):
        return "/v/a.mkv"

    def current_frame(self):
        return 120

    def current_bg_path(self):
        return None

    def project_videos(self):
        return [{"id": "a", "name": "a.mkv", "path": "/v/a.mkv"},
                {"id": "b", "name": "b.mkv", "path": "/v/b.mkv"}]

    def localizer_dir(self):
        return f"{self.project_dir}/models/localizer" if self.project_dir else None

    def localizer_path(self):
        return None if self.saved_localizer == "unset" else self.saved_localizer

    def set_localizer_path(self, path):
        self.saved_localizer = path

    def status(self, m):
        self.msgs.append(m)


class StubClient:
    def __init__(self):
        self.calls = []

    def localizer_info(self):
        return {"sam2_available": True, "sam2_detail": "transformers 5.0"}

    def localizer_seed_preview(self, video, frame, x=None, y=None, sam2_model="tiny"):
        self.calls.append(("preview", video, frame, x, y, sam2_model))
        res = {"image": _jpeg_b64(), "width": 2000, "height": 1000, "center": None, "area": 0}
        if x is not None:
            res.update(center=[x + 5, y - 5], area=81234)
        return res

    def train_localizer(self, req):
        self.calls.append(("train", req))
        return {"task_id": "t1"}

    def localizer_status(self, task_id):
        yield {"status": "running", "phase": "label", "progress": 5, "total": 10,
               "log": ["clip 1/1 ..."]}
        yield {"status": "completed", "phase": "train", "progress": 3, "total": 3,
               "log": ["saved /p/models/localizer/x.pt"],
               "result": {"weights": "/p/models/localizer/x.pt", "train_frames": 400,
                          "val_frames": 100, "dropped_frames": 3,
                          "before": {"median_error_px": 40.0, "p95_error_px": 90.0},
                          "after": {"median_error_px": 6.0, "p95_error_px": 20.0,
                                    "presence_on_animal": 0.97, "presence_on_empty": 0.04},
                          "qc_image": None}}

    def stop_localizer(self, task_id):
        self.calls.append(("stop", task_id))
        return {}


def test_request_from_project_videos_and_preprocess_weights():
    _app()
    ctx = FakeCtx()
    pre = PreprocessTab(ctx)
    tab = LocalizerTab(ctx, preprocess_tab=pre)
    tab.refresh()
    req = tab.build_request()
    assert req["videos"] == ["/v/a.mkv", "/v/b.mkv"]  # clip sources default on
    assert req["negative_videos"] == []               # empty-arena videos default off
    assert req["init_weights"] is None                 # bundled
    assert req["output_dir"] == "/p/models/localizer"
    assert req["init"] == "finetune" and req["name"] is None
    pre.loc_edit.setText("/w/mine.pt")
    tab.videos.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    tab.refresh()  # keeps ticks across refreshes
    req = tab.build_request()
    assert req["init_weights"] == "/w/mine.pt" and req["videos"] == ["/v/a.mkv"]
    assert tab.validate(req) is None


def test_validate():
    _app()
    tab = LocalizerTab(FakeCtx(project_dir=None))
    assert "project" in tab.validate(tab.build_request()).lower()
    tab = LocalizerTab(FakeCtx())
    tab.refresh()
    tab.clips_per_video.setValue(0)
    assert "nothing to label" in tab.validate(tab.build_request()).lower()
    tab.seeds = [{"video_path": "/v/a.mkv", "frame": 1, "x": 1, "y": 2}]
    assert tab.validate(tab.build_request()) is None


def test_click_seed_flow():
    _app()
    ctx = FakeCtx()
    ctx._client = client = StubClient()
    tab = LocalizerTab(ctx)
    tab._load_frame()
    assert _wait(lambda: tab._frame is not None)
    assert client.calls[-1][:3] == ("preview", "/v/a.mkv", 120)
    assert not tab.add_seed_btn.isEnabled()
    tab._on_seed_click(0.5, 0.25)  # -> native (1000, 250) on the 2000x1000 frame
    assert _wait(lambda: tab.add_seed_btn.isEnabled())
    assert client.calls[-1][3:5] == (1000.0, 250.0)
    assert "81,234" in tab.seed_info.text()
    tab._add_seed()
    assert tab.seeds == [{"video_path": "/v/a.mkv", "frame": 120, "x": 1000.0, "y": 250.0}]
    assert tab.seed_table.rowCount() == 1 and not tab.add_seed_btn.isEnabled()
    tab.seed_table.selectRow(0)
    tab._remove_seed()
    assert tab.seeds == [] and tab.seed_table.rowCount() == 0


def test_train_streams_and_hands_weights_to_preprocess():
    _app()
    ctx = FakeCtx()
    ctx._client = client = StubClient()
    pre = PreprocessTab(ctx)
    tab = LocalizerTab(ctx, preprocess_tab=pre)
    tab.refresh()
    tab._train()
    assert _wait(lambda: tab.result is not None and tab.train_btn.isEnabled())
    assert client.calls[0][0] == "train"
    assert "clip 1/1" in tab.log.toPlainText() and "saved" in tab.log.toPlainText()
    assert "40.0 px" in tab.result_info.text() and "6.0 px" in tab.result_info.text()
    assert not tab.result_box.isHidden()
    tab._use_result()
    assert pre.loc_edit.text() == "/p/models/localizer/x.pt"
    assert pre.method() == "localizer"
    assert ctx.saved_localizer == "/p/models/localizer/x.pt"


def test_no_improvement_is_said_and_not_offered():
    _app()
    tab = LocalizerTab(FakeCtx(), preprocess_tab=PreprocessTab(FakeCtx()))
    tab._show_result({"weights": "/p/x.pt", "improved": False, "dropped_clips": [3],
                      "before": {"median_error_px": 20.0, "p95_error_px": 40.0},
                      "after": {"median_error_px": 20.0, "p95_error_px": 40.0}})
    assert "never beat" in tab.result_info.text() and "left out" in tab.result_info.text()
    assert not tab.use_btn.isEnabled()
    tab._show_result({"weights": "/p/y.pt", "improved": True, "after": {"median_error_px": 3.0,
                                                                       "p95_error_px": 6.0}})
    assert tab.use_btn.isEnabled()


def test_train_refuses_without_a_project():
    _app()
    ctx = FakeCtx(project_dir=None)
    ctx._client = client = StubClient()
    tab = LocalizerTab(ctx)
    tab.refresh()
    tab._train()
    assert not client.calls and "project" in tab.phase_label.text().lower()
    assert tab.train_btn.isEnabled()


def test_sam2_status_is_shown():
    _app()
    ctx = FakeCtx()
    ctx._client = StubClient()
    tab = LocalizerTab(ctx)
    tab.refresh()
    assert _wait(lambda: "ready" in tab.sam2_label.text())


def test_preprocess_restores_and_saves_the_project_localizer():
    _app()
    ctx = FakeCtx()
    ctx.saved_localizer = "/p/models/localizer/old.pt"
    pre = PreprocessTab(ctx)
    pre.refresh()
    assert pre.loc_edit.text() == "/p/models/localizer/old.pt"
    pre.set_localizer(None)  # back to the bundled weights
    assert ctx.saved_localizer is None and pre.loc_edit.text() == ""


def test_clickable_pane_reports_source_fractions():
    _app()
    pane = ClickableImagePane()
    pane.resize(400, 400)
    pix = QtGui.QPixmap(200, 100)  # letterboxed: shown as 400x200, centred vertically
    pix.fill(QtGui.QColor("black"))
    pane.set_image(pix)
    got = []
    pane.clicked.connect(lambda fx, fy: got.append((round(fx, 2), round(fy, 2))))
    for x, y in ((100, 150), (300, 250), (200, 50)):  # the last is in the letterbox
        ev = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress, QtCore.QPointF(x, y),
                               QtCore.QPointF(x, y), QtCore.Qt.MouseButton.LeftButton,
                               QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
        pane.mousePressEvent(ev)
    assert got == [(0.25, 0.25), (0.75, 0.75)], got


def test_pipeline_has_the_tab():
    _app()
    panel = PipelinePanel(FakeCtx())
    names = [panel.tabText(i) for i in range(panel.count())]
    assert names.index("Train Localizer") == names.index("Preprocess") + 1


def test_project_settings_round_trip():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("p")
        assert store.get_setting("localizer_path") is None
        store.set_setting("localizer_path", "models/localizer/x.pt")
        assert ProjectStore(d).get_setting("localizer_path") == "models/localizer/x.pt"
        store.set_setting("localizer_path", None)
        assert "localizer_path" not in store.load_manifest().get("settings", {})


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

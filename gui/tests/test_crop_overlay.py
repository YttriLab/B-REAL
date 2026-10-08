"""Crop box over the video: recorded boxes from crops.csv, live ones from the backend."""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from labeling_shell.video_widget import PlayerWidget, VideoView  # noqa: E402
from plugins.crop_overlay import (  # noqa: E402
    CROPS_CSV, LIVE_COLOR, RECORDED_COLOR, CropBoxOverlay, read_crop_boxes,
)
from tools.index_video import write_index_video  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    return cond()


class FakeClient:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def crop_box(self, video, index, params, bg):
        self.calls.append(index)
        if self.fail:
            raise RuntimeError("boom")
        return {"box": [index, 10, 20, 20], "frame_index": index}


class Ctx:
    def __init__(self, frames_dir, client=None, method="localizer", bg=None):
        self.frames_dir, self._client, self.method, self.bg = frames_dir, client, method, bg
        self.msgs = []

    def client(self):
        return self._client

    def video_path(self):
        return "v.mp4"

    def current_frames_dir(self):
        return self.frames_dir

    def current_bg_path(self):
        return self.bg

    def crop_params(self):
        return {"method": self.method, "box_size": 20}

    def status(self, msg):
        self.msgs.append(msg)


def _player(d, n=30):
    path = os.path.join(d, "clip.mp4")
    write_index_video(path, n, fps=30, gop=10)
    p = PlayerWidget()
    p.load(path)
    return p


def _write_csv(frames_dir, rows):
    os.makedirs(frames_dir, exist_ok=True)
    with open(os.path.join(frames_dir, CROPS_CSV), "w") as f:
        f.write("frame,x,y,w,h\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


def test_view_draws_boxes_in_video_pixels():
    _app()
    view = VideoView()
    view.resize(200, 100)
    import numpy as np
    view.set_frame(np.zeros((50, 100, 3), np.uint8))  # 2x scale, no letterbox
    view.set_boxes([{"rect": (10, 10, 20, 20), "color": "#ff0000"}])
    img = QtGui.QImage(view.size(), QtGui.QImage.Format.Format_RGB32)
    view.render(img)
    edge = img.pixelColor(20, 40)  # left edge at x = 10*2, y inside the box
    assert edge.red() > 200 and edge.green() < 80, edge.name()
    assert img.pixelColor(40, 40).red() < 50  # the box is not filled


def test_recorded_box_shown_and_live_requested_elsewhere():
    _app()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        frames = os.path.join(d, "frames")
        _write_csv(frames, [(3, 1, 2, 30, 40)])
        assert read_crop_boxes(os.path.join(frames, CROPS_CSV)) == {3: (1.0, 2.0, 30.0, 40.0)}
        player = _player(d)
        client = FakeClient()
        ov = CropBoxOverlay(Ctx(frames, client), player)
        ov.set_enabled(True)

        player._show(3)
        assert player.view.boxes == [{"rect": (1.0, 2.0, 30.0, 40.0),
                                      "color": RECORDED_COLOR, "label": "crop"}]
        assert client.calls == []  # a recorded frame never asks the backend

        player._show(7)
        assert player.view.boxes == []  # not the previous frame's box meanwhile
        assert _wait_until(lambda: player.view.boxes)
        box = player.view.boxes[0]
        assert box["rect"] == (7, 10, 20, 20) and box["color"] == LIVE_COLOR and box["dashed"]
        assert client.calls == [7]

        player._show(3)
        player._show(7)  # cached: no second request
        assert player.view.boxes[0]["rect"] == (7, 10, 20, 20)
        assert client.calls == [7]

        ov.set_enabled(False)
        assert player.view.boxes == []
        ov.shutdown()
        player.reader.close()


def test_rapid_frame_changes_only_ask_for_where_the_player_rests():
    _app()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        player = _player(d)
        client = FakeClient()
        ov = CropBoxOverlay(Ctx(None, client), player)
        ov.set_enabled(True)
        for i in range(10, 20):  # like playback: faster than the debounce
            player._show(i)
        assert _wait_until(lambda: player.view.boxes)
        assert client.calls == [19]
        ov.shutdown()
        player.reader.close()


def test_no_live_box_without_what_it_needs():
    _app()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        player = _player(d)
        client = FakeClient()
        for ctx in (Ctx(None, None),                            # no backend
                    Ctx(None, client, method="background"),    # no background yet
                    Ctx(None, client, method="none")):         # whole frame
            ov = CropBoxOverlay(ctx, player)
            ov.set_enabled(True)
            player._show(5)
            QtCore.QThread.msleep(200)
            QtWidgets.QApplication.processEvents()
            assert client.calls == [] and player.view.boxes == []
            ov.set_enabled(False)
        player.reader.close()


def test_live_failure_reported_once():
    _app()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        player = _player(d)
        client = FakeClient(fail=True)
        ctx = Ctx(None, client)
        ov = CropBoxOverlay(ctx, player)
        ov.set_enabled(True)
        player._show(4)
        assert _wait_until(lambda: ctx.msgs)
        player._show(5)
        QtCore.QThread.msleep(200)
        QtWidgets.QApplication.processEvents()
        assert client.calls == [4] and len(ctx.msgs) == 1
        ov.shutdown()
        player.reader.close()


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

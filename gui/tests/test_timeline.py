"""Phase 4: uncertainty timeline mapping + wiring (offscreen)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.active_learning import AlState  # noqa: E402
from labeling_shell.timeline_widget import UncertaintyTimeline, uncertainty_color  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_frame_x_mapping():
    _app()
    tl = UncertaintyTimeline()
    al = AlState()
    al.load_results({"class_names": ["a"], "frames": []})
    tl.set_data(al, 100)
    tl.resize(101, 46)
    w = tl.width()
    assert tl.x_at_frame(0) == 0
    assert tl.x_at_frame(tl.num_frames - 1) == w - 1
    assert tl.frame_at_x(0) == 0
    assert tl.frame_at_x(w - 1) == 99


def test_empty_is_safe():
    _app()
    tl = UncertaintyTimeline()
    assert tl.x_at_frame(0) == 0
    assert tl.frame_at_x(50) == 0  # num_frames == 0


def test_click_emits_seek():
    _app()
    from PySide6 import QtCore, QtGui
    tl = UncertaintyTimeline()
    al = AlState()
    al.load_results({"class_names": ["a"], "frames": []})
    tl.set_data(al, 100)
    tl.resize(101, 46)
    seen = []
    tl.seekRequested.connect(seen.append)
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(tl.width() - 1, 10), QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    tl.mousePressEvent(ev)
    assert seen == [99]


def test_uncertainty_color_endpoints():
    lo = uncertainty_color(0.0)   # green-ish
    hi = uncertainty_color(1.0)   # red-ish
    assert lo.hslHue() > hi.hslHue()


def test_next_segment():
    from labeling_shell.timeline_widget import next_segment
    runs = [(10, 19), (40, 49), (70, 79)]
    assert next_segment(runs, 0) == 10
    assert next_segment(runs, 10) == 40
    assert next_segment(runs, 75) is None
    assert next_segment(runs, 45, forward=False) == 40  # inside: its own start first
    assert next_segment(runs, 40, forward=False) == 10  # at its start: the one before
    assert next_segment(runs, 60, forward=False) == 40
    assert next_segment(runs, 10, forward=False) is None


def test_lanes_paint_predictions_labels_and_max_uncertainty():
    from PySide6 import QtGui
    from labeling_shell.bouts import Bout
    _app()
    al = AlState()
    # one very uncertain frame among 1000: it must survive per-pixel aggregation
    al.load_results({"class_names": ["other", "rear"],
                     "frames": [{"index": i, "predicted_class": 1 if 600 <= i < 603 else 0,
                                 "uncertainty": 1.0 if i == 300 else 0.0} for i in range(1000)]})
    tl = UncertaintyTimeline()
    tl.resize(101, 50)
    tl.set_data(al, 1000)
    colors = {"other": QtGui.QColor("#5f6368"), "rear": QtGui.QColor("#ff0000")}
    tl.set_labels([Bout("r", "rear", 100, 101)], colors)
    per_px = tl._max_uncertainty_per_px(tl.width())
    assert per_px[tl.x_at_frame(300)] == 1.0 and per_px[tl.x_at_frame(500)] == 0.0

    def pixel(y, frame):
        img = QtGui.QImage(tl.size(), QtGui.QImage.Format.Format_RGB32)
        tl.render(img)
        return img.pixelColor(tl.x_at_frame(frame), y)

    tl.set_mode("predicted")
    model_y, label_y = 10, 30
    assert pixel(model_y, 601).name() == "#ff0000"  # 3 frames, still >= 1 px wide
    assert pixel(model_y, 200).name() == "#5f6368"
    assert pixel(label_y, 100).name() == "#ff0000"  # a 2-frame labeled bout
    assert pixel(label_y, 500).name() != "#ff0000"

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

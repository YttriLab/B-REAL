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

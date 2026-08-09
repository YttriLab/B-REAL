"""Phase 0 gate (display half): prove the decode -> QImage -> widget path is
also frame-exact, headlessly (offscreen Qt platform, no window shown).

Runnable two ways:
    .venv-gui/Scripts/python.exe gui/tests/test_qt_display.py
    pytest gui/tests/test_qt_display.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.qt_image import ndarray_to_qimage, qimage_to_ndarray  # noqa: E402
from labeling_shell.video_widget import PlayerWidget, format_time  # noqa: E402
from tools.index_video import make_index_frame, read_index_frame, write_index_video  # noqa: E402

N = 120


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_qimage_roundtrip_is_lossless():
    for i in (0, 1, 42, 4095):
        arr = make_index_frame(i, 320, 240)
        back = qimage_to_ndarray(ndarray_to_qimage(arr))
        assert np.array_equal(arr, back), f"round-trip changed pixels for {i}"
        assert read_index_frame(back) == i


def test_player_displays_exact_frame():
    _app()
    # ignore_cleanup_errors: on Windows PyAV may still hold the file briefly.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, N, fps=30, gop=12)

        player = PlayerWidget()
        try:
            player.load(path)
            assert player.reader.num_frames == N

            # The QImage held by the view must decode to exactly the requested
            # index, for sequential, random and backward access.
            order = [0, 1, 2, 50, 49, 119, 7, 118, 60, 0]
            for i in order:
                player._show(i)
                shown = qimage_to_ndarray(player.view.current_image)
                assert read_index_frame(shown) == i, f"view showed wrong frame for {i}"

            # Transport: play advances the index; pause stops it.
            player._show(0)
            player.play()
            for _ in range(5):
                player._on_tick()
            assert player.index == 5
            player.pause()
            player._on_tick()
            assert player.index == 5  # a stray tick after pause must not advance

            # Keyboard-equivalent stepping clamps at the ends.
            player._show(N - 1)
            player.step(+1)
            assert player.index == N - 1
            player._show(0)
            player.step(-1)
            assert player.index == 0
        finally:
            if player.reader is not None:
                player.reader.close()


def test_format_time():
    assert format_time(0) == "00:00:000"
    assert format_time(1000) == "00:01:000"
    assert format_time(61_500) == "01:01:500"


def _main() -> int:
    _app()
    failures = []
    for fn in (test_qimage_roundtrip_is_lossless,
               test_player_displays_exact_frame,
               test_format_time):
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

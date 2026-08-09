"""Phase 2: MainWindow + panels, offscreen (no window shown).

    QT_QPA_PLATFORM=offscreen .venv-gui/Scripts/python.exe gui/tests/test_app.py
    pytest gui/tests/test_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.behaviors_panel import BehaviorsPanel  # noqa: E402
from labeling_shell.bouts import Bout, LabelModel  # noqa: E402
from labeling_shell.bouts_panel import BoutsPanel  # noqa: E402
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402
from tools.index_video import write_index_video  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_behaviors_panel_add_remove():
    _app()
    m = LabelModel()
    panel = BehaviorsPanel(m)
    panel.add_behavior("groom")
    panel.add_behavior("rear")
    assert [b.name for b in m.behaviors] == ["groom", "rear"]
    assert panel.list.count() == 2
    panel.remove_behavior(m.behaviors[0].id)
    assert [b.name for b in m.behaviors] == ["rear"]
    assert panel.list.count() == 1


def test_bouts_panel_populate_delete_and_seek():
    _app()
    m = LabelModel()
    a = m.add_behavior("groom")
    m.insert_bout_sorted(Bout(a.id, a.name, 10, 20))
    m.insert_bout_sorted(Bout(a.id, a.name, 30, 40))
    panel = BoutsPanel(m, lambda: 30.0)
    assert panel.table.rowCount() == 2

    seen = []
    panel.seekRequested.connect(seen.append)
    panel._on_cell_clicked(0, 0)  # start of first bout
    panel._on_cell_clicked(1, 1)  # end of second bout
    assert seen == [10, 40]

    panel.delete_bout(0)
    assert panel.table.rowCount() == 1
    assert m.bouts[0].start_frame == 30

    # edit times via the non-modal helper
    assert panel.set_bout_times(0, 0, 2000) is True  # 0..2000ms @30fps -> 0..60
    assert (m.bouts[0].start_frame, m.bouts[0].end_frame) == (0, 60)


def test_mainwindow_key_toggle_creates_bout():
    _app()
    from app import MainWindow
    win = MainWindow()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path)
        win.behaviors_panel.add_behavior("groom")  # key "1"

        win._seek(10)
        win._handle_key("1", 0)   # start
        assert win.model.is_recording(win.model.behaviors[0].id)
        win._seek(25)
        win._handle_key("1", 0)   # end
        assert len(win.model.bouts) == 1
        assert (win.model.bouts[0].start_frame, win.model.bouts[0].end_frame) == (10, 25)
        assert win.bouts_panel.table.rowCount() == 1

        if win.player.reader is not None:
            win.player.reader.close()


def test_mainwindow_project_autosave_roundtrip():
    _app()
    from app import MainWindow
    win = MainWindow()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        win.project = ProjectStore(proj)
        win.project.create("t")
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path)
        win.behaviors_panel.add_behavior("groom")
        win._seek(5)
        win._handle_key("1", 0)
        win._seek(9)
        win._handle_key("1", 0)  # autosaves

        vid = safe_id("clip.mp4")
        reloaded = win.project.load_into_model(vid)
        assert [b.name for b in reloaded.behaviors] == ["groom"]
        assert (reloaded.bouts[0].start_frame, reloaded.bouts[0].end_frame) == (5, 9)

        if win.player.reader is not None:
            win.player.reader.close()


def test_mainwindow_al_results_drive_timeline_and_hud():
    _app()
    from app import MainWindow
    from labeling_shell.active_learning import AlState
    from labeling_shell.bouts import Bout
    win = MainWindow()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path)
        a = win.model.add_behavior("groom")
        win.model.insert_bout_sorted(Bout(a.id, a.name, 12, 12))  # label frame 12 groom

        state = AlState()
        state.load_results({
            "class_names": ["other", "groom"],
            "frames": [
                {"index": 10, "predicted_class": "groom", "uncertainty": 0.9, "predictions": [0.1, 0.9]},
                {"index": 12, "predicted_class": "other", "uncertainty": 0.6, "predictions": [0.8, 0.2]},
            ],
        })
        win.set_al_results(state)
        # timeline knows the frame count and the misclassified frame (12: pred other vs label groom)
        assert win.timeline.num_frames == 60
        assert 12 in win.timeline.misclassified
        # HUD shows a mismatch at frame 12 (isHidden reflects show/hide state even
        # when the top-level window was never shown, as in offscreen tests)
        win._update_hud(12)
        assert not win.hud.isHidden()
        assert "pred: other" in win.hud.text()

        if win.player.reader is not None:
            win.player.reader.close()


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

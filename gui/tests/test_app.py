"""Phase 2: MainWindow + panels, offscreen (no window shown).

    QT_QPA_PLATFORM=offscreen pixi run -e gui python gui/tests/test_app.py
    pytest gui/tests/test_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from labeling_shell.behaviors_panel import BehaviorsPanel  # noqa: E402
from labeling_shell.bouts import Bout, LabelModel  # noqa: E402
from labeling_shell.bouts_panel import BoutsPanel  # noqa: E402
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402
from tools.index_video import write_index_video  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_SETTINGS_DIR = tempfile.mkdtemp(prefix="breal_test_settings_")


def _settings():
    """A throwaway settings file, so tests never touch the user's recent projects."""
    from PySide6 import QtCore
    fd, path = tempfile.mkstemp(suffix=".ini", dir=_SETTINGS_DIR)
    os.close(fd)
    return QtCore.QSettings(path, QtCore.QSettings.Format.IniFormat)


def _wait_until(cond, timeout=20.0):
    import time
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    return cond()


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
    win = MainWindow(autostart_backend=False, settings=_settings())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path, wait=True)
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
    win = MainWindow(autostart_backend=False, settings=_settings())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        win.project = ProjectStore(proj)
        win.project.create("t")
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path, wait=True)
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


def test_open_project_lists_videos_and_reopens_the_last_one():
    _app()
    from app import MainWindow
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        a, b = os.path.join(d, "a.mp4"), os.path.join(d, "b.mp4")
        write_index_video(a, 30, fps=30, gop=12)
        write_index_video(b, 30, fps=30, gop=12)
        win = MainWindow(autostart_backend=False, settings=_settings())
        win.project = ProjectStore(proj)
        win.project.create("t")
        win.load_video(a, wait=True)
        win.load_video(b, wait=True)
        win.project.register_video("gone.mp4", "gone.mp4", os.path.join(d, "gone.mp4"))
        win.player.reader.close()

        win2 = MainWindow(autostart_backend=False, settings=_settings())
        win2.open_project_dir(proj)
        assert _wait_until(lambda: not win2.player.loading)
        assert win2.current_video_id == safe_id("b.mp4")  # last opened, not first
        texts = [win2.videos_panel.list.item(i).text() for i in range(win2.videos_panel.list.count())]
        assert texts == ["a.mp4", "b.mp4", "gone.mp4  (missing)"]

        win2.videos_panel.list.itemClicked.emit(win2.videos_panel.list.item(0))
        assert _wait_until(lambda: not win2.player.loading)
        assert win2.current_video_id == safe_id("a.mp4")
        assert win2.project.last_video() == safe_id("a.mp4")
        win2.player.reader.close()


def test_remove_video_from_project():
    _app()
    from app import MainWindow
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        a, b = os.path.join(d, "a.mp4"), os.path.join(d, "b.mp4")
        write_index_video(a, 30, fps=30, gop=12)
        write_index_video(b, 30, fps=30, gop=12)
        win = MainWindow(autostart_backend=False, settings=_settings())
        win.project = ProjectStore(proj)
        win.project.create("t")
        win.load_video(a, wait=True)
        win.load_video(b, wait=True)
        win.behaviors_panel.add_behavior("groom")
        win._seek(2)
        win._handle_key("1", 0)
        win._seek(6)
        win._handle_key("1", 0)

        answers = [(False, False), (True, False), (True, False)]
        win._confirm_remove = lambda name: answers.pop(0)

        win.remove_project_video(safe_id("a.mp4"))  # cancelled
        assert len(win.project.list_videos()) == 2

        win.remove_project_video(safe_id("a.mp4"))  # not the open one: b stays up
        assert [v["id"] for v in win.project.list_videos()] == [safe_id("b.mp4")]
        assert win.current_video_id == safe_id("b.mp4") and win.player.reader is not None

        win.remove_project_video(safe_id("b.mp4"))  # the open one: player cleared
        assert win.project.list_videos() == []
        assert win.current_video_id is None and win.player.reader is None
        assert win.model.bouts == [] and [x.name for x in win.model.behaviors] == ["groom"]
        assert win.videos_panel.list.count() == 0
        # labels were autosaved and kept, so adding b back restores its bout
        assert [(x.start_frame, x.end_frame) for x in
                win.project.load_into_model(safe_id("b.mp4")).bouts] == [(2, 6)]


def test_relink_saves_immediately_and_fixes_siblings():
    _app()
    import time as _time
    from app import MainWindow
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj, vids = os.path.join(d, "proj"), os.path.join(d, "vids")
        os.makedirs(vids)
        a, b = os.path.join(vids, "a.mp4"), os.path.join(vids, "b.mp4")
        write_index_video(a, 30, fps=30, gop=10)
        write_index_video(b, 40, fps=30, gop=10)
        store = ProjectStore(proj)
        store.create("t")
        store.register_video("a.mp4", "a.mp4", "/mnt/elsewhere/vids/a.mp4", 30)
        store.register_video("b.mp4", "b.mp4", "/mnt/elsewhere/vids/b.mp4", 30)  # wrong count

        warnings = []
        real_info, real_open, real_warn = (QtWidgets.QMessageBox.information,
                                           QtWidgets.QFileDialog.getOpenFileName,
                                           QtWidgets.QMessageBox.warning)
        QtWidgets.QMessageBox.information = staticmethod(lambda *x, **k: None)
        QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *x, **k: (a, ""))
        QtWidgets.QMessageBox.warning = staticmethod(lambda *x, **k: warnings.append(x[2]))
        try:
            win = MainWindow(autostart_backend=False, settings=_settings())
            win.open_project_dir(proj)  # nothing on disk: nothing opens
            win.open_project_video("a.mp4")
            # saved before the load finishes, and b was found beside a
            assert {v["id"]: v["path"] for v in store.list_videos()} == {"a.mp4": a, "b.mp4": b}
            assert _wait_until(lambda: not win.player.loading)
            assert not warnings  # a's frame count matches
            win.open_project_video("b.mp4")
            assert _wait_until(lambda: not win.player.loading)
            assert len(warnings) == 1 and "40 frames" in warnings[0]
        finally:
            QtWidgets.QMessageBox.information = real_info
            QtWidgets.QFileDialog.getOpenFileName = real_open
            QtWidgets.QMessageBox.warning = real_warn
        win.player.reader.close()


def test_predictions_follow_their_video():
    """Saved predictions reload with their video, and never show on another."""
    _app()
    from app import MainWindow
    from labeling_shell.active_learning import AlState
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        a, b = os.path.join(d, "a.mp4"), os.path.join(d, "b.mp4")
        write_index_video(a, 30, fps=30, gop=12)
        write_index_video(b, 30, fps=30, gop=12)
        win = MainWindow(autostart_backend=False, settings=_settings())
        win.project = ProjectStore(proj)
        win.project.create("t")
        res = {"class_names": ["other", "rear"],
               "frames": [{"index": i, "predicted_class": 1, "uncertainty": 0.9,
                           "predictions": [0.1, 0.9]} for i in range(30)]}
        win.project.save_predictions(safe_id("a.mp4"), res,
                                     {"checkpoint": "/m/best.ckpt", "created": "2026-10-08T12:00:00",
                                      "crop_params": {"method": "localizer"}})

        win.load_video(a, wait=True)
        assert _wait_until(lambda: win.al_state() is not None)
        assert len(win.al_state().by_index) == 30
        assert "best.ckpt" in win.pipeline.active_learning.pred_info.text()

        win.load_video(b, wait=True)  # no saved predictions for b
        QtWidgets.QApplication.processEvents()
        assert win.al_state() is None and win.timeline.al_state is None
        assert "None" in win.pipeline.active_learning.pred_info.text()

        # a slow load must not overwrite results that arrived meanwhile
        win.load_video(a, wait=True)
        fresh = AlState()
        fresh.load_results({"class_names": ["x"], "frames": []})
        win.set_al_results(fresh, {"checkpoint": "/m/new.ckpt"})
        QtCore.QThread.msleep(300)
        QtWidgets.QApplication.processEvents()
        assert _wait_until(lambda: not win._bg_workers)
        assert win.al_state() is fresh
        win.player.reader.close()


def test_inference_results_saved_to_the_video_they_ran_on():
    _app()
    from app import MainWindow
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        a, b = os.path.join(d, "a.mp4"), os.path.join(d, "b.mp4")
        write_index_video(a, 30, fps=30, gop=12)
        write_index_video(b, 30, fps=30, gop=12)
        win = MainWindow(autostart_backend=False, settings=_settings())
        win.project = ProjectStore(proj)
        win.project.create("t")
        win.load_video(b, wait=True)
        res = {"class_names": ["other", "rear"],
               "frames": [{"index": 0, "predicted_class": 0, "uncertainty": 0.1}]}
        # a run started on a.mp4 lands while b.mp4 is open
        win.pipeline.active_learning._on_results(res, safe_id("a.mp4"), {"checkpoint": "/m/c.ckpt"})
        assert win.al_state() is None  # not shown on b
        assert _wait_until(lambda: win.project.find_predictions(safe_id("a.mp4")) is not None)
        assert win.project.load_predictions(safe_id("a.mp4"))["meta"]["checkpoint"] == "/m/c.ckpt"
        win.player.reader.close()


def test_start_dialog_choices_and_recent_list():
    _app()
    from labeling_shell.start_dialog import StartDialog, recent_projects, remember_project
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        for p in (a, b):
            ProjectStore(p).create(os.path.basename(p))
        st = _settings()
        remember_project(st, a)
        remember_project(st, b)
        remember_project(st, a)                       # re-opening moves it to the front
        remember_project(st, os.path.join(d, "gone"))  # no project.json: filtered out
        assert recent_projects(st) == [a, b]

        dlg = StartDialog(recent_projects(st))
        assert dlg.recent.count() == 2
        dlg.recent.itemActivated.emit(dlg.recent.item(1))
        assert dlg.result() == QtWidgets.QDialog.DialogCode.Accepted
        assert dlg.choice == ("open", b)

        for btn, kind in (("new_btn", "new"), ("open_btn", "browse")):
            dlg = StartDialog([])
            getattr(dlg, btn).click()
            assert dlg.choice == (kind, None)
        dlg = StartDialog([])
        dlg.quit_btn.click()
        assert dlg.result() == QtWidgets.QDialog.DialogCode.Rejected and dlg.choice is None


def test_prompt_for_project_retries_until_one_opens():
    _app()
    import app as appmod
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        proj = os.path.join(d, "proj")
        ProjectStore(proj).create("p")
        choices = [("open", os.path.join(d, "not_a_project")), ("open", proj)]

        class FakeDialog:
            def __init__(self, recent, *a, **k):
                self.choice = choices.pop(0)

            def exec(self):
                return True

        real_dialog, real_warning = appmod.StartDialog, QtWidgets.QMessageBox.warning
        appmod.StartDialog = FakeDialog
        QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
        try:
            win = appmod.MainWindow(autostart_backend=False, settings=_settings())
            assert win.prompt_for_project()
        finally:
            appmod.StartDialog, QtWidgets.QMessageBox.warning = real_dialog, real_warning
        assert not choices  # the bad folder was rejected and it asked again
        assert win.project.root == Path(proj)
        from labeling_shell.start_dialog import recent_projects
        assert recent_projects(win.settings) == [os.path.abspath(proj)]


def test_prompt_for_project_quit_returns_false():
    _app()
    import app as appmod

    class QuitDialog:
        def __init__(self, *a, **k):
            self.choice = None

        def exec(self):
            return False

    real = appmod.StartDialog
    appmod.StartDialog = QuitDialog
    try:
        win = appmod.MainWindow(autostart_backend=False, settings=_settings())
        assert not win.prompt_for_project()
        assert win.project is None
    finally:
        appmod.StartDialog = real


def test_mainwindow_al_results_drive_timeline_and_hud():
    _app()
    from app import MainWindow
    from labeling_shell.active_learning import AlState
    from labeling_shell.bouts import Bout
    win = MainWindow(autostart_backend=False, settings=_settings())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path, wait=True)
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


def test_snap_back_seeks_to_the_last_bout_end():
    _app()
    from app import MainWindow
    win = MainWindow(autostart_backend=False, settings=_settings())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        path = os.path.join(d, "clip.mp4")
        write_index_video(path, 60, fps=30, gop=12)
        win.load_video(path, wait=True)
        a = win.model.add_behavior("groom")

        # no bouts yet -> back to the start
        win._seek(40)
        win.snap_back()
        assert win.player.index == 0

        # "latest" is by end frame, not list order: bouts sort by start, so a
        # long early bout can outlast a later short one
        win.model.insert_bout_sorted(Bout(a.id, a.name, 5, 50))
        win.model.insert_bout_sorted(Bout(a.id, a.name, 20, 25))
        win._seek(3)
        win.snap_back()
        assert win.player.index == 50

        if win.player.reader is not None:
            win.player.reader.close()


def test_snap_back_without_a_video_is_harmless():
    _app()
    from app import MainWindow
    win = MainWindow(autostart_backend=False, settings=_settings())
    win.snap_back()  # must not raise


def test_player_has_no_fps_lock_control():
    """The FPS lock UI was removed; the reader-level override stays."""
    _app()
    from labeling_shell.video_widget import PlayerWidget
    p = PlayerWidget()
    assert not hasattr(p, "lock_chk")
    assert not hasattr(p, "fps_spin")
    assert hasattr(p, "snap_btn")
    p.set_fps_override(60)  # no reader loaded -> no-op, must not raise


def test_bout_behavior_combo_ignores_the_wheel():
    """Scrolling the bouts table must not reassign the bout under the cursor."""
    _app()
    from PySide6 import QtCore, QtGui
    m = LabelModel()
    a = m.add_behavior("groom")
    m.add_behavior("rear")
    m.insert_bout_sorted(Bout(a.id, a.name, 0, 10))
    panel = BoutsPanel(m, lambda: 30.0)
    panel.refresh()

    combo = panel.table.cellWidget(0, 2)
    assert combo.currentText() == "groom"

    event = QtGui.QWheelEvent(
        QtCore.QPointF(0, 0), QtCore.QPointF(0, 0),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    combo.wheelEvent(event)

    assert combo.currentText() == "groom"      # unchanged
    assert m.bouts[0].behavior_name == "groom"  # and not reassigned


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

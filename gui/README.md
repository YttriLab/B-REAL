# gui/ — all-Python (PySide6) frontend

The Python successor to the Electron/npm frontend under `LabelingApp/`, built on
the `python-rewrite` branch. `main` remains the working Electron app (restore
point: the `electron-baseline` tag).

Stack: **PySide6** (Qt) + **PyAV** (ffmpeg/libav) + NumPy. The existing
FastAPI/PyTorch backend is kept and reached over HTTP/SSE via a `BackendClient`
(local-spawned or remote). No JavaScript, no Electron, no npm.

## Run the app

```bash
C:/AdenCode/RealtimeASOID/.venv-gui/Scripts/python.exe C:/AdenCode/RealtimeASOID/gui/app.py
```

Open a video, add behaviors, press `1`–`0` to record bouts, Space/←/→ to
transport, `F` fullscreen. File menu: import/export CSV, new/open project.
Backend menu: connect a local or remote backend to use the Pipeline tabs
(Preprocess / Train / Export / Active Learning).

## Layout

```
gui/
  app.py                     # MainWindow: player + Label/Pipeline tabs, menus, shortcuts, project
  run_all_tests.py           # run every suite offscreen, aggregate PASS/FAIL
  requirements.txt           # frontend deps (av, numpy, PySide6-Essentials, httpx)
  labeling_shell/            # reusable, pipeline-agnostic shell (successor to core/*.js)
    video_reader.py          #   frame-accurate PyAV decoder (replaces WebCodecs+mp4box+remux)
    video_widget.py          #   VideoView + PlayerWidget (transport, scrubber, FPS lock, keys)
    qt_image.py              #   numpy RGB <-> QImage
    timefmt.py               #   frame<->time + MM:SS:mmm format/parse
    bouts.py                 #   Behavior/Bout model: toggle, sorted insert, overlap, merge
    csv_io.py                #   bout CSV, training one-hot matrix (load_data fmt), import auto-detect
    project.py               #   ProjectStore: project.json + per-video labels/<id>.json + CSV
    frame_ranges.py          #   bouts -> merged native frame ranges (computeLabeledFrameRanges)
    backend_worker.py        #   QThread workers: SSE stream + one-shot call, off the UI thread
    active_learning.py       #   AlState + label->model-class mapping (binarized false-pos fix)
    timeline_widget.py       #   uncertainty/misclassification heatmap strip, click-to-seek
    behaviors_panel.py       #   add/rename/remove/merge behaviors
    bouts_panel.py           #   bouts table: seek, reassign, edit, delete
    theme.qss / theme.py     #   dark QSS theme (applied once at QApplication level)
    backend/                 #   seam to the FastAPI backend (core/backend.js + main.js spawn)
      config.py / process.py / client.py / manager.py
  plugins/                   # RealtimeASOID pipeline UI (successor to plugins/pipeline.js)
    pipeline_config.py       #   pure config marshalling (crop params, train config)
    pipeline_panel.py        #   Preprocess / Train / Export tabs
    active_learning_panel.py #   run inference + review (uncertainty, next-uncertain)
  tools/index_video.py       # synthetic index-marked test videos (ground-truth per frame)
  tools/screenshot_app.py    # renders MainWindow to PNG via QWidget.grab() (native platform, no display needed)
  tests/                     # one suite per area; each runs standalone or under pytest
  packaging/                 # PyInstaller spec + per-OS build docs
```

## Dev environment

Isolated venv (`.venv-gui/`, gitignored) so the frontend never drags in the
backend's CUDA/torch stack:

```bash
python -m venv .venv-gui
.venv-gui/Scripts/pip install -r gui/requirements.txt   # Windows (…/bin on mac/linux)
```

## Tests

```bash
C:/AdenCode/RealtimeASOID/.venv-gui/Scripts/python.exe gui/run_all_tests.py
```

All suites run headless (Qt offscreen; synthetic videos). What needs the backend's
pixi/torch env — live preprocess/train/inference — is not exercised here; those
paths are covered at the contract level (stub server, config building).

## Status (python-rewrite branch)

- Phase 0 — frame-accurate PyAV video (frame-exact MP4+MKV, no remux). ✅
- Phase 1 — BackendClient seam (local-spawn or remote). ✅
- Phase 2 — core labeler: model, CSV, Qt panels, project persistence. ✅
- Phase 3 — pipeline UI: preprocess / train / export. ✅
- Phase 4 — active learning + overlays (uncertainty timeline, HUD). ✅
- Phase 5 — packaging (PyInstaller spec + docs). ✅ Promotion to `main` is
  deferred until parity sign-off (then merge + tag `electron-final`).

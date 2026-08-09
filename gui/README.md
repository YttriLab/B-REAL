# gui/ — all-Python (PySide6) frontend

This is the Python successor to the Electron/npm frontend under `LabelingApp/`.
It is being built on the `python-rewrite` branch; `main` remains the working
Electron app (restore point: the `electron-baseline` tag).

Target stack: **PySide6** (Qt) + **PyAV** (ffmpeg/libav) + NumPy. The existing
FastAPI/PyTorch backend is kept and reached via a `BackendClient` seam (Phase 1).

## Layout

```
gui/
  labeling_shell/        # reusable, pipeline-agnostic shell (successor to core/*.js)
    video_reader.py      #   frame-accurate random-access decoder (replaces WebCodecs + mp4box + remux)
    video_widget.py      #   VideoView + PlayerWidget (transport, scrubber, FPS lock, shortcuts)
    qt_image.py          #   numpy RGB <-> QImage
    backend/             #   seam to the FastAPI/PyTorch backend (successor to core/backend.js + main.js spawn)
      config.py          #     BackendConfig (local-spawn vs remote host)
      process.py         #     spawn/health/stop local uvicorn; interpreter resolution
      client.py          #     HTTP client + SSE parsing, one method per endpoint
      manager.py         #     Backend: client + (optional) process behind one object
  tools/
    index_video.py       # synthetic index-marked test videos (ground-truth per frame)
  tests/
    test_video_reader.py  # frame-exactness of the decoder (MP4 + MKV)
    test_qt_display.py     # frame-exactness of the decode->QImage->widget path (offscreen)
    test_backend_client.py # backend client HTTP/SSE contract (stdlib stub server)
  spike_player.py        # runnable minimal player for manual/visual checks
```

## Backend seam (Phase 1)

The GUI talks to the existing FastAPI backend over HTTP/SSE via
`labeling_shell.backend`. It is HTTP-only by design: for a single-machine
install it spawns a local `uvicorn` (`BackendConfig(spawn_local=True)`), and for
the split setup it connects to a remote one (`BackendConfig.remote(host)`) — e.g.
labeling on a light box, training on the 3090 rig. Callers use only
`Backend(...).client`; no backend files were modified.

```python
from labeling_shell.backend import Backend, BackendConfig
with Backend(BackendConfig()) as be:          # local spawn + health wait
    be.start()
    print(be.client.health())
    for ev in be.client.training_status(task_id):
        ...  # per-epoch metric dicts, streamed from SSE
```

## Dev environment

An isolated venv (`.venv-gui/`, gitignored) is used for the frontend so it does
not drag in the backend's CUDA/torch stack:

```
python -m venv .venv-gui
.venv-gui/Scripts/python.exe -m pip install av numpy PySide6-Essentials
```

## Run

```
# headless correctness (no display required)
.venv-gui/Scripts/python.exe gui/tests/test_video_reader.py
QT_QPA_PLATFORM=offscreen .venv-gui/Scripts/python.exe gui/tests/test_qt_display.py

# visual spike (opens a window; generates a sample if no path given)
.venv-gui/Scripts/python.exe gui/spike_player.py [video.mp4|video.mkv]
```

## Phase 0 result (go/no-go gate) — PASS

The riskiest assumption of the whole rewrite is that pure-Python frame-accurate
video is viable cross-platform. It is:

- **Frame-exact random access** on MP4 **and** MKV (H.264 with B-frames + small
  GOP): every decoded frame reports exactly the index requested.
- **`seek == sequential`**: a cold random seek is pixel-identical to decoding
  straight through — the keyframe-aware seek is exact.
- **MKV needs no remux** — PyAV demuxes it directly, so the Electron app's
  `remux-video` / `ensureRemuxedForWebcodecs` path is eliminated, not ported.
- **Throughput** (dev box, 320x240): ~295 fps sequential (~3.4 ms/frame) vs. the
  30 fps playback rate; ~40 ms for a cold random seek (worst case; instant once
  cached).
- Decode -> QImage -> widget round-trips losslessly and displays the exact frame.

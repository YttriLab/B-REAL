# Packaging the RealtimeASOID GUI

Cross-platform desktop bundles via **PyInstaller** (the SLEAP/DeepLabCut approach).
The GUI bundle ships PySide6 + PyAV (with ffmpeg libs) but **not** the backend —
the app spawns a local backend from the project's pixi env, or connects to a
remote one (e.g. the 3090 rig), so torch/CUDA stay out of the GUI bundle.

## Build (run on each target OS separately)

```bash
# Windows (from repo root, in the GUI venv)
.venv-gui/Scripts/pip install -r gui/requirements.txt pyinstaller
.venv-gui/Scripts/pyinstaller gui/packaging/realtimeasoid.spec
# -> dist/RealtimeASOID/RealtimeASOID.exe
```

```bash
# macOS / Linux
python -m venv .venv-gui
.venv-gui/bin/pip install -r gui/requirements.txt pyinstaller
.venv-gui/bin/pyinstaller gui/packaging/realtimeasoid.spec
# -> dist/RealtimeASOID/RealtimeASOID
```

PyInstaller does not cross-compile: build the Windows bundle on Windows, the mac
bundle on macOS, the Linux bundle on Linux.

## Notes / TODO

- **Icon:** add `icon=...` to the `EXE(...)` in the spec once an `.ico`/`.icns`
  exists (the Electron app used `labelerlogo.ico`).
- **macOS:** for a `.app` + notarization, add a `BUNDLE(...)` step and codesign;
  unsigned builds warn on first launch.
- **Backend discovery:** the packaged app resolves the backend interpreter via
  `RTASOID_PYTHON` / `PYTHON` / the project `.pixi` env (see
  `labeling_shell/backend/process.py`). For a remote backend, use Backend →
  Connect Remote in the app.
- The spec compiles as plain Python (PyInstaller injects `SPECPATH`, `Analysis`,
  etc. at build time); a full multi-OS build is left to run on each target.

# RealtimeASOID

A desktop application for building **real-time, clip-based animal behavior classifiers** — from hand-labeling video through training to active-learning refinement.

It pairs an Electron labeling UI with a Python (FastAPI + PyTorch Lightning) backend that handles video preprocessing, TSM model training, and per-frame inference. Trained models export to TorchScript for real-time deployment (e.g. Bonsai.ML rigs).

---

## What it does

A complete pipeline in one app, organized as four tabs:

1. **Label** — Load videos, define behaviors, and mark temporal *bouts* (start/end intervals) frame-by-frame.
2. **Preprocess** — Background subtraction + animal-centered cropping → flat frame images for training.
3. **Train** — Train a Temporal Shift Module (TSM) classifier on an EfficientNet backbone, with live metrics and optional Weights & Biases logging.
4. **Active Learning** — Run inference, see an uncertainty heatmap and misclassification markers on the timeline, and jump to the frames most worth relabeling.

---

## Requirements

- **Python** 3.10+ (with `pip`)
- **Node.js** 18+ (with `npm`)
- **ffmpeg** on your `PATH` *(optional)* — only needed to re-encode unsupported videos or to render labeled-video overlays
- A **CUDA GPU** is recommended for training/inference but not required (CPU works, slower)

---

## Installation

```bash
# 1. Python backend dependencies
pip install -r backend/requirements.txt

# 2. Electron app dependencies
cd LabelingApp
npm install
```

## Running

```bash
# From LabelingApp/ — launches the desktop app and auto-starts the backend
npm start
```

The Electron main process spawns the FastAPI backend (`uvicorn backend.server:app --port 8741`) automatically and shuts it down on exit. The header shows a **Backend** status pill (green = connected).

To run the backend on its own (e.g. for API testing):

```bash
# From the project root
python -m uvicorn backend.server:app --port 8741 --host 127.0.0.1
```

## Building installers

```bash
cd LabelingApp
npm run dist:win     # Windows (NSIS installer + portable)
npm run dist:mac     # macOS (DMG)
npm run dist:linux   # Linux (AppImage + deb)
```

---

## Usage walkthrough

### 1. Create or open a project
A project is required. On launch a gate prompts you to **New Project** (choose a folder), **Open Project**, or **Import .rlt** — these are also in the header **Project ▾** menu. A project keeps your videos, labels, configs, frames, and models together (videos in `frames/<id>/`, backgrounds in `backgrounds/`) and auto-restores on next launch.

### 2. Add videos
Click **+** in the **Videos** sidebar to add one or more video files. They appear in the sidebar; click one to load it.

> **Playback:** WebCodecs mode is on by default for true frame-accurate stepping and seeking. If a video's codec/container isn't supported (HEVC/H.265, AVI are common culprits), the app prompts you to either pick a different file or re-encode it to H.264 MP4.

### 3. Define behaviors
In the **Label** tab, add behaviors (e.g. *grooming*, *walking*). Each is auto-assigned a number key (`1`–`9`, `0`).

### 4. Mark bouts
Play or step through the video and press a behavior's number key **once to start** a bout and **again to end** it. Bouts appear in the scrollable list, where you can edit times, reassign behaviors, or jump to a bout's start/end.

### 5. Preprocess
In the **Preprocess** tab: compute a background, tune the crop preview, then run preprocessing. It extracts **only the labeled frames** (plus a short clip lead-in), cropped to the animal, into the project — so label first, then preprocess.

### 6. Train
In the **Train** tab: add data sources (frame dir + label CSV), set hyperparameters, and start training. Watch live loss/accuracy. Optionally enable **Weights & Biases** logging (project name, run name, and API key — the key is passed to the training process via environment variable and never written to disk). When a project is open, the best checkpoint is saved into it and recorded automatically.

### 7. Infer & refine (Active Learning)
In the **Active Learning** tab: select a checkpoint and run inference — it scores the **current video on the fly** (no frame directory needed), so it can flag uncertain regions you haven't labeled yet. A per-frame **uncertainty heatmap** (green → red) plus **misclassification markers** appear on the timeline. Jump to the next uncertain segment / misclassified frame, fix the label, and iterate.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`–`9`, `0` | Start / end a bout for that behavior |
| `Space` | Play / pause |
| `←` / `→` | Step one frame back / forward |
| `F` | Toggle fullscreen |
| Drag trackbar | Scrub — jumps to that frame live (no need to press play) |

---

## Projects & sharing (`.rlt` files)

A project is a folder:

```
my_experiment/
├── project.json     # manifest: videos, behaviors, configs, model refs
├── labels/          # frame-indexed one-hot label CSVs per video
├── frames/          # preprocessed frames (regenerable)
├── backgrounds/     # computed backgrounds (regenerable)
└── models/          # checkpoints + TorchScript exports
```

Share a project as a single HDF5 `.rlt` file (modeled on SLEAP), in two flavors from **Project ▾**:

- **Export .rlt (labels only)** — lightweight: manifest, behaviors, configs, video references, and your hand-labeled bouts. Small and emailable; regenerable data is excluded.
- **Export Package .rlt (everything)** — self-contained bundle that also embeds preprocessed frames, backgrounds, and model checkpoints/exports. Large, but trains anywhere with no re-preprocessing.

**Import** with **Project ▾ → Import .rlt** (auto-detects which flavor). On the new machine:

| Item | Lightweight `.rlt` | Package `.rlt` |
|------|--------------------|----------------|
| **Labels & config** | Restored automatically | Restored automatically |
| **Videos** | Referenced by path; re-link if it doesn't resolve | Same (videos aren't embedded) |
| **Frames / backgrounds** | Re-run Preprocess | Restored from the file |
| **Models** | Re-link the `.ckpt`, or re-train | Restored from the file |

---

## Import / export formats

**Label CSV import** auto-detects two layouts:

- **Bout CSV** — header contains `start`, `end`, `behavior`:
  ```csv
  start,end,behavior
  00:01.000,00:05.500,grooming
  ```
- **Frame-level matrix** — first column is frame index/time, remaining columns are 0/1 per behavior:
  ```csv
  time,grooming,walking,rearing
  0,1,0,0
  1,1,0,0
  2,0,1,0
  ```

XLSX/XLS label files are also supported on import.

**Export** produces a bout CSV or a one-hot frame-level CSV (optionally with a "fill" behavior for unlabeled frames). The **Import Video+Labels** action loads a video + label file from elsewhere directly into the current session.

---

## TorchScript export (real-time deployment)

```python
from convert_torchscript_tsm import export_model

export_model(
    checkpoint_path="lightning_logs/.../epoch=22.ckpt",
    output_name="tsm_realtime",
    output_dir="bonsai_models",
    tsm_stages=[1, 2, 3, 4, 5, 6, 7],
)
```

The resulting `.pt` is compatible with `Bonsai.ML.Torch`'s `LoadScriptModule` for real-time rig deployment.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| **Backend pill stays red** | Ensure Python deps are installed and `python` is on `PATH`; check the terminal for `[backend]` errors. |
| **Video won't load / "unsupported format"** | Choose a different MP4 (H.264) or WebM file, or use the prompt's *Re-encode* option (needs ffmpeg). |
| **WebCodecs unavailable** | The checkbox disables itself and falls back to HTML5 playback automatically. |
| **Re-encode / labeled-video fails** | Install ffmpeg and make sure it's on your `PATH`. |
| **Training is very slow** | Use a CUDA GPU; CPU training works but is much slower. |

---

## Testing

The backend has a pytest suite (106 tests) using synthetic frames/videos — no GPU or real data needed:

```bash
pytest          # all tests
pytest -v       # verbose
pytest tests/test_rlt_service.py   # a single module
```

The Electron renderer UI is not covered by automated tests.

---

## Repository layout

```
├── LabelingApp/        # Electron app (UI)
│   ├── main.js         # main process — spawns the backend, IPC handlers
│   ├── renderer.js     # all UI logic
│   ├── preload.js      # contextBridge IPC
│   ├── project-manager.js
│   └── index.html / styles.css
├── backend/            # FastAPI server (port 8741)
│   ├── server.py
│   ├── routers/        # preprocess, training, inference, project endpoints
│   └── services/       # core logic incl. rlt_service.py (.rlt files)
├── model.py            # EfficientNet classifier
├── model_tsm.py        # TSM (Temporal Shift Module) variant
├── convert_torchscript_tsm.py
├── tune_cropping.py    # cropping/background utilities
├── config_tsm.yaml     # training defaults
└── tests/              # pytest suite
```

For architecture and contributor details, see [CLAUDE.md](CLAUDE.md).

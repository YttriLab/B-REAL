# B-REAL

A desktop application for building **real-time, clip-based animal behavior classifiers** — from hand-labeling video through training to active-learning refinement.

It pairs a PySide6 labeling UI with a Python (FastAPI + PyTorch Lightning) backend that handles video preprocessing, TSM model training, hyperparameter sweeps, and per-frame inference. Trained models export to ONNX, OpenVINO or TensorRT for real-time deployment (e.g. Bonsai.ML rigs).

> The original Electron frontend was removed once the all-Python UI replaced it. It is preserved at the `electron-final` tag.

---

## What it does

A complete pipeline in one app:

1. **Label** — Load videos, define behaviors, and mark temporal *bouts* (start/end intervals) frame-by-frame.
2. **Preprocess** — Background subtraction + animal-centered cropping → flat frame images for training.
3. **Train** — Train a Temporal Shift Module (TSM) classifier on an EfficientNet backbone (b0–b7), with live metrics.
4. **Tune** — Optuna hyperparameter sweeps, optionally constrained to architectures that meet a p99 inference-latency budget in the deployment runtime.
5. **Active Learning** — Run inference, see an uncertainty heatmap and misclassification markers on the timeline, and jump to the frames most worth relabeling.

---

## Requirements

- **[pixi](https://pixi.sh)** — manages the entire Python side (PyTorch+CUDA, ffmpeg, and the native stack) from a single lockfile. Install once:
  - Windows (PowerShell): `iwr -useb https://pixi.sh/install.ps1 | iex`
  - macOS/Linux: `curl -fsSL https://pixi.sh/install.sh | bash`
- A **CUDA GPU** is recommended for training/inference but not required (see CPU note below).

> You no longer install Python, PyTorch, CUDA, ffmpeg, or PySide6 yourself — `pixi` provisions two pinned, project-local environments from one `pixi.toml`/`pixi.lock`: `default` (the backend — PyTorch+CUDA, ffmpeg, the training/tuning stack) and `gui` (the PySide6 frontend — no CUDA/torch, so it installs even on a machine with no GPU). The app discovers the backend env automatically.

---

## Installation

```bash
pixi install   # both environments — .pixi/envs/default (backend) and .pixi/envs/gui (frontend)
```

**CPU-only machine?** Edit `pixi.toml` and replace `pytorch-gpu` with `pytorch-cpu` under `[feature.backend.dependencies]`, then re-run `pixi install`. This only affects the backend environment — `gui` never depended on CUDA.

## Running

```bash
pixi run gui
```

Use the **Backend** menu to spawn a local backend (`uvicorn backend.server:app --port 8742`, run from the **backend pixi env**) or to attach to a remote one. `GET /api/health` reports the interpreter + CUDA status. To run the backend on its own: `pixi run backend`.

To run the backend on its own (e.g. for API testing) or the tests:

```bash
pixi run backend     # uvicorn backend.server:app --port 8742 --host 127.0.0.1
pixi run test        # pytest
```

## Building installers

PyInstaller spec and per-OS notes live in `gui/packaging/`.

> **Distribution status:** the dev/lab workflow above (`pixi install` → `python gui/app.py`) is the supported install today and needs no manual Python/CUDA setup. A fully turnkey end-user installer — bundling the `pixi` binary + backend source and running `pixi install` once on first launch (thin installer, ~no manual steps) — is the remaining packaging work tracked in the plan; until then, distribute by sharing the repo and having the recipient run `pixi install`.

---

## Usage walkthrough

### 1. Create or open a project
A project is required. On launch a gate prompts you to **New Project** (choose a folder), **Open Project**, or **Import .rlt** — these are also in the header **Project ▾** menu. A project keeps your videos, labels, configs, frames, and models together (videos in `frames/<id>/`, backgrounds in `backgrounds/`) and auto-restores on next launch.

### 2. Add videos
Click **+** in the **Videos** sidebar to add one or more video files. They appear in the sidebar; click one to load it.

> **Playback:** decoding goes through PyAV (ffmpeg/libav), so stepping and seeking are frame-exact on every container ffmpeg can read — MP4 and MKV included, with no remux step.

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
| **Backend won't connect** | Make sure you ran `pixi install` at the repo root; check the terminal for `[backend]` errors. The startup log prints the interpreter + torch/CUDA status. |
| **Video won't load** | PyAV reads whatever ffmpeg supports; if a file still fails, re-encode it to H.264 MP4 (ffmpeg is provided by the pixi env). |
| **Training is very slow** | Use a CUDA GPU; CPU works but is much slower. |
| **Sweep refuses to start on a latency budget** | The deployment runtime is missing or unusable. TensorRT needs a GPU of compute capability 7.5+ (Turing or newer); pick the OpenVINO/CPU target otherwise. The error names the reason. |
| **Training runs on CPU despite having a GPU** | The training log shows `Device: CPU — GPU unavailable (...)` with the reason. With the default `pixi.toml` (`pytorch-gpu`) the env is CUDA-enabled; check `GET /api/health` → `device` to confirm the app is using the pixi env. If you edited `pixi.toml` to `pytorch-cpu`, switch it back and re-run `pixi install`. |

### Selecting the backend Python interpreter (advanced)

Normally the app auto-discovers the pixi env (`.pixi/envs/default/`). To force a different interpreter, set **`BREAL_PYTHON`** (or `PYTHON`) before launching — it takes priority over the pixi env:

```powershell
# Windows (PowerShell)
$env:BREAL_PYTHON = "C:\path\to\python.exe"; pixi run gui
```
```bash
# macOS / Linux
export BREAL_PYTHON="/path/to/python"; pixi run gui
```

The chosen interpreter is printed as `[backend] launching with interpreter: ...`, and `/api/health` reports its torch/CUDA status.

---

## Testing

The backend has a pytest suite using synthetic frames/videos — no GPU or real data needed:

```bash
pixi run test                          # all tests
pixi run pytest -v                     # verbose
pixi run pytest tests/test_rlt_service.py   # a single module
```

The frontend has its own headless suite (Qt offscreen, synthetic videos):

```bash
pixi run -e gui python gui/run_all_tests.py
```

---

## Repository layout

```
├── gui/                # PySide6 desktop app (UI) — see gui/README.md
│   ├── app.py          # MainWindow: player + Label/Pipeline tabs
│   ├── labeling_shell/ # reusable, pipeline-agnostic labeler + backend seam
│   └── plugins/        # this project's pipeline UI (preprocess/train/tune/export)
├── backend/            # FastAPI server (port 8742)
│   ├── server.py
│   ├── routers/        # preprocess, training, tuning, inference, project endpoints
│   └── services/       # core logic incl. rlt_service.py (.rlt files)
├── model.py            # EfficientNet classifier
├── model_tsm.py        # TSM (Temporal Shift Module) variant
├── convert_torchscript_tsm.py
├── optimized_runtime.py  # ONNX -> OpenVINO / TensorRT export + latency benchmarking
├── tune_cropping.py    # cropping/background utilities
├── config_tsm.yaml     # training defaults
└── tests/              # pytest suite
```

For architecture and contributor details, see [CLAUDE.md](CLAUDE.md).

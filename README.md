# B-REAL

A desktop application for building **real-time, clip-based animal behavior classifiers** — from hand-labeling video through training to active-learning refinement.

It pairs a PySide6 labeling UI with a Python (FastAPI + PyTorch Lightning) backend that handles video preprocessing, TSM model training, hyperparameter sweeps, and per-frame inference. Trained models export to ONNX, OpenVINO or TensorRT for real-time deployment (e.g. Bonsai.ML rigs).

> The original Electron frontend was removed once the all-Python UI replaced it. It is preserved at the `electron-final` tag.

---

## What it does

A complete pipeline in one app:

1. **Label** — Load videos, define behaviors, and mark temporal *bouts* (start/end intervals) frame-by-frame.
2. **Preprocess** — Find the animal (background subtraction, or a SAM2-distilled localizer network) and crop around it → flat frame images for training.
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

**GPU or not, nothing to edit.** `pixi install` installs CUDA PyTorch when the NVIDIA driver supports CUDA 12, and the CPU build otherwise (no NVIDIA GPU, or an older driver). Needs a recent pixi (`pixi self-update`).

## Demo project

`demo/calms21_demo/` is a small project to try the whole pipeline on: four one-minute clips of two interacting mice from the [CalMS21](https://data.caltech.edu/records/s0vdx-0k302) dataset (CC BY 4.0), three with expert labels (attack / investigation / mount / other) and one left unlabeled for Active Learning. Open it from the start dialog; see `demo/README.md`.

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
In the **Preprocess** tab: pick a **crop method**, tune the crop preview, then run preprocessing. It extracts **only the labeled frames** (plus a short clip lead-in), cropped to the animal, into the project — so label first, then preprocess.

| Crop method | Finds the animal by | Needs |
|---|---|---|
| **Background subtraction** (default) | thresholding the difference from an averaged background | a background (**Compute background**) |
| **SAM2 localizer** | a ResNet18 heatmap network distilled from SAM2 body masks, run in batches on the GPU; it also predicts whether an animal is present, and holds the box still when it isn't | nothing: the bundled weights are used unless you choose or train others |
| **None** | — (whole frames) | nothing |

The localizer copes with what background subtraction can't: a drifting background, long stillness, an empty arena, or a hand in the cage. B-REAL ships one set of weights (`weights/localizer/mouse_topdown.pt`, trained on a single top-down IR mouse rig; see [its README](weights/localizer/README.md)). Check the preview: if the box misses the animal on your rig, train your own (below). The chosen weights are saved with the project. Active Learning crops with the same settings as this tab, so keep them matching what the model was trained on.

#### Train Localizer (optional)
The **Train Localizer** tab fits the localizer to your rig. SAM2 (Meta's Segment Anything 2, via Hugging Face `transformers`) tracks the animal through short clips of your videos, and a fast network is trained to reproduce SAM2's body centres. That network runs hundreds of times faster than SAM2.

- **Clips seeded by the current localizer**: tick videos and choose clips per video. Each clip starts where the current localizer is confident it sees the animal.
- **Clips seeded by a click**: for a rig where the current localizer doesn't work at all. Load the player's frame, click the animal, check that SAM2's green mask covers it, and add the seed.
- **Empty-arena videos** (optional): real "no animal" frames for the presence head. Synthetic empty frames are made either way.
- **Start from**: fine-tune the current localizer (the default), or train from ImageNet weights.

SAM2 labels about 10 frames/s on a GPU, so 6 clips × 500 frames take about 5 minutes. Its labels are cached in the project, so retraining with other settings skips straight to training. The model downloads on first use (about 150 MB for SAM2 tiny). Whole clips are held out to measure centre error against SAM2. That error is checked as training runs, and the best checkpoint is the one kept. When fine-tuning, the starting weights count as a candidate, so a run can't hand back something worse than it started from; the log says so when nothing beat them. Clips where SAM2 lost the animal, or tracked something of the wrong size, are left out and named in the log.

The result is saved to `models/localizer/` with a JSON card and a QC sheet. The QC sheet marks SAM2's centre in green and the new network's in magenta: check it, because a wrong SAM2 track teaches the wrong thing. **Use for cropping** switches the Preprocess tab to it. Re-run preprocessing before training a classifier on the new crops.

What to expect: on a colour 1280×720 rig where the bundled weights miss the animal (over 200 px median error, presence 0.09), 8 click-seeded clips of 400 frames fine-tuned in about 4 minutes to 2.9 px median error (p95 7 px) on held-out clips. On a rig where the bundled weights already work (20 px), a few clips won't beat them. They were trained on tens of thousands of frames, so keep them.

### 6. Train
In the **Train** tab: add data sources (frame dir + label CSV), set hyperparameters, and start training. Watch live loss/accuracy. Optionally enable **Weights & Biases** logging (project name, run name, and API key — the key is passed to the training process via environment variable and never written to disk). When a project is open, the best checkpoint is saved into it and recorded automatically.

### 7. Infer & refine (Active Learning)
In the **Active Learning** tab: select a checkpoint and run inference — it scores the **current video on the fly** (no frame directory needed), so it can flag uncertain regions you haven't labeled yet. Decoding runs ahead on its own thread — on the GPU (NVDEC) when that reproduces OpenCV's frames bit-for-bit, which it does for monochrome h264/hevc video (the **Decoder** setting; *Auto* checks and falls back to OpenCV) — and clips go through the model in batches.

**TensorRT, automatically.** With **Runtime: Auto** (the default) the classifier and — in localizer crop mode — the localizer each run as a TensorRT engine compiled for your GPU. Nothing to export by hand: the first run builds both engines from the checkpoint/weights (~0.5–2 min each; the status bar says so) and caches them, keyed by the weights' content, the input shapes, the GPU and the TensorRT version, so retraining or upgrading rebuilds automatically and everything else is instant. fp16 is the default (conv trunks in fp16, softmax/pooling/heads in fp32). Without a supported GPU (compute capability 7.5+) Auto quietly uses PyTorch. Engines live in `~/.cache/b-real/trt` (`%LOCALAPPDATA%\B-REAL\trt` on Windows); set `BREAL_TRT_CACHE` to move them. A per-frame **uncertainty heatmap** (green → red) plus **misclassification markers** appear on the timeline. Jump to the next uncertain segment / misclassified frame, fix the label, and iterate.

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
│   ├── routers/        # preprocess, localizer, training, tuning, inference, project endpoints
│   └── services/       # core logic incl. rlt_service.py (.rlt files)
├── model.py            # EfficientNet classifier
├── model_tsm.py        # TSM (Temporal Shift Module) variant
├── convert_torchscript_tsm.py
├── optimized_runtime.py  # ONNX -> OpenVINO / TensorRT export + latency benchmarking
├── tune_cropping.py    # cropping/background utilities
├── localizer.py        # SAM2-distilled localizer network + crop stabilisation
├── sam2_track.py       # SAM2 teacher: one click -> body-centre track (Train Localizer)
├── localizer_train.py  # SAM2 clips -> distilled localizer weights
├── weights/localizer/  # the bundled default localizer (+ model card)
├── video_decode.py     # cv2 / NVDEC frame source (bit-identical), prefetching
├── trt_engines.py      # automatic TensorRT engines for the localizer + classifier
├── config_tsm.yaml     # training defaults
└── tests/              # pytest suite
```

For architecture and contributor details, see [CLAUDE.md](CLAUDE.md).

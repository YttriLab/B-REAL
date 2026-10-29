# CLAUDE.md

## Project Overview

Real-time clip-based animal behavior classifier. A desktop app for labeling video with behavioral bouts, backed by a FastAPI server that handles video preprocessing, TSM model training (PyTorch Lightning), hyperparameter sweeps, and per-frame inference with active learning.

The frontend is the all-Python PySide6 app in `gui/` (`gui/app.py`), talking to the backend over HTTP/SSE.
The original Electron frontend (`LabelingApp/`) was removed once this replaced it. Two tags preserve it: `electron-baseline` (before the rewrite began) and `electron-final` (the last commit where it existed).

## Project Structure

```
├── gui/                   # PySide6 desktop app (the only frontend)
│   ├── app.py             # MainWindow: player + Label/Pipeline tabs, menus
│   ├── labeling_shell/    # OURS but pipeline-agnostic: player, bouts, CSV,
│   │                      # project store, backend seam. Names no pipeline.
│   └── plugins/           # this pipeline's UI: preprocess / train / tune /
│                          # export / active learning
├── backend/               # FastAPI REST server (port 8742)
│   ├── server.py          # App init + CORS
│   ├── routers/           # preprocessing, training, tuning, inference endpoints
│   └── services/          # Core logic (async tasks with cancellation)
├── model.py               # EfficientNet-based classifier
├── model_tsm.py           # TSM (Temporal Shift Module) variant
├── convert_torchscript_tsm.py  # Export to TorchScript
├── optimized_runtime.py   # ONNX -> OpenVINO / TensorRT export + p99 benchmarking
├── preprocess_video_train.py   # Video preprocessing CLI
├── tune_cropping.py       # Background-subtraction cropper (FindRatBoundingBox)
├── localizer.py           # SAM2-distilled localizer network + LocalizerCropper
├── sam2_track.py          # SAM2 teacher (HF transformers): click -> body-centre track
├── localizer_train.py     # Train Localizer: SAM2 clips -> distilled localizer weights
├── weights/localizer/     # bundled default localizer (mouse_topdown.pt) + model card
├── demo/                  # calms21_demo/ (small ready-to-open project, CalMS21 subset) + its build script
├── video_decode.py        # cv2 / NVDEC frame source, bit-identical to cv2
├── trt_engines.py         # auto-built, cached TensorRT engines (localizer + TSM) for inference
├── config_tsm.yaml        # Training hyperparameters
└── tests/                 # pytest suite
```

## Running the App

```bash
# Both environments from one pixi.toml/pixi.lock: default (backend — CUDA torch,
# ffmpeg) and gui (frontend — PySide6/PyAV, no CUDA).
pixi install

# Launch
pixi run gui
```

The Backend menu spawns `uvicorn backend.server:app --port 8742` and tears it down on exit, resolving the interpreter in order: **`BREAL_PYTHON`/`PYTHON` override → the project-local pixi env (`.pixi/envs/default/python.exe`) → bare `python`**. The pixi env is the supported path (no env var needed); the override is an advanced escape hatch. The chosen interpreter is logged as `[backend] launching with interpreter: ...`, and `GET /api/health` reports its torch/CUDA status (`device` block). `pixi.toml` is the source of truth for Python deps for both environments (`backend/requirements.txt` is a pip fallback for the backend only — nothing pip-installs the `gui` environment; that's pixi-only). Standalone backend: `pixi run backend`, tests: `pixi run test`, GUI tests: `pixi run -e gui python gui/run_all_tests.py`.

## Running the Backend Standalone

```bash
pixi run backend     # uvicorn backend.server:app --port 8742 --host 127.0.0.1
```

## Running Tests

```bash
pixi run test                              # all tests
pixi run pytest -v                         # verbose
pixi run pytest tests/test_api_routers.py  # single module
```

Tests use synthetic frames/videos — no GPU or real data required. `pytest.ini` sets `asyncio_mode = auto`.

## Project Directory Structure

Projects are saved as self-contained directories with a `project.json` manifest:

```
my_experiment/
├── project.json              # manifest (videos, behaviors, config)
├── labels/                   # auto-saved frame-indexed one-hot label CSVs per video
├── frames/                   # preprocessed frames per video
├── backgrounds/              # computed background images
├── predictions/              # Active Learning results, <video_id>.json (+ "meta")
└── models/
    ├── checkpoints/          # training checkpoints
    └── exports/              # TorchScript exports
```

- Video paths are **absolute** (stay where they are); all other paths are **relative** to project root. Each video record also keeps `rel_path` (the video relative to the project folder): a project on a share is mounted at different places on different machines, and every open rewrites `path` as the opening machine sees it, so `ProjectStore.list_videos()` resolves `path`, else `rel_path`. Relinking (a missing file, or right-click ▸ Locate File…) saves at once and relinks other missing videos found in the same folder; a relinked file whose frame count differs from the recorded one warns, since labels are by frame number.
- `gui/labeling_shell/project.py` manages project state. The project label file is a **frame-indexed one-hot matrix** (`time` column = native frame index, one 0/1 column per behavior), generated by `csv_io.py` — this is the exact format `model.py::load_data` consumes (it `argmax`es each row). Only in-bout frames get a row (sparse; lead-in/gap frames are context-only on disk, never labeled samples). The user-facing **Export bouts** button still writes the human-readable `start,end,behavior` timestamp CSV ; import auto-detects either format and closes runs on frame-index gaps so the sparse matrix round-trips.
- **A project is required.** On launch a start dialog (`labeling_shell/start_dialog.py`) asks to create a project, open one, or pick a recent one (remembered per user in `QSettings("B-REAL", "B-REAL")`); closing it quits. The backend is spawned only after a project is chosen. The File menu offers New / Open later; a project keeps videos, labels, configs, frames and models together. Tests pass `MainWindow(settings=...)` a throwaway settings file.
- **Videos list** (top of the Label tab, `videos_panel.py`) shows the manifest's videos; click to switch (a missing file prompts to relink, keeping its id). Opening a project reloads `last_video`. Videos open on a worker thread behind a progress overlay (`PlayerWidget.load_async`; the frame-index pass reads the whole file) — `load_video(..., wait=True)` is the synchronous path tests use.
- **Training crop follows the model**: preprocessing writes `frames/<id>/crop_params.json` (`{crop_params, mixed}`; a re-run with other settings marks it mixed); `TrainTab._register_model` copies the training frame dirs' crop onto the model entry (`crop_params`, or None + `crop_note` when mixed/unrecorded); the Active Learning tab ("Project model" picker, "Crop from") infers with the model's training crop by default, else the Preprocess tab, and lists differences in video px (`plugins/crop_summary.py`). Models from before this have no record and fall back to the Preprocess tab with a note.
- **Crop box overlay** (View ▸ Show Crop Box, `B`; `gui/plugins/crop_overlay.py` drawing through the shell's generic `VideoView.set_boxes`): solid green = the box preprocessing recorded in `crops.csv`; dashed amber = live from `/api/preprocess/crop-box` with the Preprocess tab's settings, requested only after the player rests (100 ms debounce). The backend keeps one cropper + open video between calls, so stepping forward continues smoothing exactly as preprocessing (tested frame-for-frame); a jump restarts it. `FrameCropper.last_boxes` / `FindRatBoundingBox.last_crop_rect` expose the boxes. Note `box_size` is in downsampled px for background subtraction (crop = box × `downsample_factor`) but resized-frame px for the localizer.
- **Predictions persist**: each Active Learning run is saved to `predictions/<video_id>.json` (the shared inference payload + `meta`: checkpoint, crop params, decoder/runtime, created) under the video it *started* on, even if another is open when it lands; opening a video clears the shown predictions and reloads its saved ones on a worker (`ProjectStore.find_predictions` also finds the Electron app's `predictions/<stem>.json`). `MainWindow._al_token` stops a slow reload from replacing newer results. The AL tab's "Predictions" line names the run.
- **Electron-era project folders** (behaviors without `id`, videos keyed by filename stem, labels in the entry's `labels` CSV) don't open as-is: convert once with `gui/tools/migrate_electron_project.py <dir> [--video-dir DIR]` (rewrites in place, keeps `project.electron-backup.json`).
- **Artifacts route into the project automatically:** Compute Background writes to `backgrounds/bg_<name>.png` and records the path on the video's manifest entry (auto-loaded when you switch videos); Preprocess writes **only labeled** frames to `frames/<id>/` named by native index (no manual output-directory setting), plus `crops.csv` (`frame,x,y,w,h`, native video px, merged across runs; loaders only read `*.png`). Active Learning needs no frame dir — it infers on the fly from the video.
- **Training data sources** are the Train tab's **Training videos** checklist of the project's videos (`ctx.video_sources()`: label CSV + `frames/<id>/` per video). Each video shows whether it is preprocessed for its *current* labels (`gui/plugins/preprocess_status.py`: ready / partial / not preprocessed / no labels, plus the recorded crop method), in the Videos list, the Preprocess tab and this checklist. A video is ticked when it has frames unless the user (un)ticked it. There is no "Add External Source" in the Python app.

## .rlt File Format

Projects export to a single HDF5 `.rlt` file in **two flavors** (mirroring SLEAP's `.slp` vs `.pkg.slp`):

- **Lightweight** (default, "Export .rlt (labels only)") — manifest + video references + hand-annotated label bouts. Small, shareable; frames/backgrounds/models are regenerated after import.
- **Package** (`package=True`, "Export Package .rlt (everything)") — all of the above PLUS preprocessed frames (gzip), background images, and model checkpoints/exports. A self-contained, train-anywhere bundle (large).

**HDF5 layout:**
```
/                               (attrs: rlt_version = 3, packaged)
/                               (attr: manifest — JSON string with paths + config)
/labels/{video_id}              (dataset: UTF-8 CSV string of bouts)
/frames/{video_id}/{idx}        (dataset: uint8 PNG bytes, gzip)   [package only]
/backgrounds/{video_id}         (dataset: uint8 PNG bytes, gzip)   [package only]
/models/checkpoints/{name}      (dataset: uint8 blob)              [package only]
/models/exports/{name}          (dataset: uint8 blob)              [package only]
/models/localizer/{name}        (dataset: uint8 blob)              [package only]
```

- Video paths are preserved in the manifest; on import the app prompts to re-link any video whose path doesn't resolve (SLEAP-style)
- `import_rlt` auto-detects which groups are present and restores accordingly (labels always; frames/backgrounds/models if packaged), writing `project.json` + the standard subdirs
- Export/import goes through the backend (`/api/project/export-rlt` with `package` flag, `/api/project/import-rlt`, `/api/project/inspect-rlt`)
- `rlt_service.py` handles all HDF5 read/write via `h5py`

## The labeling shell is pipeline-agnostic — keep it that way

`gui/labeling_shell/` is the reusable labeler: player, bout model, CSV I/O,
project store, panels, and the backend seam. `gui/plugins/` is this project's
TSM pipeline UI. The split is the same one the old Electron app maintained
against the upstream `behavior-labeler` package, and it exists for the same
reason: **IntegratedASOiD** (https://github.com/adeneagle/IntegratedASOiD) swaps
this project's TSM classifier for A-SOID's RandomForest and shares everything
else. Before the split the two projects kept 5,500-line copies of one renderer
that had drifted 1,948 lines apart, with 91 of 125 divergent hunks sitting in
one region — the tabs that configure and run a pipeline.

**The shell must not name pipeline-specific code.** `labeling_shell/` makes no
pipeline-specific backend calls — labeling works with the backend down. The
pipeline reaches the shell through the `ctx` duck-type documented at the top of
`gui/plugins/pipeline_panel.py` (`client()`, `label_paths()`, `frame_dirs()`,
`checkpoint_dir()`, `behaviors()`, `status()`, …), which `MainWindow` provides.

**Don't change the inference payload shape.** Both apps return
`{frames: [{index, predicted_class, uncertainty}], class_names}` indexed by
native video frame. That single contract is why the whole Active Learning panel
is shared code rather than two copies, and it is what the sharing rests on.

## Key Patterns

- **Behavior names**: normalized (trim + collapse internal whitespace, case-insensitive match) at every create/import/rename site; `mergeDuplicateBehaviors()` collapses existing duplicates on load. **Merge UI**: tick behavior rows + "Merge selected" → `mergeBehaviors(ids, targetId)` reassigns their bouts to one survivor (across the current video + all stored per-video progress via `reassignBoutsToBehavior`) and removes the rest (destructive).
- **Async task management**: Services use background tasks with task IDs, progress callbacks, and cancellation support. Task registries are module-level dicts cleared between tests via `autouse` fixture.
- **Pydantic v2**: `model_config` is a reserved field name — use `extra_model_config` instead.
- **CSV import**: `parseCsvAndImport(content, sourceName)` is the reusable core; `importCsv()` and Import Project both delegate to it.
- **One label per frame**: a new bout replaces what it overlaps (`LabelModel.place_bout` trims/splits other bouts, merges a touching same-behavior bout) — that is how relabeling works, keys or the AL tab's behavior buttons alike. Every bout change snapshots first; Edit ▸ Undo / Ctrl+Z (`LabelModel.undo`, per video, 100 deep).
- **Timeline** (`timeline_widget.py`): top lane = model (uncertainty, or predicted behavior), bottom lane = your bouts, both in `behavior_colors.py`'s per-behavior colors ("other" gray; model classes matched to behaviors by name); per-pixel aggregation (max uncertainty, every segment ≥1 px) from `AlState.arrays()`/`predicted_runs()`. Under it, `TimelineControls`: lane mode (saved in QSettings) and Jump to <behavior> in labels/predictions (◀ ▶, `[` `]`; previous = start of the current segment first, like a media player).
- **Misclassification detection**: `AlState.misclassified_in_bouts(bouts)` visits only labeled frames (the old per-predicted-frame scan took 1.4 s per keypress on a 440k-frame video; now ~14 ms). Electron's `buildMisclassifiedArray()` cross-references model predictions against user-labeled bouts. Hooked into `saveProgress()` with try-catch for TDZ safety (defined after `alState`). Compares in the model's label space via `mapLabelToModelClass()` — a binarized model predicts `['other', <positive>]`, so raw behavior names must be mapped first (returns null = "not comparable", never flagged).
- **MKV / non-MP4 containers**: decoding is PyAV (`gui/labeling_shell/video_reader.py`), which reads whatever ffmpeg reads, so there is no remux step and native frame indices are always the file's own. The Electron app needed mp4box (ISO/MP4 only) and remuxed everything else; that whole class of index-drift bug is gone.
- **Model mocking in tests**: `monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint", ...)` with a `FakeModel` returning fixed logits.

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/health` | GET | Health check (polled by the GUI when it spawns the backend) |
| `/api/preprocess/background` | POST | Compute background frame |
| `/api/preprocess/preview` | POST | Preview cropped frame |
| `/api/preprocess/crop-box` | POST | One frame's crop box (native px), same FrameCropper as preprocessing |
| `/api/preprocess/run` | POST | Start video preprocessing |
| `/api/preprocess/status/{task_id}` | GET | Preprocessing progress |
| `/api/preprocess/stop/{task_id}` | POST | Stop preprocessing |
| `/api/training/start` | POST | Start model training |
| `/api/training/status/{task_id}` | GET | SSE stream of training metrics |
| `/api/training/stop/{task_id}` | POST | Stop training |
| `/api/training/checkpoints` | GET | List saved checkpoints |
| `/api/training/export-torchscript` | POST | Export a `.ckpt` → TorchScript `.pt` (Train tab GUI button) |
| `/api/inference/run` | POST | Start inference |
| `/api/inference/status/{task_id}` | GET | Inference progress (SSE) |
| `/api/inference/results/{task_id}` | GET | Fetch inference results |
| `/api/inference/stop/{task_id}` | POST | Stop inference |
| `/api/localizer/info` | GET | Bundled localizer path, SAM2 availability |
| `/api/localizer/seed-preview` | POST | A frame, or SAM2's mask for a click on it |
| `/api/localizer/train` | POST | Start Train Localizer (SAM2 → localizer) |
| `/api/localizer/status/{task_id}` | GET | Train Localizer progress + log (SSE) |
| `/api/localizer/stop/{task_id}` | POST | Stop Train Localizer |
| `/api/project/export-rlt` | POST | Pack project folder into .rlt |
| `/api/project/import-rlt` | POST | Unpack .rlt into project folder |
| `/api/project/inspect-rlt` | POST | Read .rlt metadata without extracting |

## Workflow Pipeline

1. **Label**: Load video in player, define behaviors (keys 1-9, 0), annotate temporal bouts.
2. **Preprocess**: Video → optional **full-frame pre-resize** (`crop_params.resize_factor`, applied to the frame + background via `tune_cropping.pop_resize_factor`/`resize_frame`; the Preprocess panel shows the original vs processed frame size) → background subtraction (`crop_params.difference_mode`: "directional" = subject darker than bg, or "absolute" = |frame − bg|) → contour detection → center-of-mass crop, with temporal stabilization of the crop center (`crop_params.smoothing` EMA + `deadband` + `min_contour_area`/nearest-blob selection in `FindRatBoundingBox`) so the TSM classifier sees the animal's own motion rather than box jitter. The Preprocess panel can step frame-by-frame (Prev/Next), chaining the smoothed center so you can watch the box settle; `preview_frame` accepts/returns `center` for this. Crop method `"localizer"` swaps background subtraction for the localizer network (see Crop methods below). The same crop settings are applied at inference (`run_inference_video`) so AL crops match training. Extracts **only labeled frames** (each bout expanded by `frames_per_group-1` lead-in for clip context), named by **native frame index**: `frames/<id>/<nativeIndex>.png`. The frontend computes the ranges from the video's bouts (`frame_ranges.py`) and sends `frame_ranges` to `/api/preprocess/run`.
3. **Train**: Frame dirs + label CSVs → TSM classifier (EfficientNet backbone) → checkpoint in `lightning_logs/`. Optional W&B logging (project/run name, API key passed via env var to subprocess). **Binarize** is a checklist of behaviors: 0 checked = multi-class; 1 = binarize that behavior; ≥2 = combined into one positive class via `label_union` then binarized (the frontend sends `binarize` + `label_union`). The subprocess emits **verbose startup lines** (device, data sources, sample/class counts, class distribution, train/val/test split) into the metrics JSONL; the SSE endpoint streams **every** line in order (`read_all_metrics`, not just the latest) so nothing is lost. The UI shows a scrollable train log, plus **Best Val Loss** and **Stagnation** (`MetricsWriter` tracks `best_val_loss` and `stagnation_epochs` = epochs since improvement, shown as `n / patience` to mirror EarlyStopping). **Find LR** (Train tab) sends the same config with `find_lr: true`: the subprocess runs Lightning's `Tuner.lr_find` (200 steps, 1e-6 → 1e-1, nothing saved) and emits one `{lr_find, lrs, losses, suggested_lr}` line; the dialog (`gui/plugins/lr_finder_dialog.py`) plots it and Use sets the lr field. The suggestion is fastai's valley rule (`backend/services/lr_valley.py`), not Lightning's `suggestion()`, which differentiates against the linear lr and picked ~3e-6 on rig2 off a curve descending 1e-4 → 2e-3.
4. **Infer / Active Learning**: Checkpoint + **the video itself** → `inference_service.run_inference_video` decodes + crops frames on the fly (same `FrameCropper` as preprocessing) → per-frame predictions + normalized entropy, indexed by native frame number. No pre-extracted frame dir. Decode runs on a prefetch thread, each frame is transformed once and shared by the T clips containing it, and clips are batched (`batch_size`). Uncertainty heatmap (green→red) + misclassification ticks on the timeline; jump to next uncertain segment / misclassified frame for relabeling.

**Crop methods** (`crop_params.method`, `backend/services/cropping.py::FrameCropper` — the one place preprocessing and inference both crop through, so train/inference parity is structural; `tests/test_inference_service.py` checks it frame-for-frame): `"background"` (default, `FindRatBoundingBox`), `"localizer"` (SAM2-distilled ResNet18 heatmap + presence head, `localizer.py`; `crop_params.localizer_path`, default `localizer.DEFAULT_WEIGHTS` = the bundled `weights/localizer/mouse_topdown.pt`; reuses `box_size`/`smoothing`/`deadband`; holds the box when presence < `presence_threshold`, default 0.1 — the bundled net's presence drops to 0.2–0.4 for an animal at the edge), `"none"` (whole frame; the legacy `crop=False`). `resize_factor` applies to all three. Smoothing restarts at every gap in the extracted/inferred frame indices, in both services. The GUI saves the chosen method + knobs per project (`manifest.settings.preprocess` = `{method, params}`, via ctx `setting`/`set_setting`) and restores them on open, because Active Learning crops with whatever the Preprocess tab shows; a project with nothing saved starts from an Electron-era video entry's `preprocessing` dict, else defaults.

**Decoders** (`decoder`: `"auto"`/`"nvdec"`/`"cpu"`, `video_decode.py`): NVDEC via PyNvVideoCodec yields 2-D gray frames equal to every channel of cv2's BGR frame, via a LUT on the luma plane — valid only for monochrome video (chroma 128). `"auto"` checks the first 16 frames against cv2 and falls back to cv2 otherwise, so it never changes a pixel. Croppers and `cropping.to_rgb` accept both layouts; a gray PNG reads back as the same BGR image.

**Train Localizer** (`/api/localizer/*`, `backend/services/localizer_service.py`, GUI `gui/plugins/localizer_panel.py`): plan clips (click seeds + `clips_per_video` evenly spaced clips seeded by the current localizer's first confident frame) → SAM2-label each (`sam2_track.Sam2Tracker`, cached per clip as npz in `<project>/models/localizer/sam2_labels/`) → `localizer_train.train_localizer` (heatmap CE + L1 + presence BCE; synthetic paste-over negatives + optional empty-arena videos; held-out whole clips, or the last 20% of each clip when < 3 clips; validates every `eval_every` steps and keeps the best checkpoint, init weights counting as step 0, so `improved: False` instead of a regression; `odd_clips` drops clips whose median mask area is >3× off the others'; reports before/after error vs SAM2) → `<name>.pt` + `<name>.json` card + `<name>_qc.jpg` in `models/localizer/` (packaged by a full `.rlt`, not the label cache). Training sees `localizer.network_input`, the same function inference uses. The chosen weights persist per project as `manifest.settings.localizer_path` (relative to the project; `ProjectStore.get_setting/set_setting` is a generic store the shell never reads). **SAM2 traps**: one click is ambiguous: the video model's best-IoU pick was a 3k px patch of a rat's back (tracked all clip), so `Sam2Tracker.seed_mask` scores the click with the image model (`Sam2Model`, same checkpoint), cuts each of the 3 candidates to the connected piece under the click (the "whole" candidate carried floor speckles that lost the track in 2 frames), takes the largest under 20% of the frame, and seeds the video model with that mask; bump `sam2_track.LABEL_VERSION` when the recipe changes (it's in the label cache key). bfloat16 ruins the first-click mask (run fp32); pruning a streaming session needs explicit `frame_idx`, else frames reuse one index and the mask freezes. `transformers` is optional at runtime (`sam2_status()`; `/train` 503s without it).

**TensorRT runtime** (`trt_engines.py`): inference `runtime` (`"auto"`/`"tensorrt"`/`"torch"`, API default `"auto"`, service default `"torch"`) + `precision` (`fp16`/`fp32`) for the classifier; the localizer's is the crop setting `crop_params.localizer_runtime` (+ `localizer_precision`) so preprocessing and inference locate with the same engine (the GUI sends `"auto"`). Engines are built on first use from the loaded model and cached under `BREAL_TRT_CACHE` / the OS cache dir, keyed by weights content hash + shapes/overrides + precision + max batch + GPU arch + TRT version; builds are atomic and serialized per engine. **TensorRT 11 is strongly typed only** (no `BuilderFlag.FP16`, no `platform_has_fast_fp16`): fp16 is put in the ONNX graph (`optimized_runtime.half_features` for TSM, `trt_engines._half_localizer`) with casts so normalisation/softmax/pooling/heads stay fp32. `optimized_runtime.build_tensorrt` builds strongly typed where available, takes `max_batch` for dynamic-batch graphs. TSM export swaps `TemporalShift.forward` for a concat version on the copy being exported (checked equal first). `auto` falls back to PyTorch if TRT is unusable or a build fails; explicit `tensorrt` raises. Preview always uses the PyTorch localizer (needs the heatmap).

**Frame-index invariant:** native video frame number is the single canonical index across labels (`time` column), extracted frame filenames, and inference output. There is no framerate downsampling.

## CSV Formats

**Import auto-detects two formats:**

Bout CSV (header contains `start`, `end`, `behav*`):
```csv
start,end,behavior
00:01.000,00:05.500,grooming
```

Frame-level label matrix (header contains `time` or numeric first column):
```csv
time,grooming,walking,rearing
0,1,0,0
1,1,0,0
2,0,1,0
```

**Export** produces bout CSV (start/end as MM:SS.mmm) or one-hot frame-level CSV.

## Import Project

"Import Project" button loads a video + label CSV from another project into the labeling UI. Two file dialogs (video then CSV), loads video, parses labels via `csv_io.py`. New behaviors are auto-created if they don't exist in the current session.

## Data Storage

- **Labels/bouts**: the project store on disk — `project.json` plus per-video `labels/<id>.json` + CSV (`gui/labeling_shell/project.py`)
- **Preprocessed frames**: Flat numbered PNGs (e.g., `frames_out/video_name/0.png`)
- **Checkpoints**: `lightning_logs/version_N/checkpoints/*.ckpt`
- **Backgrounds**: `backgrounds/bg_videoname.png`
- **Training metrics**: JSON-lines files tailed by SSE endpoints

## TorchScript Export

```python
from convert_torchscript_tsm import export_model
export_model(
    checkpoint_path="lightning_logs/.../epoch=22.ckpt",
    output_name="tsm_realtime",
    output_dir="bonsai_models",
    tsm_stages=[1, 2, 3, 4, 5, 6, 7],
)
```

Output `.pt` file is compatible with Bonsai.ML.Torch `LoadScriptModule` for real-time rig deployment.

## Build / Distribution

PyInstaller spec and per-OS notes: `gui/packaging/`.

## Dependencies

(`pixi.toml` is the authoritative list — conda-forge env with a `pixi.lock`; `backend/requirements.txt` is a pip fallback.)

- **Python** (via pixi/conda-forge): FastAPI, Uvicorn, PyTorch (`pytorch`; `pixi install` takes the CUDA build when the driver supports CUDA 12, else the CPU build — the CUDA platform variants come first in `platforms`), torchvision, Lightning, torchmetrics, timm, OpenCV, Albumentations, OmegaConf, numpy, pandas, h5py (`.rlt` files), wandb (optional training logging), matplotlib, pyyaml, scipy, optuna (hyperparameter sweeps), onnx + openvino (deployment-format export and latency measurement — see `optimized_runtime.py`), tensorrt-cu12 (PyPI, under `[pypi-dependencies]` — the GPU deployment runtime; needs a GPU of compute capability 7.5+, i.e. Turing or newer, so it will not run on Pascal cards like a GTX 10xx), PyNvVideoCodec (PyPI, NVDEC video decoding for `video_decode.py`; optional at runtime — `decoder="auto"` falls back to cv2), transformers (SAM2 teacher for Train Localizer, `sam2_track.py`; optional at runtime), **ffmpeg** (re-encode / labeled-video, now provided by the env)
- **Frontend** (pixi `gui` environment, `[feature.gui]` in `pixi.toml`): PySide6, PyAV, numpy, httpx, pyinstaller
- **Test**: pytest, pytest-asyncio, httpx

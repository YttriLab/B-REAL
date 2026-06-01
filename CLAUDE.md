# CLAUDE.md

## Project Overview

Real-time clip-based animal behavior classifier. An Electron desktop app for labeling video with behavioral bouts, backed by a FastAPI server that handles video preprocessing, TSM model training (PyTorch Lightning), and per-frame inference with active learning.

## Project Structure

```
├── LabelingApp/           # Electron app (labeling UI)
│   ├── main.js            # Main process — spawns Python backend
│   ├── renderer.js        # Renderer process — all UI logic (~10K LOC)
│   ├── preload.js         # IPC bridge (contextBridge)
│   └── index.html / styles.css
├── backend/               # FastAPI REST server (port 8741)
│   ├── server.py          # App init + CORS
│   ├── routers/           # preprocessing, training, inference endpoints
│   └── services/          # Core logic (async tasks with cancellation)
├── model.py               # EfficientNet-based classifier
├── model_tsm.py           # TSM (Temporal Shift Module) variant
├── convert_torchscript_tsm.py  # Export to TorchScript
├── preprocess_video_train.py   # Video preprocessing CLI
├── tune_cropping.py       # Crop parameter tuning
├── config_tsm.yaml        # Training hyperparameters
└── tests/                 # pytest suite (83 tests)
```

## Running the App

```bash
# Install Python deps
pip install -r backend/requirements.txt

# Install Node deps + launch Electron (auto-starts backend)
cd LabelingApp
npm install
npm start
```

The Electron main process spawns `uvicorn backend.server:app --port 8741` automatically and tears it down on exit.

## Running the Backend Standalone

```bash
python -m uvicorn backend.server:app --port 8741 --host 127.0.0.1
```

## Running Tests

```bash
pytest          # all 83 tests
pytest -v       # verbose
pytest tests/test_api_routers.py   # single module
```

Tests use synthetic frames/videos — no GPU or real data required. `pytest.ini` sets `asyncio_mode = auto`.

## Project Directory Structure

Projects are saved as self-contained directories with a `project.json` manifest:

```
my_experiment/
├── project.json              # manifest (videos, behaviors, config)
├── labels/                   # auto-saved bout CSVs per video
├── frames/                   # preprocessed frames per video
├── backgrounds/              # computed background images
└── models/
    ├── checkpoints/          # training checkpoints
    └── exports/              # TorchScript exports
```

- Video paths are **absolute** (stay where they are); all other paths are **relative** to project root
- `project-manager.js` manages project state; `saveProgress()` dual-writes to both localStorage and project labels
- New/Open/Close Project buttons in the header; last project auto-restored on startup
- Backward compatible: app works without a project (localStorage-only mode)

## .rlt File Format

Projects can be exported to a single `.rlt` file — a **lightweight, uncompressed HDF5** modeled on SLEAP's base `.slp` format. It stores only the irreplaceable parts of a project; regenerable data is rebuilt after import.

**Stored:** manifest (project info, behaviors, all config), video references (paths + filenames), and hand-annotated label bouts.
**Not stored:** preprocessed frames, backgrounds, model checkpoints (all regenerable from videos + config).

**HDF5 layout:**
```
/                               (attr: rlt_version = 2)
/                               (attr: manifest — JSON string with paths + config)
/labels/{video_id}              (dataset: UTF-8 CSV string of bouts)
```

- Video paths are preserved in the manifest; on import, the app prompts to re-link any video whose path doesn't resolve on the new machine (SLEAP-style)
- On import, `import_rlt` writes `project.json` + label CSVs and creates the standard empty subdirs (`frames/`, `backgrounds/`, `models/`) for regenerated data
- Export/import goes through the backend (`/api/project/export-rlt`, `/api/project/import-rlt`, `/api/project/inspect-rlt`)
- `rlt_service.py` handles all HDF5 read/write via `h5py`; no compression (labels are small text)

## Key Patterns

- **Async task management**: Services use background tasks with task IDs, progress callbacks, and cancellation support. Task registries are module-level dicts cleared between tests via `autouse` fixture.
- **Pydantic v2**: `model_config` is a reserved field name — use `extra_model_config` instead.
- **CSV import**: `parseCsvAndImport(content, sourceName)` is the reusable core; `importCsv()` and Import Project both delegate to it.
- **Misclassification detection**: `buildMisclassifiedArray()` cross-references model predictions against user-labeled bouts. Hooked into `saveProgress()` with try-catch for TDZ safety (defined after `alState`).
- **Model mocking in tests**: `monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint", ...)` with a `FakeModel` returning fixed logits.

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/preprocess/background` | POST | Compute background frame |
| `/api/preprocess/preview` | POST | Preview cropped frame |
| `/api/preprocess/start` | POST | Start video preprocessing |
| `/api/preprocess/task/{id}` | GET | Preprocessing progress |
| `/api/training/start` | POST | Start model training |
| `/api/training/metrics/{id}` | GET | Stream metrics (JSON lines) |
| `/api/training/stop/{id}` | POST | Stop training |
| `/api/training/checkpoints` | GET | List saved checkpoints |
| `/api/inference/run` | POST | Start inference |
| `/api/inference/task/{id}` | GET | Inference progress/results |
| `/api/project/export-rlt` | POST | Pack project folder into .rlt |
| `/api/project/import-rlt` | POST | Unpack .rlt into project folder |
| `/api/project/inspect-rlt` | POST | Read .rlt metadata without extracting |

## Workflow Pipeline

1. **Preprocess**: Video → background subtraction → contour detection → center-of-mass crop → flat PNGs (`output_dir/video_name/0.png, 1.png, ...`)
2. **Label**: Load video in player, define behaviors (keys 1-9, 0), annotate temporal bouts, export CSV
3. **Train**: Frame dirs + label CSVs → TSM classifier (EfficientNet backbone) → checkpoint in `lightning_logs/`. Optional W&B logging (project/run name, API key passed via env var to subprocess)
4. **Infer**: Checkpoint + frame dir → per-frame predictions + normalized entropy uncertainty
5. **Active Learning**: Uncertainty heatmap (green→red) + misclassification markers (magenta ticks) on timeline, jump to next uncertain/misclassified frame for relabeling

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

"Import Project" button loads a video + label CSV from another project into the labeling UI. Two file dialogs (video then CSV), loads video, parses labels via `parseCsvAndImport()`. New behaviors are auto-created if they don't exist in the current session.

## Data Storage

- **Labels/bouts**: `localStorage["bouts"]` and `localStorage["behaviors"]` in Electron renderer
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

```bash
cd LabelingApp
npm run dist:win    # Windows (NSIS + portable)
npm run dist:mac    # macOS (DMG)
npm run dist:linux  # Linux (AppImage + deb)
```

## Dependencies

- **Python**: FastAPI, Uvicorn, PyTorch, Lightning, timm, OpenCV, Albumentations, OmegaConf, matplotlib, pyyaml, scipy
- **Node**: Electron 31, electron-builder, mp4box
- **Test**: pytest, pytest-asyncio, httpx

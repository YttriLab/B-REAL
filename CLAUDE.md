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

## Build / Distribution

```bash
cd LabelingApp
npm run dist:win    # Windows (NSIS + portable)
npm run dist:mac    # macOS (DMG)
npm run dist:linux  # Linux (AppImage + deb)
```

## Dependencies

- **Python**: FastAPI, Uvicorn, PyTorch, Lightning, timm, OpenCV, Albumentations, OmegaConf
- **Node**: Electron 31, electron-builder, mp4box
- **Test**: pytest, pytest-asyncio, httpx

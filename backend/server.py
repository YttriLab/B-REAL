"""
FastAPI backend for B-REAL.

Provides endpoints for video preprocessing, model training, and inference
with active learning support. Spawned by the PySide6 GUI (gui/app.py).

Usage:
    python -m uvicorn backend.server:app --port 8742
    (run from the repo root)
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the project root is importable so tune_cropping, model, model_tsm resolve
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routers import preprocessing, training, tuning, inference, project  # noqa: E402


def device_info() -> dict:
    """Report the interpreter + GPU the backend (and thus training) will use."""
    info = {"python": sys.executable, "cuda_available": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda  # None for CPU-only wheels
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            info["CUDA_VISIBLE_DEVICES"] = cvd
    except Exception as e:  # torch not importable, etc.
        info["torch_error"] = f"{type(e).__name__}: {e}"
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print(f"[backend] project root: {PROJECT_ROOT}")
    print(f"[backend] sys.path includes project root: {PROJECT_ROOT in sys.path}")
    di = device_info()
    print(f"[backend] python: {di.get('python')}")
    print(f"[backend] torch {di.get('torch')} (cuda build {di.get('torch_cuda_build')}) — "
          f"cuda_available={di.get('cuda_available')} gpus={di.get('gpus')}")
    yield
    # Cleanup: cancel any running tasks
    from backend.services.preprocess_service import cancel_all as preprocess_cancel
    from backend.services.training_service import cancel_all as training_cancel
    from backend.services.tuning_service import cancel_all as tuning_cancel
    from backend.services.inference_service import cancel_all as inference_cancel
    preprocess_cancel()
    # Sweeps first: each holds a live training subprocess of its own.
    tuning_cancel()
    training_cancel()
    inference_cancel()


app = FastAPI(title="B-REAL Backend", lifespan=lifespan)

# Allow requests from Electron renderer (any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(preprocessing.router, prefix="/api/preprocess", tags=["preprocessing"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(tuning.router, prefix="/api/tuning", tags=["tuning"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(project.router, prefix="/api/project", tags=["project"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": device_info()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8742, reload=False)

"""
FastAPI backend for RealtimeASOID.

Provides endpoints for video preprocessing, model training, and inference
with active learning support. Designed to be spawned by the Electron app.

Usage:
    python -m uvicorn backend.server:app --port 8741
    (run from C:\\AdenCode\\RealtimeASOID)
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

from backend.routers import preprocessing, training, inference  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print(f"[backend] project root: {PROJECT_ROOT}")
    print(f"[backend] sys.path includes project root: {PROJECT_ROOT in sys.path}")
    yield
    # Cleanup: cancel any running tasks
    from backend.services.preprocess_service import cancel_all as preprocess_cancel
    from backend.services.training_service import cancel_all as training_cancel
    from backend.services.inference_service import cancel_all as inference_cancel
    preprocess_cancel()
    training_cancel()
    inference_cancel()


app = FastAPI(title="RealtimeASOID Backend", lifespan=lifespan)

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
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8741, reload=False)

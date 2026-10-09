"""Training router: start/stop training, stream metrics, list checkpoints."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.training_service import (
    start_training,
    read_latest_metrics,
    read_all_metrics,
    stop_training,
    list_checkpoints,
    get_task,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """Full training configuration — mirrors config_tsm.yaml structure."""
    # Data loading
    label_paths: list[str]
    frame_dirs: list[str]
    frames_per_group: int = 3
    label_union: Optional[dict] = None
    binarize: Optional[str] = None

    # Data splits
    train_percent: float = 0.7
    val_percent: float = 0.15
    test_percent: float = 0.15
    shuffle: bool = False

    # Model
    num_frames: int = 3
    variant: str = "b0"
    shift_fraction: float = 0.125
    unidirectional: bool = False
    pretrained: bool = True
    tsm_stages: list[int] = [5, 6, 7]
    dropout: float = 0.2
    lr: float = 1e-4
    weight_decay: float = 1e-3
    temporal_pool: str = "last"

    # Data loader
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 224

    # Training
    max_epochs: int = 100
    early_stopping_patience: int = 20

    # Boundary smoothing
    boundary_smoothing_enabled: bool = False
    boundary_smoothing_epsilon: float = 0.3
    boundary_smoothing_window: int = 5

    # Logging
    use_wandb: bool = False
    project_name: str = "b-real"
    run_name: Optional[str] = None
    wandb_api_key: Optional[str] = None

    # Output
    checkpoint_dir: Optional[str] = None

    # Learning-rate range test instead of training: a short run with the
    # learning rate rising exponentially, reporting loss against it and
    # Lightning's suggestion. Nothing is saved.
    find_lr: bool = False


class CheckpointSearchRequest(BaseModel):
    search_dirs: Optional[list[str]] = None


class CoverageRequest(BaseModel):
    label_paths: list[str]
    frame_dirs: list[str]
    frames_per_group: int = 3


class ExportTorchscriptRequest(BaseModel):
    checkpoint_path: str
    output_name: str = "tsm_model"
    output_dir: str
    tsm_stages: Optional[list[int]] = None


class ExportOptimizedRequest(BaseModel):
    """Export a checkpoint to a deployment format (what the rig actually runs)."""
    checkpoint_path: str
    output_dir: str
    output_name: str = "tsm_model"
    format: str = "onnx"        # onnx | openvino | tensorrt
    tsm_stages: Optional[list[int]] = None
    num_frames: Optional[int] = None
    image_size: int = 224
    batch_size: int = 1
    precision: str = "fp32"     # fp32 | fp16


class BenchmarkRequest(BaseModel):
    """Architecture to time the optimized inference path for (pre-training)."""
    variant: str = "b0"
    num_frames: int = 3
    tsm_stages: Optional[list[int]] = None
    temporal_pool: str = "last"
    unidirectional: bool = False
    image_size: int = 224
    batch_size: int = 1
    precision: str = "fp32"  # fp32 | fp16
    iters: int = 100
    warmup: int = 20
    # Deployment runtime to measure in. "auto" picks TensorRT on a CUDA box and
    # OpenVINO otherwise; "torchscript" is kept for comparison, not deployment.
    target: str = "auto"     # auto | openvino | tensorrt | torchscript


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/start")
async def api_start_training(req: TrainingConfig):
    """Start model training in a subprocess. Returns task_id."""
    # Convert to config_tsm.yaml-compatible dict
    config = {
        "data_loading": {
            "label_paths": req.label_paths,
            "frame_dirs": req.frame_dirs,
            "frames_per_group": req.frames_per_group,
            "label_union": req.label_union,
            "binarize": req.binarize,
        },
        "data": {
            "train_percent": req.train_percent,
            "val_percent": req.val_percent,
            "test_percent": req.test_percent,
            "shuffle": req.shuffle,
        },
        "model": {
            "num_frames": req.num_frames,
            "variant": req.variant,
            "shift_fraction": req.shift_fraction,
            "unidirectional": req.unidirectional,
            "pretrained": req.pretrained,
            "tsm_stages": req.tsm_stages,
            "dropout": req.dropout,
            "lr": req.lr,
            "weight_decay": req.weight_decay,
            "temporal_pool": req.temporal_pool,
        },
        # Separate from "model" on purpose: the trainer splats config.model into
        # TSMClassifier(**...), which would reject these.
        "data_loader": {
            "batch_size": req.batch_size,
            "num_workers": req.num_workers,
            "image_size": req.image_size,
        },
        "max_epochs": req.max_epochs,
        "callbacks": {
            "early_stopping_patience": req.early_stopping_patience,
        },
        "boundary_smoothing": {
            "enabled": req.boundary_smoothing_enabled,
            "epsilon": req.boundary_smoothing_epsilon,
            "window": req.boundary_smoothing_window,
        },
        "use_wandb": req.use_wandb,
        "project_name": req.project_name,
        "run_name": req.run_name or "training",
        "wandb_api_key": req.wandb_api_key,
        "fine_tune_lr": False,
        "find_lr": req.find_lr,
    }
    if req.checkpoint_dir:
        config["checkpoint_dir"] = req.checkpoint_dir

    try:
        task_id = start_training(config)
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_training_status(task_id: str):
    """SSE stream of training metrics."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        # Stream every event line in order (startup info messages + per-epoch
        # metrics), not just the latest, so nothing written between polls is lost.
        sent = 0
        while True:
            lines = read_all_metrics(task_id)
            while sent < len(lines):
                yield f"data: {json.dumps(lines[sent])}\n\n"
                sent += 1
            # Terminal states (error/cancelled) live on the task, not the file —
            # emit a final event carrying status + error detail, then stop.
            status = get_task(task_id).get("status")
            if status in ("completed", "error", "cancelled"):
                yield f"data: {json.dumps(read_latest_metrics(task_id))}\n\n"
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/stop/{task_id}")
async def api_stop_training(task_id: str):
    """Terminate a running training subprocess."""
    success = stop_training(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already stopped")
    return {"status": "cancelled"}


@router.post("/check-coverage")
async def api_check_coverage(req: CoverageRequest):
    """Report how many labeled frames have preprocessed PNGs ready for training.

    A lightweight (stdlib-only) pre-flight check so the GUI can warn before a run
    that some labels would be silently dropped for lack of extracted frames.
    """
    from data_coverage import compute_coverage
    try:
        return compute_coverage(req.frame_dirs, req.label_paths, req.frames_per_group)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints")
async def api_list_checkpoints(search_dirs: Optional[str] = None):
    """List available checkpoint files."""
    dirs = search_dirs.split(",") if search_dirs else None
    checkpoints = list_checkpoints(dirs)
    return {"checkpoints": checkpoints}


@router.post("/export-torchscript")
async def api_export_torchscript(req: ExportTorchscriptRequest):
    """Export a .ckpt to a single TorchScript .pt (for Bonsai / real-time rigs)."""
    import os

    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.checkpoint_path}")

    def _run():
        import torch
        from convert_torchscript_tsm import export_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return export_model(
            checkpoint_path=req.checkpoint_path,
            output_name=req.output_name,
            output_dir=req.output_dir,
            device=device,
            tsm_stages=req.tsm_stages,
        )

    try:
        output_path = await asyncio.to_thread(_run)
        return {"output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/export-optimized")
async def api_export_optimized(req: ExportOptimizedRequest):
    """Export a .ckpt to ONNX / OpenVINO IR / a TensorRT engine — the same
    formats the latency gate measures, so the rig runs what was benchmarked."""
    import os

    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.checkpoint_path}")

    def _run():
        from optimized_runtime import export_optimized
        return export_optimized(
            req.checkpoint_path, req.output_dir, req.output_name,
            fmt=req.format, tsm_stages=req.tsm_stages, num_frames=req.num_frames,
            image_size=req.image_size, batch_size=req.batch_size,
            precision=req.precision,
        )

    try:
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/benchmark-inference")
async def api_benchmark_inference(req: BenchmarkRequest):
    """Time inference for an architecture in its deployment runtime before
    training it — so a user can size their real-time compute budget."""
    def _run():
        from optimized_runtime import benchmark_optimized
        return benchmark_optimized(
            target=req.target,
            variant=req.variant,
            num_frames=req.num_frames,
            tsm_stages=req.tsm_stages,
            temporal_pool=req.temporal_pool,
            unidirectional=req.unidirectional,
            image_size=req.image_size,
            batch_size=req.batch_size,
            precision=req.precision,
            iters=req.iters,
            warmup=req.warmup,
        )

    try:
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # A missing runtime is a 503: the request is fine, this box can't serve it.
        from optimized_runtime import RuntimeUnavailable
        status = 503 if isinstance(e, RuntimeUnavailable) else 500
        raise HTTPException(status_code=status, detail=f"{type(e).__name__}: {e}")

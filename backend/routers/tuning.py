"""Tuning router: start/stop a hyperparameter sweep and stream trial results."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.routers.training import TrainingConfig
from backend.services.tuning_service import (
    start_tuning,
    stop_tuning,
    read_all_results,
    get_task,
)

router = APIRouter()


class RangeSpec(BaseModel):
    """A continuous hyperparameter swept over [low, high]."""
    low: float
    high: float
    log: bool = False


class CategoricalSpec(BaseModel):
    """A discrete hyperparameter swept over a fixed set of choices.

    Choices must be hashable (Optuna requirement), so `tsm_stages` is carried as
    comma-strings like "5,6,7" and parsed into list[int] after sampling.
    """
    choices: list


class LatencyConstraint(BaseModel):
    """Reject architectures whose tail inference latency blows the frame budget.

    Checked *before* a trial trains, by exporting the sampled architecture to its
    deployment format (OpenVINO on CPU, TensorRT on GPU) and timing it — minutes
    of work that saves a whole training run. Only architecture-shaped parameters
    (variant, frames, stages, pool, image size) affect it, so results are cached
    per architecture within a sweep.
    """
    max_p99_ms: float                 # budget for one batch at `batch_size`
    batch_size: int = 1               # 1 = a real-time rig scoring one clip
    precision: str = "fp32"           # fp32 | fp16
    iters: int = 100
    warmup: int = 20
    # Deployment runtime the budget is measured against. Pick the one the rig
    # runs, not the one the sweep happens to train on.
    target: str = "auto"              # auto | openvino | tensorrt | torchscript


class TuningRequest(BaseModel):
    """A baseline training config plus the search space + sweep settings."""
    base: TrainingConfig
    n_trials: int = 20
    objective_metric: str = "val_mcc"  # val_mcc | val_loss | val_balanced_acc | val_acc
    # Search space — include only the hyperparameters to sweep.
    lr: Optional[RangeSpec] = None
    weight_decay: Optional[RangeSpec] = None
    dropout: Optional[RangeSpec] = None
    variant: Optional[CategoricalSpec] = None
    batch_size: Optional[CategoricalSpec] = None
    image_size: Optional[CategoricalSpec] = None
    temporal_pool: Optional[CategoricalSpec] = None
    tsm_stages: Optional[CategoricalSpec] = None
    frames_per_group: Optional[CategoricalSpec] = None
    # Optional constraint on the sampled architecture.
    latency: Optional[LatencyConstraint] = None


_FLOAT_PARAMS = ("lr", "weight_decay", "dropout")
_CATEGORICAL_PARAMS = (
    "variant", "batch_size", "image_size", "temporal_pool",
    "tsm_stages", "frames_per_group",
)


def _base_to_config(req: TrainingConfig) -> dict:
    """Mirror training router's TrainingConfig → config_tsm.yaml-shaped dict."""
    return {
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
        "data_loader": {
            "batch_size": req.batch_size,
            "num_workers": req.num_workers,
            "image_size": req.image_size,
        },
        "max_epochs": req.max_epochs,
        "callbacks": {"early_stopping_patience": req.early_stopping_patience},
        "boundary_smoothing": {
            "enabled": req.boundary_smoothing_enabled,
            "epsilon": req.boundary_smoothing_epsilon,
            "window": req.boundary_smoothing_window,
        },
        "use_wandb": False,
        "fine_tune_lr": False,
    }


def _build_search_space(req: TuningRequest) -> dict:
    """Collect the populated specs into one dict, tagged by kind so the service
    can dispatch to suggest_float / suggest_categorical without re-deriving types."""
    space = {}
    for name in _FLOAT_PARAMS:
        spec = getattr(req, name)
        if spec:
            space[name] = {"kind": "float", **spec.model_dump()}
    for name in _CATEGORICAL_PARAMS:
        spec = getattr(req, name)
        if spec:
            if not spec.choices:
                raise HTTPException(
                    status_code=422,
                    detail=f"'{name}' is enabled but has no choices to sweep.",
                )
            space[name] = {"kind": "categorical", "choices": list(spec.choices)}
    return space


@router.post("/start")
async def api_start_tuning(req: TuningRequest):
    """Start an Optuna hyperparameter sweep. Returns task_id."""
    search_space = _build_search_space(req)

    if not search_space:
        raise HTTPException(status_code=422, detail="Select at least one hyperparameter to sweep.")
    if req.n_trials < 1:
        raise HTTPException(status_code=422, detail="n_trials must be >= 1.")
    if req.latency:
        if req.latency.max_p99_ms <= 0:
            raise HTTPException(status_code=422, detail="max_p99_ms must be > 0.")
        # Fail here rather than after the first trial: a sweep that can't measure
        # its own constraint would spend GPU hours producing undeployable models.
        from optimized_runtime import TARGETS, RuntimeUnavailable, preflight_target
        if req.latency.target not in TARGETS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown latency target '{req.latency.target}'. Choose from {TARGETS}.")
        try:
            preflight_target(req.latency.target)
        except RuntimeUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))

    base_config = _base_to_config(req.base)
    try:
        task_id = start_tuning(
            base_config=base_config,
            search_space=search_space,
            n_trials=req.n_trials,
            objective_metric=req.objective_metric,
            latency=req.latency.model_dump() if req.latency else None,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_tuning_status(task_id: str):
    """SSE stream of trial events (start / epoch / trial_done / pruned / done)."""
    if get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        sent = 0
        while True:
            events = read_all_results(task_id)
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1
            status = (get_task(task_id) or {}).get("status")
            if status in ("completed", "error", "cancelled"):
                yield f"data: {json.dumps({'event': 'status', 'status': status})}\n\n"
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/stop/{task_id}")
async def api_stop_tuning(task_id: str):
    """Request cancellation of a running sweep."""
    if not stop_tuning(task_id):
        raise HTTPException(status_code=404, detail="Task not found or already finished")
    return {"status": "cancelling"}

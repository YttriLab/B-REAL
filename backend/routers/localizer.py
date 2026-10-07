"""Localizer router: the bundled default, SAM2 seed previews, and in-app
distillation of SAM2 into a localizer for a new rig (Train Localizer)."""

import asyncio
import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services import localizer_service as svc

router = APIRouter()

Sam2Model = Literal["tiny", "small", "base-plus", "large"]


class SeedPoint(BaseModel):
    """A click on the animal: the first frame of a clip SAM2 will track."""
    video_path: str
    frame: int = Field(ge=0)
    x: float
    y: float


class SeedPreviewRequest(BaseModel):
    video_path: str
    frame: int = Field(0, ge=0)
    # Omit both to just fetch the frame (to click on).
    x: Optional[float] = None
    y: Optional[float] = None
    sam2_model: Sam2Model = "tiny"


class TrainRequest(BaseModel):
    # Videos to cut clips_per_video evenly spaced clips from, each seeded by
    # the current localizer (init_weights) on its first confident frame.
    videos: list[str] = []
    clips_per_video: int = Field(6, ge=0, le=200)
    # Clips seeded by a click instead: one clip starts at each.
    seeds: list[SeedPoint] = []
    frames_per_clip: int = Field(500, ge=10, le=20000)
    # Track every Nth native frame: wider coverage per SAM2 frame.
    frame_step: int = Field(2, ge=1, le=100)
    sam2_model: Sam2Model = "tiny"
    # Empty-arena videos: real "no animal" frames for the presence head.
    negative_videos: list[str] = []
    negatives_per_video: int = Field(200, ge=0, le=20000)
    # "finetune" starts from init_weights; "imagenet" from scratch.
    init: Literal["finetune", "imagenet"] = "finetune"
    # The current localizer: seeds auto clips and is fine-tuned from.
    # None = the bundled weights.
    init_weights: Optional[str] = None
    steps: int = Field(4000, ge=1, le=500000)
    batch_size: int = Field(64, ge=2, le=1024)
    output_dir: str
    name: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._-]+$")
    # Per-clip SAM2 labels; default <output_dir>/sam2_labels.
    cache_dir: Optional[str] = None


@router.get("/info")
async def api_info():
    """The bundled weights' path and whether SAM2 can run here."""
    return svc.info()


@router.post("/seed-preview")
async def api_seed_preview(req: SeedPreviewRequest):
    if (req.x is None) != (req.y is None):
        raise HTTPException(status_code=422, detail="give both x and y, or neither")
    try:
        return await asyncio.to_thread(svc.seed_preview, req.video_path, req.frame, req.x, req.y,
                                       req.sam2_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/train")
async def api_train(req: TrainRequest):
    if not req.seeds and not (req.videos and req.clips_per_video > 0):
        raise HTTPException(status_code=422,
                            detail="nothing to label: give seeds, or videos with clips_per_video > 0")
    status = svc.info()
    if not status["sam2_available"]:
        raise HTTPException(status_code=503, detail=status["sam2_detail"])
    return {"task_id": svc.start_localizer_training(**req.model_dump())}


@router.get("/status/{task_id}")
async def api_status(task_id: str):
    """SSE: phase/progress/total, and every log line exactly once."""
    task = svc.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def stream():
        sent = 0
        while True:
            status = task["status"]  # before the log, so no final line is missed
            new = task["log"][sent:]
            sent += len(new)
            data = {
                "status": status, "phase": task["phase"],
                "progress": task["progress"], "total": task["total"],
                "message": task["message"], "error": task["error"],
                "log": new, "result": task["result"] if status == "completed" else None,
            }
            yield f"data: {json.dumps(data, default=str)}\n\n"
            if status in ("completed", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/stop/{task_id}")
async def api_stop(task_id: str):
    task = svc.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["cancel"] = True
    return {"status": "cancelling"}

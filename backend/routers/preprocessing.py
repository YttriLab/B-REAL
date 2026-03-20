"""Preprocessing router: background computation, frame preview, full video preprocessing."""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.preprocess_service import (
    compute_background,
    preview_frame,
    start_preprocessing,
    get_task,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class BackgroundRequest(BaseModel):
    video_path: str
    num_frames: int = 500
    output_dir: Optional[str] = None


class CropParams(BaseModel):
    box_size: int = 220
    difference_threshold: int = 5
    downsample_factor: int = 1
    open1_kernel: int = 4
    close_kernel: int = 15
    open2_kernel: int = 10


class PreviewRequest(BaseModel):
    video_path: str
    bg_path: str
    frame_index: int
    crop_params: CropParams = CropParams()


class RunRequest(BaseModel):
    video_path: str
    output_dir: str
    crop_params: CropParams = CropParams()
    desired_framerate: Optional[int] = None
    bg_path: Optional[str] = None
    num_frames_bg: int = 500
    output_name: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/background")
async def api_compute_background(req: BackgroundRequest):
    """Compute background average from video frames. Returns saved path."""
    try:
        path = compute_background(req.video_path, req.num_frames, req.output_dir)
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview")
async def api_preview_frame(req: PreviewRequest):
    """Return base64-encoded preview images for a single frame."""
    try:
        result = preview_frame(
            video_path=req.video_path,
            bg_path=req.bg_path,
            frame_index=req.frame_index,
            crop_params=req.crop_params.model_dump(),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def api_run_preprocessing(req: RunRequest):
    """Start full video preprocessing in background. Returns task_id."""
    try:
        task_id = start_preprocessing(
            video_path=req.video_path,
            output_dir=req.output_dir,
            crop_params=req.crop_params.model_dump(),
            desired_framerate=req.desired_framerate,
            bg_path=req.bg_path,
            num_frames_bg=req.num_frames_bg,
            output_name=req.output_name,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_preprocess_status(task_id: str):
    """SSE stream of preprocessing progress."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        import json
        while True:
            data = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "total": task["total"],
                "error": task["error"],
                "output_dir": task["output_dir"],
            })
            yield f"data: {data}\n\n"
            if task["status"] in ("completed", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/stop/{task_id}")
async def api_stop_preprocessing(task_id: str):
    """Cancel a running preprocessing task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["cancel"] = True
    return {"status": "cancelling"}

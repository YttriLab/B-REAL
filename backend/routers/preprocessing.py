"""Preprocessing router: background computation, frame preview, full video preprocessing."""

import asyncio
from typing import Literal, Optional

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
    # "directional" (subject darker than bg) or "absolute" (|frame - bg|).
    difference_mode: str = "directional"
    downsample_factor: int = 1
    open1_kernel: int = 4
    close_kernel: int = 15
    open2_kernel: int = 10
    # Full-frame pre-resize applied before cropping (1.0 = original size).
    resize_factor: float = 1.0
    # Temporal stabilization of the crop center (see FindRatBoundingBox).
    smoothing: float = 0.0          # EMA weight on previous center, 0..1
    deadband: float = 0.0           # px (small space) before the box moves
    min_contour_area: float = 0.0   # ignore blobs smaller than this
    # How the animal is found (backend/services/cropping.py): background
    # subtraction, the SAM2-distilled localizer network, or no crop at all.
    method: Literal["background", "localizer", "none"] = "background"
    # Localizer weights (.pt state dict). None = the bundled default
    # (localizer.DEFAULT_WEIGHTS), or one trained in-app (/api/localizer/train).
    localizer_path: Optional[str] = None
    # Below this localizer presence probability the box holds still.
    presence_threshold: float = 0.1
    # How the localizer runs: "auto" = a TensorRT engine (built + cached on
    # first use) when the GPU supports it, else PyTorch. Part of the crop
    # settings so preprocessing and inference locate identically.
    localizer_runtime: Literal["auto", "tensorrt", "torch"] = "torch"
    localizer_precision: Literal["fp16", "fp32"] = "fp16"


Decoder = Literal["auto", "nvdec", "cpu"]


class PreviewRequest(BaseModel):
    video_path: str
    # Required for method="background"; unused by the localizer.
    bg_path: Optional[str] = None
    frame_index: int
    crop_params: CropParams = CropParams()
    # Smoothed center [cx, cy] (small space) carried from the previous previewed
    # frame, so stepping frame-by-frame shows the temporal smoothing in action.
    prev_center: Optional[list[float]] = None


class RunRequest(BaseModel):
    video_path: str
    output_dir: str
    crop_params: CropParams = CropParams()
    # Inclusive [start, end] native-index ranges to extract. None = all frames.
    frame_ranges: Optional[list[list[int]]] = None
    bg_path: Optional[str] = None
    num_frames_bg: int = 500
    output_name: Optional[str] = None
    # False writes whole frames and skips background subtraction entirely — for
    # rigs with no static background, or when the animal already fills the
    # frame. crop_params.resize_factor still applies. Same as method="none".
    crop: bool = True
    # "auto" = NVDEC when it reproduces cv2 bit-for-bit on this video, else cv2.
    decoder: Decoder = "auto"


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
    if req.crop_params.method == "background" and not req.bg_path:
        raise HTTPException(status_code=422, detail="bg_path is required for background cropping")
    try:
        result = preview_frame(
            video_path=req.video_path,
            bg_path=req.bg_path,
            frame_index=req.frame_index,
            crop_params=req.crop_params.model_dump(),
            prev_center=req.prev_center,
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
            frame_ranges=req.frame_ranges,
            bg_path=req.bg_path,
            num_frames_bg=req.num_frames_bg,
            output_name=req.output_name,
            crop=req.crop,
            decoder=req.decoder,
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
                "message": task.get("message"),
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

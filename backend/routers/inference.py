"""Inference router: run per-frame inference, stream progress, return results."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.inference_service import (
    start_inference,
    get_task,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    checkpoint_path: str
    frame_dir: str
    frames_per_group: int = 3
    image_size: int = 224
    extra_model_config: Optional[dict] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/run")
async def api_run_inference(req: InferenceRequest):
    """Start inference in background. Returns task_id."""
    try:
        task_id = start_inference(
            checkpoint_path=req.checkpoint_path,
            frame_dir=req.frame_dir,
            frames_per_group=req.frames_per_group,
            image_size=req.image_size,
            model_config=req.extra_model_config,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_inference_status(task_id: str):
    """SSE stream of inference progress."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        while True:
            data = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "total": task["total"],
                "error": task["error"],
            })
            yield f"data: {data}\n\n"
            if task["status"] in ("completed", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/results/{task_id}")
async def api_inference_results(task_id: str):
    """Return inference results (per-frame predictions + uncertainty)."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Inference not complete (status: {task['status']})"
        )

    results = task.get("results")
    if results is None:
        raise HTTPException(status_code=500, detail="No results available")

    return results


@router.post("/stop/{task_id}")
async def api_stop_inference(task_id: str):
    """Cancel a running inference task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["cancel"] = True
    return {"status": "cancelling"}

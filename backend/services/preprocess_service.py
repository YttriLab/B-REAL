"""
Preprocessing service: wraps tune_cropping.py for background computation,
frame preview, and full video-to-flat-frames extraction with progress.
"""

import base64
import os
import threading
import uuid
from typing import Callable, Optional

import cv2
import numpy as np

from tune_cropping import FindRatBoundingBox, get_background_avg

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


# ── Background computation ─────────────────────────────────────────────────

def compute_background(video_path: str, num_frames: int = 500, output_dir: str | None = None) -> str:
    """Compute background average and save as PNG.  Returns saved path."""
    bg = get_background_avg(video_path, num_frames)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), "backgrounds")
    os.makedirs(output_dir, exist_ok=True)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"bg_{vid_name}.png")
    cv2.imwrite(out_path, bg.astype(np.uint8))
    return out_path


# ── Single-frame preview ──────────────────────────────────────────────────

def _encode_image(img: np.ndarray, quality: int = 85) -> str:
    """Encode an image as base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf).decode("ascii")


def preview_frame(
    video_path: str,
    bg_path: str,
    frame_index: int,
    crop_params: dict,
) -> dict:
    """
    Return base64-encoded preview images for a single frame.

    Returns dict with keys: original_bbox, mask, cropped
    """
    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    cropper = FindRatBoundingBox(bg, **crop_params)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    (x, y, w, h), _small, mask, cropped = cropper(frame)

    # Draw bounding box on original frame
    s = crop_params.get("downsample_factor", 1)
    frame_bbox = frame.copy()
    cv2.rectangle(frame_bbox, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 255, 0), 2)

    # Convert mask to 3-channel for consistent encoding
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return {
        "original_bbox": _encode_image(frame_bbox),
        "mask": _encode_image(mask_bgr),
        "cropped": _encode_image(cropped),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
    }


# ── Full video preprocessing (flat frame output) ──────────────────────────

def crop_video_flat(
    video_path: str,
    output_dir: str,
    crop_params: dict,
    desired_framerate: int | None = None,
    bg_path: str | None = None,
    num_frames_bg: int = 500,
    output_name: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> str:
    """
    Extract cropped frames from a video, saving each as a flat PNG.

    Output: output_dir/video_name/0.png, 1.png, 2.png, ...

    Returns the output subdirectory path.
    """
    vid_name = output_name or os.path.splitext(os.path.basename(video_path))[0]
    out_sub_dir = os.path.join(output_dir, vid_name)
    os.makedirs(out_sub_dir, exist_ok=True)

    # Background
    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        bg = get_background_avg(video_path, num_frames_bg)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps:
        raise ValueError("Cannot read framerate from video")

    # Framerate downsampling
    if desired_framerate is not None:
        if fps % desired_framerate == 0:
            skip_size = int(fps // desired_framerate)
        else:
            raise ValueError(
                f"Video framerate ({fps}) not an integer multiple of desired framerate ({desired_framerate})"
            )
    else:
        skip_size = 1

    estimated_output = total_frames // skip_size
    cropper = FindRatBoundingBox(bg, **crop_params)

    frame_counter = 0
    processed_counter = 0

    while True:
        if cancel_flag and cancel_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % skip_size != 0:
            frame_counter += 1
            continue

        *_, cropped = cropper(frame)
        out_path = os.path.join(out_sub_dir, f"{processed_counter}.png")
        cv2.imwrite(out_path, cropped)

        frame_counter += 1
        processed_counter += 1

        if on_progress:
            on_progress(processed_counter, estimated_output)

    cap.release()
    return out_sub_dir


def start_preprocessing(
    video_path: str,
    output_dir: str,
    crop_params: dict,
    desired_framerate: int | None = None,
    bg_path: str | None = None,
    num_frames_bg: int = 500,
    output_name: str | None = None,
) -> str:
    """Launch preprocessing in a background thread. Returns task_id."""
    task_id = str(uuid.uuid4())
    task = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "cancel": False,
        "error": None,
        "output_dir": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            def on_progress(done, total):
                task["progress"] = done
                task["total"] = total

            result_dir = crop_video_flat(
                video_path=video_path,
                output_dir=output_dir,
                crop_params=crop_params,
                desired_framerate=desired_framerate,
                bg_path=bg_path,
                num_frames_bg=num_frames_bg,
                output_name=output_name,
                on_progress=on_progress,
                cancel_flag=lambda: task["cancel"],
            )
            if task["cancel"]:
                task["status"] = "cancelled"
            else:
                task["status"] = "completed"
                task["output_dir"] = result_dir
        except Exception as e:
            task["status"] = "error"
            task["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return task_id

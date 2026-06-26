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

from tune_cropping import FindRatBoundingBox, get_background_avg, pop_resize_factor, resize_frame

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
    prev_center: list | None = None,
) -> dict:
    """
    Return base64-encoded preview images for a single frame.

    Returns dict with keys: original_bbox, mask, cropped, frame size, and the
    resulting smoothed `center` (small space) so the caller can chain it into the
    next frame's preview to visualize temporal smoothing while stepping.
    """
    crop_params, resize_factor = pop_resize_factor(crop_params)

    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = resize_frame(bg, resize_factor)
    cropper = FindRatBoundingBox(bg, **crop_params)
    # Seed the smoothing state from the previous frame so stepping shows the
    # stabilized track (a fresh cropper otherwise has no temporal history).
    if prev_center and len(prev_center) == 2:
        cropper.smoothed_center = (float(prev_center[0]), float(prev_center[1]))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    orig_w, orig_h = frame.shape[1], frame.shape[0]
    if resize_factor != 1.0:
        # Resize the whole frame to exactly the (resized) background dims so the
        # detection/crop pipeline operates on the smaller image.
        frame = cv2.resize(frame, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)

    (x, y, w, h), _small, mask, cropped = cropper(frame)

    # Draw bounding box on the (resized) frame that was actually processed.
    s = crop_params.get("downsample_factor", 1)
    frame_bbox = frame.copy()
    cv2.rectangle(frame_bbox, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 255, 0), 2)

    # Convert mask to 3-channel for consistent encoding
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return {
        "original_bbox": _encode_image(frame_bbox),
        "mask": _encode_image(mask_bgr),
        "cropped": _encode_image(cropped),
        "frame_width": orig_w,           # source video frame size
        "frame_height": orig_h,
        "resized_width": frame.shape[1],  # size actually processed/saved
        "resized_height": frame.shape[0],
        "resize_factor": resize_factor,
        # Smoothed crop center (small space) — chain into the next preview frame.
        "center": list(cropper.smoothed_center) if cropper.smoothed_center else None,
    }


# ── Full video preprocessing (flat frame output) ──────────────────────────

def _wanted_set(frame_ranges, total_frames):
    """Build the set of native frame indices to extract from inclusive ranges.

    Returns None when no ranges are given (meaning: extract every frame).
    """
    if not frame_ranges:
        return None
    wanted = set()
    for r in frame_ranges:
        start, end = int(r[0]), int(r[1])
        if end < start:
            start, end = end, start
        start = max(0, start)
        for i in range(start, end + 1):
            wanted.add(i)
    return wanted


def crop_video_flat(
    video_path: str,
    output_dir: str,
    crop_params: dict,
    frame_ranges: list | None = None,
    bg_path: str | None = None,
    num_frames_bg: int = 500,
    output_name: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> str:
    """
    Extract cropped frames from a video, each saved as a flat PNG named by its
    native (0-based) video frame index: output_dir/video_name/{frame}.png.

    If ``frame_ranges`` (a list of inclusive [start, end] native-index pairs) is
    given, only those frames are written — the label-aware "sparse" extraction
    used for training. If omitted, every frame is extracted (back-compat).

    Naming by native frame index keeps the extracted frames aligned with the
    label CSV ``time`` column and the app's timeline.

    Returns the output subdirectory path.
    """
    vid_name = output_name or os.path.splitext(os.path.basename(video_path))[0]
    out_sub_dir = os.path.join(output_dir, vid_name)
    os.makedirs(out_sub_dir, exist_ok=True)

    crop_params, resize_factor = pop_resize_factor(crop_params)

    # Background
    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        bg = get_background_avg(video_path, num_frames_bg)
    bg = resize_frame(bg, resize_factor)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    wanted = _wanted_set(frame_ranges, total_frames)
    target_total = len(wanted) if wanted is not None else total_frames

    cropper = FindRatBoundingBox(bg, **crop_params)

    frame_index = 0
    written = 0

    while True:
        if cancel_flag and cancel_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        if wanted is None or frame_index in wanted:
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
            *_, cropped = cropper(frame)
            out_path = os.path.join(out_sub_dir, f"{frame_index}.png")
            cv2.imwrite(out_path, cropped)
            written += 1
            if on_progress:
                on_progress(written, target_total)

        frame_index += 1

    cap.release()
    return out_sub_dir


def start_preprocessing(
    video_path: str,
    output_dir: str,
    crop_params: dict,
    frame_ranges: list | None = None,
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
                frame_ranges=frame_ranges,
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

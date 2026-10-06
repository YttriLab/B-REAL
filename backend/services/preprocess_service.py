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

from tune_cropping import FindRatBoundingBox, get_background_avg, resize_frame

from backend.services.cropping import FrameCropper, iter_wanted, split_crop_params

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


def _read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _box_thickness(frame: np.ndarray) -> int:
    # The preview is displayed small (~200px), so a fixed 2px line would shrink
    # to near-invisible on big frames.
    return max(2, round(max(frame.shape[0], frame.shape[1]) / 100))


def _preview_localizer(video_path, frame_index, crop_params, prev_center) -> dict:
    """Localizer-mode preview: the box, the network's heatmap, and the crop."""
    frame = _read_frame(video_path, frame_index)
    cropper = FrameCropper(crop_params)
    try:
        if prev_center and len(prev_center) == 2:
            cropper._loc.smoothed_center = (float(prev_center[0]), float(prev_center[1]))
        centers, pres, heat = cropper._loc.locate([frame], return_heatmap=True)
        resized = cropper._resize(frame)
        cropped, (x, y, w, h) = cropper._loc.crop(resized, centers[0], pres[0], frame.shape)
        center = cropper._loc.smoothed_center
    finally:
        cropper.close()

    frame_bbox = resized.copy() if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame_bbox, (x, y), (x + w, y + h), (0, 255, 0), _box_thickness(frame_bbox))
    # Heatmap as probabilities, stretched for display and sized like the frame.
    p = np.exp(heat[0] - heat[0].max())
    p = (255 * p / p.max()).astype(np.uint8)
    p = cv2.resize(p, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_LINEAR)
    return {
        "original_bbox": _encode_image(frame_bbox),
        "mask": _encode_image(cv2.applyColorMap(p, cv2.COLORMAP_INFERNO)),
        "cropped": _encode_image(cropped),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "resized_width": resized.shape[1],
        "resized_height": resized.shape[0],
        "resize_factor": cropper.resize_factor,
        "center": list(center) if center else None,
        "presence": float(pres[0]),
    }


def preview_frame(
    video_path: str,
    bg_path: str | None,
    frame_index: int,
    crop_params: dict,
    prev_center: list | None = None,
) -> dict:
    """
    Return base64-encoded preview images for a single frame.

    Returns dict with keys: original_bbox, mask, cropped, frame size, and the
    resulting smoothed `center` (small space) so the caller can chain it into the
    next frame's preview to visualize temporal smoothing while stepping.

    In localizer mode (``crop_params["method"] == "localizer"``) no background
    is used; ``mask`` is the localizer's heatmap and ``presence`` its
    animal-present probability.
    """
    method = split_crop_params(crop_params)[0]
    if method == "localizer":
        return _preview_localizer(video_path, frame_index, crop_params, prev_center)
    if not bg_path:
        raise ValueError("background crop preview needs a bg_path")
    _method, resize_factor, _extras, crop_params = split_crop_params(crop_params)

    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = resize_frame(bg, resize_factor)
    cropper = FindRatBoundingBox(bg, **crop_params)
    # Seed the smoothing state from the previous frame so stepping shows the
    # stabilized track (a fresh cropper otherwise has no temporal history).
    if prev_center and len(prev_center) == 2:
        cropper.smoothed_center = (float(prev_center[0]), float(prev_center[1]))

    frame = _read_frame(video_path, frame_index)

    orig_w, orig_h = frame.shape[1], frame.shape[0]
    if resize_factor != 1.0:
        # Resize the whole frame to exactly the (resized) background dims so the
        # detection/crop pipeline operates on the smaller image.
        frame = cv2.resize(frame, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)

    (x, y, w, h), _small, mask, cropped = cropper(frame)

    # Draw bounding box on the (resized) frame that was actually processed.
    s = crop_params.get("downsample_factor", 1)
    frame_bbox = frame.copy()
    cv2.rectangle(frame_bbox, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 255, 0),
                  _box_thickness(frame))

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
    crop: bool = True,
    decoder: str = "auto",
    batch_size: int = 32,
) -> str:
    """
    Extract frames from a video, each saved as a flat PNG named by its
    native (0-based) video frame index: output_dir/video_name/{frame}.png.

    ``crop_params["method"]`` picks the crop (see backend/services/cropping.py):
    "background" (default) subtracts a background and crops around the animal,
    "localizer" finds it with the SAM2-distilled network, "none" writes the
    whole frame — no background is read or computed at all, which is the only
    thing that works when there is no static background to subtract (a moving
    camera, or a rig where the animal fills the frame). ``crop=False`` is the
    legacy spelling of "none". ``resize_factor`` applies in every mode.

    ``decoder`` is "auto" (NVDEC when it reproduces cv2 bit-for-bit on this
    video, else cv2), "nvdec" or "cpu"; see video_decode.py.

    If ``frame_ranges`` (a list of inclusive [start, end] native-index pairs) is
    given, only those frames are written — the label-aware "sparse" extraction
    used for training. If omitted, every frame is extracted (back-compat).
    Crop smoothing restarts at each gap between ranges, as it does at inference.

    Naming by native frame index keeps the extracted frames aligned with the
    label CSV ``time`` column and the app's timeline.

    Returns the output subdirectory path.
    """
    import video_decode

    vid_name = output_name or os.path.splitext(os.path.basename(video_path))[0]
    out_sub_dir = os.path.join(output_dir, vid_name)
    os.makedirs(out_sub_dir, exist_ok=True)

    cropper = FrameCropper(crop_params, video_path=video_path, bg_path=bg_path,
                           num_frames_bg=num_frames_bg, crop=crop)
    dec, reason = video_decode.resolve_decoder(decoder, video_path)
    print(f"[preprocess] decoder: {reason}", flush=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    wanted = _wanted_set(frame_ranges, total_frames)
    target_total = len(wanted) if wanted is not None else total_frames
    # Stop decoding after the last wanted frame instead of reading to the end.
    stop_after = (max(wanted) + 1) if wanted else None

    written = 0

    def cancelled():
        return bool(cancel_flag and cancel_flag())

    try:
        batches = video_decode.iter_batches(video_path, dec, batch_size=batch_size,
                                            max_frames=stop_after)
        is_wanted = (lambda i: True) if wanted is None else wanted.__contains__
        for idxs, frames, resets, prepared in iter_wanted(batches, is_wanted, cropper):
            if cancelled():
                break
            for idx, out in zip(idxs, cropper.crop_batch(frames, resets, prepared)):
                if cancelled():
                    break
                cv2.imwrite(os.path.join(out_sub_dir, f"{idx}.png"), out)
                written += 1
                if on_progress:
                    on_progress(written, target_total)
    finally:
        cropper.close()
    return out_sub_dir


def start_preprocessing(
    video_path: str,
    output_dir: str,
    crop_params: dict,
    frame_ranges: list | None = None,
    bg_path: str | None = None,
    num_frames_bg: int = 500,
    output_name: str | None = None,
    crop: bool = True,
    decoder: str = "auto",
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
                crop=crop,
                decoder=decoder,
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

"""
Inference service: loads a trained TSMClassifier checkpoint and runs per-frame
inference on preprocessed flat frame directories. Computes normalized entropy
for active learning uncertainty estimation.
"""

import math
import os
import threading
import uuid
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


# ── Inference logic ────────────────────────────────────────────────────────

def _build_val_transform(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ], additional_targets={f"image{i}": "image" for i in range(1, 10)})


def _load_frame(path: str) -> np.ndarray:
    """Load a single frame as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_inference(
    checkpoint_path: str,
    frame_dir: str,
    frames_per_group: int = 3,
    image_size: int = 224,
    model_config: dict | None = None,
    on_progress: callable = None,
    cancel_flag: callable = None,
) -> dict:
    """
    Run per-frame inference on a flat frame directory.

    Args:
        checkpoint_path: Path to .ckpt file
        frame_dir: Directory containing 0.png, 1.png, ... (flat)
        frames_per_group: Number of frames per temporal group (default 3)
        image_size: Resize dimension
        model_config: Override model hyperparameters (tsm_stages, etc.)

    Returns:
        dict with 'frames' list and 'class_names'
    """
    from model_tsm import TSMClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    load_kwargs = model_config or {}
    model = TSMClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False, **load_kwargs
    )
    model.to(device)
    model.eval()

    num_classes = model.num_classes
    class_names = model.class_names

    # Discover frames
    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    total_frames = len(frame_files)

    # Build transform
    transform = _build_val_transform(image_size)

    results = []
    T = frames_per_group

    with torch.no_grad():
        for idx in range(T - 1, total_frames):
            if cancel_flag and cancel_flag():
                break

            # Construct triplet: [idx-(T-1), ..., idx-1, idx]
            triplet_files = [frame_files[idx - (T - 1 - j)] for j in range(T)]
            triplet_paths = [os.path.join(frame_dir, f) for f in triplet_files]

            # Load frames
            images = [_load_frame(p) for p in triplet_paths]

            # Apply transforms consistently across the group
            targets = {"image": images[0]}
            for i in range(1, len(images)):
                targets[f"image{i}"] = images[i]
            transformed = transform(**targets)

            tensors = [transformed["image"]]
            for i in range(1, len(images)):
                tensors.append(transformed[f"image{i}"])

            # Stack into (1, T, C, H, W)
            triplet_tensor = torch.stack(tensors).unsqueeze(0).to(device)

            # Forward pass
            logits = model(triplet_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)  # (K,)

            # Entropy-based uncertainty
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            max_entropy = math.log(num_classes)
            normalized_entropy = (entropy / max_entropy).item() if max_entropy > 0 else 0.0

            predicted_class = probs.argmax().item()

            results.append({
                "index": idx,
                "predictions": probs.cpu().tolist(),
                "uncertainty": normalized_entropy,
                "predicted_class": predicted_class,
            })

            if on_progress:
                on_progress(len(results), total_frames - (T - 1))

    return {
        "frames": results,
        "class_names": class_names,
        "total_frames": total_frames,
        "frames_per_group": T,
    }


def _wanted_set(frame_ranges):
    """Build the set of native frame indices to predict from inclusive ranges.

    ``frame_ranges`` is a list of ``[start, end]`` pairs (inclusive, native frame
    indices). Returns ``None`` when no ranges are given (meaning: predict on every
    frame). Mirrors ``preprocess_service._wanted_set``.
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


def run_inference_video(
    checkpoint_path: str,
    video_path: str,
    bg_path: str | None,
    crop_params: dict,
    frames_per_group: int = 3,
    image_size: int = 224,
    num_frames_bg: int = 500,
    model_config: dict | None = None,
    frame_ranges: list | None = None,
    on_progress: callable = None,
    cancel_flag: callable = None,
) -> dict:
    """
    Run per-frame inference by decoding + cropping the video on the fly — no
    pre-extracted frame directory. Frames are cropped with the SAME pipeline as
    training (FindRatBoundingBox + background), so the model sees the same
    distribution. Results are indexed by native frame number, matching the
    label/timeline index space.

    If ``frame_ranges`` (inclusive [start, end] native-index pairs) is given,
    predictions are emitted only for those frames — used by Active Learning to
    scope inference to a segment or to the already-labeled frames. Decoding still
    starts at frame 0 to keep native-index alignment, but the expensive crop +
    model forward are skipped for frames outside the clip lookback of any wanted
    frame, and decoding stops once the last wanted frame is passed.
    """
    from model_tsm import TSMClassifier
    from tune_cropping import FindRatBoundingBox, get_background_avg, pop_resize_factor, resize_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs = model_config or {}
    model = TSMClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False, **load_kwargs
    )
    model.to(device)
    model.eval()

    num_classes = model.num_classes
    class_names = model.class_names

    bg = None
    if bg_path:
        loaded = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        if loaded is not None:
            bg = loaded.astype(np.float32)
        else:
            # cv2.imread silently returns None for a missing/unreadable file;
            # fall back to computing the background from the video itself rather
            # than crashing with "'NoneType' object has no attribute 'astype'".
            print(f"[inference] background not readable at {bg_path}; "
                  f"computing it from the video instead.", flush=True)
    if bg is None:
        bg = get_background_avg(video_path, num_frames_bg)
    # Apply the same full-frame pre-resize used in training so crops match.
    crop_params, resize_factor = pop_resize_factor(crop_params)
    bg = resize_frame(bg, resize_factor)
    cropper = FindRatBoundingBox(bg, **crop_params)

    transform = _build_val_transform(image_size)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    T = frames_per_group

    # Optional scoping: only emit predictions for frames in `wanted`.
    wanted = _wanted_set(frame_ranges)
    wanted_max = max(wanted) if wanted else None

    def _is_needed(i: int) -> bool:
        # Frame i must be cropped if it falls within the T-frame clip of any
        # wanted frame (i.e. one of i, i+1, ..., i+T-1 is wanted).
        if wanted is None:
            return True
        return any((i + k) in wanted for k in range(T))

    if wanted is None:
        target_total = max(0, total_frames - (T - 1))
    else:
        # Wanted frames before index T-1 can't form a full clip, so they're not emitted.
        target_total = sum(1 for i in wanted if i >= T - 1)

    window = []  # rolling window of the last T cropped RGB frames
    results = []
    frame_index = 0

    with torch.no_grad():
        while True:
            if cancel_flag and cancel_flag():
                break
            ret, frame = cap.read()
            if not ret:
                break
            if wanted is not None and frame_index > wanted_max:
                break  # past the last wanted frame — nothing more to predict

            if _is_needed(frame_index):
                if resize_factor != 1.0:
                    frame = cv2.resize(frame, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
                # Crop, then match _load_frame's BGR->RGB conversion exactly.
                *_, cropped = cropper(frame)
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                window.append(rgb)
                if len(window) > T:
                    window.pop(0)
            else:
                # Gap before the next wanted frame: restart the clip and the
                # crop-center smoothing so the next region begins cleanly.
                window = []
                cropper.smoothed_center = None

            emit = wanted is None or frame_index in wanted
            if emit and len(window) == T:
                targets = {"image": window[0]}
                for i in range(1, T):
                    targets[f"image{i}"] = window[i]
                transformed = transform(**targets)
                tensors = [transformed["image"]]
                for i in range(1, T):
                    tensors.append(transformed[f"image{i}"])
                clip = torch.stack(tensors).unsqueeze(0).to(device)

                logits = model(clip)
                probs = F.softmax(logits, dim=1).squeeze(0)
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                max_entropy = math.log(num_classes)
                normalized_entropy = (entropy / max_entropy).item() if max_entropy > 0 else 0.0

                results.append({
                    "index": frame_index,
                    "predictions": probs.cpu().tolist(),
                    "uncertainty": normalized_entropy,
                    "predicted_class": probs.argmax().item(),
                })
                if on_progress:
                    on_progress(len(results), target_total)

            frame_index += 1

    cap.release()
    return {
        "frames": results,
        "class_names": class_names,
        "total_frames": frame_index,
        "frames_per_group": T,
    }


# ── Async task management ─────────────────────────────────────────────────

def start_inference(
    checkpoint_path: str,
    frame_dir: str | None = None,
    frames_per_group: int = 3,
    image_size: int = 224,
    model_config: dict | None = None,
    video_path: str | None = None,
    bg_path: str | None = None,
    crop_params: dict | None = None,
    frame_ranges: list | None = None,
) -> str:
    """
    Launch inference in a background thread. Returns task_id.

    If ``video_path`` is given, frames are decoded + cropped on the fly
    (Active Learning). Otherwise it reads a pre-extracted ``frame_dir``.

    ``frame_ranges`` (video path only) scopes predictions to inclusive
    [start, end] native-index pairs — a segment or the labeled frames.
    """
    task_id = str(uuid.uuid4())
    task = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "cancel": False,
        "error": None,
        "results": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            def on_progress(done, total):
                task["progress"] = done
                task["total"] = total

            if video_path:
                result = run_inference_video(
                    checkpoint_path=checkpoint_path,
                    video_path=video_path,
                    bg_path=bg_path,
                    crop_params=crop_params or {},
                    frames_per_group=frames_per_group,
                    image_size=image_size,
                    model_config=model_config,
                    frame_ranges=frame_ranges,
                    on_progress=on_progress,
                    cancel_flag=lambda: task["cancel"],
                )
            else:
                result = run_inference(
                    checkpoint_path=checkpoint_path,
                    frame_dir=frame_dir,
                    frames_per_group=frames_per_group,
                    image_size=image_size,
                    model_config=model_config,
                    on_progress=on_progress,
                    cancel_flag=lambda: task["cancel"],
                )

            if task["cancel"]:
                task["status"] = "cancelled"
            else:
                task["status"] = "completed"
                task["results"] = result
        except Exception as e:
            task["status"] = "error"
            task["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return task_id

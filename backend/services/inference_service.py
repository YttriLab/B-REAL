"""
Inference service: loads a trained TSMClassifier checkpoint and runs per-frame
inference, either on a video (decoded + cropped on the fly, see
run_inference_video) or on a preprocessed flat frame directory. Computes
normalized entropy for active learning uncertainty estimation.
"""

import math
import os
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
    decoder: str = "auto",
    batch_size: int = 32,
    crop: bool = True,
    runtime: str = "torch",
    precision: str = "fp16",
    on_status: callable = None,
) -> dict:
    """
    Run per-frame inference by decoding + cropping the video on the fly — no
    pre-extracted frame directory. Frames are cropped with the SAME code as
    preprocessing (backend/services/cropping.py: background subtraction, the
    SAM2-distilled localizer, or no crop, per ``crop_params["method"]``), so
    the model sees the distribution it was trained on. Results are indexed by
    native frame number, matching the label/timeline index space.

    Throughput: decoding runs ahead on a background thread (NVDEC when
    ``decoder`` resolves to it, see video_decode.py), the localizer runs in
    batches, each frame is transformed once and reused by the T clips that
    contain it, and clips go through the model ``batch_size`` at a time.

    ``runtime`` "tensorrt" (or "auto", when the GPU supports it) runs the
    classifier as a TensorRT engine at ``precision``, built from the
    checkpoint and cached on first use (trt_engines.py). The localizer's
    runtime is a crop setting (``crop_params["localizer_runtime"]``), so
    preprocessing locates with the same engine.

    If ``frame_ranges`` (inclusive [start, end] native-index pairs) is given,
    predictions are emitted only for those frames — used by Active Learning to
    scope inference to a segment or to the already-labeled frames. Decoding still
    starts at frame 0 to keep native-index alignment, but the expensive crop +
    model forward are skipped for frames outside the clip lookback of any wanted
    frame, and decoding stops once the last wanted frame is passed.
    """
    import video_decode
    from model_tsm import TSMClassifier
    from backend.services.cropping import FrameCropper, iter_wanted, to_rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = frames_per_group
    batch_size = max(1, int(batch_size))

    rt, rt_reason = "torch", "PyTorch (requested)"
    if runtime != "torch":
        import trt_engines
        rt, rt_reason = trt_engines.resolve_runtime(runtime)

    load_kwargs = model_config or {}
    model = TSMClassifier.load_from_checkpoint(
        checkpoint_path, map_location="cpu" if rt == "tensorrt" else device,
        strict=False, **load_kwargs
    )
    model.eval()
    classify = model
    if rt == "tensorrt":
        try:
            classify = trt_engines.tsm_engine(
                model, checkpoint_path, T, image_size, precision=precision,
                max_batch=batch_size, model_config=load_kwargs, on_status=on_status)
            rt_reason = f"TensorRT {precision} ({os.path.basename(classify.path)})"
            device = "cuda"
        except Exception as e:
            if runtime == "tensorrt":
                raise
            rt_reason = f"PyTorch (TensorRT engine build failed: {type(e).__name__}: {e})"
    if classify is model:
        model.to(device)

    num_classes = model.num_classes
    class_names = model.class_names
    max_entropy = math.log(num_classes)

    cropper = FrameCropper(crop_params, video_path=video_path, bg_path=bg_path,
                           num_frames_bg=num_frames_bg, crop=crop, on_status=on_status)
    dec, reason = video_decode.resolve_decoder(decoder, video_path)
    print(f"[inference] crop: {cropper.method}; decoder: {reason}; classifier: {rt_reason}",
          flush=True)
    if on_status:
        engine = f"TensorRT {precision}" if classify is not model else "PyTorch"
        on_status(f"Running inference ({cropper.method} crop, {dec} decode, {engine})…")

    transform = _build_val_transform(image_size)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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

    window = deque(maxlen=T)  # transformed tensors of the last T contiguous needed frames
    pending_idx, pending_clips = [], []
    results = []

    def flush():
        if not pending_clips:
            return
        clips = torch.stack(pending_clips).to(device)
        with torch.no_grad():
            probs = F.softmax(classify(clips).float(), dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
        norm = (entropy / max_entropy) if max_entropy > 0 else torch.zeros_like(entropy)
        for idx, p, u, c in zip(pending_idx, probs.cpu().tolist(), norm.cpu().tolist(),
                                probs.argmax(1).cpu().tolist()):
            results.append({"index": idx, "predictions": p, "uncertainty": u,
                            "predicted_class": c})
        pending_idx.clear()
        pending_clips.clear()
        if on_progress:
            on_progress(len(results), target_total)

    def to_tensor(c):
        return transform(image=to_rgb(c))["image"]

    stop_after = (wanted_max + 1) if wanted else None
    decoded = 0  # frames actually read (CAP_PROP_FRAME_COUNT is only an estimate)

    def counted(batches):
        nonlocal decoded
        for b in batches:
            decoded += len(b)
            yield b

    pool = ThreadPoolExecutor(4)
    try:
        batches = counted(video_decode.iter_batches(video_path, dec, batch_size=32,
                                                    max_frames=stop_after))
        # A gap (reset) restarts the clip and the crop-centre smoothing so the
        # next region begins cleanly.
        for idxs, frames, resets, prepared in iter_wanted(batches, _is_needed, cropper):
            if cancel_flag and cancel_flag():
                break
            tensors = pool.map(to_tensor, cropper.crop_batch(frames, resets, prepared))
            for idx, reset, t in zip(idxs, resets, tensors):
                if reset:
                    window.clear()
                window.append(t)
                if len(window) == T and (wanted is None or idx in wanted):
                    pending_idx.append(idx)
                    pending_clips.append(torch.stack(tuple(window)))
                    if len(pending_clips) >= batch_size:
                        flush()
        flush()
    finally:
        pool.shutdown(wait=False)
        cropper.close()

    return {
        "frames": results,
        "class_names": class_names,
        "total_frames": decoded,
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
    decoder: str = "auto",
    batch_size: int = 32,
    runtime: str = "torch",
    precision: str = "fp16",
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
        "message": None,
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
                    decoder=decoder,
                    batch_size=batch_size,
                    runtime=runtime,
                    precision=precision,
                    on_status=lambda m: task.__setitem__("message", m),
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

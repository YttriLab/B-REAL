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


# ── Async task management ─────────────────────────────────────────────────

def start_inference(
    checkpoint_path: str,
    frame_dir: str,
    frames_per_group: int = 3,
    image_size: int = 224,
    model_config: dict | None = None,
) -> str:
    """Launch inference in a background thread. Returns task_id."""
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

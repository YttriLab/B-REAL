"""Pure config-building for the pipeline UI (testable without Qt).

Mirrors the DOM->config marshalling in plugins/pipeline.js: assembles the flat
dicts the backend endpoints expect (crop params, training config). Keeping this
separate from the widgets lets it be unit-tested against the backend contract.
"""

from __future__ import annotations

import os
from typing import Optional

# Architecture choices, shared by the Train, Tune and benchmark UIs so they can't
# drift apart. b4-b7 are supported by model_tsm.build_tsm_efficientnet but are
# ImageNet-trained at higher resolutions (see VARIANT_NATIVE_SIZE) and need a
# smaller batch to fit in VRAM.
VARIANTS = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
VARIANT_NATIVE_SIZE = {"b0": 224, "b1": 240, "b2": 260, "b3": 300,
                       "b4": 380, "b5": 456, "b6": 528, "b7": 600}
POOLS = ["last", "mean", "max"]

# Sweep choices.
OBJECTIVES = ["val_mcc", "val_balanced_acc", "val_acc", "val_loss"]
BATCH_CHOICES = [8, 16, 32, 64, 128]
IMAGE_SIZE_CHOICES = [224, 260, 300, 380]
FRAMES_CHOICES = [2, 3, 4, 5]
PRECISIONS = ["fp32", "fp16"]

# Deployment runtimes latency is measured in. TorchScript is kept only for
# comparison — it is not what a rig runs, and it mispredicts the frame budget
# badly (on one box, b0@224: TorchScript CPU p99 81ms vs OpenVINO 24ms).
TARGETS = ["auto", "openvino", "tensorrt", "torchscript"]
TARGET_LABELS = {
    "auto": "auto (TensorRT if CUDA, else OpenVINO)",
    "openvino": "CPU rig — OpenVINO",
    "tensorrt": "GPU rig — TensorRT",
    "torchscript": "TorchScript (comparison only)",
}
EXPORT_FORMATS = ["torchscript", "onnx", "openvino", "tensorrt"]

# CropParams defaults mirror backend/routers/preprocessing.py::CropParams.
CROP_DEFAULTS = {
    "box_size": 220,
    "difference_threshold": 5,
    "difference_mode": "directional",
    "downsample_factor": 1,
    "open1_kernel": 4,
    "close_kernel": 15,
    "open2_kernel": 10,
    "resize_factor": 1.0,
    "smoothing": 0.0,
    "deadband": 0.0,
    "min_contour_area": 0.0,
}

# (label, crop_params["method"]) — see backend/services/cropping.py
CROP_METHODS = [
    ("Background subtraction", "background"),
    ("SAM2 localizer", "localizer"),
    ("None (whole frame)", "none"),
]
# One line under the Crop method dropdown saying what the choice does.
CROP_METHOD_HINTS = {
    "background": "Finds the animal by subtracting a background image. Needs a "
                  "static camera and a background (Compute background below).",
    "localizer": "A small network finds the animal: no background needed, and it "
                 "copes with lighting or bedding changes. Uses the bundled mouse "
                 "network unless you choose or train another.",
    "none": "Writes whole frames, only resized (resize_factor). For rigs where the "
            "animal fills the frame.",
}
# Which CROP_DEFAULTS knobs each method reads.
CROP_KEYS_BY_METHOD = {
    "background": set(CROP_DEFAULTS),
    "localizer": {"box_size", "resize_factor", "smoothing", "deadband"},
    "none": {"resize_factor"},
}
# EMA the localizer pipeline was validated with (label_video: 0.5).
LOCALIZER_SMOOTHING = 0.5

# (label, runtime, precision) for the classifier at inference — see trt_engines.py
INFERENCE_RUNTIMES = [
    ("Auto (TensorRT fp16 if available)", "auto", "fp16"),
    ("TensorRT fp16", "tensorrt", "fp16"),
    ("TensorRT fp32", "tensorrt", "fp32"),
    ("PyTorch", "torch", "fp16"),
]

# (label, decoder) — see video_decode.py
DECODERS = [
    ("Auto (GPU if bit-exact)", "auto"),
    ("GPU (NVDEC)", "nvdec"),
    ("CPU (OpenCV)", "cpu"),
]


def parse_stages(text: str) -> list[int]:
    """'5,6,7' -> [5,6,7] (ignores blanks/junk)."""
    out = []
    for part in str(text).replace(" ", "").split(","):
        if part:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out


def resolve_binarize(positive: list, existing_names=()) -> tuple:
    """Turn the ticked positive behaviors into (binarize, label_union).

    Mirrors the Electron plugin's rule:
      0 ticked -> multi-class (None, None)
      1 ticked -> binarize on that behavior
      2+       -> union them under one synthetic positive label, then binarize
                  on it. The synthetic name must not collide with a real
                  behavior or load_data would merge the wrong rows, so it gains
                  trailing underscores until it is unique.
    """
    positive = [p for p in positive if p]
    if not positive:
        return None, None
    if len(positive) == 1:
        return positive[0], None

    taken = set(existing_names)
    union_name = "positive"
    while union_name in taken:
        union_name += "_"
    return union_name, {union_name: list(positive)}


def build_train_config(
    label_paths: list[str],
    frame_dirs: list[str],
    *,
    frames_per_group: int = 3,
    variant: str = "b0",
    tsm_stages: Optional[list[int]] = None,
    temporal_pool: str = "last",
    unidirectional: bool = False,
    pretrained: bool = True,
    dropout: float = 0.2,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    max_epochs: int = 100,
    early_stopping_patience: int = 20,
    binarize: Optional[str] = None,
    label_union: Optional[dict] = None,
    train_percent: float = 0.7,
    val_percent: float = 0.15,
    test_percent: float = 0.15,
    boundary_smoothing_enabled: bool = False,
    boundary_smoothing_epsilon: float = 0.3,
    boundary_smoothing_window: int = 5,
    use_wandb: bool = False,
    project_name: str = "b-real",
    wandb_api_key: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> dict:
    """Flat dict for POST /api/training/start (TrainingConfig).

    num_frames is tied to frames_per_group (a clip is that many frames), matching
    the JS. Fields not set here fall back to the server's Pydantic defaults.
    """
    cfg = {
        "label_paths": list(label_paths),
        "frame_dirs": list(frame_dirs),
        "frames_per_group": frames_per_group,
        "num_frames": frames_per_group,
        "variant": variant,
        "tsm_stages": tsm_stages if tsm_stages is not None else [5, 6, 7],
        "temporal_pool": temporal_pool,
        "unidirectional": unidirectional,
        "pretrained": pretrained,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": test_percent,
        "boundary_smoothing_enabled": boundary_smoothing_enabled,
        "boundary_smoothing_epsilon": boundary_smoothing_epsilon,
        "boundary_smoothing_window": boundary_smoothing_window,
        "use_wandb": use_wandb,
        "project_name": project_name,
    }
    if binarize:
        cfg["binarize"] = binarize
    if label_union:
        cfg["label_union"] = label_union
    if use_wandb and wandb_api_key:
        cfg["wandb_api_key"] = wandb_api_key
    if checkpoint_dir:
        cfg["checkpoint_dir"] = checkpoint_dir
    if run_name:
        cfg["run_name"] = run_name
    return cfg


def range_spec(low: float, high: float, log: bool = False) -> dict:
    """One continuous dimension of the search space (RangeSpec)."""
    return {"low": float(low), "high": float(high), "log": bool(log)}


def choice_spec(choices: list) -> dict:
    """One discrete dimension of the search space (CategoricalSpec)."""
    return {"choices": list(choices)}


def latency_spec(max_p99_ms: float, *, batch_size: int = 1, precision: str = "fp32",
                 iters: int = 100, warmup: int = 20, target: str = "auto") -> dict:
    """Frame-budget constraint (LatencyConstraint). Architectures whose p99 in
    the deployment runtime exceeds max_p99_ms are rejected before they train.

    `target` names that runtime — pick the one the rig runs, which need not be
    the machine doing the sweep.
    """
    return {
        "max_p99_ms": float(max_p99_ms), "batch_size": int(batch_size),
        "precision": precision, "iters": int(iters), "warmup": int(warmup),
        "target": target,
    }


def build_tuning_request(base: dict, *, space: dict, n_trials: int = 20,
                         objective: str = "val_mcc",
                         latency: Optional[dict] = None) -> dict:
    """Flat dict for POST /api/tuning/start (TuningRequest).

    `base` is a build_train_config() result — the fixed baseline the sweep varies
    around. `space` maps parameter name -> range_spec()/choice_spec(); only the
    parameters present are swept.
    """
    req = {
        "base": base,
        "n_trials": int(n_trials),
        "objective_metric": objective,
        **space,
    }
    if latency:
        req["latency"] = latency
    return req


def format_train_event(ev: dict) -> str:
    """One readable line for the Train tab's log from a training SSE event."""
    status = ev.get("status")
    # Checked before "message": a failed run's final event is its last line
    # (often a message) with the status and error laid over it.
    if status == "error":
        return f"Error: {ev.get('error') or 'training failed (no detail)'}"
    if status == "cancelled":
        return "Stopped."
    if ev.get("message"):
        return str(ev["message"])
    if status == "completed":
        ckpt = ev.get("best_checkpoint")
        if not ckpt:
            return "Done."
        scores = [f"{name} {ev[key]:.3f}" for key, name in
                  (("best_val_mcc", "best val MCC"), ("test_mcc", "test MCC"))
                  if isinstance(ev.get(key), (int, float))]
        return "Done. " + "".join(f"{s} · " for s in scores[:-1]) + \
            (f"{scores[-1]}: " if scores else "") + os.path.basename(ckpt)
    if "epoch" in ev:
        def num(key, fmt=".3f"):
            v = ev.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) else "?"
        line = (f"epoch {ev['epoch']}"
                + (f" (max {ev['total_epochs']})" if ev.get("total_epochs") else "")
                + f" · train loss {num('train_loss')} · val loss {num('val_loss')}"
                + f" · val MCC {num('val_mcc')} (best {num('best_val_mcc')})")
        if ev.get("patience"):
            line += f" · no gain {ev.get('stagnation_epochs', 0)}/{ev['patience']}"
        if isinstance(ev.get("lr"), (int, float)):
            line += f" · lr {ev['lr']:.1e}"
        return line
    return ", ".join(f"{k}: {v}" for k, v in ev.items() if not isinstance(v, (list, dict)))


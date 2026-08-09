"""Pure config-building for the pipeline UI (testable without Qt).

Mirrors the DOM->config marshalling in plugins/pipeline.js: assembles the flat
dicts the backend endpoints expect (crop params, training config). Keeping this
separate from the widgets lets it be unit-tested against the backend contract.
"""

from __future__ import annotations

from typing import Optional

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
    max_epochs: int = 100,
    early_stopping_patience: int = 20,
    binarize: Optional[str] = None,
    train_percent: float = 0.7,
    val_percent: float = 0.15,
    test_percent: float = 0.15,
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
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": test_percent,
    }
    if binarize:
        cfg["binarize"] = binarize
    if checkpoint_dir:
        cfg["checkpoint_dir"] = checkpoint_dir
    if run_name:
        cfg["run_name"] = run_name
    return cfg

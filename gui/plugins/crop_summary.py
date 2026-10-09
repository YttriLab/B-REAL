"""Crop settings in human terms, and what a model was trained with.

A model only works on crops like the ones it was trained on. Preprocessing
writes the settings it used beside the frames (``frames/<id>/crop_params.json``,
see backend/services/preprocess_service.py), training copies them onto the
model's manifest entry, and Active Learning compares them with what it is
about to use.

``box_size`` means different things per method — background subtraction crops
``box_size * downsample_factor`` px of the (resized) frame, the localizer
``box_size`` px — so summaries and comparisons use the box in video pixels.
"""

from __future__ import annotations

import json
import os

CROP_PARAMS_JSON = "crop_params.json"  # written by preprocess_service
METHOD_NAMES = {"background": "background subtraction", "localizer": "SAM2 localizer",
                "none": "no crop (whole frame)"}
# Knobs that change which pixels a crop contains, beyond method / box / resize.
_SHARED = ("smoothing", "deadband")
_BY_METHOD = {
    "background": ("difference_threshold", "difference_mode", "open1_kernel",
                   "close_kernel", "open2_kernel", "min_contour_area"),
    "localizer": ("localizer_path", "presence_threshold"),
    "none": (),
}


def _method(p: dict) -> str:
    return (p or {}).get("method") or "background"


def box_px(p: dict):
    """The crop's side in video pixels (None for whole frames)."""
    method = _method(p)
    if method == "none":
        return None
    box = float(p.get("box_size", 220))
    if method == "background":
        box *= float(p.get("downsample_factor", 1) or 1)
    return box / float(p.get("resize_factor", 1.0) or 1.0)


def summarize(p: dict | None) -> str:
    if not p:
        return "—"
    method = _method(p)
    parts = [METHOD_NAMES.get(method, method)]
    box = box_px(p)
    if box is not None:
        detail = ""
        if method == "background" and (p.get("downsample_factor") or 1) != 1:
            detail = f" ({p.get('box_size')} × downsample {p.get('downsample_factor')})"
        parts.append(f"{box:.0f} px box{detail}")
    if (p.get("resize_factor") or 1.0) != 1.0:
        parts.append(f"resize {p['resize_factor']}")
    if method != "none" and p.get("smoothing"):
        parts.append(f"smoothing {p['smoothing']}")
    if method == "localizer":
        path = p.get("localizer_path")
        parts.append(f"weights {os.path.basename(path)}" if path else "bundled weights")
    return " · ".join(parts)


def differences(trained: dict, used: dict) -> list[str]:
    """What differs between a model's training crop and another, in words."""
    if _method(trained) != _method(used):
        return [f"method {METHOD_NAMES.get(_method(used))} vs "
                f"{METHOD_NAMES.get(_method(trained))} in training"]
    out = []
    bt, bu = box_px(trained), box_px(used)
    if bt is not None and bu is not None and abs(bt - bu) > 0.5:
        out.append(f"box {bu:.0f} px vs {bt:.0f} px in training")
    for key in ("resize_factor", *_SHARED, *_BY_METHOD.get(_method(used), ())):
        a, b = trained.get(key), used.get(key)
        if a != b and not (a in (None, 0, 0.0) and b in (None, 0, 0.0)):
            out.append(f"{key} {b} vs {a} in training")
    return out


def read_frames_record(frame_dir: str) -> dict | None:
    try:
        with open(os.path.join(frame_dir, CROP_PARAMS_JSON)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def training_crop(frame_dirs) -> tuple[dict | None, str]:
    """The crop a model's training frames were made with: (params, note).
    params is None when it can't be stated as one setting; note says why
    ("" when it can)."""
    records = [read_frames_record(d) for d in frame_dirs]
    if not records or any(r is None for r in records):
        return None, "not recorded (frames made before crop settings were saved)"
    if any(r.get("mixed") for r in records):
        return None, "mixed (some training frames were re-made with other settings)"
    first = records[0]["crop_params"]
    if any(differences(first, r["crop_params"]) for r in records[1:]):
        return None, "mixed (training videos were cropped differently)"
    return first, ""

"""Shared data, inference, and serialization helpers for binary benchmarks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import MultiImageDataModule, load_data, train_test_split
from model_tsm import TSMClassifier


DEFAULT_DATA_ROOT = Path(r"C:\Realtime\brayden_data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarking" / "figures" / "binary_behaviors"
DEFAULT_SPLIT_CONFIG = REPO_ROOT / "benchmarking" / "configs" / "brayden_binary_chronological_splits.json"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot make an output name from behavior {value!r}")
    return slug


def discover_sources(data_root: Path) -> tuple[list[Path], list[Path], list[str]]:
    """Find matching frame directories and label CSVs beneath a project root."""
    import pandas as pd

    frame_root = data_root / "frames"
    label_root = data_root / "labels"
    if not frame_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Expected frames/ and labels/ beneath {data_root}")

    frame_dirs: list[Path] = []
    label_paths: list[Path] = []
    for frame_dir in sorted((p for p in frame_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        candidates = [
            label_root / f"{frame_dir.name}_bouts.csv",
            label_root / f"{frame_dir.name}.csv",
        ]
        label_path = next((p for p in candidates if p.is_file()), None)
        if label_path is not None:
            frame_dirs.append(frame_dir.resolve())
            label_paths.append(label_path.resolve())

    if not frame_dirs:
        raise FileNotFoundError(f"No matching frames/<name> and labels/<name>[_bouts].csv pairs in {data_root}")

    behavior_lists = [list(pd.read_csv(path, nrows=0).columns[1:]) for path in label_paths]
    behaviors = [name for name in behavior_lists[0] if all(name in names for names in behavior_lists[1:])]
    if not behaviors:
        raise ValueError("The matched label files have no shared behavior columns")
    return frame_dirs, label_paths, behaviors


def resolve_behavior(requested: str, available: list[str]) -> str:
    """Accept an exact name or a space/slash/hyphen-insensitive alias."""
    if requested in available:
        return requested
    requested_slug = slugify(requested)
    matches = [name for name in available if slugify(name) == requested_slug]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Behavior {requested!r} was not found. Available: {', '.join(available)}")


def load_binary_split(
    data_root: Path,
    behavior: str,
    *,
    frames_per_group: int = 3,
    train_percent: float = 0.70,
    val_percent: float = 0.15,
    test_percent: float = 0.15,
    shuffle: bool = False,
    random_seed: int = 42,
    split_config: Path | None = DEFAULT_SPLIT_CONFIG,
):
    frame_dirs, label_paths, available = discover_sources(data_root)
    behavior = resolve_behavior(behavior, available)
    paths, labels, class_names, boundary_dists = load_data(
        frame_dirs=[str(path) for path in frame_dirs],
        label_paths=[str(path) for path in label_paths],
        frames_per_group=frames_per_group,
        binarize=behavior,
    )
    # model.load_data names the negative class "other" for every binary
    # target. Avoid the ambiguous ['other', 'other'] case when "other" itself
    # is the positive behavior.
    if behavior == "other":
        class_names = ["not_other", "other"]
    split_order_by_dataset = None
    if not shuffle and split_config is not None and split_config.is_file():
        # Windows PowerShell 5.1 writes a UTF-8 BOM when Set-Content uses
        # -Encoding UTF8. utf-8-sig accepts those generated configs while also
        # reading ordinary BOM-free UTF-8 files unchanged.
        config = json.loads(split_config.read_text(encoding="utf-8-sig"))
        configured_percentages = config.get("percentages", {})
        requested_percentages = {
            "train": train_percent, "val": val_percent, "test": test_percent
        }
        if any(
            abs(float(configured_percentages.get(name, -1)) - value) >= 1e-9
            for name, value in requested_percentages.items()
        ):
            raise ValueError(
                f"Split config percentages {configured_percentages!r} do not match "
                f"requested percentages {requested_percentages!r}"
            )
        if int(config.get("boundary_embargo", -1)) != frames_per_group - 1:
            raise ValueError(
                f"Split config boundary embargo does not match frames_per_group={frames_per_group}"
            )
        configured_names = config.get("dataset_names")
        actual_names = [path.name for path in frame_dirs]
        if configured_names != actual_names:
            raise ValueError(
                f"Split config datasets {configured_names!r} do not match discovered datasets {actual_names!r}"
            )
        behavior_config = config.get("behaviors", {}).get(behavior)
        if behavior_config is None:
            raise ValueError(f"Split config has no entry for behavior {behavior!r}: {split_config}")
        split_order_by_dataset = behavior_config["split_order_by_dataset"]

    train_data, val_data, test_data, test_clip_sizes = train_test_split(
        paths,
        labels,
        boundary_dists,
        train_percent=train_percent,
        val_percent=val_percent,
        test_percent=test_percent,
        shuffle=shuffle,
        random_seed=random_seed,
        split_order_by_dataset=split_order_by_dataset,
        boundary_embargo=frames_per_group - 1 if not shuffle else 0,
    )
    return {
        "behavior": behavior,
        "class_names": class_names,
        "frame_dirs": frame_dirs,
        "label_paths": label_paths,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "test_clip_sizes": test_clip_sizes,
        "split": {
            "train_percent": train_percent,
            "val_percent": val_percent,
            "test_percent": test_percent,
            "shuffle": shuffle,
            "random_seed": random_seed,
            "frames_per_group": frames_per_group,
            "split_config": str(split_config.resolve()) if split_order_by_dataset is not None else None,
            "split_order_by_dataset": split_order_by_dataset,
        },
    }


def load_multibehavior_split(
    data_root: Path,
    *,
    frames_per_group: int = 3,
    train_percent: float = 0.70,
    val_percent: float = 0.15,
    test_percent: float = 0.15,
    shuffle: bool = False,
    random_seed: int = 42,
    split_config: Path | None = DEFAULT_SPLIT_CONFIG,
):
    frame_dirs, label_paths, _ = discover_sources(data_root)
    paths, labels, class_names, boundary_dists = load_data(
        frame_dirs=[str(path) for path in frame_dirs],
        label_paths=[str(path) for path in label_paths],
        frames_per_group=frames_per_group,
    )
    split_order_by_dataset = None
    if not shuffle and split_config is not None and split_config.is_file():
        config = json.loads(split_config.read_text(encoding="utf-8-sig"))
        if config.get("dataset_names") != [path.name for path in frame_dirs]:
            raise ValueError("Multibehavior split config does not match the discovered datasets")
        behavior_config = config.get("multibehavioral")
        if behavior_config is None:
            raise ValueError(f"Split config has no multibehavioral entry: {split_config}")
        split_order_by_dataset = behavior_config["split_order_by_dataset"]
    train_data, val_data, test_data, test_clip_sizes = train_test_split(
        paths, labels, boundary_dists,
        train_percent=train_percent, val_percent=val_percent, test_percent=test_percent,
        shuffle=shuffle, random_seed=random_seed,
        split_order_by_dataset=split_order_by_dataset,
        boundary_embargo=frames_per_group - 1 if not shuffle else 0,
    )
    return {
        "behavior": "multibehavioral",
        "class_names": class_names,
        "frame_dirs": frame_dirs,
        "label_paths": label_paths,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "test_clip_sizes": test_clip_sizes,
        "split": {
            "train_percent": train_percent, "val_percent": val_percent, "test_percent": test_percent,
            "shuffle": shuffle, "random_seed": random_seed, "frames_per_group": frames_per_group,
            "split_config": str(split_config.resolve()) if split_order_by_dataset is not None else None,
            "split_order_by_dataset": split_order_by_dataset,
        },
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_checkpoint_model(checkpoint: Path, device: torch.device) -> TSMClassifier:
    model = TSMClassifier.load_from_checkpoint(
        checkpoint.resolve(), map_location="cpu", strict=False, pretrained=False
    )
    model.eval().to(device)
    return model


def predict_test_split(
    checkpoint: Path,
    loaded: dict,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    device = choose_device(device_name)
    model = load_checkpoint_model(checkpoint, device)
    module = MultiImageDataModule(
        loaded["train_data"], loaded["val_data"], loaded["test_data"],
        batch_size=batch_size, num_workers=num_workers,
    )
    module.setup(stage="test")
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels, _ in module.test_dataloader():
            pred_parts.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
            true_parts.append(labels.numpy())

    y_true = np.concatenate(true_parts).astype(np.int64)
    y_pred = np.concatenate(pred_parts).astype(np.int64)
    paths = loaded["test_data"][0]
    frame_numbers = np.asarray([int(Path(group[-1]).stem) for group in paths], dtype=np.int64)
    dataset_ids = np.concatenate([
        np.full(size, index, dtype=np.int64)
        for index, size in enumerate(loaded["test_clip_sizes"])
    ])
    if not (len(y_true) == len(frame_numbers) == len(dataset_ids)):
        raise RuntimeError("Prediction metadata did not align with the held-out test samples")
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "frame_numbers": frame_numbers,
        "dataset_ids": dataset_ids,
        "device": np.asarray(str(device)),
    }


def save_predictions(path: Path, predictions: dict[str, np.ndarray], *, behavior: str, checkpoint: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **predictions,
        behavior=np.asarray(behavior),
        checkpoint=np.asarray(str(checkpoint.resolve())),
    )


def load_predictions(path: Path, *, expected_behavior: str | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if expected_behavior is not None and str(result.get("behavior", "")) != expected_behavior:
        raise ValueError(
            f"Prediction cache behavior {str(result.get('behavior', ''))!r} does not match {expected_behavior!r}"
        )
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

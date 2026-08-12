"""Evaluate a TSM Lightning checkpoint on the held-out test split.

This deliberately bypasses ``trainer.test`` and the model's optional logger
callbacks. The GUI-trained checkpoint is valid, but its original test pass
failed when a logger callback called ``ExperimentWriter.log``. This script
only performs forward passes and computes dataset-wide metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking.code.benchmark_size_learning_curve import compute_full_metrics
from model import MultiImageDataModule, load_data, train_test_split
from model_tsm import TSMClassifier


def distribution(labels: list[int], class_names: list[str]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {name: counts.get(index, 0) for index, name in enumerate(class_names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = OmegaConf.load(config_path)
    dl_config = OmegaConf.to_container(config.data_loading, resolve=True)
    file_paths, labels, class_names, boundary_dists = load_data(**dl_config)
    train_data, val_data, test_data, _test_clip_sizes = train_test_split(
        file_paths,
        labels,
        boundary_dists,
        **OmegaConf.to_container(config.data, resolve=True),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(args.device)

    torch.set_float32_matmul_precision("medium")
    # Override pretrained to prevent a checkpoint evaluation from attempting a
    # network download. All learned weights are restored from the checkpoint.
    model = TSMClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
        pretrained=False,
    )
    model.eval().to(device)

    data_module = MultiImageDataModule(
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup(stage="test")

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels, _boundary_dists in data_module.test_dataloader():
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            y_pred_parts.append(predictions)
            y_true_parts.append(batch_labels.numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    metrics = compute_full_metrics(y_true, y_pred, len(class_names), class_names)
    result = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(device),
        "split_order": list(config.data.get("split_order", ["train", "val", "test"])),
        "split_order_by_dataset": OmegaConf.to_container(
            config.data.get("split_order_by_dataset"), resolve=True
        ) if config.data.get("split_order_by_dataset") is not None else None,
        "n_train": len(train_data[1]),
        "n_val": len(val_data[1]),
        "n_test": len(test_data[1]),
        "train_distribution": distribution(train_data[1], class_names),
        "val_distribution": distribution(val_data[1], class_names),
        "test_distribution": distribution(test_data[1], class_names),
        **metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

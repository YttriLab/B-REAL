"""Compute positive-class F1, MCC, accuracy, precision, and recall."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = Path(r"C:\Realtime\brayden_data")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "f1": f1,
        "mcc": mcc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {"negative": tn + fp, "positive": tp + fn},
    }


def bout_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frame_numbers: np.ndarray,
    dataset_ids: np.ndarray,
    target_class: int,
) -> dict:
    """Fraction of true target bouts identified at least once within the bout."""
    hits = total = 0
    start = 0
    for index in range(1, len(y_true) + 1):
        boundary = index == len(y_true)
        if not boundary:
            boundary = (
                dataset_ids[index] != dataset_ids[index - 1]
                or frame_numbers[index] != frame_numbers[index - 1] + 1
                or y_true[index] != y_true[index - 1]
            )
        if boundary:
            if int(y_true[start]) == target_class:
                total += 1
                hits += int(np.any(y_pred[start:index] == target_class))
            start = index
    return {"hit_rate": hits / total if total else None, "bouts_hit": hits, "total_true_bouts": total}


def main() -> None:
    from binary_behavior_common import (
        load_binary_split,
        predict_test_split,
        save_predictions,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    loaded = load_binary_split(
        args.data_root.resolve(), args.behavior, shuffle=args.shuffle_split, random_seed=args.seed
    )
    predictions = predict_test_split(
        args.checkpoint, loaded, batch_size=args.batch_size,
        num_workers=args.num_workers, device_name=args.device,
    )
    result = {
        "behavior": loaded["behavior"],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(predictions["device"]),
        "n_test": len(predictions["y_true"]),
        "positive_class": 1,
        **binary_metrics(predictions["y_true"], predictions["y_pred"]),
        "bout_hit_rate": bout_hit_rate(
            predictions["y_true"], predictions["y_pred"],
            predictions["frame_numbers"], predictions["dataset_ids"], target_class=1,
        ),
    }
    write_json(args.output.resolve(), result)
    if args.predictions_output:
        save_predictions(
            args.predictions_output.resolve(), predictions,
            behavior=loaded["behavior"], checkpoint=args.checkpoint,
        )
    print(args.output.resolve().read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

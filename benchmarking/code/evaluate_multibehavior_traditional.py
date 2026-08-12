"""Evaluate an all-behavior checkpoint and write traditional multiclass metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_binary_traditional import bout_hit_rate


DEFAULT_DATA_ROOT = Path(r"C:\Realtime\brayden_data")


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    size = len(class_names)
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    per_class = {}
    precisions, recalls, f1s = [], [], []
    for class_id, name in enumerate(class_names):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(confusion[class_id, :].sum()),
        }
    total = int(confusion.sum())
    actual = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    numerator = float(np.trace(confusion) * total - np.dot(actual, predicted))
    denominator = float(
        ((total**2 - np.dot(predicted, predicted)) * (total**2 - np.dot(actual, actual))) ** 0.5
    )
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0
    )
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "mcc": numerator / denominator if denominator else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
    }


def multiclass_bout_hit_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frame_numbers: np.ndarray,
    dataset_ids: np.ndarray,
    class_names: list[str],
) -> dict:
    per_class = {
        name: bout_hit_rate(y_true, y_pred, frame_numbers, dataset_ids, class_id)
        for class_id, name in enumerate(class_names)
    }
    total = sum(row["total_true_bouts"] for row in per_class.values())
    hits = sum(row["bouts_hit"] for row in per_class.values())
    available_rates = [row["hit_rate"] for row in per_class.values() if row["hit_rate"] is not None]
    return {
        "overall": {
            "hit_rate": hits / total if total else None,
            "bouts_hit": hits,
            "total_true_bouts": total,
        },
        "macro_hit_rate": float(np.mean(available_rates)) if available_rates else None,
        "per_class": per_class,
    }


def save_confusion_figure(path: Path, counts: np.ndarray, class_names: list[str]) -> None:
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)
    for axis, matrix, title, formatter in (
        (axes[0], counts, "Confusion matrix (counts)", lambda value: str(int(value))),
        (axes[1], normalized, "Confusion matrix (row normalized)", lambda value: f"{value:.2f}"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=(1 if matrix is normalized else None))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), labels=class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Ground truth")
        axis.set_title(title)
        threshold = float(matrix.max()) / 2 if matrix.size else 0
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                axis.text(column, row, formatter(matrix[row, column]), ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from binary_behavior_common import (
        DEFAULT_SPLIT_CONFIG, load_multibehavior_split, predict_test_split,
        save_predictions, write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confusion-figure", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = load_multibehavior_split(
        args.data_root.resolve(), shuffle=args.shuffle_split, random_seed=args.seed,
        split_config=args.split_config.resolve() if args.split_config else None,
    )
    predictions = predict_test_split(
        args.checkpoint, loaded, batch_size=args.batch_size,
        num_workers=args.num_workers, device_name=args.device,
    )
    metrics = multiclass_metrics(predictions["y_true"], predictions["y_pred"], loaded["class_names"])
    result = {
        "mode": "multibehavioral", "checkpoint": str(args.checkpoint.resolve()),
        "device": str(predictions["device"]), "n_test": len(predictions["y_true"]),
        "class_names": loaded["class_names"], **metrics,
        "bout_hit_rate": multiclass_bout_hit_rates(
            predictions["y_true"], predictions["y_pred"],
            predictions["frame_numbers"], predictions["dataset_ids"], loaded["class_names"],
        ),
    }
    write_json(args.output.resolve(), result)
    save_confusion_figure(
        args.confusion_figure.resolve(), np.asarray(metrics["confusion_matrix"]), loaded["class_names"]
    )
    if args.predictions_output:
        save_predictions(
            args.predictions_output.resolve(), predictions,
            behavior="multibehavioral", checkpoint=args.checkpoint,
        )
    print(args.output.resolve().read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

"""Compute multiclass bout latency, coalescing edit distance, and segmental F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from binary_behavior_common import (
    DEFAULT_DATA_ROOT, DEFAULT_SPLIT_CONFIG, load_multibehavior_split,
    load_predictions, write_json,
)
from evaluate_binary_segments import (
    bouts_from_sequence, coalescing_edit_distance, segmental_f1, split_sequences,
)


def evaluate(y_true, y_pred, frames, datasets, class_names: list[str]) -> dict:
    all_gt = []
    all_pred = []
    edit_distances = []
    gt_bout_counts = []
    latency_by_class = {class_id: [] for class_id in range(len(class_names))}
    totals = np.zeros(len(class_names), dtype=np.int64)
    identified = np.zeros(len(class_names), dtype=np.int64)

    true_sequences = list(split_sequences(y_true, frames, datasets))
    pred_sequences = list(split_sequences(y_pred, frames, datasets))
    if len(true_sequences) != len(pred_sequences):
        raise RuntimeError("Ground-truth and prediction sequence counts differ")
    for (truth, seq_frames, dataset_id), (pred, pred_frames, pred_dataset_id) in zip(
        true_sequences, pred_sequences
    ):
        if dataset_id != pred_dataset_id or not np.array_equal(seq_frames, pred_frames):
            raise RuntimeError("Ground-truth and prediction sequences are misaligned")
        gt_bouts = bouts_from_sequence(truth, seq_frames, dataset_id)
        pred_bouts = bouts_from_sequence(pred, seq_frames, dataset_id)
        all_gt.extend(gt_bouts)
        all_pred.extend(pred_bouts)
        edit_distances.append(coalescing_edit_distance(
            [bout.class_id for bout in pred_bouts],
            [bout.class_id for bout in gt_bouts],
        ))
        gt_bout_counts.append(len(gt_bouts))
        for bout in gt_bouts:
            totals[bout.class_id] += 1
            within = (seq_frames >= bout.start) & (seq_frames <= bout.end)
            correct = np.flatnonzero(within & (pred == bout.class_id))
            if len(correct):
                identified[bout.class_id] += 1
                latency_by_class[bout.class_id].append(int(seq_frames[correct[0]] - bout.start))

    latency = {}
    all_latencies = []
    for class_id, name in enumerate(class_names):
        values = latency_by_class[class_id]
        all_latencies.extend(values)
        latency[name] = {
            "mean_frames": float(np.mean(values)) if values else None,
            "median_frames": float(np.median(values)) if values else None,
            "identified_bouts": int(identified[class_id]),
            "missed_bouts": int(totals[class_id] - identified[class_id]),
            "total_bouts": int(totals[class_id]),
            "identified_fraction": float(identified[class_id] / totals[class_id]) if totals[class_id] else None,
        }
    total_distance = int(sum(edit_distances))
    total_gt_bouts = int(sum(gt_bout_counts))
    result = {
        "bout_identification_latency": {
            "overall_mean_frames": float(np.mean(all_latencies)) if all_latencies else None,
            "overall_median_frames": float(np.median(all_latencies)) if all_latencies else None,
            "per_class": latency,
        },
        "segmental_edit": {
            "distance": total_distance,
            "ground_truth_bouts": total_gt_bouts,
            "normalized_score": max(0.0, 100 * (1 - total_distance / total_gt_bouts)) if total_gt_bouts else None,
            "per_contiguous_sequence": edit_distances,
            "coalescing_semantics": True,
        },
        "segmental_f1": {},
    }
    for percent in (10, 25, 50):
        threshold = percent / 100
        result["segmental_f1"][f"f1@{percent}"] = {
            "all_classes": segmental_f1(all_gt, all_pred, threshold),
            "per_class": {
                name: segmental_f1(all_gt, all_pred, threshold, class_id=class_id)
                for class_id, name in enumerate(class_names)
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    loaded = load_multibehavior_split(
        args.data_root.resolve(), shuffle=args.shuffle_split, random_seed=args.seed,
        split_config=args.split_config.resolve() if args.split_config else None,
    )
    predictions = load_predictions(args.predictions.resolve(), expected_behavior="multibehavioral")
    result = {
        "mode": "multibehavioral", "checkpoint": str(predictions.get("checkpoint", "")),
        "class_names": loaded["class_names"], "n_test": len(predictions["y_true"]),
        **evaluate(
            predictions["y_true"], predictions["y_pred"],
            predictions["frame_numbers"], predictions["dataset_ids"], loaded["class_names"],
        ),
    }
    write_json(args.output.resolve(), result)
    print(args.output.resolve().read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

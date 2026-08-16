"""Compute bout latency, coalescing edit distance, and segmental F1@k."""

from __future__ import annotations

import argparse
import heapq
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = Path(r"C:\Realtime\brayden_data")


@dataclass(frozen=True)
class Bout:
    class_id: int
    start: int
    end: int  # inclusive native frame number
    dataset_id: int


def split_sequences(values: np.ndarray, frame_numbers: np.ndarray, dataset_ids: np.ndarray):
    """Yield contiguous sequences; gaps and recording changes are hard boundaries."""
    start = 0
    for index in range(1, len(values) + 1):
        boundary = index == len(values)
        if not boundary:
            boundary = (
                dataset_ids[index] != dataset_ids[index - 1]
                or frame_numbers[index] != frame_numbers[index - 1] + 1
            )
        if boundary:
            yield values[start:index], frame_numbers[start:index], int(dataset_ids[start])
            start = index


def bouts_from_sequence(values: np.ndarray, frames: np.ndarray, dataset_id: int) -> list[Bout]:
    bouts: list[Bout] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            bouts.append(Bout(int(values[start]), int(frames[start]), int(frames[index - 1]), dataset_id))
            start = index
    return bouts


def temporal_iou(predicted: Bout, ground_truth: Bout) -> float:
    intersection = max(0, min(predicted.end, ground_truth.end) - max(predicted.start, ground_truth.start) + 1)
    union = max(predicted.end, ground_truth.end) - min(predicted.start, ground_truth.start) + 1
    return intersection / union if union else 0.0


def segmental_f1(gt_bouts: list[Bout], pred_bouts: list[Bout], threshold: float, class_id: int | None = None) -> dict:
    gt = [bout for bout in gt_bouts if class_id is None or bout.class_id == class_id]
    pred = [bout for bout in pred_bouts if class_id is None or bout.class_id == class_id]
    matched: set[int] = set()
    tp = fp = 0
    for predicted in pred:
        best_iou = -1.0
        best_index = None
        for index, actual in enumerate(gt):
            if index in matched or predicted.class_id != actual.class_id or predicted.dataset_id != actual.dataset_id:
                continue
            iou = temporal_iou(predicted, actual)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= threshold:
            tp += 1
            matched.add(best_index)
        else:
            fp += 1
    fn = len(gt) - len(matched)
    denominator = 2 * tp + fp + fn
    return {"f1": (2 * tp / denominator if denominator else 0.0), "tp": tp, "fp": fp, "fn": fn}


def levenshtein(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def coalescing_edit_distance(left: list[int], right: list[int]) -> int:
    """Edit distance between bout strings with free adjacent coalescing.

    Each reduced bout string denotes all strings made by repeating its runs.
    A shortest-path alignment between those run automata implements the useful
    coalescing rule (for example, ABA -> A costs one deletion of B).
    """
    if not left:
        return len(right)
    if not right:
        return len(left)
    m, n = len(left), len(right)
    distances = {(0, 0): 0}
    queue = [(0, 0, 0)]
    while queue:
        distance, i, j = heapq.heappop(queue)
        if distance != distances[(i, j)]:
            continue
        if i == m and j == n:
            return distance
        left_steps = [] if i == m else [(i, left[i]), (i + 1, left[i])]
        right_steps = [] if j == n else [(j, right[j]), (j + 1, right[j])]
        transitions = []
        for next_i, left_label in left_steps:
            transitions.append((next_i, j, 1))
            for next_j, right_label in right_steps:
                transitions.append((next_i, next_j, int(left_label != right_label)))
        for next_j, _ in right_steps:
            transitions.append((i, next_j, 1))
        for next_i, next_j, cost in transitions:
            if next_i == i and next_j == j:
                continue
            candidate = distance + cost
            if candidate < distances.get((next_i, next_j), 10**9):
                distances[(next_i, next_j)] = candidate
                heapq.heappush(queue, (candidate, next_i, next_j))
    raise RuntimeError("Unable to calculate coalescing edit distance")


def coalescing_binary_edit_distance(left: list[int], right: list[int]) -> int:
    """Exact edit distance for coalesced binary bouts where AA automatically becomes A.

    A binary coalesced string is completely described by its first class and
    length. Deleting/substituting an interior bout removes it and coalesces its
    equal neighbors, so ABA -> A costs one as requested.
    """
    if len(set(left + right)) > 2:
        return coalescing_edit_distance(left, right)
    if not left:
        return 0 if not right else (len(right) + 1) // 2
    if not right:
        return (len(left) + 1) // 2
    target = (len(right), right[0])
    start = (len(left), left[0])
    queue = deque([(start, 0)])
    seen = {start}
    max_length = max(len(left), len(right)) + 2
    while queue:
        (length, first), distance = queue.popleft()
        if (length, first) == target:
            return distance
        neighbors = []
        if length == 0:
            neighbors.extend([(1, 0), (1, 1)])
        else:
            if length >= 1:
                neighbors.append((length - 1, first ^ 1 if length > 1 else 0))
                neighbors.append((length - 1, first if length > 1 else 0))
            if length >= 3:
                neighbors.append((length - 2, first))
            if length < max_length:
                neighbors.extend([(length + 1, first), (length + 1, first ^ 1)])
            if length + 2 <= max_length:
                # Insert an opposite-class bout inside an existing bout. The
                # original bout remains on both sides (A -> ABA).
                neighbors.append((length + 2, first))
        for state in neighbors:
            if state[0] == 0:
                state = (0, 0)
            if state not in seen:
                seen.add(state)
                queue.append((state, distance + 1))
    raise RuntimeError("Unable to calculate coalescing edit distance")


def calculate_metrics(y_true, y_pred, frames, datasets) -> dict:
    all_gt: list[Bout] = []
    all_pred: list[Bout] = []
    edit_distances = []
    gt_bout_counts = []
    latencies = []
    identified = 0
    total_positive = 0

    true_sequences = list(split_sequences(y_true, frames, datasets))
    pred_sequences = list(split_sequences(y_pred, frames, datasets))
    for (truth, seq_frames, dataset_id), (pred, pred_frames, pred_dataset_id) in zip(true_sequences, pred_sequences):
        if dataset_id != pred_dataset_id or not np.array_equal(seq_frames, pred_frames):
            raise RuntimeError("Ground-truth and prediction sequences are misaligned")
        gt_bouts = bouts_from_sequence(truth, seq_frames, dataset_id)
        pred_bouts = bouts_from_sequence(pred, seq_frames, dataset_id)
        all_gt.extend(gt_bouts)
        all_pred.extend(pred_bouts)
        gt_labels = [bout.class_id for bout in gt_bouts]
        pred_labels = [bout.class_id for bout in pred_bouts]
        edit_distances.append(coalescing_binary_edit_distance(pred_labels, gt_labels))
        gt_bout_counts.append(len(gt_labels))

        for bout in (item for item in gt_bouts if item.class_id == 1):
            total_positive += 1
            within = (seq_frames >= bout.start) & (seq_frames <= bout.end)
            correct_indices = np.flatnonzero(within & (pred == 1))
            if len(correct_indices):
                identified += 1
                latencies.append(int(seq_frames[correct_indices[0]] - bout.start))

    total_distance = int(sum(edit_distances))
    total_gt_bouts = int(sum(gt_bout_counts))
    result = {
        "bout_identification_latency": {
            "mean_frames": float(np.mean(latencies)) if latencies else None,
            "median_frames": float(np.median(latencies)) if latencies else None,
            "identified_bouts": identified,
            "missed_bouts": total_positive - identified,
            "total_positive_bouts": total_positive,
            "identified_fraction": identified / total_positive if total_positive else None,
        },
        "segmental_edit": {
            "distance": total_distance,
            "ground_truth_bouts": total_gt_bouts,
            "normalized_score": max(0.0, 100.0 * (1.0 - total_distance / total_gt_bouts)) if total_gt_bouts else None,
            "per_contiguous_sequence": edit_distances,
            "coalescing_semantics": True,
        },
        "segmental_f1": {},
    }
    for percent in (10, 25, 50):
        threshold = percent / 100.0
        result["segmental_f1"][f"f1@{percent}"] = {
            "all_classes": segmental_f1(all_gt, all_pred, threshold),
            "positive_class": segmental_f1(all_gt, all_pred, threshold, class_id=1),
        }
    return result


def main() -> None:
    from binary_behavior_common import (
        DEFAULT_SPLIT_CONFIG,
        load_binary_split,
        load_predictions,
        predict_test_split,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = load_binary_split(
        args.data_root.resolve(), args.behavior, shuffle=args.shuffle_split, random_seed=args.seed,
        split_config=args.split_config.resolve() if args.split_config else None,
    )
    if args.predictions:
        predictions = load_predictions(args.predictions.resolve(), expected_behavior=loaded["behavior"])
        checkpoint = str(predictions.get("checkpoint", ""))
    else:
        if args.checkpoint is None:
            parser.error("Provide either --predictions or --checkpoint")
        predictions = predict_test_split(
            args.checkpoint, loaded, batch_size=args.batch_size,
            num_workers=args.num_workers, device_name=args.device,
        )
        checkpoint = str(args.checkpoint.resolve())

    result = {
        "behavior": loaded["behavior"],
        "checkpoint": checkpoint,
        "n_test": len(predictions["y_true"]),
        **calculate_metrics(
            predictions["y_true"], predictions["y_pred"],
            predictions["frame_numbers"], predictions["dataset_ids"],
        ),
    }
    write_json(args.output.resolve(), result)
    print(args.output.resolve().read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

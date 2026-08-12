"""Select behavior-specific chronological splits for the Brayden datasets.

Every recording remains divided into one contiguous train block, one contiguous
validation block, and one contiguous test block. For each binary behavior this
script exhaustively evaluates all 6^N assignments of those blocks, where N is
the number of recordings, and writes the selected per-recording orders to the
default config consumed by the binary training/evaluation pipeline.

Selection is label-only (never based on model results), in this order:
  1. Keep every class present in a recording represented in its training block.
  2. Maximize the worst val/test recording coverage for the positive behavior.
  3. Minimize the worst split-vs-full positive-prevalence difference.
  4. Minimize the mean prevalence difference and val-vs-test difference.
  5. Prefer more positive bouts and frames in the weaker evaluation split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from binary_behavior_common import DEFAULT_DATA_ROOT, discover_sources
from analyze_split_permutations import search_per_dataset_orders
from model import _chronological_split_indices, load_data


SPLITS = ("train", "val", "test")
ORDERS = tuple(itertools.permutations(SPLITS))
DEFAULT_OUTPUT = REPO_ROOT / "benchmarking" / "figures" / "binary_split_analysis" / "results.json"
DEFAULT_CONFIG = REPO_ROOT / "benchmarking" / "configs" / "brayden_binary_chronological_splits.json"
PIPELINE_ORDER = ["train", "val", "test"]
ITCH_PIPELINE_ORDERS = [
    ["test", "train", "val"],
    ["val", "train", "test"],
]


def positive_bout_count(labels: np.ndarray) -> int:
    if not len(labels):
        return 0
    return int(np.sum((labels == 1) & np.r_[True, labels[:-1] != 1]))


def option_for(labels: np.ndarray, order: tuple[str, ...], embargo: int) -> dict:
    indices = _chronological_split_indices(len(labels), 0.70, 0.15, 0.15, order)
    for split_name in order[1:]:
        indices[split_name] = indices[split_name][embargo:]
    result = {"order": list(order), "splits": {}}
    for split_name in SPLITS:
        selected = labels[indices[split_name]]
        counts = np.bincount(selected, minlength=2)
        result["splits"][split_name] = {
            "n_samples": int(len(selected)),
            "negative_frames": int(counts[0]),
            "positive_frames": int(counts[1]),
            "positive_prevalence": float(counts[1] / len(selected)) if len(selected) else None,
            "positive_bouts": positive_bout_count(selected),
        }
    return result


def analyze_behavior(all_labels: list[list[int]], dataset_names: list[str], embargo: int) -> dict:
    arrays = [np.asarray(labels, dtype=np.int64) for labels in all_labels]
    options = [
        {order: option_for(labels, order, embargo) for order in ORDERS}
        for labels in arrays
    ]
    full = np.concatenate(arrays)
    full_prevalence = float(np.mean(full == 1))
    candidates = []
    rejected_missing_training_class = 0

    for combination in itertools.product(ORDERS, repeat=len(arrays)):
        reports = [options[index][order] for index, order in enumerate(combination)]
        missing_training_class = False
        for labels, report in zip(arrays, reports):
            full_counts = np.bincount(labels, minlength=2)
            train = report["splits"]["train"]
            train_counts = (train["negative_frames"], train["positive_frames"])
            if any(full_counts[class_id] > 0 and train_counts[class_id] == 0 for class_id in (0, 1)):
                missing_training_class = True
                break
        if missing_training_class:
            rejected_missing_training_class += 1
            continue

        aggregate = {}
        for split_name in SPLITS:
            negative = sum(report["splits"][split_name]["negative_frames"] for report in reports)
            positive = sum(report["splits"][split_name]["positive_frames"] for report in reports)
            bouts = sum(report["splits"][split_name]["positive_bouts"] for report in reports)
            total = negative + positive
            aggregate[split_name] = {
                "n_samples": total,
                "negative_frames": negative,
                "positive_frames": positive,
                "positive_prevalence": positive / total if total else None,
                "positive_bouts": bouts,
                "positive_recordings": sum(
                    report["splits"][split_name]["positive_frames"] > 0 for report in reports
                ),
                "positive_frames_by_dataset": [
                    report["splits"][split_name]["positive_frames"] for report in reports
                ],
                "positive_bouts_by_dataset": [
                    report["splits"][split_name]["positive_bouts"] for report in reports
                ],
            }

        if any(aggregate[name]["positive_frames"] == 0 for name in SPLITS):
            continue
        deviations = [
            abs(aggregate[name]["positive_prevalence"] - full_prevalence)
            for name in SPLITS
        ]
        evaluation_coverage = min(
            aggregate["val"]["positive_recordings"], aggregate["test"]["positive_recordings"]
        )
        minimum_eval_bouts = min(
            aggregate["val"]["positive_bouts"], aggregate["test"]["positive_bouts"]
        )
        minimum_eval_frames = min(
            aggregate["val"]["positive_frames"], aggregate["test"]["positive_frames"]
        )
        val_test_difference = abs(
            aggregate["val"]["positive_prevalence"] - aggregate["test"]["positive_prevalence"]
        )
        selection_key = (
            -evaluation_coverage,
            max(deviations),
            float(np.mean(deviations)),
            val_test_difference,
            -minimum_eval_bouts,
            -minimum_eval_frames,
            combination,
        )
        candidates.append((selection_key, {
            "split_order_by_dataset": [list(order) for order in combination],
            "positive_evaluation_recording_coverage": evaluation_coverage,
            "worst_positive_prevalence_deviation": max(deviations),
            "mean_positive_prevalence_deviation": float(np.mean(deviations)),
            "validation_test_prevalence_difference": val_test_difference,
            "minimum_evaluation_positive_bouts": minimum_eval_bouts,
            "minimum_evaluation_positive_frames": minimum_eval_frames,
            "splits": aggregate,
        }))

    if not candidates:
        raise ValueError("No eligible chronological split combination contains positives in every split")
    candidates.sort(key=lambda item: item[0])
    return {
        "full_positive_prevalence": full_prevalence,
        "total_combinations": len(ORDERS) ** len(arrays),
        "eligible_combinations": len(candidates),
        "rejected_missing_training_class": rejected_missing_training_class,
        "dataset_names": dataset_names,
        "recommended": candidates[0][1],
        "top_candidates": [candidate for _, candidate in candidates[:10]],
        "all_candidates": [candidate for _, candidate in candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames-per-group", type=int, default=3)
    args = parser.parse_args()

    frame_dirs, label_paths, behaviors = discover_sources(args.data_root.resolve())
    analyses = {}
    config_behaviors = {}
    for behavior in behaviors:
        _, labels, _, _ = load_data(
            frame_dirs=[str(path) for path in frame_dirs],
            label_paths=[str(path) for path in label_paths],
            frames_per_group=args.frames_per_group,
            binarize=behavior,
        )
        analysis = analyze_behavior(labels, [path.name for path in frame_dirs], args.frames_per_group - 1)
        analyses[behavior] = analysis
        active_orders = (
            [order.copy() for order in ITCH_PIPELINE_ORDERS]
            if behavior == "itch"
            else [PIPELINE_ORDER.copy() for _ in frame_dirs]
        )
        config_behaviors[behavior] = {
            "split_order_by_dataset": active_orders,
            "analysis_recommended_split_order_by_dataset": analysis["recommended"]["split_order_by_dataset"],
        }
        print(f"\n{behavior}")
        for dataset, order in zip(frame_dirs, analysis["recommended"]["split_order_by_dataset"]):
            print(f"  {dataset.name}: {' -> '.join(order)}")
        for split_name, split in analysis["recommended"]["splits"].items():
            print(
                f"  {split_name:<5} positive={split['positive_frames']:4d}/"
                f"{split['n_samples']} ({100 * split['positive_prevalence']:.2f}%), "
                f"bouts={split['positive_bouts']}"
            )

    _, multiclass_labels, multiclass_names, _ = load_data(
        frame_dirs=[str(path) for path in frame_dirs],
        label_paths=[str(path) for path in label_paths],
        frames_per_group=args.frames_per_group,
    )
    multiclass_search = search_per_dataset_orders(
        multiclass_labels,
        multiclass_names,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        minimum_eval_class_count=1,
    )
    multiclass_recommended = multiclass_search["recommended"]
    print("\nmultibehavioral")
    for dataset, order in zip(frame_dirs, multiclass_recommended["split_order_by_dataset"]):
        print(f"  {dataset.name}: {' -> '.join(order)}")

    report = {
        "data_root": str(args.data_root.resolve()),
        "frame_dirs": [str(path) for path in frame_dirs],
        "label_paths": [str(path) for path in label_paths],
        "percentages": {"train": 0.70, "val": 0.15, "test": 0.15},
        "frames_per_group": args.frames_per_group,
        "boundary_embargo": args.frames_per_group - 1,
        "all_blocks_remain_contiguous": True,
        "selection_priority": [
            "retain every available class in each recording's training block",
            "maximize worst val/test recording coverage for the positive behavior",
            "minimize worst and mean positive-prevalence deviation from the full data",
            "minimize validation-vs-test prevalence difference",
            "maximize positive bouts and frames in the weaker evaluation split",
        ],
        "behaviors": analyses,
        "multibehavioral": {
            "class_names": multiclass_names,
            "selection": multiclass_search,
        },
    }
    config = {
        "description": (
            "Generated by analyze_binary_behavior_splits.py. The active pipeline policy uses "
            "train->val->test for every recording except the severely imbalanced itch binary target; "
            "the multibehavioral model uses that same distribution-improving mixed order. "
            "Label-only alternatives are retained for diagnostics."
        ),
        "data_root": str(args.data_root.resolve()),
        "dataset_names": [path.name for path in frame_dirs],
        "percentages": report["percentages"],
        "boundary_embargo": args.frames_per_group - 1,
        "behaviors": config_behaviors,
        "multibehavioral": {
            "split_order_by_dataset": [order.copy() for order in ITCH_PIPELINE_ORDERS],
            "analysis_recommended_split_order_by_dataset": multiclass_recommended["split_order_by_dataset"],
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.config_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.config_output.resolve().write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nDetailed analysis: {args.output.resolve()}")
    print(f"Pipeline config:   {args.config_output.resolve()}")


if __name__ == "__main__":
    main()

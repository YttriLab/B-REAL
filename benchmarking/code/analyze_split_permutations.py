"""Compare all chronological train/val/test block orders for stratification.

This does not train a model. It loads the same usable samples and remapped
labels as training, divides every configured dataset into exact-size contiguous
blocks for each of the six role permutations, and writes a detailed JSON
report. The recommended order is selected from labels only, never model
performance.

Balance score (lower is better):
  1. For every configured dataset and split, calculate total-variation (TV)
     distance between that split's class proportions and the full dataset's
     class proportions: TV = 0.5 * sum(abs(p_split - p_full)).
  2. Average TV equally over train, validation, and test for that dataset.
  3. Average datasets weighted by their usable sample counts.

TV is valid for any number of classes and is interpretable as the fraction of
class-probability mass that would need to move to match the reference
distribution. A score of 0% is perfect. The report also includes aggregate TV
across all datasets as a diagnostic, but it is not the selection criterion
because aggregation can hide opposing imbalances in individual datasets.

Usage:
    python benchmarking/code/analyze_split_permutations.py
    python benchmarking/code/analyze_split_permutations.py --config config_tsm.yaml
    python benchmarking/code/analyze_split_permutations.py --output path/to/results.json
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model import _chronological_split_indices, load_data


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_OUTPUT = REPO_ROOT / "benchmarking" / "figures" / "split_permutations" / "results.json"


def label_distribution(labels, num_classes):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    total = int(counts.sum())
    proportions = counts.astype(float) / total if total else None
    return counts, proportions


def total_variation(left, right):
    # A required split with no samples is maximally imbalanced.
    if left is None or right is None:
        return 1.0
    return float(0.5 * np.abs(left - right).sum())


def distribution_record(counts, proportions, class_names):
    return {
        class_name: {
            "count": int(counts[class_id]),
            "proportion": float(proportions[class_id]) if proportions is not None else None,
        }
        for class_id, class_name in enumerate(class_names)
    }


def analyze_order(all_labels, class_names, percentages, order):
    num_classes = len(class_names)
    aggregate_counts = {name: np.zeros(num_classes, dtype=np.int64) for name in SPLIT_NAMES}
    dataset_reports = []
    weighted_dataset_tv = 0.0
    total_samples = sum(len(labels) for labels in all_labels)

    for dataset_id, labels in enumerate(all_labels):
        full_counts, full_proportions = label_distribution(labels, num_classes)
        indices = _chronological_split_indices(
            len(labels),
            percentages["train"],
            percentages["val"],
            percentages["test"],
            order,
        )

        split_reports = {}
        split_tv = []
        labels_array = np.asarray(labels, dtype=int)
        for split_name in SPLIT_NAMES:
            split_labels = labels_array[indices[split_name]]
            counts, proportions = label_distribution(split_labels, num_classes)
            aggregate_counts[split_name] += counts
            tv = total_variation(proportions, full_proportions)
            split_tv.append(tv)
            split_reports[split_name] = {
                "n_samples": int(len(split_labels)),
                "total_variation": tv,
                "classes": distribution_record(counts, proportions, class_names),
            }

        dataset_mean_tv = float(np.mean(split_tv))
        if total_samples:
            weighted_dataset_tv += len(labels) / total_samples * dataset_mean_tv
        dataset_reports.append({
            "dataset_id": dataset_id,
            "n_samples": len(labels),
            "full_classes": distribution_record(full_counts, full_proportions, class_names),
            "mean_total_variation": dataset_mean_tv,
            "splits": split_reports,
        })

    full_aggregate = sum(aggregate_counts.values(), np.zeros(num_classes, dtype=np.int64))
    full_total = int(full_aggregate.sum())
    full_proportions = full_aggregate / full_total if full_total else None
    aggregate_reports = {}
    aggregate_tv = []
    for split_name in SPLIT_NAMES:
        counts = aggregate_counts[split_name]
        split_total = int(counts.sum())
        proportions = counts / split_total if split_total else None
        tv = total_variation(proportions, full_proportions)
        aggregate_tv.append(tv)
        aggregate_reports[split_name] = {
            "n_samples": split_total,
            "total_variation": tv,
            "classes": distribution_record(counts, proportions, class_names),
        }

    return {
        "order": list(order),
        "balance_score": weighted_dataset_tv,
        "aggregate_mean_total_variation": float(np.mean(aggregate_tv)),
        "aggregate_splits": aggregate_reports,
        "datasets": dataset_reports,
    }


def compact_distribution(split_report, class_names):
    values = []
    for name in class_names:
        proportion = split_report["classes"][name]["proportion"]
        values.append(f"{name}={100 * proportion:.1f}%" if proportion is not None else f"{name}=n/a")
    return ", ".join(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config_tsm.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    data_loading = OmegaConf.to_container(config.data_loading, resolve=True)
    _, all_labels, class_names, _ = load_data(**data_loading)
    percentages = {
        name: float(config.data[f"{name}_percent"])
        for name in SPLIT_NAMES
    }
    if abs(sum(percentages.values()) - 1.0) >= 1e-6:
        raise ValueError(f"Split percentages must sum to 1.0; got {percentages}")

    results = [
        analyze_order(all_labels, class_names, percentages, order)
        for order in itertools.permutations(SPLIT_NAMES)
    ]
    results.sort(key=lambda row: (row["balance_score"], row["aggregate_mean_total_variation"]))
    best = results[0]
    tied_best = [
        row for row in results
        if abs(row["balance_score"] - best["balance_score"]) <= 1e-12
        and abs(row["aggregate_mean_total_variation"] - best["aggregate_mean_total_variation"]) <= 1e-12
    ]

    report = {
        "config_path": str(args.config.resolve()),
        "class_names": class_names,
        "percentages": percentages,
        "selection_metric": {
            "name": "sample-weighted per-dataset mean total-variation distance",
            "formula": (
                "For each dataset: mean over train/val/test of "
                "0.5*sum_classes(abs(split_proportion-full_dataset_proportion)); "
                "then mean over datasets weighted by usable sample count."
            ),
            "lower_is_better": True,
        },
        "recommended_split_order": best["order"],
        "equally_best_split_orders": [row["order"] for row in tied_best],
        "recommended_balance_score": best["balance_score"],
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print("\nChronological split permutation balance (lower is better)")
    print("Balance score = sample-weighted per-dataset mean TV distance.")
    print(f"Classes: {', '.join(class_names)}")
    for rank, result in enumerate(results, start=1):
        order_text = " -> ".join(result["order"])
        print(
            f"{rank}. {order_text:<24} "
            f"balance={100 * result['balance_score']:.3f}%  "
            f"aggregate={100 * result['aggregate_mean_total_variation']:.3f}%"
        )
        for split_name in SPLIT_NAMES:
            split = result["aggregate_splits"][split_name]
            print(
                f"     {split_name:<5} n={split['n_samples']:<7} "
                f"{compact_distribution(split, class_names)}"
            )

    if len(tied_best) > 1:
        print("\nEqually best orders: " + "; ".join(" -> ".join(row["order"]) for row in tied_best))
        print("The first is shown below as the deterministic recommendation.")
    print("\nRecommended config (used only when shuffle: false):")
    print("data:")
    print("  shuffle: false")
    print(f"  split_order: [{', '.join(best['order'])}]")
    print(f"\nDetailed results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

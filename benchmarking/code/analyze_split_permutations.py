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
    python benchmarking/code/analyze_split_permutations.py --config config_tsm.yaml --per-dataset
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


def search_per_dataset_orders(all_labels, class_names, percentages,
                              minimum_eval_class_count=25, top_k=20):
    """Search one contiguous split-order permutation per source dataset.

    Candidate selection deliberately emphasizes the non-majority classes:
      1. Every class present in a source dataset must remain represented in
         that dataset's training block.
      2. Maximize the worst validation/test source-session coverage at the
         requested minimum count.
      3. Jointly minimize session dominance and validation-vs-test class
         distribution mismatch. This prevents a tiny improvement in one
         criterion from selecting the severe session confounding seen with a
         single global chronological order.
      4. Prefer broader positive-count coverage, then use aggregate distance
         from the full distribution as a final tie breaker. A session with
         only a handful of examples therefore cannot outweigh a badly
         mismatched validation/test distribution.

    No samples or bouts are shuffled: each role remains a single contiguous
    chronological block within every dataset.
    """
    orders = list(itertools.permutations(SPLIT_NAMES))
    dataset_options = []
    full_counts_by_dataset = []
    for labels in all_labels:
        full_counts, _ = label_distribution(labels, len(class_names))
        full_counts_by_dataset.append(full_counts)
        options = {}
        for order in orders:
            options[order] = analyze_order(
                [labels], class_names, percentages, order
            )["datasets"][0]
        dataset_options.append(options)

    full_counts = sum(full_counts_by_dataset, np.zeros(len(class_names), dtype=np.int64))
    majority_class_id = int(np.argmax(full_counts))
    selection_class_ids = [
        class_id for class_id in range(len(class_names))
        if class_id != majority_class_id and full_counts[class_id] > 0
    ]
    if not selection_class_ids:
        selection_class_ids = [
            class_id for class_id, count in enumerate(full_counts) if count > 0
        ]

    candidates = []
    rejected_no_train_class = 0
    for order_combination in itertools.product(orders, repeat=len(all_labels)):
        split_counts = {
            split_name: np.zeros((len(all_labels), len(class_names)), dtype=np.int64)
            for split_name in SPLIT_NAMES
        }
        split_sizes = {split_name: np.zeros(len(all_labels), dtype=np.int64)
                       for split_name in SPLIT_NAMES}

        for dataset_id, order in enumerate(order_combination):
            report = dataset_options[dataset_id][order]
            for split_name in SPLIT_NAMES:
                split = report["splits"][split_name]
                split_sizes[split_name][dataset_id] = split["n_samples"]
                split_counts[split_name][dataset_id] = [
                    split["classes"][name]["count"] for name in class_names
                ]

        missing_train_class = any(
            full_counts_by_dataset[dataset_id][class_id] > 0
            and split_counts["train"][dataset_id, class_id] == 0
            for dataset_id in range(len(all_labels))
            for class_id in range(len(class_names))
        )
        if missing_train_class:
            rejected_no_train_class += 1
            continue

        aggregate_counts = {
            split_name: split_counts[split_name].sum(axis=0)
            for split_name in SPLIT_NAMES
        }
        aggregate_proportions = {
            split_name: aggregate_counts[split_name] / aggregate_counts[split_name].sum()
            for split_name in SPLIT_NAMES
        }
        full_proportions = full_counts / full_counts.sum()

        coverage_at_minimum = min(
            int(np.sum(split_counts[split_name][:, class_id] >= minimum_eval_class_count))
            for split_name in ("val", "test")
            for class_id in selection_class_ids
        )
        coverage_positive = min(
            int(np.sum(split_counts[split_name][:, class_id] > 0))
            for split_name in ("val", "test")
            for class_id in selection_class_ids
        )
        maximum_session_share = max(
            float(split_counts[split_name][:, class_id].max()
                  / split_counts[split_name][:, class_id].sum())
            for split_name in ("val", "test")
            for class_id in selection_class_ids
            if split_counts[split_name][:, class_id].sum() > 0
        )
        val_test_tv = total_variation(
            aggregate_proportions["val"], aggregate_proportions["test"]
        )
        aggregate_mean_tv = float(np.mean([
            total_variation(aggregate_proportions[split_name], full_proportions)
            for split_name in SPLIT_NAMES
        ]))
        confounding_risk = maximum_session_share + val_test_tv

        selection_key = (
            -coverage_at_minimum,
            confounding_risk,
            maximum_session_share,
            -coverage_positive,
            aggregate_mean_tv,
            order_combination,
        )
        candidates.append((selection_key, {
            "split_order_by_dataset": [list(order) for order in order_combination],
            "evaluation_source_coverage_at_minimum": coverage_at_minimum,
            "evaluation_source_coverage_positive": coverage_positive,
            "minimum_eval_class_count": minimum_eval_class_count,
            "maximum_evaluation_session_share": maximum_session_share,
            "validation_test_total_variation": val_test_tv,
            "session_confounding_risk": confounding_risk,
            "aggregate_mean_total_variation": aggregate_mean_tv,
            "splits": {
                split_name: {
                    "n_samples": int(split_sizes[split_name].sum()),
                    "classes": {
                        name: {
                            "count": int(aggregate_counts[split_name][class_id]),
                            "proportion": float(aggregate_proportions[split_name][class_id]),
                            "counts_by_dataset": split_counts[split_name][:, class_id].tolist(),
                        }
                        for class_id, name in enumerate(class_names)
                    },
                }
                for split_name in SPLIT_NAMES
            },
        }))

    if not candidates:
        raise ValueError(
            "No per-dataset chronological ordering leaves every class present "
            "in each dataset's training block."
        )

    candidates.sort(key=lambda item: item[0])
    return {
        "selection_class_names": [class_names[class_id] for class_id in selection_class_ids],
        "minimum_eval_class_count": minimum_eval_class_count,
        "total_combinations": len(orders) ** len(all_labels),
        "eligible_combinations": len(candidates),
        "rejected_missing_training_class": rejected_no_train_class,
        "recommended": candidates[0][1],
        "top_candidates": [candidate for _key, candidate in candidates[:top_k]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config_tsm.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-dataset", action="store_true",
        help="Search one chronological split order per configured dataset",
    )
    parser.add_argument(
        "--minimum-eval-class-count", type=int, default=25,
        help="Minimum examples for a dataset to count as representing a class in val/test",
    )
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

    if args.per_dataset:
        search = search_per_dataset_orders(
            all_labels,
            class_names,
            percentages,
            minimum_eval_class_count=args.minimum_eval_class_count,
        )
        report = {
            "config_path": str(args.config.resolve()),
            "class_names": class_names,
            "percentages": percentages,
            "mode": "per_dataset_chronological_orders",
            "dataset_label_paths": [str(path) for path in data_loading["label_paths"]],
            "selection": {
                "priority": [
                    "maximize worst val/test source coverage for non-majority classes",
                    "minimize max session share plus val/test distribution mismatch",
                    "maximize worst positive source coverage",
                    "minimize aggregate distribution mismatch",
                ],
                "all_blocks_remain_contiguous": True,
                **search,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2)

        recommended = search["recommended"]
        print("\nPer-dataset chronological split-order search")
        print(f"Evaluated {search['total_combinations']:,} combinations; "
              f"{search['eligible_combinations']:,} kept every dataset class in training.")
        print("Selection classes: " + ", ".join(search["selection_class_names"]))
        for dataset_id, order in enumerate(recommended["split_order_by_dataset"]):
            print(f"  dataset {dataset_id}: {' -> '.join(order)}")
        print("\nAggregate split distributions:")
        for split_name in SPLIT_NAMES:
            split = recommended["splits"][split_name]
            values = ", ".join(
                f"{name}={100 * split['classes'][name]['proportion']:.2f}% "
                f"({split['classes'][name]['count']})"
                for name in class_names
            )
            print(f"  {split_name:<5} n={split['n_samples']}: {values}")
        print(f"Max evaluation session share: "
              f"{100 * recommended['maximum_evaluation_session_share']:.2f}%")
        print(f"Val/test distribution TV: "
              f"{100 * recommended['validation_test_total_variation']:.2f}%")
        print("\nRecommended config:")
        print("data:")
        print("  shuffle: false")
        print("  split_order_by_dataset:")
        for order in recommended["split_order_by_dataset"]:
            print(f"    - [{', '.join(order)}]")
        print(f"\nDetailed results written to {args.output.resolve()}")
        return

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

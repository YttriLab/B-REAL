"""
Data-size learning curve for the TSM-EfficientNet classifier.

Trains the model on 0%, 20%, 40%, 60%, 80%, 100% of the (fixed) training split
and evaluates each one on the same held-out test split. For every fraction
other than 0% and 100% (where there's only one possible sample, or no sample
at all), a different random subset of that size is drawn 5 times so the plot
shows run-to-run variance. Val/test splits never shrink or change across runs
-- only the training subset size changes -- so accuracy/F1 differences are
attributable to training-set size alone.

The 0% point can't correspond to an actual trained model (there's nothing to
fit), so it's reported as a majority-class baseline: predict the single most
common behavior across all available labels (train+val+test) and score that
fixed prediction against the test set.

The x-axis is "seconds of training data" = (# distinct labeled training
frames used) / fps. Each training sample corresponds to exactly one distinct
native video frame, so this is the amount of labeled footage the model
actually trained on. fps is not auto-detected -- pass it explicitly with
--fps since it depends on the source video(s).

Usage:
    python benchmarking/code/benchmark_size_learning_curve.py --fps 30
    python benchmarking/code/benchmark_size_learning_curve.py --config config_tsm.yaml --fps 30 --repeats 5
    python benchmarking/code/benchmark_size_learning_curve.py --fps 30 --max-epochs 30 --patience 8
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from model import load_data, MultiImageDataModule, train_test_split
from model_tsm import TSMClassifier

FIGURES_DIR = REPO_ROOT / "benchmarking" / "figures" / "size_learning_curve"

# Same palette used by benchmark_inference_speed.py (dataviz skill default).
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Metrics (numpy-only, no sklearn dependency in this repo's env)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float]:
    """Returns (accuracy, macro_f1). Classes with no support get F1=0, matching
    sklearn's f1_score(..., average='macro', zero_division=0)."""
    accuracy = float(np.mean(y_true == y_pred))

    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return accuracy, macro_f1


# ---------------------------------------------------------------------------
# Data loading / subsetting
# ---------------------------------------------------------------------------

def load_experiment_data(config):
    dl = OmegaConf.to_container(config.data_loading, resolve=True)
    all_file_paths, all_labels, all_class_names_union, all_boundary_dists = load_data(**dl)

    train_data, valid_data, test_data, _test_clip_sizes = train_test_split(
        all_file_paths, all_labels, all_boundary_dists,
        **OmegaConf.to_container(config.data, resolve=True)
    )
    return train_data, valid_data, test_data, all_class_names_union


def subsample_train(train_data, n: int, seed: int):
    paths, labels, boundary_dists = train_data
    total = len(labels)
    if n >= total:
        idx = np.arange(total)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(total, size=n, replace=False)
    sub_paths = [paths[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    sub_bd = [boundary_dists[i] for i in idx]
    return sub_paths, sub_labels, sub_bd


def class_weights_for(labels: list, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts_safe = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / counts_safe
    return torch.tensor(weights, dtype=torch.float32)


def majority_class_baseline(train_data, valid_data, test_data, num_classes: int) -> int:
    """Most common label across every available split -- the 0%-training-data
    reference point (no model can be fit with zero samples)."""
    all_labels = train_data[1] + valid_data[1] + test_data[1]
    counts = np.bincount(all_labels, minlength=num_classes)
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Train + evaluate one (fraction, repeat) run
# ---------------------------------------------------------------------------

def train_and_evaluate(subset, valid_data, test_data, config, class_names, args, seed: int) -> tuple[float, float]:
    L.seed_everything(seed, workers=True)

    num_classes = len(class_names)
    max_epochs = args.max_epochs if args.max_epochs is not None else config.max_epochs
    patience = args.patience if args.patience is not None else config.callbacks.early_stopping_patience

    data_module = MultiImageDataModule(
        subset, valid_data, test_data,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    weights = class_weights_for(subset[1], num_classes)
    boundary_smoothing = (
        OmegaConf.to_container(config.boundary_smoothing, resolve=True)
        if "boundary_smoothing" in config else None
    )

    model = TSMClassifier(
        **OmegaConf.to_container(config.model, resolve=True),
        max_epochs=max_epochs,
        weights=weights,
        class_names=class_names,
        boundary_smoothing=boundary_smoothing,
    )

    tmp_dir = tempfile.mkdtemp(prefix="size_curve_")
    try:
        checkpoint_cb = ModelCheckpoint(dirpath=tmp_dir, monitor="val_loss", mode="min", save_top_k=1)
        early_stop_cb = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb, early_stop_cb],
            max_epochs=max_epochs,
            accelerator=args.accelerator,
            enable_progress_bar=args.progress_bar,
        )
        trainer.fit(model, data_module)

        best_path = checkpoint_cb.best_model_path
        if best_path:
            model = TSMClassifier.load_from_checkpoint(best_path, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for images, labels, _boundary_dists in data_module.test_dataloader():
                logits = model(images.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred_list.append(preds)
                y_true_list.append(labels.numpy())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return compute_metrics(y_true, y_pred, num_classes)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(args) -> list[dict]:
    config = OmegaConf.load(args.config)
    train_data, valid_data, test_data, class_names = load_experiment_data(config)
    num_classes = len(class_names)
    total_train = len(train_data[1])

    print(f"Train/val/test samples: {total_train}/{len(valid_data[1])}/{len(test_data[1])}")
    print(f"Classes ({num_classes}): {class_names}")

    fractions = [int(f.strip()) for f in args.fractions.split(",")]

    # Flatten to a single task list up front so the progress bar knows the
    # true total (0%/100% only run once; the rest run --repeats times).
    tasks = []
    for frac in fractions:
        repeats = 1 if frac in (0, 100) else args.repeats
        for rep in range(repeats):
            tasks.append((frac, rep, repeats))

    results = []
    pbar = tqdm(tasks, desc="Learning curve", unit="run")
    for frac, rep, repeats in pbar:
        n = round(frac / 100 * total_train)
        seed = args.seed + frac * 1000 + rep
        pbar.set_description(f"fraction={frac}% rep={rep + 1}/{repeats}")
        tqdm.write(f"\n=== fraction={frac}% repeat={rep + 1}/{repeats} (n={n} frames, seed={seed}) ===")
        start = time.time()

        if frac == 0:
            majority = majority_class_baseline(train_data, valid_data, test_data, num_classes)
            y_true = np.array(test_data[1])
            y_pred = np.full_like(y_true, majority)
            accuracy, f1 = compute_metrics(y_true, y_pred, num_classes)
            n_used = 0
        else:
            subset = subsample_train(train_data, n, seed)
            n_used = len(subset[1])
            accuracy, f1 = train_and_evaluate(subset, valid_data, test_data, config, class_names, args, seed)

        elapsed = time.time() - start
        seconds_train = n_used / args.fps

        tqdm.write(f"  n_used={n_used}  seconds_train={seconds_train:.1f}  "
                   f"accuracy={accuracy:.4f}  macro_f1={f1:.4f}  ({elapsed:.1f}s wall)")
        pbar.set_postfix(acc=f"{accuracy:.3f}", f1=f"{f1:.3f}")

        results.append({
            "fraction": frac,
            "repeat": rep,
            "seed": seed,
            "n_frames_train": n_used,
            "seconds_train": seconds_train,
            "accuracy": accuracy,
            "macro_f1": f1,
            "wall_seconds": elapsed,
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(results: list[dict], metric_key: str, ylabel: str, title: str, out_path: Path):
    fractions = sorted({r["fraction"] for r in results})
    xs_mean, ys_mean, ys_std = [], [], []

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for frac in fractions:
        rows = [r for r in results if r["fraction"] == frac]
        xs = [r["seconds_train"] for r in rows]
        ys = [r[metric_key] for r in rows]

        ax.scatter(xs, ys, color=PALETTE[0], alpha=0.45, s=28, zorder=2,
                   label="Individual runs" if frac == fractions[0] else None)

        xs_mean.append(float(np.mean(xs)))
        ys_mean.append(float(np.mean(ys)))
        ys_std.append(float(np.std(ys)))

    order = np.argsort(xs_mean)
    xs_mean = np.array(xs_mean)[order]
    ys_mean = np.array(ys_mean)[order]
    ys_std = np.array(ys_std)[order]

    ax.errorbar(xs_mean, ys_mean, yerr=ys_std, color=PALETTE[1], marker="o",
                markersize=6, linewidth=1.8, capsize=4, zorder=3, label="Mean ± std")

    ax.set_xlabel("Training data (seconds)", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=14)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config_tsm.yaml"),
                         help="Training config YAML (same shape as config_tsm.yaml) providing "
                              "data_loading, data (split percentages), model, and boundary_smoothing")
    parser.add_argument("--fps", type=float, required=True,
                         help="Frame rate of the source video(s), used to convert "
                              "# training frames into seconds for the x-axis")
    parser.add_argument("--fractions", type=str, default="0,20,40,60,80,100",
                         help="Comma-separated train-data percentages to evaluate")
    parser.add_argument("--repeats", type=int, default=5,
                         help="Random subset draws per fraction (ignored for 0%% and 100%%, "
                              "which only have one possible sample)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--max-epochs", type=int, default=None,
                         help="Override config's max_epochs (useful for faster benchmarking runs)")
    parser.add_argument("--patience", type=int, default=None,
                         help="Override config's early_stopping_patience")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(args)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {results_path}")

    accuracy_path = output_dir / "accuracy_vs_seconds.svg"
    f1_path = output_dir / "f1_vs_seconds.svg"
    plot_learning_curve(results, "accuracy", "Accuracy", "Accuracy vs. training data size", accuracy_path)
    plot_learning_curve(results, "macro_f1", "Macro F1", "Macro F1 vs. training data size", f1_path)

    print(f"Saved {accuracy_path}")
    print(f"Saved {f1_path}")


if __name__ == "__main__":
    main()

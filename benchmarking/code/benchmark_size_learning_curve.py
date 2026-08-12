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

Each training sample corresponds to exactly one distinct native video frame;
results.json always stores that raw count (n_frames_train). The x-axis
itself is converted to seconds using --fps (default 60) at plot time --
pass --fps to match your actual recording, or --fps -1 to skip the
conversion and plot raw frame count instead. Either way results.json is
unaffected, so the same run can be replotted in either unit without
retraining anything.

Progress (args + results-so-far) is saved to <output-dir>/run_state.json after
every run, so a killed/crashed invocation can be continued with --resume
instead of starting over. --resume ignores all other CLI args and reuses
whatever the original run was called with; if every planned run is already
done, it just reports that and exits.

Each training fraction's subset is drawn *stratified* by (source video,
behavior class): sampling happens independently within every (dataset_id,
label) cell of the training pool, so a fraction's subset always keeps both
the same video mix and the same per-video class balance as the full
training pool -- not just an overall random draw that could, by chance,
lean heavily toward one video or drop a rare behavior entirely.

Every run also writes, under --output-dir:
  - epoch_metrics/frac{f}_rep{r}.jsonl -- one JSON line per training epoch
    (train/val loss, balanced acc, MCC, LR), so a run's trajectory can be
    inspected later without re-running it.
  - checkpoints/frac100_rep0_best.ckpt -- the best-val-mcc checkpoint from
    the 100%-data run only (kept as the "reference" model; every other
    run's checkpoint is discarded after evaluation to bound disk usage).
    Checkpoint selection (and early stopping) monitor val_mcc, not val_loss
    -- the two can diverge on this model, and val_mcc is the metric this
    whole benchmark exists to report.
  - benchmarking/models/frac100_rep0_best.pt -- that same checkpoint
    exported to TorchScript.
  - run_meta.json -- one-time reproducibility snapshot: git commit, torch/
    lightning versions, the resolved config, class names, and per-dataset
    label-row counts.
and each entry in results.json additionally records epochs_trained,
epoch_of_best, best_val_mcc, per-class precision/recall/F1/support, a
confusion matrix, MCC, and the class weights used for that run.

Email notifications (GMAIL_ADDRESS/GMAIL_APP_PASSWORD env vars required,
see send_completion_email's docstring; --no-notify disables all of this):
  - A short progress email after every single (fraction, repeat) run
    finishes, with that run's accuracy/macro_f1/mcc.
  - A full report -- results.json, the accuracy/f1/mcc-vs-training-size
    plots, and a val_mcc + val_loss vs. epoch plot for the 100% run --
    the moment the 100%-data run finishes. This happens *before* the
    (separately) risky TorchScript export step is attempted, so an export
    failure can never take the report down with it; export failures and
    email failures are both caught and logged rather than raised, so
    neither can crash the run.
  - If 100% wasn't part of this run's --fractions (or was already reported
    in a prior --resume invocation), main() sends an equivalent fallback
    report (without the epoch plot) at the very end instead.

Usage:
    python benchmarking/code/benchmark_size_learning_curve.py
    python benchmarking/code/benchmark_size_learning_curve.py --max-epochs 5 --repeats 3 --notify-email braydenk@andrew.cmu.edu
    python benchmarking/code/benchmark_size_learning_curve.py --fractions 100 --max-epochs 5 --notify-email braydenk@andrew.cmu.edu --output-dir benchmarking/figures/size_learning_curve_test

    python benchmarking/code/benchmark_size_learning_curve.py --max-epochs 22 --patience 22
    python benchmarking/code/benchmark_size_learning_curve.py --max-epochs 5 --repeats 3
    python benchmarking/code/benchmark_size_learning_curve.py --resume
"""

import argparse
import json
import os
import shutil
import smtplib
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from email.message import EmailMessage
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from torchmetrics import ConfusionMatrix, MatthewsCorrCoef
import matplotlib.pyplot as plt

from model import load_data, MultiImageDataModule, train_test_split
from model_tsm import TSMClassifier
from convert_torchscript_tsm import export_model

FIGURES_DIR = REPO_ROOT / "benchmarking" / "figures" / "size_learning_curve"
MODELS_DIR = REPO_ROOT / "benchmarking" / "models"

# Same palette used by benchmark_inference_speed.py (dataviz skill default).
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                          class_names: list) -> dict:
    """Accuracy, macro F1, MCC, a confusion matrix, and per-class precision/
    recall/F1/support -- enough to diagnose *which* behaviors benefited from
    more data, not just a single aggregate number per run.

    Accuracy/F1/precision/recall are hand-rolled with numpy (no sklearn
    dependency in this repo's env); MCC and the confusion matrix reuse
    torchmetrics, which is already a hard dependency via model_tsm.py, rather
    than hand-rolling the multiclass MCC generalization.
    """
    accuracy = float(np.mean(y_true == y_pred))

    per_class = {}
    f1s = []
    for c, name in enumerate(class_names):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    macro_f1 = float(np.mean(f1s))

    task = "binary" if num_classes <= 2 else "multiclass"
    t_true = torch.as_tensor(y_true, dtype=torch.long)
    t_pred = torch.as_tensor(y_pred, dtype=torch.long)
    mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)(t_pred, t_true).item()
    confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)(t_pred, t_true).tolist()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


# ---------------------------------------------------------------------------
# Data loading / subsetting
# ---------------------------------------------------------------------------

def load_experiment_data(config):
    dl = OmegaConf.to_container(config.data_loading, resolve=True)
    all_file_paths, all_labels, all_class_names_union, all_boundary_dists = load_data(**dl)

    train_data, valid_data, test_data, _test_clip_sizes, train_dataset_ids = train_test_split(
        all_file_paths, all_labels, all_boundary_dists,
        **OmegaConf.to_container(config.data, resolve=True),
        return_dataset_ids=True,
    )
    # all_labels (nested, pre-split, per source dataset) is returned too --
    # save_run_meta uses it to report total labeled rows per dataset.
    return train_data, valid_data, test_data, all_class_names_union, all_labels, train_dataset_ids


def stratified_subsample_train(train_data, dataset_ids: np.ndarray, n: int, seed: int):
    """Draw ~n samples from the training pool, stratified by (dataset, label).

    Sampling happens independently within every (video, behavior class) cell
    of the training pool, so the resulting subset keeps both the same video
    mix and each video's class balance as the full pool -- not just an
    overall random draw that could, by chance, lean toward one video or drop
    a rare behavior at low fractions.

    Returns (subset, subset_dataset_ids): subset is the usual (paths, labels,
    boundary_dists) 3-tuple; subset_dataset_ids is a parallel array (same
    order) recording which dataset each chosen sample came from, for
    composition reporting.
    """
    paths, labels, boundary_dists = train_data
    total = len(labels)
    if n >= total:
        return train_data, np.array(dataset_ids)

    labels_arr = np.asarray(labels)
    rng = np.random.RandomState(seed)
    frac = n / total

    cells = defaultdict(list)
    for i in range(total):
        cells[(int(dataset_ids[i]), int(labels_arr[i]))].append(i)

    chosen = []
    for cell_idx in cells.values():
        k = len(cell_idx)
        # Floor at 1 so a fraction never silently zeroes out a (video, class)
        # combination that actually has samples -- the achieved n can
        # therefore slightly exceed the nominal target at low fractions when
        # there are many small cells; the caller records the real n used.
        take = min(k, max(1, round(frac * k)))
        chosen.extend(rng.choice(cell_idx, size=take, replace=False).tolist())

    idx = np.array(sorted(chosen))
    sub_paths = [paths[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    sub_bd = [boundary_dists[i] for i in idx]
    return (sub_paths, sub_labels, sub_bd), np.asarray(dataset_ids)[idx]


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
# Per-epoch trajectory logging
# ---------------------------------------------------------------------------

class EpochMetricsWriter(Callback):
    """Appends one JSON line per training epoch (train/val loss, balanced
    acc, MCC, LR) to `path`, so a run's trajectory can be inspected or
    replotted later without re-running it."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        def _get(key):
            val = trainer.callback_metrics.get(key)
            return float(val) if val is not None else None

        record = {
            "epoch": trainer.current_epoch,
            "train_loss": _get("train_loss"),
            "train_balanced_acc": _get("train_balanced_acc"),
            "train_mcc": _get("train_mcc"),
            "val_loss": _get("val_loss"),
            "val_balanced_acc": _get("val_balanced_acc"),
            "val_mcc": _get("val_mcc"),
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Train + evaluate one (fraction, repeat) run
# ---------------------------------------------------------------------------

def train_and_evaluate(subset, valid_data, test_data, config, class_names, args, seed: int,
                        epoch_log_path: Path, keep_checkpoint_to: Path | None = None) -> dict:
    """Trains one run and returns a dict with the test-set metrics plus
    enough bookkeeping (epochs actually trained, which epoch was best, the
    class weights used) to reconstruct what happened without re-running it."""
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
        # Monitor val_mcc (not val_loss): on this model, val_loss and val_mcc
        # can diverge after the first couple of epochs (loss creeps up from
        # increasingly confident wrong predictions while MCC keeps
        # improving), so a val_loss-best checkpoint can be measurably worse
        # on the metric this whole benchmark exists to report. See
        # benchmarking/figures/size_learning_curve_test's frac100 run for a
        # concrete example: val_loss was lowest at epoch 0, but val_mcc kept
        # climbing through epoch 4 and was never captured by that checkpoint.
        checkpoint_cb = ModelCheckpoint(dirpath=tmp_dir, monitor="val_mcc", mode="max", save_top_k=1)
        early_stop_cb = EarlyStopping(monitor="val_mcc", patience=patience, mode="max")
        epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_writer_cb = EpochMetricsWriter(epoch_log_path)

        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb, early_stop_cb, metrics_writer_cb],
            max_epochs=max_epochs,
            accelerator=args.accelerator,
            enable_progress_bar=args.progress_bar,
        )
        trainer.fit(model, data_module)

        epochs_trained = trainer.current_epoch
        best_val_mcc = (
            float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else None
        )

        best_path = checkpoint_cb.best_model_path
        epoch_of_best = None
        if best_path:
            # Lightning checkpoints store the epoch they were saved at --
            # read that off before reloading the module from it.
            ckpt_meta = torch.load(best_path, map_location="cpu", weights_only=False)
            epoch_of_best = int(ckpt_meta.get("epoch")) if "epoch" in ckpt_meta else None
            del ckpt_meta
            model = TSMClassifier.load_from_checkpoint(best_path, strict=False)

        if keep_checkpoint_to is not None and best_path:
            keep_checkpoint_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, keep_checkpoint_to)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        data_module.setup(stage="test")
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

        metrics = compute_full_metrics(y_true, y_pred, num_classes, class_names)
        return {
            **metrics,
            "epochs_trained": epochs_trained,
            "epoch_of_best": epoch_of_best,
            "best_val_mcc": best_val_mcc,
            "class_weights": weights.tolist(),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def state_file_for(output_dir: Path) -> Path:
    return output_dir / "run_state.json"


def save_state(state_path: Path, args, results: list[dict]) -> None:
    """Atomically persist args + results-so-far so a killed/crashed run can
    be continued with --resume."""
    state = {"args": vars(args), "results": results}
    tmp_path = state_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)


def load_resume_state(state_path: Path, output_dir: Path):
    """Reconstruct the args Namespace + completed results from a prior run.
    All args other than --output-dir come from the saved state -- resuming
    re-runs the exact same experiment, just skipping finished (fraction,
    repeat) pairs."""
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    saved_args = dict(state["args"])
    saved_args["output_dir"] = str(output_dir)
    saved_args["resume"] = True
    args = argparse.Namespace(**saved_args)
    return args, state["results"]


# ---------------------------------------------------------------------------
# TorchScript export (the reference / best model only)
# ---------------------------------------------------------------------------

def export_best_model_to_torchscript(checkpoint_path: Path, config, frac: int, rep: int) -> Path:
    """Exports the frac100 run's kept checkpoint straight to TorchScript
    alongside its .ckpt, so the benchmark's reference model is immediately
    usable by anything that loads a .pt (the LabelingApp, Bonsai,
    eval_torchscript_model.py) without a separate manual conversion step.

    Runs on CPU. The export includes a self-verification forward pass
    (compares traced vs. reloaded output) through the
    torch.jit.optimize_for_inference'd graph, which on CUDA can try to
    JIT-compile fused kernels via nvrtc -- and this box's torch build needs
    an nvrtc-builtins DLL that isn't installed anywhere on it (same issue
    documented in eval_torchscript_model.py). CPU sidesteps that entirely;
    the resulting .pt still loads and runs fine on CUDA later (a
    TorchScript export isn't device-locked -- see eval_torchscript_model.py
    for how to run one on this box's GPU despite the missing DLL).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tsm_stages = OmegaConf.to_container(config.model, resolve=True).get("tsm_stages")
    output_path = export_model(
        str(checkpoint_path), f"frac{frac}_rep{rep}_best", str(MODELS_DIR),
        device="cpu", tsm_stages=tsm_stages,
    )
    return Path(output_path)


# ---------------------------------------------------------------------------
# Reproducibility metadata (written once per experiment, not per run)
# ---------------------------------------------------------------------------

def get_git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def save_run_meta(output_dir: Path, args, config, class_names: list, all_labels: list) -> None:
    """One-time snapshot of everything needed to say what a run *was*, as
    opposed to results.json's per-run numbers: git commit, package versions,
    the fully-resolved config, class order, and how many labeled rows each
    source dataset contributed."""
    dl = OmegaConf.to_container(config.data_loading, resolve=True)
    frame_dirs = dl.get("frame_dirs", [])
    label_paths = dl.get("label_paths", [])

    meta = {
        "git_commit": get_git_commit(REPO_ROOT),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
        "config_path": str(args.config),
        "config": OmegaConf.to_container(config, resolve=True),
        "class_names": class_names,
        "dataset_entries": [
            {"dataset_id": i, "frame_dir": fd, "label_path": lp, "n_labeled_rows": len(all_labels[i])}
            for i, (fd, lp) in enumerate(zip(frame_dirs, label_paths))
        ],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "base_seed": args.seed,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(args, completed: list[dict]) -> tuple[list[dict], bool]:
    """Returns (results, final_report_sent) -- final_report_sent is True iff
    the frac=100 report (results.json + plots + email) was already sent
    from inside this call, so main() knows whether it still needs to do its
    own fallback report at the end (e.g. when 100% wasn't part of this run's
    fractions, or was already completed and reported in a prior --resume
    invocation)."""
    output_dir = Path(args.output_dir)
    state_path = state_file_for(output_dir)

    config = OmegaConf.load(args.config)
    train_data, valid_data, test_data, class_names, all_labels, dataset_ids = load_experiment_data(config)
    num_classes = len(class_names)
    total_train = len(train_data[1])

    save_run_meta(output_dir, args, config, class_names, all_labels)

    print(f"Train/val/test samples: {total_train}/{len(valid_data[1])}/{len(test_data[1])}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training pool spans {len(all_labels)} source dataset(s): "
          f"{[len(d) for d in all_labels]} labeled rows each")

    fractions = [int(f.strip()) for f in args.fractions.split(",")]

    # Flatten to a single task list up front so the progress bar knows the
    # true total (0%/100% only run once; the rest run --repeats times).
    tasks = []
    for frac in fractions:
        repeats = 1 if frac in (0, 100) else args.repeats
        for rep in range(repeats):
            tasks.append((frac, rep, repeats))

    completed_keys = {(r["fraction"], r["repeat"]) for r in completed}
    pending_tasks = [t for t in tasks if (t[0], t[1]) not in completed_keys]
    n_skipped = len(tasks) - len(pending_tasks)
    if n_skipped:
        print(f"Resuming: {n_skipped}/{len(tasks)} run(s) already completed in {state_path}, skipping.")
    if not pending_tasks:
        print("All planned runs already completed -- nothing to do.")
        return completed, False

    results = list(completed)
    final_report_sent = False
    pbar = tqdm(pending_tasks, desc="Learning curve", unit="run")
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
            metrics = compute_full_metrics(y_true, y_pred, num_classes, class_names)
            n_used = 0
            epoch_log_path = None
            keep_checkpoint_to = None
            run_extra = {
                "epochs_trained": None, "epoch_of_best": None, "best_val_mcc": None,
                "class_weights": None, "samples_per_dataset": None,
                "epoch_metrics_path": None, "checkpoint_path": None, "torchscript_path": None,
            }
        else:
            subset, subset_dataset_ids = stratified_subsample_train(train_data, dataset_ids, n, seed)
            n_used = len(subset[1])
            samples_per_dataset = {
                str(d): int(c) for d, c in zip(*np.unique(subset_dataset_ids, return_counts=True))
            }

            epoch_log_path = output_dir / "epoch_metrics" / f"frac{frac}_rep{rep}.jsonl"
            if epoch_log_path.exists():
                epoch_log_path.unlink()  # fresh log in case a prior crashed attempt left one

            # Keep the 100%-data run's checkpoint as the "reference" model;
            # every other run's checkpoint is discarded to bound disk usage.
            keep_checkpoint_to = (
                output_dir / "checkpoints" / f"frac{frac}_rep{rep}_best.ckpt" if frac == 100 else None
            )

            metrics = train_and_evaluate(
                subset, valid_data, test_data, config, class_names, args, seed,
                epoch_log_path=epoch_log_path, keep_checkpoint_to=keep_checkpoint_to,
            )

            run_extra = {
                "epochs_trained": metrics.pop("epochs_trained"),
                "epoch_of_best": metrics.pop("epoch_of_best"),
                "best_val_mcc": metrics.pop("best_val_mcc"),
                "class_weights": metrics.pop("class_weights"),
                "samples_per_dataset": samples_per_dataset,
                "epoch_metrics_path": str(epoch_log_path),
                "checkpoint_path": str(keep_checkpoint_to) if keep_checkpoint_to else None,
                # Filled in below, after the report is safely out, if frac==100
                # and the (separately risky) TorchScript export succeeds.
                "torchscript_path": None,
            }

        elapsed = time.time() - start
        accuracy, f1 = metrics["accuracy"], metrics["macro_f1"]

        tqdm.write(f"  n_used={n_used}  "
                   f"accuracy={accuracy:.4f}  macro_f1={f1:.4f}  mcc={metrics['mcc']:.4f}  "
                   f"({elapsed:.1f}s wall)")
        pbar.set_postfix(acc=f"{accuracy:.3f}", f1=f"{f1:.3f}")

        results.append({
            "fraction": frac,
            "repeat": rep,
            "seed": seed,
            "n_frames_train": n_used,
            **metrics,
            **run_extra,
            "wall_seconds": elapsed,
        })
        save_state(state_path, args, results)

        if args.notify:
            send_progress_email(args.notify_email, frac, rep, repeats, metrics, n_used, elapsed)

        if frac == 100:
            # Get the report (results.json + every plot, emailed) out the
            # door *before* touching TorchScript export -- that step has
            # already crashed once on this codebase (a torch.jit bug, not
            # anything about this training run), and a fully-trained run's
            # results are worth far more than the .pt file. Export is
            # attempted afterward, still safely wrapped, so a second failure
            # there still can't take the report down with it.
            results_path = output_dir / "results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            x_unit = "seconds" if args.fps > 0 else "frames"
            accuracy_path = output_dir / f"accuracy_vs_{x_unit}.svg"
            f1_path = output_dir / f"f1_vs_{x_unit}.svg"
            mcc_path = output_dir / f"mcc_vs_{x_unit}.svg"
            plot_learning_curve(results, "accuracy", "Accuracy", "Accuracy vs. training data size",
                                 accuracy_path, fps=args.fps)
            plot_learning_curve(results, "macro_f1", "Macro F1", "Macro F1 vs. training data size",
                                 f1_path, fps=args.fps)
            plot_learning_curve(results, "mcc", "MCC", "MCC vs. training data size", mcc_path, fps=args.fps)
            report_plot_paths = [accuracy_path, f1_path, mcc_path]

            if epoch_log_path is not None and epoch_log_path.exists():
                val_metrics_path = output_dir / "val_mcc_and_loss_vs_epoch_frac100.svg"
                plot_val_metrics_vs_epoch(epoch_log_path, val_metrics_path,
                                           epoch_of_best=run_extra["epoch_of_best"])
                report_plot_paths.append(val_metrics_path)

            final_report_sent = True
            if args.notify:
                send_completion_email(args.notify_email, output_dir, report_plot_paths, results)

            if keep_checkpoint_to is not None and keep_checkpoint_to.exists():
                tqdm.write("  Exporting best model to TorchScript...")
                try:
                    torchscript_path = export_best_model_to_torchscript(keep_checkpoint_to, config, frac, rep)
                    tqdm.write(f"  Saved {torchscript_path}")
                    results[-1]["torchscript_path"] = str(torchscript_path)
                    save_state(state_path, args, results)
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)
                except Exception as e:
                    # Same reasoning as above: don't let this take anything
                    # else down. The .ckpt is still on disk either way, so
                    # export can always be retried separately later.
                    tqdm.write(f"  WARNING: TorchScript export failed ({type(e).__name__}: {e}) -- "
                               f"continuing without it. The checkpoint is still at {keep_checkpoint_to}.")

    return results, final_report_sent


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(results: list[dict], metric_key: str, ylabel: str, title: str, out_path: Path,
                         fps: float = 60.0):
    """fps > 0 converts each run's raw n_frames_train into seconds for the
    x-axis (x = n_frames_train / fps); fps <= 0 (e.g. -1) leaves the x-axis
    as raw frame count instead. n_frames_train itself is always what's
    stored in results.json -- this conversion happens at plot time, so the
    same results.json can be replotted in either unit without re-running
    anything."""
    use_seconds = fps is not None and fps > 0
    xlabel = "Training data (seconds)" if use_seconds else "Training data (# frames)"

    fractions = sorted({r["fraction"] for r in results})
    xs_mean, ys_mean, ys_std = [], [], []

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    has_folds = any("fold" in r for r in results)

    for frac in fractions:
        rows = [r for r in results if r["fraction"] == frac]
        xs = [r["n_frames_train"] / fps if use_seconds else r["n_frames_train"] for r in rows]
        ys = [r[metric_key] for r in rows]

        ax.scatter(xs, ys, color=PALETTE[0], alpha=0.18 if has_folds else 0.45, s=28, zorder=2,
                   label="Individual runs" if frac == fractions[0] else None)

        xs_mean.append(float(np.mean(xs)))
        ys_mean.append(float(np.mean(ys)))
        ys_std.append(float(np.std(ys)))

    order = np.argsort(xs_mean)
    xs_mean = np.array(xs_mean)[order]
    ys_mean = np.array(ys_mean)[order]
    ys_std = np.array(ys_std)[order]

    # Retain the aggregate as the visual focus while exposing the temporal
    # fold trajectories that produced it.
    if has_folds:
        for fold_idx, fold in enumerate(sorted({r["fold"] for r in results})):
            fold_xs, fold_ys = [], []
            fold_rows = [r for r in results if r["fold"] == fold]
            for frac in fractions:
                rows = [r for r in fold_rows if r["fraction"] == frac]
                if not rows:
                    continue
                raw_xs = [r["n_frames_train"] / fps if use_seconds else r["n_frames_train"]
                          for r in rows]
                fold_xs.append(float(np.mean(raw_xs)))
                fold_ys.append(float(np.mean([r[metric_key] for r in rows])))
            fold_order = np.argsort(fold_xs)
            label = " → ".join(fold.split("_"))
            ax.plot(np.asarray(fold_xs)[fold_order], np.asarray(fold_ys)[fold_order],
                    color=PALETTE[(fold_idx + 2) % len(PALETTE)], alpha=0.38,
                    marker="o", markersize=3.5, linewidth=1.25, zorder=2,
                    label=f"Fold: {label}")

    ax.errorbar(xs_mean, ys_mean, yerr=ys_std, color=PALETTE[1], marker="o",
                markersize=6, linewidth=2.1, capsize=4, zorder=3, label="Overall mean ± std")

    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, pad=14)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_val_metrics_vs_epoch(epoch_log_path: Path, out_path: Path, epoch_of_best: int | None = None) -> None:
    """Val MCC and val loss across training epochs for one run (the 100%
    run, in practice -- the only one whose per-epoch log survives past its
    own run). Two stacked panels sharing an epoch x-axis, since MCC and loss
    aren't on comparable scales (never a dual y-axis chart for two
    different-unit measures -- see the dataviz "one axis" rule).

    Answers "did this run need more epochs?" at a glance: if val_mcc is
    still climbing at the last epoch plotted, more training likely would
    have helped; if it's flat or falling, it had already converged (or
    started overfitting) within the epochs actually run. The dashed vertical
    line marks the epoch whose checkpoint was actually kept (now selected by
    best val_mcc, not val_loss -- see train_and_evaluate).
    """
    records = []
    with open(epoch_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return

    epochs = [r["epoch"] for r in records]
    val_mcc = [r["val_mcc"] for r in records]
    val_loss = [r["val_loss"] for r in records]

    fig, (ax_mcc, ax_loss) = plt.subplots(2, 1, figsize=(7, 7.5), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    panels = [(ax_mcc, val_mcc, "Val MCC", PALETTE[1]), (ax_loss, val_loss, "Val loss", PALETTE[5])]
    for ax, ys, ylabel, color in panels:
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, ys, color=color, marker="o", markersize=6, linewidth=1.8, zorder=3)
        if epoch_of_best is not None:
            ax.axvline(epoch_of_best, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
        ax.set_ylabel(ylabel, color=INK_PRIMARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRIDLINE)
        ax.spines["bottom"].set_color(GRIDLINE)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)

    ax_mcc.set_title("Val MCC and val loss vs. epoch (100% run)", color=INK_PRIMARY, fontsize=13, pad=14)
    if epoch_of_best is not None:
        ax_mcc.annotate("checkpoint kept", xy=(epoch_of_best, val_mcc[epochs.index(epoch_of_best)]),
                         xytext=(6, 6), textcoords="offset points", color=INK_SECONDARY, fontsize=8)
    ax_loss.set_xlabel("Epoch", color=INK_SECONDARY)
    ax_loss.set_xticks(epochs)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Completion email
# ---------------------------------------------------------------------------

def send_completion_email(to_addr: str, output_dir: Path, plot_paths: list, results: list) -> None:
    """Sends a "run finished" notification via Gmail SMTP once the SVGs are
    saved, with the plots and raw results.json attached directly (not just
    referenced by path -- the recipient is often not at the machine that ran
    the benchmark). Credentials come from GMAIL_ADDRESS / GMAIL_APP_PASSWORD
    env vars (never hardcoded or passed on the CLI) -- generate an App
    Password at https://myaccount.google.com/apppasswords (requires 2FA
    enabled on that Gmail account) and set both before running, e.g.
    (PowerShell):
        $env:GMAIL_ADDRESS = "you@gmail.com"
        $env:GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"
    A missing/failed send is logged but never raises -- losing the
    notification shouldn't discard a multi-hour run's completed results.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        print("Skipping completion email: set GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
              "env vars to enable it (see send_completion_email's docstring).")
        return

    # Everything below -- message construction, attaching files, the SMTP
    # send -- is one try/except block on purpose: a failure anywhere in here
    # (a bad attachment read, a network error, an SMTP auth problem) should
    # never propagate and take down a multi-hour benchmark run. Worst case
    # you just don't get this email.
    try:
        best_100 = [r for r in results if r["fraction"] == 100]
        summary_line = ""
        if best_100:
            r = best_100[0]
            summary_line = (f"\n100% run: accuracy={r['accuracy']:.4f}  "
                             f"macro_f1={r['macro_f1']:.4f}  mcc={r['mcc']:.4f}\n")

        msg = EmailMessage()
        msg["Subject"] = "Size learning curve benchmark finished"
        msg["From"] = sender
        msg["To"] = to_addr
        msg.set_content(
            f"benchmark_size_learning_curve.py finished. Plots and raw results are attached.\n\n"
            f"Output directory: {output_dir.resolve()}\n"
            f"{summary_line}\n"
            f"Plots:\n" + "\n".join(f"  - {p}" for p in plot_paths) + "\n\n"
            f"Raw results: {output_dir.resolve() / 'results.json'}\n"
        )

        results_path = output_dir / "results.json"
        attachments = list(plot_paths) + [results_path]
        for path in attachments:
            path = Path(path)
            if not path.exists():
                print(f"Skipping attachment (not found): {path}")
                continue
            if path.suffix == ".svg":
                maintype, subtype = "image", "svg+xml"
            elif path.suffix == ".json":
                maintype, subtype = "application", "json"
            else:
                maintype, subtype = "application", "octet-stream"
            msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        print(f"Sent completion email to {to_addr} with {len(attachments)} attachment(s)")
    except Exception as e:
        print(f"Failed to send completion email: {type(e).__name__}: {e}")


def send_progress_email(to_addr: str, frac: int, rep: int, repeats: int, metrics: dict,
                         n_used: int, elapsed: float) -> None:
    """Best-effort per-run notification -- sent after every (fraction,
    repeat) finishes, so progress on a long sweep can be tracked without
    watching the terminal. Same env-var gating as send_completion_email; the
    whole body is one try/except for the same reason (a failed/skipped send
    must never take down the run) -- it just doesn't re-print the setup
    instructions every single run.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"Size learning curve: fraction={frac}% repeat={rep + 1}/{repeats} done"
        msg["From"] = sender
        msg["To"] = to_addr
        msg.set_content(
            f"fraction={frac}%  repeat={rep + 1}/{repeats}\n"
            f"n_frames_train={n_used}\n"
            f"accuracy={metrics['accuracy']:.4f}\n"
            f"macro_f1={metrics['macro_f1']:.4f}\n"
            f"mcc={metrics['mcc']:.4f}\n"
            f"wall_seconds={elapsed:.1f}\n"
        )
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        tqdm.write(f"  Sent progress email to {to_addr}")
    except Exception as e:
        tqdm.write(f"  WARNING: progress email failed ({type(e).__name__}: {e}) -- continuing.")


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
    parser.add_argument("--fps", type=float, default=60.0,
                         help="Frame rate assumed for the source video(s), used to convert each "
                              "run's # training frames into seconds for the plot x-axis (default: "
                              "60). Pass -1 to skip the conversion and plot raw frame count instead.")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--resume", action="store_true",
                         help="Continue an incomplete run found in --output-dir, skipping "
                              "(fraction, repeat) runs already saved there. All other args "
                              "are loaded from that saved run, not from the command line.")
    parser.add_argument("--notify-email", type=str, default="braydenk@andrew.cmu.edu",
                         help="Email address to notify when the run finishes and plots are "
                              "saved. Requires GMAIL_ADDRESS/GMAIL_APP_PASSWORD env vars to "
                              "be set (see send_completion_email's docstring); silently "
                              "skipped otherwise.")
    parser.add_argument("--no-notify", dest="notify", action="store_false",
                         help="Disable the completion email entirely.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_file_for(output_dir)

    if args.resume:
        if not state_path.exists():
            print(f"No previous run found at {state_path} -- nothing to resume. "
                  f"Run without --resume to start a new one.")
            sys.exit(1)
        args, completed = load_resume_state(state_path, output_dir)
        print(f"Resuming from {state_path} ({len(completed)} run(s) already completed)")
    else:
        if state_path.exists():
            print(f"Note: {state_path} exists and will be overwritten "
                  f"(pass --resume to continue it instead)")
        completed = []

    email_address_present = bool(os.environ.get("GMAIL_ADDRESS"))
    email_password_present = bool(os.environ.get("GMAIL_APP_PASSWORD"))
    if not args.notify:
        email_status = "DISABLED by --no-notify"
    elif email_address_present and email_password_present:
        email_status = "READY"
    else:
        email_status = "NOT READY -- missing credential variable(s)"
    print("\nEmail notification configuration:")
    print(f"  enabled: {args.notify}")
    print(f"  recipient: {args.notify_email}")
    print(f"  GMAIL_ADDRESS: {'set' if email_address_present else 'MISSING'}")
    print(f"  GMAIL_APP_PASSWORD: {'set' if email_password_present else 'MISSING'}")
    print(f"  status: {email_status}\n")

    results, final_report_sent = run_experiment(args, completed)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {results_path}")

    if final_report_sent:
        # run_experiment already wrote results.json (again, harmlessly, just
        # above), generated every plot, and emailed the report the moment
        # the frac=100 run completed -- before touching TorchScript export.
        # Redoing it here would just duplicate the email.
        print("Final report (plots + email) was already sent when the 100% run completed.")
        return

    x_unit = "seconds" if args.fps > 0 else "frames"
    accuracy_path = output_dir / f"accuracy_vs_{x_unit}.svg"
    f1_path = output_dir / f"f1_vs_{x_unit}.svg"
    mcc_path = output_dir / f"mcc_vs_{x_unit}.svg"
    plot_learning_curve(results, "accuracy", "Accuracy", "Accuracy vs. training data size", accuracy_path, fps=args.fps)
    plot_learning_curve(results, "macro_f1", "Macro F1", "Macro F1 vs. training data size", f1_path, fps=args.fps)
    plot_learning_curve(results, "mcc", "MCC", "MCC vs. training data size", mcc_path, fps=args.fps)

    print(f"Saved {accuracy_path}")
    print(f"Saved {f1_path}")
    print(f"Saved {mcc_path}")

    if args.notify:
        send_completion_email(args.notify_email, output_dir,
                               [accuracy_path, f1_path, mcc_path], results)


if __name__ == "__main__":
    main()

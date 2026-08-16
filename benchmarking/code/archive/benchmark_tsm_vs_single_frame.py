"""Archived study.
Compare three-frame TSM against a CNN that sees only the most recent frame.

Both arms use the same EfficientNet variant, pretrained initialization,
classifier head, training settings, augmentation, samples, class weights, and
paired random seed.  The only intended difference is temporal input:

  * tsm_3frame: the existing TSMClassifier sees [t-2, t-1, t].
  * cnn_latest: the same backbone sees only t.

The split order is deliberately forced to train -> val -> test regardless of
the order stored in the source config.  This benchmark defaults to the five
recording config used by the most recent split-order experiment.

Examples:
    python benchmarking/code/archive/benchmark_tsm_vs_single_frame.py
    python benchmarking/code/archive/benchmark_tsm_vs_single_frame.py --max-epochs 5
    python benchmarking/code/archive/benchmark_tsm_vs_single_frame.py --repeats 3
    python benchmarking/code/archive/benchmark_tsm_vs_single_frame.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import smtplib
import sys
import tempfile
import time
from email.message import EmailMessage
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
from torchmetrics import ConfusionMatrix, MatthewsCorrCoef

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from model import MultiImageDataModule, load_data, train_test_split
from model_tsm import TSMClassifier


DEFAULT_CONFIG = (
    REPO_ROOT / "benchmarking" / "configs" / "gui_global_split_order_folds.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarking" / "figures" / "tsm_vs_single_frame"
)
MODEL_ORDER = ("tsm_3frame", "cnn_latest")
DISPLAY_NAMES = {
    "tsm_3frame": "TSM (3 frames)",
    "cnn_latest": "CNN (latest frame)",
}


class LatestFrameCNN(TSMClassifier):
    """TSMClassifier training behavior with a strictly single-frame forward.

    The parent constructs the same EfficientNet blocks and classifier head as
    the TSM arm. ``num_frames=1`` and ``shift_fraction=0`` make its wrappers
    identities, and this forward method sends only x[:, -1] through the
    backbone. Earlier frames therefore cannot affect activations, gradients,
    or BatchNorm statistics.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, T, C, H, W), got shape {tuple(x.shape)}")
        x = x[:, -1, :, :, :]
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class EpochMetricsWriter(Callback):
    """Write one JSON record per epoch for later trajectory inspection."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        def value(name):
            metric = trainer.callback_metrics.get(name)
            return float(metric) if metric is not None else None

        record = {
            "epoch": int(trainer.current_epoch),
            "train_loss": value("train_loss"),
            "train_balanced_acc": value("train_balanced_acc"),
            "train_mcc": value("train_mcc"),
            "val_loss": value("val_loss"),
            "val_balanced_acc": value("val_balanced_acc"),
            "val_mcc": value("val_mcc"),
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def class_weights_for(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split has no samples for class indices {missing}")
    return torch.tensor(counts.sum() / counts, dtype=torch.float32)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> dict:
    accuracy = float(np.mean(y_true == y_pred))
    recalls = []
    f1s = []
    per_class = {}

    for class_id, name in enumerate(class_names):
        true_class = y_true == class_id
        pred_class = y_pred == class_id
        tp = int(np.sum(true_class & pred_class))
        fp = int(np.sum(~true_class & pred_class))
        fn = int(np.sum(true_class & ~pred_class))
        support = int(np.sum(true_class))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1s.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    task = "binary" if num_classes <= 2 else "multiclass"
    true_tensor = torch.as_tensor(y_true, dtype=torch.long)
    pred_tensor = torch.as_tensor(y_pred, dtype=torch.long)
    mcc = float(
        MatthewsCorrCoef(task=task, num_classes=num_classes)(
            pred_tensor, true_tensor
        )
    )
    confusion = ConfusionMatrix(task=task, num_classes=num_classes)(
        pred_tensor, true_tensor
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluation_device(accelerator: str) -> torch.device:
    accelerator = accelerator.lower()
    if accelerator in {"auto", "gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    model_name: str,
    config,
    class_names: list[str],
    weights: torch.Tensor,
    max_epochs: int,
):
    model_args = OmegaConf.to_container(config.model, resolve=True)
    boundary_smoothing = (
        OmegaConf.to_container(config.boundary_smoothing, resolve=True)
        if config.get("boundary_smoothing") is not None
        else None
    )

    if model_name == "cnn_latest":
        # Keep the same wrapper locations and trainable parameters but turn
        # shifting off. LatestFrameCNN.forward then evaluates only frame t.
        model_args["num_frames"] = 1
        model_args["shift_fraction"] = 0.0
        model_class = LatestFrameCNN
    elif model_name == "tsm_3frame":
        model_class = TSMClassifier
    else:
        raise ValueError(f"Unknown model arm: {model_name}")

    return model_class(
        **model_args,
        max_epochs=max_epochs,
        weights=weights,
        class_names=class_names,
        boundary_smoothing=boundary_smoothing,
    )


def predict(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    predictions = []
    targets = []
    with torch.inference_mode():
        for images, labels, _boundary_distances in dataloader:
            logits = model(images.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    model.to("cpu")
    return np.concatenate(targets), np.concatenate(predictions)


def train_one(
    model_name: str,
    repeat: int,
    seed: int,
    config,
    class_names: list[str],
    train_data,
    val_data,
    test_data,
    args,
    output_dir: Path,
) -> dict:
    L.seed_everything(seed, workers=True)
    max_epochs = args.max_epochs or int(config.max_epochs)
    patience = args.patience or int(config.callbacks.early_stopping_patience)
    weights = class_weights_for(train_data[1], len(class_names))

    data_module = MultiImageDataModule(
        train_data,
        val_data,
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(model_name, config, class_names, weights, max_epochs)

    temporary_checkpoint_dir = Path(tempfile.mkdtemp(prefix=f"{model_name}_"))
    epoch_log = output_dir / "epoch_metrics" / f"{model_name}_repeat{repeat}.jsonl"
    if epoch_log.exists():
        epoch_log.unlink()

    checkpoint_callback = ModelCheckpoint(
        dirpath=temporary_checkpoint_dir,
        monitor="val_mcc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_mcc:.4f}",
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_mcc", patience=patience, mode="max"),
        EpochMetricsWriter(epoch_log),
    ]

    started = time.perf_counter()
    try:
        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=True,
            callbacks=callbacks,
            max_epochs=max_epochs,
            accelerator=args.accelerator,
            devices=1,
            deterministic=args.deterministic,
            enable_progress_bar=args.progress_bar,
        )
        trainer.fit(model, data_module)

        best_path = checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError(f"{model_name} did not produce a best checkpoint")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        best_epoch = int(checkpoint.get("epoch", -1))

        data_module.setup(stage="test")
        y_true, y_pred = predict(
            model,
            data_module.test_dataloader(),
            evaluation_device(args.accelerator),
        )
        metrics = compute_metrics(y_true, y_pred, len(class_names), class_names)

        if args.keep_checkpoints:
            destination = (
                output_dir / "checkpoints" / f"{model_name}_repeat{repeat}_best.ckpt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, destination)

        return {
            "model": model_name,
            "display_name": DISPLAY_NAMES[model_name],
            "repeat": repeat,
            "seed": seed,
            "epochs_trained": int(trainer.current_epoch),
            "best_epoch": best_epoch,
            "best_val_mcc": float(checkpoint_callback.best_model_score),
            "wall_seconds": time.perf_counter() - started,
            "class_weights": weights.tolist(),
            **metrics,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(temporary_checkpoint_dir, ignore_errors=True)


def summarize(results: list[dict]) -> dict:
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "mcc")
    summary = {"models": {}, "paired_delta_tsm_minus_cnn": {}}

    for model_name in MODEL_ORDER:
        rows = [row for row in results if row["model"] == model_name]
        summary["models"][model_name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "std": float(np.std([row[metric] for row in rows], ddof=1))
                if len(rows) > 1
                else 0.0,
            }
            for metric in metric_names
        }

    by_key = {(row["model"], row["repeat"]): row for row in results}
    paired_repeats = sorted(
        set(row["repeat"] for row in results if row["model"] == "tsm_3frame")
        & set(row["repeat"] for row in results if row["model"] == "cnn_latest")
    )
    for metric in metric_names:
        deltas = [
            by_key[("tsm_3frame", repeat)][metric]
            - by_key[("cnn_latest", repeat)][metric]
            for repeat in paired_repeats
        ]
        summary["paired_delta_tsm_minus_cnn"][metric] = {
            "values": deltas,
            "mean": float(np.mean(deltas)) if deltas else None,
            "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        }
    return summary


def write_csv(results: list[dict], path: Path) -> None:
    columns = [
        "model",
        "repeat",
        "seed",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "best_val_mcc",
        "best_epoch",
        "epochs_trained",
        "wall_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: list[dict], path: Path) -> None:
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("mcc", "MCC"),
    ]
    colors = {"tsm_3frame": "#2a78d6", "cnn_latest": "#e34948"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4), dpi=160)
    rng = np.random.default_rng(0)

    for axis, (metric, title) in zip(axes, metrics):
        for x, model_name in enumerate(MODEL_ORDER):
            values = np.array(
                [row[metric] for row in results if row["model"] == model_name]
            )
            axis.bar(
                x,
                values.mean(),
                color=colors[model_name],
                width=0.64,
                alpha=0.82,
            )
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            axis.scatter(
                np.full(len(values), x) + jitter,
                values,
                color="#111111",
                s=22,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels([DISPLAY_NAMES[name] for name in MODEL_ORDER], rotation=15)
        axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        axis.set_axisbelow(True)

    fig.suptitle("Three-frame TSM vs. most-recent-frame CNN")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def send_progress_email(to_addr: str, result: dict, repeats: int) -> None:
    """Best-effort notification after each model arm finishes."""
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        return

    try:
        message = EmailMessage()
        message["Subject"] = (
            f"TSM comparison: {result['display_name']} "
            f"repeat {result['repeat'] + 1}/{repeats} finished"
        )
        message["From"] = sender
        message["To"] = to_addr
        message.set_content(
            f"Model: {result['display_name']}\n"
            f"Repeat: {result['repeat'] + 1}/{repeats}\n"
            f"Seed: {result['seed']}\n"
            f"Balanced accuracy: {result['balanced_accuracy']:.4f}\n"
            f"Macro F1: {result['macro_f1']:.4f}\n"
            f"MCC: {result['mcc']:.4f}\n"
            f"Best validation MCC: {result['best_val_mcc']:.4f}\n"
            f"Best epoch: {result['best_epoch']}\n"
            f"Wall seconds: {result['wall_seconds']:.1f}\n"
        )
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(message)
        print(f"Sent progress email to {to_addr}")
    except Exception as error:
        print(
            f"WARNING: progress email failed "
            f"({type(error).__name__}: {error}) -- continuing."
        )


def send_completion_email(
    to_addr: str, output_dir: Path, results: list[dict], summary: dict
) -> None:
    """Best-effort final email with the comparison results attached."""
    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        print(
            "Skipping completion email: set GMAIL_ADDRESS and "
            "GMAIL_APP_PASSWORD to enable it."
        )
        return

    try:
        delta = summary["paired_delta_tsm_minus_cnn"]
        message = EmailMessage()
        message["Subject"] = "TSM vs latest-frame CNN benchmark finished"
        message["From"] = sender
        message["To"] = to_addr
        message.set_content(
            "benchmark_tsm_vs_single_frame.py finished.\n\n"
            "Mean paired difference (TSM - latest-frame CNN):\n"
            f"  Balanced accuracy: {delta['balanced_accuracy']['mean']:+.4f}\n"
            f"  Macro F1: {delta['macro_f1']['mean']:+.4f}\n"
            f"  MCC: {delta['mcc']['mean']:+.4f}\n\n"
            f"Completed model runs: {len(results)}\n"
            f"Output directory: {output_dir.resolve()}\n\n"
            "The plot, summary, CSV, and full JSON results are attached."
        )

        attachments = [
            output_dir / "comparison.svg",
            output_dir / "summary.json",
            output_dir / "results.csv",
            output_dir / "results.json",
        ]
        attached_count = 0
        for path in attachments:
            if not path.exists():
                print(f"Skipping email attachment (not found): {path}")
                continue
            if path.suffix == ".svg":
                maintype, subtype = "image", "svg+xml"
            elif path.suffix == ".json":
                maintype, subtype = "application", "json"
            elif path.suffix == ".csv":
                maintype, subtype = "text", "csv"
            else:
                maintype, subtype = "application", "octet-stream"
            message.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
            attached_count += 1

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(message)
        print(
            f"Sent completion email to {to_addr} with "
            f"{attached_count} attachment(s)"
        )
    except Exception as error:
        print(
            f"Failed to send completion email: "
            f"{type(error).__name__}: {error}"
        )


def save_outputs(results: list[dict], output_dir: Path) -> None:
    results = sorted(results, key=lambda row: (row["repeat"], MODEL_ORDER.index(row["model"])))
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_csv(results, output_dir / "results.csv")
    if all(any(row["model"] == name for row in results) for name in MODEL_ORDER):
        summary = summarize(results)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        plot_results(results, output_dir / "comparison.svg")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Paired seeds to run for both models (default: 1; use 3+ for uncertainty)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--notify-email",
        default="braydenk@andrew.cmu.edu",
        help="Completion/progress recipient; Gmail credentials come from "
        "GMAIL_ADDRESS and GMAIL_APP_PASSWORD",
    )
    parser.add_argument(
        "--no-notify",
        dest="notify",
        action="store_false",
        help="Disable progress and completion emails",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model/repeat pairs already present in results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    torch.set_float32_matmul_precision("medium")

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(config_path)

    # This is the split requested for this comparison. Remove any per-dataset
    # override so the global order is guaranteed to apply to every recording.
    config.data.split_order = ["train", "val", "test"]
    if config.data.get("split_order_by_dataset") is not None:
        del config.data["split_order_by_dataset"]

    loading_args = OmegaConf.to_container(config.data_loading, resolve=True)
    all_paths, all_labels, class_names, all_boundary_distances = load_data(
        **loading_args
    )
    split_args = OmegaConf.to_container(config.data, resolve=True)
    train_data, val_data, test_data, _test_clip_sizes = train_test_split(
        all_paths, all_labels, all_boundary_distances, **split_args
    )

    run_metadata = {
        "config_path": str(config_path),
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "forced_split_order": ["train", "val", "test"],
        "class_names": class_names,
        "split_samples": {
            "train": len(train_data[1]),
            "val": len(val_data[1]),
            "test": len(test_data[1]),
        },
        "comparison": {
            "tsm_3frame": "Frames [t-2, t-1, t] with configured TSM shifts",
            "cnn_latest": "Only frame t through the otherwise matched EfficientNet",
        },
        "arguments": vars(args),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    results_path = output_dir / "results.json"
    results = []
    if args.resume and results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"Resuming with {len(results)} completed run(s) from {results_path}")
    elif args.resume:
        print(f"No prior results at {results_path}; starting a new benchmark")

    completed = {(row["model"], int(row["repeat"])) for row in results}
    print(f"Forced split order: train -> val -> test")
    print(
        f"Samples: train={len(train_data[1]):,}, val={len(val_data[1]):,}, "
        f"test={len(test_data[1]):,}"
    )
    print(f"Classes: {class_names}")
    if not args.notify:
        email_status = "disabled"
    elif os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD"):
        email_status = f"ready; recipient={args.notify_email}"
    else:
        email_status = "missing GMAIL_ADDRESS and/or GMAIL_APP_PASSWORD"
    print(f"Email notifications: {email_status}")

    for repeat in range(args.repeats):
        paired_seed = args.seed + repeat
        for model_name in MODEL_ORDER:
            key = (model_name, repeat)
            if key in completed:
                print(f"Skipping completed {model_name}, repeat {repeat}")
                continue
            print(
                f"\n=== {DISPLAY_NAMES[model_name]} | repeat {repeat + 1}/"
                f"{args.repeats} | seed {paired_seed} ===",
                flush=True,
            )
            result = train_one(
                model_name,
                repeat,
                paired_seed,
                config,
                class_names,
                train_data,
                val_data,
                test_data,
                args,
                output_dir,
            )
            results.append(result)
            save_outputs(results, output_dir)
            print(
                f"Test: balanced_acc={result['balanced_accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}, mcc={result['mcc']:.4f}"
            )
            if args.notify:
                send_progress_email(args.notify_email, result, args.repeats)

    save_outputs(results, output_dir)
    final_summary = summarize(results)
    delta = final_summary["paired_delta_tsm_minus_cnn"]
    print("\n=== Paired TSM - CNN result ===")
    for metric in ("balanced_accuracy", "macro_f1", "mcc"):
        value = delta[metric]["mean"]
        print(f"{metric}: {value:+.4f}")
    print(f"\nSaved benchmark outputs to {output_dir}")
    if args.notify:
        send_completion_email(args.notify_email, output_dir, results, final_summary)


if __name__ == "__main__":
    main()

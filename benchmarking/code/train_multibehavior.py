"""Train one TSM model jointly over every behavior class."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from binary_behavior_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_CONFIG,
    load_multibehavior_split,
    write_json,
)
from behavior_benchmark_callbacks import ValidationMetricsPrinter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from model import MultiImageDataModule
from model_tsm import TSMClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-patience", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    loaded = load_multibehavior_split(
        args.data_root.resolve(), shuffle=args.shuffle_split, random_seed=args.seed,
        split_config=args.split_config.resolve() if args.split_config else None,
    )
    output_dir = args.output_root.resolve() / "multibehavioral"
    checkpoint_path = output_dir / "model.ckpt"
    if checkpoint_path.is_file() and not args.force:
        print(f"Checkpoint already exists; skipping training: {checkpoint_path}")
        return

    counts = np.bincount(loaded["train_data"][1], minlength=len(loaded["class_names"]))
    if np.any(counts == 0):
        missing = [loaded["class_names"][i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"Training split lacks classes: {missing}")
    weights = torch.tensor(counts.sum() / counts, dtype=torch.float32)
    L.seed_everything(args.seed, workers=True)
    module = MultiImageDataModule(
        loaded["train_data"], loaded["val_data"], loaded["test_data"],
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = TSMClassifier(
        num_frames=3, variant="b0", shift_fraction=0.125, pretrained=True,
        tsm_stages=[1, 2, 3, 4, 5, 6, 7], dropout=0.2, lr=args.learning_rate,
        weight_decay=1e-3, max_epochs=args.max_epochs,
        lr_scheduler="plateau", lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_patience=args.lr_plateau_patience, lr_min=args.min_learning_rate,
        temporal_pool="last",
        weights=weights, class_names=loaded["class_names"],
        boundary_smoothing={"enabled": True, "epsilon": 0.3, "window": 5},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="multibehavioral_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=temporary_dir, monitor="val_mcc", mode="max", save_top_k=1,
        filename="best-{epoch:02d}-{val_mcc:.4f}",
    )
    started = time.perf_counter()
    try:
        trainer = L.Trainer(
            logger=False,
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="val_mcc", mode="max", patience=args.patience),
                ValidationMetricsPrinter(),
            ],
            max_epochs=args.max_epochs, accelerator=args.accelerator, devices=1,
            deterministic=True, enable_progress_bar=not args.no_progress_bar,
        )
        trainer.fit(model, module)
        if not checkpoint_callback.best_model_path:
            raise RuntimeError("Training completed without producing a checkpoint")
        shutil.copy2(checkpoint_callback.best_model_path, checkpoint_path)
        checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu", weights_only=False)
        write_json(output_dir / "train_metadata.json", {
            "mode": "multibehavioral", "checkpoint": str(checkpoint_path),
            "data_root": str(args.data_root.resolve()), "class_names": loaded["class_names"],
            "split": loaded["split"], "n_train": len(loaded["train_data"][1]),
            "n_val": len(loaded["val_data"][1]), "n_test": len(loaded["test_data"][1]),
            "train_class_counts": counts.tolist(), "class_weights": weights.tolist(),
            "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
            "learning_rate": args.learning_rate, "lr_scheduler": "plateau",
            "lr_plateau_factor": args.lr_plateau_factor,
            "lr_plateau_patience": args.lr_plateau_patience,
            "min_learning_rate": args.min_learning_rate,
            "epochs_trained": int(trainer.current_epoch) + 1,
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "best_val_mcc": float(checkpoint_callback.best_model_score),
            "wall_seconds": time.perf_counter() - started,
        })
        print(f"Saved best checkpoint: {checkpoint_path}")
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

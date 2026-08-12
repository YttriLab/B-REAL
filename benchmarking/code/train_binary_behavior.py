"""Train one three-frame TSM checkpoint for one binarized behavior."""

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
    discover_sources,
    load_binary_split,
    slugify,
    write_json,
)
from behavior_benchmark_callbacks import ValidationMetricsPrinter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from model import MultiImageDataModule
from model_tsm import TSMClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior")
    parser.add_argument("--list-behaviors", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing checkpoint")
    parser.add_argument("--shuffle-split", action="store_true", help="Use bout-stratified rather than chronological splits")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()
    if not args.list_behaviors and not args.behavior:
        parser.error("--behavior is required unless --list-behaviors is used")
    return args


def main() -> None:
    args = parse_args()
    _, _, available = discover_sources(args.data_root.resolve())
    if args.list_behaviors:
        print("\n".join(available))
        return

    loaded = load_binary_split(
        args.data_root.resolve(), args.behavior, shuffle=args.shuffle_split, random_seed=args.seed
    )
    behavior = loaded["behavior"]
    output_dir = args.output_root.resolve() / slugify(behavior)
    checkpoint_path = output_dir / "model.ckpt"
    metadata_path = output_dir / "train_metadata.json"
    if checkpoint_path.is_file() and not args.force:
        print(f"Checkpoint already exists; skipping training: {checkpoint_path}", flush=True)
        return

    train_labels = np.asarray(loaded["train_data"][1], dtype=np.int64)
    counts = np.bincount(train_labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Training split lacks a class for {behavior!r}: counts={counts.tolist()}")
    weights = torch.tensor(counts.sum() / counts, dtype=torch.float32)

    L.seed_everything(args.seed, workers=True)
    module = MultiImageDataModule(
        loaded["train_data"], loaded["val_data"], loaded["test_data"],
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = TSMClassifier(
        num_frames=3,
        variant="b0",
        shift_fraction=0.125,
        pretrained=True,
        tsm_stages=[1, 2, 3, 4, 5, 6, 7],
        dropout=0.2,
        lr=1e-4,
        weight_decay=1e-3,
        max_epochs=args.max_epochs,
        lr_scheduler_t_max=args.max_epochs,
        temporal_pool="last",
        weights=weights,
        class_names=loaded["class_names"],
        boundary_smoothing={"enabled": True, "epsilon": 0.3, "window": 5},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"binary_{slugify(behavior)}_"))
    callback = ModelCheckpoint(
        dirpath=temporary_dir, monitor="val_mcc", mode="max", save_top_k=1,
        filename="best-{epoch:02d}-{val_mcc:.4f}",
    )
    started = time.perf_counter()
    try:
        trainer = L.Trainer(
            logger=False,
            callbacks=[
                callback,
                EarlyStopping(monitor="val_mcc", mode="max", patience=args.patience),
                ValidationMetricsPrinter(),
            ],
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=1,
            deterministic=True,
            enable_progress_bar=not args.no_progress_bar,
        )
        trainer.fit(model, module)
        if not callback.best_model_path:
            raise RuntimeError("Training completed without producing a checkpoint")
        shutil.copy2(callback.best_model_path, checkpoint_path)
        checkpoint = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
        write_json(metadata_path, {
            "behavior": behavior,
            "checkpoint": str(checkpoint_path),
            "data_root": str(args.data_root.resolve()),
            "frame_dirs": [str(path) for path in loaded["frame_dirs"]],
            "label_paths": [str(path) for path in loaded["label_paths"]],
            "class_names": loaded["class_names"],
            "split": loaded["split"],
            "n_train": len(loaded["train_data"][1]),
            "n_val": len(loaded["val_data"][1]),
            "n_test": len(loaded["test_data"][1]),
            "train_class_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "epochs_trained": int(trainer.current_epoch) + 1,
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "best_val_mcc": float(callback.best_model_score),
            "wall_seconds": time.perf_counter() - started,
        })
        print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

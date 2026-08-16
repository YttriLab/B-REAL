"""Callbacks shared by binary and multibehavioral behavior benchmarks."""

from __future__ import annotations

import time

from lightning.pytorch.callbacks import Callback


class ValidationMetricsPrinter(Callback):
    """Print one stable, grep-friendly validation summary after each epoch."""

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        def format_metric(name: str) -> str:
            value = trainer.callback_metrics.get(name)
            return "n/a" if value is None else f"{float(value):.6f}"

        print(
            f"Epoch {trainer.current_epoch + 1:03d} validation: "
            f"val_loss={format_metric('val_loss')} "
            f"val_balanced_acc={format_metric('val_balanced_acc')} "
            f"val_mcc={format_metric('val_mcc')}",
            flush=True,
        )


class TrainingProgressPrinter(Callback):
    """Write append-only batch progress suitable for nohup and redirected logs."""

    def __init__(self, every_n_batches: int = 50):
        super().__init__()
        self.every_n_batches = max(1, every_n_batches)
        self.epoch_started = 0.0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_started = time.perf_counter()
        print(
            f"Epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs:03d} training started",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        total = trainer.num_training_batches
        completed = batch_idx + 1
        if completed != total and completed % self.every_n_batches:
            return
        elapsed = time.perf_counter() - self.epoch_started
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (total - completed) / rate if rate > 0 else 0.0
        loss = trainer.callback_metrics.get("train_loss")
        loss_text = "n/a" if loss is None else f"{float(loss):.6f}"
        print(
            f"Epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs:03d} training: "
            f"batch {completed}/{total} ({100 * completed / total:.1f}%) "
            f"train_loss={loss_text} elapsed={elapsed:.0f}s eta={remaining:.0f}s",
            flush=True,
        )

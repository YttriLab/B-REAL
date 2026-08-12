"""Callbacks shared by binary and multibehavioral behavior benchmarks."""

from __future__ import annotations

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

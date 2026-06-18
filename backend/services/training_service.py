"""
Training service: manages model training as a subprocess with live metrics streaming.

Training runs in a subprocess to isolate CUDA memory and allow clean termination.
Metrics are written to a JSON-lines file that the SSE endpoint tails.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        proc = task.get("process")
        if proc and proc.poll() is None:
            proc.terminate()


# ── Training subprocess script ─────────────────────────────────────────────

TRAIN_SCRIPT = r'''
"""Training subprocess: loads config, trains model, writes JSON-lines metrics."""
import sys
import os
import json
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from omegaconf import OmegaConf

# Ensure project root is importable
sys.path.insert(0, os.environ["PROJECT_ROOT"])

from model import load_data, MultiImageDataModule, train_test_split
from model_tsm import TSMClassifier


def _emit(metrics_path, payload):
    """Append one JSON-lines event to the metrics file (tailed by the SSE endpoint)."""
    with open(metrics_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


class MetricsWriter(Callback):
    """Write per-epoch metrics to a JSON-lines file for the SSE endpoint."""

    def __init__(self, metrics_path: str, max_epochs: int, patience: int):
        super().__init__()
        self.metrics_path = metrics_path
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_val_loss = float("inf")
        self.stagnation = 0  # epochs since val_loss last improved (early-stopping counter)

    def on_train_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", 0))
        # Track best val_loss and how many epochs we've gone without improving it
        # (mirrors EarlyStopping's wait_count so the UI can show progress to stop).
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.stagnation = 0
        else:
            self.stagnation += 1

        metrics = {
            "epoch": trainer.current_epoch,
            "total_epochs": self.max_epochs,
            "train_loss": float(trainer.callback_metrics.get("train_loss", 0)),
            "train_acc": float(trainer.callback_metrics.get("train_balanced_acc", 0)),
            "val_loss": val_loss,
            "val_acc": float(trainer.callback_metrics.get("val_balanced_acc", 0)),
            "val_mcc": float(trainer.callback_metrics.get("val_mcc", 0)),
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "best_val_loss": self.best_val_loss,
            "stagnation_epochs": self.stagnation,
            "patience": self.patience,
            "status": "training",
        }
        _emit(self.metrics_path, metrics)


def main():
    config_path = sys.argv[1]
    metrics_path = sys.argv[2]

    def info(message, **extra):
        """Surface a human-readable startup/progress line to the UI via the SSE stream."""
        print(message, flush=True)
        _emit(metrics_path, {"status": "starting", "message": message, **extra})

    config = OmegaConf.load(config_path)
    torch.set_float32_matmul_precision("medium")

    dl = OmegaConf.to_container(config.data_loading, resolve=True)
    accel = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {accel.upper()}"
         + (f" ({torch.cuda.get_device_name(0)})" if accel == "cuda" else ""))
    info(f"Loading data from {len(dl['frame_dirs'])} source(s)...")

    # Load data
    all_file_paths, all_labels, all_class_names_union, all_boundary_dists = load_data(**dl)

    total_samples = sum(len(d) for d in all_labels)
    flat = [lbl for d in all_labels for lbl in d]
    dist = {all_class_names_union[i]: int(c)
            for i, c in zip(*np.unique(flat, return_counts=True))} if flat else {}
    info(f"Loaded {total_samples} samples across {len(all_labels)} dataset(s).")
    info(f"Classes ({len(all_class_names_union)}): {', '.join(all_class_names_union)}")
    info(f"Class distribution: {dist}")
    if total_samples == 0:
        raise RuntimeError(
            "No training samples found. Have you run Preprocess so the labeled "
            "frames exist on disk, and do the frame indices match the labels?"
        )

    train_data, valid_data, test_data, test_clip_sizes = train_test_split(
        all_file_paths, all_labels, all_boundary_dists,
        **OmegaConf.to_container(config.data, resolve=True)
    )
    info(f"Split: {len(train_data[1])} train / {len(valid_data[1])} val / {len(test_data[1])} test")

    # Class weights
    class_ids, counts = torch.unique(torch.tensor(train_data[1]), return_counts=True)
    weights = counts.sum() / counts

    info(f"Building model (variant {config.model.get('variant', 'b0')}, "
         f"max {config.max_epochs} epochs)...")

    data_module = MultiImageDataModule(train_data, valid_data, test_data)

    boundary_smoothing = (
        OmegaConf.to_container(config.boundary_smoothing, resolve=True)
        if "boundary_smoothing" in config else None
    )

    model = TSMClassifier(
        **OmegaConf.to_container(config.model, resolve=True),
        max_epochs=config.max_epochs,
        weights=weights,
        class_names=all_class_names_union,
        boundary_smoothing=boundary_smoothing,
    )
    model.test_clip_sizes = test_clip_sizes

    # Callbacks
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss", mode="min", save_top_k=1
    )
    patience = config.callbacks.early_stopping_patience
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )
    metrics_cb = MetricsWriter(metrics_path, config.max_epochs, patience)

    # Logger (optional wandb)
    logger = None
    if config.get("use_wandb", False):
        from lightning.pytorch.loggers.wandb import WandbLogger
        wandb_kwargs = dict(
            project=config.get("project_name", "realtime-asoid"),
            name=config.get("run_name", "training"),
            config=OmegaConf.to_container(config, resolve=True),
        )
        logger = WandbLogger(**wandb_kwargs)

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, metrics_cb],
        max_epochs=config.max_epochs,
        accelerator="auto",
    )

    info(f"Starting training (early stopping after {patience} stagnant epochs)...")
    trainer.fit(model, data_module)

    # Write completion
    completion = {
        "status": "completed",
        "best_checkpoint": checkpoint_cb.best_model_path,
        "best_val_loss": float(checkpoint_cb.best_model_score or 0),
        "epoch": trainer.current_epoch,
        "total_epochs": config.max_epochs,
    }
    with open(metrics_path, "a") as f:
        f.write(json.dumps(completion) + "\n")

    if logger and hasattr(logger, "experiment"):
        logger.experiment.finish()


if __name__ == "__main__":
    main()
'''


# ── Service functions ──────────────────────────────────────────────────────

def start_training(config: dict) -> str:
    """
    Launch training in a subprocess.

    Args:
        config: Full training configuration dict (same structure as config_tsm.yaml)

    Returns:
        task_id for status polling.
    """
    task_id = str(uuid.uuid4())

    # Write config to temp file
    tmp_dir = tempfile.mkdtemp(prefix="asoid_train_")
    config_path = os.path.join(tmp_dir, "config.yaml")
    metrics_path = os.path.join(tmp_dir, "metrics.jsonl")
    script_path = os.path.join(tmp_dir, "train_subprocess.py")

    # Add checkpoint dir to config
    config.setdefault("checkpoint_dir", os.path.join(tmp_dir, "checkpoints"))

    # Extract API key before writing config to disk
    wandb_api_key = config.pop("wandb_api_key", None)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(script_path, "w") as f:
        f.write(TRAIN_SCRIPT)

    # Touch metrics file
    open(metrics_path, "w").close()

    task = {
        "status": "starting",
        "metrics_path": metrics_path,
        "config_path": config_path,
        "tmp_dir": tmp_dir,
        "process": None,
        "latest_metrics": {},
        "error": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            env = os.environ.copy()
            env["PROJECT_ROOT"] = PROJECT_ROOT
            if wandb_api_key:
                env["WANDB_API_KEY"] = wandb_api_key

            proc = subprocess.Popen(
                [sys.executable, script_path, config_path, metrics_path],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            task["process"] = proc
            task["status"] = "running"

            # communicate() drains stdout/stderr concurrently — required to avoid
            # a deadlock where the subprocess blocks on a full OS pipe buffer
            # while we wait() on it.
            stdout_b, stderr_b = proc.communicate()

            if proc.returncode != 0:
                stderr = (stderr_b or b"").decode("utf-8", errors="replace").strip()
                stdout = (stdout_b or b"").decode("utf-8", errors="replace").strip()
                # Some crashes print the useful traceback to stdout (or nothing to
                # stderr at all); fall back so the user never sees a bare "error".
                detail = stderr or stdout or f"Training process exited with code {proc.returncode} (no output)."
                # error must be set before status: the SSE poller breaks as soon
                # as it sees status=="error", so the detail has to be there first.
                task["error"] = detail[-2000:]  # Last 2000 chars
                task["status"] = "error"
            else:
                # Read final metrics line for completion info
                try:
                    with open(metrics_path, "r") as f:
                        lines = f.readlines()
                    if lines:
                        final = json.loads(lines[-1])
                        task["latest_metrics"] = final
                        task["status"] = final.get("status", "completed")
                except Exception:
                    task["status"] = "completed"
        except Exception as e:
            task["error"] = str(e) or repr(e)
            task["status"] = "error"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return task_id


def read_all_metrics(task_id: str) -> list[dict]:
    """Read every JSON-lines event written so far (for ordered SSE streaming)."""
    task = get_task(task_id)
    if not task:
        return []
    metrics_path = task.get("metrics_path")
    if not metrics_path or not os.path.exists(metrics_path):
        return []
    out = []
    try:
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def read_latest_metrics(task_id: str) -> dict:
    """Read the latest metrics from the JSON-lines file."""
    task = get_task(task_id)
    if not task:
        return {}

    metrics_path = task.get("metrics_path")
    if not metrics_path or not os.path.exists(metrics_path):
        return {"status": task["status"], "error": task.get("error")}

    try:
        with open(metrics_path, "r") as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            latest["status"] = task["status"]
            # If the run errored after writing some metrics, the last line is a
            # normal "training" record with no error field — attach the detail so
            # the client doesn't just see a bare "error".
            if task["status"] == "error" and task.get("error"):
                latest["error"] = task["error"]
            return latest
    except Exception:
        pass

    return {"status": task["status"], "error": task.get("error")}


def stop_training(task_id: str) -> bool:
    """Terminate the training subprocess."""
    task = get_task(task_id)
    if not task:
        return False
    proc = task.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        task["status"] = "cancelled"
        return True
    return False


def list_checkpoints(search_dirs: list[str] | None = None) -> list[dict]:
    """
    Scan directories for checkpoint files.
    Returns list of {path, filename, size_mb}.
    """
    if search_dirs is None:
        # Search in common locations
        search_dirs = [PROJECT_ROOT]

    results = []
    for base_dir in search_dirs:
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".ckpt") or f.endswith(".pt"):
                    full_path = os.path.join(root, f)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    results.append({
                        "path": full_path,
                        "filename": f,
                        "size_mb": round(size_mb, 1),
                    })
    return results

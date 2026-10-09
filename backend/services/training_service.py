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
import re
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
        self.best_val_mcc = float("-inf")  # monitored metric (higher is better)
        self.best_val_loss = float("inf")  # kept for display only
        self.stagnation = 0  # epochs since val_mcc last improved (early-stopping counter)

    def on_train_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", 0))
        val_mcc = float(trainer.callback_metrics.get("val_mcc", 0))
        # Stagnation tracks the monitored metric (val_mcc, maximized) with a strict
        # ">" so it mirrors EarlyStopping's wait_count (min_delta=0, mode="max") and
        # the UI's "stagnation X/patience" matches the actual stop trigger.
        if val_mcc > self.best_val_mcc:
            self.best_val_mcc = val_mcc
            self.stagnation = 0
        else:
            self.stagnation += 1
        # Best val_loss is still reported for context (not what selection/stopping use).
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        metrics = {
            "epoch": trainer.current_epoch,
            "total_epochs": self.max_epochs,
            "train_loss": float(trainer.callback_metrics.get("train_loss", 0)),
            "train_acc": float(trainer.callback_metrics.get("train_balanced_acc", 0)),
            "val_loss": val_loss,
            "val_acc": float(trainer.callback_metrics.get("val_balanced_acc", 0)),
            "val_mcc": val_mcc,
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "best_val_mcc": self.best_val_mcc,
            "best_val_loss": self.best_val_loss,
            "stagnation_epochs": self.stagnation,
            "patience": self.patience,
            "status": "training",
        }
        _emit(self.metrics_path, metrics)


LR_FIND_MIN, LR_FIND_MAX, LR_FIND_STEPS = 1e-6, 1e-1, 200


def find_lr(config, model, data_module, metrics_path, info):
    """Lightning's learning-rate range test: LR_FIND_STEPS training steps with
    the learning rate rising exponentially from LR_FIND_MIN to LR_FIND_MAX
    (stopping early once the loss blows up), reporting the smoothed loss curve
    and a suggestion (valley_lr). The model's weights are restored afterwards
    and nothing is saved."""
    from lightning.pytorch.tuner import Tuner
    from backend.services.lr_valley import valley_lr

    info(f"Finding a learning rate: {LR_FIND_STEPS} steps from {LR_FIND_MIN:g} to "
         f"{LR_FIND_MAX:g} (currently {config.model.get('lr')})...")
    trainer = L.Trainer(logger=False, enable_checkpointing=False,
                        max_epochs=config.max_epochs, accelerator="auto")
    finder = Tuner(trainer).lr_find(
        model, datamodule=data_module, min_lr=LR_FIND_MIN, max_lr=LR_FIND_MAX,
        num_training=LR_FIND_STEPS, update_attr=False)
    pairs = [(float(lr), float(l)) for lr, l in zip(finder.results["lr"], finder.results["loss"])
             if np.isfinite(l)]
    lrs, losses = [p[0] for p in pairs], [p[1] for p in pairs]
    suggestion = valley_lr(lrs, losses)
    lightning = finder.suggestion()
    if suggestion is not None:
        info(f"Suggested learning rate: {suggestion:.2e} (fastai valley rule; Lightning's "
             f"steepest-slope pick: {float(lightning or 0):.2e}). Check the curve.")
    else:
        info("No suggestion: the loss never fell over the range tested.")
    _emit(metrics_path, {
        "status": "completed", "lr_find": True, "lrs": lrs, "losses": losses,
        "suggested_lr": suggestion,
        "lightning_lr": float(lightning) if lightning is not None else None,
        "current_lr": float(config.model.get("lr")),
    })


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
    if torch.cuda.is_available():
        # Free memory up front: another job on the GPU is the usual reason a
        # run dies with "CUDA out of memory" partway in.
        try:
            free, total = torch.cuda.mem_get_info(0)
            mem = f", {free / 2**30:.1f} of {total / 2**30:.1f} GB free"
        except Exception:
            mem = ""
        info(f"Device: CUDA ({torch.cuda.get_device_name(0)}{mem})")
    else:
        # Explain WHY we fell back to CPU so the log is self-diagnosing.
        reasons = [f"torch {torch.__version__}", f"torch.version.cuda={torch.version.cuda}"]
        if torch.version.cuda is None:
            reasons.append("this is a CPU-only torch build — reinstall the CUDA wheel")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and cvd.strip() in ("", "-1"):
            reasons.append(f"CUDA_VISIBLE_DEVICES={cvd!r} hides all GPUs")
        try:
            torch.zeros(1).cuda()  # surface the real CUDA init error, if any
        except Exception as e:
            reasons.append(f"cuda init failed: {type(e).__name__}: {e}")
        info("Device: CPU — GPU unavailable (" + "; ".join(reasons) + ")")
    info(f"Loading data from {len(dl['frame_dirs'])} source(s)...")

    # Coverage pre-check: load_data silently drops labeled frames whose PNGs (or
    # clip context) aren't on disk, so report the shortfall in the log. The GUI
    # also blocks on this before starting, but log it here as a safety net.
    try:
        from data_coverage import compute_coverage
        cov = compute_coverage(dl["frame_dirs"], dl["label_paths"], dl.get("frames_per_group", 3))
        info(f"Frame coverage: {cov['total_ready']}/{cov['total_labeled']} "
             f"labeled frames preprocessed and ready for training.")
        if cov["total_missing"] > 0:
            for d in cov["datasets"]:
                if d["error"]:
                    name = os.path.basename(str(d["frame_dir"]).rstrip("/\\")) or d["frame_dir"]
                    info(f"  WARNING [{name}]: {d['error']}")
                elif d["missing"] > 0:
                    name = os.path.basename(str(d["frame_dir"]).rstrip("/\\")) or d["frame_dir"]
                    info(f"  WARNING [{name}]: {d['missing']} of {d['labeled']} labeled frames "
                         f"will be skipped ({d['not_extracted']} not preprocessed, "
                         f"{d['too_early']} below the clip horizon).")
            info("  Re-run Preprocess on those videos to include the missing labels.")
    except Exception as e:
        info(f"Coverage check skipped: {type(e).__name__}: {e}")

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

    dl_cfg = (OmegaConf.to_container(config.data_loader, resolve=True)
              if "data_loader" in config else {})
    num_frames = config.model.get("num_frames", 3)

    info(f"Building model (variant {config.model.get('variant', 'b0')}, "
         f"max {config.max_epochs} epochs)...")
    info(f"Data loader: batch_size={dl_cfg.get('batch_size', 64)}, "
         f"image_size={dl_cfg.get('image_size', 224)}, num_frames={num_frames} "
         f"(effective batch {dl_cfg.get('batch_size', 64) * num_frames} images/step).")

    data_module = MultiImageDataModule(
        train_data, valid_data, test_data, num_frames=num_frames, **dl_cfg
    )

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

    if config.get("find_lr", False):
        find_lr(config, model, data_module, metrics_path, info)
        return

    # Callbacks
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    # Select and early-stop on val_mcc (maximized): Matthews correlation is a
    # single imbalance-robust summary of the confusion matrix, so it tracks
    # decision quality better than the (class-weighted) val_loss, whose minimum
    # tends to fall a few epochs before the best-MCC model as probabilities
    # miscalibrate. Both callbacks monitor the same metric so the saved best
    # checkpoint is the best-MCC one, not the best-loss one.
    # Self-describing checkpoint names: <target>-epoch=NN-val_mcc=0.NNN.ckpt, so a
    # file on disk says what it predicts and how well, instead of Lightning's
    # default epoch=N-step=M. The target is the binarized positive class — union
    # members joined, since the positive class is otherwise a synthetic name like
    # "positive" — or "multiclass".
    binarize_name = dl.get("binarize")
    label_union = dl.get("label_union") or {}
    if binarize_name:
        members = label_union.get(binarize_name)
        target = "+".join(members) if members else binarize_name
    else:
        target = "multiclass"
    # Behavior names are free text (config_tsm.yaml ships "face/head/paw groom"),
    # so strip anything unsafe before it reaches a path, and cap the length.
    target = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(target)).strip("-._") or "model"
    target = target[:40].rstrip("-._+") or "model"
    info(f"Checkpoints: {checkpoint_dir}/{target}-epoch=NN-val_mcc=N.NNN.ckpt")

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=target + "-{epoch:02d}-{val_mcc:.3f}",
        monitor="val_mcc", mode="max", save_top_k=1
    )
    patience = config.callbacks.early_stopping_patience
    early_stop_cb = EarlyStopping(
        monitor="val_mcc",
        patience=patience,
        mode="max",
    )
    metrics_cb = MetricsWriter(metrics_path, config.max_epochs, patience)

    # Logger (optional wandb)
    logger = None
    if config.get("use_wandb", False):
        from lightning.pytorch.loggers.wandb import WandbLogger
        wandb_kwargs = dict(
            project=config.get("project_name", "b-real"),
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

    # Run the test pass so test metrics + per-clip videos get logged (to W&B when
    # enabled). test_step / on_test_epoch_end only execute under trainer.test(),
    # so without this the standalone script's test logging never happened here.
    best_ckpt = checkpoint_cb.best_model_path or None
    try:
        info("Running test pass (logging test metrics / clips)...")
        trainer.test(model, data_module, ckpt_path=best_ckpt or "best")
    except Exception as e:
        # Don't fail a successful training run just because testing/logging did.
        info(f"Test pass skipped: {type(e).__name__}: {e}")

    # Write completion
    # checkpoint_cb monitors val_mcc now, so best_model_score is the best MCC.
    best_mcc = checkpoint_cb.best_model_score
    best_loss = metrics_cb.best_val_loss
    completion = {
        "status": "completed",
        "best_checkpoint": checkpoint_cb.best_model_path,
        "best_val_mcc": float(best_mcc) if best_mcc is not None else None,
        "best_val_loss": float(best_loss) if best_loss != float("inf") else None,
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

def explain_failure(detail: str) -> str:
    """Put a plain-words cause in front of a failed run's output when it's
    recognisable (the traceback itself stays, after it)."""
    if "out of memory" in detail.lower() or "OutOfMemoryError" in detail:
        return ("The GPU ran out of memory. Is another job using it (nvidia-smi)? "
                "Otherwise lower batch_size or image_size.\n\n" + detail)
    return detail


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
    tmp_dir = tempfile.mkdtemp(prefix="breal_train_")
    config_path = os.path.join(tmp_dir, "config.yaml")
    metrics_path = os.path.join(tmp_dir, "metrics.jsonl")
    script_path = os.path.join(tmp_dir, "train_subprocess.py")

    # Add checkpoint dir to config
    config.setdefault("checkpoint_dir", os.path.join(tmp_dir, "checkpoints"))

    # Extract API key before writing config to disk
    wandb_api_key = config.pop("wandb_api_key", None)

    # Always write UTF-8 — the default on Windows is the locale codec (cp1252),
    # which mangles non-ASCII chars in the script (e.g. em dashes) into bytes
    # Python can't decode when it imports the file as UTF-8 source.
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(TRAIN_SCRIPT)

    # Touch metrics file
    open(metrics_path, "w", encoding="utf-8").close()

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
            # Force UTF-8 for the child's source/IO so non-ASCII (em dashes in
            # log messages, unicode in paths) never trips the locale codec.
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
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
                task["error"] = explain_failure(detail[-2000:])  # Last 2000 chars
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

"""Tests for backend.services.training_service."""

import json
import os
import time

import pytest

from backend.services.training_service import (
    get_task,
    list_checkpoints,
    read_latest_metrics,
    start_training,
    stop_training,
)


# ── list_checkpoints ──────────────────────────────────────────────────────

class TestListCheckpoints:
    def test_finds_ckpt_files(self, tmp_path):
        # Create some fake checkpoint files
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch=5.ckpt").write_bytes(b"x" * 1024)
        (ckpt_dir / "model.pt").write_bytes(b"y" * 2048)
        (ckpt_dir / "readme.txt").write_text("not a checkpoint")

        results = list_checkpoints([str(tmp_path)])
        filenames = [r["filename"] for r in results]
        assert "epoch=5.ckpt" in filenames
        assert "model.pt" in filenames
        assert "readme.txt" not in filenames

    def test_returns_size_mb(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        data = b"x" * (1024 * 1024)  # 1 MB
        (ckpt_dir / "big.ckpt").write_bytes(data)

        results = list_checkpoints([str(tmp_path)])
        assert len(results) == 1
        assert results[0]["size_mb"] == 1.0

    def test_empty_directory(self, tmp_path):
        results = list_checkpoints([str(tmp_path)])
        assert results == []

    def test_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "deep.ckpt").write_bytes(b"z" * 100)

        results = list_checkpoints([str(tmp_path)])
        assert len(results) == 1
        assert results[0]["filename"] == "deep.ckpt"


# ── read_latest_metrics ──────────────────────────────────────────────────

class TestReadLatestMetrics:
    def test_nonexistent_task(self):
        result = read_latest_metrics("nonexistent-id")
        assert result == {}

    def test_task_without_metrics_file(self):
        from backend.services.training_service import _tasks
        _tasks["test-task"] = {
            "status": "starting",
            "metrics_path": "/nonexistent/metrics.jsonl",
            "error": None,
        }
        result = read_latest_metrics("test-task")
        assert result["status"] == "starting"

    def test_reads_latest_line(self, tmp_path):
        from backend.services.training_service import _tasks
        metrics_file = tmp_path / "metrics.jsonl"
        lines = [
            json.dumps({"epoch": 0, "train_loss": 1.5}),
            json.dumps({"epoch": 1, "train_loss": 0.8}),
            json.dumps({"epoch": 2, "train_loss": 0.3}),
        ]
        metrics_file.write_text("\n".join(lines) + "\n")

        _tasks["metrics-test"] = {
            "status": "running",
            "metrics_path": str(metrics_file),
            "error": None,
        }
        result = read_latest_metrics("metrics-test")
        assert result["epoch"] == 2
        assert result["train_loss"] == 0.3
        assert result["status"] == "running"


# ── stop_training ────────────────────────────────────────────────────────

class TestStopTraining:
    def test_nonexistent_task(self):
        assert stop_training("nonexistent") is False

    def test_already_finished_task(self):
        from backend.services.training_service import _tasks
        _tasks["done-task"] = {
            "status": "completed",
            "process": None,
        }
        assert stop_training("done-task") is False


# ── start_training ────────────────────────────────────────────────────────

class TestStartTraining:
    def test_returns_task_id(self, tmp_path):
        config = {
            "data_loading": {
                "label_paths": ["/fake/labels.csv"],
                "frame_dirs": ["/fake/frames"],
                "frames_per_group": 3,
            },
            "data": {"train_percent": 0.7, "val_percent": 0.15, "test_percent": 0.15, "shuffle": False},
            "model": {"num_frames": 3, "variant": "b0", "lr": 0.001},
            "max_epochs": 1,
            "callbacks": {"early_stopping_patience": 5},
            "checkpoint_dir": str(tmp_path / "ckpts"),
        }
        task_id = start_training(config)
        assert isinstance(task_id, str)
        task = get_task(task_id)
        assert task is not None
        assert task["status"] in ("starting", "running", "error")

    def test_creates_temp_files(self, tmp_path):
        config = {
            "data_loading": {"label_paths": [], "frame_dirs": [], "frames_per_group": 3},
            "data": {"train_percent": 0.7, "val_percent": 0.15, "test_percent": 0.15, "shuffle": False},
            "model": {"num_frames": 3, "variant": "b0", "lr": 0.001},
            "max_epochs": 1,
            "callbacks": {"early_stopping_patience": 5},
        }
        task_id = start_training(config)
        task = get_task(task_id)
        assert os.path.exists(task["config_path"])
        assert os.path.exists(task["metrics_path"])

    def test_bad_config_results_in_error(self, tmp_path):
        """Training with invalid data paths should fail gracefully."""
        config = {
            "data_loading": {
                "label_paths": ["/totally/fake/path.csv"],
                "frame_dirs": ["/totally/fake/frames"],
                "frames_per_group": 3,
            },
            "data": {"train_percent": 0.7, "val_percent": 0.15, "test_percent": 0.15, "shuffle": False},
            "model": {"num_frames": 3, "variant": "b0", "lr": 0.001},
            "max_epochs": 1,
            "callbacks": {"early_stopping_patience": 5},
            "checkpoint_dir": str(tmp_path / "ckpts"),
        }
        task_id = start_training(config)
        # Wait for subprocess to fail
        for _ in range(100):
            task = get_task(task_id)
            if task["status"] in ("error", "completed"):
                break
            time.sleep(0.2)
        assert task["status"] == "error"
        assert task["error"] is not None

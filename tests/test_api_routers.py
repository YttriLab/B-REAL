"""
Integration tests for FastAPI routers using httpx AsyncClient.

Tests the HTTP layer: request validation, response shapes, SSE streaming,
and error handling. Heavy operations (video I/O, model loading) are mocked.
"""

import asyncio
import json
import os
import time

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.server import app


@pytest_asyncio.fixture
async def ac():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ── Health endpoint ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(ac):
    resp = await ac.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Preprocessing router ────────────────────────────────────────────────

class TestPreprocessingRouter:
    @pytest.mark.asyncio
    async def test_background_missing_video(self, ac):
        resp = await ac.post("/api/preprocess/background", json={
            "video_path": "/nonexistent/video.mp4",
        })
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_background_success(self, ac, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        resp = await ac.post("/api/preprocess/background", json={
            "video_path": video_path,
            "num_frames": 5,
            "output_dir": str(tmp_path / "bg"),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data
        assert os.path.exists(data["path"])

    @pytest.mark.asyncio
    async def test_preview_success(self, ac, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        resp = await ac.post("/api/preprocess/preview", json={
            "video_path": video_path,
            "bg_path": synthetic_background,
            "frame_index": 5,
            "crop_params": {"box_size": 32, "difference_threshold": 5},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "original_bbox" in data
        assert "mask" in data
        assert "cropped" in data

    @pytest.mark.asyncio
    async def test_preview_invalid_frame(self, ac, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        resp = await ac.post("/api/preprocess/preview", json={
            "video_path": video_path,
            "bg_path": synthetic_background,
            "frame_index": 99999,
            "crop_params": {"box_size": 32, "difference_threshold": 5},
        })
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_run_returns_task_id(self, ac, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        resp = await ac.post("/api/preprocess/run", json={
            "video_path": video_path,
            "output_dir": str(tmp_path / "out"),
            "crop_params": {"box_size": 32, "difference_threshold": 5},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data

    @pytest.mark.asyncio
    async def test_status_not_found(self, ac):
        resp = await ac.get("/api/preprocess/status/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_not_found(self, ac):
        resp = await ac.post("/api/preprocess/stop/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_request_validation(self, ac):
        # Missing required fields
        resp = await ac.post("/api/preprocess/run", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_crop_params_defaults(self, ac, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        resp = await ac.post("/api/preprocess/run", json={
            "video_path": video_path,
            "output_dir": str(tmp_path / "defaults"),
        })
        assert resp.status_code == 200


# ── Training router ──────────────────────────────────────────────────────

class TestTrainingRouter:
    @pytest.mark.asyncio
    async def test_start_returns_task_id(self, ac, tmp_path):
        resp = await ac.post("/api/training/start", json={
            "label_paths": ["/fake/labels.csv"],
            "frame_dirs": ["/fake/frames"],
            "max_epochs": 1,
            "checkpoint_dir": str(tmp_path / "ckpts"),
        })
        assert resp.status_code == 200
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_start_validation(self, ac):
        # Missing required label_paths and frame_dirs
        resp = await ac.post("/api/training/start", json={
            "max_epochs": 1,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_status_not_found(self, ac):
        resp = await ac.get("/api/training/status/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_not_found(self, ac):
        resp = await ac.post("/api/training/stop/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_checkpoints_empty(self, ac, tmp_path):
        resp = await ac.get("/api/training/checkpoints", params={
            "search_dirs": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["checkpoints"] == []

    @pytest.mark.asyncio
    async def test_checkpoints_finds_files(self, ac, tmp_path):
        (tmp_path / "model.ckpt").write_bytes(b"x" * 100)
        resp = await ac.get("/api/training/checkpoints", params={
            "search_dirs": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["checkpoints"]) == 1

    @pytest.mark.asyncio
    async def test_config_defaults(self, ac, tmp_path):
        """All optional fields should have defaults."""
        resp = await ac.post("/api/training/start", json={
            "label_paths": ["/fake/labels.csv"],
            "frame_dirs": ["/fake/frames"],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_wandb_fields_accepted(self, ac, tmp_path):
        """W&B logging fields are optional and accepted when provided."""
        resp = await ac.post("/api/training/start", json={
            "label_paths": ["/fake/labels.csv"],
            "frame_dirs": ["/fake/frames"],
            "max_epochs": 1,
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "use_wandb": True,
            "project_name": "my-project",
            "run_name": "run-001",
            "wandb_api_key": "secret-key",
        })
        assert resp.status_code == 200
        assert "task_id" in resp.json()


# ── Inference router ──────────────────────────────────────────────────────

class TestInferenceRouter:
    @pytest.mark.asyncio
    async def test_run_returns_task_id(self, ac, synthetic_frames, monkeypatch):
        import model_tsm
        import torch

        class FakeModel(torch.nn.Module):
            num_classes = 2
            class_names = ["a", "b"]
            def forward(self, x):
                return torch.tensor([[0.5, 0.5]] * x.shape[0])
            def eval(self):
                return self
            def to(self, device):
                return self

        monkeypatch.setattr(
            model_tsm.TSMClassifier,
            "load_from_checkpoint",
            staticmethod(lambda *a, **kw: FakeModel()),
        )

        resp = await ac.post("/api/inference/run", json={
            "checkpoint_path": "fake.ckpt",
            "frame_dir": synthetic_frames,
        })
        assert resp.status_code == 200
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_run_video_backed(self, ac, synthetic_video, synthetic_background, monkeypatch):
        import model_tsm
        import torch

        class FakeModel(torch.nn.Module):
            num_classes = 2
            class_names = ["a", "b"]
            def forward(self, x):
                return torch.tensor([[0.5, 0.5]] * x.shape[0])
            def eval(self):
                return self
            def to(self, device):
                return self

        monkeypatch.setattr(
            model_tsm.TSMClassifier,
            "load_from_checkpoint",
            staticmethod(lambda *a, **kw: FakeModel()),
        )

        video_path, _ = synthetic_video
        resp = await ac.post("/api/inference/run", json={
            "checkpoint_path": "fake.ckpt",
            "video_path": video_path,
            "bg_path": synthetic_background,
            "crop_params": {"box_size": 32, "difference_threshold": 5},
        })
        assert resp.status_code == 200
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_run_validation(self, ac):
        # Neither video_path nor frame_dir → 422
        resp = await ac.post("/api/inference/run", json={"checkpoint_path": "x.ckpt"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_status_not_found(self, ac):
        resp = await ac.get("/api/inference/status/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_results_not_found(self, ac):
        resp = await ac.get("/api/inference/results/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_results_not_complete(self, ac):
        from backend.services.inference_service import _tasks
        _tasks["test-inf"] = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "cancel": False,
            "error": None,
            "results": None,
        }
        resp = await ac.get("/api/inference/results/test-inf")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_results_success(self, ac):
        from backend.services.inference_service import _tasks
        _tasks["test-done"] = {
            "status": "completed",
            "progress": 10,
            "total": 10,
            "cancel": False,
            "error": None,
            "results": {
                "frames": [{"index": 2, "predictions": [0.8, 0.2], "uncertainty": 0.3, "predicted_class": 0}],
                "class_names": ["walk", "rest"],
                "total_frames": 10,
                "frames_per_group": 3,
            },
        }
        resp = await ac.get("/api/inference/results/test-done")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["frames"]) == 1
        assert data["class_names"] == ["walk", "rest"]

    @pytest.mark.asyncio
    async def test_stop_not_found(self, ac):
        resp = await ac.post("/api/inference/stop/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_sets_cancel(self, ac):
        from backend.services.inference_service import _tasks
        _tasks["test-cancel"] = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "cancel": False,
            "error": None,
            "results": None,
        }
        resp = await ac.post("/api/inference/stop/test-cancel")
        assert resp.status_code == 200
        assert _tasks["test-cancel"]["cancel"] is True


# ── Project router (.rlt export/import/inspect) ───────────────────────────

class TestProjectRouter:
    @pytest.mark.asyncio
    async def test_export_success(self, ac, sample_project, tmp_path):
        project_dir, manifest = sample_project
        out = str(tmp_path / "exported.rlt")
        resp = await ac.post("/api/project/export-rlt", json={
            "project_dir": project_dir,
            "output_path": out,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["summary"]["labels"] == 2
        assert data["summary"]["videos"] == len(manifest["videos"])
        assert os.path.exists(out)

    @pytest.mark.asyncio
    async def test_export_missing_project_returns_404(self, ac, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        resp = await ac.post("/api/project/export-rlt", json={
            "project_dir": str(empty),
            "output_path": str(tmp_path / "x.rlt"),
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_export_validation(self, ac):
        resp = await ac.post("/api/project/export-rlt", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_import_round_trip(self, ac, sample_project, tmp_path):
        project_dir, manifest = sample_project
        out = str(tmp_path / "exported.rlt")
        # Export first
        await ac.post("/api/project/export-rlt", json={
            "project_dir": project_dir,
            "output_path": out,
        })
        # Then import into a fresh directory
        dest = str(tmp_path / "imported")
        resp = await ac.post("/api/project/import-rlt", json={
            "rlt_path": out,
            "output_dir": dest,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["manifest"] == manifest
        assert os.path.exists(os.path.join(dest, "project.json"))
        assert os.path.exists(os.path.join(dest, "labels", "vid_a_bouts.csv"))

    @pytest.mark.asyncio
    async def test_import_missing_file_returns_404(self, ac, tmp_path):
        resp = await ac.post("/api/project/import-rlt", json={
            "rlt_path": str(tmp_path / "nope.rlt"),
            "output_dir": str(tmp_path / "out"),
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_inspect(self, ac, sample_project, tmp_path):
        project_dir, manifest = sample_project
        out = str(tmp_path / "exported.rlt")
        await ac.post("/api/project/export-rlt", json={
            "project_dir": project_dir,
            "output_path": out,
        })
        resp = await ac.post("/api/project/inspect-rlt", json={"rlt_path": out})
        assert resp.status_code == 200
        info = resp.json()
        assert info["labels"] == 2
        assert info["videos"] == len(manifest["videos"])
        assert info["manifest"]["name"] == manifest["name"]

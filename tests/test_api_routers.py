"""
Integration tests for FastAPI routers using httpx AsyncClient.

Tests the HTTP layer: request validation, response shapes, SSE streaming,
and error handling. Heavy operations (video I/O, model loading) are mocked.
"""

import asyncio
import os

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
    body = resp.json()
    assert body["status"] == "ok"
    # Health also reports the interpreter + GPU the backend will train on.
    assert "device" in body
    assert "cuda_available" in body["device"]
    assert "python" in body["device"]


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
    async def test_crop_box(self, ac, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        body = {"video_path": video_path, "frame_index": 5,
                "crop_params": {"box_size": 16, "difference_threshold": 5}}
        assert (await ac.post("/api/preprocess/crop-box", json=body)).status_code == 422
        resp = await ac.post("/api/preprocess/crop-box",
                             json={**body, "bg_path": synthetic_background})
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_index"] == 5 and len(data["box"]) == 4
        assert data["method"] == "background"

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

class TestCropMethodAndDecoder:
    """The localizer crop method and the decoder choice, at the HTTP layer."""

    @pytest.mark.asyncio
    async def test_localizer_preview_defaults_to_bundled_weights(self, ac, synthetic_video):
        """No weights path and no background: the bundled network runs."""
        resp = await ac.post("/api/preprocess/preview", json={
            "video_path": synthetic_video[0], "frame_index": 0,
            "crop_params": {"method": "localizer", "box_size": 24},
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["cropped"] and len(resp.json()["center"]) == 2

    @pytest.mark.asyncio
    async def test_localizer_preview(self, ac, synthetic_video, stub_localizer):
        resp = await ac.post("/api/preprocess/preview", json={
            "video_path": synthetic_video[0], "frame_index": 5,
            "crop_params": {"method": "localizer", "localizer_path": stub_localizer,
                            "box_size": 24},
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["cropped"] and data["mask"] and data["original_bbox"]
        assert data["presence"] > 0.5 and len(data["center"]) == 2

    @pytest.mark.asyncio
    async def test_background_preview_still_needs_a_background(self, ac, synthetic_video):
        resp = await ac.post("/api/preprocess/preview", json={
            "video_path": synthetic_video[0], "frame_index": 0,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_run_rejects_unknown_method_and_decoder(self, ac, synthetic_video, tmp_path):
        base = {"video_path": synthetic_video[0], "output_dir": str(tmp_path)}
        r1 = await ac.post("/api/preprocess/run", json={**base, "crop_params": {"method": "sam3"}})
        r2 = await ac.post("/api/preprocess/run", json={**base, "decoder": "vaapi"})
        assert (r1.status_code, r2.status_code) == (422, 422)

    @pytest.mark.asyncio
    async def test_inference_accepts_decoder_and_batch_size(self, ac, synthetic_video):
        base = {"checkpoint_path": "x.ckpt", "video_path": synthetic_video[0]}
        bad = await ac.post("/api/inference/run", json={**base, "decoder": "vaapi"})
        assert bad.status_code == 422
        bad = await ac.post("/api/inference/run", json={**base, "batch_size": 0})
        assert bad.status_code == 422


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
    async def test_check_coverage(self, ac, tmp_path):
        import csv
        labels = tmp_path / "vid.csv"
        with open(labels, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "a", "b"])
            for fn in (2, 3, 4):
                w.writerow([fn, 1, 0])
        frames = tmp_path / "vid"
        frames.mkdir()
        for i in (2, 3, 4):  # missing context 0,1
            (frames / f"{i}.png").write_bytes(b"")

        resp = await ac.post("/api/training/check-coverage", json={
            "frame_dirs": [str(frames)],
            "label_paths": [str(labels)],
            "frames_per_group": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_labeled"] == 3
        assert data["total_ready"] == 1  # only frame 4 has its full triplet
        assert data["datasets"][0]["not_extracted"] == 2

    @pytest.mark.asyncio
    async def test_benchmark_inference(self, ac):
        """Pre-training latency benchmark returns positive timings for the arch."""
        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0",
            "num_frames": 3,
            "tsm_stages": [5, 6, 7],
            "iters": 2,
            "warmup": 1,  # keep the test fast (CPU on CI)
            "target": "torchscript",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ms_per_clip"] > 0
        assert data["clips_per_sec"] > 0
        assert data["variant"] == "b0"
        assert data["num_frames"] == 3
        assert data["precision"] == "fp32"
        # every result names the runtime it came from, so a number can't be
        # mistaken for a deployment measurement
        assert data["runtime"] == "torchscript"

    @pytest.mark.asyncio
    async def test_benchmark_inference_fp16_on_cpu_downgrades(self, ac):
        """FP16 without CUDA must not error — it downgrades to FP32 with a note."""
        import torch
        if torch.cuda.is_available():
            pytest.skip("CUDA present; downgrade path not exercised")
        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0", "iters": 2, "warmup": 1, "precision": "fp16",
            "target": "torchscript",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["precision"] == "fp32"
        assert "note" in data

    @pytest.mark.asyncio
    async def test_benchmark_unknown_target_rejected(self, ac):
        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0", "iters": 1, "warmup": 1, "target": "coreml",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_benchmark_openvino(self, ac):
        """The CPU deployment path: ONNX -> OpenVINO, timed through the runtime."""
        pytest.importorskip("openvino")
        pytest.importorskip("onnx")
        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0", "num_frames": 3, "tsm_stages": [5, 6, 7],
            "iters": 2, "warmup": 1, "target": "openvino",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["runtime"] == "openvino"
        assert data["device"] == "cpu"
        assert data["ms_per_batch_p99"] > 0

    @pytest.mark.asyncio
    async def test_benchmark_missing_runtime_returns_503(self, ac, monkeypatch):
        """A runtime this box can't provide is 503, not 500 — the request is fine."""
        import optimized_runtime

        def missing(module):
            raise optimized_runtime.RuntimeUnavailable(f"{module} is not installed")
        monkeypatch.setattr(optimized_runtime, "_require", missing)

        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0", "iters": 1, "warmup": 1, "target": "openvino",
        })
        assert resp.status_code == 503
        assert "not installed" in resp.json()["detail"]

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

    @pytest.mark.asyncio
    async def test_find_lr_reaches_the_trainer(self, ac):
        """find_lr runs the range test instead of training (training_service.find_lr)."""
        from omegaconf import OmegaConf
        from backend.services.training_service import get_task
        base = {"label_paths": ["/fake/labels.csv"], "frame_dirs": ["/fake/frames"]}
        for flag in (True, False):
            resp = await ac.post("/api/training/start", json={**base, "find_lr": flag})
            assert resp.status_code == 200
            cfg = OmegaConf.load(get_task(resp.json()["task_id"])["config_path"])
            assert cfg.find_lr is flag

    @pytest.mark.asyncio
    async def test_data_loader_fields_accepted(self, ac):
        """batch_size / num_workers / image_size reach the trainer."""
        resp = await ac.post("/api/training/start", json={
            "label_paths": ["/fake/labels.csv"],
            "frame_dirs": ["/fake/frames"],
            "batch_size": 16,
            "num_workers": 2,
            "image_size": 380,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_benchmark_reports_tail_latency(self, ac):
        """p99 is what the tuning latency gate is checked against."""
        resp = await ac.post("/api/training/benchmark-inference", json={
            "variant": "b0", "iters": 3, "warmup": 1, "target": "torchscript",
        })
        assert resp.status_code == 200
        data = resp.json()
        for key in ("ms_per_batch_p50", "ms_per_batch_p95", "ms_per_batch_p99",
                    "ms_per_clip_p99"):
            assert key in data
        assert data["ms_per_batch_p50"] <= data["ms_per_batch_p99"]


# ── Frontend/backend config contract ──────────────────────────────────────

class TestGuiConfigContract:
    """The GUI builds the training payload itself, so drift between
    build_train_config and TrainingConfig is silent — a dropped field just
    reverts to a server default mid-experiment. Check the seam directly."""

    @staticmethod
    def _gui_config():
        """Import the frontend's pure config module (no Qt) from gui/."""
        import sys
        gui = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui")
        if gui not in sys.path:
            sys.path.insert(0, gui)
        from plugins import pipeline_config
        return pipeline_config

    def _build(self, **kw):
        return self._gui_config().build_train_config(["l.csv"], ["f"], **kw)

    def test_every_field_survives_validation(self):
        from backend.routers.training import TrainingConfig

        binarize, union = self._gui_config().resolve_binarize(
            ["groom", "rear"], ["groom", "rear", "walk"])
        cfg = self._build(
            binarize=binarize, label_union=union,
            boundary_smoothing_enabled=True, boundary_smoothing_epsilon=0.25,
            boundary_smoothing_window=7, use_wandb=True, project_name="p",
            run_name="r", wandb_api_key="k", batch_size=16, image_size=380)
        tc = TrainingConfig(**cfg)

        assert tc.binarize == "positive"
        assert tc.label_union == {"positive": ["groom", "rear"]}
        assert (tc.boundary_smoothing_enabled, tc.boundary_smoothing_epsilon,
                tc.boundary_smoothing_window) == (True, 0.25, 7)
        assert tc.use_wandb and tc.project_name == "p" and tc.run_name == "r"
        assert tc.wandb_api_key == "k"
        assert tc.batch_size == 16 and tc.image_size == 380

    def test_api_key_is_not_sent_when_wandb_is_off(self):
        cfg = self._build(use_wandb=False, wandb_api_key="secret")
        assert "wandb_api_key" not in cfg

    @pytest.mark.asyncio
    async def test_gui_config_is_accepted_by_the_endpoint(self, ac):
        cfg = self._build(binarize="groom", boundary_smoothing_enabled=True)
        cfg["label_paths"] = ["/fake/labels.csv"]
        cfg["frame_dirs"] = ["/fake/frames"]
        resp = await ac.post("/api/training/start", json=cfg)
        assert resp.status_code == 200


# ── Tuning router ─────────────────────────────────────────────────────────

class TestTuningRouter:
    BASE = {"label_paths": ["/fake/labels.csv"], "frame_dirs": ["/fake/frames"],
            "max_epochs": 1}

    @pytest.mark.asyncio
    async def test_start_returns_task_id(self, ac):
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE,
            "n_trials": 1,
            "lr": {"low": 1e-5, "high": 1e-2, "log": True},
        })
        assert resp.status_code == 200
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_empty_search_space_rejected(self, ac):
        resp = await ac.post("/api/tuning/start", json={"base": self.BASE})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_choice_list_rejected(self, ac):
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "variant": {"choices": []},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_trials_rejected(self, ac):
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "n_trials": 0,
            "dropout": {"low": 0.0, "high": 0.6},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_nonpositive_latency_budget_rejected(self, ac):
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "dropout": {"low": 0.0, "high": 0.6},
            "latency": {"max_p99_ms": 0},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_latency_target_rejected(self, ac):
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "dropout": {"low": 0.0, "high": 0.6},
            "latency": {"max_p99_ms": 33.0, "target": "coreml"},
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_runtime_refuses_to_start(self, ac, monkeypatch):
        """The sweep is preflighted: a budget that can't be measured stops here,
        not after burning a trial."""
        import optimized_runtime

        def missing(module):
            raise optimized_runtime.RuntimeUnavailable(f"{module} is not installed")
        monkeypatch.setattr(optimized_runtime, "_require", missing)

        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "dropout": {"low": 0.0, "high": 0.6},
            "latency": {"max_p99_ms": 33.0, "target": "openvino"},
        })
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_categorical_space_accepted(self, ac, monkeypatch):
        pytest.importorskip("openvino")
        # The sweep runs in a background thread. A real latency gate would
        # export ONNX there, racing torch's process-global ONNX export flag in
        # whichever test runs next, so stub it — and wait for the sweep to end
        # below, since monkeypatch restores the real one at teardown.
        from backend.services import tuning_service
        monkeypatch.setattr(tuning_service, "_measure_latency",
                            lambda cfg, latency: {"ms_per_batch_p99": 1e9})
        resp = await ac.post("/api/tuning/start", json={
            "base": self.BASE, "n_trials": 1,
            "variant": {"choices": ["b0", "b4"]},
            "batch_size": {"choices": [8, 16]},
            "tsm_stages": {"choices": ["5,6,7"]},
            "frames_per_group": {"choices": [2, 3]},
            "latency": {"max_p99_ms": 33.0, "precision": "fp16",
                        "target": "openvino"},
        })
        assert resp.status_code == 200
        task = tuning_service.get_task(resp.json()["task_id"])
        for _ in range(600):
            if task["status"] in ("completed", "error", "cancelled"):
                break
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_status_404_for_unknown_task(self, ac):
        resp = await ac.get("/api/tuning/status/does-not-exist")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_404_for_unknown_task(self, ac):
        resp = await ac.post("/api/tuning/stop/does-not-exist")
        assert resp.status_code == 404


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

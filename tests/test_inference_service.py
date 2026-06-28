"""Tests for backend.services.inference_service."""

import math
import os

import cv2
import numpy as np
import pytest
import torch

from backend.services.inference_service import (
    _build_val_transform,
    _load_frame,
    get_task,
    start_inference,
)


# ── _build_val_transform ─────────────────────────────────────────────────

class TestBuildValTransform:
    def test_returns_compose(self):
        t = _build_val_transform(224)
        assert t is not None

    def test_resizes_image(self):
        t = _build_val_transform(128)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = t(image=img)
        assert result["image"].shape == (3, 128, 128)

    def test_additional_targets(self):
        t = _build_val_transform(64)
        img1 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = t(image=img1, image1=img2)
        assert result["image"].shape == (3, 64, 64)
        assert result["image1"].shape == (3, 64, 64)

    def test_normalizes_to_float(self):
        t = _build_val_transform(32)
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = t(image=img)
        # After normalization, values should be floats near 0 (not 0-255)
        assert result["image"].dtype == torch.float32
        assert result["image"].abs().max() < 10  # normalized values


# ── _load_frame ──────────────────────────────────────────────────────────

class TestLoadFrame:
    def test_loads_rgb(self, tmp_path):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red in BGR
        path = str(tmp_path / "red.png")
        cv2.imwrite(path, img)

        result = _load_frame(path)
        # Should be RGB, so red channel (index 0) should be 255
        assert result[:, :, 0].mean() == 255
        assert result[:, :, 2].mean() == 0

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="Cannot read frame"):
            _load_frame("/nonexistent/frame.png")


# ── run_inference (integration) ──────────────────────────────────────────
# These tests require a model checkpoint. We mock TSMClassifier.load_from_checkpoint
# to avoid needing actual trained weights.

class TestRunInference:
    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Patch TSMClassifier.load_from_checkpoint to return a fake model."""
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes = 3
                self.class_names = ["walk", "groom", "rear"]
                self.linear = torch.nn.Linear(1, 3)  # dummy param

            def forward(self, x):
                B = x.shape[0]
                # Return fixed logits
                return torch.tensor([[1.0, 0.5, 0.2]] * B)

            def eval(self):
                return self

            def to(self, device):
                return self

        fake = FakeModel()

        def fake_load(*args, **kwargs):
            return fake

        # Patch at the module where it's imported
        import model_tsm
        monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint", staticmethod(fake_load))
        return fake

    def test_run_inference_returns_results(self, synthetic_frames, mock_model, tmp_path):
        from backend.services.inference_service import run_inference

        results = run_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
        )
        assert "frames" in results
        assert "class_names" in results
        assert results["class_names"] == ["walk", "groom", "rear"]
        # With 20 frames and T=3, we get results for frames 2..19 = 18 results
        assert len(results["frames"]) == 18

    def test_uncertainty_is_normalized(self, synthetic_frames, mock_model):
        from backend.services.inference_service import run_inference

        results = run_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
        )
        for frame_result in results["frames"]:
            assert 0.0 <= frame_result["uncertainty"] <= 1.0

    def test_predictions_sum_to_one(self, synthetic_frames, mock_model):
        from backend.services.inference_service import run_inference

        results = run_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
        )
        for frame_result in results["frames"]:
            total = sum(frame_result["predictions"])
            assert abs(total - 1.0) < 1e-4

    def test_cancel_stops_early(self, synthetic_frames, mock_model):
        from backend.services.inference_service import run_inference

        counter = {"n": 0}

        def on_progress(done, total):
            counter["n"] = done

        results = run_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
            on_progress=on_progress,
            cancel_flag=lambda: counter["n"] >= 5,
        )
        assert len(results["frames"]) < 18

    def test_progress_callback(self, synthetic_frames, mock_model):
        from backend.services.inference_service import run_inference

        progress_log = []

        def on_progress(done, total):
            progress_log.append((done, total))

        run_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
            on_progress=on_progress,
        )
        assert len(progress_log) == 18
        assert progress_log[-1] == (18, 18)


# ── run_inference_video (on-the-fly Active Learning) ─────────────────────

class TestRunInferenceVideo:
    @pytest.fixture
    def mock_model(self, monkeypatch):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes = 3
                self.class_names = ["walk", "groom", "rear"]
                self.linear = torch.nn.Linear(1, 3)

            def forward(self, x):
                return torch.tensor([[1.0, 0.5, 0.2]] * x.shape[0])

            def eval(self):
                return self

            def to(self, device):
                return self

        fake = FakeModel()
        import model_tsm
        monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint", staticmethod(lambda *a, **k: fake))
        return fake

    def test_results_indexed_by_native_frame(self, synthetic_video, synthetic_background, mock_model):
        from backend.services.inference_service import run_inference_video

        video_path, num_frames = synthetic_video
        results = run_inference_video(
            checkpoint_path="fake.ckpt",
            video_path=video_path,
            bg_path=synthetic_background,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frames_per_group=3,
            image_size=64,
        )
        assert results["class_names"] == ["walk", "groom", "rear"]
        # T=3 over num_frames → results for native indices 2..num_frames-1.
        indices = [f["index"] for f in results["frames"]]
        assert indices == list(range(2, num_frames))
        for f in results["frames"]:
            assert 0.0 <= f["uncertainty"] <= 1.0
            assert abs(sum(f["predictions"]) - 1.0) < 1e-4

    def test_frame_ranges_scopes_to_segment(self, synthetic_video, synthetic_background, mock_model):
        from backend.services.inference_service import run_inference_video

        video_path, num_frames = synthetic_video
        # A mid-video segment well inside the clip-able range.
        seg_start, seg_end = 5, 9
        results = run_inference_video(
            checkpoint_path="fake.ckpt",
            video_path=video_path,
            bg_path=synthetic_background,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frames_per_group=3,
            image_size=64,
            frame_ranges=[[seg_start, seg_end]],
        )
        indices = [f["index"] for f in results["frames"]]
        # Predictions only for the requested segment — and native indices preserved.
        assert indices == list(range(seg_start, seg_end + 1))

    def test_frame_ranges_multiple_disjoint(self, synthetic_video, synthetic_background, mock_model):
        from backend.services.inference_service import run_inference_video

        video_path, num_frames = synthetic_video
        results = run_inference_video(
            checkpoint_path="fake.ckpt",
            video_path=video_path,
            bg_path=synthetic_background,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frames_per_group=3,
            image_size=64,
            frame_ranges=[[3, 4], [8, 10]],
        )
        indices = [f["index"] for f in results["frames"]]
        assert indices == [3, 4, 8, 9, 10]

    def test_frame_ranges_drops_uncliplable_early_frames(self, synthetic_video, synthetic_background, mock_model):
        from backend.services.inference_service import run_inference_video

        video_path, num_frames = synthetic_video
        # Frames 0,1 can't form a T=3 clip; only frame 2 is emittable.
        results = run_inference_video(
            checkpoint_path="fake.ckpt",
            video_path=video_path,
            bg_path=synthetic_background,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frames_per_group=3,
            image_size=64,
            frame_ranges=[[0, 2]],
        )
        indices = [f["index"] for f in results["frames"]]
        assert indices == [2]


# ── start_inference (async task) ─────────────────────────────────────────

class TestStartInference:
    def test_returns_task_id(self, synthetic_frames, monkeypatch):
        # Patch the model loading to avoid checkpoint file
        import model_tsm

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

        task_id = start_inference(
            checkpoint_path="fake.ckpt",
            frame_dir=synthetic_frames,
            frames_per_group=3,
            image_size=64,
        )
        assert isinstance(task_id, str)
        task = get_task(task_id)
        assert task is not None
        assert task["status"] in ("running", "completed")

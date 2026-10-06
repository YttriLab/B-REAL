"""Tests for backend.services.inference_service."""


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


# ── batched video pipeline: batching, crop methods, train/inference parity ─

class _RecordingModel(torch.nn.Module):
    """Logits depend on the clip's content (so batching mistakes show up), and
    every clip the model sees is recorded."""
    num_classes = 3
    class_names = ["a", "b", "c"]

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):  # (B, T, C, H, W)
        self.seen.extend(x.detach().clone())
        m = x.mean(dim=(2, 3, 4))  # (B, T)
        return torch.stack([m[:, 0], m[:, -1], m.std(1) * 10], 1)

    def eval(self):
        return self

    def to(self, device):
        return self


@pytest.fixture
def recording_model(monkeypatch):
    import model_tsm
    model = _RecordingModel()
    monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint",
                        staticmethod(lambda *a, **k: model))
    return model


def _run_video(video, bg, crop_params, **kw):
    from backend.services.inference_service import run_inference_video
    kw.setdefault("decoder", "cpu")
    return run_inference_video("fake.ckpt", video, bg, dict(crop_params),
                               frames_per_group=3, image_size=32, **kw)


class TestBatchedVideoInference:
    BG_PARAMS = {"box_size": 24, "difference_threshold": 30, "smoothing": 0.5}

    def test_batch_size_does_not_change_results(self, synthetic_video, synthetic_background,
                                                recording_model):
        video, _ = synthetic_video
        runs = [_run_video(video, synthetic_background, self.BG_PARAMS, batch_size=b,
                           frame_ranges=[[2, 9], [15, 27]])["frames"]
                for b in (1, 4, 64)]
        for other in runs[1:]:
            assert [f["index"] for f in other] == [f["index"] for f in runs[0]]
            for a, b in zip(runs[0], other):
                assert a["predicted_class"] == b["predicted_class"]
                assert np.allclose(a["predictions"], b["predictions"], atol=1e-6)

    @pytest.mark.parametrize("method", ["background", "localizer", "none"])
    def test_inference_sees_exactly_the_preprocessed_crops(
            self, method, synthetic_video, synthetic_background, recording_model,
            stub_localizer, tmp_path):
        """The model must be evaluated on what it was trained on: for every crop
        method, the clips inference builds equal the training transform applied
        to the PNGs preprocessing writes, frame for frame, across range gaps."""
        from backend.services.inference_service import _build_val_transform, _load_frame
        from backend.services.preprocess_service import crop_video_flat

        video, _ = synthetic_video
        params = dict(self.BG_PARAMS, method=method, localizer_path=stub_localizer,
                      resize_factor=0.75)
        ranges = [[3, 10], [14, 20], [22, 29]]  # gaps restart the smoothing
        out = crop_video_flat(video, str(tmp_path / "frames"), dict(params),
                              frame_ranges=ranges, bg_path=synthetic_background,
                              decoder="cpu")
        # ranges already include the T-1 lead-in, as the GUI sends them
        wanted = [[s + 2, e] for s, e in ranges]
        res = _run_video(video, synthetic_background, params, frame_ranges=wanted,
                         batch_size=5)

        transform = _build_val_transform(32)
        emitted = [f["index"] for f in res["frames"]]
        assert emitted == [i for s, e in wanted for i in range(s, e + 1)]
        assert len(recording_model.seen) == len(emitted)
        for idx, clip in zip(emitted, recording_model.seen):
            ref = torch.stack([transform(image=_load_frame(f"{out}/{i}.png"))["image"]
                               for i in (idx - 2, idx - 1, idx)])
            assert torch.equal(clip, ref), f"clip ending at frame {idx} differs"

    def test_whole_frame_method_ignores_the_background(self, synthetic_video,
                                                       recording_model):
        video, _ = synthetic_video
        res = _run_video(video, None, {"method": "none"}, frame_ranges=[[2, 4]])
        assert [f["index"] for f in res["frames"]] == [2, 3, 4]

    def test_payload_contract(self, synthetic_video, synthetic_background, recording_model):
        """Both apps' Active Learning panels read this shape — don't change it."""
        video, n = synthetic_video
        res = _run_video(video, synthetic_background, self.BG_PARAMS)
        assert set(res) == {"frames", "class_names", "total_frames", "frames_per_group"}
        assert res["total_frames"] == n
        assert [f["index"] for f in res["frames"]] == list(range(2, n))
        f = res["frames"][0]
        assert set(f) == {"index", "predictions", "uncertainty", "predicted_class"}
        assert 0.0 <= f["uncertainty"] <= 1.0

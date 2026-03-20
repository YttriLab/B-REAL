"""Tests for backend.services.preprocess_service."""

import os
import time
import threading

import cv2
import numpy as np
import pytest

from backend.services.preprocess_service import (
    _encode_image,
    compute_background,
    crop_video_flat,
    get_task,
    preview_frame,
    start_preprocessing,
)


# ── _encode_image ─────────────────────────────────────────────────────────

class TestEncodeImage:
    def test_returns_base64_string(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _encode_image(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_is_valid_base64(self):
        import base64
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _encode_image(img)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_quality_affects_size(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        low_q = _encode_image(img, quality=10)
        high_q = _encode_image(img, quality=95)
        # Higher quality should generally produce larger output
        assert len(high_q) >= len(low_q)


# ── compute_background ────────────────────────────────────────────────────

class TestComputeBackground:
    def test_saves_background_png(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "bg_out")
        result = compute_background(video_path, num_frames=10, output_dir=out_dir)
        assert os.path.exists(result)
        assert result.endswith(".png")

    def test_default_output_dir(self, synthetic_video):
        video_path, _ = synthetic_video
        result = compute_background(video_path, num_frames=10)
        assert "backgrounds" in result
        assert os.path.exists(result)

    def test_background_matches_video_dimensions(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "bg_out2")
        result = compute_background(video_path, num_frames=10, output_dir=out_dir)
        bg = cv2.imread(result, cv2.IMREAD_GRAYSCALE)
        assert bg.shape == (64, 64)


# ── preview_frame ─────────────────────────────────────────────────────────

class TestPreviewFrame:
    def test_returns_three_images(self, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        result = preview_frame(
            video_path=video_path,
            bg_path=synthetic_background,
            frame_index=10,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        assert "original_bbox" in result
        assert "mask" in result
        assert "cropped" in result
        assert isinstance(result["original_bbox"], str)
        assert isinstance(result["mask"], str)
        assert isinstance(result["cropped"], str)

    def test_returns_frame_dimensions(self, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        result = preview_frame(
            video_path=video_path,
            bg_path=synthetic_background,
            frame_index=5,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        assert result["frame_width"] == 64
        assert result["frame_height"] == 64

    def test_invalid_frame_index_raises(self, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        with pytest.raises(RuntimeError, match="Could not read frame"):
            preview_frame(
                video_path=video_path,
                bg_path=synthetic_background,
                frame_index=99999,
                crop_params={"box_size": 32, "difference_threshold": 5},
            )


# ── crop_video_flat ───────────────────────────────────────────────────────

class TestCropVideoFlat:
    def test_creates_flat_png_files(self, synthetic_video, tmp_path):
        video_path, num_frames = synthetic_video
        out_dir = str(tmp_path / "cropped")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        assert os.path.isdir(result)
        pngs = [f for f in os.listdir(result) if f.endswith(".png")]
        assert len(pngs) == num_frames
        # Check sequential naming
        indices = sorted(int(f.replace(".png", "")) for f in pngs)
        assert indices == list(range(num_frames))

    def test_framerate_downsampling(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_ds")
        os.makedirs(out_dir, exist_ok=True)
        # Video is 30fps, downsample to 15fps → skip_size=2 → half the frames
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            desired_framerate=15,
        )
        pngs = [f for f in os.listdir(result) if f.endswith(".png")]
        assert len(pngs) == 15

    def test_invalid_framerate_raises(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_bad_fps")
        os.makedirs(out_dir, exist_ok=True)
        with pytest.raises(ValueError, match="not an integer multiple"):
            crop_video_flat(
                video_path=video_path,
                output_dir=out_dir,
                crop_params={"box_size": 32, "difference_threshold": 5},
                desired_framerate=7,
            )

    def test_cancel_stops_early(self, synthetic_video, tmp_path):
        video_path, num_frames = synthetic_video
        out_dir = str(tmp_path / "cropped_cancel")
        os.makedirs(out_dir, exist_ok=True)
        counter = {"n": 0}

        def cancel_after_5():
            return counter["n"] >= 5

        def on_progress(done, total):
            counter["n"] = done

        crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            on_progress=on_progress,
            cancel_flag=cancel_after_5,
        )
        pngs = [f for f in os.listdir(os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0])) if f.endswith(".png")]
        assert len(pngs) < num_frames

    def test_progress_callback(self, synthetic_video, tmp_path):
        video_path, num_frames = synthetic_video
        out_dir = str(tmp_path / "cropped_prog")
        os.makedirs(out_dir, exist_ok=True)
        progress_log = []

        def on_progress(done, total):
            progress_log.append((done, total))

        crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            on_progress=on_progress,
        )
        assert len(progress_log) == num_frames
        assert progress_log[-1][0] == num_frames  # final done count
        assert all(total == num_frames for _, total in progress_log)

    def test_custom_bg_path(self, synthetic_video, synthetic_background, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_bg")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            bg_path=synthetic_background,
        )
        pngs = [f for f in os.listdir(result) if f.endswith(".png")]
        assert len(pngs) > 0

    def test_custom_output_name(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_named")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            output_name="my_video",
        )
        assert os.path.basename(result) == "my_video"


# ── start_preprocessing (async task) ─────────────────────────────────────

class TestStartPreprocessing:
    def test_returns_task_id(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "async_out")
        os.makedirs(out_dir, exist_ok=True)
        task_id = start_preprocessing(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    def test_task_completes(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "async_out2")
        os.makedirs(out_dir, exist_ok=True)
        task_id = start_preprocessing(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        # Poll until done
        for _ in range(100):
            task = get_task(task_id)
            if task["status"] != "running":
                break
            time.sleep(0.1)
        assert task["status"] == "completed"
        assert task["output_dir"] is not None
        assert os.path.isdir(task["output_dir"])

    def test_task_can_be_cancelled(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "async_cancel")
        os.makedirs(out_dir, exist_ok=True)
        task_id = start_preprocessing(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        # Cancel immediately
        task = get_task(task_id)
        task["cancel"] = True
        for _ in range(100):
            if task["status"] != "running":
                break
            time.sleep(0.1)
        assert task["status"] in ("cancelled", "completed")  # May finish before cancel

    def test_invalid_video_sets_error(self, tmp_path):
        out_dir = str(tmp_path / "async_err")
        os.makedirs(out_dir, exist_ok=True)
        task_id = start_preprocessing(
            video_path="/nonexistent/video.mp4",
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
        )
        for _ in range(50):
            task = get_task(task_id)
            if task["status"] != "running":
                break
            time.sleep(0.1)
        assert task["status"] == "error"
        assert task["error"] is not None

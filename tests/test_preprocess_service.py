"""Tests for backend.services.preprocess_service."""

import os
import time

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

    def test_difference_mode_absolute_detects_brighter(self):
        """Directional misses a uniformly-brighter frame; absolute detects it."""
        from tune_cropping import FindRatBoundingBox
        bg = np.full((64, 64), 100, dtype=np.float32)
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)  # brighter than bg everywhere

        directional = FindRatBoundingBox(bg, difference_threshold=5, box_size=20,
                                         difference_mode="directional")
        bb_d, *_ = directional(frame)
        assert bb_d[2] == 64 and bb_d[3] == 64  # nothing darker -> previous full-frame box

        absolute = FindRatBoundingBox(bg, difference_threshold=5, box_size=20,
                                      difference_mode="absolute")
        bb_a, *_ = absolute(frame)
        assert bb_a[2] == 20 and bb_a[3] == 20  # change detected -> fixed-size box

    def test_resize_factor_reports_original_and_resized(self, synthetic_video, synthetic_background):
        video_path, _ = synthetic_video
        result = preview_frame(
            video_path=video_path,
            bg_path=synthetic_background,
            frame_index=5,
            crop_params={"box_size": 32, "difference_threshold": 5, "resize_factor": 0.5},
        )
        # frame_width/height stay the ORIGINAL source size; resized_* is what's processed.
        assert result["frame_width"] == 64 and result["frame_height"] == 64
        assert result["resized_width"] == 32 and result["resized_height"] == 32

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

    def test_resize_factor_shrinks_output(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        # box_size larger than the frame → crop is the whole (resized) frame,
        # so the saved PNG dimensions directly reflect the pre-resize.
        full = str(tmp_path / "full")
        half = str(tmp_path / "half")
        os.makedirs(full, exist_ok=True)
        os.makedirs(half, exist_ok=True)
        crop_video_flat(video_path=video_path, output_dir=full,
                        crop_params={"box_size": 200, "difference_threshold": 5},
                        frame_ranges=[[5, 5]])
        crop_video_flat(video_path=video_path, output_dir=half,
                        crop_params={"box_size": 200, "difference_threshold": 5, "resize_factor": 0.5},
                        frame_ranges=[[5, 5]])
        import cv2
        full_img = cv2.imread(os.path.join(full, os.listdir(full)[0], "5.png"))
        half_img = cv2.imread(os.path.join(half, os.listdir(half)[0], "5.png"))
        assert full_img.shape[0] == 64 and full_img.shape[1] == 64
        assert half_img.shape[0] == 32 and half_img.shape[1] == 32

    def test_raw_mode_writes_whole_frames(self, synthetic_video, tmp_path):
        """crop=False writes the full frame, not a box_size crop."""
        import cv2
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "raw")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 16, "difference_threshold": 5},
            frame_ranges=[[5, 5]],
            crop=False,
        )
        img = cv2.imread(os.path.join(result, "5.png"))
        # the synthetic video is 64x64; a cropped run would give 16x16
        assert img.shape[:2] == (64, 64)

    def test_raw_mode_never_touches_the_background(self, synthetic_video, tmp_path,
                                                   monkeypatch):
        """Skipping the crop must also skip building a background — decoding
        hundreds of frames for one that gets discarded is the whole cost."""
        from backend.services import preprocess_service

        def boom(*a, **kw):
            raise AssertionError("background computed in raw mode")

        monkeypatch.setattr(preprocess_service, "get_background_avg", boom)

        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "raw_nobg")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path, output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frame_ranges=[[2, 3]], crop=False,
        )
        assert sorted(os.listdir(result)) == ["2.png", "3.png"]

    def test_raw_mode_still_applies_resize_factor(self, synthetic_video, tmp_path):
        import cv2
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "raw_half")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path, output_dir=out_dir,
            crop_params={"box_size": 200, "resize_factor": 0.5},
            frame_ranges=[[5, 5]], crop=False,
        )
        img = cv2.imread(os.path.join(result, "5.png"))
        assert img.shape[:2] == (32, 32)

    def test_sparse_extraction_only_writes_requested_frames(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_sparse")
        os.makedirs(out_dir, exist_ok=True)
        # Only frames 5..8 (inclusive) — named by native index.
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frame_ranges=[[5, 8]],
        )
        pngs = sorted(int(f.replace(".png", "")) for f in os.listdir(result) if f.endswith(".png"))
        assert pngs == [5, 6, 7, 8]

    def test_sparse_extraction_merges_multiple_ranges(self, synthetic_video, tmp_path):
        video_path, _ = synthetic_video
        out_dir = str(tmp_path / "cropped_sparse2")
        os.makedirs(out_dir, exist_ok=True)
        result = crop_video_flat(
            video_path=video_path,
            output_dir=out_dir,
            crop_params={"box_size": 32, "difference_threshold": 5},
            frame_ranges=[[1, 2], [10, 12]],
        )
        pngs = sorted(int(f.replace(".png", "")) for f in os.listdir(result) if f.endswith(".png"))
        assert pngs == [1, 2, 10, 11, 12]

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


# ── Crop-center stabilization (jitter fix) ──────────────────────────────────

class TestCropperStabilization:
    def _bg(self):
        return np.zeros((64, 64), dtype=np.float32)

    def test_smoothing_off_passes_through(self):
        from tune_cropping import FindRatBoundingBox
        c = FindRatBoundingBox(self._bg())  # smoothing defaults to 0
        assert c._apply_smoothing(10, 10) == (10, 10)
        assert c._apply_smoothing(40, 40) == (40, 40)

    def test_smoothing_ema_and_deadband(self):
        from tune_cropping import FindRatBoundingBox
        c = FindRatBoundingBox(self._bg(), smoothing=0.5, deadband=3.0)
        assert c._apply_smoothing(10, 10) == (10, 10)  # first call seeds center
        assert c._apply_smoothing(12, 10) == (10, 10)  # within dead-band -> hold
        nx, ny = c._apply_smoothing(20, 10)            # beyond dead-band -> EMA
        assert nx == 0.5 * 20 + 0.5 * 10 and ny == 10

    def test_min_contour_area_filters_specks(self):
        import cv2
        from tune_cropping import FindRatBoundingBox
        c = FindRatBoundingBox(self._bg(), min_contour_area=50)
        mask = np.zeros((64, 64), np.uint8)
        cv2.rectangle(mask, (5, 5), (25, 25), 255, -1)    # large blob
        cv2.rectangle(mask, (60, 60), (62, 62), 255, -1)  # speck
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = c._contour_centroid(c._select_contour(contours))
        assert 5 <= cx <= 25 and 5 <= cy <= 25  # tracked the large blob, not the speck

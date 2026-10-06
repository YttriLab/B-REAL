"""Tests for the SAM2-distilled localizer crop path (localizer.py, cropping.py)."""

import numpy as np
import pytest
import torch

import localizer
from localizer import IN_H, IN_W, Localizer, LocalizerCropper
from tests.conftest import DarkCentroidLocalizer


def _frame(h, w, cx, cy, r=6, bgr=True):
    f = np.full((h, w), 200, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    f[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 20
    return np.stack([f] * 3, -1) if bgr else f


def _cropper(**kw):
    return LocalizerCropper(None, model=DarkCentroidLocalizer(), device="cpu", **kw)


class TestNetwork:
    def test_output_shapes(self):
        model = Localizer(pretrained=False).eval()
        with torch.no_grad():
            xy, heat, pres = model(torch.zeros(2, IN_H, IN_W, dtype=torch.uint8))
        assert xy.shape == (2, 2)
        assert heat.shape == (2, IN_H // 8, IN_W // 8)
        assert pres.shape == (2,)
        # soft-argmax: a convex combination of cell centres, so inside the image
        assert (xy[:, 0] > 0).all() and (xy[:, 0] < IN_W).all()
        assert (xy[:, 1] > 0).all() and (xy[:, 1] < IN_H).all()

    def test_weights_round_trip_and_cache(self, tmp_path):
        """label_video's Localizer2 checkpoints are plain state dicts with the
        same keys; load_localizer must accept them and reuse the loaded model."""
        path = str(tmp_path / "loc.pt")
        torch.save(Localizer(pretrained=False).state_dict(), path)
        a = localizer.load_localizer(path, "cpu")
        b = localizer.load_localizer(path, "cpu")
        assert a is b
        assert not a.training


class TestLocate:
    def test_centres_are_in_native_pixels(self):
        frames = [_frame(704, 768, 300, 500), _frame(352, 384, 100, 50, bgr=False)]
        centers, pres = _cropper().locate(frames)
        assert np.allclose(centers[0], [300, 500], atol=3)
        assert np.allclose(centers[1], [100, 50], atol=2)
        assert (pres > 0.5).all()

    def test_empty_batch(self):
        centers, pres = _cropper().locate([])
        assert centers.shape == (0, 2) and pres.shape == (0,)


class TestCrop:
    def test_box_is_centred_and_clamped_inside_the_frame(self):
        c = _cropper(box_size=40, smoothing=0.0)
        f = _frame(100, 120, 60, 50)
        crop, (x0, y0, w, h) = c.crop(f, (60, 50), 1.0, f.shape)
        assert (x0, y0, w, h) == (40, 30, 40, 40) and crop.shape == (40, 40, 3)
        c.reset()
        crop, (x0, y0, _, _) = c.crop(f, (2, 98), 1.0, f.shape)  # near a corner
        assert (x0, y0) == (0, 60) and crop.shape == (40, 40, 3)

    def test_crop_is_a_copy(self):
        c = _cropper(box_size=10)
        f = _frame(50, 50, 25, 25)
        crop, _ = c.crop(f, (25, 25), 1.0, f.shape)
        f[:] = 0
        assert crop.max() > 0

    def test_resized_frame_scales_the_centre(self):
        """box_size is in the cropped frame's pixels, like background mode."""
        c = _cropper(box_size=20, smoothing=0.0)
        native = (200, 240)
        half = _frame(100, 120, 30, 40)
        _, (x0, y0, _, _) = c.crop(half, (60, 80), 1.0, native)
        assert (x0, y0) == (30 - 10, 40 - 10)

    def test_smoothing_deadband_and_presence_hold(self):
        c = _cropper(box_size=10, smoothing=0.5, deadband=0.0, presence_threshold=0.5)
        shape = (100, 100)
        f = np.zeros(shape, np.uint8)
        c.crop(f, (20, 20), 1.0, shape)
        assert c.smoothed_center == (20, 20)  # first frame: raw centre
        c.crop(f, (40, 20), 1.0, shape)
        assert c.smoothed_center == (30, 20)  # EMA 0.5
        c.crop(f, (90, 90), 0.1, shape)
        assert c.smoothed_center == (30, 20)  # animal absent: hold still
        c.deadband = 5.0
        c.crop(f, (33, 20), 1.0, shape)
        assert c.smoothed_center == (30, 20)  # inside the dead-band
        c.reset()
        c.crop(f, (70, 70), 1.0, shape)
        assert c.smoothed_center == (70, 70)


class TestFrameCropper:
    def test_localizer_method_needs_weights(self):
        from backend.services.cropping import FrameCropper
        with pytest.raises(ValueError, match="localizer_path"):
            FrameCropper({"method": "localizer"})

    def test_unknown_method(self):
        from backend.services.cropping import FrameCropper
        with pytest.raises(ValueError, match="method"):
            FrameCropper({"method": "sam3"})

    def test_legacy_crop_false_means_whole_frame(self):
        from backend.services.cropping import FrameCropper
        fc = FrameCropper({"resize_factor": 0.5}, crop=False)
        assert fc.method == "none"
        out = fc.crop_batch([np.zeros((40, 60, 3), np.uint8)])
        assert out[0].shape == (20, 30, 3)

    def test_localizer_batch_with_gap_reset(self, stub_localizer):
        from backend.services.cropping import FrameCropper
        fc = FrameCropper({"method": "localizer", "localizer_path": stub_localizer,
                           "box_size": 20, "smoothing": 0.5})
        frames = [_frame(100, 100, 30, 30), _frame(100, 100, 50, 30), _frame(100, 100, 80, 70)]
        fc.crop_batch(frames[:2])
        assert fc._loc.smoothed_center[0] == pytest.approx(40, abs=1.5)  # smoothed
        fc.crop_batch(frames[2:], resets=[True])
        assert fc._loc.smoothed_center == pytest.approx((80, 70), abs=1.5)  # restarted
        fc.close()

    def test_gray_and_bgr_frames_crop_identically(self, stub_localizer):
        """NVDEC yields 2-D gray frames for mono video; the crop must be the same
        pixels cv2's BGR frame would give."""
        from backend.services.cropping import FrameCropper, to_rgb
        params = {"method": "localizer", "localizer_path": stub_localizer, "box_size": 24}
        bgr = [_frame(90, 110, 40 + i, 45) for i in range(4)]
        gray = [f[..., 0].copy() for f in bgr]
        a = FrameCropper(dict(params)).crop_batch(bgr)
        b = FrameCropper(dict(params)).crop_batch(gray)
        for x, y in zip(a, b):
            assert np.array_equal(to_rgb(x), to_rgb(y))

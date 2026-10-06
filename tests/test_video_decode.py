"""Tests for video_decode: the cv2 and NVDEC frame sources and decoder choice."""

import threading

import cv2
import numpy as np
import pytest

import video_decode


def _cv2_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def _nvdec_available():
    try:
        import torch
        import PyNvVideoCodec  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


needs_nvdec = pytest.mark.skipif(not _nvdec_available(),
                                 reason="needs PyNvVideoCodec and a CUDA GPU")


class TestCpuDecoder:
    def test_frames_match_cv2_in_order(self, synthetic_video):
        path, n = synthetic_video
        got = list(video_decode.iter_frames(path, "cpu", batch_size=7))
        ref = _cv2_frames(path)
        assert len(got) == len(ref) == n
        assert all(np.array_equal(a, b) for a, b in zip(got, ref))

    def test_batches_and_max_frames(self, synthetic_video):
        path, _ = synthetic_video
        sizes = [len(b) for b in video_decode.iter_batches(path, "cpu", batch_size=8,
                                                           max_frames=20)]
        assert sizes == [8, 8, 4]

    def test_breaking_out_stops_the_decode_thread(self, synthetic_video):
        path, _ = synthetic_video
        before = {t.name for t in threading.enumerate()}
        gen = video_decode.iter_batches(path, "cpu", batch_size=2, prefetch=1)
        next(gen)
        gen.close()
        leftover = [t for t in threading.enumerate()
                    if t.name.startswith("decode-") and t.name not in before and t.is_alive()]
        assert not leftover

    def test_unreadable_video_raises_in_the_consumer(self, tmp_path):
        bad = tmp_path / "nope.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(RuntimeError):
            list(video_decode.iter_frames(str(bad), "cpu"))

    def test_rejects_unresolved_decoder(self, synthetic_video):
        with pytest.raises(ValueError):
            list(video_decode.iter_frames(synthetic_video[0], "auto"))


class TestResolveDecoder:
    def test_cpu_is_taken_as_is(self, synthetic_video):
        assert video_decode.resolve_decoder("cpu", synthetic_video[0])[0] == "cpu"

    def test_unknown_decoder(self, synthetic_video):
        with pytest.raises(ValueError):
            video_decode.resolve_decoder("vaapi", synthetic_video[0])

    def test_auto_falls_back_to_cv2_and_says_why(self, synthetic_video, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        dec, reason = video_decode.resolve_decoder("auto", synthetic_video[0])
        assert dec == "cpu" and "no CUDA" in reason

    def test_forced_nvdec_raises_when_unusable(self, synthetic_video, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="NVDEC"):
            video_decode.resolve_decoder("nvdec", synthetic_video[0])

    def test_auto_rejects_a_colour_video(self, tmp_path, monkeypatch):
        """NVDEC only reproduces cv2 when every BGR channel is the same."""
        path = str(tmp_path / "colour.avi")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
        for i in range(5):
            f = np.zeros((32, 32, 3), np.uint8)
            f[..., 2] = 200  # red
            w.write(f)
        w.release()
        ok, msg = video_decode.check_nvdec_matches_cv2(path)
        assert not ok and "colour" in msg


@needs_nvdec
class TestNvdec:
    def test_bit_identical_to_cv2_on_a_mono_video(self, mono_h264_video):
        path, n = mono_h264_video
        got = list(video_decode.iter_frames(path, "nvdec", batch_size=16))
        ref = _cv2_frames(path)
        assert len(got) == len(ref) == n
        for g, f in zip(got, ref):
            assert g.ndim == 2
            assert np.array_equal(g, f[..., 0])

    def test_auto_picks_nvdec_for_a_mono_video(self, mono_h264_video):
        dec, reason = video_decode.resolve_decoder("auto", mono_h264_video[0])
        assert dec == "nvdec", reason

    def test_max_frames(self, mono_h264_video):
        path, _ = mono_h264_video
        assert len(list(video_decode.iter_frames(path, "nvdec", batch_size=8,
                                                 max_frames=13))) == 13

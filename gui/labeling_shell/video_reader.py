"""Frame-accurate random-access video reader built on PyAV (libav / ffmpeg).

This is the Python replacement for the Electron app's frame-accurate video
engine, which was a browser workaround: chunked disk reads -> `mp4box` demux ->
a decode-order sample table with keyframe indices -> `VideoDecoder` -> canvas,
plus a hand-rolled keyframe-aware seek/reset state machine
(core/player.js at the electron-final tag) and an MKV->MP4 remux path because
`mp4box` is MP4-only.

libav gives us all of that natively:

  * `av.open()` handles MP4/MKV/AVI/WebM directly -> the remux path disappears.
  * seeking with `backward=True` lands on the nearest preceding keyframe, so the
    keyframe-location logic we hand-rolled is done for us; we just decode forward
    to the target.
  * B-frame reordering is handled by the decoder, which outputs frames in
    presentation (display) order.

Frame addressing is by integer *display-order* index (0 .. num_frames-1), the
same contract as the Electron app's `currentFrameIndex`.

Frame <-> time uses `index * frame_duration` deliberately (see
`player.js:1032-1037`): real container timestamps (CTS/PTS) are non-monotonic in
decode order with B-frames, which would make the displayed time jump around while
stepping frame-by-frame. Indexing by frame keeps stepping and labeling exact.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict

import av
import numpy as np


class VideoReader:
    """Open a video and read any frame by integer display-order index.

    Args:
        path: path to a video file (any container libav can demux).
        cache_size: number of decoded RGB frames to keep in an LRU cache
            (mirrors the Electron app's 300-bitmap cache).
    """

    def __init__(self, path, cache_size: int = 300):
        self.path = str(path)
        self._container = av.open(self.path)
        if not self._container.streams.video:
            self._container.close()
            raise ValueError(f"no video stream in {self.path!r}")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"  # let libav use frame/slice threading
        self._time_base = self._stream.time_base  # Fraction: seconds per pts unit

        # Display-order index: pts of every frame + whether it is a keyframe.
        self._pts_list: list[int] = []
        self._is_keyframe: list[bool] = []
        self._build_index()

        self._detected_fps = self._detect_fps()
        self._fps_override: float | None = None

        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = int(cache_size)

        # Running forward-decode state, so sequential access (playback / single
        # frame stepping) does not re-seek on every frame.
        self._gen = None
        self._gen_pos: int | None = None

    # ------------------------------------------------------------------ index
    def _build_index(self) -> None:
        """One demux-only pass to enumerate frames in display order.

        Demuxing (no decoding) is fast: one compressed packet == one coded
        frame. Packets arrive in decode order; sorting by pts yields display
        order. Keyframe flags travel with their pts.
        """
        entries: list[tuple[int, bool]] = []
        for packet in self._container.demux(self._stream):
            if packet.size == 0:
                continue  # flush/sentinel packet
            pts = packet.pts if packet.pts is not None else packet.dts
            if pts is None:
                continue
            entries.append((int(pts), bool(packet.is_keyframe)))
        entries.sort(key=lambda e: e[0])
        self._pts_list = [pts for pts, _ in entries]
        self._is_keyframe = [kf for _, kf in entries]
        self.num_frames = len(self._pts_list)
        self._reset_decoder()

    def _reset_decoder(self) -> None:
        self._container.seek(0, stream=self._stream, backward=True, any_frame=False)
        self._stream.codec_context.flush_buffers()
        self._gen = None
        self._gen_pos = None

    # -------------------------------------------------------------------- fps
    def _detect_fps(self) -> float:
        rate = self._stream.average_rate
        if rate is None or float(rate) <= 0:
            rate = self._stream.guessed_rate
        if rate is not None and float(rate) > 0:
            return float(rate)
        # last resort: derive from duration
        if self._stream.duration and self.num_frames:
            dur_s = float(self._stream.duration * self._time_base)
            if dur_s > 0:
                return self.num_frames / dur_s
        return 30.0

    @property
    def fps(self) -> float:
        """Effective fps: the manual lock override if set, else auto-detected."""
        return self._fps_override if self._fps_override else self._detected_fps

    @property
    def detected_fps(self) -> float:
        return self._detected_fps

    def set_fps_override(self, fps: float | None) -> None:
        """Manual FPS lock (mirrors the Electron `fpsLock` checkbox)."""
        self._fps_override = float(fps) if fps else None

    @property
    def frame_duration_ms(self) -> float:
        return 1000.0 / self.fps

    def frame_to_ms(self, index: int) -> float:
        return index * self.frame_duration_ms

    def ms_to_frame(self, ms: float) -> int:
        return int(round(ms / self.frame_duration_ms))

    # ----------------------------------------------------------------- decode
    def _nearest_keyframe_index(self, index: int) -> int:
        """Largest display index <= `index` that is a keyframe (0 is always ok)."""
        j = index
        while j > 0 and not self._is_keyframe[j]:
            j -= 1
        return j

    def _pts_to_index(self, pts: int) -> int | None:
        i = bisect.bisect_left(self._pts_list, pts)
        if i < self.num_frames and self._pts_list[i] == pts:
            return i
        return None

    def _decode_from_keyframe(self, kf_index: int):
        """Generator yielding (display_index, av.VideoFrame) from a keyframe on."""
        kf_pts = self._pts_list[kf_index]
        self._container.seek(kf_pts, stream=self._stream, backward=True, any_frame=False)
        self._stream.codec_context.flush_buffers()
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            idx = self._pts_to_index(int(frame.pts))
            if idx is None:
                continue
            yield idx, frame

    def _advance_gen_to(self, index: int) -> np.ndarray | None:
        """Pull frames from the running generator until we reach `index`."""
        if self._gen is None:
            return None
        for idx, frame in self._gen:
            arr = frame.to_ndarray(format="rgb24")
            self._cache_put(idx, arr)
            self._gen_pos = idx + 1
            if idx == index:
                return arr
            if idx > index:  # overshoot: shouldn't happen (ascending order)
                return None
        self._gen = None
        self._gen_pos = None
        return None

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame `index` as an (H, W, 3) uint8 RGB array.

        Random access seeks to the nearest keyframe and decodes forward; nearby
        forward access reuses the running decoder (the same policy as
        `player.js`: continue forward for small jumps, re-seek for backward or
        far-forward jumps).
        """
        if not (0 <= index < self.num_frames):
            raise IndexError(f"frame {index} out of range [0, {self.num_frames})")

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        # Fast path: continue the current forward decode for small forward jumps.
        if self._gen_pos is not None and self._gen_pos <= index <= self._gen_pos + 120:
            arr = self._advance_gen_to(index)
            if arr is not None:
                return arr

        # Slow path: seek to nearest keyframe, decode forward to target.
        self._gen = self._decode_from_keyframe(self._nearest_keyframe_index(index))
        self._gen_pos = None
        arr = self._advance_gen_to(index)
        if arr is None:
            raise RuntimeError(f"failed to decode frame {index}")
        return arr

    # ------------------------------------------------------------------ misc
    def _cache_put(self, index: int, arr: np.ndarray) -> None:
        self._cache[index] = arr
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @property
    def width(self) -> int:
        return self._stream.codec_context.width

    @property
    def height(self) -> int:
        return self._stream.codec_context.height

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:
            pass

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

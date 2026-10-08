"""Frame -> crop, shared by preprocessing (training frames) and inference.

Training crops and inference crops have to come from the same code, or the
model is evaluated on a distribution it never saw. Both services build a
``FrameCropper`` from the same ``crop_params`` and feed it decoded frames.

``crop_params["method"]`` picks how the animal is found:

- ``"background"`` (default): background subtraction + contour centroid
  (``tune_cropping.FindRatBoundingBox``). Needs a background image.
- ``"localizer"``: the SAM2-distilled network (``localizer.py``), batched on
  the GPU. ``crop_params["localizer_path"]`` picks the weights (default: the
  bundled ``localizer.DEFAULT_WEIGHTS``); no background.
- ``"none"``: no crop, the whole (resized) frame.

``resize_factor`` pre-resizes the whole frame in every mode. Smoothing state
is reset at every gap in the extracted frame indices, so each extracted run
starts from a fresh centre rather than one carried over from a distant bout.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

METHODS = ("background", "localizer", "none")
# crop_params keys that aren't FindRatBoundingBox kwargs
_NON_BG_KEYS = ("method", "localizer_path", "presence_threshold", "localizer_runtime",
                "localizer_precision")


def split_crop_params(crop_params: dict | None, crop: bool = True):
    """-> (method, resize_factor, extras, rest). ``crop=False`` (the legacy raw
    flag) forces method "none"."""
    from tune_cropping import pop_resize_factor
    cp, resize_factor = pop_resize_factor(crop_params or {})
    extras = {k: cp.pop(k) for k in _NON_BG_KEYS if k in cp}
    method = (extras.get("method") or "background") if crop else "none"
    if method not in METHODS:
        raise ValueError(f"crop method must be one of {METHODS}, got {method!r}")
    return method, resize_factor, extras, cp


class FrameCropper:
    def __init__(self, crop_params: dict | None, *, video_path: str | None = None,
                 bg_path: str | None = None, num_frames_bg: int = 500, crop: bool = True,
                 localizer=None, on_status=None):
        """``localizer``: an already-built LocalizerCropper (tests, or reuse).
        ``on_status(msg)``: progress notes (e.g. a one-time TensorRT build)."""
        from tune_cropping import FindRatBoundingBox, get_background_avg, resize_frame

        self.method, self.resize_factor, extras, rest = split_crop_params(crop_params, crop)
        # Per-frame work that doesn't depend on crop state (the full-frame
        # resize) runs here, a batch ahead; see prepare().
        self._pool = ThreadPoolExecutor(8)
        self.bg = None
        self._bg_cropper = None
        self._loc = None
        # crop_batch's boxes, (x, y, w, h) in native video px, one per frame.
        self.last_boxes: list[tuple[float, float, float, float]] = []

        if self.method == "background":
            if bg_path:
                loaded = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    self.bg = loaded.astype(np.float32)
                else:
                    # imread returns None for a missing/unreadable file; rebuild
                    # the background from the video rather than crash.
                    print(f"[crop] background not readable at {bg_path}; "
                          f"computing it from the video instead.", flush=True)
            if self.bg is None:
                if not video_path:
                    raise ValueError("background crop needs a bg_path or a video to compute one from")
                self.bg = get_background_avg(video_path, num_frames_bg)
            self.bg = resize_frame(self.bg, self.resize_factor)
            self._bg_cropper = FindRatBoundingBox(self.bg, **rest)
        elif self.method == "localizer":
            from localizer import PRESENCE_THRESHOLD, LocalizerCropper
            path = extras.get("localizer_path")  # None -> the bundled weights
            loc_kw = dict(box_size=rest.get("box_size", 220),
                          smoothing=rest.get("smoothing", 0.5),
                          deadband=rest.get("deadband", 0.0),
                          presence_threshold=extras.get("presence_threshold",
                                                        PRESENCE_THRESHOLD))
            if localizer is not None:
                for k, v in loc_kw.items():
                    setattr(localizer, k, v)
                self._loc = localizer
            else:
                # The runtime is a crop setting, not an inference one, so
                # preprocessing and inference locate with the same engine.
                self._loc = LocalizerCropper(
                    path, **loc_kw, runtime=extras.get("localizer_runtime", "torch"),
                    precision=extras.get("localizer_precision", "fp16"), on_status=on_status)

    # -- helpers --------------------------------------------------------------
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_factor == 1.0:
            return frame
        if self.bg is not None:
            # Exactly the background's dims, so detection and crop operate on
            # identically sized images.
            return cv2.resize(frame, (self.bg.shape[1], self.bg.shape[0]),
                              interpolation=cv2.INTER_AREA)
        return cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor,
                          interpolation=cv2.INTER_AREA)

    def reset(self) -> None:
        if self._bg_cropper is not None:
            self._bg_cropper.smoothed_center = None
        if self._loc is not None:
            self._loc.reset()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        if self._loc is not None:
            self._loc.close()

    # -- main entry point -----------------------------------------------------
    def prepare(self, frames: list):
        """Start the per-frame work that doesn't depend on crop state — the
        full-frame resize, and the localizer's input downscale — on thread
        pools, for a batch that will be passed to :meth:`crop_batch` next. Lets
        callers overlap it with the current batch's model work. Same calls,
        same pixels; only where and when they run changes."""
        resized = ([self._pool.submit(self._resize, f) for f in frames]
                   if self.resize_factor != 1.0 else None)
        loc = self._loc.prepare(frames) if self.method == "localizer" else None
        return resized, loc

    def crop_batch(self, frames: list, resets: list | None = None, prepared=None) -> list:
        """Crop consecutive *wanted* frames. ``resets[i]`` True resets the
        smoothing before frame i (a gap in the frame indices). ``prepared`` is
        this batch's :meth:`prepare` result, if the caller started it early.
        Returns crops in the frames' own layout (BGR, or 2-D gray from the
        NVDEC decoder); each crop's box, in native video pixels, is left in
        ``self.last_boxes``."""
        resets = resets or [False] * len(frames)
        resized_f, loc_prepared = prepared if prepared is not None else (None, None)
        if self.method == "localizer":
            # The network is stateless, so locate the whole batch at once; only
            # the smoothing below is sequential.
            centers, pres = self._loc.locate(frames, prepared=loc_prepared)
        out, boxes = [], []
        for i, frame in enumerate(frames):
            if resets[i]:
                self.reset()
            resized = resized_f[i].result() if resized_f is not None else self._resize(frame)
            if self.method == "none":
                out.append(resized)
                rect = (0, 0, resized.shape[1], resized.shape[0])
            elif self.method == "background":
                out.append(self._bg_cropper(resized)[-1])
                rect = self._bg_cropper.last_crop_rect
            else:
                crop, rect = self._loc.crop(resized, centers[i], pres[i], frame.shape)
                out.append(crop)
            # Crops are taken from the resized frame; report the box in the
            # video's own pixels so it can be drawn over the original.
            sx = frame.shape[1] / resized.shape[1]
            sy = frame.shape[0] / resized.shape[0]
            x, y, w, h = rect
            boxes.append((x * sx, y * sy, w * sx, h * sy))
        self.last_boxes = boxes
        return out


def iter_wanted(batches, is_wanted, cropper: FrameCropper | None = None):
    """Decoded batches -> (indices, frames, resets) of the wanted frames only,
    tracking native frame indices from 0. ``resets[i]`` marks a gap before
    frame i. With a ``cropper``, each item also carries its
    :meth:`FrameCropper.prepare` handle, started one batch ahead so it runs
    while the caller is busy with the previous batch."""
    frame_index = 0
    last = None

    def select(batch):
        nonlocal frame_index, last
        idxs, frames, resets = [], [], []
        for frame in batch:
            if is_wanted(frame_index):
                idxs.append(frame_index)
                frames.append(frame)
                resets.append(last is not None and frame_index != last + 1)
                last = frame_index
            frame_index += 1
        return idxs, frames, resets

    ahead = None
    for batch in batches:
        item = select(batch)
        prepared = cropper.prepare(item[1]) if cropper is not None else None
        if ahead is not None:
            yield ahead
        ahead = (*item, prepared)
    if ahead is not None:
        yield ahead


def to_rgb(crop: np.ndarray) -> np.ndarray:
    """A crop (BGR, or 2-D gray) -> RGB, as training's cv2.imread + BGR2RGB sees it."""
    if crop.ndim == 2:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

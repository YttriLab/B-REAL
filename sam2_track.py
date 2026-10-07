"""SAM2 teacher tracks: where the animal's body is, frame by frame, for
training a localizer (localizer_train.py).

SAM2 (Meta's Segment Anything 2, run through Hugging Face ``transformers``)
segments whatever a click lands on and tracks it through a video. It is
accurate but slow (~8-15 fps on a desktop GPU), so it labels a few short clips
and a small network (localizer.py) learns to reproduce its body centres at
hundreds of frames per second.

The recipe is the one the bundled localizer was distilled with:
``sam2.1-hiera-tiny``, one positive click on a clip's first frame, then the
centre of the mask after eroding away its thin parts (tail, tether cable).
The erosion kernel is ``TAIL_EROSION * sqrt(mask area)``, which works out to
the 60 px used on the bundled rig (~80k px mice at 2248x2064), but scales
with the animal for other rigs.

**Which mask a click means.** One click is ambiguous: SAM2 proposes three
masks (roughly sub-part, part, whole) and its video model keeps the one with
the best predicted IoU, which can be a patch of fur on the animal's back
(seen on a rat: 3k px instead of ~70k, then tracked for the whole clip). So the
click is first scored by the image model, each candidate is cut down to the
connected piece the click is on (the "whole" candidate can carry speckles
scattered over a textured floor, which made the tracker lose the animal within
two frames), the largest piece under ``MAX_MASK_FRACTION`` of the frame is
taken as the animal, and the video model is seeded with that mask. On the
bundled rig this picks the same mask the original run's best-IoU choice did.

Two more traps, both found by checking this port against the original
Ultralytics run (median 3.3 px apart over 1500 frames once fixed):

- bfloat16 breaks the first-click mask (a 4k px patch instead of the 95k px
  mouse), so the model runs in float32.
- A streaming session keeps every frame and every frame's memory features
  unless pruned. Pruning needs explicit frame indices: the session otherwise
  numbers a new frame ``len(frames kept)``, so it reuses one index and the
  mask freezes.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, NamedTuple, Optional

import cv2
import numpy as np

SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}
DEFAULT_MODEL = "tiny"
TAIL_EROSION = 0.2
# Masks are post-processed at 1/WORK_DOWNSCALE of native resolution: the
# erosion and centroid don't need 4.6 MP, and full-res morphology would cost
# more than SAM2 itself.
WORK_DOWNSCALE = 4
# SAM2 attends to the last 6 frames' mask memories and up to 16 frames of
# object pointers; keep a little more than that, drop the rest.
_HISTORY = 20
# A seed candidate bigger than this share of the frame is background, not an animal.
MAX_MASK_FRACTION = 0.2

FLAG_OK, FLAG_NO_TAIL, FLAG_LOST = 0, 1, 2
# Bump when the labelling recipe changes, so cached clip labels are redone.
LABEL_VERSION = 3


class TrackPoint(NamedTuple):
    x: float      # body centre, native px (nan when lost)
    y: float
    area: int     # full mask area, native px
    flag: int     # FLAG_OK, FLAG_NO_TAIL (erosion removed everything; whole mask used), FLAG_LOST


def sam2_status() -> tuple[bool, str]:
    """-> (usable, detail). SAM2 is optional: only Train Localizer needs it."""
    try:
        import transformers
        from transformers import Sam2VideoModel, Sam2VideoProcessor  # noqa: F401
    except Exception as e:
        return False, (f"SAM2 needs Hugging Face transformers >= 4.56 "
                       f"({type(e).__name__}: {e}); `pixi install` adds it.")
    return True, f"transformers {transformers.__version__}"


def body_center(mask: np.ndarray, erosion: float = TAIL_EROSION):
    """Binary mask -> (centre (x, y) in mask px or None, area, flag).

    Opening with a kernel scaled to the animal removes thin appendages; the
    largest surviving blob is the body. If nothing survives (a very thin or
    tiny mask) the whole mask's centroid is used and flagged."""
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return None, 0, FLAG_LOST
    k = max(3, int(round(erosion * np.sqrt(area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.erode(m, kernel), connectivity=8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        body, flag = cv2.dilate((labels == best).astype(np.uint8), kernel), FLAG_OK
    else:
        body, flag = m, FLAG_NO_TAIL
    ys, xs = np.nonzero(body)
    return (float(xs.mean()), float(ys.mean())), area, flag


def clicked_piece(mask: np.ndarray, xy, search: int = 15) -> np.ndarray:
    """The connected part of ``mask`` under the click (or, if the click is
    just off it, the part most of a small window around the click falls on)."""
    m = mask.astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    x = int(np.clip(round(xy[0]), 0, m.shape[1] - 1))
    y = int(np.clip(round(xy[1]), 0, m.shape[0] - 1))
    lab = labels[y, x]
    if lab == 0:
        win = labels[max(0, y - search):y + search + 1, max(0, x - search):x + search + 1]
        hits = win[win > 0]
        if not len(hits):
            return np.zeros(mask.shape, bool)
        lab = np.bincount(hits).argmax()
    return labels == lab


def _rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=2)
    return np.ascontiguousarray(frame[..., ::-1])


class Sam2Tracker:
    """One click -> a body-centre track through a stream of frames.

    The model loads on first use (and downloads into the Hugging Face cache
    the very first time: ~150 MB for tiny, ~900 MB for large)."""

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 erosion: float = TAIL_EROSION):
        self.model_id = SAM2_MODELS.get(model, model)  # a short name, HF repo id or local dir
        self.device = device
        self.erosion = erosion
        self._model = None
        self._processor = None
        self._image_model = None
        self._image_processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        ok, detail = sam2_status()
        if not ok:
            raise RuntimeError(detail)
        import torch
        from transformers import (Sam2Model, Sam2Processor, Sam2VideoModel,
                                  Sam2VideoProcessor)
        from transformers.utils import logging as hf_logging
        self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # "sam2_video checkpoint into Sam2Model" is expected
        try:
            self._processor = Sam2VideoProcessor.from_pretrained(self.model_id)
            self._model = Sam2VideoModel.from_pretrained(self.model_id).to(
                self.device, dtype=torch.float32).eval()
            # Same checkpoint, image head: it returns all three candidate masks.
            self._image_processor = Sam2Processor.from_pretrained(self.model_id)
            self._image_model = Sam2Model.from_pretrained(self.model_id).to(
                self.device, dtype=torch.float32).eval()
        finally:
            hf_logging.set_verbosity(level)

    def close(self) -> None:
        """Free the GPU memory before something else (training) needs it."""
        self._model = self._processor = self._image_model = self._image_processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def seed_mask(self, frame: np.ndarray, seed_xy) -> tuple[np.ndarray, list[int]]:
        """A click -> (the animal's mask at native res, all candidate areas):
        the largest of SAM2's candidates under MAX_MASK_FRACTION of the frame
        (the best-IoU one if every candidate is that big)."""
        import torch
        self.load()
        proc = self._image_processor
        inputs = proc(images=_rgb(frame), input_points=[[[[float(seed_xy[0]), float(seed_xy[1])]]]],
                      input_labels=[[[1]]], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._image_model(**inputs, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0]
        masks = [clicked_piece(m, seed_xy) for m in masks.numpy().astype(bool)]  # 3 x (H, W)
        areas = [int(m.sum()) for m in masks]
        ok = [i for i, a in enumerate(areas) if 0 < a <= MAX_MASK_FRACTION * masks[0].size]
        pick = max(ok, key=lambda i: areas[i]) if ok else int(out.iou_scores[0, 0].argmax())
        return masks[pick], areas

    def _segment(self, frames: Iterable[np.ndarray], seed_xy) -> Iterator[tuple]:
        """-> per frame: (mask at 1/WORK_DOWNSCALE res (uint8), native (h, w), object present)."""
        import torch
        import torch.nn.functional as F
        self.load()
        proc, model = self._processor, self._model
        session = proc.init_video_session(inference_device=self.device, dtype=torch.float32)
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            inputs = proc(images=_rgb(frame), device=self.device, return_tensors="pt")
            if i == 0:
                mask0, _areas = self.seed_mask(frame, seed_xy)
                session.video_height, session.video_width = h, w
                proc.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1,
                                                     input_masks=mask0)
            with torch.inference_mode():
                out = model(inference_session=session, frame_idx=i, frame=inputs.pixel_values[0])
                size = (max(1, h // WORK_DOWNSCALE), max(1, w // WORK_DOWNSCALE))
                logits = F.interpolate(out.pred_masks[0:1].float(), size, mode="bilinear",
                                       align_corners=False)[0, 0]
                mask = (logits > 0).to(torch.uint8).cpu().numpy()
                present = float(out.object_score_logits.reshape(-1)[0]) > 0
            old = i - _HISTORY
            if old > 0:  # never frame 0: it holds the click (the conditioning frame)
                session.processed_frames.pop(old, None)
                for d in session.output_dict_per_obj.values():
                    d["non_cond_frame_outputs"].pop(old, None)
                for d in session.frames_tracked_per_obj.values():
                    d.pop(old, None)
            yield mask, (h, w), present

    def track(self, frames: Iterable[np.ndarray], seed_xy,
              cancel: Optional[Callable[[], bool]] = None) -> Iterator[TrackPoint]:
        """Yield one TrackPoint per frame; ``seed_xy`` is a click on the first
        frame, native px. Stops early if ``cancel()`` turns true."""
        for mask, (h, w), present in self._segment(frames, seed_xy):
            if cancel is not None and cancel():
                return
            centre, area, flag = body_center(mask, self.erosion) if present else (None, 0, FLAG_LOST)
            if centre is None:
                yield TrackPoint(float("nan"), float("nan"), 0, FLAG_LOST)
                continue
            sx, sy = w / mask.shape[1], h / mask.shape[0]
            yield TrackPoint((centre[0] + 0.5) * sx - 0.5, (centre[1] + 0.5) * sy - 0.5,
                             int(round(area * sx * sy)), flag)

    def segment_one(self, frame: np.ndarray, seed_xy):
        """A single click on a single frame -> (TrackPoint, mask at native res).
        For checking a seed before committing a clip to it."""
        mask, (h, w), present = next(iter(self._segment([frame], seed_xy)))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if present else np.zeros((h, w), np.uint8)
        centre, area, flag = body_center(mask, self.erosion)
        if centre is None:
            return TrackPoint(float("nan"), float("nan"), 0, FLAG_LOST), mask
        return TrackPoint(centre[0], centre[1], area, flag), mask

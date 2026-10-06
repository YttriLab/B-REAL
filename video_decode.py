"""Sequential video decoding for preprocessing and inference: cv2 or NVDEC.

Everything downstream (labels, extracted frame filenames, inference output) is
indexed by native frame number, and a frame that's decoded differently crops
differently, so both decoders have to produce exactly what
``cv2.VideoCapture.read()`` produces.

- ``"cpu"``: cv2. Yields BGR frames (H, W, 3).
- ``"nvdec"``: GPU decode through PyNvVideoCodec. Monochrome sources only: on a
  video whose chroma planes are a constant 128 (the usual h264 encode of a mono
  camera), every channel of cv2's yuv420p -> BGR conversion is a fixed function
  of luma, ``LUT[Y]``. So the GPU path is NVDEC -> NV12 luma plane -> LUT gather
  -> pinned host buffer, and it yields 2-D gray frames (H, W) that equal each
  channel of cv2's BGR frame bit-for-bit. Croppers and transforms accept either
  shape, and a gray crop saved as PNG reads back as the same BGR image.
- ``"auto"``: NVDEC if it's importable, the GPU has a decoder, and the first
  frames match cv2 exactly; otherwise cv2 (and the reason is recorded).

Decoding runs on a background thread a few batches ahead, so it overlaps the
crop + model work of the consumer.

Ported from the label_video pipeline (video_processing/nvdec_reader.py).
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional

import cv2
import numpy as np

DECODERS = ("auto", "nvdec", "cpu")

# swscale yuv420p (tv range) -> bgr24, any channel, at U=V=128. Tabulated rather
# than computed because swscale's fixed-point math differs from
# round(255/219*(Y-16)) by 1 at 108 of 256 values. derive_cv2_lut() regenerates it.
CV2_GRAY_LUT = np.array(
    [0] * 17 + [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31,
     32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64,
     65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96,
     97, 98, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 121, 122, 123,
     124, 125, 126, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 149,
     150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174,
     175, 176, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 199, 200,
     201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225,
     227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251,
     252, 253] + [255] * 21, dtype=np.uint8)
assert CV2_GRAY_LUT.shape == (256,)


class NotMonochrome(RuntimeError):
    """The video has real chroma, so cv2's BGR isn't a function of luma alone."""


def derive_cv2_lut(workdir: str = "/tmp") -> np.ndarray:
    """Regenerate CV2_GRAY_LUT: losslessly encode a 0..255 luma ramp (chroma
    128) with ffmpeg and read it back with cv2."""
    import os
    import subprocess
    w, h = 256, 64
    frame = np.concatenate([np.tile(np.arange(256, dtype=np.uint8), h),
                            np.full(w * h // 2, 128, np.uint8)])
    path = os.path.join(workdir, "luma_ramp.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", "-c:v", "libx264", "-qp", "0",
                    "-color_range", "tv", path], input=frame.tobytes(), check=True)
    ok, f = cv2.VideoCapture(path).read()
    return f[h // 2, :, 0].copy()


# ── producers ───────────────────────────────────────────────────────────────
# Each takes (video, batch_size, max_frames, stop) and yields lists of frames.

def _produce_cpu(video, batch_size, max_frames, stop):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video}")
    n = 0
    try:
        while not stop.is_set():
            batch = []
            while len(batch) < batch_size and (max_frames is None or n < max_frames):
                ok, f = cap.read()
                if not ok:
                    break
                batch.append(f)
                n += 1
            if not batch:
                return
            yield batch
            if len(batch) < batch_size:
                return
    finally:
        cap.release()


def _produce_nvdec(video, batch_size, max_frames, stop, *, gpu_id=0, ring_size=6):
    import PyNvVideoCodec as nvc
    import torch

    dec = nvc.ThreadedDecoder(video, buffer_size=2 * batch_size, gpu_id=gpu_id,
                              output_color_type=nvc.OutputColorType.NATIVE)
    try:
        meta = dec.get_stream_metadata()
        h, w = meta.height, meta.width
        dev = torch.device("cuda", gpu_id)
        lut = torch.from_numpy(CV2_GRAY_LUT).to(dev)
        stream = torch.cuda.Stream(device=dev)
        # Yielded frames are views into this ring of pinned buffers. A batch's
        # buffer is reused ring_size batches later, which is longer than the
        # prefetch queue plus the batch the consumer is working on.
        ring = [torch.empty((batch_size, h, w), dtype=torch.uint8).pin_memory()
                for _ in range(ring_size)]
        gpu = torch.empty((batch_size, h, w), dtype=torch.uint8, device=dev)
        n = k = 0
        with torch.cuda.stream(stream):
            while not stop.is_set() and (max_frames is None or n < max_frames):
                want = batch_size if max_frames is None else min(batch_size, max_frames - n)
                frames = dec.get_batch_frames(want)
                if not frames:
                    return
                for i, fr in enumerate(frames):
                    if fr.format != nvc.Pixel_Format.NV12:
                        raise NotMonochrome(f"decoder output is {fr.format}, expected NV12")
                    y_plane, uv_plane = fr.cuda()
                    y = torch.as_tensor(y_plane, device=dev)[..., 0]
                    if n == 0 and i == 0 and not bool(
                            (torch.as_tensor(uv_plane, device=dev) == 128).all()):
                        raise NotMonochrome("chroma isn't constant 128 (colour video?)")
                    torch.index_select(lut, 0, y.reshape(-1).int(), out=gpu[i].view(-1))
                host = ring[k % ring_size][:len(frames)]
                host.copy_(gpu[:len(frames)], non_blocking=True)
                stream.synchronize()  # host filled; decoder surfaces released
                yield list(host.numpy())
                n += len(frames)
                k += 1
                if len(frames) < want:
                    return
    finally:
        dec.end()


_PRODUCERS = {"cpu": _produce_cpu, "nvdec": _produce_nvdec}


def iter_batches(video: str, decoder: str = "cpu", batch_size: int = 32,
                 max_frames: Optional[int] = None, prefetch: int = 3) -> Iterator[list]:
    """Yield lists of consecutive frames from frame 0, decoded ``prefetch``
    batches ahead on a background thread. ``decoder`` must be a concrete
    "cpu"/"nvdec" (see :func:`resolve_decoder`). Breaking out of the loop (or
    closing the generator) stops the decode thread."""
    if decoder not in _PRODUCERS:
        raise ValueError(f"decoder must be 'cpu' or 'nvdec' here, got {decoder!r}")
    produce = _PRODUCERS[decoder]
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    _END = object()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def run():
        try:
            for batch in produce(video, batch_size, max_frames, stop):
                put(batch)
            put(_END)
        except BaseException as e:  # surface decode errors in the consumer
            put(e)

    t = threading.Thread(target=run, daemon=True, name=f"decode-{decoder}")
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=5)


def iter_frames(video: str, decoder: str = "cpu", **kw) -> Iterator[np.ndarray]:
    """Frame-at-a-time view of :func:`iter_batches`."""
    for batch in iter_batches(video, decoder, **kw):
        yield from batch


# ── decoder selection ──────────────────────────────────────────────────────

def check_nvdec_matches_cv2(video: str, n: int = 16) -> tuple[bool, str]:
    """Bit-exact check of the first ``n`` frames, NVDEC vs cv2. -> (ok, message)."""
    cap = cv2.VideoCapture(video)
    ref = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        ref.append(f)
    cap.release()
    if not ref:
        return False, "cv2 could not read any frames"
    if any(not (np.array_equal(f[..., 0], f[..., 1]) and np.array_equal(f[..., 0], f[..., 2]))
           for f in ref):
        return False, "video is colour (BGR channels differ); NVDEC path is monochrome-only"
    try:
        got = list(iter_frames(video, "nvdec", batch_size=len(ref), max_frames=len(ref)))
    except NotMonochrome as e:
        return False, str(e)
    if len(got) != len(ref):
        return False, f"NVDEC returned {len(got)} of the first {len(ref)} frames"
    bad = [i for i, (g, f) in enumerate(zip(got, ref)) if not np.array_equal(g, f[..., 0])]
    if bad:
        return False, f"frames {bad} differ from cv2"
    return True, f"first {len(ref)} frames bit-identical to cv2"


def resolve_decoder(decoder: str, video: str) -> tuple[str, str]:
    """Map "auto"/"nvdec"/"cpu" to a concrete decoder. -> (decoder, reason).

    "nvdec" raises if NVDEC can't reproduce cv2 on this video; "auto" falls
    back to cv2 instead."""
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
    if decoder == "cpu":
        return "cpu", "cv2 (requested)"
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        import PyNvVideoCodec  # noqa: F401
        ok, msg = check_nvdec_matches_cv2(video)
    except Exception as e:  # not installed, no NVDEC, unsupported codec, ...
        ok, msg = False, f"{type(e).__name__}: {e}"
    if ok:
        return "nvdec", f"NVDEC ({msg})"
    if decoder == "nvdec":
        raise RuntimeError(f"NVDEC decoding unusable for {video}: {msg}")
    return "cpu", f"cv2 (NVDEC unusable: {msg})"

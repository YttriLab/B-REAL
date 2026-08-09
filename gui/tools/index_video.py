"""Generate synthetic videos whose every frame encodes its own frame index.

Used to prove *frame-exact* random access: we can decode any frame and read back
which frame index it claims to be, so we know the reader landed on exactly the
frame we asked for (not merely a consistent-but-wrong neighbour).

The index is drawn as a row of large high-contrast luma blocks (a binary code),
which survives H.264 chroma subsampling and lossy quantisation. A moving square
is added lower down purely so the frames look different to a human during manual
GUI testing.
"""

from __future__ import annotations

import av
import numpy as np

_HI = 235  # "1" luma (safe TV range)
_LO = 16   # "0" luma


def make_index_frame(i: int, w: int, h: int, nbits: int = 14) -> np.ndarray:
    """Return an (h, w, 3) uint8 RGB frame encoding integer `i`."""
    arr = np.full((h, w, 3), _LO, dtype=np.uint8)
    bw = w // nbits
    band_h = max(1, h // 4)
    for k in range(nbits):
        if (i >> k) & 1:
            x0 = k * bw
            x1 = w if k == nbits - 1 else (k + 1) * bw
            arr[0:band_h, x0:x1, :] = _HI
    # Moving marker (visual variety only; ignored by the decoder).
    sq = max(4, h // 12)
    span = max(1, w - sq)
    cx = (i * 7) % span
    y0 = h - band_h
    arr[y0:y0 + sq, cx:cx + sq, :] = 200
    return arr


def read_index_frame(arr: np.ndarray, nbits: int = 14) -> int:
    """Recover the integer encoded by `make_index_frame` from a decoded frame."""
    h, w, _ = arr.shape
    bw = w // nbits
    band_h = max(1, h // 4)
    y = band_h // 2
    val = 0
    for k in range(nbits):
        x = k * bw + bw // 2
        if int(arr[y, x, :].mean()) > 128:
            val |= (1 << k)
    return val


def write_index_video(
    path,
    num_frames: int,
    w: int = 320,
    h: int = 240,
    fps: int = 30,
    gop: int = 12,
    codec: str | None = None,
    nbits: int = 14,
) -> str:
    """Write an index-marked video. Returns the codec name actually used.

    A small GOP (default 12) plus H.264's default B-frames ensures multiple
    keyframes and out-of-order coding, so the reader's keyframe-seek and
    display-order logic are genuinely exercised.
    """
    candidates = [codec] if codec else ["libx264", "mpeg4"]
    last_err: Exception | None = None
    for cname in candidates:
        try:
            container = av.open(str(path), mode="w")
            try:
                stream = container.add_stream(cname, rate=fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                stream.gop_size = gop
                if cname == "libx264":
                    stream.options = {"crf": "18", "g": str(gop)}
                for i in range(num_frames):
                    img = make_index_frame(i, w, h, nbits)
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():  # flush encoder
                    container.mux(packet)
            finally:
                container.close()
            return cname
        except Exception as e:  # noqa: BLE001 - try the next codec
            last_err = e
            continue
    raise RuntimeError(f"no usable encoder among {candidates}: {last_err}")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    used = write_index_video(out, n)
    print(f"wrote {n} frames to {out} using {used}")

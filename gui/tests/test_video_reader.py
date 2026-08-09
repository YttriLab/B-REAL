"""Phase 0 gate: prove frame-exact random access with the PyAV VideoReader.

Runnable two ways:
    .venv-gui/Scripts/python.exe gui/tests/test_video_reader.py     # prints PASS/FAIL
    pytest gui/tests/test_video_reader.py                            # if pytest installed

Each frame of the synthetic videos encodes its own index (see tools/index_video),
so `read_index_frame(reader.get_frame(i)) == i` proves the reader landed on
exactly frame i -- the core claim of the whole rewrite (frame-accurate labeling
without WebCodecs).
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.video_reader import VideoReader  # noqa: E402
from tools.index_video import read_index_frame, write_index_video  # noqa: E402

N = 150
FPS = 30
W, H = 320, 240


def _make(tmpdir, ext):
    path = os.path.join(tmpdir, f"clip{ext}")
    used = write_index_video(path, N, w=W, h=H, fps=FPS, gop=12)
    return path, used


def _check_container(path, codec_used):
    results = []

    def ok(name, cond):
        results.append((name, bool(cond)))

    with VideoReader(path) as r:
        ok("frame count", r.num_frames == N)
        ok("fps detect", abs(r.fps - FPS) < 0.5)
        ok("dimensions", (r.width, r.height) == (W, H))

        # 1. Sequential forward access (exercises the running-decode fast path).
        bad = [i for i in range(N) if read_index_frame(r.get_frame(i)) != i]
        ok("sequential frame-exact", not bad)

        # 2. Random access incl. backward jumps (exercises keyframe re-seek).
        order = list(range(N))
        random.Random(1234).shuffle(order)
        bad_rand = [i for i in order if read_index_frame(r.get_frame(i)) != i]
        ok("random-access frame-exact", not bad_rand)

        # 3. Boundaries.
        ok("first frame", read_index_frame(r.get_frame(0)) == 0)
        ok("last frame", read_index_frame(r.get_frame(N - 1)) == N - 1)

        # 4. Determinism: same index twice -> identical pixels.
        a, b = r.get_frame(73), r.get_frame(73)
        ok("repeat identical", np.array_equal(a, b))

        # 5. Seek-vs-sequential agreement: a fresh reader that only random-seeks
        #    to frame 99 must match a reader that decoded straight through to 99.
        with VideoReader(path) as r2:
            seek_only = r2.get_frame(99).copy()
        with VideoReader(path) as r3:
            for i in range(100):
                seq = r3.get_frame(i)
            ok("seek == sequential", np.array_equal(seek_only, seq))

        # 6. Manual FPS lock overrides detection + time mapping.
        r.set_fps_override(60)
        ok("fps override", r.fps == 60 and abs(r.frame_to_ms(60) - 1000.0) < 1e-6)
        r.set_fps_override(None)
        ok("ms<->frame roundtrip", r.ms_to_frame(r.frame_to_ms(42)) == 42)

    return results


def _throughput(path):
    with VideoReader(path, cache_size=8) as r:
        t0 = time.perf_counter()
        for i in range(r.num_frames):
            r.get_frame(i)
        seq = (time.perf_counter() - t0) / r.num_frames * 1000.0

        idxs = [random.Random(7).randrange(r.num_frames) for _ in range(60)]
        t0 = time.perf_counter()
        for i in idxs:
            r._cache.clear()  # force a real seek+decode each time
            r.get_frame(i)
        rnd = (time.perf_counter() - t0) / len(idxs) * 1000.0
    return seq, rnd


# ------- pytest entry points -------
def test_mp4_frame_exact():
    with tempfile.TemporaryDirectory() as d:
        path, used = _make(d, ".mp4")
        for name, passed in _check_container(path, used):
            assert passed, f"MP4 [{used}]: {name} failed"


def test_mkv_frame_exact_without_remux():
    with tempfile.TemporaryDirectory() as d:
        path, used = _make(d, ".mkv")
        for name, passed in _check_container(path, used):
            assert passed, f"MKV [{used}]: {name} failed"


def _main():
    all_ok = True
    with tempfile.TemporaryDirectory() as d:
        for ext in (".mp4", ".mkv"):
            path, used = _make(d, ext)
            print(f"\n=== {ext}  (codec: {used}) ===")
            for name, passed in _check_container(path, used):
                print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
                all_ok = all_ok and passed
            seq, rnd = _throughput(path)
            print(f"  decode: {seq:.2f} ms/frame sequential, "
                  f"{rnd:.2f} ms/frame random-seek "
                  f"(~{1000.0/seq:.0f} fps sequential)")
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())

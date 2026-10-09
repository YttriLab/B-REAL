"""Has a video been preprocessed for its current labels? One status per video.

Preprocessing writes only labeled frames (each bout plus frames_per_group-1
lead-in frames) to ``frames/<id>/<native index>.png``, so "preprocessed" means
those frames are all on disk — for the labels as they are now. Labeling more
after preprocessing leaves the video partly done, and training silently skips
the frames that aren't there (data_coverage.py), so partial is its own state.

Shown in the Videos list, the Preprocess tab and the Train tab's video
checklist. Listing a frames dir is cheap but not free (a network mount, 160k
files), so the set of frames on disk is cached until the directory changes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from labeling_shell.frame_ranges import compute_labeled_frame_ranges

from .crop_summary import METHOD_NAMES, read_frames_record

READY, PARTIAL, MISSING, UNLABELED = "ready", "partial", "missing", "unlabeled"
COLORS = {READY: "#3fb950", PARTIAL: "#e3b341", MISSING: "#f85149", UNLABELED: "#8b949e"}


@dataclass
class VideoStatus:
    state: str
    needed: int = 0   # frames the current labels need on disk
    present: int = 0  # of those, how many are
    method: str | None = None  # crop method the frames were made with, if recorded
    mixed: bool = False

    @property
    def color(self) -> str:
        return COLORS[self.state]

    @property
    def short(self) -> str:
        """A few words for a list row."""
        if self.state == UNLABELED:
            return "no labels yet"
        if self.state == MISSING:
            return "not preprocessed"
        method = f" · {METHOD_NAMES.get(self.method, self.method)}" if self.method else ""
        if self.state == PARTIAL:
            return f"preprocessed {self.present:,}/{self.needed:,} frames{method}"
        return f"preprocessed{method}"

    @property
    def detail(self) -> str:
        """A sentence for a tooltip or the Preprocess tab."""
        if self.state == UNLABELED:
            return "No labeled bouts, so nothing to preprocess or train on."
        if self.state == MISSING:
            return (f"Not preprocessed: none of the {self.needed:,} frames the labels "
                    "need are on disk. Run Preprocess on this video.")
        crop = (f"Cropped with {METHOD_NAMES.get(self.method, self.method)}."
                if self.method else "Crop settings not recorded (made before they were saved).")
        if self.mixed:
            crop += " Some frames were re-made with other settings (mixed)."
        if self.state == PARTIAL:
            return (f"Partly preprocessed: {self.present:,} of the {self.needed:,} frames "
                    "the labels need are on disk (labels added since?). Run Preprocess "
                    f"again to add the rest; training skips them until then. {crop}")
        return f"Preprocessed: all {self.needed:,} frames the labels need are on disk. {crop}"


_listing: dict[str, tuple[int, frozenset]] = {}  # frames dir -> (mtime_ns, frame indices)
# A directory's mtime moves in coarse ticks (milliseconds, or seconds on some
# network filesystems), so files added within the tick of a listing leave it
# unchanged. Only a directory that has been still this long is cached.
SETTLED_S = 2.0


def frames_on_disk(frame_dir) -> frozenset:
    """Native indices of the PNGs in a frames dir (cached until it changes)."""
    frame_dir = str(frame_dir)
    try:
        mtime = os.stat(frame_dir).st_mtime_ns
    except OSError:
        return frozenset()
    hit = _listing.get(frame_dir)
    if hit and hit[0] == mtime:
        return hit[1]
    indices = set()
    try:
        with os.scandir(frame_dir) as it:
            for e in it:
                stem, ext = os.path.splitext(e.name)
                if ext.lower() == ".png" and stem.isdigit():
                    indices.add(int(stem))
    except OSError:
        pass
    if time.time() - mtime / 1e9 > SETTLED_S:
        _listing[frame_dir] = (mtime, frozenset(indices))
    else:
        _listing.pop(frame_dir, None)  # still changing: list it again next time
    return frozenset(indices)


def video_status(bouts, frame_dir, frames_per_group: int = 3) -> VideoStatus:
    if not bouts:
        return VideoStatus(UNLABELED)
    needed = set()
    for start, end in compute_labeled_frame_ranges(bouts, frames_per_group):
        needed.update(range(start, end + 1))
    present = len(needed & frames_on_disk(frame_dir)) if frame_dir else 0
    record = read_frames_record(frame_dir) if frame_dir and present else None
    method = ((record or {}).get("crop_params") or {}).get("method") if record else None
    state = READY if present == len(needed) else PARTIAL if present else MISSING
    return VideoStatus(state, len(needed), present, method,
                       bool((record or {}).get("mixed")))


def statuses(sources, frames_per_group: int = 3) -> dict:
    """{video id: VideoStatus} for ctx.video_sources() records."""
    return {s["id"]: video_status(s.get("bouts") or [], s.get("frame_dir"), frames_per_group)
            for s in sources}

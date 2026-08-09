"""Convert bouts into native-frame ranges for preprocessing / inference.

Python port of computeLabeledFrameRanges (plugins/pipeline.js): each bout is
extended backward by frames_per_group-1 (the lead-in frames a clip needs as
temporal context), then overlapping/adjacent ranges are merged. Returned as
inclusive [start, end] pairs in native frame indices, the shape the backend's
preprocess/inference `frame_ranges` expects.
"""

from __future__ import annotations


def compute_labeled_frame_ranges(bouts, frames_per_group: int = 3) -> list[list[int]]:
    lead = max(0, frames_per_group - 1)
    spans = sorted((max(0, b.start_frame - lead), b.end_frame) for b in bouts)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

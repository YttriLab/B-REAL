"""
Pre-training coverage check: how many labeled frames actually have their
preprocessed PNGs (with full clip context) on disk.

Mirrors the frame-existence logic in ``model.load_data`` (flat layout): a labeled
frame ``N`` is "ready" for training only if frames ``N-(T-1) .. N`` all exist as
``{n}.png`` in the frame directory. Frames below ``T-1`` can never form a clip.

Kept dependency-free (stdlib only) so the FastAPI server can call it cheaply as a
pre-flight check without importing the torch/lightning stack.
"""

import csv
import os


def _labeled_frame_indices(label_path):
    """Read the frame indices (first column) from a frame-indexed label CSV."""
    with open(label_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    indices = []
    for row in rows[1:]:  # skip header
        if not row:
            continue
        try:
            indices.append(int(float(row[0])))
        except (ValueError, IndexError):
            continue
    return indices


def compute_coverage(frame_dirs, label_paths, frames_per_group=3):
    """
    For each (frame_dir, label_path) pair, count how many labeled frames are
    "ready" (have a complete preprocessed clip on disk) vs missing.

    Returns a dict::

        {
          "datasets": [
            {"frame_dir", "label_path", "labeled", "ready", "missing",
             "too_early", "not_extracted", "missing_examples", "error"},
            ...
          ],
          "total_labeled", "total_ready", "total_missing",
        }

    where ``missing == too_early + not_extracted``. ``too_early`` frames sit below
    the clip horizon (unfixable by preprocessing); ``not_extracted`` frames are
    the ones a preprocessing run would add. ``missing_examples`` lists up to 10
    not-extracted frame indices to aid debugging.
    """
    if isinstance(frame_dirs, str):
        frame_dirs = [frame_dirs]
        label_paths = [label_paths]

    datasets = []
    for frame_dir, label_path in zip(frame_dirs, label_paths):
        entry = {
            "frame_dir": frame_dir,
            "label_path": label_path,
            "labeled": 0,
            "ready": 0,
            "missing": 0,
            "too_early": 0,
            "not_extracted": 0,
            "missing_examples": [],
            "error": None,
        }
        try:
            frame_nums = _labeled_frame_indices(label_path)
        except FileNotFoundError:
            entry["error"] = "label file not found"
            datasets.append(entry)
            continue
        except Exception as e:  # malformed CSV, encoding, etc.
            entry["error"] = f"{type(e).__name__}: {e}"
            datasets.append(entry)
            continue

        entry["labeled"] = len(frame_nums)
        min_idx = frames_per_group - 1
        dir_exists = os.path.isdir(frame_dir)
        for fn in frame_nums:
            if fn < min_idx:
                entry["too_early"] += 1
                continue
            if dir_exists:
                triplet = [
                    os.path.join(frame_dir, f"{fn - (frames_per_group - 1 - j)}.png")
                    for j in range(frames_per_group)
                ]
                if all(os.path.exists(p) for p in triplet):
                    entry["ready"] += 1
                    continue
            entry["not_extracted"] += 1
            if len(entry["missing_examples"]) < 10:
                entry["missing_examples"].append(fn)
        entry["missing"] = entry["labeled"] - entry["ready"]
        datasets.append(entry)

    return {
        "datasets": datasets,
        "total_labeled": sum(d["labeled"] for d in datasets),
        "total_ready": sum(d["ready"] for d in datasets),
        "total_missing": sum(d["missing"] for d in datasets),
    }

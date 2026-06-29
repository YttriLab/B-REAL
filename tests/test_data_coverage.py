"""Tests for data_coverage.compute_coverage (pre-training frame coverage check)."""

import csv
import os

from data_coverage import compute_coverage


def _write_labels(path, frame_indices, classes=("a", "b")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", *classes])
        for fn in frame_indices:
            w.writerow([fn, 1, 0])


def _touch_frames(dirpath, indices):
    os.makedirs(dirpath, exist_ok=True)
    for i in indices:
        open(os.path.join(dirpath, f"{i}.png"), "wb").close()


class TestComputeCoverage:
    def test_all_ready(self, tmp_path):
        labels = str(tmp_path / "vid.csv")
        frames = str(tmp_path / "vid")
        _write_labels(labels, [2, 3, 4])
        _touch_frames(frames, [0, 1, 2, 3, 4])  # full triplet context present

        cov = compute_coverage([frames], [labels], frames_per_group=3)
        d = cov["datasets"][0]
        assert d["labeled"] == 3
        assert d["ready"] == 3
        assert d["missing"] == 0
        assert cov["total_ready"] == 3

    def test_missing_context_frames_not_ready(self, tmp_path):
        labels = str(tmp_path / "vid.csv")
        frames = str(tmp_path / "vid")
        _write_labels(labels, [2, 3, 4])
        _touch_frames(frames, [2, 3, 4])  # the leading context (0,1) was not extracted

        cov = compute_coverage([frames], [labels], frames_per_group=3)
        d = cov["datasets"][0]
        # Only frame 4 has its full triplet (2,3,4); 2 and 3 are missing context.
        assert d["ready"] == 1
        assert d["not_extracted"] == 2
        assert d["missing"] == 2
        assert set(d["missing_examples"]) == {2, 3}

    def test_frames_below_clip_horizon_are_too_early(self, tmp_path):
        labels = str(tmp_path / "vid.csv")
        frames = str(tmp_path / "vid")
        _write_labels(labels, [0, 1, 2])
        _touch_frames(frames, [0, 1, 2])

        cov = compute_coverage([frames], [labels], frames_per_group=3)
        d = cov["datasets"][0]
        assert d["too_early"] == 2  # frames 0,1 can't form a 3-clip
        assert d["ready"] == 1      # frame 2 can
        assert d["not_extracted"] == 0
        assert d["missing"] == d["too_early"] + d["not_extracted"]

    def test_missing_frame_dir(self, tmp_path):
        labels = str(tmp_path / "vid.csv")
        _write_labels(labels, [2, 3])
        cov = compute_coverage([str(tmp_path / "does_not_exist")], [labels], frames_per_group=3)
        d = cov["datasets"][0]
        assert d["ready"] == 0
        assert d["not_extracted"] == 2

    def test_missing_label_file_reports_error(self, tmp_path):
        cov = compute_coverage([str(tmp_path)], [str(tmp_path / "nope.csv")])
        d = cov["datasets"][0]
        assert d["error"] == "label file not found"
        assert d["labeled"] == 0

    def test_totals_aggregate_across_datasets(self, tmp_path):
        l1, f1 = str(tmp_path / "a.csv"), str(tmp_path / "a")
        l2, f2 = str(tmp_path / "b.csv"), str(tmp_path / "b")
        _write_labels(l1, [2, 3]); _touch_frames(f1, [0, 1, 2, 3])      # both ready
        _write_labels(l2, [2, 3]); _touch_frames(f2, [3])               # neither ready

        cov = compute_coverage([f1, f2], [l1, l2], frames_per_group=3)
        assert cov["total_labeled"] == 4
        assert cov["total_ready"] == 2
        assert cov["total_missing"] == 2

    def test_missing_examples_capped_at_ten(self, tmp_path):
        labels = str(tmp_path / "vid.csv")
        frames = str(tmp_path / "vid")
        _write_labels(labels, list(range(2, 30)))
        os.makedirs(frames, exist_ok=True)  # empty dir → all not extracted
        cov = compute_coverage([frames], [labels], frames_per_group=3)
        assert len(cov["datasets"][0]["missing_examples"]) == 10

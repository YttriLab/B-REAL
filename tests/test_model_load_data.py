"""
Tests for model.py load_data() function.

Covers flat vs subfolder layout detection, label union, binarization,
boundary distance computation, and multi-dataset merging.
"""

import os

import numpy as np
import pandas as pd
import pytest

from model import load_data


# ── Flat layout ──────────────────────────────────────────────────────────

class TestLoadDataFlat:
    def test_basic_flat_loading(self, synthetic_frames, label_csv_3class):
        paths, labels, class_names, boundary_dists = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[label_csv_3class],
            frames_per_group=3,
        )
        # Nested structure: [dataset][sample]
        assert len(paths) == 1
        assert len(labels) == 1
        assert len(paths[0]) > 0
        assert len(paths[0]) == len(labels[0])
        assert len(class_names) == 3
        assert set(class_names) == {"walk", "groom", "rear"}

    def test_triplet_structure(self, synthetic_frames, label_csv_3class):
        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[label_csv_3class],
            frames_per_group=3,
        )
        # Each sample should be a list of 3 paths
        for triplet in paths[0]:
            assert len(triplet) == 3
            for p in triplet:
                assert p.endswith(".png")
                assert os.path.exists(p)

    def test_triplet_ordering(self, synthetic_frames, label_csv_3class):
        """Triplets should be temporally ordered: [N-2, N-1, N]."""
        paths, _, _, _ = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[label_csv_3class],
            frames_per_group=3,
        )
        for triplet in paths[0]:
            nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in triplet]
            assert nums[1] == nums[0] + 1
            assert nums[2] == nums[1] + 1

    def test_skips_frames_without_labels(self, synthetic_frames, tmp_path):
        """Frames not in the label CSV should be skipped."""
        csv_path = tmp_path / "sparse_labels.csv"
        rows = [
            {"time": 5, "walk": 1, "rest": 0},
            {"time": 10, "walk": 0, "rest": 1},
        ]
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[str(csv_path)],
            frames_per_group=3,
        )
        assert len(paths[0]) == 2  # Only frames 5 and 10 have labels

    def test_frames_per_group_5(self, tmp_path):
        """Test with frames_per_group=5."""
        frame_dir = tmp_path / "frames5"
        frame_dir.mkdir()
        rng = np.random.RandomState(1)
        for i in range(30):
            img = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frame_dir / f"{i}.png"), img)

        csv_path = tmp_path / "labels5.csv"
        rows = []
        for frame_num in range(4, 30):  # min is frames_per_group - 1 = 4
            rows.append({"time": frame_num, "a": frame_num % 2, "b": 1 - frame_num % 2})
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        paths, labels, _, _ = load_data(
            frame_dirs=[str(frame_dir)],
            label_paths=[str(csv_path)],
            frames_per_group=5,
        )
        for triplet in paths[0]:
            assert len(triplet) == 5


# ── Subfolder layout ────────────────────────────────────────────────────

class TestLoadDataSubfolder:
    def test_subfolder_loading(self, synthetic_frames_subfolder, tmp_path):
        csv_path = tmp_path / "labels_sub.csv"
        rows = []
        for frame_num in range(2, 20):
            rows.append({"time": frame_num, "walk": frame_num % 2, "rest": 1 - frame_num % 2})
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames_subfolder],
            label_paths=[str(csv_path)],
            frames_per_group=3,
        )
        assert len(paths[0]) > 0
        for triplet in paths[0]:
            assert len(triplet) == 3


# ── Label union ──────────────────────────────────────────────────────────

class TestLabelUnion:
    def test_union_merges_classes(self, synthetic_frames, tmp_path):
        csv_path = tmp_path / "labels_union.csv"
        rows = []
        for i in range(2, 20):
            rows.append({"time": i, "walk": i % 3 == 0, "groom": i % 3 == 1, "rear": i % 3 == 2})
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[str(csv_path)],
            frames_per_group=3,
            label_union={"exploration": ["groom", "rear"]},
        )
        assert "exploration" in class_names
        assert "groom" not in class_names
        assert "rear" not in class_names
        assert "walk" in class_names

    def test_union_invalid_label_raises(self, synthetic_frames, label_csv_3class):
        with pytest.raises(ValueError, match="not found"):
            load_data(
                frame_dirs=[synthetic_frames],
                label_paths=[label_csv_3class],
                frames_per_group=3,
                label_union={"x": ["nonexistent_behavior"]},
            )


# ── Binarization ────────────────────────────────────────────────────────

class TestBinarization:
    def test_binarize_creates_two_classes(self, synthetic_frames, label_csv_3class):
        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[label_csv_3class],
            frames_per_group=3,
            binarize="walk",
        )
        assert class_names == ["other", "walk"]
        flat = np.array(labels[0])
        assert set(np.unique(flat)).issubset({0, 1})

    def test_binarize_invalid_name_raises(self, synthetic_frames, label_csv_3class):
        with pytest.raises(ValueError, match="not found"):
            load_data(
                frame_dirs=[synthetic_frames],
                label_paths=[label_csv_3class],
                frames_per_group=3,
                binarize="nonexistent",
            )


# ── Multi-dataset ────────────────────────────────────────────────────────

class TestMultiDataset:
    def test_two_datasets(self, synthetic_frames, label_csv_3class, tmp_path):
        # Create a second dataset with a different frame dir
        frame_dir2 = tmp_path / "frames2"
        frame_dir2.mkdir()
        import cv2
        rng = np.random.RandomState(99)
        for i in range(20):
            img = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir2 / f"{i}.png"), img)

        csv_path2 = tmp_path / "labels2.csv"
        rows = []
        for i in range(2, 20):
            rows.append({"time": i, "walk": i % 2, "groom": 0, "rear": 1 - i % 2})
        pd.DataFrame(rows).to_csv(str(csv_path2), index=False)

        paths, labels, class_names, _ = load_data(
            frame_dirs=[synthetic_frames, str(frame_dir2)],
            label_paths=[label_csv_3class, str(csv_path2)],
            frames_per_group=3,
        )
        assert len(paths) == 2
        assert len(labels) == 2
        assert len(paths[0]) > 0
        assert len(paths[1]) > 0

    def test_string_args_backward_compat(self, synthetic_frames, label_csv_3class):
        """Single string args should work (backward compatibility)."""
        paths, labels, class_names, _ = load_data(
            frame_dirs=synthetic_frames,
            label_paths=label_csv_3class,
            frames_per_group=3,
        )
        assert len(paths) == 1
        assert len(paths[0]) > 0


# ── Boundary distances ──────────────────────────────────────────────────

class TestBoundaryDistances:
    def test_boundary_dists_returned(self, synthetic_frames, label_csv_3class):
        _, _, _, boundary_dists = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[label_csv_3class],
            frames_per_group=3,
        )
        assert len(boundary_dists) == 1
        assert len(boundary_dists[0]) > 0
        # All distances should be non-negative integers
        for d in boundary_dists[0]:
            assert d >= 0

    def test_transition_has_zero_distance(self, synthetic_frames, tmp_path):
        """Frames at label transitions should have distance 0 or 1."""
        csv_path = tmp_path / "labels_trans.csv"
        rows = []
        for i in range(2, 20):
            # Transition at frame 10
            label = "walk" if i < 10 else "rest"
            rows.append({
                "time": i,
                "walk": 1 if label == "walk" else 0,
                "rest": 1 if label == "rest" else 0,
            })
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)

        _, labels, _, boundary_dists = load_data(
            frame_dirs=[synthetic_frames],
            label_paths=[str(csv_path)],
            frames_per_group=3,
        )
        # The minimum boundary distance should be small (near the transition)
        assert min(boundary_dists[0]) <= 1

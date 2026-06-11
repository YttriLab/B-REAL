"""
Tests for backend.services.rlt_service.

Covers the lightweight .rlt (HDF5) export/import/inspect round-trip and the
deliberate exclusion of regenerable data (frames, backgrounds, checkpoints).
No GPU or real video required.
"""

import json
import os

import h5py
import pytest

from backend.services.rlt_service import (
    RLT_VERSION,
    export_rlt,
    import_rlt,
    inspect_rlt,
)


# ── export_rlt ─────────────────────────────────────────────────────────────

class TestExportRlt:
    def test_creates_valid_hdf5(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        out = str(tmp_path / "out.rlt")
        export_rlt(project_dir, out)
        assert os.path.exists(out)
        # It must be a real HDF5 file
        assert h5py.is_hdf5(out)

    def test_summary_counts(self, sample_project, tmp_path):
        project_dir, manifest = sample_project
        out = str(tmp_path / "out.rlt")
        summary = export_rlt(project_dir, out)
        assert summary["videos"] == len(manifest["videos"])
        assert summary["labels"] == 2

    def test_writes_version_and_manifest_attrs(self, sample_project, tmp_path):
        project_dir, manifest = sample_project
        out = str(tmp_path / "out.rlt")
        export_rlt(project_dir, out)
        with h5py.File(out, "r") as hf:
            assert int(hf.attrs["rlt_version"]) == RLT_VERSION
            packed = json.loads(hf.attrs["manifest"])
            assert packed["name"] == manifest["name"]
            assert packed["behaviors"] == manifest["behaviors"]

    def test_labels_are_embedded(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        out = str(tmp_path / "out.rlt")
        export_rlt(project_dir, out)
        with h5py.File(out, "r") as hf:
            assert "labels" in hf
            assert set(hf["labels"].keys()) == {"vid_a", "vid_b"}
            content = hf["labels"]["vid_a"][()]
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            assert "grooming" in content

    def test_excludes_regenerable_data(self, sample_project, tmp_path):
        """Frames, backgrounds, and checkpoints must NOT be embedded."""
        project_dir, _ = sample_project
        out = str(tmp_path / "out.rlt")
        export_rlt(project_dir, out)
        with h5py.File(out, "r") as hf:
            assert "frames" not in hf
            assert "backgrounds" not in hf
            assert "models" not in hf

    def test_stays_small(self, sample_project, tmp_path):
        """File should be tiny — only labels + manifest, no frame data."""
        project_dir, _ = sample_project
        out = str(tmp_path / "out.rlt")
        export_rlt(project_dir, out)
        # The checkpoint alone is 1 KB and frames are excluded; a few KB max.
        assert os.path.getsize(out) < 64 * 1024

    def test_missing_project_json_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            export_rlt(str(empty), str(tmp_path / "out.rlt"))


# ── import_rlt ─────────────────────────────────────────────────────────────

class TestImportRlt:
    def test_writes_project_json(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        import_rlt(rlt, str(dest))
        assert (dest / "project.json").exists()

    def test_recreates_label_csvs(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        import_rlt(rlt, str(dest))
        assert (dest / "labels" / "vid_a_bouts.csv").exists()
        assert (dest / "labels" / "vid_b_bouts.csv").exists()
        text = (dest / "labels" / "vid_a_bouts.csv").read_text(encoding="utf-8")
        assert "grooming" in text

    def test_creates_empty_subdirs(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        import_rlt(rlt, str(dest))
        for sub in ("frames", "backgrounds",
                    os.path.join("models", "checkpoints"),
                    os.path.join("models", "exports")):
            assert (dest / sub).is_dir()

    def test_preserves_video_paths(self, sample_project, tmp_path):
        project_dir, manifest = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        returned = import_rlt(rlt, str(dest))
        on_disk = json.loads((dest / "project.json").read_text(encoding="utf-8"))
        paths = sorted(v["path"] for v in on_disk["videos"])
        expected = sorted(v["path"] for v in manifest["videos"])
        assert paths == expected
        # The returned manifest matches what was written
        assert returned["videos"] == on_disk["videos"]


# ── Round-trip ─────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_manifest_identity(self, sample_project, tmp_path):
        project_dir, manifest = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        imported = import_rlt(rlt, str(dest))
        # Full manifest survives the round-trip unchanged
        assert imported == manifest

    def test_labels_identity(self, sample_project, tmp_path):
        project_dir, _ = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        dest = tmp_path / "imported"
        import_rlt(rlt, str(dest))
        for vid in ("vid_a", "vid_b"):
            orig = os.path.join(project_dir, "labels", f"{vid}_bouts.csv")
            new = dest / "labels" / f"{vid}_bouts.csv"
            assert open(orig, encoding="utf-8").read() == new.read_text(encoding="utf-8")


# ── inspect_rlt ────────────────────────────────────────────────────────────

class TestInspectRlt:
    def test_reports_counts(self, sample_project, tmp_path):
        project_dir, manifest = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        info = inspect_rlt(rlt)
        assert info["version"] == RLT_VERSION
        assert info["videos"] == len(manifest["videos"])
        assert info["labels"] == 2
        assert info["manifest"]["name"] == manifest["name"]

    def test_does_not_extract(self, sample_project, tmp_path):
        """inspect_rlt reads metadata only; it writes nothing to disk."""
        project_dir, _ = sample_project
        rlt = str(tmp_path / "out.rlt")
        export_rlt(project_dir, rlt)

        before = set(os.listdir(tmp_path))
        inspect_rlt(rlt)
        after = set(os.listdir(tmp_path))
        assert before == after

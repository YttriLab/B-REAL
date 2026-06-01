"""
RLT file format service: lightweight HDF5 project files with .rlt extension.

An .rlt file is a small HDF5 container holding only the irreplaceable parts of
a project — the manifest (project info, behaviors, config), video references
(paths + filenames), and the hand-annotated label bouts. Regenerable data
(preprocessed frames, backgrounds, model checkpoints) is NOT stored; it is
reproduced from the videos + config after the project is opened.

This mirrors SLEAP's base `.slp` format, which stores labels and video
references rather than embedding raw frames.

HDF5 layout:
    /                         (attr: rlt_version)
    /                         (attr: manifest — JSON string with paths + config)
    /labels/{video_id}        (dataset: UTF-8 CSV string of bouts)
"""

import json
import os
from pathlib import Path
from typing import Optional

import h5py


RLT_VERSION = 2


# ── Export (folder → .rlt) ────────────────────────────────────────────────


def export_rlt(
    project_dir: str,
    output_path: str,
    *,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Pack a project's manifest + labels into a lightweight .rlt file.

    Video paths and all configuration are preserved in the manifest. Frames,
    backgrounds, and checkpoints are intentionally excluded (regenerable).

    Args:
        project_dir: Path to the project folder (must contain project.json).
        output_path: Where to write the .rlt file.
        progress_cb: Optional callback(stage: str, pct: float).

    Returns:
        Summary dict with counts of packed items.
    """
    project_dir = Path(project_dir)
    manifest_path = project_dir / "project.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No project.json in {project_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    summary = {"videos": len(manifest.get("videos", [])), "labels": 0}

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(output_path, "w") as hf:
        hf.attrs["rlt_version"] = RLT_VERSION
        hf.attrs["manifest"] = json.dumps(manifest, ensure_ascii=False)

        _report("labels", 0)
        labels_grp = hf.create_group("labels")
        labels_dir = project_dir / "labels"
        if labels_dir.is_dir():
            csv_files = sorted(labels_dir.glob("*.csv"))
            for i, csv_path in enumerate(csv_files):
                vid_id = csv_path.stem.replace("_bouts", "")
                content = csv_path.read_text(encoding="utf-8")
                # Store label CSVs verbatim (text — no compression).
                labels_grp.create_dataset(vid_id, data=content)
                summary["labels"] += 1
                _report("labels", (i + 1) / max(len(csv_files), 1) * 100)

        _report("done", 100)

    return summary


# ── Import (.rlt → folder) ────────────────────────────────────────────────


def import_rlt(
    rlt_path: str,
    output_dir: str,
    *,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Unpack an .rlt file into a project directory.

    Writes project.json and the label CSVs, and creates the standard (empty)
    subdirectories so the regenerable data can be rebuilt in place. Video paths
    from the manifest are preserved as-is; the app will prompt to re-link any
    video whose path does not resolve on this machine.

    Args:
        rlt_path:    Path to the .rlt file.
        output_dir:  Where to create the project directory.
        progress_cb: Optional callback(stage: str, pct: float).

    Returns:
        The manifest dict from the file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(rlt_path, "r") as hf:
        manifest = json.loads(hf.attrs["manifest"])

        # Standard project subdirs (empty — regenerable data rebuilt later)
        for sub in ("labels", "frames", "backgrounds",
                    os.path.join("models", "checkpoints"),
                    os.path.join("models", "exports")):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        _report("labels", 0)
        labels_dir = output_dir / "labels"
        if "labels" in hf:
            label_keys = list(hf["labels"].keys())
            for i, vid_id in enumerate(label_keys):
                content = hf["labels"][vid_id][()]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                (labels_dir / f"{vid_id}_bouts.csv").write_text(
                    content, encoding="utf-8"
                )
                _report("labels", (i + 1) / max(len(label_keys), 1) * 100)

        manifest_path = output_dir / "project.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _report("done", 100)

    return manifest


# ── Inspect (read metadata only) ─────────────────────────────────────────


def inspect_rlt(rlt_path: str) -> dict:
    """
    Read metadata from an .rlt file without extracting anything.

    Returns dict with manifest, version, and content counts.
    """
    with h5py.File(rlt_path, "r") as hf:
        version = int(hf.attrs.get("rlt_version", 1))
        manifest = json.loads(hf.attrs["manifest"])

        return {
            "version": version,
            "manifest": manifest,
            "videos": len(manifest.get("videos", [])),
            "labels": len(hf["labels"]) if "labels" in hf else 0,
        }

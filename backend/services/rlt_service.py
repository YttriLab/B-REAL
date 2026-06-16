"""
RLT file format service: HDF5 project files with .rlt extension.

Two flavors, both HDF5:
  - **Lightweight** (default): manifest (project info, behaviors, config), video
    references, and the hand-annotated label bouts only. Small and shareable —
    modeled on SLEAP's base `.slp`. Frames/backgrounds/models are regenerated.
  - **Package** (`package=True`): everything above PLUS the regenerable data —
    preprocessed frames (gzip), background images, and model checkpoints/exports.
    A self-contained, train-anywhere bundle — modeled on SLEAP's `.pkg.slp`.

Import auto-detects which groups are present and restores accordingly.

HDF5 layout:
    /                               (attr: rlt_version, packaged)
    /                               (attr: manifest — JSON string)
    /labels/{video_id}              (dataset: UTF-8 CSV string of bouts)
    /frames/{video_id}/{idx}        (dataset: uint8 PNG bytes, gzip)   [package]
    /backgrounds/{video_id}         (dataset: uint8 PNG bytes, gzip)   [package]
    /models/checkpoints/{name}      (dataset: uint8 binary blob)       [package]
    /models/exports/{name}          (dataset: uint8 binary blob)       [package]
"""

import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


RLT_VERSION = 3


# ── Export (folder → .rlt) ────────────────────────────────────────────────


def export_rlt(
    project_dir: str,
    output_path: str,
    *,
    package: bool = False,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Pack a project into an .rlt file.

    Args:
        project_dir: Path to the project folder (must contain project.json).
        output_path: Where to write the .rlt file.
        package:     If True, also embed frames, backgrounds, and models
                     (a self-contained bundle). If False (default), labels +
                     manifest only.
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

    summary = {
        "videos": len(manifest.get("videos", [])),
        "labels": 0,
        "frames": 0,
        "backgrounds": 0,
        "checkpoints": 0,
        "exports": 0,
        "packaged": package,
    }

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(output_path, "w") as hf:
        hf.attrs["rlt_version"] = RLT_VERSION
        hf.attrs["packaged"] = bool(package)
        hf.attrs["manifest"] = json.dumps(manifest, ensure_ascii=False)

        # ── Labels (always) ───────────────────────────────────────────
        _report("labels", 0)
        labels_grp = hf.create_group("labels")
        labels_dir = project_dir / "labels"
        if labels_dir.is_dir():
            csv_files = sorted(labels_dir.glob("*.csv"))
            for i, csv_path in enumerate(csv_files):
                vid_id = csv_path.stem.replace("_bouts", "")
                labels_grp.create_dataset(vid_id, data=csv_path.read_text(encoding="utf-8"))
                summary["labels"] += 1
                _report("labels", (i + 1) / max(len(csv_files), 1) * 100)

        if not package:
            _report("done", 100)
            return summary

        # ── Frames (package only) ─────────────────────────────────────
        frames_grp = hf.create_group("frames")
        frames_dir = project_dir / "frames"
        if frames_dir.is_dir():
            vdirs = sorted(d for d in frames_dir.iterdir() if d.is_dir())
            for vi, vdir in enumerate(vdirs):
                vid_grp = frames_grp.create_group(vdir.name)
                for png in sorted(vdir.glob("*.png"), key=lambda p: p.stem):
                    raw = np.frombuffer(png.read_bytes(), dtype=np.uint8)
                    vid_grp.create_dataset(png.stem, data=raw, compression="gzip", compression_opts=4)
                    summary["frames"] += 1
                _report("frames", (vi + 1) / max(len(vdirs), 1) * 100)

        # ── Backgrounds (package only) ────────────────────────────────
        bg_grp = hf.create_group("backgrounds")
        bg_dir = project_dir / "backgrounds"
        if bg_dir.is_dir():
            bgs = sorted(bg_dir.glob("*.png"))
            for i, bg in enumerate(bgs):
                raw = np.frombuffer(bg.read_bytes(), dtype=np.uint8)
                bg_grp.create_dataset(bg.stem.replace("bg_", ""), data=raw, compression="gzip", compression_opts=4)
                summary["backgrounds"] += 1
                _report("backgrounds", (i + 1) / max(len(bgs), 1) * 100)

        # ── Models (package only) ─────────────────────────────────────
        models_grp = hf.create_group("models")
        for sub, key in (("checkpoints", "checkpoints"), ("exports", "exports")):
            grp = models_grp.create_group(sub)
            mdir = project_dir / "models" / sub
            if mdir.is_dir():
                files = sorted(p for p in mdir.glob("*") if p.is_file())
                for i, f in enumerate(files):
                    raw = np.frombuffer(f.read_bytes(), dtype=np.uint8)
                    grp.create_dataset(f.name, data=raw)
                    summary[key] += 1
                    _report("models", (i + 1) / max(len(files), 1) * 100)

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
    Unpack an .rlt into a project directory. Restores labels always, and
    frames/backgrounds/models if the file is a package. Video paths from the
    manifest are preserved (re-link on the new machine if they don't resolve).

    Returns the manifest dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    def _write_blobs(group, dest_dir, namer):
        dest_dir.mkdir(parents=True, exist_ok=True)
        keys = list(group.keys())
        for i, key in enumerate(keys):
            data = group[key][()]
            (dest_dir / namer(key)).write_bytes(data.tobytes())
            _report(dest_dir.name, (i + 1) / max(len(keys), 1) * 100)

    with h5py.File(rlt_path, "r") as hf:
        manifest = json.loads(hf.attrs["manifest"])

        for sub in ("labels", "frames", "backgrounds",
                    os.path.join("models", "checkpoints"),
                    os.path.join("models", "exports")):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        # Labels
        if "labels" in hf:
            labels_dir = output_dir / "labels"
            for i, vid_id in enumerate(hf["labels"].keys()):
                content = hf["labels"][vid_id][()]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                (labels_dir / f"{vid_id}_bouts.csv").write_text(content, encoding="utf-8")

        # Frames (package)
        if "frames" in hf:
            frames_dir = output_dir / "frames"
            vids = list(hf["frames"].keys())
            for vi, vid_id in enumerate(vids):
                _write_blobs(hf["frames"][vid_id], frames_dir / vid_id, lambda k: f"{k}.png")
                _report("frames", (vi + 1) / max(len(vids), 1) * 100)

        # Backgrounds (package)
        if "backgrounds" in hf:
            _write_blobs(hf["backgrounds"], output_dir / "backgrounds", lambda k: f"bg_{k}.png")

        # Models (package)
        if "models" in hf:
            if "checkpoints" in hf["models"]:
                _write_blobs(hf["models"]["checkpoints"], output_dir / "models" / "checkpoints", lambda k: k)
            if "exports" in hf["models"]:
                _write_blobs(hf["models"]["exports"], output_dir / "models" / "exports", lambda k: k)

        (output_dir / "project.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _report("done", 100)

    return manifest


# ── Inspect (read metadata only) ─────────────────────────────────────────


def inspect_rlt(rlt_path: str) -> dict:
    """Read metadata from an .rlt without extracting. Returns counts + flags."""
    with h5py.File(rlt_path, "r") as hf:
        manifest = json.loads(hf.attrs["manifest"])
        frame_count = 0
        if "frames" in hf:
            for vid_id in hf["frames"]:
                frame_count += len(hf["frames"][vid_id])
        return {
            "version": int(hf.attrs.get("rlt_version", 1)),
            "packaged": bool(hf.attrs.get("packaged", "frames" in hf)),
            "manifest": manifest,
            "videos": len(manifest.get("videos", [])),
            "labels": len(hf["labels"]) if "labels" in hf else 0,
            "frames": frame_count,
            "backgrounds": len(hf["backgrounds"]) if "backgrounds" in hf else 0,
            "checkpoints": len(hf["models"]["checkpoints"]) if "models" in hf and "checkpoints" in hf["models"] else 0,
            "exports": len(hf["models"]["exports"]) if "models" in hf and "exports" in hf["models"] else 0,
        }

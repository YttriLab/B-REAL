"""
RLT file format service: pack/unpack projects as HDF5 files with .rlt extension.

An .rlt file is an HDF5 container that holds everything needed to reproduce
or transfer a project — labels, preprocessed frames, backgrounds, model
checkpoints, and configuration — in a single portable file.

Video files are NOT stored inside (they're too large); only their filenames
are recorded so the user can re-link them after transfer.

HDF5 layout:
    /manifest                 (JSON string as attribute)
    /labels/{video_id}        (dataset: UTF-8 CSV string)
    /frames/{video_id}/{idx}  (dataset: uint8 PNG bytes, chunked + gzip)
    /backgrounds/{video_id}   (dataset: uint8 PNG bytes)
    /models/checkpoints/{name}(dataset: uint8 binary blob)
    /models/exports/{name}    (dataset: uint8 binary blob)
"""

import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


RLT_VERSION = 1


# ── Export (folder → .rlt) ────────────────────────────────────────────────


def export_rlt(
    project_dir: str,
    output_path: str,
    *,
    include_frames: bool = True,
    include_models: bool = True,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Pack a project directory into an .rlt file.

    Args:
        project_dir:    Path to the project folder (must contain project.json).
        output_path:    Where to write the .rlt file.
        include_frames: Whether to pack preprocessed frames (can be large).
        include_models: Whether to pack checkpoints and exports.
        progress_cb:    Optional callback(stage: str, pct: float).

    Returns:
        Summary dict with counts of packed items.
    """
    project_dir = Path(project_dir)
    manifest_path = project_dir / "project.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No project.json in {project_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    summary = {
        "labels": 0,
        "frames": 0,
        "backgrounds": 0,
        "checkpoints": 0,
        "exports": 0,
    }

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(output_path, "w") as hf:
        # ── Metadata ──────────────────────────────────────────────────
        hf.attrs["rlt_version"] = RLT_VERSION
        hf.attrs["manifest"] = json.dumps(manifest, ensure_ascii=False)

        _report("labels", 0)

        # ── Labels ────────────────────────────────────────────────────
        labels_grp = hf.create_group("labels")
        labels_dir = project_dir / "labels"
        if labels_dir.is_dir():
            csv_files = sorted(labels_dir.glob("*.csv"))
            for i, csv_path in enumerate(csv_files):
                vid_id = csv_path.stem.replace("_bouts", "")
                content = csv_path.read_text(encoding="utf-8")
                labels_grp.create_dataset(vid_id, data=content)
                summary["labels"] += 1
                _report("labels", (i + 1) / max(len(csv_files), 1) * 100)

        # ── Frames ────────────────────────────────────────────────────
        if include_frames:
            frames_grp = hf.create_group("frames")
            frames_dir = project_dir / "frames"
            if frames_dir.is_dir():
                video_dirs = sorted(
                    d for d in frames_dir.iterdir() if d.is_dir()
                )
                total_videos = len(video_dirs)
                for vi, vdir in enumerate(video_dirs):
                    vid_id = vdir.name
                    vid_grp = frames_grp.create_group(vid_id)
                    pngs = sorted(vdir.glob("*.png"), key=lambda p: p.stem)
                    for fi, png_path in enumerate(pngs):
                        raw = png_path.read_bytes()
                        vid_grp.create_dataset(
                            png_path.stem,
                            data=np.frombuffer(raw, dtype=np.uint8),
                            compression="gzip",
                            compression_opts=4,
                        )
                        summary["frames"] += 1
                    pct = (vi + 1) / max(total_videos, 1) * 100
                    _report("frames", pct)
        else:
            hf.create_group("frames")

        # ── Backgrounds ───────────────────────────────────────────────
        _report("backgrounds", 0)
        bg_grp = hf.create_group("backgrounds")
        bg_dir = project_dir / "backgrounds"
        if bg_dir.is_dir():
            bg_files = sorted(bg_dir.glob("*.png"))
            for i, bg_path in enumerate(bg_files):
                vid_id = bg_path.stem.replace("bg_", "")
                raw = bg_path.read_bytes()
                bg_grp.create_dataset(
                    vid_id,
                    data=np.frombuffer(raw, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )
                summary["backgrounds"] += 1
                _report("backgrounds", (i + 1) / max(len(bg_files), 1) * 100)

        # ── Models ────────────────────────────────────────────────────
        models_grp = hf.create_group("models")
        ckpt_grp = models_grp.create_group("checkpoints")
        export_grp = models_grp.create_group("exports")

        if include_models:
            _report("models", 0)
            ckpt_dir = project_dir / "models" / "checkpoints"
            if ckpt_dir.is_dir():
                ckpt_files = sorted(ckpt_dir.glob("*"))
                for i, ckpt_path in enumerate(ckpt_files):
                    if ckpt_path.is_file():
                        raw = ckpt_path.read_bytes()
                        ckpt_grp.create_dataset(
                            ckpt_path.name,
                            data=np.frombuffer(raw, dtype=np.uint8),
                        )
                        summary["checkpoints"] += 1
                        _report("models", (i + 1) / max(len(ckpt_files), 1) * 50)

            export_dir = project_dir / "models" / "exports"
            if export_dir.is_dir():
                export_files = sorted(export_dir.glob("*"))
                for i, exp_path in enumerate(export_files):
                    if exp_path.is_file():
                        raw = exp_path.read_bytes()
                        export_grp.create_dataset(
                            exp_path.name,
                            data=np.frombuffer(raw, dtype=np.uint8),
                        )
                        summary["exports"] += 1
                        _report("models", 50 + (i + 1) / max(len(export_files), 1) * 50)

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
        version = int(hf.attrs.get("rlt_version", 1))
        manifest = json.loads(hf.attrs["manifest"])

        # Clear absolute video paths — user must re-link
        for v in manifest.get("videos", []):
            v["path"] = ""

        # ── Labels ────────────────────────────────────────────────────
        _report("labels", 0)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        if "labels" in hf:
            label_keys = list(hf["labels"].keys())
            for i, vid_id in enumerate(label_keys):
                content = hf["labels"][vid_id][()]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                out_path = labels_dir / f"{vid_id}_bouts.csv"
                out_path.write_text(content, encoding="utf-8")
                _report("labels", (i + 1) / max(len(label_keys), 1) * 100)

        # ── Frames ────────────────────────────────────────────────────
        _report("frames", 0)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        if "frames" in hf:
            video_ids = list(hf["frames"].keys())
            for vi, vid_id in enumerate(video_ids):
                vid_dir = frames_dir / vid_id
                vid_dir.mkdir(exist_ok=True)
                frame_keys = list(hf["frames"][vid_id].keys())
                for fi, frame_key in enumerate(frame_keys):
                    data = hf["frames"][vid_id][frame_key][()]
                    out_path = vid_dir / f"{frame_key}.png"
                    out_path.write_bytes(data.tobytes())
                pct = (vi + 1) / max(len(video_ids), 1) * 100
                _report("frames", pct)

        # ── Backgrounds ───────────────────────────────────────────────
        _report("backgrounds", 0)
        bg_dir = output_dir / "backgrounds"
        bg_dir.mkdir(exist_ok=True)
        if "backgrounds" in hf:
            bg_keys = list(hf["backgrounds"].keys())
            for i, vid_id in enumerate(bg_keys):
                data = hf["backgrounds"][vid_id][()]
                out_path = bg_dir / f"bg_{vid_id}.png"
                out_path.write_bytes(data.tobytes())
                _report("backgrounds", (i + 1) / max(len(bg_keys), 1) * 100)

        # ── Models ────────────────────────────────────────────────────
        _report("models", 0)
        ckpt_dir = output_dir / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_dir = output_dir / "models" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        if "models" in hf:
            if "checkpoints" in hf["models"]:
                ckpt_keys = list(hf["models"]["checkpoints"].keys())
                for i, name in enumerate(ckpt_keys):
                    data = hf["models"]["checkpoints"][name][()]
                    out_path = ckpt_dir / name
                    out_path.write_bytes(data.tobytes())
                    _report("models", (i + 1) / max(len(ckpt_keys), 1) * 50)

            if "exports" in hf["models"]:
                exp_keys = list(hf["models"]["exports"].keys())
                for i, name in enumerate(exp_keys):
                    data = hf["models"]["exports"][name][()]
                    out_path = export_dir / name
                    out_path.write_bytes(data.tobytes())
                    _report("models", 50 + (i + 1) / max(len(exp_keys), 1) * 50)

        # ── Write manifest ────────────────────────────────────────────
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

        frame_count = 0
        if "frames" in hf:
            for vid_id in hf["frames"]:
                frame_count += len(hf["frames"][vid_id])

        return {
            "version": version,
            "manifest": manifest,
            "labels": len(hf["labels"]) if "labels" in hf else 0,
            "frames": frame_count,
            "backgrounds": len(hf["backgrounds"]) if "backgrounds" in hf else 0,
            "checkpoints": len(hf["models"]["checkpoints"]) if "models" in hf and "checkpoints" in hf["models"] else 0,
            "exports": len(hf["models"]["exports"]) if "models" in hf and "exports" in hf["models"] else 0,
        }

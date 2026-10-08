"""On-disk project persistence.

Python successor to project-manager.js + saveProgress. A project is a directory
with a `project.json` manifest (global behaviors + registered videos) and a
`labels/` folder holding, per video: `<id>.json` (bouts) and `<id>_labels.csv`
(the training one-hot matrix load_data reads). Behaviors are global; bouts are
per video — the same split the Electron app used (localStorage "behaviors" +
per-video "progress_<file>").

`.rlt` packaging is delegated to the backend (rlt_service) via BackendClient;
this module handles the plain on-disk layout.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .bouts import Behavior, Bout, LabelModel
from .csv_io import generate_training_label_csv

PROJECT_VERSION = 1


def safe_id(name: str) -> str:
    """Filesystem-safe id derived from a video filename (the per-video key)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


class ProjectStore:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "project.json"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def create(self, name: str) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"version": PROJECT_VERSION, "name": name, "behaviors": [], "videos": []}
        self._write_manifest(manifest)
        return manifest

    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def save_behaviors(self, behaviors) -> None:
        manifest = self.load_manifest()
        manifest["behaviors"] = [vars(b) for b in behaviors]
        self._write_manifest(manifest)

    def register_video(self, video_id, name, path, frame_count=None) -> None:
        manifest = self.load_manifest()
        videos = manifest.setdefault("videos", [])
        record = {"id": video_id, "name": name, "path": str(path), "frame_count": frame_count}
        existing = next((v for v in videos if v.get("id") == video_id), None)
        if existing:
            existing.update(record)
        else:
            videos.append(record)
        self._write_manifest(manifest)

    def list_videos(self) -> list:
        return self.load_manifest().get("videos", []) or []

    def set_last_video(self, video_id) -> None:
        """Remember the open video so reopening the project returns to it."""
        manifest = self.load_manifest()
        if manifest.get("last_video") != video_id:
            manifest["last_video"] = video_id
            self._write_manifest(manifest)

    def last_video(self):
        return self.load_manifest().get("last_video")

    def remove_video(self, video_id, delete_data: bool = False) -> None:
        """Drop a video from the project. The video file itself is never touched.

        By default its labels, frames and background stay on disk, so adding the
        same video back restores its labels. `delete_data` removes those too.
        """
        if not str(video_id or "").strip():
            raise ValueError("empty video id")  # frames_dir("") is all of frames/
        manifest = self.load_manifest()
        manifest["videos"] = [v for v in manifest.get("videos", []) if v.get("id") != video_id]
        if manifest.get("last_video") == video_id:
            manifest.pop("last_video")
        self._write_manifest(manifest)
        if delete_data:
            for p in (self.labels_dir / f"{video_id}.json", self.label_csv_path(video_id),
                      self.background_path(video_id)):
                p.unlink(missing_ok=True)
            shutil.rmtree(self.frames_dir(video_id), ignore_errors=True)

    def label_csv_path(self, video_id) -> Path:
        return self.labels_dir / f"{video_id}_labels.csv"

    def frames_dir(self, video_id) -> Path:
        """Where preprocessing extracts this video's frames (the trainer's frame_dir)."""
        return self.root / "frames" / str(video_id)

    def backgrounds_dir(self) -> Path:
        return self.root / "backgrounds"

    def background_path(self, video_id) -> Path:
        """Where this video's background lives.

        The backend names it ``bg_<stem>.png`` (compute_background), so match
        that rather than inventing a second convention — otherwise a computed
        background is never found again and never travels with the .rlt.
        """
        stem = os.path.splitext(str(video_id))[0]
        current = self.backgrounds_dir() / f"bg_{stem}.png"
        if current.exists():
            return current
        # Projects written before the naming was reconciled.
        legacy = self.backgrounds_dir() / f"{video_id}.png"
        return legacy if legacy.exists() else current

    def models_dir(self) -> Path:
        return self.root / "models"

    def checkpoint_dir(self) -> Path:
        # models/checkpoints, not checkpoints/: rlt_service packages
        # models/checkpoints and models/exports, so anything written elsewhere
        # never travels with the .rlt.
        return self.models_dir() / "checkpoints"

    def exports_dir(self) -> Path:
        return self.models_dir() / "exports"

    def add_model(self, entry: dict) -> None:
        """Record a trained checkpoint in the manifest, deduped by path.

        Keeps enough of the training config to interpret the model later — in
        particular `binarize`/`label_union`, without which Active Learning can't
        map a behavior name into a binarized model's collapsed label space.
        """
        manifest = self.load_manifest()
        models = manifest.setdefault("models", [])
        path = entry.get("path")
        existing = next((i for i, m in enumerate(models)
                         if path and m.get("path") == path), None)
        if existing is not None:
            models[existing] = entry
        else:
            models.append(entry)
        self._write_manifest(manifest)

    def list_models(self) -> list:
        return self.load_manifest().get("models", []) or []

    # Free-form per-project settings for whichever pipeline is attached (the
    # shell never reads them): a JSON object under manifest["settings"].
    def get_setting(self, key: str, default=None):
        return (self.load_manifest().get("settings") or {}).get(key, default)

    def set_setting(self, key: str, value) -> None:
        manifest = self.load_manifest()
        settings = manifest.setdefault("settings", {})
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value
        self._write_manifest(manifest)

    def save_video_labels(self, video_id, model: LabelModel) -> None:
        """Persist this video's bouts (+ the training CSV) and the global behaviors."""
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        data = {"bouts": [vars(b) for b in model.bouts]}
        (self.labels_dir / f"{video_id}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
        self.label_csv_path(video_id).write_text(
            generate_training_label_csv(model.bouts), encoding="utf-8")
        self.save_behaviors(model.behaviors)

    def load_into_model(self, video_id) -> LabelModel:
        """A model with the project's global behaviors + this video's bouts."""
        model = LabelModel()
        if self.exists():
            for b in self.load_manifest().get("behaviors", []):
                model.behaviors.append(Behavior(**b))
        p = self.labels_dir / f"{video_id}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            model.bouts = [Bout(**b) for b in data.get("bouts", [])]
        return model

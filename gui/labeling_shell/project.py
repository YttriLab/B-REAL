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
import re
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

    def label_csv_path(self, video_id) -> Path:
        return self.labels_dir / f"{video_id}_labels.csv"

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

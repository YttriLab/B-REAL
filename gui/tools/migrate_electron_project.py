"""Convert a project folder written by the old Electron app to the current layout.

    pixi run -e gui python gui/tools/migrate_electron_project.py <project_dir> \
        [--video-dir DIR] [--dry-run] [--fps N]

The Electron app's project.json differs from what `labeling_shell.project`
reads in three ways, and this rewrites each one in place:

  * behaviors were ``{name, key}`` with no ``id`` -> each gets a fresh id.
  * a video's id was its filename stem; the app now keys videos by
    ``safe_id(<filename with extension>)``. Entries are re-keyed, and
    ``frames/<old id>/`` + the background are renamed to match.
  * labels lived in the CSV named by the entry's ``labels`` key; the app reads
    ``labels/<id>.json`` (bouts) + ``labels/<id>_labels.csv``. Those are written
    from the old CSV, which is left in place.

Moving a project to another machine also leaves its absolute paths behind:
``--video-dir`` repoints each video to the same filename in that folder, and a
``models`` entry is repointed to the file of the same name under the project's
own ``models/`` when one is there.

The original manifest is kept as ``project.electron-backup.json``. Safe to
re-run: an already converted project is reported and left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling_shell.bouts import Behavior, LabelModel, normalize_behavior_name  # noqa: E402
from labeling_shell.csv_io import generate_training_label_csv, parse_csv_and_import  # noqa: E402
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402

BACKUP_NAME = "project.electron-backup.json"


def _basename(path: str) -> str:
    # Manifests written on Windows hold backslash paths; Path() on POSIX won't split them.
    return re.split(r"[\\/]", str(path))[-1]


def _rel(root: Path, rel: str) -> Path:
    return root / Path(*re.split(r"[\\/]", rel))


def _video_fps(path: str):
    try:
        import av
        with av.open(path) as c:
            rate = c.streams.video[0].average_rate
            return float(rate) if rate else None
    except Exception:
        return None


def needs_migration(manifest: dict) -> bool:
    if any("id" not in b for b in manifest.get("behaviors", [])):
        return True
    return any(v.get("id") != safe_id(_basename(v.get("path", "")))
               for v in manifest.get("videos", []) if v.get("path"))


def migrate(root, dry_run: bool = False, fps=None, video_dir=None, log=print) -> bool:
    """Convert the project at `root`. Returns True if anything was (or would be) changed."""
    store = ProjectStore(root)
    root = store.root
    manifest = store.load_manifest()
    if not needs_migration(manifest):
        log(f"{root}: already in the current format, nothing to do.")
        return False

    behaviors = [Behavior(id=b.get("id") or uuid.uuid4().hex,
                          name=normalize_behavior_name(b.get("name")),
                          key=str(b.get("key") or ""))
                 for b in manifest.get("behaviors", [])]

    new_videos, label_writes, renames = [], [], []
    seen_ids, used_csvs = set(), set()
    for entry in manifest.get("videos", []):
        old_id = entry.get("id")
        name = _basename(entry.get("path", "")) or str(old_id)
        new_id = safe_id(name)
        if new_id in seen_ids:
            log(f"  ! {old_id}: another entry already maps to {new_id}; skipped")
            continue
        seen_ids.add(new_id)

        path = entry.get("path")
        if video_dir:
            moved = Path(video_dir) / name
            if moved.exists():
                path = str(moved.resolve())
            else:
                log(f"  ! {new_id}: not found in {video_dir}; path left as {path}")
        record = {k: v for k, v in entry.items() if k not in ("labels", "frames")}
        record.update(id=new_id, name=name, path=path)
        record.setdefault("frame_count", None)

        # Labels -> labels/<new id>.json + <new id>_labels.csv
        csv_rel = entry.get("labels") or f"labels/{old_id}_bouts.csv"
        csv_path = _rel(root, csv_rel)
        if (store.labels_dir / f"{new_id}.json").exists():
            log(f"  - {new_id}: labels/{new_id}.json already exists; labels kept as is")
        elif csv_path.exists():
            used_csvs.add(csv_path.resolve())
            model = LabelModel()
            model.behaviors = behaviors
            vfps = fps or _video_fps(path or "") or 30.0
            n = parse_csv_and_import(csv_path.read_text(encoding="utf-8"), model, vfps)
            behaviors = model.behaviors  # the import may add behaviors the manifest lacked
            label_writes.append((new_id, model.bouts))
            log(f"  + {new_id}: {n} bouts from {csv_rel}")
        else:
            log(f"  - {new_id}: no label CSV found ({csv_rel})")

        # frames/<old id>/ -> frames/<new id>/
        old_frames = _rel(root, entry["frames"]) if entry.get("frames") else root / "frames" / str(old_id)
        new_frames = store.frames_dir(new_id)
        if old_frames.is_dir() and old_frames.resolve() != new_frames.resolve():
            if new_frames.exists():
                log(f"  ! {new_id}: both {old_frames.name}/ and {new_frames.name}/ exist under frames/; left alone")
            else:
                renames.append((old_frames, new_frames))

        # Background: the app looks for bg_<stem of new id>.png
        if entry.get("background"):
            old_bg = _rel(root, entry["background"])
            new_bg = store.background_path(new_id)
            if old_bg.exists() and not new_bg.exists():
                renames.append((old_bg, new_bg))
            record["background"] = new_bg.relative_to(root).as_posix()

        new_videos.append(record)

    models = []
    for m in manifest.get("models", []) or []:
        m = dict(m)
        hits = [p for p in (store.models_dir().rglob(_basename(m.get("path", "")))
                            if m.get("path") else []) if p.is_file()]
        if len(hits) == 1 and hits[0].resolve() != Path(m["path"]).resolve():
            m["path"] = str(hits[0].resolve())
        elif not Path(m.get("path", "")).exists():
            log(f"  ! model {m.get('name')}: {m.get('path')} not found in the project; left as is")
        models.append(m)

    for csv in sorted(store.labels_dir.glob("*.csv")):
        if csv.resolve() not in used_csvs and not csv.name.endswith("_labels.csv"):
            log(f"  ? {csv.name} is not referenced by any video entry; left in place")
    for src, dst in renames:
        log(f"  > rename {src.relative_to(root)} -> {dst.relative_to(root)}")

    if dry_run:
        log("Dry run: nothing written.")
        return True

    backup = root / BACKUP_NAME
    if not backup.exists():
        shutil.copy2(store.manifest_path, backup)
    store.labels_dir.mkdir(parents=True, exist_ok=True)
    for new_id, bouts in label_writes:
        (store.labels_dir / f"{new_id}.json").write_text(
            json.dumps({"bouts": [vars(b) for b in bouts]}, indent=2), encoding="utf-8")
        store.label_csv_path(new_id).write_text(generate_training_label_csv(bouts), encoding="utf-8")
    for src, dst in renames:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    # Manifest last, so an interrupted run is still detected as unconverted and can be re-run.
    manifest["behaviors"] = [vars(b) for b in behaviors]
    manifest["videos"] = new_videos
    if "models" in manifest:
        manifest["models"] = models
    store._write_manifest(manifest)
    log(f"Converted {root} (original manifest saved as {BACKUP_NAME}).")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("project_dir")
    p.add_argument("--video-dir", default=None,
                   help="folder the videos now live in (matched by filename)")
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument("--fps", type=float, default=None,
                   help="fps for timestamp (start,end,behavior) label CSVs, if the video "
                        "can't be opened to read it; frame-indexed CSVs don't need it")
    a = p.parse_args(argv)
    if not ProjectStore(a.project_dir).exists():
        print(f"No project.json in {a.project_dir}", file=sys.stderr)
        return 1
    migrate(a.project_dir, dry_run=a.dry_run, fps=a.fps, video_dir=a.video_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Converting an Electron-era project folder to the current layout."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.project import ProjectStore  # noqa: E402
from tools.migrate_electron_project import migrate  # noqa: E402

# Shape of a manifest the Electron app wrote: behaviors without ids, a video
# keyed by its filename stem, labels in the CSV its entry names.
ELECTRON_MANIFEST = {
    "name": "rig", "version": 1,
    "behaviors": [{"name": "rear", "key": "1"}, {"name": "other", "key": "2"}],
    "videos": [{
        "id": "clip",
        "path": "C:\\data\\clip.mp4",
        "labels": "labels/clip_bouts.csv",
        "frames": "frames/clip/",
        "background": "backgrounds/bg_clip.png",
    }],
    "models": [{"name": "a.ckpt", "path": "D:\\rig\\a.ckpt"}],
}
LABEL_CSV = "time,other,rear\n10,1,0\n11,1,0\n12,0,1\n13,0,1\n20,1,0\n"


def _electron_project(d: str) -> Path:
    root = Path(d)
    (root / "labels").mkdir()
    (root / "frames" / "clip").mkdir(parents=True)
    (root / "frames" / "clip" / "10.png").write_bytes(b"x")
    (root / "backgrounds").mkdir()
    (root / "backgrounds" / "bg_clip.png").write_bytes(b"x")
    (root / "labels" / "clip_bouts.csv").write_text(LABEL_CSV)
    (root / "project.json").write_text(json.dumps(ELECTRON_MANIFEST))
    return root


def test_converts_and_opens():
    with tempfile.TemporaryDirectory() as d:
        root = _electron_project(d)
        assert migrate(root, log=lambda *_: None)
        store = ProjectStore(root)
        man = store.load_manifest()

        assert [b["name"] for b in man["behaviors"]] == ["rear", "other"]
        assert all(b["id"] for b in man["behaviors"])
        assert [v["id"] for v in man["videos"]] == ["clip.mp4"]
        assert man["models"] == ELECTRON_MANIFEST["models"]
        assert (root / "project.electron-backup.json").exists()

        # Loads through the same path the app uses, with the original labels.
        model = store.load_into_model("clip.mp4")
        assert sorted((b.start_frame, b.end_frame, b.behavior_name) for b in model.bouts) == \
            [(10, 11, "other"), (12, 13, "rear"), (20, 20, "other")]
        assert store.label_csv_path("clip.mp4").exists()
        assert (store.frames_dir("clip.mp4") / "10.png").exists()
        assert store.background_path("clip.mp4").exists()


def test_relinks_moved_videos_and_models():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "proj").mkdir()
        root = _electron_project(os.path.join(d, "proj"))
        videos = Path(d) / "videos"
        videos.mkdir()
        (videos / "clip.mp4").write_bytes(b"x")
        (root / "models" / "checkpoints").mkdir(parents=True)
        (root / "models" / "checkpoints" / "a.ckpt").write_bytes(b"x")

        migrate(root, video_dir=videos, log=lambda *_: None)
        man = ProjectStore(root).load_manifest()
        assert Path(man["videos"][0]["path"]) == (videos / "clip.mp4").resolve()
        assert Path(man["models"][0]["path"]) == (root / "models" / "checkpoints" / "a.ckpt").resolve()


def test_rerun_is_noop():
    with tempfile.TemporaryDirectory() as d:
        root = _electron_project(d)
        migrate(root, log=lambda *_: None)
        before = (root / "project.json").read_text()
        assert not migrate(root, log=lambda *_: None)
        assert (root / "project.json").read_text() == before


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        root = _electron_project(d)
        assert migrate(root, dry_run=True, log=lambda *_: None)
        assert json.loads((root / "project.json").read_text()) == ELECTRON_MANIFEST
        assert not (root / "project.electron-backup.json").exists()
        assert (root / "frames" / "clip").is_dir()


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

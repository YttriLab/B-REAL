"""Phase 2: project persistence (pure logic)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.bouts import Bout, LabelModel  # noqa: E402
from labeling_shell.project import ProjectStore, safe_id  # noqa: E402


def test_safe_id():
    assert safe_id("my video (1).mp4") == "my_video__1_.mp4"


def test_create_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        assert store.exists()
        assert store.load_manifest()["name"] == "demo"

        m = LabelModel()
        a = m.add_behavior("groom")
        b = m.add_behavior("rear")
        m.insert_bout_sorted(Bout(a.id, a.name, 5, 8))
        m.insert_bout_sorted(Bout(b.id, b.name, 12, 15))

        vid = safe_id("clip.mp4")
        store.register_video(vid, "clip.mp4", "/x/clip.mp4", frame_count=100)
        store.save_video_labels(vid, m)

        # global behaviors + this video's bouts round-trip
        loaded = store.load_into_model(vid)
        assert [x.name for x in loaded.behaviors] == ["groom", "rear"]
        assert sorted((x.start_frame, x.end_frame, x.behavior_name) for x in loaded.bouts) == \
            [(5, 8, "groom"), (12, 15, "rear")]

        # video registered in manifest, training CSV written
        man = store.load_manifest()
        assert man["videos"][0]["frame_count"] == 100
        assert store.label_csv_path(vid).exists()
        assert store.label_csv_path(vid).read_text().splitlines()[0] == "time,groom,rear"


def test_per_video_bouts_global_behaviors():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        m = LabelModel()
        a = m.add_behavior("groom")
        m.insert_bout_sorted(Bout(a.id, a.name, 1, 2))
        store.save_video_labels("v1.mp4", m)
        m.bouts = [Bout(a.id, a.name, 50, 60)]
        store.save_video_labels("v2.mp4", m)

        v1 = store.load_into_model("v1.mp4")
        v2 = store.load_into_model("v2.mp4")
        assert [b.start_frame for b in v1.bouts] == [1]
        assert [b.start_frame for b in v2.bouts] == [50]
        # behaviors are shared/global
        assert [b.name for b in v1.behaviors] == [b.name for b in v2.behaviors] == ["groom"]


def test_background_path_matches_what_the_backend_writes():
    """compute_background saves bg_<stem>.png. If the project looks somewhere
    else the background is never found again — it won't auto-load on a video
    switch and won't travel with the .rlt."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        vid = safe_id("clip.mp4")

        expected = store.backgrounds_dir() / "bg_clip.png"
        assert store.background_path(vid) == expected

        # and it resolves once the file is actually there
        store.backgrounds_dir().mkdir(parents=True, exist_ok=True)
        expected.write_bytes(b"\x89PNG")
        assert store.background_path(vid) == expected
        assert store.background_path(vid).exists()


def test_background_path_falls_back_to_the_legacy_name():
    """Projects written before the naming was reconciled keep working."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        vid = safe_id("clip.mp4")
        store.backgrounds_dir().mkdir(parents=True, exist_ok=True)
        legacy = store.backgrounds_dir() / f"{vid}.png"
        legacy.write_bytes(b"\x89PNG")
        assert store.background_path(vid) == legacy


def test_model_dirs_match_what_rlt_packages():
    """rlt_service packages models/checkpoints and models/exports — anything
    written elsewhere never travels with the .rlt."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        assert store.checkpoint_dir() == store.root / "models" / "checkpoints"
        assert store.exports_dir() == store.root / "models" / "exports"


def test_add_model_records_and_dedupes():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        assert store.list_models() == []

        store.add_model({"name": "a.ckpt", "path": "/p/a.ckpt", "val_mcc": 0.5})
        store.add_model({"name": "b.ckpt", "path": "/p/b.ckpt", "val_mcc": 0.7})
        assert [m["name"] for m in store.list_models()] == ["a.ckpt", "b.ckpt"]

        # retraining to the same path replaces the entry rather than appending
        store.add_model({"name": "a.ckpt", "path": "/p/a.ckpt", "val_mcc": 0.9})
        models = store.list_models()
        assert len(models) == 2
        assert next(m for m in models if m["path"] == "/p/a.ckpt")["val_mcc"] == 0.9


def test_add_model_survives_a_manifest_without_models():
    """Projects created before the registry existed have no "models" key."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        man = store.load_manifest()
        man.pop("models", None)
        store._write_manifest(man)

        store.add_model({"name": "a.ckpt", "path": "/p/a.ckpt"})
        assert len(store.list_models()) == 1


def test_remove_video_keeps_or_deletes_its_data():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore(d)
        store.create("demo")
        m = LabelModel()
        g = m.add_behavior("groom")
        m.insert_bout_sorted(Bout(g.id, g.name, 1, 3))
        for vid in ("a.mp4", "b.mp4"):
            store.register_video(vid, vid, f"/x/{vid}")
            store.save_video_labels(vid, m)
            (store.frames_dir(vid)).mkdir(parents=True)
            (store.frames_dir(vid) / "1.png").write_bytes(b"x")
        store.set_last_video("a.mp4")

        store.remove_video("a.mp4")  # default: keep its data
        assert [v["id"] for v in store.list_videos()] == ["b.mp4"]
        assert store.last_video() is None
        assert store.load_into_model("a.mp4").bouts  # re-adding brings labels back
        assert store.frames_dir("a.mp4").is_dir()

        store.remove_video("b.mp4", delete_data=True)
        assert store.list_videos() == []
        assert not store.load_into_model("b.mp4").bouts
        assert not store.label_csv_path("b.mp4").exists()
        assert not store.frames_dir("b.mp4").exists()
        assert store.frames_dir("a.mp4").is_dir()  # the other video's frames untouched

        try:
            store.remove_video("", delete_data=True)
            raise AssertionError("empty id must not reach frames/")
        except ValueError:
            pass


def test_video_paths_resolve_on_another_machine():
    """A shared project is mounted at different places on different machines;
    the path relative to the project still finds the video."""
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "share", "proj")
        vids = os.path.join(d, "share", "videos")
        os.makedirs(vids)
        for n in ("a.mp4", "b.mp4", "c.mp4"):
            open(os.path.join(vids, n), "wb").close()
        store = ProjectStore(root)
        store.create("demo")
        store.register_video("a.mp4", "a.mp4", os.path.join(vids, "a.mp4"), 10)
        rec = store.load_manifest()["videos"][0]
        assert rec["rel_path"] == "../videos/a.mp4"

        # the other machine wrote its own absolute path last
        man = store.load_manifest()
        man["videos"][0]["path"] = "/mnt/howlin/share/videos/a.mp4"
        man["videos"].append({"id": "b.mp4", "name": "b.mp4",
                              "path": "D:\\old\\videos\\b.mp4", "frame_count": 10})
        man["videos"].append({"id": "c.mp4", "name": "c.mp4",
                              "path": "/gone/c.mp4", "frame_count": 10})
        store._write_manifest(man)
        paths = {v["id"]: v["path"] for v in store.list_videos()}
        assert paths["a.mp4"] == os.path.join(vids, "a.mp4")      # via rel_path
        assert paths["b.mp4"] == "D:\\old\\videos\\b.mp4"       # no rel_path: unresolved

        # relinking one finds the others in the same folder (Windows paths too)
        store.relink_video("b.mp4", os.path.join(vids, "b.mp4"))
        assert store.relink_missing_in(vids) == ["c.mp4"]
        assert all(os.path.exists(v["path"]) for v in store.list_videos())
        assert {v["id"]: v.get("rel_path") for v in store.load_manifest()["videos"]}["c.mp4"] \
            == "../videos/c.mp4"


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

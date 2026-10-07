"""Train Localizer: SAM2 teacher tracks -> distilled localizer.

SAM2 itself is replaced by FakeTracker (a dark-pixel mask), so these run
without transformers, a GPU, or a model download; the real model was checked
against the original Ultralytics teacher on the bundled rig (sam2_track.py).
"""

import json
import os

import cv2
import numpy as np
import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient

import localizer
import localizer_train as lt
import sam2_track
from sam2_track import FLAG_LOST, FLAG_NO_TAIL, FLAG_OK, Sam2Tracker, body_center


class FakeTracker(Sam2Tracker):
    """SAM2 stand-in: the 'mask' is the frame's dark pixels."""

    def __init__(self):
        super().__init__("fake")
        self.calls = 0

    def load(self):
        pass

    def close(self):
        pass

    def _segment(self, frames, seed_xy):
        self.calls += 1
        for f in frames:
            g = f if f.ndim == 2 else f[..., 0]
            yield (g < 100).astype(np.uint8), g.shape[:2], True


def _disk_frames(n=12, h=80, w=96):
    out = []
    for i in range(n):
        f = np.full((h, w, 3), 190, np.uint8)
        cv2.circle(f, (12 + 6 * i, 30 + (i % 5) * 3), 9, (25, 25, 25), -1)
        out.append(f)
    return out


# ── SAM2 mask -> body centre ────────────────────────────────────────────────

class TestBodyCenter:
    def test_tail_does_not_pull_the_centre(self):
        m = np.zeros((200, 300), np.uint8)
        cv2.circle(m, (100, 100), 40, 1, -1)
        cv2.line(m, (140, 100), (290, 100), 1, 4)  # a long thin tail
        (cx, cy), area, flag = body_center(m)
        ys, xs = np.nonzero(m)
        assert flag == FLAG_OK and area == int(m.sum())
        assert abs(cx - 100) < 2 and abs(cy - 100) < 2
        assert xs.mean() > 110  # the naive centroid is dragged toward the tail

    def test_empty_mask_is_lost(self):
        centre, area, flag = body_center(np.zeros((10, 10), np.uint8))
        assert centre is None and area == 0 and flag == FLAG_LOST

    def test_all_thin_falls_back_to_the_whole_mask(self):
        m = np.zeros((100, 100), np.uint8)
        m[50, 10:90] = 1
        (cx, cy), _a, flag = body_center(m)
        assert flag == FLAG_NO_TAIL and abs(cx - 49.5) < 1 and cy == 50


class TestClickedPiece:
    def test_keeps_only_the_part_under_the_click(self):
        m = np.zeros((100, 100), bool)
        m[40:60, 40:60] = True        # the animal
        m[5:8, 5:8] = m[90:93, 10:13] = True  # speckles elsewhere
        piece = sam2_track.clicked_piece(m, (50, 50))
        assert piece.sum() == 400 and not piece[5:8, 5:8].any()

    def test_click_just_off_the_mask_and_empty(self):
        m = np.zeros((100, 100), bool)
        m[40:60, 40:60] = True
        assert sam2_track.clicked_piece(m, (65, 50)).sum() == 400
        assert not sam2_track.clicked_piece(m, (95, 5)).any()
        assert not sam2_track.clicked_piece(np.zeros((10, 10), bool), (5, 5)).any()


class TestTracker:
    def test_track_maps_low_res_masks_to_native_px(self):
        """Masks come back at 1/WORK_DOWNSCALE; centres must land in native px."""
        class Quarter(FakeTracker):
            def _segment(self, frames, seed_xy):
                for f in frames:
                    m = np.zeros((f.shape[0] // 4, f.shape[1] // 4), np.uint8)
                    cv2.circle(m, (30, 20), 6, 1, -1)
                    yield m, f.shape[:2], True

        (p,) = list(Quarter().track([np.zeros((160, 240), np.uint8)], (0, 0)))
        assert p.flag == FLAG_OK
        assert p.x == pytest.approx(30 * 4 + 1.5, abs=0.5)
        assert p.y == pytest.approx(20 * 4 + 1.5, abs=0.5)

    def test_absent_object_is_lost_and_cancel_stops(self):
        class Gone(FakeTracker):
            def _segment(self, frames, seed_xy):
                for f in frames:
                    yield np.ones((8, 8), np.uint8), f.shape[:2], False

        frames = [np.zeros((32, 32), np.uint8)] * 5
        pts = list(Gone().track(frames, (0, 0)))
        assert all(p.flag == FLAG_LOST and np.isnan(p.x) for p in pts)
        n = iter(range(100))
        assert len(list(Gone().track(frames, (0, 0), cancel=lambda: next(n) >= 2))) == 2

    def test_sam2_status_reports_without_raising(self):
        ok, detail = sam2_track.sam2_status()
        assert isinstance(ok, bool) and detail


# ── clips ───────────────────────────────────────────────────────────────────

class TestClips:
    def test_read_clip_steps_through_native_frames(self, mono_h264_video):
        path, n = mono_h264_video
        seq = []
        cap = cv2.VideoCapture(path)
        for _ in range(n):
            seq.append(cap.read()[1])
        got = list(lt.read_clip(path, 5, 6, step=3))
        assert [i for i, _f in got] == [5, 8, 11, 14, 17, 20]
        for i, f in got:
            assert np.array_equal(f, seq[i])
        assert len(list(lt.read_clip(path, n - 2, 10))) == 2  # stops at the end

    def test_auto_starts_spread_and_fit(self):
        s = lt.auto_starts(1000, 4, 100)
        assert s == sorted(s) and s[0] >= 0 and s[-1] + 100 <= 1000
        assert lt.auto_starts(50, 3, 100) == [0, 0, 0] and lt.auto_starts(1000, 0, 10) == []

    def test_label_clip_pairs_frames_with_teacher_centres(self, mono_h264_video):
        path, _n = mono_h264_video
        clip = lt.label_clip(FakeTracker(), path, 4, 10, 2, (24, 34))
        assert clip["x"].shape == (10, localizer.IN_H, localizer.IN_W)
        assert list(clip["index"]) == list(range(4, 24, 2))
        assert tuple(clip["native_shape"]) == (80, 96)
        # conftest's blob: centre (20 + i, 30 + i % 10)
        i = clip["index"]
        assert np.allclose(clip["center"][:, 0], 20 + i, atol=1.0)
        assert np.allclose(clip["center"][:, 1], 30 + i % 10, atol=1.0)

    def test_qc_drops_lost_frames_and_area_jumps(self):
        clip = {"flag": np.array([0, 0, 2, 0, 0, 0]),
                "area": np.array([100, 110, 0, 400, 95, 30])}
        assert list(lt.qc_mask(clip)) == [True, True, False, False, True, False]

    def test_odd_clip_out_by_mask_area(self):
        clips = [{"area": np.full(5, a)} for a in (70000, 75000, 3200, 68000)]
        keeps = [np.ones(5, bool)] * 4
        assert lt.odd_clips(clips, keeps) == [2]
        assert lt.odd_clips(clips[:2], keeps[:2]) == []  # too few to tell

    def test_clip_cache_round_trip(self, mono_h264_video, tmp_path):
        clip = lt.label_clip(FakeTracker(), mono_h264_video[0], 0, 4, 1, (20, 30))
        p = str(tmp_path / "c.npz")
        lt.save_clip(p, clip)
        back = lt.load_clip(p)
        assert set(back) == set(clip)
        assert np.array_equal(back["x"], clip["x"]) and str(back["video"]) == mono_h264_video[0]


# ── training ────────────────────────────────────────────────────────────────

def _clip_from_frames(frames, video="synthetic"):
    t = FakeTracker()
    pts = list(t.track(frames, (0, 0)))
    return {
        "index": np.arange(len(frames)),
        "x": np.stack([localizer.network_input(f) for f in frames]),
        "center": np.array([[p.x, p.y] for p in pts], np.float32),
        "area": np.array([p.area for p in pts]),
        "flag": np.array([p.flag for p in pts], np.uint8),
        "native_shape": np.array(frames[0].shape[:2]),
        "seed": np.zeros(2, np.float32),
        "video": np.array(video),
    }


class TestTrain:
    def test_split_holds_out_whole_clips_or_a_tail(self):
        keeps = [np.ones(10, bool) for _ in range(4)]
        tr, va = lt.split_clips([{}] * 4, keeps)
        assert [v.any() for v in va] == [False, False, False, True]
        assert not (tr[3].any())
        tr, va = lt.split_clips([{}] * 2, keeps[:2])
        assert all(v[8:].all() and not v[:8].any() for v in va)
        assert all(t[:8].all() and not t[8:].any() for t in tr)

    def test_synthetic_negative_removes_the_animal(self):
        frames = _disk_frames(12)
        data = lt._Data([_clip_from_frames(frames)], [np.ones(12, bool)])
        neg = data.synth_negative(0, np.random.default_rng(0))
        assert neg is not None
        cx, cy = (int(v) for v in data.c[0])
        assert int(data.x[0, cy, cx]) < 100 and int(neg[cy, cx]) > 100

    def test_fine_tune_saves_loadable_weights_and_reports(self, tmp_path):
        clips = [_clip_from_frames(_disk_frames(12, h=88, w=96), f"v{k}") for k in range(3)]
        out = str(tmp_path / "loc.pt")
        seen = []
        res = lt.train_localizer(clips, out, init_weights=localizer.DEFAULT_WEIGHTS, steps=3,
                                 batch_size=4, device="cpu",
                                 on_progress=lambda s, t, m: seen.append((s, t)))
        model = res.pop("_model")
        assert os.path.exists(out) and seen[-1] == (3, 3)
        assert res["before"]["frames"] == res["after"]["frames"] == res["val_frames"] > 0
        assert res["after"]["median_error_px"] is not None and res["after"]["negatives"] > 0
        loaded = localizer.Localizer()
        loaded.load_state_dict(torch.load(out, weights_only=True))
        qc = lt.qc_sheet(model, clips, str(tmp_path / "qc.jpg"))
        assert cv2.imread(qc) is not None

    def test_no_usable_frames_is_an_error(self, tmp_path):
        clip = _clip_from_frames(_disk_frames(4))
        clip["flag"][:] = FLAG_LOST
        with pytest.raises(ValueError, match="no usable"):
            lt.train_localizer([clip], str(tmp_path / "x.pt"), steps=1, device="cpu")

    def test_cancel(self, tmp_path):
        clips = [_clip_from_frames(_disk_frames(8))]
        with pytest.raises(InterruptedError):
            lt.train_localizer(clips, str(tmp_path / "x.pt"), init_weights=localizer.DEFAULT_WEIGHTS,
                               steps=5, batch_size=2, device="cpu", cancel=lambda: True)
        assert not os.path.exists(tmp_path / "x.pt")


# ── service ─────────────────────────────────────────────────────────────────

def _task():
    return {"status": "running", "phase": None, "progress": 0, "total": 0, "cancel": False,
            "error": None, "message": None, "log": [], "result": None}


class TestService:
    def _run(self, video, out, tracker, **kw):
        from backend.services import localizer_service as svc
        args = dict(videos=[video], seeds=[{"video_path": video, "frame": 3, "x": 23, "y": 33}],
                    clips_per_video=2, frames_per_clip=8, frame_step=1, sam2_model="tiny",
                    negative_videos=[video], negatives_per_video=3, init="finetune",
                    init_weights=None, steps=2, batch_size=4, output_dir=out, name="rig_loc",
                    cache_dir=None, device="cpu", tracker=tracker)
        args.update(kw)
        task = _task()
        return task, svc.run_localizer_training(task, **args)

    def test_end_to_end_then_cached(self, mono_h264_video, stub_localizer, tmp_path):
        """Click seed + localizer-seeded clips -> labels cached -> weights,
        card and QC sheet written; a rerun reuses the SAM2 labels."""
        video, _n = mono_h264_video
        out = str(tmp_path / "models" / "localizer")
        tracker = FakeTracker()
        task, card = self._run(video, out, tracker)
        assert tracker.calls == 3  # 1 click + 2 auto clips
        assert os.path.exists(os.path.join(out, "rig_loc.pt"))
        on_disk = json.load(open(os.path.join(out, "rig_loc.json")))
        assert on_disk["teacher"] == "facebook/sam2.1-hiera-tiny"
        assert [c["seed_source"] for c in on_disk["clips"]] == ["click", "localizer", "localizer"]
        assert on_disk["seed_weights"] == localizer.DEFAULT_WEIGHTS
        assert card["qc_image"] and os.path.exists(card["qc_image"])
        assert len(os.listdir(os.path.join(out, "sam2_labels"))) == 3
        assert any("held-out error" in line for line in task["log"])

        tracker2 = FakeTracker()
        task, _card = self._run(video, out, tracker2)
        assert tracker2.calls == 0
        assert sum("cached labels" in line for line in task["log"]) == 3

    def test_nothing_to_label(self, mono_h264_video, tmp_path):
        with pytest.raises(ValueError, match="nothing to label"):
            self._run(mono_h264_video[0], str(tmp_path), FakeTracker(), seeds=[], clips_per_video=0)

    def test_unconfident_localizer_skips_clips(self, mono_h264_video, stub_localizer, tmp_path,
                                               monkeypatch):
        monkeypatch.setattr(lt, "find_seed", lambda *a, **k: None)
        task, card = self._run(mono_h264_video[0], str(tmp_path), FakeTracker())
        assert [c["seed_source"] for c in card["clips"]] == ["click"]
        assert sum("clip skipped" in line for line in task["log"]) == 2


@pytest_asyncio.fixture
async def ac():
    from backend.server import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


class TestRouter:
    @pytest.mark.asyncio
    async def test_info(self, ac):
        r = await ac.get("/api/localizer/info")
        assert r.status_code == 200
        d = r.json()
        assert d["default_weights"] == localizer.DEFAULT_WEIGHTS and d["default_weights_exist"]
        assert "tiny" in d["sam2_models"]

    @pytest.mark.asyncio
    async def test_train_validation(self, ac, tmp_path, monkeypatch):
        from backend.services import localizer_service as svc
        r = await ac.post("/api/localizer/train", json={"output_dir": str(tmp_path)})
        assert r.status_code == 422
        r = await ac.post("/api/localizer/train", json={
            "output_dir": str(tmp_path), "videos": ["v.mkv"], "name": "../escape"})
        assert r.status_code == 422
        real = svc.info
        monkeypatch.setattr(svc, "info", lambda: {**real(), "sam2_available": False,
                                                  "sam2_detail": "no transformers"})
        r = await ac.post("/api/localizer/train", json={"output_dir": str(tmp_path), "videos": ["v.mkv"]})
        assert r.status_code == 503 and "transformers" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_seed_preview(self, ac, mono_h264_video, monkeypatch):
        from backend.services import localizer_service as svc
        monkeypatch.setattr(svc, "_tracker", lambda model: FakeTracker())
        video = mono_h264_video[0]
        r = await ac.post("/api/localizer/seed-preview", json={"video_path": video, "frame": 5})
        assert r.status_code == 200 and r.json()["image"] and r.json()["center"] is None
        r = await ac.post("/api/localizer/seed-preview",
                          json={"video_path": video, "frame": 5, "x": 25, "y": 35})
        d = r.json()
        assert r.status_code == 200 and d["area"] > 0
        assert d["center"] == pytest.approx([25, 35], abs=1.0)
        r = await ac.post("/api/localizer/seed-preview", json={"video_path": video, "x": 3})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_train_streams_log_and_result(self, ac, mono_h264_video, stub_localizer,
                                                tmp_path, monkeypatch):
        from backend.services import localizer_service as svc
        monkeypatch.setattr(svc, "info", lambda: {"sam2_available": True, "sam2_detail": ""})
        real = svc.run_localizer_training
        monkeypatch.setattr(svc, "run_localizer_training",
                            lambda task, **kw: real(task, **kw, device="cpu", tracker=FakeTracker())
                            if "device" not in kw else real(task, **kw))
        video = mono_h264_video[0]
        r = await ac.post("/api/localizer/train", json={
            "output_dir": str(tmp_path), "seeds": [{"video_path": video, "frame": 2, "x": 22, "y": 32}],
            "frames_per_clip": 10, "frame_step": 1, "steps": 2, "batch_size": 4, "name": "n1"})
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        lines, last = [], None
        async with ac.stream("GET", f"/api/localizer/status/{task_id}") as resp:
            async for raw in resp.aiter_lines():
                if raw.startswith("data: "):
                    last = json.loads(raw[6:])
                    lines += last["log"]
        assert last["status"] == "completed", last
        assert last["result"]["weights"].endswith("n1.pt")
        assert len(lines) == len(set(lines)) and any("saved" in x for x in lines)

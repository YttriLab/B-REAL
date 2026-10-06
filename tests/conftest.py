"""
Shared fixtures for the B-REAL test suite.

Provides synthetic video frames, label CSVs, fake checkpoints,
and a FastAPI test client — all without requiring GPU or real video files.
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Temporary directory fixtures ──────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory (pytest built-in)."""
    return tmp_path


# ── Synthetic frame fixtures ──────────────────────────────────────────────

@pytest.fixture
def synthetic_frames(tmp_path):
    """
    Create a flat directory of synthetic 64x64 RGB PNG frames: 0.png .. 19.png.
    Returns the directory path.
    """
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    rng = np.random.RandomState(42)
    for i in range(20):
        img = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"{i}.png"), img)
    return str(frame_dir)


@pytest.fixture
def synthetic_frames_subfolder(tmp_path):
    """
    Create a subfolder-layout frame directory: frames/N/{0,1,2}.png for N in 2..19.
    Returns the directory path.
    """
    frame_dir = tmp_path / "frames_sub"
    frame_dir.mkdir()
    rng = np.random.RandomState(42)
    for n in range(2, 20):
        sub = frame_dir / str(n)
        sub.mkdir()
        for j in range(3):
            img = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(sub / f"{j}.png"), img)
    return str(frame_dir)


# ── Label CSV fixtures ────────────────────────────────────────────────────

@pytest.fixture
def label_csv_3class(tmp_path, synthetic_frames):
    """
    Create a 3-class one-hot label CSV matching frames 2..19.
    Columns: time, walk, groom, rear
    Returns the CSV path.
    """
    csv_path = tmp_path / "labels.csv"
    rows = []
    rng = np.random.RandomState(0)
    for frame_num in range(2, 20):
        label_idx = rng.randint(0, 3)
        row = {"time": frame_num, "walk": 0, "groom": 0, "rear": 0}
        cols = ["walk", "groom", "rear"]
        row[cols[label_idx]] = 1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(str(csv_path), index=False)
    return str(csv_path)


@pytest.fixture
def label_csv_2class(tmp_path):
    """
    Create a 2-class one-hot label CSV for frames 2..19.
    Columns: time, active, inactive
    """
    csv_path = tmp_path / "labels_2class.csv"
    rows = []
    for frame_num in range(2, 20):
        active = 1 if frame_num % 3 == 0 else 0
        rows.append({"time": frame_num, "active": active, "inactive": 1 - active})
    df = pd.DataFrame(rows)
    df.to_csv(str(csv_path), index=False)
    return str(csv_path)


# ── Synthetic video fixture ──────────────────────────────────────────────

@pytest.fixture
def synthetic_video(tmp_path):
    """
    Create a short synthetic 64x64 .avi video (30 frames) with a bright blob.
    Returns (video_path, num_frames).
    """
    video_path = str(tmp_path / "test_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 30
    size = (64, 64)
    writer = cv2.VideoWriter(video_path, fourcc, fps, size)
    rng = np.random.RandomState(7)

    # Create a static-ish background with a moving blob
    bg = rng.randint(180, 220, (64, 64, 3), dtype=np.uint8)
    for i in range(30):
        frame = bg.copy()
        # Draw a dark circle (simulates the animal)
        cx = 20 + i % 24
        cy = 32
        cv2.circle(frame, (cx, cy), 8, (30, 30, 30), -1)
        writer.write(frame)
    writer.release()
    return video_path, 30


@pytest.fixture
def synthetic_background(tmp_path, synthetic_video):
    """
    Create a background image (grayscale) for the synthetic video.
    Returns the background image path.
    """
    video_path, _ = synthetic_video
    bg_path = str(tmp_path / "bg.png")
    # Use a uniform gray as background (the average of frames without blob)
    bg = np.full((64, 64), 200, dtype=np.uint8)
    cv2.imwrite(bg_path, bg)
    return bg_path


@pytest.fixture
def mono_h264_video(tmp_path):
    """A monochrome h264 .mkv (yuv420p, chroma 128) — the kind of file the
    NVDEC path handles. 40 frames, 96x80, a dark blob moving on a lit floor.
    Returns (video_path, num_frames); skips if ffmpeg/libx264 is unavailable."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available")
    n, h, w = 40, 80, 96
    rng = np.random.RandomState(3)
    floor = rng.randint(150, 210, (h, w), dtype=np.uint8)
    frames = []
    for i in range(n):
        f = floor.copy()
        cv2.circle(f, (20 + i, 30 + (i % 10)), 9, 25, -1)
        frames.append(f)
    path = str(tmp_path / "mono.mkv")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        input=np.stack(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg/libx264 failed: {proc.stderr.decode()[-200:]}")
    return path, n


class DarkCentroidLocalizer(__import__("torch").nn.Module):
    """Stands in for the SAM2-distilled localizer network: same input/output
    contract, but the centre is just the centroid of the dark pixels, and the
    animal is "present" when there are any."""

    def forward(self, x_u8):
        import torch
        x = x_u8.float()
        dark = (x < 100).float()
        B, H, W = x.shape
        mass = dark.sum((1, 2))
        xs = torch.arange(W).float() + 0.5
        ys = torch.arange(H).float() + 0.5
        cx = (dark.sum(1) * xs).sum(1) / mass.clamp(min=1)
        cy = (dark.sum(2) * ys).sum(1) / mass.clamp(min=1)
        heat = torch.zeros(B, H // 8, W // 8)
        presence_logit = torch.where(mass > 0, torch.tensor(10.0), torch.tensor(-10.0))
        return torch.stack([cx, cy], 1), heat, presence_logit


@pytest.fixture
def stub_localizer(monkeypatch, tmp_path):
    """Route every localizer load to DarkCentroidLocalizer (on CPU) and return
    a weights path to put in crop_params — the file only has to exist."""
    import localizer
    model = DarkCentroidLocalizer()
    monkeypatch.setattr(localizer, "load_localizer", lambda path, device=None: model)
    monkeypatch.setattr(localizer.torch.cuda, "is_available", lambda: False)
    path = tmp_path / "loc_stub.pt"
    path.write_bytes(b"")
    return str(path)


# ── Sample project fixture (for .rlt / project router tests) ──────────────

@pytest.fixture
def sample_project(tmp_path):
    """
    Build a realistic project directory on disk and return (project_dir, manifest).

    Includes the regenerable data (frames, backgrounds, a fake checkpoint) so
    tests can assert that .rlt export deliberately *excludes* them.
    """
    proj = tmp_path / "sample_project"
    for sub in ("labels", "frames", "backgrounds",
                os.path.join("models", "checkpoints"),
                os.path.join("models", "exports")):
        os.makedirs(proj / sub, exist_ok=True)

    manifest = {
        "name": "sample_project",
        "created": "2026-01-01",
        "version": 1,
        "behaviors": [
            {"name": "grooming", "key": "1"},
            {"name": "walking", "key": "2"},
        ],
        "videos": [
            {
                "id": "vid_a",
                "path": "/abs/path/to/vid_a.mp4",
                "labels": "labels/vid_a_bouts.csv",
                "frames": "frames/vid_a/",
                "background": "backgrounds/bg_vid_a.png",
            },
            {
                "id": "vid_b",
                "path": "/abs/path/to/vid_b.mp4",
                "labels": "labels/vid_b_bouts.csv",
                "frames": "frames/vid_b/",
                "background": "backgrounds/bg_vid_b.png",
            },
        ],
        "preprocessing": {"box_size": 128, "difference_threshold": 10},
        "training": {"variant": "b0", "max_epochs": 50},
        "models": [
            {
                "name": "epoch=1.ckpt",
                "path": "/abs/path/to/models/checkpoints/epoch=1.ckpt",
                "created": "2026-01-02",
                "val_loss": 0.123,
                "config": {"variant": "b0", "binarize": None,
                           "num_frames": 3, "tsm_stages": [5, 6, 7]},
            }
        ],
    }
    (proj / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Label CSVs (the irreplaceable hand-annotation — should be packed)
    (proj / "labels" / "vid_a_bouts.csv").write_text(
        "start,end,behavior\n00:01.000,00:05.500,grooming\n", encoding="utf-8"
    )
    (proj / "labels" / "vid_b_bouts.csv").write_text(
        "start,end,behavior\n00:02.000,00:03.000,walking\n", encoding="utf-8"
    )

    # Regenerable data (should NOT be packed)
    rng = np.random.RandomState(1)
    for vid in ("vid_a", "vid_b"):
        fdir = proj / "frames" / vid
        fdir.mkdir(exist_ok=True)
        for i in range(3):
            img = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(fdir / f"{i}.png"), img)
        bg = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(proj / "backgrounds" / f"bg_{vid}.png"), bg)
    (proj / "models" / "checkpoints" / "epoch=1.ckpt").write_bytes(b"\x00" * 1024)

    return str(proj), manifest


# ── FastAPI test client ──────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Provide an httpx AsyncClient for testing FastAPI endpoints.
    Requires httpx to be installed.
    """
    from httpx import AsyncClient, ASGITransport
    from backend.server import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Cleanup task registries between tests ─────────────────────────────────

@pytest.fixture(autouse=True)
def clear_task_registries():
    """Cancel and clear all service task registries after each test."""
    yield
    # Cleanup after test. Cancel first so background work a test started
    # stops instead of running on into the next test.
    from backend.services import (
        preprocess_service, training_service, tuning_service, inference_service,
    )
    for svc in (tuning_service, preprocess_service, training_service,
                inference_service):
        try:
            svc.cancel_all()
        except Exception:
            pass
    preprocess_service._tasks.clear()
    training_service._tasks.clear()
    tuning_service._tasks.clear()
    inference_service._tasks.clear()

"""
Shared fixtures for the RealtimeASOID test suite.

Provides synthetic video frames, label CSVs, fake checkpoints,
and a FastAPI test client — all without requiring GPU or real video files.
"""

import json
import os
import sys
import tempfile
import shutil

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
    """Clear all service task registries before each test."""
    yield
    # Cleanup after test
    from backend.services import preprocess_service, training_service, inference_service
    preprocess_service._tasks.clear()
    training_service._tasks.clear()
    inference_service._tasks.clear()

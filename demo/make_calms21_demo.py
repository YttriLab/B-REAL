"""Build the small CalMS21 demo project in demo/calms21_demo/.

Cuts a few one-minute clips out of CalMS21 Task 1 training videos, turns the
expert per-frame labels for those frames into bouts, and writes a normal B-REAL
project folder through ProjectStore — so it is exactly what the app writes.

The output is committed, so you only need this to rebuild or change the demo.
Needs a local copy of CalMS21 Task 1 (videos + one-hot label CSVs):

    pixi run python demo/make_calms21_demo.py --calms21 /path/to/Calms21

CalMS21: Sun et al. 2021, https://data.caltech.edu/records/s0vdx-0k302, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gui"))
from labeling_shell.bouts import Behavior, Bout, LabelModel  # noqa: E402
from labeling_shell.csv_io import generate_training_label_csv  # noqa: E402
from labeling_shell.project import ProjectStore  # noqa: E402

CLASSES = ["attack", "investigation", "mount", "other"]  # CalMS21 Task 1 vocab order
CLIP_FRAMES = 1800  # 60 s at 30 fps

# (source sequence, first frame, labeled?). Windows chosen for a mix of attack,
# mount and investigation; the unlabeled clip is for trying Active Learning,
# with its expert labels kept in ground_truth/ to compare against.
CLIPS = [
    ("mouse070_task1_annotator1", 4200, True),
    ("mouse018_task1_annotator1", 1350, True),
    ("mouse055_task1_annotator1", 4500, True),
    ("mouse001_task1_annotator1", 1350, False),
]


def runs_to_bouts(labels: np.ndarray, behaviors: dict[str, Behavior]) -> list[Bout]:
    """Per-frame class ids -> one bout per run of equal labels (end inclusive)."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges - 1, len(labels) - 1]
    out = []
    for s, e in zip(starts, ends):
        b = behaviors[CLASSES[labels[s]]]
        out.append(Bout(b.id, b.name, int(s), int(e)))
    return out


def cut_clip(src: Path, dst: Path, start: int, n: int) -> None:
    """Frames [start, start+n) of src at half size (512 px wide), re-encoded
    small (~2 MB/min) so the demo can live in git. trim by frame number is
    exact, so clip frame k is source frame start+k."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", f"trim=start_frame={start}:end_frame={start + n},setpts=PTS-STARTPTS,format=gray,scale=512:-2,format=yuv420p",
         "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "32", "-g", "30",
         "-movflags", "+faststart", str(dst)],
        check=True)


def check_alignment(src: Path, dst: Path, start: int, n: int) -> None:
    """The clip must have n frames, and clip frame k must match source frame
    start+k better than its neighbours (lossy, so not bit-equal)."""
    cap = cv2.VideoCapture(str(dst))
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert got == n, f"{dst.name}: {got} frames, expected {n}"
    for k in (0, n // 2, n - 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        clip = cap.read()[1][..., 0].astype(np.float32)
        s = cv2.VideoCapture(str(src))
        s.set(cv2.CAP_PROP_POS_FRAMES, start + k - 1)
        size = clip.shape[::-1]
        err = [np.abs(cv2.resize(s.read()[1][..., 0], size, interpolation=cv2.INTER_AREA)
                      .astype(np.float32) - clip).mean() for _ in range(3)]
        assert int(np.argmin(err)) == 1, f"{dst.name} frame {k} misaligned: {err}"
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calms21", required=True, type=Path,
                    help="folder with task1_videos_mp4/train/*.mp4 and labels_calms21/*.csv")
    ap.add_argument("--out", type=Path, default=REPO / "demo" / "calms21_demo")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    store = ProjectStore(out)
    store.create("CalMS21 demo")
    (out / "videos").mkdir()
    (out / "ground_truth").mkdir()

    # Fixed ids so a rebuild doesn't churn every file.
    behaviors = {name: Behavior(id=f"calms21-{name}", name=name, key=str(i + 1))
                 for i, name in enumerate(CLASSES)}
    store.save_behaviors(list(behaviors.values()))

    for seq, start, labeled in CLIPS:
        src = args.calms21 / "task1_videos_mp4" / "train" / f"{seq}.mp4"
        csv = args.calms21 / "labels_calms21" / f"{seq}.csv"
        vid = f"{seq.split('_')[0]}_f{start}"
        dst = out / "videos" / f"{vid}.mp4"
        print(f"{vid}: frames {start}-{start + CLIP_FRAMES - 1} of {seq}")

        df = pd.read_csv(csv)
        assert (df["time"].to_numpy() == np.arange(len(df))).all(), f"{csv} is not dense"
        labels = df[CLASSES].to_numpy().argmax(1)[start:start + CLIP_FRAMES]

        cut_clip(src, dst, start, CLIP_FRAMES)
        check_alignment(src, dst, start, CLIP_FRAMES)
        store.register_video(vid, dst.name, dst, frame_count=CLIP_FRAMES)

        model = LabelModel()
        model.behaviors = list(behaviors.values())
        model.bouts = runs_to_bouts(labels, behaviors)
        if labeled:
            store.save_video_labels(vid, model)
        else:
            (out / "ground_truth" / f"{vid}_labels.csv").write_text(
                generate_training_label_csv(model.bouts), encoding="utf-8")

    # Crop: whole frame. Both mice matter for these social behaviors, and the
    # bedding shifts too much for background subtraction (it is a demo).
    store.set_setting("preprocess", {"method": "none", "params": {}})

    # Open on the first clip. Paths: `path` is absolute for this machine, but
    # ProjectStore falls back to `rel_path`, so the committed folder opens anywhere.
    store.set_last_video(f"{CLIPS[0][0].split('_')[0]}_f{CLIPS[0][1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

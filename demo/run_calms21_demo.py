"""Run the CalMS21 demo end to end, headless, and score it.

Does what the demo's README walks through in the GUI, through the same backend
API calls: preprocess the three labeled clips, train with the Train tab's
defaults, run Active Learning inference on the unlabeled clip, then compare
its predictions with the expert labels in ground_truth/. Prints the time each
step took and the scores; this is where the README's expected results come from.

    pixi run python demo/run_calms21_demo.py            # GUI defaults
    pixi run python demo/run_calms21_demo.py --max-epochs 1 --batch-size 16   # quick check

Works on a temporary copy of the demo, so demo/calms21_demo/ is left as it is.
Starts its own backend on --port (not the GUI's 8742).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "demo" / "calms21_demo"
sys.path.insert(0, str(REPO / "gui"))
from labeling_shell.frame_ranges import compute_labeled_frame_ranges  # noqa: E402
from labeling_shell.project import ProjectStore  # noqa: E402
from plugins.pipeline_config import build_train_config  # noqa: E402

UNLABELED = "mouse001_f1350"


def sse(client: httpx.Client, url: str):
    """Yield each JSON event of a server-sent-event stream."""
    with client.stream("GET", url, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


def wait_task(client, url):
    last = None
    for ev in sse(client, url):
        last = ev
        if ev.get("status") in ("error", "cancelled"):
            raise RuntimeError(f"{url}: {ev.get('error') or ev}")
    return last


def scores(truth: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    """Balanced accuracy and macro F1 over the classes present in the truth."""
    present = [c for c in range(len(classes)) if (truth == c).any()]
    recall, f1 = {}, {}
    for c in present:
        tp = np.sum((pred == c) & (truth == c))
        rec = tp / np.sum(truth == c)
        prec = tp / max(np.sum(pred == c), 1)
        recall[classes[c]] = rec
        f1[classes[c]] = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return {"balanced_accuracy": float(np.mean(list(recall.values()))),
            "macro_f1": float(np.mean(list(f1.values()))),
            "accuracy": float(np.mean(truth == pred)),
            "recall": {k: round(float(v), 3) for k, v in recall.items()},
            "f1": {k: round(float(v), 3) for k, v in f1.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8743)
    ap.add_argument("--max-epochs", type=int, default=None, help="default: the Train tab's")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the Train tab's")
    ap.add_argument("--keep", action="store_true", help="keep the temporary project copy")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="breal_demo_"))
    proj = work / "calms21_demo"
    shutil.copytree(DEMO, proj, ignore=shutil.ignore_patterns("frames", "models", "predictions", "backgrounds"))
    store = ProjectStore(proj)
    videos = {v["id"]: v for v in store.list_videos()}
    saved = store.get_setting("preprocess") or {}
    crop = {**saved.get("params", {}), "method": saved.get("method", "none")}

    base = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.server:app", "--port", str(args.port), "--host", "127.0.0.1"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    times = {}
    try:
        client = httpx.Client(base_url=base, timeout=60)
        for _ in range(120):
            try:
                health = client.get("/api/health").json()
                break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            raise RuntimeError("backend did not start")
        print("device:", json.dumps(health.get("device")))

        # 1. Preprocess the labeled clips (labeled frames + lead-in, as the GUI does).
        t0 = time.monotonic()
        label_paths, frame_dirs = [], []
        for vid, rec in videos.items():
            if vid == UNLABELED:
                continue
            model = store.load_into_model(vid)
            ranges = compute_labeled_frame_ranges(model.bouts, 3)
            out = store.frames_dir(vid)
            r = client.post("/api/preprocess/run", json={
                # The backend writes to output_dir/output_name, as the GUI calls it.
                "video_path": rec["path"], "output_dir": str(out.parent), "output_name": out.name,
                "crop_params": crop, "frame_ranges": ranges, "crop": crop["method"] != "none"})
            r.raise_for_status()
            wait_task(client, f"/api/preprocess/status/{r.json()['task_id']}")
            label_paths.append(str(store.label_csv_path(vid)))
            frame_dirs.append(str(out))
        times["preprocess"] = time.monotonic() - t0
        print(f"preprocess: {times['preprocess']:.0f} s")

        # 2. Train with the Train tab's defaults.
        overrides = {k: v for k, v in (("max_epochs", args.max_epochs), ("batch_size", args.batch_size)) if v}
        config = build_train_config(label_paths, frame_dirs, **overrides)
        config["checkpoint_dir"] = str(store.checkpoint_dir())
        t0 = time.monotonic()
        r = client.post("/api/training/start", json=config)
        r.raise_for_status()
        epochs, test, ckpt = 0, {}, None
        for ev in sse(client, f"/api/training/status/{r.json()['task_id']}"):
            if ev.get("status") in ("error", "cancelled"):
                raise RuntimeError(f"training: {ev.get('error') or ev}")
            if "epoch" in ev:
                epochs = max(epochs, int(ev["epoch"]))  # the final event says how many ran
            test.update({k: v for k, v in ev.items() if str(k).startswith("test")})
            ckpt = ev.get("best_checkpoint") or ckpt
        times["train"] = time.monotonic() - t0
        if not ckpt or not os.path.exists(ckpt):
            raise RuntimeError(f"training finished without a checkpoint ({ckpt})")
        print(f"train: {times['train']:.0f} s, {epochs} epochs, checkpoint {Path(ckpt).name}")
        if test:
            print("held-out split of the training clips:", json.dumps(test))

        # 3. Active Learning inference on the unlabeled clip.
        t0 = time.monotonic()
        r = client.post("/api/inference/run", json={
            "checkpoint_path": ckpt, "video_path": videos[UNLABELED]["path"], "crop_params": crop})
        r.raise_for_status()
        tid = r.json()["task_id"]
        wait_task(client, f"/api/inference/status/{tid}")
        res = client.get(f"/api/inference/results/{tid}").json()
        times["inference"] = time.monotonic() - t0
        print(f"inference: {times['inference']:.0f} s")

        # 4. Score against the expert labels.
        names = res["class_names"]
        gt = pd.read_csv(proj / "ground_truth" / f"{UNLABELED}_labels.csv")
        truth_name = dict(zip(gt["time"], gt.drop(columns="time").idxmax(axis=1)))
        idx, truth, pred = [], [], []
        for f in res["frames"]:
            if f["index"] in truth_name:
                p = f["predicted_class"]
                truth.append(names.index(truth_name[f["index"]]))
                pred.append(p if isinstance(p, int) else names.index(p))
        s = scores(np.array(truth), np.array(pred), names)
        print(f"{UNLABELED} vs expert labels ({len(truth)} frames):")
        print(json.dumps(s, indent=2))
        print(json.dumps({"seconds": {k: round(v) for k, v in times.items()},
                          "total_minutes": round(sum(times.values()) / 60, 1)}))
    finally:
        server.terminate()
        server.wait(timeout=30)
        if args.keep:
            print("kept", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

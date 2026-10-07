"""Train Localizer: SAM2 labels clips, a localizer is distilled from them.

The long-running job (``start_localizer_training``) runs in phases, reported
on the task as ``phase``/``progress``/``total`` plus a ``log`` of lines:

1. **plan**: clips from the user's seed clicks, plus ``clips_per_video``
   evenly spaced clips per video, each seeded by the current localizer's
   guess on its first confident frame;
2. **label**: SAM2 tracks each clip. Labels are cached per clip in
   ``cache_dir``, so changing only the training settings and rerunning skips
   straight to training;
3. **negatives**: frames sampled from empty-arena videos, if any;
4. **train**: see localizer_train.train_localizer.

``seed_preview`` runs SAM2 on one frame for one click, so a seed can be checked
before a clip is committed to it.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import os
import threading
import uuid
from typing import Optional

import cv2
import numpy as np

_tasks: dict[str, dict] = {}
_tracker_lock = threading.Lock()
_preview_tracker = None  # one SAM2 model kept warm for seed previews


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


def info() -> dict:
    from localizer import DEFAULT_WEIGHTS
    from sam2_track import DEFAULT_MODEL, SAM2_MODELS, sam2_status
    ok, detail = sam2_status()
    return {
        "default_weights": DEFAULT_WEIGHTS,
        "default_weights_exist": os.path.exists(DEFAULT_WEIGHTS),
        "sam2_available": ok,
        "sam2_detail": detail,
        "sam2_models": list(SAM2_MODELS),
        "sam2_default_model": DEFAULT_MODEL,
    }


def _encode_jpg(img: np.ndarray, max_side: int = 1280) -> str:
    scale = min(1.0, max_side / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf).decode("ascii")


def _tracker(model: str):
    global _preview_tracker
    from sam2_track import SAM2_MODELS, Sam2Tracker
    want = SAM2_MODELS.get(model, model)
    if _preview_tracker is None or _preview_tracker.model_id != want:
        if _preview_tracker is not None:
            _preview_tracker.close()
        _preview_tracker = Sam2Tracker(model)
    return _preview_tracker


def release_preview_tracker() -> None:
    global _preview_tracker
    with _tracker_lock:
        if _preview_tracker is not None:
            _preview_tracker.close()
            _preview_tracker = None


def seed_preview(video_path: str, frame_index: int, x: Optional[float] = None,
                 y: Optional[float] = None, sam2_model: str = "tiny") -> dict:
    """The frame (to click on), and with a click, SAM2's mask and the body
    centre the localizer would be trained toward."""
    from localizer_train import read_frame
    frame = read_frame(video_path, frame_index)
    h, w = frame.shape[:2]
    out = {"frame_index": frame_index, "width": w, "height": h, "center": None, "area": 0}
    vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if x is not None and y is not None:
        with _tracker_lock:
            pt, mask = _tracker(sam2_model).segment_one(frame, (x, y))
        r = max(4, round(max(h, w) / 150))
        if pt.area:
            tint = vis[mask > 0].astype(np.float32) * 0.5 + np.array([0, 127, 0], np.float32)
            vis[mask > 0] = tint.astype(np.uint8)
            cv2.drawMarker(vis, (int(pt.x), int(pt.y)), (255, 0, 255), cv2.MARKER_CROSS, 4 * r, max(2, r // 2))
            out.update(center=[pt.x, pt.y], area=pt.area, flag=pt.flag)
        cv2.circle(vis, (int(x), int(y)), r, (0, 0, 255), -1)
    out["image"] = _encode_jpg(vis)
    out["image_scale"] = min(1.0, 1280 / max(h, w))
    return out


def _clip_cache_path(cache_dir: str, video: str, start: int, n: int, step: int,
                     model: str, seed) -> str:
    from sam2_track import LABEL_VERSION
    key = f"{os.path.abspath(video)}|{os.path.getsize(video)}|{start}|{n}|{step}|{model}|" \
          f"{round(seed[0])},{round(seed[1])}|v{LABEL_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(cache_dir, f"{stem}_{start}_{n}x{step}_{digest}.npz")


def _label_all(task, plan, cache_dir, frames_per_clip, frame_step, sam2_model, tracker,
               device, log, cancel) -> list[dict]:
    """SAM2-label every planned clip (or load its cached labels)."""
    import localizer_train as lt
    from sam2_track import SAM2_MODELS, Sam2Tracker
    clips, made = [], None
    try:
        for k, (v, start, seed, source) in enumerate(plan):
            path = _clip_cache_path(cache_dir, v, start, frames_per_clip, frame_step,
                                    SAM2_MODELS.get(sam2_model, sam2_model), seed)
            if os.path.exists(path):
                clip = lt.load_clip(path)
                task["progress"] += frames_per_clip
                log(f"clip {k + 1}/{len(plan)} ({os.path.basename(v)} @ {start}): cached labels")
            else:
                if tracker is None:
                    release_preview_tracker()  # one SAM2 on the GPU at a time
                    tracker = made = Sam2Tracker(sam2_model, device=device)
                done0 = task["progress"]

                def tick():
                    task["progress"] += 1
                clip = lt.label_clip(tracker, v, start, frames_per_clip, frame_step, seed,
                                     cancel=cancel, on_frame=tick)
                if cancel():
                    raise InterruptedError("cancelled")
                task["progress"] = done0 + frames_per_clip
                lt.save_clip(path, clip)
                kept = int(lt.qc_mask(clip).sum())
                log(f"clip {k + 1}/{len(plan)} ({os.path.basename(v)} @ {start}, {source} seed): "
                    f"SAM2 kept {kept}/{len(clip['flag'])} frames")
            clip["source"] = np.array(source)
            clips.append(clip)
    finally:
        if made is not None:
            made.close()  # free the GPU for training
    return clips


def run_localizer_training(task: dict, *, videos: list[str], seeds: list[dict],
                           clips_per_video: int, frames_per_clip: int, frame_step: int,
                           sam2_model: str, negative_videos: list[str], negatives_per_video: int,
                           init: str, init_weights: Optional[str], steps: int, batch_size: int,
                           output_dir: str, name: Optional[str], cache_dir: Optional[str],
                           device: Optional[str] = None, tracker=None) -> dict:
    import torch

    import localizer_train as lt
    from localizer import load_localizer, resolve_weights
    from sam2_track import SAM2_MODELS

    cancel = lambda: task["cancel"]  # noqa: E731

    def log(msg: str) -> None:
        task["log"].append(msg)
        task["message"] = msg
        print(f"[localizer] {msg}", flush=True)

    def phase(name: str, total: int) -> None:
        task.update(phase=name, progress=0, total=total)

    seed_weights = resolve_weights(init_weights)
    name = name or f"localizer_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(output_dir, "sam2_labels")
    os.makedirs(cache_dir, exist_ok=True)
    span = frames_per_clip * frame_step

    # 1. plan --------------------------------------------------------------
    phase("plan", len(videos))
    plan = [(s["video_path"], int(s["frame"]), (float(s["x"]), float(s["y"])), "click")
            for s in seeds]
    if clips_per_video > 0 and videos:
        seeder = load_localizer(seed_weights)
        for v in videos:
            if cancel():
                raise InterruptedError("cancelled")
            starts = lt.auto_starts(lt.frame_count(v), clips_per_video, span)
            found = 0
            for st in starts:
                hit = lt.find_seed(seeder, v, st)
                if hit is None:
                    log(f"{os.path.basename(v)} @ {st}: localizer never confident the "
                        f"animal is there; clip skipped (add a click seed instead)")
                    continue
                plan.append((v, hit[0], hit[1], "localizer"))
                found += 1
            task["progress"] += 1
            log(f"{os.path.basename(v)}: {found}/{len(starts)} clips seeded by the localizer")
    if not plan:
        raise ValueError("nothing to label: add click seeds, or videos with clips per video > 0")
    log(f"{len(plan)} clips x {frames_per_clip} frames (frame step {frame_step}) to label with "
        f"SAM2 {sam2_model}")

    # 2. label -------------------------------------------------------------
    phase("label", len(plan) * frames_per_clip)
    clips = _label_all(task, plan, cache_dir, frames_per_clip, frame_step, sam2_model,
                       tracker, device, log, cancel)

    # 3. negatives ---------------------------------------------------------
    negatives = None
    if negative_videos and negatives_per_video > 0:
        phase("negatives", len(negative_videos))
        negatives = lt.sample_negatives(negative_videos, negatives_per_video)
        task["progress"] = len(negative_videos)
        log(f"{len(negatives)} empty-arena frames from {len(negative_videos)} videos")

    # 4. train -------------------------------------------------------------
    phase("train", steps)
    out_path = os.path.join(output_dir, f"{name}.pt")

    def on_progress(step, total, msg):
        task["progress"] = step
        if step % 500 == 0 or step == total:
            log(msg)
        else:
            task["message"] = msg

    res = lt.train_localizer(
        clips, out_path, negatives=negatives,
        init_weights=seed_weights if init == "finetune" else None,
        steps=steps, batch_size=batch_size, device=device, on_progress=on_progress, cancel=cancel)
    model = res.pop("_model")
    for i in res["dropped_clips"]:
        c, keep = clips[i], lt.qc_mask(clips[i])
        what = (f"SAM2 kept only {int(keep.sum())}/{len(keep)} frames (it lost the animal)"
                if keep.sum() < 0.5 * len(keep) else
                f"its SAM2 mask is {int(np.median(c['area'][keep])):,} px, far from the other "
                f"clips' (a part of the animal, or the animal plus something else)")
        log(f"clip {i + 1} ({os.path.basename(str(c['video']))} @ {int(c['index'][0])}) left out: "
            f"{what}; check its seed")
    qc = lt.qc_sheet(model, clips, os.path.join(output_dir, f"{name}_qc.jpg"))
    res["qc_image"] = qc or None
    card = {
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "teacher": SAM2_MODELS.get(sam2_model, sam2_model),
        "seed_weights": seed_weights,
        "clips": [{"video": str(c["video"]), "start": int(c["index"][0]) if len(c["index"]) else None,
                   "frames": int(len(c["index"])), "step": frame_step,
                   "kept": int(lt.qc_mask(c).sum()), "seed": [float(v) for v in c["seed"]],
                   "seed_source": str(c["source"])} for c in clips],
        "negative_videos": negative_videos,
        **res,
    }
    lt.write_card(os.path.join(output_dir, f"{name}.json"), card)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def fmt(m):
        return "n/a" if not m or m["median_error_px"] is None else \
            f"median {m['median_error_px']:.1f} px, p95 {m['p95_error_px']:.1f} px"
    if res["before"]:
        log(f"held-out error: before {fmt(res['before'])} -> after {fmt(res['after'])} "
            f"(best at step {res['best_step']})")
        if not res["improved"]:
            log("training never beat the starting weights on the held-out clips; the saved "
                "file is a copy of them. Keep the current localizer, or label more clips.")
    else:
        log(f"held-out error: {fmt(res['after'])}")
    log(f"saved {out_path}")
    return card


def start_localizer_training(**kw) -> str:
    task_id = str(uuid.uuid4())
    task = {"status": "running", "phase": "plan", "progress": 0, "total": 0, "cancel": False,
            "error": None, "message": None, "log": [], "result": None}
    _tasks[task_id] = task

    def _run():
        try:
            task["result"] = run_localizer_training(task, **kw)
            task["status"] = "completed"
        except InterruptedError:
            task["status"] = "cancelled"
        except Exception as e:
            if task["cancel"]:
                task["status"] = "cancelled"
            else:
                task["status"] = "error"
                task["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=_run, daemon=True).start()
    return task_id

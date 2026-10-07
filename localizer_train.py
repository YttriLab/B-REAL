"""Train a localizer for a new rig by distilling SAM2 (sam2_track.py).

1. **Label clips.** Each clip is ``n`` frames, every ``step``-th native frame
   from ``start``, tracked by SAM2 from one click (a user's click, or the
   current localizer's guess on the first frame). Per frame this keeps the
   network input (``localizer.network_input``, so training sees exactly what
   inference will), the SAM2 body centre and mask area. Frames where SAM2 lost
   the animal, or where the mask area jumps (the track merged with or
   switched to something else), are dropped (``qc_mask``).
2. **Train.** ``localizer.Localizer`` learns heatmap + soft-argmax centre on
   the kept frames, and presence on two kinds of negatives: synthetic ones (the
   animal's window pasted over from a frame where it is elsewhere, so the
   arena looks empty) and, optionally, real empty-arena frames. Augmentation:
   brightness/contrast/gamma/noise, flips, scale and shift.

Ported from the scripts the bundled weights came from (paul_all/code:
sam2_chunk_extract.py, build_small.py, train_loc2.py).
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from localizer import IN_H, IN_W, STRIDE, Localizer, network_input
from sam2_track import FLAG_LOST, Sam2Tracker

Cancel = Optional[Callable[[], bool]]


# ── clips ───────────────────────────────────────────────────────────────────

def read_clip(video: str, start: int, n: int, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """-> (native index, frame) for every ``step``-th frame from ``start``.

    Seeks once (cv2 lands on the exact frame for the indexed containers this
    app records; a clip only needs its frames to match their own SAM2 labels,
    which they do whatever the seek does) then reads sequentially."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for k in range(n * step):
            if k % step:
                if not cap.grab():
                    return
                continue
            ok, frame = cap.read()
            if not ok:
                return
            yield start + k, frame
    finally:
        cap.release()


def read_frame(video: str, index: int) -> np.ndarray:
    for _i, frame in read_clip(video, index, 1):
        return frame
    raise RuntimeError(f"could not read frame {index} of {video}")


def frame_count(video: str) -> int:
    cap = cv2.VideoCapture(video)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def auto_starts(total: int, clips: int, span: int) -> list[int]:
    """``clips`` start frames spread evenly through a ``total``-frame video,
    each leaving room for ``span`` frames."""
    room = max(0, total - span)
    return [int(round(room * (k + 0.5) / clips)) for k in range(clips)] if clips > 0 else []


@torch.no_grad()
def find_seed(model: Localizer, video: str, start: int, tries: int = 20, every: int = 15,
              min_presence: float = 0.5):
    """Look for a frame at or after ``start`` where ``model`` is confident the
    animal is there -> (frame index, (x, y) native px) or None."""
    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    for idx, frame in read_clip(video, start, tries, every):
        x = torch.from_numpy(network_input(frame)[None]).to(device)
        xy, _lg, pl = model(x)
        if torch.sigmoid(pl.float())[0] >= min_presence:
            sx, sy = frame.shape[1] / IN_W, frame.shape[0] / IN_H
            return idx, (float(xy[0, 0]) * sx, float(xy[0, 1]) * sy)
    return None


def label_clip(tracker: Sam2Tracker, video: str, start: int, n: int, step: int, seed_xy,
               cancel: Cancel = None, on_frame: Optional[Callable[[], None]] = None) -> dict:
    """SAM2-track one clip -> arrays: index, x (network inputs), center
    (native px), area, flag; plus native_shape and the seed."""
    small, shape = [], None

    def frames():
        nonlocal shape
        for _idx, frame in read_clip(video, start, n, step):
            shape = frame.shape[:2]
            small.append(network_input(frame))
            yield frame

    pts = []
    for p in tracker.track(frames(), seed_xy, cancel=cancel):
        pts.append(p)
        if on_frame:
            on_frame()
    pts = np.array(pts, dtype=np.float64).reshape(-1, 4)
    k = len(pts)
    return {
        "index": start + step * np.arange(k, dtype=np.int64),
        "x": np.stack(small[:k]) if k else np.zeros((0, IN_H, IN_W), np.uint8),
        "center": pts[:, :2].astype(np.float32),
        "area": pts[:, 2].astype(np.int64),
        "flag": pts[:, 3].astype(np.uint8),
        "native_shape": np.array(shape or (0, 0), np.int64),
        "seed": np.array(seed_xy, np.float32),
        "video": np.array(video),
    }


def qc_mask(clip: dict, low: float = 0.4, high: float = 2.5) -> np.ndarray:
    """Frames worth training on: SAM2 still had the animal, and the mask area
    is within [low, high] x the clip's median (a sudden jump means the track
    swallowed or switched to something else)."""
    ok = clip["flag"] != FLAG_LOST
    if ok.any():
        med = np.median(clip["area"][ok])
        ok &= (clip["area"] >= low * med) & (clip["area"] <= high * med)
    return ok


def odd_clips(clips: list[dict], keeps: list[np.ndarray], low: float = 1 / 3,
              high: float = 3.0) -> list[int]:
    """Clips whose typical mask area is far from the others' (SAM2 tracked a
    part of the animal, or the animal plus something else). Needs 3+ clips
    to say which is odd one out."""
    med = {i: float(np.median(c["area"][k])) for i, (c, k) in enumerate(zip(clips, keeps)) if k.any()}
    if len(med) < 3:
        return []
    typical = float(np.median(list(med.values())))
    return [i for i, a in med.items() if not low * typical <= a <= high * typical]


def save_clip(path: str, clip: dict) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **clip)
    os.replace(tmp, path)


def load_clip(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_negatives(videos: list[str], per_video: int) -> np.ndarray:
    """Network inputs sampled evenly across empty-arena videos."""
    out = []
    for v in videos:
        total = frame_count(v)
        for idx in np.linspace(0, max(0, total - 1), per_video).astype(int):
            try:
                out.append(network_input(read_frame(v, int(idx))))
            except RuntimeError:
                continue
    return np.stack(out) if out else np.zeros((0, IN_H, IN_W), np.uint8)


# ── training ────────────────────────────────────────────────────────────────

def _augment(x: torch.Tensor, c: torch.Tensor):
    """x (B, H, W) float 0..255, c (B, 2) input px -> augmented copies."""
    B, dev = x.shape[0], x.device
    xf = x.unsqueeze(1)

    def u(lo, hi, *shape):
        return torch.empty(*(shape or (B,)), device=dev).uniform_(lo, hi)

    a, b, g = u(0.6, 1.6, B, 1, 1, 1), u(-15, 25, B, 1, 1, 1), u(0.7, 1.5, B, 1, 1, 1)
    xf = ((xf / 255.0).clamp(0, 1) ** g) * 255.0 * a + b \
        + torch.randn_like(xf) * u(0, 6, B, 1, 1, 1)
    s = u(0.85, 1.15)
    fx = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    fy = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    ax, ay, tx, ty = fx / s, fy / s, u(-0.25, 0.25), u(-0.25, 0.25)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0], th[:, 0, 2], th[:, 1, 1], th[:, 1, 2] = ax, tx, ay, ty
    xf = F.grid_sample(xf, F.affine_grid(th, xf.shape, align_corners=False), mode="bilinear",
                       padding_mode="reflection", align_corners=False)
    un = torch.stack([2 * c[:, 0] / IN_W - 1, 2 * c[:, 1] / IN_H - 1], 1)
    o = torch.stack([(un[:, 0] - tx) / ax, (un[:, 1] - ty) / ay], 1)
    return xf[:, 0].clamp(0, 255), torch.stack([(o[:, 0] + 1) * IN_W / 2, (o[:, 1] + 1) * IN_H / 2], 1)


def _gauss_target(c: torch.Tensor, h: int, w: int, sigma: float = 1.5) -> torch.Tensor:
    gx = (torch.arange(w, device=c.device).float() + 0.5) * STRIDE
    gy = (torch.arange(h, device=c.device).float() + 0.5) * STRIDE
    d2 = (gx[None, None, :] - c[:, 0, None, None]) ** 2 + (gy[None, :, None] - c[:, 1, None, None]) ** 2
    t = torch.exp(-d2 / (2 * (sigma * STRIDE) ** 2))
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-8)


class _Data:
    """All kept frames of a set of clips, in network-input px."""

    def __init__(self, clips: list[dict], rows: list[np.ndarray]):
        xs, cs, cid, half = [], [], [], []
        for k, (clip, keep) in enumerate(zip(clips, rows)):
            if not keep.any():
                continue
            h, w = clip["native_shape"]
            scale = np.array([IN_W / w, IN_H / h], np.float32)
            xs.append(clip["x"][keep])
            cs.append(clip["center"][keep] * scale)
            cid.append(np.full(int(keep.sum()), len(half)))
            # Paste window for synthetic negatives: big enough to cover the
            # animal (80 px half-width for the bundled rig's ~48 px mice).
            body = np.sqrt(np.median(clip["area"][keep]) * scale[0] * scale[1])
            half.append(int(np.clip(1.7 * body, 8, 120)))
        self.n = sum(len(x) for x in xs)
        if not self.n:
            self.x = torch.zeros((0, IN_H, IN_W), dtype=torch.uint8)
            self.c = torch.zeros((0, 2))
            self.clip = torch.zeros(0, dtype=torch.long)
            self.half = []
            self.by_clip = []
            return
        self.x = torch.from_numpy(np.concatenate(xs))
        self.c = torch.from_numpy(np.concatenate(cs))
        self.clip = torch.from_numpy(np.concatenate(cid))
        self.half = half
        self.by_clip = [torch.nonzero(self.clip == k).flatten() for k in range(len(half))]

    def synth_negative(self, i: int, rng: np.random.Generator) -> Optional[torch.Tensor]:
        """Frame i with the animal's window replaced by the same window from a
        frame of the same clip where the animal is far away."""
        k = int(self.clip[i])
        half, pool = self.half[k], self.by_clip[k]
        cand = pool[torch.from_numpy(rng.integers(0, len(pool), 16))]
        far = cand[(self.c[cand] - self.c[i]).norm(dim=1) > 1.5 * half]
        if len(far) == 0:
            return None
        j = int(far[0])
        cx, cy = int(self.c[i, 0]), int(self.c[i, 1])
        x0, x1, y0, y1 = max(0, cx - half), min(IN_W, cx + half), max(0, cy - half), min(IN_H, cy + half)
        img = self.x[i].clone()
        img[y0:y1, x0:x1] = self.x[j, y0:y1, x0:x1]
        return img


@torch.no_grad()
def evaluate(model: Localizer, data: _Data, negatives: torch.Tensor, device,
             native_scale: np.ndarray, batch: int = 128) -> dict:
    """Centre error (native px) on held-out positives; presence on positives
    and on negatives (synthetic + empty-arena)."""
    model.eval()
    errs, pres = [], []
    for i in range(0, data.n, batch):
        x = data.x[i:i + batch].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            xy, _lg, pl = model(x)
        d = (xy.float().cpu() - data.c[i:i + batch]).numpy() * native_scale[data.clip[i:i + batch].numpy()]
        errs.append(np.hypot(d[:, 0], d[:, 1]))
        pres.append(torch.sigmoid(pl.float()).cpu().numpy())
    neg_p = np.zeros(0)
    if len(negatives):
        ps = []
        for i in range(0, len(negatives), batch):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _xy, _lg, pl = model(negatives[i:i + batch].to(device))
            ps.append(torch.sigmoid(pl.float()).cpu().numpy())
        neg_p = np.concatenate(ps)
    model.train()
    e = np.concatenate(errs) if errs else np.zeros(0)
    p = np.concatenate(pres) if pres else np.zeros(0)

    def stat(f, a):
        return float(f(a)) if len(a) else None

    return {
        "frames": int(len(e)),
        "median_error_px": stat(np.median, e),
        "p95_error_px": stat(lambda a: np.percentile(a, 95), e),
        "presence_on_animal": stat(np.mean, p),
        "negatives": int(len(neg_p)),
        "presence_on_empty": stat(np.mean, neg_p),
    }


def split_clips(clips: list[dict], keeps: list[np.ndarray], val_every: int = 4):
    """Hold out whole clips when there are enough (every ``val_every``-th, at
    least one); otherwise the last 20% of each clip. -> (train rows, val rows)."""
    train = [k.copy() for k in keeps]
    val = [np.zeros_like(k) for k in keeps]
    usable = [i for i, k in enumerate(keeps) if k.any()]
    if len(usable) >= 3:
        held = usable[val_every - 1::val_every] or usable[-1:]
        for i in held:
            val[i], train[i] = keeps[i].copy(), np.zeros_like(keeps[i])
    else:
        for i in usable:
            cut = int(len(keeps[i]) * 0.8)
            val[i][cut:] = keeps[i][cut:]
            train[i][cut:] = False
    return train, val


def train_localizer(clips: list[dict], out_path: str, *, negatives: Optional[np.ndarray] = None,
                    init_weights: Optional[str] = None, steps: int = 4000, batch_size: int = 64,
                    lr: Optional[float] = None, device: Optional[str] = None, seed: int = 0,
                    eval_every: Optional[int] = None,
                    on_progress: Optional[Callable[[int, int, str], None]] = None,
                    cancel: Cancel = None) -> dict:
    """Distil SAM2-labelled ``clips`` into a localizer saved at ``out_path``.

    ``init_weights``: fine-tune from these (e.g. the bundled localizer);
    None = ImageNet ResNet18. Held-out centre error is checked every
    ``eval_every`` steps and the best checkpoint is the one saved; when
    fine-tuning, the starting weights count as step 0, so a run on too little
    data can't hand back something worse than it started from (it reports
    ``improved: False`` instead). -> metrics, before and after, on the
    held-out frames (which also pick the checkpoint, so "after" is a little
    optimistic)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    keeps = [qc_mask(c) for c in clips]
    odd = odd_clips(clips, keeps)
    for i in odd:
        keeps[i] = np.zeros_like(keeps[i])
    tr_rows, va_rows = split_clips(clips, keeps)
    train, val = _Data(clips, tr_rows), _Data(clips, va_rows)
    if train.n == 0:
        raise ValueError("no usable SAM2-labelled frames to train on (every clip lost the animal?)")
    native_scale = np.array([[c["native_shape"][1] / IN_W, c["native_shape"][0] / IN_H]
                             for c, k in zip(clips, va_rows) if k.any()], np.float32)
    neg = torch.from_numpy(negatives if negatives is not None and len(negatives)
                           else np.zeros((0, IN_H, IN_W), np.uint8))
    ncut = int(len(neg) * 0.8)
    neg_tr, neg_va = neg[:ncut], neg[ncut:]
    # Held-out synthetic negatives, so presence is checked even without
    # empty-arena videos.
    synth_va = [s for s in (val.synth_negative(i, rng) for i in
                            rng.permutation(val.n)[:200]) if s is not None] if val.n else []
    neg_eval = torch.cat([neg_va] + ([torch.stack(synth_va)] if synth_va else []))

    model = Localizer(pretrained=init_weights is None)
    if init_weights:
        model.load_state_dict(torch.load(init_weights, map_location="cpu", weights_only=True))
    model = model.to(device).train()
    before = evaluate(model, val, neg_eval, device, native_scale) if init_weights and val.n else None

    def score(m):
        return m["median_error_px"] if m and m["median_error_px"] is not None else float("inf")

    eval_every = eval_every or max(50, steps // 10)
    best = (score(before), 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}) \
        if before else (float("inf"), 0, None)
    curve = [(0, before["median_error_px"])] if before else []
    lr = lr or (2e-4 if init_weights else 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    zero = torch.zeros(2)
    t0, run_loss = time.time(), 0.0
    for step in range(steps):
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
        imgs, cs, present = [], [], []
        for i in rng.integers(0, train.n, batch_size):
            r = rng.random()
            if r < 0.05 and len(neg_tr):
                imgs.append(neg_tr[rng.integers(len(neg_tr))]); cs.append(zero); present.append(0.0)
                continue
            if r < 0.25:
                s = train.synth_negative(int(i), rng)
                if s is not None:
                    imgs.append(s); cs.append(zero); present.append(0.0)
                    continue
            imgs.append(train.x[int(i)]); cs.append(train.c[int(i)]); present.append(1.0)
        x = torch.stack(imgs).to(device, non_blocking=True).float()
        c = torch.stack(cs).to(device)
        pres = torch.tensor(present, device=device)
        x, c = _augment(x, c)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            xy, lg, pl = model(x)
        lg, pl = lg.float(), pl.float()
        tgt = _gauss_target(c, lg.shape[1], lg.shape[2])
        ce = -(tgt.flatten(1) * F.log_softmax(lg.flatten(1), 1)).sum(1)
        l1 = (xy.float() - c).abs().mean(1) / STRIDE
        loss = ((ce + 0.5 * l1) * pres).sum() / pres.sum().clamp_min(1) \
            + F.binary_cross_entropy_with_logits(pl, pres)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.item()
        if on_progress and ((step + 1) % 50 == 0 or step + 1 == steps):
            n = 50 if (step + 1) % 50 == 0 else (step + 1) % 50
            on_progress(step + 1, steps, f"step {step + 1}/{steps} loss {run_loss / n:.4f} "
                                         f"({time.time() - t0:.0f}s)")
            run_loss = 0.0
        if val.n and ((step + 1) % eval_every == 0 or step + 1 == steps):
            m = evaluate(model, val, neg_eval, device, native_scale)
            curve.append((step + 1, m["median_error_px"]))
            if score(m) < best[0]:
                best = (score(m), step + 1, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()})

    if best[2] is not None:
        model.load_state_dict(best[2])
    after = evaluate(model, val, neg_eval, device, native_scale) if val.n else None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(model.cpu().state_dict(), tmp)
    os.replace(tmp, out_path)
    return {
        "weights": out_path,
        "train_frames": train.n,
        "val_frames": val.n,
        "dropped_frames": int(sum(len(k) - k.sum() for k in keeps)),
        "dropped_clips": odd,
        "steps": steps,
        "best_step": best[1] if val.n else steps,
        "improved": (best[1] > 0) if before else None,
        "curve": curve,
        "init": init_weights or "imagenet",
        "before": before,
        "after": after,
        "seconds": round(time.time() - t0, 1),
        "_model": model,  # popped by the caller; kept for the QC sheet
    }


@torch.no_grad()
def qc_sheet(model: Localizer, clips: list[dict], path: str, n: int = 12, seed: int = 0) -> str:
    """A grid of labelled frames: SAM2's centre (green) vs the trained
    localizer's (magenta). The quickest way to see whether the teacher
    tracked the right thing."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    picks = []
    for clip in clips:
        ok = np.nonzero(qc_mask(clip))[0]
        if len(ok):
            picks += [(clip, int(i)) for i in rng.choice(ok, min(len(ok), max(1, n // len(clips))), replace=False)]
    picks = picks[:n]
    if not picks:
        return ""
    model.eval()
    tiles = []
    for clip, i in picks:
        img = cv2.cvtColor(clip["x"][i], cv2.COLOR_GRAY2BGR)
        h, w = clip["native_shape"]
        teacher = clip["center"][i] * np.array([IN_W / w, IN_H / h])
        xy, _lg, _pl = model(torch.from_numpy(clip["x"][i][None]).to(device))
        cv2.circle(img, tuple(int(v) for v in teacher), 6, (0, 255, 0), 2)
        cv2.drawMarker(img, tuple(int(v) for v in xy[0].float().cpu().numpy()), (255, 0, 255),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"{os.path.basename(str(clip['video']))[:28]} #{int(clip['index'][i])}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def write_card(path: str, info: dict) -> None:
    """The JSON sidecar next to a trained localizer: where it came from and how
    well it did on held-out frames."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

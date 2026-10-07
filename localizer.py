"""SAM2-distilled animal localizer: a fast learned alternative to background
subtraction for finding the crop centre.

The network was distilled from SAM2 body masks (tail eroded): SAM2 itself runs
at ~3 fps, this runs at hundreds. It takes the whole frame downscaled to
352x384 gray and returns the body centre (soft-argmax of a stride-8 heatmap)
plus an "animal present" logit, so an empty arena or a hand in the cage
doesn't drag the box away.

ResNet18 (ImageNet) through layer3 with a layer2 skip -> heatmap head; a small
MLP on pooled features -> presence. Weights are a plain state dict
(``torch.save(model.state_dict())``), compatible with the label_video
pipeline's ``Localizer2`` checkpoints. ``DEFAULT_WEIGHTS`` (bundled, see
weights/localizer/README.md) is used when no path is given; train one for a new
rig with localizer_train.py (Pipeline -> Train Localizer in the app).

``LocalizerCropper`` turns centres into fixed-size crops with the same
stabilisation rules as ``tune_cropping.FindRatBoundingBox`` (EMA ``smoothing``
+ ``deadband``), and holds the box still on frames where the animal isn't
present.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_H, IN_W = 352, 384  # network input (rows, cols); any source resolution is resized to this
STRIDE = 8
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weights", "localizer", "mouse_topdown.pt")
# Below this presence probability the box holds still. 0.1, not 0.5: the
# bundled network's presence drops to ~0.2-0.4 for an animal at the arena edge
# or half hidden, and those frames still have a usable centre.
PRESENCE_THRESHOLD = 0.1


def resolve_weights(path: str | None) -> str:
    """``path``, or the bundled default when none is given."""
    path = path or DEFAULT_WEIGHTS
    if not os.path.exists(path):
        raise FileNotFoundError(f"localizer weights not found: {path}")
    return path


class Localizer(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.lat3 = nn.Conv2d(256, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(True), nn.Conv2d(64, 1, 1))
        self.pres = nn.Sequential(nn.Linear(256 * 2 + 128, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_u8: torch.Tensor):
        """x_u8: (B, IN_H, IN_W) uint8 gray.
        -> (xy (B, 2) in input-image px, heatmap logits (B, IN_H/8, IN_W/8), presence logit (B,))"""
        x = (x_u8.float() / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        f2 = self.l2(self.l1(self.stem(x)))  # stride 8, 128 ch
        f3 = self.l3(f2)                      # stride 16, 256 ch
        f = f2 + F.interpolate(self.lat3(f3), size=f2.shape[-2:], mode="bilinear",
                               align_corners=False)
        lg = self.head(f)[:, 0]
        B, h, w = lg.shape
        p = F.softmax(lg.flatten(1), 1).view(B, h, w)
        gx = (torch.arange(w, device=lg.device).float() + 0.5) * STRIDE
        gy = (torch.arange(h, device=lg.device).float() + 0.5) * STRIDE
        xy = torch.stack([(p.sum(1) * gx).sum(1), (p.sum(2) * gy).sum(1)], 1)
        feat = torch.cat([f3.mean((2, 3)), f3.amax((2, 3)), f.amax((2, 3))], 1)
        return xy, lg, self.pres(feat)[:, 0]


_CACHE: dict = {}


def load_localizer(path: str, device: str | torch.device | None = None) -> Localizer:
    """Load weights onto ``device``. Cached per (file, mtime, device): the
    preview endpoint asks for the same model on every frame step, and an eval
    model is stateless so sharing one between requests is safe."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (os.path.abspath(path), os.path.getmtime(path), str(device))
    if key not in _CACHE:
        model = Localizer(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _CACHE.clear()  # keep one; switching weights shouldn't pile up GPU memory
        _CACHE[key] = model.to(device).eval()
    return _CACHE[key]


def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def network_input(frame: np.ndarray) -> np.ndarray:
    """Native frame (gray or BGR) -> the (IN_H, IN_W) uint8 the network sees."""
    return cv2.resize(to_gray(frame), (IN_W, IN_H), interpolation=cv2.INTER_AREA)


class LocalizerCropper:
    """Locate the animal in batches, then crop a ``box_size`` square around the
    stabilised centre.

    Centres come out of :meth:`locate` in native-frame pixels. :meth:`crop`
    works in whatever frame it is handed (e.g. one pre-resized by
    ``resize_factor``), scaling the centre to that frame's size, so
    ``box_size``/``deadband`` mean the same thing as in background mode: pixels
    of the frame being cropped.
    """

    def __init__(self, weights_path: str, box_size: int = 220, smoothing: float = 0.5,
                 deadband: float = 0.0, presence_threshold: float = PRESENCE_THRESHOLD,
                 device: str | None = None, model: Localizer | None = None,
                 workers: int = 8, runtime: str = "torch", precision: str = "fp16",
                 max_batch: int = 32, on_status=None):
        """``runtime``: "torch", "tensorrt" or "auto" (TensorRT when usable;
        the engine is built and cached on first use, see trt_engines.py)."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = weights_path if model is not None else resolve_weights(weights_path)
        self.engine = None
        if model is None and runtime != "torch":
            import trt_engines
            rt, reason = trt_engines.resolve_runtime(runtime)
            if rt == "tensorrt":
                try:
                    self.engine = trt_engines.localizer_engine(
                        self.weights_path, precision=precision, max_batch=max_batch,
                        on_status=on_status)
                    self.device = torch.device("cuda")
                    reason = f"TensorRT {precision} ({os.path.basename(self.engine.path)})"
                except Exception as e:
                    if runtime == "tensorrt":
                        raise
                    reason = f"PyTorch (TensorRT engine build failed: {type(e).__name__}: {e})"
            print(f"[localizer] runtime: {reason}", flush=True)
        self._model = model
        self.box_size = int(box_size)
        self.smoothing = max(0.0, min(1.0, float(smoothing)))
        self.deadband = float(deadband)
        self.presence_threshold = float(presence_threshold)
        self.smoothed_center = None  # (cx, cy) in cropped-frame px
        self._pool = ThreadPoolExecutor(workers)

    @property
    def model(self) -> Localizer:
        """The PyTorch network — loaded lazily, so a TensorRT run never pays
        for it unless something asks for the heatmap."""
        if self._model is None:
            self._model = load_localizer(self.weights_path, self.device)
        return self._model

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def reset(self) -> None:
        self.smoothed_center = None

    def prepare(self, frames: list) -> list:
        """Start downscaling ``frames`` to the network input on the thread pool
        (the most expensive CPU step, ~4 ms for a 2 MP frame). Pass the result
        to :meth:`locate` later, so it overlaps whatever runs in between."""
        return [self._pool.submit(network_input, f) for f in frames]

    @torch.no_grad()
    def locate(self, frames: list, return_heatmap: bool = False, prepared: list | None = None):
        """Native frames -> (centers (N, 2) native px, presence prob (N,)[, heatmap logits])."""
        if not frames:
            empty = (np.zeros((0, 2), np.float32), np.zeros(0, np.float32))
            return empty + (np.zeros((0, IN_H // STRIDE, IN_W // STRIDE), np.float32),) \
                if return_heatmap else empty
        small = [f.result() for f in (prepared if prepared is not None else self.prepare(frames))]
        x = torch.from_numpy(np.stack(small)).to(self.device)
        if self.engine is not None and not return_heatmap:
            xy, pl = self.engine(x)
            lg = None
        else:
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                xy, lg, pl = self.model(x)
        xy = xy.float().cpu().numpy()
        scale = np.array([[f.shape[1] / IN_W, f.shape[0] / IN_H] for f in frames], np.float32)
        centers = xy * scale
        pres = torch.sigmoid(pl.float()).cpu().numpy()
        if return_heatmap:
            return centers, pres, lg.float().cpu().numpy()
        return centers, pres

    def _update_center(self, raw_cx: float, raw_cy: float, present: bool):
        """EMA + dead-band, same rule as FindRatBoundingBox, plus: when the
        animal isn't present, hold the previous centre."""
        if self.smoothed_center is None:
            self.smoothed_center = (raw_cx, raw_cy)
            return self.smoothed_center
        if not present:
            return self.smoothed_center
        px, py = self.smoothed_center
        if self.deadband > 0 and ((raw_cx - px) ** 2 + (raw_cy - py) ** 2) ** 0.5 <= self.deadband:
            return self.smoothed_center
        a = self.smoothing
        self.smoothed_center = ((1 - a) * raw_cx + a * px, (1 - a) * raw_cy + a * py)
        return self.smoothed_center

    def crop(self, frame: np.ndarray, center_native, presence: float, native_shape):
        """Crop ``frame`` (already resized, if resizing) around the stabilised
        centre. -> (crop, (x0, y0, box, box) in ``frame`` px)."""
        sx = frame.shape[1] / native_shape[1]
        sy = frame.shape[0] / native_shape[0]
        cx, cy = self._update_center(float(center_native[0]) * sx, float(center_native[1]) * sy,
                                     presence >= self.presence_threshold)
        box = self.box_size
        x0 = int(max(0, min(round(cx) - box // 2, frame.shape[1] - box)))
        y0 = int(max(0, min(round(cy) - box // 2, frame.shape[0] - box)))
        return frame[y0:y0 + box, x0:x0 + box].copy(), (x0, y0, box, box)

"""TensorRT engines for video inference: the localizer and the TSM classifier,
built automatically on first use and cached.

    loc = localizer_engine(weights_path)        # (B, 352, 384) uint8 cuda -> (xy (B,2), presence logit (B,)) f32
    tsm = tsm_engine(model, ckpt_path, T, S)    # (B, T, 3, S, S) f32 cuda  -> logits (B, K) f32

An engine only runs on the GPU architecture and TensorRT version it was built
with, and only computes what the weights said at build time, so the cache key
encodes all of it (weights by content hash, plus every shape/config knob).
Retraining, a new GPU or a TensorRT upgrade builds a new engine (~0.5-2 min,
once) instead of loading a stale or incompatible one. Builds are atomic, so a
killed build never leaves a half-written engine behind.

Precision: "fp16" runs the conv trunks in fp16 with everything numerically
delicate (normalisation, the soft-argmax softmax, pooling, heads) in fp32 —
the split autocast makes. It's baked into the exported graph because
TensorRT 11 builds strongly typed networks only (no FP16 builder flag).

Cache location: $BREAL_TRT_CACHE, else the OS cache dir
(~/.cache/b-real/trt, %LOCALAPPDATA%/B-REAL/trt).

Ported from the label_video pipeline (video_processing/trt_engines.py).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import types

import torch
import torch.nn as nn

RUNTIMES = ("auto", "tensorrt", "torch")
PRECISIONS = ("fp16", "fp32")
WORKSPACE_GB = 2.0
_LOGGER = None
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()
_DIGESTS: dict = {}


def cache_dir() -> str:
    env = os.environ.get("BREAL_TRT_CACHE")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "B-REAL", "trt")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "b-real", "trt")


def _logger():
    """One process-wide TRT logger (TRT warns about and ignores a second one)."""
    global _LOGGER
    if _LOGGER is None:
        import tensorrt as trt
        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


def tensorrt_available() -> tuple[bool, str]:
    """-> (usable, reason). Usable = tensorrt importable + a GPU it supports."""
    try:
        from optimized_runtime import _require, check_tensorrt_gpu
        trt = _require("tensorrt")
        check_tensorrt_gpu()
        return True, f"TensorRT {trt.__version__} on {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def resolve_runtime(runtime: str) -> tuple[str, str]:
    """"auto" -> "tensorrt" when usable, else "torch". -> (runtime, reason).
    An explicit "tensorrt" that isn't usable raises."""
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {RUNTIMES}, got {runtime!r}")
    if runtime == "torch":
        return "torch", "PyTorch (requested)"
    ok, reason = tensorrt_available()
    if ok:
        return "tensorrt", reason
    if runtime == "tensorrt":
        raise RuntimeError(f"TensorRT requested but unusable: {reason}")
    return "torch", f"PyTorch (TensorRT unusable: {reason})"


# ── runtime ─────────────────────────────────────────────────────────────────

class TRTModule:
    """Executes a serialized engine on torch CUDA tensors.

    Runs on its own stream, fenced both ways against torch's current stream,
    so callers see ordinary stream semantics. Inputs larger than the engine's
    max batch are split into chunks. One instance per thread: a TensorRT
    execution context isn't thread-safe.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(_logger()).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path} (TensorRT version / GPU mismatch?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode.INPUT
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == mode]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) != mode]
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.uint8: torch.uint8,
              trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        self.dtypes = {n: dt[self.engine.get_tensor_dtype(n)] for n in names}
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.inputs[0], 0)[2][0])
        self.path = engine_path
        self.stream = torch.cuda.Stream()
        meta = engine_path[:-len(".engine")] + ".json"
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}

    def _run(self, xs):
        for name, x in zip(self.inputs, xs):
            if x.dtype != self.dtypes[name] or not x.is_cuda:
                raise TypeError(f"input {name!r}: engine wants cuda {self.dtypes[name]}, "
                                f"got {x.device} {x.dtype}")
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, x.data_ptr())
        outs = []
        for name in self.outputs:
            out = torch.empty(tuple(self.context.get_tensor_shape(name)),
                              dtype=self.dtypes[name], device=xs[0].device)
            self.context.set_tensor_address(name, out.data_ptr())
            outs.append(out)
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.path}")
        cur.wait_stream(self.stream)
        return outs

    def __call__(self, *xs):
        if len(xs) != len(self.inputs):
            raise ValueError(f"{os.path.basename(self.path)} takes {len(self.inputs)} inputs, got {len(xs)}")
        xs = [x.contiguous() for x in xs]
        n = xs[0].shape[0]
        chunks = [self._run([x[i:i + self.max_batch] for x in xs])
                  for i in range(0, n, self.max_batch)]
        outs = [torch.cat(parts) if len(chunks) > 1 else parts[0] for parts in zip(*chunks)]
        return outs[0] if len(outs) == 1 else tuple(outs)


# ── build / cache ───────────────────────────────────────────────────────────

def _digest(path: str, n: int = 12) -> str:
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _DIGESTS[key] = h.hexdigest()[:n]
    return _DIGESTS[key]


def engine_path(kind: str, weights_path: str, precision: str, max_batch: int,
                config: dict | None = None) -> str:
    import tensorrt as trt
    major, minor = torch.cuda.get_device_capability()
    cfg = hashlib.sha256(json.dumps(config or {}, sort_keys=True, default=str).encode()).hexdigest()[:8]
    tag = (f"{kind}_{_digest(weights_path)}_{cfg}_{precision}_b{max_batch}"
           f"_sm{major}{minor}_trt{trt.__version__}")
    return os.path.join(cache_dir(), tag + ".engine")


def _export_onnx(module, example, onnx_path, input_name, output_names):
    axes = {input_name: {0: "batch"}, **{o: {0: "batch"} for o in output_names}}
    with torch.no_grad():
        torch.onnx.export(module, (example,), onnx_path, input_names=[input_name],
                          output_names=output_names, dynamic_axes=axes,
                          opset_version=17, dynamo=False)


def get_engine(kind, weights_path, precision, max_batch, config, make_export,
               on_status=None) -> TRTModule:
    """Load the cached engine, building it first if needed.
    ``make_export()`` -> (module, example_input, input_name, output_names, meta)."""
    from optimized_runtime import build_tensorrt
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    path = engine_path(kind, weights_path, precision, max_batch, config)
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.setdefault(path, threading.Lock())
    with lock:  # one build per engine even if two tasks ask at once
        if not os.path.exists(path):
            os.makedirs(cache_dir(), exist_ok=True)
            msg = (f"Building the {kind} TensorRT engine ({precision}, batch ≤{max_batch}) — "
                   f"one-time, ~0.5-2 min")
            print(f"[trt] {msg}: {os.path.basename(path)}", flush=True)
            if on_status:
                on_status(msg)
            t0 = time.time()
            module, example, in_name, out_names, meta = make_export()
            onnx_path = path[:-len(".engine")] + ".onnx"
            _export_onnx(module, example, onnx_path, in_name, out_names)
            try:
                build_tensorrt(onnx_path, path, precision=precision,
                               workspace_gb=WORKSPACE_GB, max_batch=max_batch)
            finally:
                os.remove(onnx_path)  # regenerable from the weights; engines are what's reused
            with open(path[:-len(".engine")] + ".json", "w") as f:
                json.dump(dict(meta, weights=os.path.abspath(weights_path), precision=precision,
                               max_batch=max_batch, config=config,
                               gpu=torch.cuda.get_device_name(0)), f, indent=1, default=str)
            print(f"[trt] built {os.path.basename(path)} in {time.time() - t0:.0f}s", flush=True)
    return TRTModule(path)


class _Cast(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


# ── localizer ───────────────────────────────────────────────────────────────

class _LocalizerExport(nn.Module):
    """Localizer minus the heatmap output (cropping only needs xy + presence)."""

    def __init__(self, loc):
        super().__init__()
        self.loc = loc

    def forward(self, x_u8):
        xy, _, pres = self.loc(x_u8)
        return xy, pres


def _half_localizer(loc):
    """fp16 conv trunk; normalisation, soft-argmax and the presence MLP in fp32.
    Localizer.forward itself is unchanged: the casts live in the submodules."""
    loc.stem = nn.Sequential(_Cast(torch.float16), loc.stem.half())
    for name in ("l1", "l2", "l3", "lat3"):
        getattr(loc, name).half()
    loc.head = nn.Sequential(loc.head.half(), _Cast(torch.float32))
    loc.pres = nn.Sequential(_Cast(torch.float32), loc.pres)
    return loc


def localizer_engine(weights_path: str, *, precision: str = "fp16", max_batch: int = 32,
                     on_status=None) -> TRTModule:
    from localizer import IN_H, IN_W, Localizer

    def make():
        loc = Localizer(pretrained=False)
        loc.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        loc = loc.cuda().eval()
        if precision == "fp16":
            loc = _half_localizer(loc)
        example = torch.randint(0, 256, (2, IN_H, IN_W), dtype=torch.uint8, device="cuda")
        return (_LocalizerExport(loc).eval(), example, "frames_u8", ["xy", "presence_logit"],
                dict(model="Localizer", input=[IN_H, IN_W]))

    return get_engine("localizer", weights_path, precision, max_batch,
                      {"in": [IN_H, IN_W]}, make, on_status)


# ── TSM classifier ──────────────────────────────────────────────────────────

def _temporal_shift_concat(self, x):
    """Export-friendly TemporalShift.forward: the same values as model_tsm's
    clone + in-place slice assignment, built from slices + concat so ONNX gets
    Slice/Concat instead of ScatterND. Checked equal before every export."""
    BT, C, H, W = x.shape
    T = self.num_frames
    x = x.view(-1, T, C, H, W)
    s = int(C * self.shift_fraction)
    if self.unidirectional:
        n = min(2 * s, C)
        past = torch.cat([torch.zeros_like(x[:, :1, :n]), x[:, :-1, :n]], 1)
        out = torch.cat([past, x[:, :, n:]], 2)
    else:
        z = torch.zeros_like(x[:, :1, :s])
        fwd = torch.cat([z, x[:, :-1, :s]], 1)        # frame t <- t-1
        bwd = torch.cat([x[:, 1:, s:2 * s], z], 1)    # frame t <- t+1
        out = torch.cat([fwd, bwd, x[:, :, 2 * s:]], 2)
    return out.reshape(-1, C, H, W)


def tsm_engine(model, weights_path: str, num_frames: int, image_size: int, *,
               precision: str = "fp16", max_batch: int = 32, model_config: dict | None = None,
               on_status=None) -> TRTModule:
    """``model``: the loaded TSMClassifier (any device; it is copied, not
    modified). ``weights_path`` identifies it in the cache; ``model_config``
    is whatever load_from_checkpoint overrides were applied (also keyed)."""
    from model_tsm import TemporalShift
    from optimized_runtime import half_features

    def make():
        m = copy.deepcopy(model).float().cuda().eval()
        x = torch.randn(2, num_frames, 3, image_size, image_size, device="cuda")
        with torch.no_grad():
            ref = m(x)
            shifts = [mod for mod in m.modules() if isinstance(mod, TemporalShift)]
            for mod in shifts:  # this copy only, never the class
                mod.forward = types.MethodType(_temporal_shift_concat, mod)
            got = m(x)
        if shifts and not torch.equal(ref, got):
            raise RuntimeError(f"export-friendly TemporalShift disagrees with the original "
                               f"(max |d|={(ref - got).abs().max():.3g}); refusing to export")
        if precision == "fp16":
            m = half_features(m)
        return (m, x, "clips", ["logits"],
                dict(model="TSMClassifier", class_names=list(model.class_names),
                     temporal_pool=model.hparams.get("temporal_pool", "last")))

    config = {"T": num_frames, "S": image_size, "overrides": model_config or {}}
    return get_engine("tsm", weights_path, precision, max_batch, config, make, on_status)

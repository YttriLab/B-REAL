"""Deployment-format export and benchmarking: ONNX -> OpenVINO (CPU) / TensorRT (GPU).

TorchScript is convenient but it is not what runs on a rig. A frame budget checked
against TorchScript numbers is checked against the wrong thing: OpenVINO and
TensorRT reorder, fuse and re-precision the graph, and the gap is large enough to
flip a "fits / doesn't fit" verdict. So the tuning sweep's latency gate and the
Export tab both go through here.

Both backends consume the same ONNX intermediate:

    checkpoint / architecture
              |
        export_onnx()  ->  model.onnx
              |                     |
      build_openvino()        build_tensorrt()
       (CPU, OpenVINO IR)      (GPU, .engine)

Every import of onnx / openvino / tensorrt is lazy and raises RuntimeError with an
install hint. Callers are expected to surface that rather than silently falling
back — a latency number from the wrong runtime is worse than no number.
"""

from __future__ import annotations

import math
import os
import time

import torch

from convert_torchscript_tsm import TSMInferenceWrapper
from model_tsm import TSMClassifier

# Deployment targets. "auto" resolves to tensorrt when CUDA is present.
TARGETS = ("auto", "openvino", "tensorrt", "torchscript")

# The pixi env has no pip, so `pixi run pip install X` silently hits whatever pip
# is on PATH (usually a system Python) and the package lands in the wrong
# interpreter. Everything is declared in pixi.toml and installed by `pixi install`.
_INSTALL_HINTS = {
    "onnx": "it is declared in pixi.toml — run `pixi install`",
    "openvino": "it is declared in pixi.toml — run `pixi install`",
    "tensorrt": ("it is declared in pixi.toml as the pypi package tensorrt-cu12 — "
                 "run `pixi install`. Needs a GPU of compute capability 7.5+"),
}


class RuntimeUnavailable(RuntimeError):
    """The requested deployment runtime isn't installed or can't be used here."""


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise RuntimeUnavailable(
            f"{module} is not installed, so latency can't be measured in the "
            f"deployment format. {_INSTALL_HINTS.get(module, f'pip install {module}')}"
        ) from e


# TensorRT 10+ dropped Pascal: the builder asserts sm_75 <= arch, and the failure
# surfaces as an opaque "pybind11::init(): factory function returned nullptr".
# Check up front so the message names the actual problem.
_TRT_MIN_SM = (7, 5)


def resolve_target(target: str = "auto") -> str:
    """'auto' -> the best runtime this machine can actually serve.

    TensorRT when there's a CUDA GPU new enough for it, OpenVINO otherwise. An
    explicitly named target is never substituted — if you asked for TensorRT and
    this box can't run it, that's an error, not a silent downgrade to numbers
    from a different runtime.
    """
    if target and target != "auto":
        if target not in TARGETS:
            raise ValueError(f"Unknown target '{target}'. Choose from {TARGETS}.")
        return target
    if not torch.cuda.is_available():
        return "openvino"
    props = torch.cuda.get_device_properties(0)
    return "tensorrt" if (props.major, props.minor) >= _TRT_MIN_SM else "openvino"


def check_tensorrt_gpu():
    """Raise RuntimeUnavailable if the visible GPU is too old for TensorRT."""
    if not torch.cuda.is_available():
        raise RuntimeUnavailable(
            "TensorRT was requested but no CUDA device is visible. Pick the "
            "OpenVINO (CPU) target, or run the backend on the GPU box.")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) < _TRT_MIN_SM:
        raise RuntimeUnavailable(
            f"{props.name} is compute capability sm_{props.major}{props.minor}; "
            f"TensorRT 10+ needs sm_{_TRT_MIN_SM[0]}{_TRT_MIN_SM[1]} (Turing) or "
            f"newer, and the last TensorRT supporting Pascal (8.6) has no wheels "
            f"for this Python. Benchmark the CPU target with OpenVINO, or run the "
            f"sweep on a machine with the deployment GPU.")


def preflight_target(target: str = "auto") -> str:
    """Check a target is usable *before* committing to long work.

    Raises RuntimeUnavailable with an install hint otherwise, so a sweep refuses
    to start rather than discovering after the first trial that it can't measure
    its own frame budget.
    """
    resolved = resolve_target(target)
    if resolved == "torchscript":
        return resolved
    _require("onnx")
    if resolved == "openvino":
        _require("openvino")
    else:
        _require("tensorrt")
        check_tensorrt_gpu()
    return resolved


# ── Model construction ─────────────────────────────────────────────────────

def build_wrapper(
    checkpoint_path: str | None = None,
    *,
    variant: str = "b0",
    num_frames: int = 3,
    tsm_stages=None,
    temporal_pool: str = "last",
    unidirectional: bool = False,
    device: str = "cpu",
) -> TSMInferenceWrapper:
    """The inference-only module, either from a checkpoint or as an untrained
    architecture. Timing depends on the graph, not the weights, so the sweep's
    gate builds untrained models and skips the checkpoint load entirely."""
    if checkpoint_path:
        load_kwargs = {}
        if tsm_stages is not None:
            load_kwargs["tsm_stages"] = tsm_stages
        model = TSMClassifier.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False, **load_kwargs
        )
    else:
        model = TSMClassifier(
            variant=variant, num_frames=num_frames, tsm_stages=tsm_stages,
            temporal_pool=temporal_pool, unidirectional=unidirectional,
            pretrained=False, class_names=["a", "b"],
        )
    model.eval()
    wrapper = TSMInferenceWrapper(model).to(device).eval()
    return wrapper


# ── ONNX ───────────────────────────────────────────────────────────────────

def export_onnx(
    wrapper: TSMInferenceWrapper,
    output_path: str,
    *,
    num_frames: int = 3,
    image_size: int = 224,
    batch_size: int = 1,
    opset: int = 17,
) -> str:
    """Trace the wrapper to ONNX at a fixed input shape.

    Shapes are static on purpose: TensorRT builds a faster engine for a fixed
    profile, and a deployment rig scores a fixed clip shape anyway.
    """
    _require("onnx")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    dummy = torch.randn(batch_size, num_frames, 3, image_size, image_size,
                        device=next(wrapper.parameters()).device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_path,
            input_names=["clip"], output_names=["logits"],
            opset_version=opset, dynamo=False,
        )
    return output_path


# ── OpenVINO (CPU) ─────────────────────────────────────────────────────────

def build_openvino(onnx_path: str, *, precision: str = "fp32", output_dir=None):
    """Compile an ONNX graph for the CPU plugin, optionally saving the IR.

    Returns (compiled_model, ir_path|None).
    """
    ov = _require("openvino")
    core = ov.Core()
    model = core.read_model(onnx_path)

    config = {}
    if str(precision).lower() in ("fp16", "half", "16"):
        # The CPU plugin keeps fp32 accumulation; this permits bf16/fp16 kernels
        # where the hardware has them.
        config["INFERENCE_PRECISION_HINT"] = "f16"
    config["PERFORMANCE_HINT"] = "LATENCY"  # a rig scores one clip at a time

    compiled = core.compile_model(model, "CPU", config)

    ir_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(onnx_path))[0]
        ir_path = os.path.join(output_dir, f"{name}.xml")
        ov.save_model(model, ir_path, compress_to_fp16=("fp16" in str(precision).lower()))
    return compiled, ir_path


def _benchmark_openvino(onnx_path, *, precision, batch_size, num_frames,
                        image_size, iters, warmup):
    import numpy as np

    compiled, _ = build_openvino(onnx_path, precision=precision)
    request = compiled.create_infer_request()
    x = np.random.randn(batch_size, num_frames, 3, image_size, image_size).astype("float32")

    for _ in range(max(1, warmup)):
        request.infer({0: x})

    samples = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        request.infer({0: x})
        samples.append((time.perf_counter() - t0) * 1000.0)

    ov = _require("openvino")
    try:
        device_name = ov.Core().get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        device_name = "CPU"
    return samples, device_name


# ── TensorRT (GPU) ─────────────────────────────────────────────────────────

def build_tensorrt(onnx_path: str, engine_path: str, *, precision: str = "fp32",
                   workspace_gb: float = 4.0) -> str:
    """Build a serialized TensorRT engine from ONNX.

    This is the slow step (tens of seconds to minutes, depending on the variant)
    because TRT profiles kernels for the actual GPU. Callers cache the result.
    """
    trt = _require("tensorrt")
    check_tensorrt_gpu()
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
    except TypeError as e:
        # pybind11 surfaces a null builder as a TypeError with no useful text.
        raise RuntimeUnavailable(
            f"TensorRT could not create a builder on this GPU: {e}") from e
    # TRT 10+ dropped implicit-batch mode entirely, so EXPLICIT_BATCH no longer
    # exists on NetworkDefinitionCreationFlag — create_network() defaults to
    # explicit batch (flags=0) on those versions. Only pass the flag on older
    # TRT where it's still required.
    explicit_batch_flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(
        1 << int(explicit_batch_flag) if explicit_batch_flag is not None else 0
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeUnavailable(f"TensorRT could not parse the ONNX graph: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if str(precision).lower() in ("fp16", "half", "16"):
        if not builder.platform_has_fast_fp16:
            raise RuntimeUnavailable("This GPU has no fast FP16; benchmark in fp32.")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeUnavailable("TensorRT engine build failed (no engine produced).")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path


def _benchmark_tensorrt(onnx_path, engine_path, *, precision, batch_size,
                        num_frames, image_size, iters, warmup):
    trt = _require("tensorrt")
    build_tensorrt(onnx_path, engine_path, precision=precision)

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Bind torch tensors directly — they carry device pointers TRT can consume,
    # so there's no extra host copy in the timed region.
    dtype = torch.float16 if str(precision).lower() in ("fp16", "half", "16") else torch.float32
    inp = torch.randn(batch_size, num_frames, 3, image_size, image_size,
                      device="cuda", dtype=torch.float32)
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, tuple(inp.shape))
            context.set_tensor_address(name, inp.data_ptr())
        else:
            shape = tuple(context.get_tensor_shape(name))
            out = torch.empty(shape, device="cuda",
                              dtype=torch.float16 if dtype == torch.float16 else torch.float32)
            outputs[name] = out
            context.set_tensor_address(name, out.data_ptr())

    stream = torch.cuda.Stream()
    handle = stream.cuda_stream

    for _ in range(max(1, warmup)):
        context.execute_async_v3(handle)
    torch.cuda.synchronize()

    samples = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        context.execute_async_v3(handle)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    return samples, torch.cuda.get_device_name(0)


# ── Stats ──────────────────────────────────────────────────────────────────

def summarize(samples: list, *, batch_size: int) -> dict:
    """Mean + tail latency from per-iteration millisecond samples."""
    ordered = sorted(samples)

    def pct(p):
        k = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
        return ordered[k]

    total_s = sum(samples) / 1000.0
    clips = len(samples) * max(1, batch_size)
    per_clip = float(max(1, batch_size))
    p50, p95, p99 = pct(50), pct(95), pct(99)
    return {
        "iters": len(samples),
        "ms_per_batch": sum(samples) / len(samples),
        "ms_per_clip": total_s / clips * 1000.0,
        "clips_per_sec": clips / total_s,
        "ms_per_batch_p50": p50,
        "ms_per_batch_p95": p95,
        "ms_per_batch_p99": p99,
        "ms_per_clip_p50": p50 / per_clip,
        "ms_per_clip_p95": p95 / per_clip,
        "ms_per_clip_p99": p99 / per_clip,
    }


# ── Public entry points ────────────────────────────────────────────────────

def benchmark_optimized(
    *,
    target: str = "auto",
    checkpoint_path: str | None = None,
    variant: str = "b0",
    num_frames: int = 3,
    tsm_stages=None,
    temporal_pool: str = "last",
    unidirectional: bool = False,
    image_size: int = 224,
    batch_size: int = 1,
    precision: str = "fp32",
    iters: int = 100,
    warmup: int = 20,
    work_dir: str | None = None,
) -> dict:
    """Time an architecture in its deployment format.

    Raises RuntimeUnavailable when the runtime isn't installed or the export
    fails — never silently degrades to a different runtime, because the caller
    is making a deploy/don't-deploy decision with the number.
    """
    import tempfile

    resolved = resolve_target(target)

    if resolved == "torchscript":
        # Kept for comparison against the deployment formats, not as a fallback.
        from convert_torchscript_tsm import benchmark_model
        result = benchmark_model(
            variant=variant, num_frames=num_frames, tsm_stages=tsm_stages,
            temporal_pool=temporal_pool, unidirectional=unidirectional,
            image_size=image_size, batch_size=batch_size, precision=precision,
            iters=iters, warmup=warmup,
        )
        result["runtime"] = "torchscript"
        result["target"] = resolved
        return result

    owns_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="breal_opt_")
    try:
        # ONNX export runs on CPU: TRT consumes the graph, not the torch device.
        wrapper = build_wrapper(
            checkpoint_path, variant=variant, num_frames=num_frames,
            tsm_stages=tsm_stages, temporal_pool=temporal_pool,
            unidirectional=unidirectional, device="cpu",
        )
        onnx_path = export_onnx(
            wrapper, os.path.join(work_dir, "model.onnx"),
            num_frames=num_frames, image_size=image_size, batch_size=batch_size,
        )

        if resolved == "openvino":
            samples, device_name = _benchmark_openvino(
                onnx_path, precision=precision, batch_size=batch_size,
                num_frames=num_frames, image_size=image_size,
                iters=iters, warmup=warmup)
        else:
            samples, device_name = _benchmark_tensorrt(
                onnx_path, os.path.join(work_dir, "model.engine"),
                precision=precision, batch_size=batch_size, num_frames=num_frames,
                image_size=image_size, iters=iters, warmup=warmup)
    finally:
        if owns_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    result = summarize(samples, batch_size=batch_size)
    result.update({
        "runtime": resolved,
        "target": resolved,
        "device": "cuda" if resolved == "tensorrt" else "cpu",
        "device_name": device_name,
        "precision": "fp16" if str(precision).lower() in ("fp16", "half", "16") else "fp32",
        "variant": variant,
        "num_frames": num_frames,
        "tsm_stages": tsm_stages,
        "temporal_pool": temporal_pool,
        "unidirectional": unidirectional,
        "image_size": image_size,
        "batch_size": batch_size,
    })
    return result


def export_optimized(
    checkpoint_path: str,
    output_dir: str,
    output_name: str = "tsm_model",
    *,
    fmt: str = "onnx",
    tsm_stages=None,
    num_frames: int | None = None,
    image_size: int = 224,
    batch_size: int = 1,
    precision: str = "fp32",
) -> dict:
    """Export a checkpoint to a deployment format.

    fmt: onnx | openvino | tensorrt. Returns {"output_path", "format", "files"}.
    OpenVINO and TensorRT both go through ONNX, which is kept alongside the
    result so the rig can be re-targeted without re-exporting from the .ckpt.
    """
    if fmt not in ("onnx", "openvino", "tensorrt"):
        raise ValueError(f"Unknown export format '{fmt}'.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    wrapper = build_wrapper(checkpoint_path, tsm_stages=tsm_stages, device="cpu")
    frames = num_frames or wrapper.num_frames

    onnx_path = export_onnx(
        wrapper, os.path.join(output_dir, f"{output_name}.onnx"),
        num_frames=frames, image_size=image_size, batch_size=batch_size,
    )
    files = [onnx_path]
    output_path = onnx_path

    if fmt == "openvino":
        _, ir_path = build_openvino(onnx_path, precision=precision,
                                    output_dir=output_dir)
        output_path = ir_path
        files += [ir_path, os.path.splitext(ir_path)[0] + ".bin"]
    elif fmt == "tensorrt":
        output_path = build_tensorrt(
            onnx_path, os.path.join(output_dir, f"{output_name}.engine"),
            precision=precision)
        files.append(output_path)

    return {"output_path": output_path, "format": fmt, "files": files,
            "num_frames": frames, "image_size": image_size,
            "batch_size": batch_size, "precision": precision}

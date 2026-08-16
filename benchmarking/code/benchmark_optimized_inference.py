"""Benchmark the same ONNX TSM graph with OpenVINO CPU or TensorRT GPU.

Examples:
  pixi run python benchmarking/code/benchmark_optimized_inference.py openvino --model model.onnx
  pixi run python benchmarking/code/benchmark_optimized_inference.py tensorrt --model model.onnx --fp16

TensorRT uses NVIDIA's trtexec executable because it reports engine build and
GPU timing details more reliably than duplicating its runner in Python.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarking" / "figures" / "inference_time"
REPORT_PATH = RESULTS_DIR / "device_report.txt"

# Reuse the exact crop and validation transform used by the app and the
# non-optimized benchmark rather than maintaining a second approximation.
from benchmark_inference_speed import (
    _crop_frame,
    _preprocess_clip,
    _synthetic_crop_inputs,
)
from backend.services.inference_service import _build_val_transform
from tune_cropping import FindRatBoundingBox


def scenario_suffix(scenario: str | None) -> str:
    """Return a short, filesystem-safe suffix while preserving the full tag in metadata."""
    if not scenario:
        return ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", scenario.strip()).strip("._-")
    return f"_{slug[:80]}" if slug else ""


def device_specs(backend: str) -> dict:
    """Collect stable hardware/runtime facts for the append-only report."""
    import psutil

    def windows_power_context() -> dict:
        context: dict[str, object] = {}
        battery = psutil.sensors_battery()
        if battery:
            context.update({
                "ac_power_connected": battery.power_plugged,
                "battery_percent": battery.percent,
            })
        if platform.system() != "Windows":
            return context
        try:
            context["power_plan"] = subprocess.check_output(
                ["powercfg", "/getactivescheme"], text=True, timeout=5
            ).strip()
        except Exception:
            pass

        # These settings can expose Windows-side changes made by a vendor
        # profile even when every profile keeps the visible plan "Balanced".
        settings = {
            "max_processor_state": "PROCTHROTTLEMAX",
            "min_processor_state": "PROCTHROTTLEMIN",
            "processor_energy_performance_preference": "36687f9e-e3a5-4dbf-b1dc-15eb381c6863",
            "processor_boost_mode": "be337238-0d82-4146-a960-4f3749d470c7",
        }
        for label, setting in settings.items():
            try:
                output = subprocess.check_output(
                    ["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", setting],
                    text=True, timeout=5,
                )
                ac = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", output)
                dc = re.search(r"Current DC Power Setting Index:\s*0x([0-9a-fA-F]+)", output)
                if ac:
                    context[f"{label}_ac"] = int(ac.group(1), 16)
                if dc:
                    context[f"{label}_dc"] = int(dc.group(1), 16)
            except Exception:
                pass
        return context

    if backend == "openvino":
        specs = {
            "processor": platform.processor() or "Unknown CPU",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        }
        try:
            import cpuinfo
            specs["processor"] = cpuinfo.get_cpu_info().get("brand_raw", specs["processor"])
        except Exception:
            pass
        frequency = psutil.cpu_freq()
        if frequency:
            specs["current_frequency_mhz_at_query"] = round(frequency.current, 1)
            specs["max_frequency_mhz"] = round(frequency.max, 1)
        specs.update(windows_power_context())
        return specs

    import torch
    props = torch.cuda.get_device_properties(0)
    specs = {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
        "torch_cuda_runtime": torch.version.cuda,
    }
    try:
        query = subprocess.check_output([
            "nvidia-smi", "--id=0",
            "--query-gpu=driver_version,pstate,power.limit,power.draw,clocks.current.sm,"
            "clocks.current.memory,clocks.max.sm,temperature.gpu,utilization.gpu,"
            "utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip().split(",")
        specs.update({
            "driver_version": query[0].strip(),
            "performance_state_at_query": query[1].strip(),
            "power_limit_w": query[2].strip(),
            "power_draw_w_at_query": query[3].strip(),
            "sm_clock_mhz_at_query": query[4].strip(),
            "memory_clock_mhz_at_query": query[5].strip(),
            "max_sm_clock_mhz": query[6].strip(),
            "temperature_c_at_query": query[7].strip(),
            "gpu_utilization_percent_at_query": query[8].strip(),
            "memory_utilization_percent_at_query": query[9].strip(),
            "vram_used_mib_at_query": query[10].strip(),
        })
    except Exception:
        pass
    specs.update(windows_power_context())
    return specs


def save_bar_plot(model_path: Path, result: dict, tested_at: datetime) -> Path:
    """Save crop, preprocessing, and optimized inference mean ± std bars."""
    import matplotlib.pyplot as plt

    if result.get("mean_ms") is None:
        raise RuntimeError("Backend did not report a mean latency; cannot create plot")
    stages = [
        ("Crop", result["cropping_mean_ms"], result["cropping_std_ms"]),
        ("Preprocess", result["preprocessing_mean_ms"], result["preprocessing_std_ms"]),
        ("Inference", result["mean_ms"], result.get("std_ms") or 0.0),
    ]
    labels = [stage[0] for stage in stages]
    means = [stage[1] for stage in stages]
    stds = [stage[2] for stage in stages]
    backend = result["backend"]
    mode = result.get("precision") or result.get("performance_hint", "")
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(labels, means, yerr=stds, capsize=5, width=0.55,
                  color=["#1baf7a", "#eda100", "#2a78d6"], edgecolor="none",
                  error_kw={"ecolor": "#52514e", "elinewidth": 1.2})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + max(max(means) * 0.02, 0.02),
                f"{mean:.3f} ± {std:.3f} ms", ha="center", va="bottom",
                color="#0b0b0b", fontsize=9)
    ax.margins(y=0.16)
    ax.set_ylabel("Time per pipeline step (ms)", color="#52514e")
    title = f"Optimized pipeline — {model_path.name}\n{backend} {mode}".rstrip()
    if result.get("scenario"):
        title += f"\nScenario: {result['scenario']}"
    ax.set_title(title, color="#0b0b0b", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e1e0d9")
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    tag = scenario_suffix(result.get("scenario"))
    path = RESULTS_DIR / f"{model_path.stem}_{backend}{tag}_{tested_at:%Y%m%d_%H%M%S}.svg"
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def append_device_report(model_path: Path, result: dict, tested_at: datetime) -> Path:
    specs = result.get("device_specs", {})
    lines = [
        "=" * 70,
        f"Model: {model_path}",
        f"Tested at: {tested_at.isoformat(timespec='seconds')}",
        f"Backend: {result['backend']}",
        "",
        "[benchmark]",
    ]
    omitted = {"raw_log", "command", "device", "device_specs", "backend"}
    for key, value in result.items():
        if key not in omitted:
            lines.append(f"    {key}: {value}")
    lines.extend(["", "[device]"])
    lines.extend(f"    {key}: {value}" for key, value in specs.items())
    lines.extend(["", ""])
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write("\n".join(lines))
    return REPORT_PATH


def summarize(times_ms: list[float]) -> dict[str, float]:
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "iterations": int(values.size),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "std_ms": float(values.std()),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "inferences_per_second": float(1000.0 / values.mean()),
    }


def benchmark_pipeline_stages(args: argparse.Namespace, device: str) -> dict:
    """Time the app's crop and tensor preprocessing around an optimized backend."""
    import torch

    frame, background = _synthetic_crop_inputs(args.frame_height, args.frame_width)
    cropper = FindRatBoundingBox(background, box_size=args.crop_size)
    transform = _build_val_transform(args.resolution)
    torch_device = torch.device(device)

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    for _ in range(args.pipeline_warmup):
        rgb = _crop_frame(cropper, frame)
        _preprocess_clip([rgb] * args.num_frames, transform, torch_device)
    sync()

    cropping = []
    preprocessing = []
    for _ in range(args.trials):
        start = time.perf_counter_ns()
        rgb = _crop_frame(cropper, frame)
        cropping.append((time.perf_counter_ns() - start) / 1e6)

        sync()
        start = time.perf_counter_ns()
        _preprocess_clip([rgb] * args.num_frames, transform, torch_device)
        sync()
        preprocessing.append((time.perf_counter_ns() - start) / 1e6)

    crop_summary = summarize(cropping)
    preprocess_summary = summarize(preprocessing)
    return {
        "cropping_mean_ms": crop_summary["mean_ms"],
        "cropping_std_ms": crop_summary["std_ms"],
        "preprocessing_mean_ms": preprocess_summary["mean_ms"],
        "preprocessing_std_ms": preprocess_summary["std_ms"],
        "source_frame_shape": (args.frame_height, args.frame_width, 3),
        "crop_size": args.crop_size,
        "preprocessing_includes_device_transfer": True,
    }


def benchmark_openvino(model_path: Path, trials: int, warmup: int, threads: int | None) -> dict:
    import openvino as ov

    config: dict[str, object] = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    if threads:
        config["INFERENCE_NUM_THREADS"] = threads
    core = ov.Core()
    compiled = core.compile_model(str(model_path), "CPU", config)
    input_port = compiled.input(0)
    shape = tuple(int(x) for x in input_port.shape)
    sample = np.random.default_rng(12345).standard_normal(shape, dtype=np.float32)
    request = compiled.create_infer_request()
    for _ in range(warmup):
        request.infer({input_port: sample})
    times = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        request.infer({input_port: sample})
        times.append((time.perf_counter_ns() - start) / 1e6)
    result = summarize(times)
    result.update({
        "backend": "openvino",
        "runtime_version": ov.__version__,
        "device": compiled.get_property("EXECUTION_DEVICES"),
        "input_shape": shape,
        "threads": threads,
        "performance_hint": "LATENCY",
        "num_streams": 1,
    })
    return result


def find_trtexec(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    found = shutil.which("trtexec") or shutil.which("trtexec.exe")
    if found:
        return Path(found)
    candidates = list(Path(os.environ.get("ProgramFiles", r"C:\Program Files")).glob("NVIDIA*/**/trtexec.exe"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("trtexec was not found; pass --trtexec PATH after installing the TensorRT SDK")


def benchmark_tensorrt(args: argparse.Namespace) -> dict:
    trtexec = find_trtexec(args.trtexec)
    engine = args.engine or args.model.with_suffix(".fp16.engine" if args.fp16 else ".fp32.engine")
    shape = f"frames:{args.batch_size}x{args.num_frames}x3x{args.resolution}x{args.resolution}"
    command = [
        str(trtexec), f"--onnx={args.model}", f"--saveEngine={engine}", f"--shapes={shape}",
        f"--warmUp={args.warmup_ms}", "--duration=0", f"--iterations={args.trials}",
        "--infStreams=1", "--noDataTransfers",
    ]
    if args.fp16:
        command.append("--fp16")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"trtexec failed ({completed.returncode})\n{log}")

    def metric(pattern: str) -> float | None:
        match = re.search(pattern, log, re.IGNORECASE)
        return float(match.group(1)) if match else None

    result = {
        "backend": "tensorrt",
        "precision": "fp16" if args.fp16 else "fp32",
        "engine": str(engine.resolve()),
        "input_shape": shape.split(":", 1)[1],
        "mean_ms": metric(r"GPU Compute Time:.*?mean\s*=\s*([0-9.]+)\s*ms"),
        "std_ms": metric(r"GPU Compute Time:.*?(?:std|variation)\s*=\s*([0-9.]+)\s*ms"),
        "median_ms": metric(r"GPU Compute Time:.*?median\s*=\s*([0-9.]+)\s*ms"),
        "throughput_qps": metric(r"Throughput:\s*([0-9.]+)\s*qps"),
        "command": command,
        "raw_log": log,
    }
    return result


def benchmark_tensorrt_rtx(args: argparse.Namespace) -> dict:
    """Build and benchmark a TensorRT-RTX engine with GPU-resident buffers."""
    try:
        import tensorrt_rtx as trt
        runtime_name = "tensorrt-rtx"
    except ImportError:
        import tensorrt as trt
        runtime_name = "tensorrt-python"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.model)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT-RTX could not parse ONNX:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT-RTX engine build failed")
    engine_path = args.engine or args.model.with_suffix(".trt_rtx.engine")
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    tensors: dict[str, torch.Tensor] = {}
    input_name = None
    output_name = None
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(int(x) for x in engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        if engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f"Unexpected tensor dtype for {name}: {engine.get_tensor_dtype(name)}")
        tensor = (torch.randn(shape, device="cuda", dtype=torch.float32)
                  if mode == trt.TensorIOMode.INPUT else
                  torch.empty(shape, device="cuda", dtype=torch.float32))
        tensors[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    stream = torch.cuda.Stream()
    for _ in range(args.warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT-RTX warmup execution failed")
    stream.synchronize()
    # Validate one TensorRT result against OpenVINO's execution of the shared
    # ONNX graph before timing. This catches parser/build errors independently
    # of the original TorchScript-vs-ONNX validation performed at export time.
    import openvino as ov
    ov_compiled = ov.Core().compile_model(str(args.model), "CPU")
    reference = ov_compiled({ov_compiled.input(0): tensors[input_name].cpu().numpy()})[ov_compiled.output(0)]
    actual = tensors[output_name].cpu().numpy()
    delta = np.abs(reference - actual)
    max_abs = float(delta.max())
    max_rel = float((delta / np.maximum(np.abs(reference), 1e-6)).max())
    # The FP32 graph may select TF32 Tensor Core kernels; a 2% relative bound
    # is appropriate for that execution mode and remains strict enough to catch
    # wrong bindings, shapes, or materially incorrect conversions.
    if not np.allclose(reference, actual, rtol=2e-2, atol=2e-3):
        raise RuntimeError(f"TensorRT validation failed: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    times = []
    for _ in range(args.trials):
        stream.synchronize()
        start = time.perf_counter_ns()
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT-RTX execution failed")
        stream.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    result = summarize(times)
    result.update({
        "backend": runtime_name,
        "runtime_version": trt.__version__,
        "precision": "fp32/tf32",
        "device": torch.cuda.get_device_name(0),
        "input_name": input_name,
        "input_shape": tuple(tensors[input_name].shape),
        "validation_max_abs": max_abs,
        "validation_max_rel": max_rel,
        "engine": str(engine_path.resolve()),
        "workspace_mib": args.workspace_mib,
        "transfers_included": False,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="backend", required=True)
    ovp = sub.add_parser("openvino")
    ovp.add_argument("--model", type=Path, required=True)
    ovp.add_argument("--trials", type=int, default=1000)
    ovp.add_argument("--warmup", type=int, default=100)
    ovp.add_argument("--threads", type=int)
    ovp.add_argument("--scenario", help="Free-text run tag included in outputs and filenames")
    trtp = sub.add_parser("tensorrt")
    trtp.add_argument("--model", type=Path, required=True)
    trtp.add_argument("--engine", type=Path)
    trtp.add_argument("--trtexec", type=Path)
    trtp.add_argument("--fp16", action="store_true")
    trtp.add_argument("--batch-size", type=int, default=1)
    trtp.add_argument("--num-frames", type=int, default=3)
    trtp.add_argument("--resolution", type=int, default=224)
    trtp.add_argument("--trials", type=int, default=1000)
    trtp.add_argument("--warmup-ms", type=int, default=1000)
    trtp.add_argument("--scenario", help="Free-text run tag included in outputs and filenames")
    rtxp = sub.add_parser("tensorrt-rtx")
    rtxp.add_argument("--model", type=Path, required=True)
    rtxp.add_argument("--engine", type=Path)
    rtxp.add_argument("--trials", type=int, default=1000)
    rtxp.add_argument("--warmup", type=int, default=100)
    rtxp.add_argument("--workspace-mib", type=int, default=1024)
    rtxp.add_argument("--scenario", help="Free-text run tag included in outputs and filenames")
    for backend_parser in (ovp, trtp, rtxp):
        if backend_parser is not trtp:
            backend_parser.add_argument("--num-frames", type=int, default=3)
            backend_parser.add_argument("--resolution", type=int, default=224)
        backend_parser.add_argument("--frame-height", type=int, default=1080)
        backend_parser.add_argument("--frame-width", type=int, default=1920)
        backend_parser.add_argument("--crop-size", type=int, default=220)
        backend_parser.add_argument("--pipeline-warmup", type=int, default=10)
    args = parser.parse_args()
    args.model = args.model.resolve()
    if args.backend == "openvino":
        result = benchmark_openvino(args.model, args.trials, args.warmup, args.threads)
    elif args.backend == "tensorrt-rtx":
        result = benchmark_tensorrt_rtx(args)
    else:
        result = benchmark_tensorrt(args)
    pipeline_device = "cpu" if args.backend == "openvino" else "cuda:0"
    result.update(benchmark_pipeline_stages(args, pipeline_device))
    result["scenario"] = args.scenario
    # Query immediately after the timed loop, before plotting or file I/O, so
    # transient clocks/power are as representative of the selected mode as a
    # point-in-time device query can be. The full snapshot is also saved in JSON.
    result["device_specs"] = device_specs(result["backend"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tested_at = datetime.now()
    tag = scenario_suffix(args.scenario)
    output = RESULTS_DIR / f"{args.model.stem}_{result['backend']}{tag}_{tested_at:%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    plot = save_bar_plot(args.model, result, tested_at)
    report = append_device_report(args.model, result, tested_at)
    printable = {k: v for k, v in result.items() if k != "raw_log"}
    print(json.dumps(printable, indent=2, default=str))
    print(f"Saved: {output}")
    print(f"Saved: {plot}")
    print(f"Appended: {report}")


if __name__ == "__main__":
    main()

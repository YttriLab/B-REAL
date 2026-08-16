"""
Benchmark TSM-EfficientNet inference speed across devices (CPU / CUDA GPU(s) / MPS).

Loads either a Lightning checkpoint (.ckpt) or a TorchScript export (.pt, from
convert_torchscript_tsm.py) and times the app's crop, preprocessing, and model
forward-pass stages on synthetic frames, so no real video or labels are needed.

Usage:
    python benchmarking/code/benchmark_inference_speed.py
    python benchmarking/code/benchmark_inference_speed.py --model path/to/epoch=22.ckpt
    python benchmarking/code/benchmark_inference_speed.py --model tsm_binary_groom.pt --num-frames 3
    python benchmarking/code/benchmark_inference_speed.py --model model.ckpt --devices cpu,cuda:0 --scenario "MSI Extreme Performance"

Run with no --model / --devices to be prompted interactively (device selection
supports picking several devices in one run, e.g. "0,1" or "all").
"""

import argparse
import os
import platform
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

# On Windows, a conda/pixi env's CUDA/cuDNN DLLs live in Library/bin, which is
# only put on PATH by activation (`pixi run ...`). Launching this script
# directly against the env's python.exe (e.g. VS Code's Run button) skips
# that — and torch's cuDNN loader uses plain LoadLibrary, which only
# consults PATH, not Python's os.add_dll_directory — so prepend it manually
# before importing torch.
if sys.platform == "win32":
    _library_bin = Path(sys.prefix) / "Library" / "bin"
    if _library_bin.is_dir():
        os.environ["PATH"] = str(_library_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "benchmarking" / "figures" / "inference_time"

from backend.services.inference_service import _build_val_transform
from tune_cropping import FindRatBoundingBox

# Categorical palette (dataviz skill default) — fixed slot order, not cycled/reassigned.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: torch.device, num_frames: int):
    """Returns (callable_model, resolved_num_frames)."""
    if model_path.suffix == ".ckpt":
        from model_tsm import TSMClassifier
        model = TSMClassifier.load_from_checkpoint(
            str(model_path), map_location=device, strict=False
        )
        model.eval().to(device)
        return model, model.num_frames
    else:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model, num_frames


# ---------------------------------------------------------------------------
# Device discovery + specs
# ---------------------------------------------------------------------------

def cpu_label():
    label = platform.processor() or "Unknown CPU"
    if label in {"x86_64", "AMD64", "Unknown CPU"}:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except (OSError, IndexError):
            pass
    return label


def discover_devices():
    devices = [("cpu", cpu_label())]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append((f"cuda:{i}", torch.cuda.get_device_name(i)))
    if torch.backends.mps.is_available():
        devices.append(("mps", "Apple MPS"))
    return devices


def prompt_device_selection(devices):
    print("\nAvailable devices:")
    for idx, (dev_str, label) in enumerate(devices):
        print(f"  [{idx}] {dev_str} — {label}")
    raw = input("\nSelect device(s) to benchmark (comma-separated indices, or 'all'): ").strip()
    if raw.lower() == "all":
        return devices
    chosen = [int(x) for x in raw.split(",") if x.strip() != ""]
    return [devices[i] for i in chosen]


def windows_power_plan():
    """Active Windows power plan — a proxy for vendor 'scenario' tools (MSI Center,
    etc.), which set proprietary WMI values this script can't read directly."""
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def cpu_specs():
    import psutil

    specs = {
        "label": cpu_label(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    }
    freq = psutil.cpu_freq()
    if freq:
        specs["max_freq_mhz"] = round(freq.max, 1)
    try:
        import cpuinfo
        specs["label"] = cpuinfo.get_cpu_info().get("brand_raw", specs["label"])
    except ImportError:
        pass
    specs["power_plan"] = windows_power_plan()
    return specs


def gpu_specs(index: int):
    props = torch.cuda.get_device_properties(index)
    specs = {
        "label": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "multiprocessor_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=driver_version,power.limit,power.draw,clocks.max.sm",
             "--format=csv,noheader"],
            text=True, timeout=5,
        )
        driver, power_limit, power_draw, max_clock = [s.strip() for s in out.strip().split(",")]
        specs["driver_version"] = driver
        specs["power_limit"] = power_limit
        specs["power_draw_at_query"] = power_draw
        specs["max_sm_clock"] = max_clock
    except Exception:
        pass  # nvidia-smi not on PATH — skip the extra fields
    return specs


def get_specs(dev_str: str):
    if dev_str == "cpu":
        return cpu_specs()
    if dev_str.startswith("cuda"):
        index = int(dev_str.split(":")[1]) if ":" in dev_str else 0
        return gpu_specs(index)
    if dev_str == "mps":
        return {"label": "Apple MPS", "os": f"{platform.system()} {platform.release()}"}
    raise ValueError(f"Unknown device: {dev_str}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _synthetic_crop_inputs(frame_height: int, frame_width: int):
    """Create a stable BGR frame/background pair that exercises the app cropper."""
    background = np.full((frame_height, frame_width), 180, dtype=np.float32)
    frame = np.full((frame_height, frame_width, 3), 180, dtype=np.uint8)
    cy, cx = frame_height // 2, frame_width // 2
    radius = max(12, min(frame_height, frame_width) // 10)
    import cv2
    cv2.circle(frame, (cx, cy), radius, (40, 40, 40), thickness=-1)
    return frame, background


def _crop_frame(cropper, frame):
    """Match run_inference_video: crop BGR, then convert the result to RGB."""
    import cv2
    *_, cropped = cropper(frame)
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def _preprocess_clip(rgb_frames, transform, device):
    """Match the app's transform, stack, batch, and device-transfer pipeline."""
    targets = {"image": rgb_frames[0]}
    targets.update({f"image{i}": image for i, image in enumerate(rgb_frames[1:], 1)})
    transformed = transform(**targets)
    tensors = [transformed["image"]]
    tensors.extend(transformed[f"image{i}"] for i in range(1, len(rgb_frames)))
    return torch.stack(tensors).unsqueeze(0).to(device)


def time_pipeline(model, device: torch.device, trials: int, warmup: int,
                  num_frames: int, resolution: int, batch_size: int,
                  frame_height: int, frame_width: int, crop_size: int):
    is_cuda = device.type == "cuda"
    stage_times = {"cropping_ms": [], "preprocessing_ms": [], "inference_ms": []}
    frame, background = _synthetic_crop_inputs(frame_height, frame_width)
    cropper = FindRatBoundingBox(background, box_size=crop_size)
    transform = _build_val_transform(resolution)

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in tqdm(range(warmup), desc=f"[{device}] warmup", leave=False):
            rgb = _crop_frame(cropper, frame)
            clip = _preprocess_clip([rgb] * num_frames, transform, device)
            if batch_size > 1:
                clip = clip.repeat(batch_size, 1, 1, 1, 1)
            model(clip)
        synchronize()

        for _ in tqdm(range(trials), desc=f"[{device}] trials"):
            synchronize()
            start = time.perf_counter()
            rgb = _crop_frame(cropper, frame)
            stage_times["cropping_ms"].append((time.perf_counter() - start) * 1000)

            synchronize()
            start = time.perf_counter()
            clip = _preprocess_clip([rgb] * num_frames, transform, device)
            if batch_size > 1:
                clip = clip.repeat(batch_size, 1, 1, 1, 1)
            synchronize()
            stage_times["preprocessing_ms"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            model(clip)
            synchronize()
            stage_times["inference_ms"].append((time.perf_counter() - start) * 1000)
    return stage_times


# ---------------------------------------------------------------------------
# Output: bar plot + device report
# ---------------------------------------------------------------------------

def save_bar_plot(model_path: Path, results: dict, tested_at: datetime) -> Path:
    dev_strs = list(results.keys())
    stages = [("cropping_ms", "Crop"), ("preprocessing_ms", "Preprocess"),
              ("inference_ms", "Inference")]
    labels = [f"{d}\n" + "\n".join(textwrap.wrap(results[d]["label"], width=16))
              for d in dev_strs]
    x = np.arange(len(dev_strs), dtype=float)
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(2.4 * len(dev_strs), 4) + 1.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for stage_idx, (key, stage_label) in enumerate(stages):
        means = [float(np.mean(results[d][key])) for d in dev_strs]
        stds = [float(np.std(results[d][key])) for d in dev_strs]
        positions = x + (stage_idx - 1) * width
        bars = ax.bar(positions, means, yerr=stds, capsize=3,
                      color=PALETTE[stage_idx], edgecolor="none", width=width,
                      label=stage_label,
                      error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0})
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std,
                    f"{mean:.2f}", ha="center", va="bottom",
                    color=INK_PRIMARY, fontsize=8, rotation=45)

    ax.margins(y=0.18)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Time per pipeline step (ms)", color=INK_SECONDARY)
    ax.set_title(f"Crop, preprocessing, and inference — {model_path.name}",
                 color=INK_PRIMARY, fontsize=13, pad=14)
    ax.legend(frameon=False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    stamp = tested_at.strftime("%Y%m%d_%H%M%S")
    out_path = FIGURES_DIR / f"{model_path.stem}_{stamp}.svg"
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def save_device_report(model_path: Path, specs_all: dict, results: dict,
                        tested_at: datetime, args) -> Path:
    report_path = FIGURES_DIR / "device_report.txt"
    lines = [
        "=" * 70,
        f"Model: {model_path}",
        f"Tested at: {tested_at.isoformat(timespec='seconds')}",
        f"Trials: {args.trials}  Warmup: {args.warmup}  Batch size: {args.batch_size}",
    ]
    if args.scenario:
        lines.append(f"Scenario label (user-supplied): {args.scenario}")
    lines.append("")

    for dev_str, specs in specs_all.items():
        lines.append(f"[{dev_str}] {results[dev_str]['label']}")
        for key, value in specs.items():
            if key == "label":
                continue
            lines.append(f"    {key}: {value}")
        for key, label in (("cropping_ms", "cropping"),
                           ("preprocessing_ms", "preprocessing"),
                           ("inference_ms", "inference")):
            values = results[dev_str][key]
            lines.append(f"    {label}_mean_ms: {np.mean(values):.3f}")
            lines.append(f"    {label}_std_ms: {np.std(values):.3f}")
        inference_mean = np.mean(results[dev_str]["inference_ms"])
        lines.append(f"    inference_fps: {1000.0 / inference_mean:.2f}")
        lines.append("")
    lines.append("")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a .ckpt (Lightning) or .pt (TorchScript) model")
    parser.add_argument("--num-frames", type=int, default=3,
                         help="Temporal window T (ignored for .ckpt — read from the checkpoint's hparams)")
    parser.add_argument("--resolution", type=int, default=224, help="Square input resolution H=W")
    parser.add_argument("--frame-height", type=int, default=1080,
                        help="Synthetic source-frame height used for crop timing")
    parser.add_argument("--frame-width", type=int, default=1920,
                        help="Synthetic source-frame width used for crop timing")
    parser.add_argument("--crop-size", type=int, default=220,
                        help="Crop box size, matching the app default")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trials", type=int, default=50, help="Timed forward passes per device")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warm-up passes per device")
    parser.add_argument("--devices", type=str, default=None,
                         help="Comma-separated device strings (e.g. 'cpu,cuda:0') — skips the interactive prompt")
    parser.add_argument("--scenario", type=str, default=None,
                         help="Free-text label for the current power/performance profile "
                              "(e.g. an MSI Center 'User Scenario' setting) — not auto-detectable, "
                              "so pass whatever you set manually before this run")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else Path(
        input("Path to model (.ckpt or .pt): ").strip().strip('"')
    )
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    available = discover_devices()
    if args.devices:
        wanted = {d.strip() for d in args.devices.split(",")}
        selected = [d for d in available if d[0] in wanted]
    else:
        selected = prompt_device_selection(available)

    if not selected:
        sys.exit("No devices selected.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    specs_all = {}
    tested_at = datetime.now()

    for dev_str, label in selected:
        print(f"\n--- Benchmarking on {dev_str} ({label}) ---")
        device = torch.device(dev_str)
        model, num_frames = load_model(model_path, device, args.num_frames)
        stage_times = time_pipeline(
            model, device, args.trials, args.warmup, num_frames, args.resolution,
            args.batch_size, args.frame_height, args.frame_width, args.crop_size,
        )
        results[dev_str] = {"label": label, **stage_times}
        specs_all[dev_str] = get_specs(dev_str)

        for key, stage_label in (("cropping_ms", "cropping"),
                                 ("preprocessing_ms", "preprocessing"),
                                 ("inference_ms", "inference")):
            values = stage_times[key]
            print(f"  {stage_label}: mean={np.mean(values):.3f} ms  "
                  f"std={np.std(values):.3f} ms")
        inference_mean = float(np.mean(stage_times["inference_ms"]))
        print(f"  inference throughput: {1000.0 / inference_mean:.1f} fps")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_path = save_bar_plot(model_path, results, tested_at)
    report_path = save_device_report(model_path, specs_all, results, tested_at, args)

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved device report to {report_path}")


if __name__ == "__main__":
    main()

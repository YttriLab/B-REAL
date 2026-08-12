"""
Benchmark TSM-EfficientNet inference speed across devices (CPU / CUDA GPU(s) / MPS).

Loads either a Lightning checkpoint (.ckpt) or a TorchScript export (.pt, from
convert_torchscript_tsm.py) and times forward passes on synthetic input — content
doesn't affect speed, only tensor shape/device/precision do, so no real video or
labels are needed.

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

def discover_devices():
    devices = [("cpu", platform.processor() or "CPU")]
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
        "label": platform.processor() or "Unknown CPU",
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

def time_inference(model, dummy_input, device: torch.device, trials: int, warmup: int):
    is_cuda = device.type == "cuda"
    times_ms = []
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc=f"[{device}] warmup", leave=False):
            model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize(device)

        for _ in tqdm(range(trials), desc=f"[{device}] trials"):
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms


# ---------------------------------------------------------------------------
# Output: bar plot + device report
# ---------------------------------------------------------------------------

def save_bar_plot(model_path: Path, results: dict, tested_at: datetime) -> Path:
    dev_strs = list(results.keys())
    means = [float(np.mean(results[d]["times_ms"])) for d in dev_strs]
    stds = [float(np.std(results[d]["times_ms"])) for d in dev_strs]
    labels = [
        f"{d}\n" + "\n".join(textwrap.wrap(results[d]["label"], width=16))
        for d in dev_strs
    ]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(dev_strs))]

    fig, ax = plt.subplots(figsize=(max(2.4 * len(dev_strs), 4) + 1.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                   edgecolor="none", width=0.55,
                   error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})

    label_pad = 0.03 * max(means)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + label_pad,
                 f"{mean:.1f} ms", ha="center", va="bottom", color=INK_PRIMARY, fontsize=10)

    ax.set_ylabel("Inference time (ms)", color=INK_SECONDARY)
    ax.set_title(f"Inference speed — {model_path.name}", color=INK_PRIMARY, fontsize=13, pad=14)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    stamp = tested_at.strftime("%Y%m%d_%H%M%S")
    out_path = FIGURES_DIR / f"{model_path.stem}_{stamp}.svgs"
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
        times_ms = results[dev_str]["times_ms"]
        lines.append(f"[{dev_str}] {results[dev_str]['label']}")
        for key, value in specs.items():
            if key == "label":
                continue
            lines.append(f"    {key}: {value}")
        lines.append(f"    mean_ms: {np.mean(times_ms):.3f}")
        lines.append(f"    std_ms: {np.std(times_ms):.3f}")
        lines.append(f"    fps: {1000.0 / np.mean(times_ms):.2f}")
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
        dummy_input = torch.randn(
            args.batch_size, num_frames, 3, args.resolution, args.resolution, device=device
        )

        times_ms = time_inference(model, dummy_input, device, args.trials, args.warmup)
        results[dev_str] = {"label": label, "times_ms": times_ms}
        specs_all[dev_str] = get_specs(dev_str)

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        print(f"  mean={mean_ms:.2f} ms  std={std_ms:.2f} ms  ({1000.0 / mean_ms:.1f} fps)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_path = save_bar_plot(model_path, results, tested_at)
    report_path = save_device_report(model_path, specs_all, results, tested_at, args)

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved device report to {report_path}")


if __name__ == "__main__":
    main()

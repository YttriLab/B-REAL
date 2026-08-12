"""
Plots MCC vs. epoch for a single size-learning-curve run, using the per-epoch
trajectory logged by EpochMetricsWriter (benchmark_size_learning_curve.py) to
epoch_metrics/frac{f}_rep{r}.jsonl. Diagnoses whether a low final test MCC is
an under-training problem (curves still climbing) or something else (val
curve already plateaued/falling, or a gap between val and test MCC).

Usage:
    python benchmarking/code/plot_epoch_mcc.py --fraction 100 --repeat 0
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "benchmarking" / "figures" / "size_learning_curve"

# Same palette/tokens as benchmark_size_learning_curve.py, for visual consistency.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_epoch_metrics(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_test_mcc(results_path: Path, fraction: int, repeat: int) -> float | None:
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    for r in results:
        if r["fraction"] == fraction and r["repeat"] == repeat:
            return r["mcc"]
    return None


def plot_mcc_vs_epoch(records: list[dict], fraction: int, repeat: int,
                       test_mcc: float | None, out_path: Path) -> None:
    epochs = [r["epoch"] for r in records]
    train_mcc = [r["train_mcc"] for r in records]
    val_mcc = [r["val_mcc"] for r in records]
    val_loss = [r["val_loss"] for r in records]

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(epochs, train_mcc, color=PALETTE[0], marker="o", markersize=6,
            linewidth=1.8, zorder=3, label="Train MCC")
    ax.plot(epochs, val_mcc, color=PALETTE[1], marker="o", markersize=6,
            linewidth=1.8, zorder=3, label="Val MCC")

    ax.axvline(epochs[best_idx], color=INK_SECONDARY, linewidth=1, linestyle="--",
               zorder=1, label=f"Checkpoint used (epoch {epochs[best_idx]}, best val_loss)")

    if test_mcc is not None:
        ax.axhline(test_mcc, color=PALETTE[5], linewidth=1.5, linestyle=":",
                   zorder=2, label=f"Held-out test MCC ({test_mcc:.3f})")

    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel("MCC", color=INK_PRIMARY)
    ax.set_title(f"MCC vs. epoch -- fraction={fraction}% repeat={repeat}",
                 color=INK_PRIMARY, fontsize=13, pad=14)
    ax.set_xticks(epochs)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraction", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    epoch_log_path = output_dir / "epoch_metrics" / f"frac{args.fraction}_rep{args.repeat}.jsonl"
    if not epoch_log_path.exists():
        raise FileNotFoundError(f"No epoch log at {epoch_log_path}")

    records = load_epoch_metrics(epoch_log_path)
    test_mcc = load_test_mcc(output_dir / "results.json", args.fraction, args.repeat)

    out_path = output_dir / f"mcc_vs_epoch_frac{args.fraction}_rep{args.repeat}.svg"
    plot_mcc_vs_epoch(records, args.fraction, args.repeat, test_mcc, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

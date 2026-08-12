"""Run a size-learning curve over predeclared global chronological folds.

The YAML config must contain ``benchmark_folds``, a list of mappings with a
unique ``name`` and a ``split_order`` permutation of train/val/test. For each
fold this runner writes a resolved child config, invokes
benchmark_size_learning_curve.py in a separate fold directory, and combines
the completed fold results into parent-level JSON and plots.

Examples:
    python benchmarking/code/benchmark_split_order_folds.py \
        --config benchmarking/configs/gui_global_split_order_folds.yaml \
        --fractions 0,1,5,10,15,20,40,60,80,100 --repeats 3 \
        --output-dir benchmarking/figures/gui_global_split_order_folds

    python benchmarking/code/benchmark_split_order_folds.py \
        --output-dir benchmarking/figures/gui_global_split_order_folds --resume
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking.code.benchmark_size_learning_curve import plot_learning_curve


SPLIT_NAMES = {"train", "val", "test"}
CHILD_SCRIPT = Path(__file__).with_name("benchmark_size_learning_curve.py")


def load_fold_specs(config) -> list[dict]:
    raw_folds = OmegaConf.to_container(config.get("benchmark_folds"), resolve=True)
    if not raw_folds:
        raise ValueError("Config must define a non-empty benchmark_folds list")

    folds = []
    seen_names = set()
    for raw in raw_folds:
        name = str(raw.get("name", ""))
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"Invalid fold name {name!r}; use letters, numbers, '.', '_' or '-'")
        if name in seen_names:
            raise ValueError(f"Duplicate fold name: {name}")
        order = list(raw.get("split_order", []))
        if len(order) != 3 or set(order) != SPLIT_NAMES:
            raise ValueError(
                f"Fold {name!r} split_order must contain train, val, and test exactly once; got {order!r}"
            )
        seen_names.add(name)
        folds.append({"name": name, "split_order": order})
    return folds


def child_arguments(args, fold_config_path: Path, fold_dir: Path, fold_index: int) -> list[str]:
    state_path = fold_dir / "run_state.json"
    if args.resume and state_path.exists():
        return ["--output-dir", str(fold_dir), "--resume"]

    command = [
        "--config", str(fold_config_path),
        "--fractions", args.fractions,
        "--repeats", str(args.repeats),
        "--seed", str(args.seed + fold_index * 1_000_000),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--fps", str(args.fps),
        "--output-dir", str(fold_dir),
        "--notify-email", args.notify_email,
    ]
    if args.max_epochs is not None:
        command += ["--max-epochs", str(args.max_epochs)]
    if args.patience is not None:
        command += ["--patience", str(args.patience)]
    if not args.progress_bar:
        command.append("--no-progress-bar")
    if not args.notify:
        command.append("--no-notify")
    return command


def save_combined_outputs(output_dir: Path, rows: list[dict], fps: float) -> None:
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summaries = []
    for fraction in sorted({row["fraction"] for row in rows}):
        fraction_rows = [row for row in rows if row["fraction"] == fraction]
        summaries.append({
            "fraction": fraction,
            "n_runs": len(fraction_rows),
            "folds": sorted({row["fold"] for row in fraction_rows}),
            "mcc_mean": float(np.mean([row["mcc"] for row in fraction_rows])),
            "mcc_std": float(np.std([row["mcc"] for row in fraction_rows])),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fraction_rows])),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in fraction_rows])),
        })
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    x_unit = "seconds" if fps > 0 else "frames"
    for key, ylabel, title in (
        ("accuracy", "Accuracy", "Accuracy vs. training data size across temporal folds"),
        ("macro_f1", "Macro F1", "Macro F1 vs. training data size across temporal folds"),
        ("mcc", "MCC", "MCC vs. training data size across temporal folds"),
    ):
        plot_learning_curve(
            rows, key, ylabel, title, output_dir / f"{key}_vs_{x_unit}.svg", fps=fps
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="benchmarking/configs/gui_global_split_order_folds.yaml")
    parser.add_argument("--fractions", default="0,1,5,10,15,20,40,60,80,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument("--output-dir", default="benchmarking/figures/gui_global_split_order_folds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notify-email", default="braydenk@andrew.cmu.edu")
    parser.add_argument("--no-notify", dest="notify", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print child commands without writing or training")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "fold_manifest.json"

    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No fold run to resume at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in manifest["arguments"].items():
            setattr(args, key, value)
        config_path = output_dir / "base_config.yaml"
        config = OmegaConf.load(config_path)
        folds = manifest["folds"]
    else:
        config_path = Path(args.config).resolve()
        config = OmegaConf.load(config_path)
        folds = load_fold_specs(config)

    planned = []
    for fold_index, fold in enumerate(folds):
        fold_dir = output_dir / "folds" / fold["name"]
        fold_config_path = output_dir / "fold_configs" / f"{fold['name']}.yaml"
        planned.append((fold_index, fold, fold_dir, fold_config_path))

    if args.dry_run:
        for fold_index, fold, fold_dir, fold_config_path in planned:
            child_args = child_arguments(args, fold_config_path, fold_dir, fold_index)
            print(f"{fold['name']}: {sys.executable} -u {CHILD_SCRIPT} {' '.join(child_args)}")
        return

    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory or pass --resume."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_config": str(config_path),
        "folds": folds,
        "arguments": {
            key: value for key, value in vars(args).items()
            if key not in {"config", "output_dir", "resume", "dry_run"}
        },
    }
    if not args.resume:
        OmegaConf.save(config, output_dir / "base_config.yaml")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    combined = []
    for fold_index, fold, fold_dir, fold_config_path in planned:
        fold_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        del fold_config["benchmark_folds"]
        fold_config.data.split_order = fold["split_order"]
        if fold_config.data.get("split_order_by_dataset") is not None:
            del fold_config.data["split_order_by_dataset"]
        fold_config.run_name = f"{config.run_name}_{fold['name']}"
        fold_config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(fold_config, fold_config_path)
        fold_dir.mkdir(parents=True, exist_ok=True)

        child_args = child_arguments(args, fold_config_path, fold_dir, fold_index)
        print(f"\n=== Fold {fold_index + 1}/{len(folds)}: {fold['name']} "
              f"({'/'.join(fold['split_order'])}) ===", flush=True)
        subprocess.run([sys.executable, "-u", str(CHILD_SCRIPT), *child_args], cwd=REPO_ROOT, check=True)

        fold_rows = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
        for row in fold_rows:
            combined.append({"fold": fold["name"], "split_order": fold["split_order"], **row})
        save_combined_outputs(output_dir, combined, args.fps)

    print(f"\nCompleted {len(folds)} folds. Combined results: {output_dir / 'results.json'}")
    print(f"Combined summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

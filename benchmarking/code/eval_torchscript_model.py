"""
Evaluates an exported TorchScript model (from convert_torchscript_tsm.py)
against the real val/test splits produced by a training config, rather than
the synthetic timing-only input benchmark_inference_speed.py uses. Reports
accuracy, macro F1, MCC, per-class precision/recall/F1, and a confusion
matrix -- the same metrics benchmark_size_learning_curve.py records -- so a
production checkpoint's quality can be sanity-checked against whatever data
a training config points at.

GPU note: a torch.jit.optimize_for_inference'd export can bake
`prim::FusionGroup` nodes into the graph. On CUDA those JIT-compile via
nvrtc at first call; if the box has no matching nvrtc-builtins DLL (no
system CUDA toolkit, and conda-forge's pytorch-gpu doesn't vendor it), that
compile fails with "failed to open nvrtc-builtins64_1xx.dll". This script
disables the fusers up front so the JIT interpreter runs each op
individually (plain cuDNN/cuBLAS kernels, no nvrtc involved) -- slightly
slower, but avoids the missing DLL entirely.

Usage:
    python benchmarking/code/eval_torchscript_model.py --model _models/torchscript_models/tsm_binary_groom.pt
    python benchmarking/code/eval_torchscript_model.py --model path/to/model.pt --device cpu --split test
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Same fix as benchmark_inference_speed.py: a conda/pixi env's CUDA/cuDNN DLLs
# live in Library/bin, which is only put on PATH by activation (`pixi run`/
# `conda run`). Launching the env's python.exe directly skips that.
if sys.platform == "win32":
    _library_bin = Path(sys.prefix) / "Library" / "bin"
    if _library_bin.is_dir():
        os.environ["PATH"] = str(_library_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch

# Must happen before the first forward pass -- see module docstring.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_gpu(False)
for _flag_setter in ("_jit_set_texpr_fuser_enabled", "_jit_set_nvfuser_enabled"):
    if hasattr(torch._C, _flag_setter):
        getattr(torch._C, _flag_setter)(False)

import albumentations as A
from omegaconf import OmegaConf
from torchmetrics import ConfusionMatrix, MatthewsCorrCoef
from torch.utils.data import DataLoader

from model import load_data, train_test_split, MultiImageDataset


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, class_names: list) -> dict:
    accuracy = float(np.mean(y_true == y_pred))
    per_class = {}
    f1s = []
    for c, name in enumerate(class_names):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    macro_f1 = float(np.mean(f1s))

    task = "binary" if num_classes <= 2 else "multiclass"
    t_true = torch.as_tensor(y_true, dtype=torch.long)
    t_pred = torch.as_tensor(y_pred, dtype=torch.long)
    mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)(t_pred, t_true).item()
    cm = ConfusionMatrix(task=task, num_classes=num_classes)(t_pred, t_true).tolist()
    return {"accuracy": accuracy, "macro_f1": macro_f1, "mcc": mcc, "per_class": per_class, "confusion_matrix": cm}


def evaluate_split(model, split_data, device, batch_size, num_workers, class_names, num_classes) -> dict:
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        A.pytorch.ToTensorV2(),
    ], additional_targets={"image1": "image", "image2": "image"})

    dataset = MultiImageDataset(split_data, val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for images, labels, _boundary_dists in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.append(preds)
            y_true_list.append(labels.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return compute_full_metrics(y_true, y_pred, num_classes, class_names)


def print_metrics(split_name: str, n: int, metrics: dict) -> None:
    print(f"\n=== {split_name} ({n} samples) ===")
    print(f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  mcc={metrics['mcc']:.4f}")
    for name, pc in metrics["per_class"].items():
        print(f"  {name}: precision={pc['precision']:.4f} recall={pc['recall']:.4f} "
              f"f1={pc['f1']:.4f} support={pc['support']}")
    print(f"  confusion_matrix={metrics['confusion_matrix']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Path to a TorchScript .pt export")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config_tsm.yaml"),
                         help="Training config providing data_loading + data (split percentages)")
    parser.add_argument("--split", type=str, default="val,test", help="Comma-separated: val, test, or both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    dl = OmegaConf.to_container(config.data_loading, resolve=True)
    all_file_paths, all_labels, class_names, all_boundary_dists = load_data(**dl)

    train_data, valid_data, test_data, test_clip_sizes = train_test_split(
        all_file_paths, all_labels, all_boundary_dists,
        **OmegaConf.to_container(config.data, resolve=True)
    )
    num_classes = len(class_names)
    print(f"class_names={class_names}  test_clip_sizes={test_clip_sizes}")
    print(f"train={len(train_data[1])} val={len(valid_data[1])} test={len(test_data[1])}")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    splits = {"val": valid_data, "test": test_data}
    for split_name in args.split.split(","):
        split_name = split_name.strip()
        split_data = splits[split_name]
        metrics = evaluate_split(model, split_data, device, args.batch_size, args.num_workers,
                                  class_names, num_classes)
        print_metrics(split_name, len(split_data[1]), metrics)


if __name__ == "__main__":
    main()

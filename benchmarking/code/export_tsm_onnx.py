"""Export a deployed TSM TorchScript model to static-shape ONNX.

The exported graph is shared by TensorRT and OpenVINO benchmarks.  Keeping all
dimensions static gives both compilers the best opportunity to optimize the
real-time batch=1 workload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def export_torchscript(model_path: Path, output_path: Path, shape: tuple[int, ...], opset: int) -> None:
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    example = torch.randn(*shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # A ScriptModule has already been captured by TorchScript, so the legacy
    # ONNX bridge is the compatible route. For checkpoint/nn.Module exports we
    # should use the newer dynamo exporter instead.
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=["frames"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )

    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)

    # OpenVINO is both a target runtime and an independent numerical check of
    # the ONNX graph. Multiple randomized inputs catch conversion mistakes that
    # a single all-zero input can hide.
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(str(output_path), "CPU")
    ov_input = compiled.input(0)
    max_abs = 0.0
    max_rel = 0.0
    rng = np.random.default_rng(12345)
    with torch.inference_mode():
        for _ in range(3):
            sample = rng.standard_normal(shape, dtype=np.float32)
            expected = model(torch.from_numpy(sample)).detach().cpu().numpy()
            actual = compiled({ov_input: sample})[compiled.output(0)]
            delta = np.abs(expected - actual)
            max_abs = max(max_abs, float(delta.max()))
            max_rel = max(max_rel, float((delta / np.maximum(np.abs(expected), 1e-6)).max()))

    if not np.allclose(expected, actual, rtol=1e-3, atol=1e-4):
        raise RuntimeError(f"ONNX validation failed: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

    print(f"Saved: {output_path}")
    print(f"Input: frames {shape} float32")
    print(f"Validation: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="TorchScript .pt model")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    shape = (args.batch_size, args.num_frames, 3, args.resolution, args.resolution)
    export_torchscript(args.model.resolve(), args.output.resolve(), shape, args.opset)


if __name__ == "__main__":
    main()

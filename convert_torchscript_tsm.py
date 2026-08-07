"""
Convert a TSM-EfficientNet checkpoint to a single TorchScript model.

Unlike the DINOv2 approach (which splits into extractor.pt + classifier.pt),
TSM cannot be split because temporal shifting happens inside the convolutions.
The entire model is exported as one TorchScript file.

Usage:
    python convert_torchscript_tsm.py

The exported model expects input shape (1, T, 3, 224, 224) and outputs logits.
"""

import torch
from torch import nn
from model_tsm import TSMClassifier
import os


class TSMInferenceWrapper(nn.Module):
    """
    Thin wrapper that extracts just the forward pass components
    from the Lightning module for TorchScript export.
    """

    def __init__(self, lightning_model: TSMClassifier):
        super().__init__()
        self.backbone = lightning_model.backbone
        self.head = lightning_model.head
        self.num_frames = lightning_model.num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — batch of frame triplets

        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Reshape for 2D backbone: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # EfficientNet features + avgpool
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (B*T, feature_dim)

        # Reshape back: (B, T, feature_dim)
        x = x.view(B, T, -1)

        # Take last frame's features
        x = x[:, -1, :]  # (B, feature_dim)

        logits = self.head(x)
        return logits


def export_model(checkpoint_path, output_name, output_dir, device="cuda",
                 tsm_stages=None):
    """
    Export TSM-EfficientNet checkpoint to TorchScript.

    Args:
        checkpoint_path: Path to .ckpt file
        output_name: Name for the output file (without .pt)
        output_dir: Directory to save the exported model
        device: Device to use for export
        tsm_stages: Override tsm_stages if checkpoint used a different config.
            Pass [1,2,3,4,5,6,7] for checkpoints trained with all stages.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.pt")

    # Load the trained model
    load_kwargs = {}
    if tsm_stages is not None:
        load_kwargs["tsm_stages"] = tsm_stages

    base_model = TSMClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False, **load_kwargs
    )
    base_model.eval()

    # Wrap for clean export
    wrapper = TSMInferenceWrapper(base_model)
    wrapper.to(device)
    wrapper.eval()

    print(f"Model loaded on {device}")

    # Dummy input: (batch=1, T=3, C=3, H=224, W=224)
    num_frames = base_model.num_frames
    dummy_input = torch.randn(1, num_frames, 3, 224, 224, device=device)

    print(f"Tracing with input shape: {dummy_input.shape}")

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    # Optimize for inference
    traced = torch.jit.optimize_for_inference(traced)

    # Save
    traced.save(output_path)
    print(f"\n[OK] Model exported to: {output_path}")

    # Verify
    print("Testing exported model...")
    loaded = torch.jit.load(output_path, map_location=device)

    with torch.no_grad():
        orig_out = wrapper(dummy_input)
        loaded_out = loaded(dummy_input)

    max_diff = (orig_out - loaded_out).abs().max().item()
    print(f"Max output difference: {max_diff:.2e}")
    print("[OK] Export successful!")

    return output_path


def benchmark_model(
    variant="b0",
    num_frames=3,
    tsm_stages=None,
    temporal_pool="last",
    unidirectional=False,
    image_size=224,
    batch_size=1,
    precision="fp32",
    iters=100,
    warmup=20,
    device=None,
):
    """Time the *optimized* (TorchScript + optimize_for_inference) forward pass
    for an untrained model of the given architecture.

    Inference latency depends only on architecture + input shape, not on trained
    weights, so this lets a user size their "compute budget" before training. It
    mirrors what export_model produces (same wrapper, same optimization), so the
    numbers reflect the artifact actually deployed to the rig.

    Returns a dict of timings and the resolved environment.
    """
    import time

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    note = None
    use_half = str(precision).lower() in ("fp16", "half", "16")
    if use_half and device != "cuda":
        # Half convs are unsupported / pathologically slow on CPU — don't pretend.
        use_half = False
        note = "FP16 requested but no CUDA device; benchmarked FP32 instead."

    torch.backends.cudnn.benchmark = True

    # Untrained model with the requested architecture (weights don't affect timing;
    # pretrained=False avoids a download). Two classes is representative.
    model = TSMClassifier(
        variant=variant,
        num_frames=num_frames,
        tsm_stages=tsm_stages,
        temporal_pool=temporal_pool,
        unidirectional=unidirectional,
        pretrained=False,
        class_names=["a", "b"],
    )
    wrapper = TSMInferenceWrapper(model).to(device).eval()
    dtype = torch.float16 if use_half else torch.float32
    if use_half:
        wrapper = wrapper.half()

    x = torch.randn(batch_size, num_frames, 3, image_size, image_size,
                    device=device, dtype=dtype)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, x)
        try:
            traced = torch.jit.optimize_for_inference(traced)
        except Exception:
            pass  # some graphs reject it; fall back to the plain traced module

        for _ in range(max(1, warmup)):
            traced(x)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            traced(x)
        if device == "cuda":
            torch.cuda.synchronize()
        total = time.perf_counter() - t0

    clips = max(1, iters) * batch_size
    clips_per_sec = clips / total
    ms_per_clip = 1000.0 / clips_per_sec
    ms_per_batch = (total / max(1, iters)) * 1000.0

    result = {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "precision": "fp16" if use_half else "fp32",
        "variant": variant,
        "num_frames": num_frames,
        "tsm_stages": tsm_stages,
        "temporal_pool": temporal_pool,
        "unidirectional": unidirectional,
        "image_size": image_size,
        "batch_size": batch_size,
        "iters": iters,
        "ms_per_clip": ms_per_clip,
        "ms_per_batch": ms_per_batch,
        "clips_per_sec": clips_per_sec,
    }
    if note:
        result["note"] = note
    return result


def main():
    checkpoint_path = r"aeagle-yttri-lab\zmjox2sf\checkpoints\epoch=22-step=19711.ckpt"
    output_name = "tsm_binary_groom"
    output_dir = "torchscript_models"

    # This checkpoint was trained with TSM on all stages
    tsm_stages = [1, 2, 3, 4, 5, 6, 7]

    export_model(checkpoint_path, output_name, output_dir, tsm_stages=tsm_stages)


if __name__ == "__main__":
    main()

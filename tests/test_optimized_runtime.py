"""Tests for optimized_runtime: ONNX export and the OpenVINO / TensorRT paths.

The GPU path needs `tensorrt` and a CUDA device, so it is skipped where either
is absent; the CPU (OpenVINO) path and the shared plumbing are always covered.
"""

import os

import pytest
import torch

import optimized_runtime
from optimized_runtime import (
    RuntimeUnavailable,
    benchmark_optimized,
    build_wrapper,
    export_onnx,
    export_optimized,
    preflight_target,
    resolve_target,
    summarize,
)

# Architecture kwargs shared by build_wrapper and benchmark_optimized. image_size
# is an input-shape knob, not a model one, so it is kept separate — a small one
# keeps these tests quick.
ARCH = dict(variant="b0", num_frames=3, tsm_stages=[6, 7])
IMAGE_SIZE = 64


# ── target resolution ──────────────────────────────────────────────────────

def _trt_capable_gpu() -> bool:
    """TensorRT needs both a CUDA device and sm_75+ (Turing or newer)."""
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    return (p.major, p.minor) >= optimized_runtime._TRT_MIN_SM


class TestResolveTarget:
    def test_explicit_target_is_kept(self):
        assert resolve_target("openvino") == "openvino"
        assert resolve_target("torchscript") == "torchscript"

    def test_auto_follows_the_hardware(self):
        """auto picks what this box can actually serve — a CUDA device is not
        enough, it has to be new enough for TensorRT."""
        expected = "tensorrt" if _trt_capable_gpu() else "openvino"
        assert resolve_target("auto") == expected

    def test_auto_skips_tensorrt_on_an_old_gpu(self, monkeypatch):
        class FakeProps:
            name, major, minor = "NVIDIA GeForce GTX 1080 Ti", 6, 1

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps())
        assert resolve_target("auto") == "openvino"

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            resolve_target("coreml")


class TestPreflight:
    def test_torchscript_needs_nothing(self):
        assert preflight_target("torchscript") == "torchscript"

    def test_missing_package_raises_with_an_install_hint(self, monkeypatch):
        def missing(module):
            raise RuntimeUnavailable(f"{module} is not installed, pip install {module}")
        monkeypatch.setattr(optimized_runtime, "_require", missing)
        with pytest.raises(RuntimeUnavailable) as e:
            preflight_target("openvino")
        assert "pip install" in str(e.value)

    def test_tensorrt_without_cuda_is_refused(self, monkeypatch):
        monkeypatch.setattr(optimized_runtime, "_require", lambda m: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeUnavailable) as e:
            preflight_target("tensorrt")
        assert "no CUDA device" in str(e.value)

    def test_old_gpu_is_named_in_the_error(self, monkeypatch):
        """TensorRT 10+ dropped Pascal. Left to itself the builder fails with an
        opaque pybind11 nullptr TypeError, so the arch is checked up front."""
        class FakeProps:
            name, major, minor = "NVIDIA GeForce GTX 1080 Ti", 6, 1

        monkeypatch.setattr(optimized_runtime, "_require", lambda m: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps())
        with pytest.raises(RuntimeUnavailable) as e:
            preflight_target("tensorrt")
        msg = str(e.value)
        assert "sm_61" in msg and "1080 Ti" in msg and "sm_75" in msg

    def test_modern_gpu_passes_the_arch_check(self, monkeypatch):
        class FakeProps:
            name, major, minor = "NVIDIA RTX 4090", 8, 9

        monkeypatch.setattr(optimized_runtime, "_require", lambda m: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: FakeProps())
        assert preflight_target("tensorrt") == "tensorrt"


# ── stats ──────────────────────────────────────────────────────────────────

class TestSummarize:
    def test_percentiles_are_nearest_rank(self):
        s = summarize([float(i) for i in range(1, 101)], batch_size=1)
        assert s["ms_per_batch_p50"] == 50.0
        assert s["ms_per_batch_p95"] == 95.0
        assert s["ms_per_batch_p99"] == 99.0

    def test_per_clip_divides_by_batch(self):
        s = summarize([10.0] * 10, batch_size=4)
        assert s["ms_per_batch_p99"] == 10.0
        assert s["ms_per_clip_p99"] == 2.5

    def test_tail_is_never_below_the_median(self):
        s = summarize([5.0, 100.0, 6.0, 7.0], batch_size=1)
        assert s["ms_per_batch_p50"] <= s["ms_per_batch_p95"] <= s["ms_per_batch_p99"]


# ── ONNX ───────────────────────────────────────────────────────────────────

class TestOnnxExport:
    def test_exports_a_loadable_graph(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        wrapper = build_wrapper(**ARCH, device="cpu")
        path = export_onnx(wrapper, str(tmp_path / "m.onnx"),
                           num_frames=ARCH["num_frames"], image_size=IMAGE_SIZE)
        assert os.path.exists(path)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        assert [i.name for i in model.graph.input] == ["clip"]
        assert [o.name for o in model.graph.output] == ["logits"]

    def test_matches_the_torch_output(self, tmp_path):
        pytest.importorskip("onnx")
        ov = pytest.importorskip("openvino")
        wrapper = build_wrapper(**ARCH, device="cpu")
        path = export_onnx(wrapper, str(tmp_path / "m.onnx"),
                           num_frames=ARCH["num_frames"], image_size=IMAGE_SIZE)

        x = torch.randn(1, ARCH["num_frames"], 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            expected = wrapper(x).numpy()

        compiled = ov.Core().compile_model(ov.Core().read_model(path), "CPU")
        got = compiled({0: x.numpy()})[compiled.output(0)]
        # The exported graph must compute the same thing, or the latency number
        # describes a model that isn't the one being trained.
        assert abs(expected - got).max() < 1e-3


# ── OpenVINO (CPU) ─────────────────────────────────────────────────────────

class TestOpenVino:
    def test_benchmark_reports_tail_latency_and_runtime(self):
        pytest.importorskip("onnx")
        pytest.importorskip("openvino")
        r = benchmark_optimized(target="openvino", iters=3, warmup=1,
                                image_size=IMAGE_SIZE, **ARCH)
        assert r["runtime"] == "openvino"
        assert r["device"] == "cpu"
        assert 0 < r["ms_per_batch_p50"] <= r["ms_per_batch_p99"]
        assert r["clips_per_sec"] > 0
        assert r["variant"] == "b0"

    def test_export_writes_ir_alongside_onnx(self, tmp_path, monkeypatch):
        pytest.importorskip("onnx")
        pytest.importorskip("openvino")

        # export_optimized loads a checkpoint; stub that out and hand it the
        # same untrained architecture the other tests use.
        monkeypatch.setattr(optimized_runtime, "build_wrapper",
                            lambda *a, **k: build_wrapper(**ARCH, device="cpu"))
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"\x00")

        res = export_optimized(str(ckpt), str(tmp_path / "out"), "m",
                               fmt="openvino", num_frames=ARCH["num_frames"],
                               image_size=IMAGE_SIZE)
        assert res["format"] == "openvino"
        assert res["output_path"].endswith(".xml")
        # the ONNX is kept so the rig can be re-targeted without the .ckpt
        assert any(f.endswith(".onnx") for f in res["files"])
        for f in res["files"]:
            assert os.path.exists(f)


# ── TensorRT (GPU) ─────────────────────────────────────────────────────────

class TestTensorRT:
    @pytest.mark.skipif(not _trt_capable_gpu(),
                        reason="needs a CUDA device with compute capability >= 7.5")
    def test_benchmark_reports_tail_latency(self):
        pytest.importorskip("onnx")
        pytest.importorskip("tensorrt")
        r = benchmark_optimized(target="tensorrt", iters=3, warmup=1,
                                image_size=IMAGE_SIZE, **ARCH)
        assert r["runtime"] == "tensorrt"
        assert r["device"] == "cuda"
        assert 0 < r["ms_per_batch_p50"] <= r["ms_per_batch_p99"]


# ── validation ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_unknown_export_format(self, tmp_path):
        with pytest.raises(ValueError):
            export_optimized("x.ckpt", str(tmp_path), fmt="coreml")

    def test_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_optimized(str(tmp_path / "nope.ckpt"), str(tmp_path), fmt="onnx")

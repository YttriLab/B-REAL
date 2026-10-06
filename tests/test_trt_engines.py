"""Tests for automatic TensorRT packaging of the localizer and classifier
(trt_engines.py), and the TensorRT-11-compatible build path it shares with
optimized_runtime."""

import os

import numpy as np
import pytest
import torch

import trt_engines


def _trt_ok():
    return trt_engines.tensorrt_available()[0]


needs_trt = pytest.mark.skipif(not _trt_ok(), reason="needs TensorRT and a supported GPU")


# ── runtime choice (no GPU needed) ─────────────────────────────────────────

class TestResolveRuntime:
    def test_torch_is_taken_as_is(self):
        assert trt_engines.resolve_runtime("torch")[0] == "torch"

    def test_unknown(self):
        with pytest.raises(ValueError):
            trt_engines.resolve_runtime("onnxruntime")

    def test_auto_falls_back_without_a_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rt, reason = trt_engines.resolve_runtime("auto")
        assert rt == "torch" and "TensorRT unusable" in reason

    def test_explicit_tensorrt_raises_without_a_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="TensorRT"):
            trt_engines.resolve_runtime("tensorrt")

    def test_cache_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BREAL_TRT_CACHE", str(tmp_path))
        assert trt_engines.cache_dir() == str(tmp_path)


# ── export wrappers (CPU) ──────────────────────────────────────────────────

def _tsm(pool="last", T=3):
    from model_tsm import TSMClassifier
    torch.manual_seed(0)
    return TSMClassifier(num_frames=T, variant="b0", pretrained=False, tsm_stages=[6, 7],
                         temporal_pool=pool, class_names=["a", "b", "c"]).eval()


@pytest.mark.parametrize("pool", ["last", "avg", "max"])
def test_torchscript_wrapper_honours_temporal_pool(pool):
    """The deployment wrapper used to take the last frame for every model."""
    from convert_torchscript_tsm import TSMInferenceWrapper
    m = _tsm(pool)
    x = torch.randn(2, 3, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(TSMInferenceWrapper(m)(x), m(x), atol=1e-6)


def test_concat_temporal_shift_matches_the_original():
    import copy
    import types
    from model_tsm import TemporalShift
    for uni in (False, True):
        ts = TemporalShift(num_frames=3, unidirectional=uni)
        alt = copy.deepcopy(ts)
        alt.forward = types.MethodType(trt_engines._temporal_shift_concat, alt)
        x = torch.randn(6, 16, 4, 4)
        assert torch.equal(ts(x), alt(x))


# ── engines (GPU + TensorRT) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def trt_cache(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("trt_cache"))
    old = os.environ.get("BREAL_TRT_CACHE")
    os.environ["BREAL_TRT_CACHE"] = path
    yield path
    if old is None:
        os.environ.pop("BREAL_TRT_CACHE")
    else:
        os.environ["BREAL_TRT_CACHE"] = old


@pytest.fixture(scope="module")
def localizer_weights(tmp_path_factory):
    from localizer import Localizer
    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("loc") / "loc.pt")
    torch.save(Localizer(pretrained=False).state_dict(), path)
    return path


@pytest.fixture(scope="module")
def tsm_and_ckpt(tmp_path_factory):
    """A random TSM model and a file standing in for its checkpoint (the cache
    key hashes the file; the model itself is passed in)."""
    path = tmp_path_factory.mktemp("tsm") / "model.ckpt"
    path.write_bytes(os.urandom(64))
    return _tsm(), str(path)


@needs_trt
class TestLocalizerEngine:
    def test_matches_torch_and_is_cached(self, trt_cache, localizer_weights):
        from localizer import IN_H, IN_W, load_localizer
        eng = trt_engines.localizer_engine(localizer_weights, precision="fp16", max_batch=8)
        mtime = os.path.getmtime(eng.path)
        x = torch.randint(0, 256, (11, IN_H, IN_W), dtype=torch.uint8, device="cuda")
        xy, pres = eng(x)  # 11 > max batch 8: chunked
        ref = load_localizer(localizer_weights, "cuda")
        with torch.no_grad():
            rxy, _, rpres = ref(x)
        assert xy.shape == (11, 2) and pres.shape == (11,)
        assert xy.dtype == torch.float32
        assert (xy - rxy).abs().max() < 2.0  # input px (352x384), fp16 trunk
        assert (torch.sigmoid(pres) - torch.sigmoid(rpres)).abs().max() < 0.05
        again = trt_engines.localizer_engine(localizer_weights, precision="fp16", max_batch=8)
        assert again.path == eng.path and os.path.getmtime(again.path) == mtime

    def test_cropper_uses_the_engine(self, trt_cache, localizer_weights):
        from localizer import LocalizerCropper
        c = LocalizerCropper(localizer_weights, runtime="auto", max_batch=8)
        assert c.engine is not None
        frames = [np.random.randint(0, 255, (200, 240), np.uint8) for _ in range(3)]
        centers, pres = c.locate(frames)
        assert centers.shape == (3, 2) and pres.shape == (3,)
        _, _, heat = c.locate(frames, return_heatmap=True)  # falls back to torch for this
        assert heat.shape[0] == 3


@needs_trt
class TestTsmEngine:
    @pytest.mark.parametrize("precision,tol", [("fp32", 1e-3), ("fp16", 3e-2)])
    def test_matches_torch(self, trt_cache, tsm_and_ckpt, precision, tol):
        model, ckpt = tsm_and_ckpt
        eng = trt_engines.tsm_engine(model, ckpt, 3, 64, precision=precision, max_batch=4)
        assert eng.meta["class_names"] == ["a", "b", "c"]
        x = torch.randn(6, 3, 3, 64, 64, device="cuda")
        with torch.no_grad():
            ref = torch.softmax(model.cuda()(x), 1)
        got = torch.softmax(eng(x), 1)
        model.cpu()
        assert (got - ref).abs().max() < tol
        assert (got.argmax(1) == ref.argmax(1)).float().mean() >= 5 / 6

    def test_inference_service_runs_on_the_engine(self, trt_cache, tsm_and_ckpt, monkeypatch,
                                                  synthetic_video, synthetic_background):
        """End to end: runtime="tensorrt" gives what PyTorch gives."""
        import model_tsm
        from backend.services.inference_service import run_inference_video
        model, ckpt = tsm_and_ckpt
        monkeypatch.setattr(model_tsm.TSMClassifier, "load_from_checkpoint",
                            staticmethod(lambda *a, **k: model))
        kw = dict(frames_per_group=3, image_size=64, decoder="cpu", batch_size=4)
        params = {"box_size": 24, "difference_threshold": 30}
        video, _ = synthetic_video
        trt_res = run_inference_video(ckpt, video, synthetic_background, dict(params),
                                      runtime="tensorrt", precision="fp32", **kw)
        torch_res = run_inference_video(ckpt, video, synthetic_background, dict(params),
                                        runtime="torch", **kw)
        model.cpu()
        a = np.array([f["predictions"] for f in trt_res["frames"]])
        b = np.array([f["predictions"] for f in torch_res["frames"]])
        assert a.shape == b.shape and np.abs(a - b).max() < 1e-3


@needs_trt
def test_fp16_deployment_benchmark_builds_on_strongly_typed_tensorrt():
    """TensorRT 11 removed the FP16 builder flag; fp16 now comes from the graph."""
    from optimized_runtime import benchmark_optimized
    r = benchmark_optimized(target="tensorrt", precision="fp16", iters=3, warmup=1,
                            image_size=64, variant="b0", num_frames=3, tsm_stages=[6, 7])
    assert r["precision"] == "fp16" and r["ms_per_batch_p50"] > 0

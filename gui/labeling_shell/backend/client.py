"""HTTP client for the FastAPI backend.

One method per endpoint the Electron renderer used (see core/*.js and
plugins/pipeline.js at the electron-final tag), matching the same
request/response contract so the backend
needs no changes. Long-running jobs (preprocess/train/inference) return a
task_id; progress is consumed by iterating the corresponding *_status generator,
which parses the backend's Server-Sent Events into plain dicts.
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Optional

import httpx


class BackendError(RuntimeError):
    """A backend request failed (transport error or non-2xx response)."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _extract_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(data, dict) and "detail" in data:
        d = data["detail"]
        return d if isinstance(d, str) else json.dumps(d)
    return json.dumps(data)


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BackendClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- low-level
    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            resp = self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise BackendError(f"{method} {path} failed: {e}") from e
        if resp.status_code >= 400:
            detail = _extract_detail(resp)
            raise BackendError(
                f"{method} {path} -> {resp.status_code}: {detail}",
                status_code=resp.status_code, detail=detail,
            )
        return resp.json() if resp.content else {}

    def _post(self, path: str, payload: Optional[dict] = None) -> dict:
        return self._request("POST", path, json=payload or {})

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", path, params=params)

    def _stream(self, method: str, path: str, payload: Optional[dict] = None) -> Iterator[dict]:
        """Yield parsed dicts from an SSE endpoint until the server closes it."""
        try:
            with self._client.stream(method, path, json=payload) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    detail = _extract_detail(resp)
                    raise BackendError(
                        f"{method} {path} -> {resp.status_code}: {detail}",
                        status_code=resp.status_code, detail=detail,
                    )
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw:
                            yield json.loads(raw)
        except httpx.HTTPError as e:
            raise BackendError(f"{method} {path} stream failed: {e}") from e

    # --------------------------------------------------------------- health
    def health(self) -> dict:
        return self._get("/api/health")

    def wait_healthy(self, timeout: float = 30.0, interval: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.health().get("status") == "ok":
                    return True
            except BackendError:
                pass
            time.sleep(interval)
        return False

    # ----------------------------------------------------------- preprocess
    def compute_background(self, video_path: str, num_frames: int = 500,
                           output_dir: Optional[str] = None) -> dict:
        return self._post("/api/preprocess/background", {
            "video_path": video_path, "num_frames": num_frames, "output_dir": output_dir,
        })

    def preview_frame(self, video_path: str, bg_path: Optional[str], frame_index: int,
                      crop_params: Optional[dict] = None,
                      prev_center: Optional[list] = None) -> dict:
        body = {"video_path": video_path, "bg_path": bg_path,
                "frame_index": frame_index, "prev_center": prev_center}
        if crop_params is not None:  # omit so the server applies its CropParams() default
            body["crop_params"] = crop_params
        return self._post("/api/preprocess/preview", body)

    def run_preprocessing(self, video_path: str, output_dir: str,
                          crop_params: Optional[dict] = None,
                          frame_ranges: Optional[list] = None,
                          bg_path: Optional[str] = None, num_frames_bg: int = 500,
                          output_name: Optional[str] = None,
                          crop: bool = True, decoder: str = "auto") -> dict:
        """`crop=False` extracts whole frames and skips background subtraction.
        `decoder`: "auto" | "nvdec" | "cpu"."""
        body = {"video_path": video_path, "output_dir": output_dir,
                "frame_ranges": frame_ranges, "bg_path": bg_path,
                "num_frames_bg": num_frames_bg, "output_name": output_name,
                "crop": crop, "decoder": decoder}
        if crop_params is not None:
            body["crop_params"] = crop_params
        return self._post("/api/preprocess/run", body)

    # ------------------------------------------------------------- localizer
    def localizer_info(self) -> dict:
        return self._get("/api/localizer/info")

    def localizer_seed_preview(self, video_path: str, frame: int, x: Optional[float] = None,
                               y: Optional[float] = None, sam2_model: str = "tiny") -> dict:
        """The frame, plus SAM2's mask for a click at (x, y) native px. The
        first call loads (and may download) SAM2, hence the long timeout."""
        return self._request("POST", "/api/localizer/seed-preview", timeout=900.0, json={
            "video_path": video_path, "frame": frame, "x": x, "y": y, "sam2_model": sam2_model})

    def train_localizer(self, request: dict) -> dict:
        return self._post("/api/localizer/train", request)

    def localizer_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/localizer/status/{task_id}")

    def stop_localizer(self, task_id: str) -> dict:
        return self._post(f"/api/localizer/stop/{task_id}")

    def preprocess_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/preprocess/status/{task_id}")

    def stop_preprocessing(self, task_id: str) -> dict:
        return self._post(f"/api/preprocess/stop/{task_id}")

    # ------------------------------------------------------------- training
    def start_training(self, config: dict) -> dict:
        """`config` is the flat TrainingConfig dict (label_paths, frame_dirs, …)."""
        return self._post("/api/training/start", config)

    def training_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/training/status/{task_id}")

    def stop_training(self, task_id: str) -> dict:
        return self._post(f"/api/training/stop/{task_id}")

    def check_coverage(self, frame_dirs: list, label_paths: list,
                       frames_per_group: int = 3) -> dict:
        return self._post("/api/training/check-coverage", {
            "frame_dirs": frame_dirs, "label_paths": label_paths,
            "frames_per_group": frames_per_group,
        })

    def list_checkpoints(self, search_dirs: Optional[list] = None) -> dict:
        params = {"search_dirs": ",".join(search_dirs)} if search_dirs else None
        return self._get("/api/training/checkpoints", params=params)

    def export_torchscript(self, checkpoint_path: str, output_dir: str,
                           output_name: str = "tsm_model",
                           tsm_stages: Optional[list] = None) -> dict:
        return self._post("/api/training/export-torchscript", {
            "checkpoint_path": checkpoint_path, "output_dir": output_dir,
            "output_name": output_name, "tsm_stages": tsm_stages,
        })

    def export_optimized(self, checkpoint_path: str, output_dir: str,
                         output_name: str = "tsm_model", fmt: str = "onnx",
                         tsm_stages: Optional[list] = None,
                         num_frames: Optional[int] = None, image_size: int = 224,
                         batch_size: int = 1, precision: str = "fp32") -> dict:
        """Export to a deployment format (onnx | openvino | tensorrt)."""
        return self._post("/api/training/export-optimized", {
            "checkpoint_path": checkpoint_path, "output_dir": output_dir,
            "output_name": output_name, "format": fmt, "tsm_stages": tsm_stages,
            "num_frames": num_frames, "image_size": image_size,
            "batch_size": batch_size, "precision": precision,
        })

    def benchmark_inference(self, **kw) -> dict:
        """Kwargs mirror BenchmarkRequest (variant, num_frames, tsm_stages, …,
        target: auto | openvino | tensorrt | torchscript)."""
        return self._post("/api/training/benchmark-inference", kw)

    # --------------------------------------------------------------- tuning
    def start_tuning(self, request: dict) -> dict:
        """`request` is the TuningRequest dict: {base, n_trials, objective_metric,
        per-parameter specs, latency}."""
        return self._post("/api/tuning/start", request)

    def tuning_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/tuning/status/{task_id}")

    def stop_tuning(self, task_id: str) -> dict:
        return self._post(f"/api/tuning/stop/{task_id}")

    # ------------------------------------------------------------ inference
    def run_inference(self, checkpoint_path: str, frame_dir: Optional[str] = None,
                      video_path: Optional[str] = None, bg_path: Optional[str] = None,
                      crop_params: Optional[dict] = None, frames_per_group: int = 3,
                      image_size: int = 224, extra_model_config: Optional[dict] = None,
                      frame_ranges: Optional[list] = None, decoder: str = "auto",
                      batch_size: int = 32, runtime: str = "auto",
                      precision: str = "fp16") -> dict:
        return self._post("/api/inference/run", {
            "checkpoint_path": checkpoint_path, "frame_dir": frame_dir,
            "video_path": video_path, "bg_path": bg_path, "crop_params": crop_params,
            "frames_per_group": frames_per_group, "image_size": image_size,
            "extra_model_config": extra_model_config, "frame_ranges": frame_ranges,
            "decoder": decoder, "batch_size": batch_size,
            "runtime": runtime, "precision": precision,
        })

    def inference_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/inference/status/{task_id}")

    def inference_results(self, task_id: str) -> dict:
        return self._get(f"/api/inference/results/{task_id}")

    def stop_inference(self, task_id: str) -> dict:
        return self._post(f"/api/inference/stop/{task_id}")

    # -------------------------------------------------------------- project
    def export_rlt(self, project_dir: str, output_path: str, package: bool = False) -> dict:
        return self._post("/api/project/export-rlt", {
            "project_dir": project_dir, "output_path": output_path, "package": package,
        })

    def import_rlt(self, rlt_path: str, output_dir: str) -> dict:
        return self._post("/api/project/import-rlt", {
            "rlt_path": rlt_path, "output_dir": output_dir,
        })

    def inspect_rlt(self, rlt_path: str) -> dict:
        return self._post("/api/project/inspect-rlt", {"rlt_path": rlt_path})

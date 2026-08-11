"""Phase 1: BackendClient against a stdlib stub server (no torch / FastAPI).

Validates the HTTP + SSE contract the client speaks, plus interpreter
resolution. The real backend needs the CUDA/torch pixi env; this exercises the
*client* seam in isolation so it runs in the gui pixi env.

    pixi run -e gui python gui/tests/test_backend_client.py
    pytest gui/tests/test_backend_client.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.backend.client import BackendClient, BackendError  # noqa: E402
from labeling_shell.backend.config import BackendConfig  # noqa: E402
from labeling_shell.backend.process import resolve_python  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._json(200, {"status": "ok", "device": {"cuda_available": False}})
        elif self.path.startswith("/api/preprocess/status/"):
            self._sse([
                {"status": "running", "progress": 1, "total": 3, "error": None, "output_dir": None},
                {"status": "running", "progress": 2, "total": 3, "error": None, "output_dir": None},
                {"status": "completed", "progress": 3, "total": 3, "error": None, "output_dir": "out"},
            ])
        elif self.path.startswith("/api/training/checkpoints"):
            self._json(200, {"checkpoints": [], "query": self.path})
        elif self.path == "/api/boom":
            self._json(500, {"detail": "boom"})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n)) if n else {}
        if self.path == "/api/preprocess/run":
            self._json(200, {"task_id": "t1", "echo": body})
        else:
            self._json(200, {"ok": True, "path": self.path, "echo": body})


class _StubServer:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"


def test_json_roundtrip_and_health():
    with _StubServer() as srv, BackendClient(srv.base_url) as client:
        assert client.health()["status"] == "ok"
        assert client.wait_healthy(timeout=3) is True
        out = client.run_preprocessing("v.mp4", "out_dir", crop_params={"box_size": 200})
        assert out["task_id"] == "t1"
        assert out["echo"]["video_path"] == "v.mp4"
        assert out["echo"]["crop_params"] == {"box_size": 200}


def test_sse_stream_parses_events():
    with _StubServer() as srv, BackendClient(srv.base_url) as client:
        events = list(client.preprocess_status("t1"))
        assert [e["progress"] for e in events] == [1, 2, 3]
        assert events[-1]["status"] == "completed"
        assert events[-1]["output_dir"] == "out"


def test_query_params_and_omitted_crop_params():
    with _StubServer() as srv, BackendClient(srv.base_url) as client:
        res = client.list_checkpoints(["a", "b"])
        assert "search_dirs=a%2Cb" in res["query"] or "search_dirs=a,b" in res["query"]
        # crop_params omitted -> not sent, so the server default applies
        echo = client.preview_frame("v.mp4", "bg.png", 5)["echo"]
        assert "crop_params" not in echo
        assert echo["prev_center"] is None


def test_error_surfaces_detail():
    with _StubServer() as srv, BackendClient(srv.base_url) as client:
        try:
            client._get("/api/boom")
        except BackendError as e:
            assert e.status_code == 500
            assert e.detail == "boom"
        else:
            raise AssertionError("expected BackendError")


def test_transport_error_is_backend_error():
    # Nothing listening on this port -> connection refused -> BackendError.
    with BackendClient("http://127.0.0.1:9") as client:
        try:
            client.health()
        except BackendError:
            pass
        else:
            raise AssertionError("expected BackendError")


def test_resolve_python_precedence(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    cfg = BackendConfig(repo_root=str(d))
    saved = {k: os.environ.get(k) for k in ("RTASOID_PYTHON", "PYTHON")}
    try:
        os.environ["RTASOID_PYTHON"] = "X_INTERP"
        assert resolve_python(cfg) == "X_INTERP"
        os.environ.pop("RTASOID_PYTHON")
        os.environ.pop("PYTHON", None)
        cfg2 = BackendConfig(repo_root=str(d), python="Y_INTERP")
        assert resolve_python(cfg2) == "Y_INTERP"
        # no override, no pixi env present -> bare interpreter name
        assert resolve_python(cfg) in ("python", "python3")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

"""Local backend subprocess management.

Ports the Electron main.js backend lifecycle (`resolveBackendPython`,
`startBackend`, `stopBackend`) to Python: pick an interpreter, spawn
`uvicorn backend.server:app`, and drain its output so it neither deadlocks on a
full pipe nor vanishes silently.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

from .config import BackendConfig

LogCallback = Callable[[str], None]


def _pixi_env_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".pixi" / "envs" / "default" / "python.exe"
    return root / ".pixi" / "envs" / "default" / "bin" / "python"


def resolve_python(config: BackendConfig) -> str:
    """Choose the interpreter to run the backend.

    Resolution order (mirrors the Electron app):
      1. BREAL_PYTHON / PYTHON env override.
      2. config.python explicit override.
      3. Project-local pixi env at <repo>/.pixi/envs/default.
      4. Bare python / python3 on PATH.
    """
    override = os.environ.get("BREAL_PYTHON") or os.environ.get("PYTHON")
    if override and override.strip():
        return override.strip()
    if config.python:
        return config.python
    pixi = _pixi_env_python(config.repo_root)
    if pixi.exists():
        return str(pixi)
    return "python" if os.name == "nt" else "python3"


class BackendProcess:
    """A spawned local uvicorn process running the FastAPI backend."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, on_log: LogCallback | None = None) -> None:
        if self.running:
            return
        python = resolve_python(self.config)
        cmd = [
            python, "-m", "uvicorn", self.config.module,
            "--port", str(self.config.port), "--host", self.config.host,
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.config.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(
            target=self._drain, args=(on_log,), daemon=True
        )
        self._log_thread.start()

    def _drain(self, on_log: LogCallback | None) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            if on_log is not None:
                on_log(line)
            else:
                print(f"[backend] {line}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

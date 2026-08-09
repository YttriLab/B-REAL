"""Backend manager: ties together the HTTP client and (when local) the spawned
uvicorn process behind one object.

The GUI creates one `Backend` from a `BackendConfig` and thereafter uses only
`backend.client`. Whether the backend was spawned locally or lives on a remote
rig is invisible past this point.
"""

from __future__ import annotations

from typing import Callable, Optional

from .client import BackendClient
from .config import BackendConfig
from .process import BackendProcess


class Backend:
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self.client = BackendClient(self.config.base_url, timeout=self.config.request_timeout)
        self.process = BackendProcess(self.config) if self.config.spawn_local else None

    def start(self, wait: bool = True, timeout: float = 30.0,
              on_log: Optional[Callable[[str], None]] = None) -> bool:
        """Spawn the local backend (if configured) and optionally wait for health.

        Returns True if the backend is reachable (or wait=False). For a remote
        backend nothing is spawned; we just health-check.
        """
        if self.process is not None:
            self.process.start(on_log=on_log)
        return self.client.wait_healthy(timeout=timeout) if wait else True

    def stop(self) -> None:
        if self.process is not None:
            self.process.stop()
        self.client.close()

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

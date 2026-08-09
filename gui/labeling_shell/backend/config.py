"""Backend connection config.

The GUI talks to the FastAPI backend over HTTP either by spawning a local
uvicorn (single-machine install) or by connecting to a remote one (e.g. the
3090 rig, with the labeling GUI running on a lighter box). Which of the two is
purely a config decision here; callers only ever use the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = 8742
DEFAULT_MODULE = "backend.server:app"


def repo_root() -> Path:
    """RealtimeASOID repo root (…/gui/labeling_shell/backend/config.py -> repo)."""
    return Path(__file__).resolve().parents[3]


@dataclass
class BackendConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    spawn_local: bool = True          # True: spawn local uvicorn; False: connect to a remote backend
    python: str | None = None         # interpreter override; None -> auto-resolve (see process.resolve_python)
    repo_root: Path | None = None     # cwd for the spawned backend; defaults to the repo root
    module: str = DEFAULT_MODULE
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root) if self.repo_root is not None else repo_root()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def remote(cls, host: str, port: int = DEFAULT_PORT, **kw) -> "BackendConfig":
        """Config for a backend running on another machine (no local spawn)."""
        return cls(host=host, port=port, spawn_local=False, **kw)

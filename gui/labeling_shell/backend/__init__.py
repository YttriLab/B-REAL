"""labeling_shell.backend — the GUI's seam to the FastAPI/PyTorch backend.

Named under `labeling_shell` (not top-level `backend`) so it never collides with
the repo's own `backend/` FastAPI package on sys.path.
"""

from .client import BackendClient, BackendError
from .config import BackendConfig
from .manager import Backend
from .process import BackendProcess, resolve_python

__all__ = [
    "Backend",
    "BackendClient",
    "BackendConfig",
    "BackendError",
    "BackendProcess",
    "resolve_python",
]

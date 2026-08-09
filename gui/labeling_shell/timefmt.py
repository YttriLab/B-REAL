"""Frame <-> time helpers, matching the Electron app's conventions exactly.

Time strings are ``MM:SS:mmm`` (colon before milliseconds), the format the JS
`formatTime`/`parseTime` use and therefore the format existing bout CSVs are in.
Frame<->time uses ``frame * (1000/fps)`` deliberately: see video_reader for why
(B-frame CTS is non-monotonic; indexing by frame keeps stepping exact).
"""

from __future__ import annotations

from typing import Optional


def format_time(ms: float) -> str:
    """Milliseconds -> ``MM:SS:mmm`` (mirrors JS formatTime)."""
    total = max(0, round(ms))
    minutes, rem = divmod(total, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes:02d}:{seconds:02d}:{millis:03d}"


def parse_time(text: str) -> Optional[float]:
    """Parse ``MM:SS``, ``MM:SS:mmm`` or ``HH:MM:SS:mmm`` -> ms (mirrors JS parseTime).

    Returns None on anything unparseable, so callers can skip bad rows.
    """
    if text is None:
        return None
    parts = text.strip().split(":")
    if not (2 <= len(parts) <= 4):
        return None
    try:
        if len(parts) == 4:
            h, m, s, ms = (int(p) for p in parts)
        elif len(parts) == 3:  # MM:SS:mmm
            h = 0
            m, s, ms = (int(p) for p in parts)
        else:  # MM:SS
            h = 0
            m, s = int(parts[0]), int(parts[1])
            ms = 0
    except ValueError:
        return None
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def frame_to_ms(frame: int, fps: float) -> float:
    return frame * (1000.0 / fps)


def ms_to_frame(ms: float, fps: float) -> int:
    return int(round(ms / (1000.0 / fps)))

"""CSV import/export for bouts and label matrices.

Python successor to `core/csv.js` + the CSV generators in `core/videos.js`. The
formats are kept byte-compatible with the Electron app and with what
`model.py::load_data` consumes, so projects interoperate across the two frontends:

  * Bout CSV        -- ``start,end,behavior`` with MM:SS:mmm times (human-facing).
  * Training labels -- ``time,<beh1>,<beh2>,...`` where ``time`` is the NATIVE
                       frame index and each row is one-hot; load_data takes argmax
                       per row. Sparse: only frames inside a bout get a row.
  * One-hot export  -- dense variant (minFrame..maxFrame) with an optional fill.
"""

from __future__ import annotations

from typing import Optional

from .bouts import Bout, LabelModel
from .timefmt import format_time, frame_to_ms, ms_to_frame, parse_time


# ── low-level CSV helpers ────────────────────────────────────────────────
def csv_escape(value) -> str:
    s = "" if value is None else str(value)
    if any(c in s for c in (',', '"', '\n', '\r')):
        return '"' + s.replace('"', '""') + '"'
    return s


def parse_csv_line(line: str) -> list[str]:
    """RFC-4180-ish parser matching the JS parseCsvLine (handles quoted commas)."""
    fields: list[str] = []
    cur = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                cur.append('"')
                i += 1
            elif ch == '"':
                in_quotes = False
            else:
                cur.append(ch)
        else:
            if ch == '"':
                in_quotes = True
            elif ch == ',':
                fields.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        i += 1
    fields.append("".join(cur).strip())
    return fields


def _unique_behavior_names(bouts) -> list[str]:
    """Behavior names in first-appearance order."""
    names: list[str] = []
    seen = set()
    for b in bouts:
        if b.behavior_name not in seen:
            seen.add(b.behavior_name)
            names.append(b.behavior_name)
    return names


# ── export ───────────────────────────────────────────────────────────────
def generate_bouts_csv(bouts, fps: float) -> str:
    """Human-facing ``start,end,behavior`` CSV with MM:SS:mmm times."""
    header = "start,end,behavior"
    rows = [
        ",".join([
            csv_escape(format_time(frame_to_ms(b.start_frame, fps))),
            csv_escape(format_time(frame_to_ms(b.end_frame, fps))),
            csv_escape(b.behavior_name),
        ])
        for b in bouts
    ]
    return "\n".join([header, *rows])


def generate_training_label_csv(bouts) -> str:
    """Sparse frame-indexed one-hot matrix — the format load_data consumes.

    Only frames inside a bout get a row (lead-in/gap frames are context on disk,
    never labeled samples). Exactly one column is set per row.
    """
    names = _unique_behavior_names(bouts)
    if not names:
        return "time\n"
    col_of = {n: i for i, n in enumerate(names)}
    frame_to_col: dict[int, int] = {}
    for b in bouts:  # last-write-wins if bouts overlap (fine for argmax one-hot)
        c = col_of[b.behavior_name]
        for f in range(b.start_frame, b.end_frame + 1):
            frame_to_col[f] = c
    header = ",".join(["time", *[csv_escape(n) for n in names]])
    rows = []
    for f in sorted(frame_to_col):
        cols = ["0"] * len(names)
        cols[frame_to_col[f]] = "1"
        rows.append(str(f) + "," + ",".join(cols))
    return "\n".join([header, *rows])


def generate_onehot_csv(bouts, fill_behavior: Optional[str] = None) -> str:
    """Dense one-hot matrix minFrame..maxFrame, with an optional fill behavior for
    otherwise-unlabeled frames (mirrors exportOneHot / fill_missing_frames_onehot)."""
    if not bouts:
        return "time\n"
    max_frame = max(b.end_frame for b in bouts)
    names = _unique_behavior_names(bouts)
    fill = (fill_behavior or "").strip()
    if fill and fill not in names:
        names.append(fill)
    fill_idx = names.index(fill) if fill else -1
    min_frame = min((b.start_frame for b in bouts)) if fill_idx >= 0 else 0

    header = ",".join(["time", *[csv_escape(n) for n in names]])
    rows = []
    for f in range(min_frame, max_frame + 1):
        cols = [0] * len(names)
        has_label = False
        for b in bouts:
            if b.start_frame <= f <= b.end_frame:
                cols[names.index(b.behavior_name)] = 1
                has_label = True
        if not has_label and fill_idx >= 0:
            cols[fill_idx] = 1
        rows.append(str(f) + "," + ",".join(str(c) for c in cols))
    return "\n".join([header, *rows])


# ── import ───────────────────────────────────────────────────────────────
def _looks_like_bout_header(header_lower: str) -> bool:
    return "start" in header_lower and "end" in header_lower and "behav" in header_lower


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def parse_csv_and_import(content: str, model: LabelModel, fps: float) -> int:
    """Detect bout CSV vs frame-level matrix, import into `model`. Returns count.

    Mirrors parseCsvAndImport: adds missing behaviors, inserts bouts sorted, and
    merges duplicate behaviors at the end.
    """
    lines = [l for l in content.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0

    header_fields = parse_csv_line(lines[0])
    header_lower = lines[0].lower()
    is_bout = _looks_like_bout_header(header_lower)
    is_matrix = ("time" in header_lower) or (_is_number(header_fields[0]) and len(header_fields) > 3)

    imported = 0

    if is_bout:
        for line in lines[1:]:
            fields = parse_csv_line(line)
            if len(fields) < 3:
                continue
            start_ms, end_ms, name = parse_time(fields[0]), parse_time(fields[1]), fields[2]
            if start_ms is None or end_ms is None or not name:
                continue
            sf, ef = ms_to_frame(start_ms, fps), ms_to_frame(end_ms, fps)
            if sf == ef:
                continue
            behavior = model.add_behavior(name)
            model.insert_bout_sorted(Bout(behavior.id, behavior.name, min(sf, ef), max(sf, ef)))
            imported += 1

    elif is_matrix:
        has_text_header = not _is_number(header_fields[0])
        if has_text_header:
            names = header_fields[1:]
            data_lines = lines[1:]
        else:
            names = [f"Behavior {i + 1}" for i in range(len(header_fields) - 1)]
            data_lines = lines
        imported += _import_matrix(data_lines, names, model)

    else:
        return 0

    if imported > 0:
        model.merge_duplicate_behaviors()
    return imported


def _import_matrix(data_lines, names, model: LabelModel) -> int:
    """Convert a frame-level one-hot matrix into contiguous-run bouts.

    Sparse matrices have no explicit 'off' row, so a gap in the frame index
    closes any open run at the previous present frame (mirrors the JS logic).
    """
    n = len(names)
    active = [False] * n
    starts = [0] * n
    prev_frame: Optional[int] = None
    imported = 0

    def close(b: int, end_frame: int):
        nonlocal imported
        behavior = model.add_behavior(names[b])
        model.insert_bout_sorted(Bout(behavior.id, behavior.name, starts[b], end_frame))
        imported += 1

    last_frame: Optional[int] = None
    for line in data_lines:
        fields = parse_csv_line(line)
        if not fields or not _is_number(fields[0]):
            continue
        frame_idx = int(float(fields[0]))
        last_frame = frame_idx
        if prev_frame is not None and frame_idx > prev_frame + 1:
            for b in range(n):
                if active[b]:
                    active[b] = False
                    close(b, prev_frame)
        prev_frame = frame_idx
        for b in range(n):
            val = float(fields[b + 1]) if b + 1 < len(fields) and fields[b + 1] != "" else 0.0
            on = val >= 0.5
            if on and not active[b]:
                active[b] = True
                starts[b] = frame_idx
            elif not on and active[b]:
                active[b] = False
                close(b, frame_idx - 1)

    for b in range(n):
        if active[b] and last_frame is not None:
            close(b, last_frame)
    return imported

"""Phase 2: CSV import/export (bout CSV, training matrix, one-hot) + round-trips.

    .venv-gui/Scripts/python.exe gui/tests/test_csv_io.py
    pytest gui/tests/test_csv_io.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labeling_shell.bouts import Bout, LabelModel  # noqa: E402
from labeling_shell.csv_io import (  # noqa: E402
    generate_bouts_csv, generate_onehot_csv, generate_training_label_csv,
    parse_csv_and_import, parse_csv_line,
)
from labeling_shell.timefmt import format_time, parse_time  # noqa: E402

FPS = 30.0


def _model_with_bouts():
    m = LabelModel()
    a = m.add_behavior("groom")
    b = m.add_behavior("rear")
    m.insert_bout_sorted(Bout(a.id, a.name, 5, 8))
    m.insert_bout_sorted(Bout(b.id, b.name, 12, 15))
    m.insert_bout_sorted(Bout(a.id, a.name, 20, 22))
    return m


def test_time_format_roundtrip():
    for ms in (0, 999, 1000, 61_500, 3_599_999):
        assert parse_time(format_time(ms)) == ms
    assert parse_time("01:30") == 90_000        # MM:SS
    assert parse_time("00:01:250") == 1250      # MM:SS:mmm
    assert parse_time("01:00:00:000") == 3_600_000  # HH:MM:SS:mmm
    assert parse_time("garbage") is None


def test_parse_csv_line_quotes():
    assert parse_csv_line('a,b,c') == ["a", "b", "c"]
    assert parse_csv_line('"a,b",c,"d""e"') == ["a,b", "c", 'd"e']


def test_bout_csv_roundtrip():
    m = _model_with_bouts()
    csv = generate_bouts_csv(m.bouts, FPS)
    assert csv.splitlines()[0] == "start,end,behavior"

    m2 = LabelModel()
    n = parse_csv_and_import(csv, m2, FPS)
    assert n == 3
    got = sorted((b.start_frame, b.end_frame, b.behavior_name) for b in m2.bouts)
    want = sorted((b.start_frame, b.end_frame, b.behavior_name) for b in m.bouts)
    assert got == want


def test_training_label_csv_shape_and_argmax():
    m = _model_with_bouts()
    csv = generate_training_label_csv(m.bouts)
    lines = csv.splitlines()
    assert lines[0] == "time,groom,rear"
    # sparse: only frames inside a bout (4+4+3 = 11 rows)
    assert len(lines) - 1 == 11
    # each row is one-hot; argmax over columns recovers the behavior
    header = lines[0].split(",")[1:]
    for row in lines[1:]:
        cells = row.split(",")
        onehot = [int(x) for x in cells[1:]]
        assert sum(onehot) == 1
        frame = int(cells[0])
        behavior = header[onehot.index(1)]
        assert (frame in range(5, 9) or frame in range(20, 23)) == (behavior == "groom")


def test_training_csv_reimports_to_same_frame_labels():
    """The training matrix, re-imported, yields the same frame->behavior mapping.

    This is the interop guarantee: what the GUI writes for load_data can be read
    back into identical bouts (contiguous runs), even across the bout/gap logic.
    """
    m = _model_with_bouts()
    csv = generate_training_label_csv(m.bouts)

    m2 = LabelModel()
    parse_csv_and_import(csv, m2, FPS)

    def frame_labels(model):
        out = {}
        for b in model.bouts:
            for f in range(b.start_frame, b.end_frame + 1):
                out[f] = b.behavior_name
        return out

    assert frame_labels(m2) == frame_labels(m)


def test_onehot_dense_with_fill():
    m = LabelModel()
    a = m.add_behavior("groom")
    m.insert_bout_sorted(Bout(a.id, a.name, 2, 3))
    m.insert_bout_sorted(Bout(a.id, a.name, 6, 7))
    csv = generate_onehot_csv(m.bouts, fill_behavior="other")
    lines = csv.splitlines()
    assert lines[0] == "time,groom,other"
    # dense from min(2)..max(7); unlabeled frames 4,5 get the fill class
    assert len(lines) - 1 == 6
    rows = {int(r.split(",")[0]): [int(x) for x in r.split(",")[1:]] for r in lines[1:]}
    assert rows[2] == [1, 0]  # groom
    assert rows[4] == [0, 1]  # fill
    assert rows[6] == [1, 0]


def test_matrix_import_sparse_gap_closes_runs():
    # sparse matrix: groom 0-2, then a gap, then rear 5-6
    csv = "time,groom,rear\n0,1,0\n1,1,0\n2,1,0\n5,0,1\n6,0,1\n"
    m = LabelModel()
    n = parse_csv_and_import(csv, m, FPS)
    assert n == 2
    got = sorted((b.start_frame, b.end_frame, b.behavior_name) for b in m.bouts)
    assert got == [(0, 2, "groom"), (5, 6, "rear")]


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

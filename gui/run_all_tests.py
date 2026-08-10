"""Run every gui/tests/test_*.py suite, each in its own subprocess, and
aggregate results.

    C:/AdenCode/RealtimeASOID/.venv-gui/Scripts/python.exe gui/run_all_tests.py

Each suite is self-contained (prints its own PASS/FAIL and raises SystemExit with
a nonzero code on failure). Qt runs offscreen so no display is needed.

Runs each file as a separate process rather than in-process (runpy) in the same
interpreter: qt-material's stylesheet/resource registration, combined with
multiple MainWindow instances created and torn down across test files, produced
a delayed segfault when several suites shared one long-lived QApplication (the
crash surfaced in whichever suite happened to run next, unrelated to its own
content) — a real user only ever constructs one QApplication per process, so
this is purely a test-runner isolation issue, not an app bug. One process per
file sidesteps it and is more robust to any future cross-file Qt state anyway.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    files = sorted(glob.glob(os.path.join(HERE, "tests", "test_*.py")))
    env = {**os.environ, "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen")}
    failed = []
    for path in files:
        name = os.path.basename(path)
        print(f"\n===== {name} =====")
        result = subprocess.run([sys.executable, "-u", path], env=env)
        if result.returncode != 0:
            failed.append(f"{name} (exit {result.returncode})")
    print("\n" + "=" * 40)
    print("ALL SUITES PASS" if not failed else f"FAILED SUITES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

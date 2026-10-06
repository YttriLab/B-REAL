"""Run every gui/tests/test_*.py suite in one process and aggregate results.

    pixi run -e gui python gui/run_all_tests.py

Each suite is self-contained (prints its own PASS/FAIL and raises SystemExit with
a nonzero code on failure). Qt runs offscreen so no display is needed.
"""

from __future__ import annotations

import glob
import os
import runpy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    files = sorted(glob.glob(os.path.join(HERE, "tests", "test_*.py")))
    failed = []
    for path in files:
        name = os.path.basename(path)
        print(f"\n===== {name} =====")
        try:
            runpy.run_path(path, run_name="__main__")
        except SystemExit as e:
            if e.code:
                failed.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed.append(name)
    print("\n" + "=" * 40)
    print("ALL SUITES PASS" if not failed else f"FAILED SUITES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

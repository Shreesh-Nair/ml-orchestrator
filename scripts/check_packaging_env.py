"""Fail fast if the packaging environment is not compatible with PyInstaller builds.

Current short-term policy: package with numpy<2 to avoid ABI warnings from
compiled wheels used by SciPy/scikit-learn/PySide-related tooling.
"""

from __future__ import annotations

import re
import sys


def main() -> int:
    try:
        import numpy as np
    except Exception as exc:
        print(f"ERROR: unable to import numpy before packaging: {exc}")
        return 1

    version = getattr(np, "__version__", "0")
    match = re.match(r"^(\d+)\.", version)
    major = int(match.group(1)) if match else 0
    print(f"Detected numpy {version}")

    if major >= 2:
        print(
            "ERROR: PyInstaller packaging should run in an environment with numpy<2 "
            "to avoid ABI warnings from compiled wheels."
        )
        return 2

    print("Packaging environment looks compatible (numpy<2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

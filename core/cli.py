# core/cli.py
from __future__ import annotations

import sys
from pathlib import Path

from core.executor import run_pipeline


EXAMPLES_DIR = Path("examples")


def list_pipelines() -> None:
    print("Available pipelines:")
    for yml in sorted(EXAMPLES_DIR.glob("*.yml")):
        print(f"  - {yml.stem} ({yml})")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage:")
        print("  python -m core.cli list")
        print("  python -m core.cli run <pipeline_name_or_path>")
        raise SystemExit(0)

    cmd = argv[0]

    if cmd == "list":
        list_pipelines()
        return

    if cmd == "run":
        if len(argv) < 2:
            print("Usage: python -m core.cli run <pipeline_name_or_path>")
            raise SystemExit(1)

        name_or_path = argv[1]
        path = Path(name_or_path)

        if not path.exists():
            # Try examples/<name>.yml
            candidate = EXAMPLES_DIR / f"{name_or_path}.yml"
            if not candidate.exists():
                raise SystemExit(f"Pipeline not found: {name_or_path}")
            path = candidate

        run_pipeline(str(path))
        return

    raise SystemExit(f"Unknown command: {cmd!r}")


if __name__ == "__main__":
    main()

"""Apply quick-fix recommendations to example datasets and write cleaned CSVs.

Writes outputs to `build/cleaned_examples/` so CI can upload or inspect results.
Run with the same interpreter used by CI/packaging.
"""
import datetime
import json
import pathlib
import sys

import pandas as pd

from core.data_quality import analyze_data_quality, recommend_quick_fixes, apply_quick_fixes


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "cleaned_examples"
OUT.mkdir(parents=True, exist_ok=True)


def apply_to_csv(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    report = analyze_data_quality(df, target_column=None)
    rec = recommend_quick_fixes(report)
    fixed_df, actions = apply_quick_fixes(
        df,
        target_column=None,
        drop_constant_columns=rec.get("drop_constant_columns", False),
        drop_duplicate_rows=rec.get("drop_duplicate_rows", False),
        missing_strategy=rec.get("missing_strategy", "none"),
    )

    out_name = f"{csv_path.stem}_cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = OUT / out_name
    fixed_df.to_csv(out_path, index=False)
    return str(out_path), actions


def main():
    # discover example YAMLs that reference local CSVs
    examples_dir = ROOT / "examples" / "generated"
    produced = {}
    for y in examples_dir.glob("*.yml"):
        try:
            text = y.read_text(encoding="utf-8")
        except Exception:
            continue
        # crude parse for `source: ` path line
        src_line = None
        for line in text.splitlines():
            if "source:" in line:
                src_line = line.split("source:", 1)[1].strip()
                break
        if not src_line:
            continue
        src_path = pathlib.Path(src_line)
        if not src_path.exists():
            # try relative to repo
            src_path = ROOT / src_line
        if not src_path.exists():
            produced[y.name] = {"error": f"source not found: {src_line}"}
            continue

        out_path, actions = apply_to_csv(src_path)
        produced[y.name] = {"cleaned_csv": out_path, "actions": actions}

    out_file = ROOT / "build" / "recommender_examples_report.json"
    out_file.write_text(json.dumps(produced, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

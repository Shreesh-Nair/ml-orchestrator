"""
Scans installed locations for compiled extensions (.pyd, .dll, .so) for key packages
and writes a JSON report to build/compiled_extensions.json.
Run from the repo root with the same Python interpreter used for packaging.
"""
import importlib
import json
import pathlib
import sys

pkgs = ["numpy", "scipy", "sklearn", "pandas", "numexpr", "scipy.linalg", "skimage"]
report = {"python_executable": sys.executable, "rows": []}

for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", None)
        ppth = pathlib.Path(m.__file__).parent
        exts = list(ppth.glob("**/*.pyd")) + list(ppth.glob("**/*.dll")) + list(ppth.glob("**/*.so"))
        exts = [str(e) for e in exts]
        report["rows"].append({"package": p, "version": v, "path": str(ppth), "ext_count": len(exts), "exts": exts})
    except Exception as e:
        report["rows"].append({"package": p, "error": str(e)})

out = pathlib.Path("build")
out.mkdir(parents=True, exist_ok=True)
out_file = out / "compiled_extensions.json"
with out_file.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Wrote {out_file} (python: {sys.executable})")
print("Summary:")
for r in report["rows"]:
    if "error" in r:
        print(f" - {r['package']}: ERROR: {r['error']}")
    else:
        print(f" - {r['package']}: {r['ext_count']} extensions in {r['path']}")

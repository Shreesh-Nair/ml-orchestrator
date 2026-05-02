import os
from pathlib import Path
import csv
import random

# Small synthetic fraud dataset generator used for CI and local tests when the real dataset
# is not available (large files are intentionally gitignored).

OUT = Path(__file__).resolve().parents[1] / "data" / "fraud.csv"

if OUT.exists():
    print(f"Test dataset already exists: {OUT}")
    exit(0)

OUT.parent.mkdir(parents=True, exist_ok=True)

# Create a tiny dataset with numeric features and a binary Class column
fieldnames = ["V1", "V2", "V3", "V4", "Amount", "Class"]
rows = []
random.seed(42)
for i in range(500):
    # Generate mostly normal rows (Class=0) and a few frauds (Class=1)
    if random.random() < 0.02:
        cls = 1
        amount = round(random.uniform(1000, 10000), 2)
        v1 = round(random.uniform(-5, 5), 4)
        v2 = round(random.uniform(-5, 5), 4)
        v3 = round(random.uniform(-5, 5), 4)
        v4 = round(random.uniform(-5, 5), 4)
    else:
        cls = 0
        amount = round(random.uniform(0, 500), 2)
        v1 = round(random.uniform(-2, 2), 4)
        v2 = round(random.uniform(-2, 2), 4)
        v3 = round(random.uniform(-2, 2), 4)
        v4 = round(random.uniform(-2, 2), 4)
    rows.append({"V1": v1, "V2": v2, "V3": v3, "V4": v4, "Amount": amount, "Class": cls})

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote synthetic test dataset to {OUT} (rows={len(rows)})")

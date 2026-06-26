#!/usr/bin/env python3
"""Aggregate per-seed metrics into mean ± std.

Reads outputs/metrics/ (seed 42) + outputs/metrics_seed*/ and reports mean±std per model.
No deps beyond the stdlib + the JSONs evaluate.py writes.

  python experiments/scripts/aggregate_seeds.py
"""
import json
import statistics as st
from pathlib import Path

PROJ = Path("/home/40a2/DV/Code")
MODELS = ["Diamond-DETR", "RT-DETR", "YOLOv8n", "YOLO11n"]
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]

# seed dirs: base outputs/metrics is seed 42; outputs/metrics_seed<N> are the rest
seed_dirs = [PROJ / "outputs" / "metrics"] + sorted((PROJ / "outputs").glob("metrics_seed*"))
print(f"seed dirs: {[d.name for d in seed_dirs if d.exists()]}\n")

for model in MODELS:
    vals = {m: [] for m in METRICS}
    n = 0
    for d in seed_dirs:
        f = d / f"{model}.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        n += 1
        for m in METRICS:
            if j.get(m) is not None:
                vals[m].append(j[m])
    if n == 0:
        print(f"{model:14s}  (no results)"); continue
    cells = []
    for m in METRICS:
        v = vals[m]
        if len(v) >= 2:
            cells.append(f"{m}={st.mean(v):.3f}±{st.pstdev(v):.3f}")
        elif v:
            cells.append(f"{m}={v[0]:.3f}(n=1)")
    print(f"{model:14s}  n={n}  " + "  ".join(cells))

print("\n(±std over seeds; n=1 means only seed 42 so far — run multiseed.sh for more.)")

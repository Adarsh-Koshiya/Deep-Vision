#!/usr/bin/env python3
"""
evaluate.py, validate one trained model on the test split and dump metrics to JSON.

For DETR models (rtdetr) it inserts the bundled fork (code/diamond_detr/RTDETR-main, ultralytics
8.0.201) at the front of sys.path so the fork-only modules load; for YOLO models it uses stock
ultralytics. The notebook then just reads results/metrics/*.json, no model is loaded in the
notebook kernel, which avoids the fork-vs-stock Ultralytics version clash.

Usage (from the repository root):
  # DETR, also export the fork on PYTHONPATH so the unpickled custom classes resolve:
  PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
    python code/evaluate.py --type rtdetr --weights weights/diamond_detr/best.pt \
      --data dataset/tile_yolo/tile.yaml --name Diamond-DETR --out results/metrics
  # YOLO, stock ultralytics, no PYTHONPATH needed:
  python code/evaluate.py --type yolo --weights weights/yolov8n/best.pt \
      --data dataset/tile_yolo/tile.yaml --name YOLOv8n --out results/metrics
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def load(model_type, weights):
    if model_type == "rtdetr":
        # Diamond-DETR checkpoints contain custom classes (DRBC3, FasterNet) that live
        # only in the fork. Insert the fork before stock ultralytics in sys.path so the
        # right ultralytics is imported when loading these weights.
        import sys
        fork = Path(__file__).parent / "diamond_detr" / "RTDETR-main"
        if fork.exists() and str(fork) not in sys.path:
            sys.path.insert(0, str(fork))
        from ultralytics import RTDETR
        return RTDETR(weights)
    from ultralytics import YOLO
    return YOLO(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["rtdetr", "yolo"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="display name, e.g. Diamond-DETR")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/metrics", type=Path)
    args = ap.parse_args()

    model = load(args.type, args.weights)

    # FPS on a dummy frame (median of 50 forward passes after warmup)
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=args.imgsz, verbose=False)
    t0 = time.perf_counter()
    for _ in range(50):
        model.predict(dummy, imgsz=args.imgsz, verbose=False)
    fps = 50 / (time.perf_counter() - t0)

    r = model.val(data=args.data, split="test", imgsz=args.imgsz, verbose=False, plots=False)
    per_class = {r.names[i]: round(float(ap), 4) for i, ap in enumerate(r.box.maps)}

    rec = {
        "model": args.name,
        "weights": str(args.weights),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "fps": round(fps, 1),
        "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        "per_class_mAP50_95": per_class,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    safe = args.name.replace("/", "_").replace(" ", "_")
    (args.out / f"{safe}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

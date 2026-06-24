#!/usr/bin/env python3
"""
evaluate.py — validate one trained model on the test split and dump metrics to JSON.

Runs in whichever environment / cwd it is launched from, so call it:
  - from RTDETR-main/  for Diamond-DETR and RT-DETR (the fork, ultralytics 8.0.201)
  - from the project root for YOLOv8 / YOLO11 (stock ultralytics)

The notebook then just reads outputs/metrics/*.json — no model is loaded in the
notebook kernel, which avoids the fork-vs-stock Ultralytics version clash.

Usage:
  python evaluate.py --type rtdetr --weights checkpoints/diamond_detr/weights/best.pt \
      --data tile_yolo/tile.yaml --name Diamond-DETR --out outputs/metrics
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def load(model_type, weights):
    if model_type == "rtdetr":
        # DETR checkpoints reference modules (DRBC3, RepFasterNet) that exist only in
        # the fork. Put the fork first on sys.path so `import ultralytics` resolves to
        # 8.0.201, not the stock install — otherwise unpickling the weights fails.
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
    ap.add_argument("--out", default="outputs/metrics", type=Path)
    args = ap.parse_args()

    model = load(args.type, args.weights)

    # Throughput on a blank frame: one warm-up pass (lets cuDNN pick kernels), then
    # mean FPS over 50 timed passes. Decoupled from val() so accuracy and speed are
    # measured independently.
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

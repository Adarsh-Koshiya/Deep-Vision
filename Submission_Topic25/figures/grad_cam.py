#!/usr/bin/env python3
"""
grad_cam.py — Grad-CAM via the official jacobgil/pytorch-grad-cam library.

Standalone (outside the project repo); reads repo checkpoints/images, writes only here.

Detection heads return per-box outputs, not a single class logit, and the library iterates
over a (B, K) score tensor. So we wrap each detector to return (B, nc) = the per-class max
score over all detections, then run the library's GradCAM and explain the top class.

  python grad_cam.py --type rtdetr --weights .../diamond_detr/weights/best.pt --name Diamond-DETR
  python grad_cam.py --type yolo   --weights .../yolov8n/weights/best.pt      --name YOLOv8n
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

REPO = Path("/home/40a2/DV/Code")
FORK = REPO / "diamond_detr" / "RTDETR-main"
HERE = Path(__file__).resolve().parent
NC = 5  # blowhole/break/crack/fray/uneven


def load_model(mtype, weights):
    if mtype == "rtdetr":
        sys.path.insert(0, str(FORK))
        from ultralytics import RTDETR
        m = RTDETR(weights).model
    else:
        from ultralytics import YOLO
        m = YOLO(weights).model
    return m.float().eval()


class DetWrapper(nn.Module):
    """Wrap a detector so forward() -> (B, nc) per-class max score (Grad-CAM-friendly)."""
    def __init__(self, model, mtype):
        super().__init__()
        self.model, self.mtype = model, mtype

    def _class_scores(self, out):
        if self.mtype == "yolo":
            preds = out[0] if isinstance(out, (tuple, list)) else out   # (B, 4+nc, N)
            return preds[:, 4:, :].amax(dim=2)                          # (B, nc)
        # rtdetr: locate the (B, ., nc) or (B, nc, .) score tensor in the output
        tens = []
        def rec(o):
            if isinstance(o, torch.Tensor): tens.append(o)
            elif isinstance(o, (list, tuple)): [rec(e) for e in o]
            elif isinstance(o, dict): [rec(e) for e in o.values()]
        rec(out)
        for t in tens:
            if t.dim() == 3 and t.shape[-1] == NC:                      # (B, nq, nc)
                return t.amax(dim=1)
            if t.dim() == 3 and t.shape[1] == NC:                       # (B, nc, nq)
                return t.amax(dim=2)
        raise RuntimeError("no (.,nc) score tensor found in RT-DETR output")

    def forward(self, x):
        return self._class_scores(self.model(x))


def pick_target_layer(model, x):
    """Deepest backbone Conv2d (smallest spatial map, >=8)."""
    seen = []
    handles = [m.register_forward_hook(
        lambda mod, i, o, r=m: seen.append((r, o.shape))
        if isinstance(o, torch.Tensor) and o.dim() == 4 else None)
        for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in handles: h.remove()
    spatial = [(mod, shp) for mod, shp in seen if shp[-1] >= 8]
    return min(spatial, key=lambda t: t[1][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["rtdetr", "yolo"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = load_model(args.type, args.weights)
    for p in model.parameters():
        p.requires_grad_(True)
    wrapped = DetWrapper(model, args.type).eval()

    test_dir = REPO / "tile_yolo" / "images" / "test"
    classes = ["Blowhole", "Break", "Crack", "Fray", "Uneven"]
    safe = args.name.replace("/", "_").replace(" ", "_")
    (HERE / "overlays").mkdir(exist_ok=True)

    first = sorted(test_dir.glob("MT_Blowhole_*.jpg"))[0]
    probe = cv2.cvtColor(cv2.resize(cv2.imread(str(first)), (args.imgsz,)*2), cv2.COLOR_BGR2RGB)
    pt = torch.from_numpy(probe.astype(np.float32)/255.).permute(2, 0, 1).unsqueeze(0)
    target_layer, shp = pick_target_layer(model, pt)
    print(f"  target layer output {tuple(shp)}")

    cam = GradCAM(model=wrapped, target_layers=[target_layer])

    for cls in classes:
        matches = sorted(test_dir.glob(f"MT_{cls}_*.jpg"))
        if not matches:
            print(f"  [skip] {cls}"); continue
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(matches[0])), (args.imgsz,)*2), cv2.COLOR_BGR2RGB)
        rgb01 = rgb.astype(np.float32) / 255.0
        t = torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            top = int(wrapped(t).argmax(1).item())              # explain the top predicted class
        grayscale = cam(input_tensor=t, targets=[ClassifierOutputTarget(top)])[0]
        vis = show_cam_on_image(rgb01, grayscale, use_rgb=True)
        out = HERE / "overlays" / f"{safe}__{cls}.png"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  saved {out.name}  (top class idx={top})")
    print(f"done: {args.name}")


if __name__ == "__main__":
    main()

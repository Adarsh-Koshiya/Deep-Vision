#!/usr/bin/env python3
"""Build an RT-DETR/Diamond ablation config and run one forward pass to confirm it is valid.

Needs the fork on PYTHONPATH. Prints the parameter count and output shapes, or the error.

  PYTHONPATH=$DIAMOND python validate_cfg.py <abs_or_rel_yaml>
"""
import sys
import torch
from ultralytics import RTDETR

yaml = sys.argv[1]
try:
    m = RTDETR(yaml).model.eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 640, 640))
    shapes = []
    def rec(o):
        if isinstance(o, torch.Tensor): shapes.append(tuple(o.shape))
        elif isinstance(o, (list, tuple)): [rec(e) for e in o]
    rec(out)
    print(f"OK  {yaml}  params={n:.2f}M  out_shapes={shapes[:4]}")
except Exception as e:
    print(f"FAIL  {yaml}  ->  {type(e).__name__}: {e}")
    raise

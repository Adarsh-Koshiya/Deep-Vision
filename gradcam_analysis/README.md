# Grad-CAM attention analysis (standalone)

**Separate from the project repo** (`/home/40a2/DV/Code`) on purpose. This folder only
*reads* the repo's checkpoints and test images; it writes nothing back into the repo.

## What this is
**Grad-CAM via the official `jacobgil/pytorch-grad-cam` library** (Selvaraju et al., 2017) —
showing where each trained detector attends on one magnetic-tile image per defect class.

Method (`grad_cam.py`):
- Detection heads emit per-box outputs, and the library iterates over a `(B, K)` score tensor,
  so each model is wrapped (`DetWrapper`) to return `(B, nc)` = the **per-class max score over all
  detections** (YOLO: `preds[:,4:,:]`; RT-DETR: the `(·,·,nc)` score tensor).
- `GradCAM(model=wrapped, target_layers=[deepest_backbone_conv])`, explained class =
  `ClassifierOutputTarget(top_predicted_class)`. The library does the hooks, gradient GAP,
  ReLU and upsampling.
- Target layer = deepest backbone conv (P5, 20×20) for every model, for a fair comparison.

## Files
- `grad_cam.py` — compute Grad-CAM overlays for one model on all 5 class images.
- `overlays/` — 20 per-model/per-class overlays (`<Model>__<Class>.png`).
- `gradcam_comparison.png` — the figure: rows = 5 defect classes, cols = input + 4 models.

## Reproduce
```bash
cd /home/40a2/DV/gradcam_analysis
python grad_cam.py --type rtdetr --weights /home/40a2/DV/Code/checkpoints/diamond_detr/weights/best.pt   --name Diamond-DETR
python grad_cam.py --type rtdetr --weights /home/40a2/DV/Code/checkpoints/rtdetr_baseline/weights/best.pt --name RT-DETR
python grad_cam.py --type yolo   --weights /home/40a2/DV/Code/checkpoints/yolov8n/weights/best.pt --name YOLOv8n
python grad_cam.py --type yolo   --weights /home/40a2/DV/Code/checkpoints/yolo11n/weights/best.pt --name YOLO11n
# then re-stitch the grid (see the compose snippet in the project plan, writes gradcam_comparison.png)
```

## Findings (consistent with the quantitative results)
- **Top predicted class exposes the gap directly:** both transformers predict the *correct* defect
  class for all 5 images (Blowhole→0, Break→1, Crack→2, Fray→3, Uneven→4); **YOLO11n predicts
  class 0 for almost everything** and YOLOv8n is mostly wrong on the rare classes — matching their
  near-zero rare-class AP.
- **Diamond-DETR** gives the most focused, defect-aligned maps (crisp line along the *crack*,
  tight region on *uneven*/*blowhole*); the **CNNs** are broader and more diffuse.

## Notes
- Target = deepest backbone layer (coarse 20×20 localisation — standard Grad-CAM granularity).
  For sharper maps, target a shallower layer (P4/40×40).
- To embed in the notebook (§6) later: copy `gradcam_comparison.png` → `Code/outputs/gradcam.png`
  (kept separate here by request).

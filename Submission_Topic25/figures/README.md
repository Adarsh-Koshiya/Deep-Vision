# Grad-CAM attention analysis (`figures/`)

The Grad-CAM attention study for the four detectors. `grad_cam.py` *reads* the trained
checkpoints in `weights/` and the test images in `dataset/tile_yolo/`, and writes its outputs
(overlays + comparison grid) into this folder. The notebook §6 displays `gradcam_comparison.png`.

## What this is
**Grad-CAM via the official `jacobgil/pytorch-grad-cam` library** (Selvaraju et al., 2017), 
showing where each trained detector attends on one magnetic-tile image per defect class.

Method (`grad_cam.py`):
- Detection heads emit per-box outputs, and the library iterates over a `(B, K)` score tensor,
 so each model is wrapped (`DetWrapper`) to return `(B, nc)` = the **per-class max score over all
 detections** (YOLO: `preds[:,4::]`; RT-DETR: the `(·,·,nc)` score tensor).
- `GradCAM(model=wrapped, target_layers=[deepest_backbone_conv])`, explained class =
 `ClassifierOutputTarget(top_predicted_class)`. The library does the hooks, gradient GAP,
 ReLU and upsampling.
- Target layer = deepest backbone conv (P5, 20×20) for every model, for a fair comparison.

## Files
- `grad_cam.py`, compute Grad-CAM overlays for one model on all 5 class images.
- `overlays/`, 20 per-model/per-class overlays (`<Model>__<Class>.png`).
- `gradcam_comparison.png`, the figure: rows = 5 defect classes, cols = input + 4 models.
- `make_figs.py`, regenerates the report's result figures from `results/metrics/*.json`:
  `fig_overall_map.png`, `fig_perclass_map.png`, `fig_speed_accuracy.png` (used by `report/REPORT.tex`).
  Run `python figures/make_figs.py` from anywhere.

## Reproduce
Run from the repository root (needs the dataset at `dataset/tile_yolo/`, not bundled). The DETR
models need the bundled fork on `PYTHONPATH`:
```bash
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
 python figures/grad_cam.py --type rtdetr --weights weights/diamond_detr/best.pt --name Diamond-DETR
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
 python figures/grad_cam.py --type rtdetr --weights weights/rtdetr_baseline/best.pt --name RT-DETR
python figures/grad_cam.py --type yolo --weights weights/yolov8n/best.pt --name YOLOv8n
python figures/grad_cam.py --type yolo --weights weights/yolo11n/best.pt --name YOLO11n
# then re-stitch the grid (see the compose snippet in ABLATION_GRADCAM_PLAN.md → gradcam_comparison.png)
```

## Findings (consistent with the quantitative results)
- **Top predicted class exposes the gap directly:** both transformers predict the *correct* defect
 class for all 5 images (Blowhole→0, Break→1, Crack→2, Fray→3, Uneven→4); **YOLO11n predicts
 class 0 for almost everything** and YOLOv8n is mostly wrong on the rare classes, matching their
 near-zero rare-class AP.
- **Diamond-DETR** gives the most focused, defect-aligned maps (crisp line along the *crack*,
 tight region on *uneven*/*blowhole*); the **CNNs** are broader and more diffuse.

## Notes
- Target = deepest backbone layer (coarse 20×20 localisation, standard Grad-CAM granularity).
 For sharper maps, target a shallower layer (P4/40×40).
- Notebook §6 reads `figures/gradcam_comparison.png` directly, no copy step needed.

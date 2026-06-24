# Deep Vision — Topic 25: Transformers vs CNNs for Surface-Defect Detection

**Cross-Domain Evaluation of Diamond-DETR on the Magnetic Tile Surface Defect dataset**
OTH Amberg-Weiden · Deep Vision · Sommersemester 2026 · Author: Adarsh Koshiya

> **Primary deliverable: `report/REPORT.pdf`.** This archive holds the report and source code.
> The **trained model weights are on GitLab** (too large for the archive) and the **dataset is public**
> — see "Not in this archive" below. Absolute paths in some scripts point at the original working
> directory `/home/40a2/DV/Code`; adjust if re-running elsewhere.

## What was done

Four detectors were trained and evaluated under matched conditions on the 5-class Magnetic Tile defect
task (blowhole · break · crack · fray · uneven), and Diamond-DETR's three components were ablated.

| Model | Family | Params | Test mAP50 | mAP50-95 | FPS |
|-------|--------|-------:|-----------:|---------:|----:|
| **Diamond-DETR** | Transformer | 14.3 M | 0.762 | 0.489 | 56.5 |
| RT-DETR-r18 | Transformer | 19.9 M | 0.763 | 0.446 | 40.2 |
| YOLOv8n | CNN | 3.0 M | 0.403 | 0.275 | 205 |
| YOLO11n | CNN | 2.6 M | 0.335 | 0.204 | 153 |

Headline findings (full analysis in the report): transformers far outperform the CNNs on rare defect
classes; under matched (from-scratch) initialisation Diamond-DETR ties RT-DETR at 28% fewer parameters
(efficiency, not an accuracy win); and the **complete 2³ component ablation** (matching the source
paper's Table 2) shows the HS-FPN neck, the paper's strongest component on diamonds, is *detrimental* on
tiles — so the best tile configuration (A+C+D, 0.587 mAP50-95) beats the full Diamond-DETR (0.489).
All numbers are reproduced verbatim from `results/**/*.json`.

## Folder layout (this archive)

```
report/      REPORT.pdf  — primary deliverable; REPORT.tex — its LaTeX source
notebook/    MagTile_DiamondDETR_Notebook.ipynb  (reads results/*.json only — no model loading)
code/        prepare_data.py, evaluate.py, train.sh, clean_fork_warnings.py
             experiments/   — ablation + multi-seed scripts and RUNBOOK
             diamond_detr/  — patched Ultralytics 8.0.201 fork; config YAMLs incl. the 8 ablation configs
gradcam_analysis/   Grad-CAM overlays (model × defect class) + comparison grid + grad_cam.py
results/     metrics/*.json            — the 4 main models (single source of the headline numbers)
             metrics_ablation/*.json   — the full 2³ factorial (8 configurations)
```

## Not in this archive

- **Trained weights** (`best.pt` for all models, ~78 MB) → provided on **GitLab** (repository link
  submitted alongside this archive).
- **Dataset** (public): Magnetic Tile Surface Defect — Huang, Qiu, Yuan, *The Visual Computer* 36, 85–96
  (2020), DOI 10.1007/s00371-018-1588-5; images at https://github.com/abin24/Magnetic-tile-defect-datasets .
  Rebuild the split with `python code/prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo`
  → **seed-42 stratified 70/20/10 split = 938 train / 268 val / 138 test, 0 train∩test overlap (verified).**

## How to reproduce

```bash
# 1. Obtain the dataset (link above) and build the split:
python code/prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo

# 2. Download the trained weights from GitLab (or retrain: FORCE=1 bash code/train.sh, ~15-20 h).

# 3. Evaluate (writes results/metrics/<Name>.json):
#    YOLO models — stock Ultralytics, from project root:
python code/evaluate.py --type yolo --weights <yolov8n_best.pt> \
  --data tile_yolo/tile.yaml --name YOLOv8n --out results/metrics
#    DETR models — require the fork on PYTHONPATH:
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
  python code/evaluate.py --type rtdetr --weights <diamond_best.pt> \
  --data tile_yolo/tile.yaml --name Diamond-DETR --out results/metrics
```

## Environment

Python 3.12 · torch 2.8.0+cu129 · NVIDIA RTX 3060 Ti 8 GB.
**Two Ultralytics versions are required and cannot share one Python process:** stock 8.4.x for the YOLO
models, and the bundled fork (`code/diamond_detr/RTDETR-main/`, 8.0.201, 6 applied patches) for the DETR
models. `train.sh` and `evaluate.py` already handle the split (the 6 fork patches are listed in the
report's appendix).

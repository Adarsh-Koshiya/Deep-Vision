# Deep Vision — Topic 25: Transformers vs CNNs for Surface-Defect Detection

**Cross-Domain Evaluation of Diamond-DETR on the Magnetic Tile Surface Defect dataset**
OTH Amberg-Weiden · Deep Vision · Author: Adarsh Koshiya

> **Note for the evaluator:** This is a self-contained snapshot of the completed study.
> The primary deliverable is **`report/REPORT.pdf`**. Everything needed to read, verify, or
> re-run the work is included. Absolute paths inside scripts (e.g. `train.sh`) point at the
> original working directory `/home/40a2/DV/Code`; adjust them if re-running elsewhere.

## What was done

Four object detectors were trained and evaluated under matched conditions on the 5-class
Magnetic Tile defect task (blowhole · break · crack · fray · uneven):

| Model | Family | Params | Test mAP50 | mAP50-95 | FPS |
|-------|--------|-------:|-----------:|---------:|----:|
| **Diamond-DETR** | Transformer | 14.3 M | 0.762 | **0.489** | 56.5 |
| RT-DETR-r18 | Transformer | 19.9 M | **0.763** | 0.446 | 40.2 |
| YOLOv8n | CNN | 3.0 M | 0.403 | 0.275 | 205 |
| YOLO11n | CNN | 2.6 M | 0.335 | 0.204 | 153 |

Numbers are reproduced verbatim from `results/metrics/*.json`. Full analysis, limitations,
and per-class breakdowns are in the report.

## Folder layout

```
report/      REPORT.pdf  — primary deliverable (PDF), and REPORT.md source
notebook/    MagTile_DiamondDETR_Notebook.ipynb (source) + .executed.ipynb + .html (rendered)
code/        prepare_data.py, evaluate.py, train.sh, clean_fork_warnings.py
             yolov8n.pt / yolo11n.pt  — COCO init weights for the CNNs
             diamond_detr/            — patched Ultralytics 8.0.201 fork (RT-DETR/Diamond-DETR)
                                        + adamas/.../RTDETR-r18/best.pt (paper's pretrained init)
results/     metrics/*.json           — the four evaluation result files (single source of numbers)
figures/     Grad-CAM attention overlays (one per model × defect class) + comparison grid
weights/     best.pt for all 4 trained models (+ results.csv / args.yaml where available)
dataset/     tile_yolo/  — prepared YOLO-format dataset (train 1910 / val 727 / test 380 images)
             Magnetic-Tile-Defect/  — raw images + segmentation masks (source dataset)
docs/        PROJECT_NOTES.md (design rationale), Execution_Guide.md, CLAUDE.md, ABLATION_GRADCAM_PLAN.md
```

## How to reproduce

```bash
# 1. Rebuild the dataset from the raw masks (optional — already provided in dataset/):
python code/prepare_data.py --src dataset/Magnetic-Tile-Defect --out dataset/tile_yolo

# 2. Retrain all 4 models (~15–20 h on an RTX 3060 Ti). FORCE=1 overrides existing metrics:
FORCE=1 bash code/train.sh

# 3. Evaluate a single model (writes results/metrics/<Name>.json):
#    YOLO models — from the project root:
python code/evaluate.py --type yolo --weights weights/yolov8n/best.pt \
  --data dataset/tile_yolo/tile.yaml --name YOLOv8n --out results/metrics
#    DETR models — require the fork on PYTHONPATH:
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
  python code/evaluate.py --type rtdetr --weights weights/diamond_detr/best.pt \
  --data dataset/tile_yolo/tile.yaml --name Diamond-DETR --out results/metrics
```

## Environment

Python 3.12 · torch 2.8.0+cu129 · NVIDIA RTX 3060 Ti 8 GB.
**Two Ultralytics versions are required and cannot share one Python process:** stock 8.4.x
for the YOLO models, and the bundled fork (`code/diamond_detr/RTDETR-main/`, 8.0.201, with
6 applied patches) for the DETR models. `train.sh` and `evaluate.py` already handle the split.
See `docs/CLAUDE.md` and `docs/PROJECT_NOTES.md` for the patch list and rationale.

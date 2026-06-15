# Execution Guide, Diamond-DETR on Magnetic Tile Defect Dataset

5-class defect detection: **blowhole / break / crack / fray / uneven** 
Comparison: Diamond-DETR vs RT-DETR-r18 vs YOLOv8n vs YOLO11n

> **Status:** Study complete, all four models trained and evaluated; results are final in
> `results/metrics/*.json` and locked in `report/REPORT.md` §6. This guide describes how to
> reproduce the runs. Single entry point: `bash code/train.sh` (from the repository root).
> Paths below use the final submission layout (`dataset/`, `weights/`, `results/metrics/`).

---

## Prerequisites

```bash
# The fork is already bundled at code/diamond_detr/. To re-clone it from scratch:
git clone https://github.com/bigboyliang/DIAMOND.git code/diamond_detr

# Stock ultralytics for YOLO models:
pip install ultralytics

# The fork provides its own ultralytics 8.0.201 at code/diamond_detr/RTDETR-main/, do NOT pip it.

# Other deps (for notebook, Grad-CAM, data prep):
pip install albumentations pytorch-grad-cam scipy scikit-learn \
 matplotlib seaborn pandas opencv-python-headless
```

> `train.sh` resolves the repository root from its own location (`code/`), so the fork must
> stay at `code/diamond_detr/` and the dataset at `dataset/` (repo root).

---

## Run everything

```bash
cd <repository-root>
screen -S train bash -c 'bash code/train.sh 2>&1 | tee train_full.log'
```

`train.sh` handles everything in order:
1. Builds `dataset/tile_yolo/` if missing (prepare_data.py)
2. Trains **YOLOv8n** (stock ultralytics), skips if `results/metrics/YOLOv8n.json` exists
3. Trains **YOLO11n** (stock ultralytics), skips if JSON exists
4. Trains **RT-DETR-r18** from pretrained paper checkpoint, skips if JSON exists
5. Trains **Diamond-DETR** from random init, skips if JSON exists
6. Writes `results/metrics/<Name>.json` after each model

Set `FORCE=1` to retrain even if the JSON exists:
```bash
FORCE=1 bash train.sh # retrain all
```

Smoke test (5 epochs):
```bash
EPOCHS=5 bash train.sh
```

Monitor progress:
```bash
tail -f train_full.log
tail -1 weights/rtdetr_baseline/results.csv # current epoch
nvidia-smi # GPU
```

---

## Dataset

**Magnetic Tile Surface Defect** (Huang et al. 2020), 1,344 images, 5 classes + defect-free.
Not bundled in the submission; place it at `dataset/Magnetic-Tile-Defect/` (repo root) with
subfolders `MT_Blowhole/`, `MT_Break/`, `MT_Crack/`, `MT_Fray/`, `MT_Uneven/`, `MT_Free/`.
Each defect folder has `Imgs/` + `Masks/` (the loader also handles masks sitting beside images).

`prepare_data.py` converts masks → YOLO bounding boxes and writes `dataset/tile_yolo/`:
```
dataset/tile_yolo/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── tile.yaml (nc:5, names:[blowhole,break,crack,fray,uneven])
```
Split: **70/20/10**, seed 42 → 938 train / 268 val / 138 test. Zero leakage verified.

---

## Model details

| Model | Params | Pretrained init | Framework |
|-------|--------|-----------------|-----------|
| YOLOv8n | 3.0 M | COCO (`yolov8n.pt`) | stock ultralytics 8.4.x |
| YOLO11n | 2.6 M | COCO (`yolo11n.pt`) | stock ultralytics 8.4.x |
| RT-DETR-r18 | 20 M | Paper's diamond checkpoint (`adamas/…/best.pt`, nc=2→5) | fork 8.0.201 |
| Diamond-DETR | 14.4 M | None (random init, authors never published weights) | fork 8.0.201 |

Diamond-DETR = RT-DETR-r18 with three components swapped (B+C+D from paper):
- Backbone: `BasicBlock_Faster_Block_Rep` (FasterNet re-parameterizable)
- Neck: `ChannelAttention_HSFPN` + `DRBC3`
- Head/encoder: identical to RT-DETR

---

## Training config

| Setting | CNNs | Transformers |
|---------|------|--------------|
| Epochs | 150 | 200 |
| Batch | 4 | 4 |
| Optimizer | SGD | **AdamW** |
| lr0 | 1e-4 | 1e-4 |
| momentum | 0.8 | 0.9 |
| weight_decay | 1e-4 | 1e-4 |
| imgsz | 640 | 640 |
| seed | 42 | 42 |

The reported CNN results come from a 150-epoch run (already converged from COCO weights); the
transformers ran 200. `train.sh` now defaults to `EPOCHS=200`, so a fresh `FORCE=1` run would
re-train the CNNs at 200, see REPORT.md §5 / Limitation 5 for why this asymmetry is harmless.

---

## Fork patches applied (code/diamond_detr/RTDETR-main/)

All already in place, only needed if you re-clone the fork:

| # | File | Change |
|---|------|--------|
| 1 | `ultralytics/utils/metrics.py` | `np.trapz` → `np.trapezoid` (numpy 2.0 removed trapz) |
| 2 | `ultralytics/utils/torch_utils.py` | `torch.load(..., weights_only=False)` (torch 2.6 default changed) |
| 3 | `ultralytics/nn/tasks.py` | Same `weights_only=False` (2 more call sites) |
| 4 | `ultralytics/nn/tasks.py` | `attempt_load_one_weight` must return `(model, ckpt)`, fork bug |
| 5 | `ultralytics/cfg/models/rt-detr/Diamond-detr.yaml` | Replaced wrong FasterNet-T2 YAML with real paper arch |
| 6 | `ultralytics/nn/tasks.py` ConvNormLayer fuse | Added `and hasattr(m, 'norm')` guard, prevents crash if `fuse()` called twice |

---

## Evaluate manually (if needed)

evaluate.py must be called with the fork on PYTHONPATH for DETR models:

All commands below run from the repository root (the bundled `best.pt` weights are in `weights/`).

```bash
# RT-DETR, needs the fork on PYTHONPATH
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
 python code/evaluate.py --type rtdetr \
 --weights weights/rtdetr_baseline/best.pt \
 --data dataset/tile_yolo/tile.yaml \
 --name "RT-DETR" --imgsz 640 --out results/metrics

# Diamond-DETR, same fork on PYTHONPATH
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
 python code/evaluate.py --type rtdetr \
 --weights weights/diamond_detr/best.pt \
 --data dataset/tile_yolo/tile.yaml \
 --name "Diamond-DETR" --imgsz 640 --out results/metrics

# YOLO models (stock ultralytics, no PYTHONPATH needed)
python code/evaluate.py --type yolo \
 --weights weights/yolov8n/best.pt \
 --data dataset/tile_yolo/tile.yaml --name YOLOv8n --out results/metrics
```

---

## Results (test split, 138 images)

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS | Params |
|-------|-------|----------|-----------|--------|-----|--------|
| Diamond-DETR | 0.762 | **0.489** | 0.745 | 0.746 | 56.5 | 14.3 M |
| RT-DETR-r18 | **0.763** | 0.446 | 0.733 | **0.819** | 40.2 | 19.9 M |
| YOLOv8n | 0.403 | 0.275 | 0.613 | 0.349 | 205 | 3.0 M |
| YOLO11n | 0.335 | 0.204 | 0.579 | 0.227 | 153 | 2.6 M |

Per-class mAP50-95 (blowhole · break · crack · fray · uneven):
- Diamond-DETR: 0.670 · 0.502 · 0.536 · 0.383 · 0.356
- RT-DETR-r18: 0.578 · 0.415 · 0.388 · 0.463 · 0.383
- YOLOv8n: 0.707 · 0.074 · 0.280 · 0.069 · 0.244
- YOLO11n: 0.563 · 0.078 · 0.175 · 0.023 · 0.181

(Final test-split numbers, identical to `results/metrics/*.json` and REPORT.md §6.)

---

## Notebook

`notebook/MagTile_DiamondDETR_Notebook.ipynb`, reads `results/metrics/*.json` and the
Grad-CAM figure only, no model loading. A pre-executed copy (`.executed.ipynb`) and an HTML
render (`.html`) are included. To re-run:

```bash
jupyter notebook notebook/MagTile_DiamondDETR_Notebook.ipynb
```

---

## Report

The primary deliverable is the typeset LaTeX report. `report/REPORT.tex` is the source,
`report/REPORT.pdf` (13 pages) is the build, and `report/REPORT.md` is the working draft kept in
sync with it. To rebuild:

```bash
# 1. regenerate the result figures from the metrics JSON (writes into figures/):
python figures/make_figs.py
# 2. compile the PDF (run xelatex twice for refs and the table of contents):
cd report && xelatex REPORT.tex && xelatex REPORT.tex
```

Style rule for any edit: never use the em-dash (U+2014) or en-dash (U+2013); in LaTeX that also
means never write `--` or `---`, since those render as en/em dashes.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AttributeError: Can't get attribute 'ConvNormLayer'` | Missing PYTHONPATH, set `PYTHONPATH="$DIAMOND:${PYTHONPATH:-}"` before calling evaluate.py |
| `AttributeError: 'ConvNormLayer' has no attribute 'norm'` | Double-fuse bug, patch #6 already applied to tasks.py |
| CUDA OOM on RT-DETR | Reduce batch to 2; shared GPU may have another user's process |
| Diamond-DETR mAP ≈ 0 | Only if trained with SGD, transformers need AdamW (the shipped run used AdamW and converged) |
| `nc` mismatch warning | Normal for RT-DETR (nc=2→5 via intersect_dicts); head re-initialised automatically |
| `train.sh` skips a model | `results/metrics/<Name>.json` exists, run `FORCE=1 bash code/train.sh` to retrain |

---

## File map

```
<repo root>/
├── README.md ← top-level overview + reproduce commands
├── code/
│ ├── train.sh ← single entry point; trains all 4 models
│ ├── evaluate.py ← test-split eval → results/metrics/<name>.json
│ ├── prepare_data.py ← masks → YOLO 5-class boxes; builds dataset/tile_yolo/
│ ├── clean_fork_warnings.py ← cosmetic: removes torch/albumentations deprecation warnings
│ ├── yolov8n.pt / yolo11n.pt ← COCO init weights for the CNNs
│ └── diamond_detr/ ← bundled DIAMOND repo (fork ultralytics 8.0.201)
│ ├── RTDETR-main/ultralytics/ ← patched in-place (patches #1-#6 above)
│ └── adamas/runs/train/RTDETR-r18/weights/best.pt ← RT-DETR pretrained init (39 MB)
├── docs/ ← PROJECT_NOTES, Execution_Guide (this file), CLAUDE,
│ ABLATION_GRADCAM_PLAN, FUTURE_WORK
├── report/ ← REPORT.tex (LaTeX source) + REPORT.pdf (deliverable) + REPORT.md (working draft)
├── notebook/ ← MagTile_DiamondDETR_Notebook.ipynb + .executed.ipynb + .html
├── figures/ ← grad_cam.py + make_figs.py + overlays/ + gradcam_comparison.png + fig_*.png
├── results/metrics/*.json ← final test metrics (written by evaluate.py)
├── weights/ ← best.pt + results.csv + args.yaml for all 4 models
│ ├── rtdetr_baseline/ ← RT-DETR
│ ├── diamond_detr/ ← Diamond-DETR
│ ├── yolov8n/ └── yolo11n/
└── dataset/ ← NOT bundled, create here when reproducing:
 ├── Magnetic-Tile-Defect/ ← raw images + masks (download from source repo)
 └── tile_yolo/ ← generated YOLO split (tile.yaml + images + labels)
```

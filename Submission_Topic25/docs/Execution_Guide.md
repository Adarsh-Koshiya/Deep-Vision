# Execution Guide — Diamond-DETR on Magnetic Tile Defect Dataset

5-class defect detection: **blowhole / break / crack / fray / uneven**  
Comparison: Diamond-DETR vs RT-DETR-r18 vs YOLOv8n vs YOLO11n

> **Status as of 2026-06-14:** CNN results are final (outputs/metrics/YOLOv8n.json,
> YOLO11n.json). RT-DETR is training (third run, screen `1869829.train`). Diamond-DETR
> will train automatically after RT-DETR finishes. Single entry point: `bash train.sh`.

---

## Prerequisites

```bash
# Inside Claude code/ — fork must live here:
git clone https://github.com/bigboyliang/DIAMOND.git diamond_detr

# Stock ultralytics for YOLO models:
pip install ultralytics

# The fork provides its own ultralytics 8.0.201 at diamond_detr/RTDETR-main/ — do NOT pip it.

# Other deps (for notebook, Grad-CAM, data prep):
pip install albumentations pytorch-grad-cam scipy scikit-learn \
            matplotlib seaborn pandas opencv-python-headless
```

> `diamond_detr/` must sit inside `Claude code/` — `train.sh` constructs all paths relative
> to its own location.

---

## Run everything

```bash
cd "/home/40a2/DV/Claude code"
screen -S train bash -c 'bash train.sh 2>&1 | tee train_full.log'
```

`train.sh` handles everything in order:
1. Builds `tile_yolo/` if missing (prepare_data.py)
2. Trains **YOLOv8n** (stock ultralytics) — skips if `outputs/metrics/YOLOv8n.json` exists
3. Trains **YOLO11n** (stock ultralytics) — skips if JSON exists
4. Trains **RT-DETR-r18** from pretrained paper checkpoint — skips if JSON exists
5. Trains **Diamond-DETR** from random init — skips if JSON exists
6. Writes `outputs/metrics/<Name>.json` after each model

Set `FORCE=1` to retrain even if the JSON exists:
```bash
FORCE=1 bash train.sh   # retrain all
```

Smoke test (5 epochs):
```bash
EPOCHS=5 bash train.sh
```

Monitor progress:
```bash
tail -f train_full.log
tail -1 checkpoints/rtdetr_baseline/results.csv   # current epoch
nvidia-smi                                         # GPU
```

---

## Dataset

**Magnetic Tile Surface Defect** (Huang et al. 2020) — 1,344 images, 5 classes + defect-free.
Stored in `Magnetic-Tile-Defect/` with subfolders `Blowhole/`, `Break/`, `Crack/`, `Fray/`,
`Uneven/`, `Free/`. Each defect folder has `Imgs/` + `Masks/`.

`prepare_data.py` converts masks → YOLO bounding boxes and writes `tile_yolo/`:
```
tile_yolo/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── tile.yaml   (nc:5, names:[blowhole,break,crack,fray,uneven])
```
Split: **70/20/10**, seed 42 → 938 train / 268 val / 138 test. Zero leakage verified.

---

## Model details

| Model | Params | Pretrained init | Framework |
|-------|--------|-----------------|-----------|
| YOLOv8n | 3.0 M | COCO (`yolov8n.pt`) | stock ultralytics 8.4.x |
| YOLO11n | 2.6 M | COCO (`yolo11n.pt`) | stock ultralytics 8.4.x |
| RT-DETR-r18 | 20 M | Paper's diamond checkpoint (`adamas/…/best.pt`, nc=2→5) | fork 8.0.201 |
| Diamond-DETR | 14.4 M | None (random init — authors never published weights) | fork 8.0.201 |

Diamond-DETR = RT-DETR-r18 with three components swapped (B+C+D from paper):
- Backbone: `BasicBlock_Faster_Block_Rep` (FasterNet re-parameterizable)
- Neck: `ChannelAttention_HSFPN` + `DRBC3`
- Head/encoder: identical to RT-DETR

---

## Training config

| Setting | CNNs | Transformers |
|---------|------|--------------|
| Epochs | 200 | 200 |
| Batch | 4 | 4 |
| Optimizer | SGD | **AdamW** |
| lr0 | 1e-4 | 1e-4 |
| momentum | 0.8 | 0.9 |
| weight_decay | 1e-4 | 1e-4 |
| imgsz | 640 | 640 |
| seed | 42 | 42 |

---

## Fork patches applied (diamond_detr/RTDETR-main/)

All already in place — only needed if you re-clone the fork:

| # | File | Change |
|---|------|--------|
| 1 | `ultralytics/utils/metrics.py` | `np.trapz` → `np.trapezoid` (numpy 2.0 removed trapz) |
| 2 | `ultralytics/utils/torch_utils.py` | `torch.load(..., weights_only=False)` (torch 2.6 default changed) |
| 3 | `ultralytics/nn/tasks.py` | Same `weights_only=False` (2 more call sites) |
| 4 | `ultralytics/nn/tasks.py` | `attempt_load_one_weight` must return `(model, ckpt)` — fork bug |
| 5 | `ultralytics/cfg/models/rt-detr/Diamond-detr.yaml` | Replaced wrong FasterNet-T2 YAML with real paper arch |
| 6 | `ultralytics/nn/tasks.py` ConvNormLayer fuse | Added `and hasattr(m, 'norm')` guard — prevents crash if `fuse()` called twice |

---

## Evaluate manually (if needed)

evaluate.py must be called with the fork on PYTHONPATH for DETR models:

```bash
cd "/home/40a2/DV/Claude code/diamond_detr/RTDETR-main"

# RT-DETR
PYTHONPATH="/home/40a2/DV/Claude code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
  python "/home/40a2/DV/Claude code/evaluate.py" \
  --type rtdetr \
  --weights "/home/40a2/DV/Claude code/checkpoints/rtdetr_baseline/weights/best.pt" \
  --data "/home/40a2/DV/Claude code/tile_yolo/tile.yaml" \
  --name "RT-DETR" --imgsz 640 \
  --out "/home/40a2/DV/Claude code/outputs/metrics"

# Diamond-DETR
PYTHONPATH="/home/40a2/DV/Claude code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
  python "/home/40a2/DV/Claude code/evaluate.py" \
  --type rtdetr \
  --weights "/home/40a2/DV/Claude code/checkpoints/diamond_detr/weights/best.pt" \
  --data "/home/40a2/DV/Claude code/tile_yolo/tile.yaml" \
  --name "Diamond-DETR" --imgsz 640 \
  --out "/home/40a2/DV/Claude code/outputs/metrics"

# YOLO models (from project root, no PYTHONPATH needed)
cd "/home/40a2/DV/Claude code"
python evaluate.py --type yolo \
  --weights checkpoints/yolov8n/weights/best.pt \
  --data tile_yolo/tile.yaml --name YOLOv8n --out outputs/metrics
```

---

## Results (test split, 138 images)

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS |
|-------|-------|----------|-----------|--------|-----|
| YOLOv8n | **0.403** | **0.275** | 0.613 | 0.349 | 205 |
| YOLO11n | 0.335 | 0.204 | 0.579 | 0.227 | 153 |
| RT-DETR-r18 | ⏳ training | — | — | — | ~28 |
| Diamond-DETR | ⏳ training | — | — | — | — |

Per-class mAP50-95: blowhole 0.707/0.563 · break 0.074/0.078 · crack 0.280/0.175 · fray 0.069/0.023 · uneven 0.244/0.181

---

## Notebook

`MagTile_DiamondDETR_Notebook.ipynb` — reads `outputs/metrics/*.json` only, no model
loading. Open after all four JSONs exist and run all cells.

```bash
jupyter notebook MagTile_DiamondDETR_Notebook.ipynb
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AttributeError: Can't get attribute 'ConvNormLayer'` | Missing PYTHONPATH — set `PYTHONPATH="$DIAMOND:${PYTHONPATH:-}"` before calling evaluate.py |
| `AttributeError: 'ConvNormLayer' has no attribute 'norm'` | Double-fuse bug — patch #6 already applied to tasks.py |
| CUDA OOM on RT-DETR | Reduce batch to 2; shared GPU may have another user's process |
| Diamond-DETR mAP ≈ 0 | Expected — no pretrained weights; random init on 938 images, 200 epochs |
| `nc` mismatch warning | Normal for RT-DETR (nc=2→5 via intersect_dicts); head re-initialised automatically |
| `train.sh` skips a model | `outputs/metrics/<Name>.json` exists — run `FORCE=1 bash train.sh` to retrain |

---

## File map

```
Claude code/
├── train.sh                    ← single entry point; trains all 4 models
├── evaluate.py                 ← test-split eval → outputs/metrics/<name>.json
├── prepare_data.py             ← masks → YOLO 5-class boxes; builds tile_yolo/
├── gradcam.py                  ← Grad-CAM visualisation helper
├── clean_fork_warnings.py      ← suppresses timm/DETR deprecation warnings
├── REPORT.md                   ← living report (single source of truth)
├── PROJECT_NOTES.md            ← design decisions, incident log, agent handoff
├── Execution_Guide.md          ← this file
├── MagTile_DiamondDETR_Notebook.ipynb
├── diamond_detr/               ← cloned DIAMOND repo (fork ultralytics 8.0.201)
│   └── RTDETR-main/ultralytics/  ← patched in-place (patches #1-#6 above)
├── Magnetic-Tile-Defect/       ← raw dataset (read-only)
├── tile_yolo/                  ← generated YOLO split (tile.yaml + images + labels)
├── checkpoints/
│   ├── rtdetr_baseline/        ← RT-DETR weights + results.csv
│   ├── diamond_detr/           ← Diamond-DETR weights + results.csv
│   ├── yolov8n/
│   └── yolo11n/
├── outputs/metrics/*.json      ← final test metrics (written by evaluate.py)
├── memory/                     ← in-project AI state notes (travels with repo)
├── train_full.log              ← run 2 log (RT-DETR reached mAP50=0.723 at ep 200)
└── train_full2.log             ← run 3 log (current; RT-DETR epoch ~70/200)
```

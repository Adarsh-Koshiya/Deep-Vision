# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project is

Deep Vision thesis (OTH Amberg-Weiden, Topic 25): compare **Diamond-DETR** and **RT-DETR-r18** (transformers) against **YOLOv8n** and **YOLO11n** (CNNs) on the 5-class Magnetic Tile Surface Defect dataset. Single ground truth: `REPORT.md`. Agent handoff doc: `PROJECT_NOTES.md`. **Always read PROJECT_NOTES.md before doing anything.**

---

## Commands

```bash
# Full training pipeline (all 4 models, ~15–20 hrs):
cd "/home/40a2/DV/Claude code"
screen -S train bash -c 'bash train.sh 2>&1 | tee train_full.log'

# Smoke test (5 epochs, verifies pipeline end-to-end):
EPOCHS=5 bash train.sh

# Force retrain even if outputs/metrics/*.json already exist:
FORCE=1 bash train.sh

# Monitor current run:
tail -f train_full2.log
tail -1 checkpoints/rtdetr_baseline/results.csv   # current epoch
nvidia-smi

# Manual evaluate.py — DETR models require PYTHONPATH:
cd diamond_detr/RTDETR-main
PYTHONPATH="/home/40a2/DV/Claude code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
  python /home/40a2/DV/Claude\ code/evaluate.py \
  --type rtdetr --weights checkpoints/rtdetr_baseline/weights/best.pt \
  --data tile_yolo/tile.yaml --name RT-DETR --out outputs/metrics

# YOLO models (from project root, no PYTHONPATH needed):
python evaluate.py --type yolo --weights checkpoints/yolov8n/weights/best.pt \
  --data tile_yolo/tile.yaml --name YOLOv8n --out outputs/metrics

# Rebuild dataset from scratch:
python prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo
```

---

## Critical: two incompatible Ultralytics versions

| Family | Ultralytics | Location | Key constraint |
|--------|-------------|----------|----------------|
| YOLO (YOLOv8n, YOLO11n) | stock 8.4.x | pip-installed | run from project root |
| DETR (RT-DETR, Diamond-DETR) | fork 8.0.201 | `diamond_detr/RTDETR-main/` | run with `PYTHONPATH="$DIAMOND:..."` |

They **cannot share a Python process**. `train.sh` handles this by running each model in a separate subprocess. `evaluate.py` also handles it via `sys.path.insert` for DETR models. Never `import ultralytics` for a DETR model without the fork on PYTHONPATH first.

---

## Architecture

```
train.sh          ─── orchestrates all 4 training runs sequentially
  │                   skips a model if outputs/metrics/<Name>.json exists
  │                   set FORCE=1 to override
  ├─ python (stock ultralytics)  ─── trains YOLOv8n / YOLO11n
  │     └─ evaluate.py --type yolo
  └─ python (fork, from RTDETR-main/)  ─── trains RT-DETR / Diamond-DETR
        └─ evaluate.py --type rtdetr (needs PYTHONPATH=$DIAMOND)

evaluate.py       ─── loads model → FPS benchmark → model.val(split=test) → JSON
prepare_data.py   ─── masks→YOLO boxes, stratified split → tile_yolo/
MagTile_DiamondDETR_Notebook.ipynb  ─── reads outputs/metrics/*.json ONLY (no model loading)
REPORT.md         ─── living report; fill ⏳ TODO markers from JSONs when training completes
```

Diamond-DETR = RT-DETR-r18 with 3 components swapped (see §3 of REPORT.md):
- **B**: `BasicBlock_Faster_Block_Rep` backbone (14.4M params, same channel dims)
- **C**: `ChannelAttention_HSFPN` neck
- **D**: `DRBC3` fusion blocks  
All extra modules are in `diamond_detr/RTDETR-main/ultralytics/extra_modules/block.py`.

---

## Fork patches — all applied to `diamond_detr/RTDETR-main/`

Must reapply if fork is re-cloned:

| # | File | Fix |
|---|------|-----|
| 1 | `utils/metrics.py` | `np.trapz` → `np.trapezoid` |
| 2 | `utils/torch_utils.py` | `torch.load(..., weights_only=False)` |
| 3 | `nn/tasks.py` | same `weights_only=False` (2 call sites) |
| 4 | `nn/tasks.py` | `attempt_load_one_weight` returns `(model, ckpt)` not just `model` |
| 5 | `cfg/models/rt-detr/Diamond-detr.yaml` | real paper arch (was wrong FasterNet-T2); **must keep top-level `nc: 80`** — its absence → `KeyError: 'nc'` at `tasks.py:263` |
| 6 | `nn/tasks.py` | `if isinstance(m, ConvNormLayer) and hasattr(m, 'norm'):` fuse guard |

Patch #6 is critical for `evaluate.py`: `model.predict()` (FPS benchmark) triggers fuse via AutoBackend; `model.val()` then calls fuse again and crashes without the guard.

---

## Current training state

RT-DETR is on its **third run** (screen `1869829.train`, `train_full2.log`). Second run reached **mAP50=0.723** at epoch 200 (`train_full.log`). Third run expects same result. Diamond-DETR starts automatically after RT-DETR finishes.

**Do not start a second training. Do not kill the screen session.**

CNN results are final:
- YOLOv8n: mAP50=0.403, mAP50-95=0.275, 205 FPS
- YOLO11n: mAP50=0.335, mAP50-95=0.204, 153 FPS

---

## Completing the report

Once both transformer JSONs exist in `outputs/metrics/`:
1. Fill every `⏳ TODO` in `REPORT.md` §1 (Abstract), §6 (Results table), §7 (Discussion), §11 (Conclusion) from the JSON values.
2. Confirm author name / matriculation in `REPORT.md` header.
3. Run the notebook: `jupyter notebook MagTile_DiamondDETR_Notebook.ipynb` (all cells).
4. Export: `pandoc REPORT.md -o REPORT.pdf`

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Can't get attribute 'ConvNormLayer'` | Missing fork on PYTHONPATH | `PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python evaluate.py ...` |
| `'ConvNormLayer' has no attribute 'norm'` | Double fuse (patch #6 missing) | Verify patch #6 is in `nn/tasks.py` |
| CUDA OOM on RT-DETR | Shared GPU contention | Check `nvidia-smi`; wait or use `batch=2` |
| Diamond-DETR mAP ≈ 0 | Expected — random init, no public pretrained weights | Document as Limitation 3 |
| train.sh skips a model unexpectedly | `outputs/metrics/<Name>.json` exists | `FORCE=1 bash train.sh` |

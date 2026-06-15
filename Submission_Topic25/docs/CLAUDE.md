# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## STYLE RULE (highest priority, applies to everything)

Never use the em-dash (Unicode U+2014) or the en-dash (Unicode U+2013) anywhere: not in the
report, the docs, code comments, the notebook, filenames, or chat replies. Use a comma, colon,
semicolon, or parentheses for a break in a sentence, and a plain hyphen for numeric ranges
(for example, write `0.38-0.58`, not the en-dash form). This is an explicit author requirement
and overrides any default formatting habit. Before saving any file, check that it contains zero
U+2014 and U+2013 characters.

---

## What this project is

Deep Vision project (OTH Amberg-Weiden, Topic 25): compare **Diamond-DETR** and **RT-DETR-r18** (transformers) against **YOLOv8n** and **YOLO11n** (CNNs) on the 5-class Magnetic Tile Surface Defect dataset. Primary deliverable: the LaTeX report `report/REPORT.tex` compiled to `report/REPORT.pdf`; `report/REPORT.md` is the working draft kept in sync. Design rationale / handoff doc: `docs/PROJECT_NOTES.md`; prioritised follow-ups: `docs/FUTURE_WORK.md`. **Read PROJECT_NOTES.md before doing anything.**

**Submission layout** (all paths in this file are relative to the repository root): scripts in `code/`, the bundled fork in `code/diamond_detr/`, trained checkpoints in `weights/`, metrics in `results/metrics/`, the dataset (not bundled) in `dataset/`, figures in `figures/`, the report in `report/`, the notebook in `notebook/`.

---

## Commands

All commands run from the repository root.

```bash
# Full training pipeline (all 4 models, ~15-20 hrs):
screen -S train bash -c 'bash code/train.sh 2>&1 | tee train_full.log'

# Smoke test (5 epochs, verifies pipeline end-to-end):
EPOCHS=5 bash code/train.sh

# Force retrain even if results/metrics/*.json already exist:
FORCE=1 bash code/train.sh

# Monitor a run:
tail -f train_full.log
tail -1 weights/rtdetr_baseline/results.csv # current epoch
nvidia-smi

# Manual evaluate.py, DETR models require the fork on PYTHONPATH:
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" \
 python code/evaluate.py \
 --type rtdetr --weights weights/rtdetr_baseline/best.pt \
 --data dataset/tile_yolo/tile.yaml --name RT-DETR --out results/metrics

# YOLO models (stock ultralytics, no PYTHONPATH needed):
python code/evaluate.py --type yolo --weights weights/yolov8n/best.pt \
 --data dataset/tile_yolo/tile.yaml --name YOLOv8n --out results/metrics

# Rebuild dataset from scratch (dataset/ not bundled, see README):
python code/prepare_data.py --src dataset/Magnetic-Tile-Defect --out dataset/tile_yolo
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
code/train.sh ─── orchestrates all 4 training runs sequentially
 │ skips a model if results/metrics/<Name>.json exists
 │ set FORCE=1 to override
 ├─ python (stock ultralytics) ─── trains YOLOv8n / YOLO11n
 │ └─ code/evaluate.py --type yolo
 └─ python (fork, from RTDETR-main/) ─── trains RT-DETR / Diamond-DETR
 └─ code/evaluate.py --type rtdetr (needs PYTHONPATH=$DIAMOND)

code/evaluate.py ─── loads model → FPS benchmark → model.val(split=test) → JSON
code/prepare_data.py ─ masks→YOLO boxes, stratified split → dataset/tile_yolo/
notebook/MagTile_DiamondDETR_Notebook.ipynb ─ reads results/metrics/*.json + figures/ ONLY (no model loading)
report/REPORT.tex ── typeset deliverable (xelatex to REPORT.pdf); REPORT.md is the synced working draft
```

Diamond-DETR = RT-DETR-r18 with 3 components swapped (see §3 of REPORT.md):
- **B**: `BasicBlock_Faster_Block_Rep` backbone (14.4M params, same channel dims)
- **C**: `ChannelAttention_HSFPN` neck
- **D**: `DRBC3` fusion blocks 
All extra modules are in `diamond_detr/RTDETR-main/ultralytics/extra_modules/block.py`.

---

## Fork patches, all applied to `diamond_detr/RTDETR-main/`

Must reapply if fork is re-cloned:

| # | File | Fix |
|---|------|-----|
| 1 | `utils/metrics.py` | `np.trapz` → `np.trapezoid` |
| 2 | `utils/torch_utils.py` | `torch.load(..., weights_only=False)` |
| 3 | `nn/tasks.py` | same `weights_only=False` (2 call sites) |
| 4 | `nn/tasks.py` | `attempt_load_one_weight` returns `(model, ckpt)` not just `model` |
| 5 | `cfg/models/rt-detr/Diamond-detr.yaml` | real paper arch (was wrong FasterNet-T2); **must keep top-level `nc: 80`**, its absence → `KeyError: 'nc'` at `tasks.py:263` |
| 6 | `nn/tasks.py` | `if isinstance(m, ConvNormLayer) and hasattr(m, 'norm'):` fuse guard |

Patch #6 is critical for `evaluate.py`: `model.predict()` (FPS benchmark) triggers fuse via AutoBackend; `model.val()` then calls fuse again and crashes without the guard.

---

## Training state, COMPLETE

All four models are trained and evaluated; results are final and locked in `report/REPORT.md` §6
and `results/metrics/*.json`. Nothing is training. Re-running requires the dataset (not bundled).

| Model | Test mAP50 | mAP50-95 | FPS | Params |
|-------|-----------:|---------:|----:|-------:|
| Diamond-DETR (random init, AdamW, 200 ep) | 0.762 | 0.489 | 56.5 | 14.3 M |
| RT-DETR-r18 (pretrained, AdamW, 200 ep) | 0.763 | 0.446 | 40.2 | 19.9 M |
| YOLOv8n (COCO, SGD, 150 ep) | 0.403 | 0.275 | 205 | 3.0 M |
| YOLO11n (COCO, SGD, 150 ep) | 0.335 | 0.204 | 153 | 2.6 M |

---

## Maintaining the report

The deliverable PDF is built from the hand-written LaTeX source `report/REPORT.tex` (the
professor-grade version). `report/REPORT.md` is the working draft of the same content. To rebuild:
1. Keep every number traceable to `results/metrics/*.json` / `weights/<model>/results.csv`.
2. Regenerate the result figures if metrics changed (the `make_figs` step writes
   `figures/fig_overall_map.png`, `fig_perclass_map.png`, `fig_speed_accuracy.png`).
3. Compile twice for refs and toc: `cd report && xelatex REPORT.tex && xelatex REPORT.tex`
   (figures resolve via `\graphicspath{{../figures/}}`).
4. Remember the STYLE RULE: no em-dash or en-dash, which in LaTeX also means never write
   `-` `-` (two hyphens) or three hyphens, since those become en/em dashes.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Can't get attribute 'ConvNormLayer'` | Missing fork on PYTHONPATH | `PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python evaluate.py ...` |
| `'ConvNormLayer' has no attribute 'norm'` | Double fuse (patch #6 missing) | Verify patch #6 is in `nn/tasks.py` |
| CUDA OOM on RT-DETR | Shared GPU contention | Check `nvidia-smi`; wait or use `batch=2` |
| Diamond-DETR mAP ≈ 0 | Trained with SGD instead of AdamW | Use AdamW for both DETRs (the shipped run did, and converged from random init) |
| train.sh skips a model unexpectedly | `results/metrics/<Name>.json` exists | `FORCE=1 bash code/train.sh` |

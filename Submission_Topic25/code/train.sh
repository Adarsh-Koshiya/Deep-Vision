#!/usr/bin/env bash
# train.sh, full training harness for the Diamond-DETR magnetic-tile study.
#
# Runs each model in the RIGHT environment, as a SEPARATE process, because the two
# model families need incompatible Ultralytics versions:
#   - DETR family (Diamond-DETR, RT-DETR): the FORK ultralytics 8.0.201, run from
#     code/diamond_detr/RTDETR-main/ (its local package shadows any installed one).
#   - YOLO family (YOLOv8, YOLO11): STOCK modern ultralytics (pip install ultralytics),
#     run from code/ (where the COCO init weights yolov8n.pt / yolo11n.pt live).
#
# Each model trains, then evaluate.py dumps test metrics to results/metrics/<name>.json.
# The notebook reads those JSON files, it never loads a model, so the version clash
# never reaches the notebook kernel.
#
# Layout (paths resolve relative to this script, which lives in code/):
#   <repo>/code/        this script, evaluate.py, prepare_data.py, diamond_detr/ (fork)
#   <repo>/dataset/     Magnetic-Tile-Defect/ (raw) + tile_yolo/ (generated), NOT bundled
#   <repo>/checkpoints/ Ultralytics run dir created here by training (per-model weights/results)
#   <repo>/weights/     curated copy: best.pt + results.csv + args.yaml per model (the submission set)
#   <repo>/results/metrics/  final test metrics written by evaluate.py
#
# Prereqs (see docs/Execution_Guide.md):
#   git clone https://github.com/bigboyliang/DIAMOND.git code/diamond_detr
#   pip install ultralytics            # stock, for the YOLO models + evaluate.py
#   (the fork provides its own ultralytics 8.0.201 locally, do not pip it)
#
# Usage:  bash code/train.sh           # full run (transformers 200 ep, CNNs default 200 ep)
#         EPOCHS=3 bash code/train.sh  # quick smoke test
#         FORCE=1 bash code/train.sh   # retrain even if results/metrics/<name>.json exists
set -euo pipefail

# Reduce CUDA memory fragmentation, the transformer models (RT-DETR ~80M params,
# 240 GFLOPs) are tight on an 8 GB card and fragment easily.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE="$(cd "$(dirname "$0")" && pwd)"          # <repo>/code
ROOT="$(cd "$CODE/.." && pwd)"                 # <repo> root
DIAMOND="$CODE/diamond_detr/RTDETR-main"       # forked ultralytics 8.0.201
DATA="$ROOT/dataset/tile_yolo/tile.yaml"
CKPT="$ROOT/checkpoints"                       # Ultralytics run dir; curated copies live in weights/
METRICS="$ROOT/results/metrics"

# Batch 4 = the paper's intended config (Diamond-DETR repo default.yaml, Execution_Guide,
# notebook all specify batch=4). Verified to fit on the 8 GB RTX 3060 Ti: RT-DETR-r18
# peaks at ~6.3 GB / 7.65 GB at batch 4. Batch 8 OOMs (do not raise); batch 2 also works
# but needlessly deviates from the paper. All four models share this batch for a fair comparison.
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
HP="optimizer='SGD', lr0=1e-4, momentum=0.8, weight_decay=1e-4, seed=42"
HP_DETR="optimizer='AdamW', lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42"
# RT-DETR pretrained checkpoint: paper's own RTDETR-r18 trained on diamond dataset.
# nc=2 in checkpoint vs nc=5 here, Ultralytics intersect_dicts partial-loads backbone/neck,
# reinits the final detection head. Same pattern as YOLO COCO→custom fine-tuning.
RTDETR_PT="$CODE/diamond_detr/adamas/runs/train/RTDETR-r18/weights/best.pt"

mkdir -p "$CKPT" "$METRICS"

# ── Step 0: data ────────────────────────────────────────────────────────────
if [ ! -f "$DATA" ]; then
  echo "[data] building dataset/tile_yolo/ ..."
  python "$CODE/prepare_data.py" --src "$ROOT/dataset/Magnetic-Tile-Defect" --out "$ROOT/dataset/tile_yolo"
fi
# Fix absolute path in tile.yaml for this machine (safe to run every time)
sed -i "s|^path:.*|path: $ROOT/dataset/tile_yolo|" "$DATA"

# ── CNN baselines: stock ultralytics, from code/ (where the COCO .pt weights live) ────
cd "$CODE"
for spec in "yolov8n.pt:yolov8n:YOLOv8n" "yolo11n.pt:yolo11n:YOLO11n"; do
  CKPT_NAME="${spec#*:}"; CKPT_NAME="${CKPT_NAME%:*}"
  DISP="${spec##*:}"; WEIGHTS="${spec%%:*}"
  if [ -f "$METRICS/$DISP.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $DISP, metrics already exist (set FORCE=1 to retrain)"; continue
  fi
  echo "[train] $DISP (stock ultralytics)"
  python - "$WEIGHTS" "$CKPT_NAME" <<PYEOF
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).train(data="$DATA", epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
                        device=$DEVICE, project="$CKPT", name=sys.argv[2], exist_ok=True, $HP)
PYEOF
  python "$CODE/evaluate.py" --type yolo \
    --weights "$CKPT/$CKPT_NAME/weights/best.pt" --data "$DATA" \
    --name "$DISP" --imgsz "$IMGSZ" --out "$METRICS"
done

# ── Transformer models: the fork, from RTDETR-main/ ───────────────────────────
cd "$DIAMOND"
if [ -f "$METRICS/RT-DETR.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[skip] RT-DETR, metrics already exist (set FORCE=1 to retrain)"
else
  echo "[train] RT-DETR-r18 (pretrained from paper's diamond checkpoint, fork ultralytics 8.0.201)"
  python - "$RTDETR_PT" <<PYEOF
import sys
from ultralytics import RTDETR
RTDETR(sys.argv[1]).train(data="$DATA", epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
                          device=$DEVICE, project="$CKPT", name="rtdetr_baseline", exist_ok=True,
                          plots=False, $HP_DETR)
PYEOF
  # RT-DETR checkpoint uses fork-only modules (ConvNormLayer etc), must run with fork on PYTHONPATH
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$CODE/evaluate.py" --type rtdetr \
    --weights "$CKPT/rtdetr_baseline/weights/best.pt" --data "$DATA" \
    --name "RT-DETR" --imgsz "$IMGSZ" --out "$METRICS"
fi

if [ -f "$METRICS/Diamond-DETR.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[skip] Diamond-DETR, metrics already exist (set FORCE=1 to retrain)"
else
  echo "[train] Diamond-DETR (random init, no public pretrained weights, fork ultralytics 8.0.201)"
  python - <<PYEOF
from ultralytics.models.rtdetr.train import RTDETRTrainer
RTDETRTrainer(overrides=dict(
    model="Diamond-detr.yaml", data="$DATA", pretrained=True,
    epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ, device=$DEVICE,
    optimizer="AdamW", lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42,
    project="$CKPT", name="diamond_detr", exist_ok=True, plots=False,
)).train()
PYEOF
  # Diamond-DETR checkpoint uses fork-only ultralytics.nn.backbone, must run with fork on PYTHONPATH
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$CODE/evaluate.py" --type rtdetr \
    --weights "$CKPT/diamond_detr/weights/best.pt" --data "$DATA" \
    --name "Diamond-DETR" --imgsz "$IMGSZ" --out "$METRICS"
fi

cd "$ROOT"
echo ""
echo "Done. Metrics in $METRICS/*.json, open notebook/MagTile_DiamondDETR_Notebook.ipynb to analyse."
echo "Curated per-model best.pt / results.csv / args.yaml live in $ROOT/weights/<model>/."

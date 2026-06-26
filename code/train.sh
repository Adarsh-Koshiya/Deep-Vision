#!/usr/bin/env bash
# Train all four models and write their test metrics to outputs/metrics/<name>.json.
# The two model families need incompatible Ultralytics versions, so each trains in its own
# process: the DETRs (Diamond-DETR, RT-DETR) on the forked 8.0.201 in diamond_detr/RTDETR-main/,
# the YOLOs on the stock 8.4.x install.
# Prerequisites: clone https://github.com/bigboyliang/DIAMOND.git as diamond_detr, and pip install ultralytics.
# Run with: bash train.sh   (EPOCHS=3 for a quick smoke test, FORCE=1 overwrites existing metrics).
set -euo pipefail

# Reduce CUDA memory fragmentation. The transformers rely on deformable attention, which is
# memory-hungry and fragments an 8 GB card even though RT-DETR-r18 is only about 20M params.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ="$(cd "$(dirname "$0")" && pwd)"
DIAMOND="$PROJ/diamond_detr/RTDETR-main"
DATA="$PROJ/tile_yolo/tile.yaml"
CKPT="$PROJ/checkpoints"
METRICS="$PROJ/outputs/metrics"

# Batch 4 = the paper's config; fits the 8 GB GPU (RT-DETR peaks ~6.3 GB). Batch 8 OOMs. Shared by all 4.
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
HP="optimizer='SGD', lr0=1e-4, momentum=0.8, weight_decay=1e-4, seed=42"
HP_DETR="optimizer='AdamW', lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42"
# Pretrained checkpoint for RT-DETR: the paper's own RTDETR-r18, trained on the diamond data.
# It carries nc=2 against nc=5 here, so Ultralytics intersect_dicts partial-loads the backbone
# and neck and re-initialises the detection head (the usual COCO-to-custom fine-tuning pattern).
RTDETR_PT="$PROJ/diamond_detr/adamas/runs/train/RTDETR-r18/weights/best.pt"

mkdir -p "$CKPT" "$METRICS"

# ── Step 0: data ────────────────────────────────────────────────────────────
if [ ! -f "$DATA" ]; then
  echo "[data] building tile_yolo/ ..."
  python "$PROJ/prepare_data.py" --src "$PROJ/Magnetic-Tile-Defect" --out "$PROJ/tile_yolo"
fi
# Fix absolute path in tile.yaml for this machine (safe to run every time)
sed -i "s|^path:.*|path: $PROJ/tile_yolo|" "$DATA"

# ── CNN baselines: stock ultralytics, from project root ───────────────────────
cd "$PROJ"
for spec in "yolov8n.pt:yolov8n:YOLOv8n" "yolo11n.pt:yolo11n:YOLO11n"; do
  CKPT_NAME="${spec#*:}"; CKPT_NAME="${CKPT_NAME%:*}"
  DISP="${spec##*:}"; WEIGHTS="${spec%%:*}"
  if [ -f "$METRICS/$DISP.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $DISP — metrics already exist (set FORCE=1 to retrain)"; continue
  fi
  echo "[train] $DISP (stock ultralytics)"
  python - "$WEIGHTS" "$CKPT_NAME" <<PYEOF
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).train(data="$DATA", epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
                        device=$DEVICE, project="$CKPT", name=sys.argv[2], exist_ok=True, $HP)
PYEOF
  python "$PROJ/evaluate.py" --type yolo \
    --weights "$CKPT/$CKPT_NAME/weights/best.pt" --data "$DATA" \
    --name "$DISP" --imgsz "$IMGSZ" --out "$METRICS"
done

# ── Transformer models: the fork, from RTDETR-main/ ───────────────────────────
cd "$DIAMOND"
if [ -f "$METRICS/RT-DETR.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[skip] RT-DETR — metrics already exist (set FORCE=1 to retrain)"
else
  echo "[train] RT-DETR-r18 (pretrained from paper's diamond checkpoint, fork ultralytics 8.0.201)"
  python - "$RTDETR_PT" <<PYEOF
import sys
from ultralytics import RTDETR
RTDETR(sys.argv[1]).train(data="$DATA", epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ,
                          device=$DEVICE, project="$CKPT", name="rtdetr_baseline", exist_ok=True,
                          plots=False, $HP_DETR)
PYEOF
  # The RT-DETR checkpoint uses fork-only modules (ConvNormLayer and friends), so the fork must be on PYTHONPATH
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/evaluate.py" --type rtdetr \
    --weights "$CKPT/rtdetr_baseline/weights/best.pt" --data "$DATA" \
    --name "RT-DETR" --imgsz "$IMGSZ" --out "$METRICS"
fi

if [ -f "$METRICS/Diamond-DETR.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[skip] Diamond-DETR — metrics already exist (set FORCE=1 to retrain)"
else
  echo "[train] Diamond-DETR (pretrained=True for backbone init, fork ultralytics 8.0.201)"
  python - <<PYEOF
from ultralytics.models.rtdetr.train import RTDETRTrainer
RTDETRTrainer(overrides=dict(
    model="Diamond-detr.yaml", data="$DATA", pretrained=True,
    epochs=$EPOCHS, batch=$BATCH, imgsz=$IMGSZ, device=$DEVICE,
    optimizer="AdamW", lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42,
    project="$CKPT", name="diamond_detr", exist_ok=True, plots=False,
)).train()
PYEOF
  # The Diamond-DETR checkpoint uses fork-only ultralytics.nn.backbone, so the fork must be on PYTHONPATH
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/evaluate.py" --type rtdetr \
    --weights "$CKPT/diamond_detr/weights/best.pt" --data "$DATA" \
    --name "Diamond-DETR" --imgsz "$IMGSZ" --out "$METRICS"
fi

cd "$PROJ"
echo ""
echo "Done. Metrics in $METRICS/*.json — see report/REPORT.pdf for the analysis."

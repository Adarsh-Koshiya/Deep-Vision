#!/usr/bin/env bash
# multiseed.sh — re-train all 4 final models under one extra seed for mean±std (Task 3).
#
# Each model keeps its exact final recipe; only the seed and output dir change.
#   SEED=1 bash experiments/scripts/multiseed.sh   → outputs/metrics_seed1/*.json
#   SEED=2 bash experiments/scripts/multiseed.sh   → outputs/metrics_seed2/*.json
# Then: python experiments/scripts/aggregate_seeds.py   (mean±std across seed42 + seedN dirs)
#
# Cost per seed: 2 CNNs (~40 min) + 2 transformers (~5–7 h) ≈ 6–8 h on a free GPU. Run in screen.
# Statistical note: seed 42 results already live in outputs/metrics/. Add seeds 1,2(,3) for n≥3.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED="${SEED:?set SEED, e.g. SEED=1 bash multiseed.sh}"

PROJ="/home/40a2/DV/Code"
DIAMOND="$PROJ/diamond_detr/RTDETR-main"
DATA="$PROJ/tile_yolo/tile.yaml"
OUT="$PROJ/outputs/metrics_seed$SEED"
CKPT="$PROJ/checkpoints/seed$SEED"
RTDETR_PT="$PROJ/diamond_detr/adamas/runs/train/RTDETR-r18/weights/best.pt"
mkdir -p "$OUT" "$CKPT"

run_done() { [ -f "$OUT/$1.json" ] && [ "${FORCE:-0}" != "1" ]; }

# ── CNNs (stock ultralytics, SGD, 150 ep, COCO-pretrained) ──
cd "$PROJ"
for spec in "yolov8n.pt:yolov8n:YOLOv8n" "yolo11n.pt:yolo11n:YOLO11n"; do
  w="${spec%%:*}"; ck="${spec#*:}"; ck="${ck%:*}"; name="${spec##*:}"
  if run_done "$name"; then echo "[skip] seed$SEED $name"; continue; fi
  echo "[train] seed$SEED $name"
  python - "$w" "$ck" <<PYEOF
import sys
from ultralytics import YOLO
YOLO(sys.argv[1]).train(data="$DATA", epochs=150, batch=4, imgsz=640, device=0,
    project="$CKPT", name=sys.argv[2], exist_ok=True,
    optimizer="SGD", lr0=1e-4, momentum=0.8, weight_decay=1e-4, seed=$SEED)
PYEOF
  python "$PROJ/evaluate.py" --type yolo --weights "$CKPT/$ck/weights/best.pt" \
    --data "$DATA" --name "$name" --imgsz 640 --out "$OUT"
done

# ── Transformers (fork, AdamW, 200 ep) ──
cd "$DIAMOND"
if ! run_done "RT-DETR"; then
  echo "[train] seed$SEED RT-DETR (pretrained init)"
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python - "$RTDETR_PT" <<PYEOF
import sys
from ultralytics import RTDETR
RTDETR(sys.argv[1]).train(data="$DATA", epochs=200, batch=4, imgsz=640, device=0,
    project="$CKPT", name="rtdetr_baseline", exist_ok=True, plots=False,
    optimizer="AdamW", lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=$SEED)
PYEOF
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/evaluate.py" --type rtdetr \
    --weights "$CKPT/rtdetr_baseline/weights/best.pt" --data "$DATA" --name "RT-DETR" --imgsz 640 --out "$OUT"
fi
if ! run_done "Diamond-DETR"; then
  echo "[train] seed$SEED Diamond-DETR (from scratch)"
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python - <<PYEOF
from ultralytics.models.rtdetr.train import RTDETRTrainer
RTDETRTrainer(overrides=dict(model="Diamond-detr.yaml", data="$DATA",
    epochs=200, batch=4, imgsz=640, device=0, optimizer="AdamW",
    lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=$SEED,
    project="$CKPT", name="diamond_detr", exist_ok=True, plots=False)).train()
PYEOF
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/evaluate.py" --type rtdetr \
    --weights "$CKPT/diamond_detr/weights/best.pt" --data "$DATA" --name "Diamond-DETR" --imgsz 640 --out "$OUT"
fi

echo ""
echo "Done seed $SEED → $OUT/  (run aggregate_seeds.py once you have ≥2 seeds)"

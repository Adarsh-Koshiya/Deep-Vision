#!/usr/bin/env bash
# Component ablation of Diamond-DETR (backbone B, neck C, fusion D).
# Trains A, A+B, A+C, A+D and A+B+C from random init (AdamW, lr 1e-4, 200 ep, batch 4, seed 42) and
# reuses the full model as A+B+C+D. Together with A+B+D and A+C+D these are the eight configurations
# of the paper's Table 2 factorial. A row is skipped if its JSON already exists (FORCE=1 retrains);
# wait_gpu blocks until 7 GB of GPU memory is free.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ="/home/40a2/DV/Code"
DIAMOND="$PROJ/diamond_detr/RTDETR-main"
DATA="$PROJ/tile_yolo/tile.yaml"
OUT="$PROJ/outputs/metrics_ablation"
CKPT="$PROJ/checkpoints/ablation"
ABL="ultralytics/cfg/models/rt-detr/ablation"
HP="optimizer='AdamW', lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42"
mkdir -p "$OUT" "$CKPT"
cd "$DIAMOND"

# Pre-flight guard: abort if any image is shared between the train and test splits.
ov=$(comm -12 <(ls "$PROJ/tile_yolo/images/train"|sort) <(ls "$PROJ/tile_yolo/images/test"|sort)|wc -l)
[ "$ov" -eq 0 ] || { echo "ABORT: train/test overlap=$ov — dataset leaked. Run prepare_data.py."; exit 1; }

# Wait until the shared GPU has enough free memory for batch-4 (~6.5 GB peak; require 7 GB headroom).
# Proceeds immediately if enough is free; only blocks while busy, grabbing the GPU within ~2s of it freeing.
wait_gpu() {
  local need="${1:-7000}" free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${free:-0}" -ge "$need" ] && break
    echo "[wait $(date +%H:%M:%S)] GPU ${free}MB free (<${need}MB) — waiting for the GPU to free; retry in 2s"; sleep 2
  done
}

# row_name:config_basename  (config lives in $ABL/<basename>.yaml)
for spec in "A:rtdetr_r18_paper" "A+B:diamond_AB" "A+C:diamond_AC" "A+D:diamond_AD" "A+B+C:diamond_ABC"; do
  name="${spec%%:*}"; cfg="${spec##*:}"
  if [ -f "$OUT/$name.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $name — metrics exist (FORCE=1 to retrain)"; continue
  fi
  wait_gpu 7000
  echo "[train] $name  ($cfg.yaml)  from scratch, AdamW, batch 4, 200ep"
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python - "$ABL/$cfg.yaml" "$cfg" <<PYEOF
import sys
from ultralytics import RTDETR
RTDETR(sys.argv[1]).train(data="$DATA", epochs=200, batch=4, imgsz=640, device=0,
                          project="$CKPT", name=sys.argv[2], exist_ok=True, plots=False, $HP)
PYEOF
  PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/evaluate.py" --type rtdetr \
    --weights "$CKPT/$cfg/weights/best.pt" --data "$DATA" --name "$name" --imgsz 640 --out "$OUT"
done

# A+B+C+D point = the already-trained full Diamond-DETR (batch 4, no need to retrain)
[ -f "$OUT/A+B+C+D.json" ] || cp "$PROJ/outputs/metrics/Diamond-DETR.json" "$OUT/A+B+C+D.json" 2>/dev/null || \
  echo "WARN: outputs/metrics/Diamond-DETR.json missing — full point not copied"

echo ""
echo "Done. Ablation metrics in $OUT/ (A, A+B, A+C, A+D, A+B+C, A+B+C+D)."

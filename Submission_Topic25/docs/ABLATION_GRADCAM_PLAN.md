# Plan, Ablation study & Grad-CAM (the two remaining notebook deliverables)

Status: **Grad-CAM DONE ✅ · Ablation NOT YET RUN.**
- Grad-CAM is complete and shipped in **`figures/`** (`grad_cam.py`, `overlays/`,
 `gradcam_comparison.png`) using the official `jacobgil/pytorch-grad-cam` library. See Part B.
 The notebook §6 displays `figures/gradcam_comparison.png` directly.
- The ablation study (Part A) is an optional extension that still needs free GPU time. Notebook §5
 shows an honest "not yet run" message until `results/metrics_ablation/*.json` exists.

Paths below use the submission layout. Repo root = the folder containing `code/ docs/ results/
weights/`. DETR fork: `code/diamond_detr/RTDETR-main` (run with `PYTHONPATH="$DIAMOND:..."`).
Both tasks reuse the existing trained checkpoints in `weights/` where possible.

---

## Part A, Ablation study (isolate Diamond-DETR's 3 components)

### What we are measuring
Diamond-DETR = RT-DETR-r18 baseline (**A**) + three components, all visible in
`diamond_detr/RTDETR-main/ultralytics/cfg/models/rt-detr/Diamond-detr.yaml`:

| Tag | Component | Module(s) in the YAML |
|-----|-----------|------------------------|
| **B** | RepFasterNet backbone | `Blocks [..., BasicBlock_Faster_Block_Rep...]` (backbone lines 4-7) |
| **C** | HS-FPN attention neck | `ChannelAttention_HSFPN` + `Multiply`/`Add` (head lines 11-30) |
| **D** | DRBC3 fusion blocks | `DRBC3 [256, 0.5]` (head lines 14, 22, 30) |

Target configs (the paper's Table 2 shape): `A`, `A+B`, `A+C`, `A+D`, `A+B+C`, `A+B+C+D` (= full Diamond-DETR).

### Step 1, author the 6 config YAMLs ← the real work, do this carefully
Create `diamond_detr/RTDETR-main/ultralytics/cfg/models/rt-detr/ablation/<config>.yaml`.
- **A** = copy `rtdetr-r18.yaml` (standard ResNet-18 backbone + standard RepC3 neck). Baseline.
- **A+B+C+D** = the existing `Diamond-detr.yaml` (already trained, reuse, see Step 3).
- **A+B** = rtdetr-r18 neck (RepC3, no HS-FPN) **+** `BasicBlock_Faster_Block_Rep` backbone.
- **A+C** = rtdetr-r18 backbone (BasicBlock) **+** HS-FPN neck, but DRBC3→RepC3.
- **A+D** = rtdetr-r18 backbone **+** standard neck with DRBC3 fusion blocks.
- **A+B+C** = B backbone + HS-FPN neck, DRBC3→RepC3.

⚠️ **These are NOT 1-line module swaps.** HS-FPN changes the neck's connectivity (it adds
`Multiply`/`Add` branches with specific `from:` indices); removing it means reverting those
head lines to the plain `rtdetr-r18` head, not just renaming a module. After writing each
YAML, **validate it builds and does a forward pass before training** (cheap, no full run):

```bash
cd diamond_detr/RTDETR-main
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python - <<'PY'
from ultralytics import RTDETR
m = RTDETR("ultralytics/cfg/models/rt-detr/ablation/A+B.yaml") # one per config
import torch; m.model(torch.zeros(1,3,640,640)) # must not raise
print("OK", sum(p.numel() for p in m.model.parameters())/1e6, "M params")
PY
```
Keep the top-level `nc: 80` key in every YAML (fork bug, see CLAUDE.md patch #5; missing → `KeyError: 'nc'`).

### Step 2, fair-init caveat (important for the report)
The ablation must be **init-controlled**: train **all 6 configs from scratch** with the same
recipe (AdamW, 200 epochs, batch 4, seed 42). Do **not** reuse the *pretrained* RT-DETR
(`results/metrics/RT-DETR.json`, 0.763) as config A, that one had a diamond-pretraining
advantage and would make B/C/D look falsely weak. The existing **Diamond-DETR** result
(0.762, from scratch) *is* a valid `A+B+C+D` point and can be reused directly.

### Step 3, train + evaluate each config → `scripts/ablation.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROJ="$(cd "$(dirname "$0")/.." && pwd)" # repo root (script lives in code/ or a scripts/ dir)
DIAMOND="$PROJ/code/diamond_detr/RTDETR-main"
DATA="$PROJ/dataset/tile_yolo/tile.yaml"; OUT="$PROJ/results/metrics_ablation"
mkdir -p "$OUT"; cd "$DIAMOND"
HP="optimizer='AdamW', lr0=1e-4, momentum=0.9, weight_decay=1e-4, seed=42"

for cfg in A_rtdetr A+B A+C A+D A+B+C; do # A+B+C+D reuse diamond_detr result
 [ -f "$OUT/$cfg.json" ] && { echo "[skip] $cfg"; continue; }
 YAML="ultralytics/cfg/models/rt-detr/ablation/$cfg.yaml"
 python - "$YAML" "$cfg" <<PYEOF
import sys
from ultralytics import RTDETR
RTDETR(sys.argv[1]).train(data="$DATA", epochs=200, batch=4, imgsz=640, device=0,
 project="$PROJ/weights/ablation", name=sys.argv[2],
 exist_ok=True, plots=False, $HP)
PYEOF
 PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/code/evaluate.py" --type rtdetr \
 --weights "$PROJ/weights/ablation/$cfg/weights/best.pt" \
 --data "$DATA" --name "$cfg" --imgsz 640 --out "$OUT"
done
# copy the already-trained full model as the A+B+C+D point:
cp "$PROJ/results/metrics/Diamond-DETR.json" "$OUT/A+B+C+D_diamond.json"
```
Run in a screen session (GPU must be free): `screen -dmS abl bash -c 'bash scripts/ablation.sh 2>&1 | tee abl.log'`

### Step 4, view results
Notebook §5 (cell already updated) auto-loads `results/metrics_ablation/*.json` and draws the
ΔmAP50 bar chart once all 6 files exist. Then update REPORT.md with a real Table 2.

### Cost
~5 from-scratch DETR trainings × ~2.5 h each ≈ **12-15 h GPU** (A+B+C+D reused). Single 8 GB
GPU, batch 4. Watch for shared-GPU contention (`nvidia-smi`).

---

## Part B, Grad-CAM attention maps ✅ DONE

**Completed and shipped in `figures/`.** Used the official **`jacobgil/pytorch-grad-cam`** library
via `figures/grad_cam.py`: each detector wrapped to emit `(B, nc)` class scores, `GradCAM` on the
deepest backbone conv, `ClassifierOutputTarget(top_class)`. Outputs: `figures/overlays/*.png`
(20 maps) + `figures/gradcam_comparison.png` (5 classes × [input + 4 models]).

**Findings:** transformers (esp. Diamond-DETR) give focused, defect-aligned maps (sharp line on the
crack); CNNs are diffuse. The top predicted class also separates them, both transformers classify
all 5 defects correctly, YOLO11n collapses to class 0, mirroring the rare-class AP gap.

**Embedded in the notebook §6:** the cell displays `figures/gradcam_comparison.png` directly (no
copy step needed).

The original (pre-library) procedure below is retained for reference.

### What we had
`figures/grad_cam.py` iterates all 5 class images per model and saves overlays to
`figures/overlays/`. It uses `pytorch-grad-cam` on the deepest backbone conv. Reuse the trained
checkpoints in `weights/`, **no retraining**, inference only (minutes).

### Step 1, pick representative images
One clear example per defect class from the test split, e.g.:
```bash
ls dataset/tile_yolo/images/test/ | grep -iE "Blowhole|Break|Crack|Fray|Uneven" | head
```

### Step 2, generate per-model overlays (run from the repo root)
```bash
#!/usr/bin/env bash
set -euo pipefail
PROJ="$(pwd)"; DIAMOND="$PROJ/code/diamond_detr/RTDETR-main"
# DETR models, fork on PYTHONPATH:
for m in "diamond_detr:Diamond-DETR" "rtdetr_baseline:RT-DETR"; do
 ck="${m%%:*}"; nm="${m##*:}"
 PYTHONPATH="$DIAMOND:${PYTHONPATH:-}" python "$PROJ/figures/grad_cam.py" --type rtdetr \
 --weights "$PROJ/weights/$ck/best.pt" --name "$nm"
done
# CNN models, stock ultralytics:
for m in "yolov8n:YOLOv8n" "yolo11n:YOLO11n"; do
 ck="${m%%:*}"; nm="${m##*:}"
 python "$PROJ/figures/grad_cam.py" --type yolo --weights "$PROJ/weights/$ck/best.pt" --name "$nm"
done
```
`grad_cam.py` honours `DV_REPO` (repo root) and `DV_TESTDIR` (test-split images) env vars to
override its defaults (see the script header) when the dataset lives elsewhere.

### Step 3, compose the comparison figure → `figures/gradcam_comparison.png`
A tiny compositor stitches the per-class overlays into the 5×(input + 4 models) grid. The shipped
`figures/gradcam_comparison.png` already contains this grid; regenerate it only if the overlays
change. Notebook §6 displays `figures/gradcam_comparison.png` automatically.

### Known risks (document, don't hide)
- **pytorch-grad-cam on detectors is finicky:** there is no single class logit. The default
 target may give weak maps; you may need a custom `targets=` (e.g. sum of objectness/cls for
 the top box). If maps look uniform, this is why, tune `--layer` and the CAM target.
- **DETR vs YOLO layer indices differ**, print `model.model` and set `--layer` per model
 (last backbone conv: RT-DETR/Diamond ≈ backbone stage output; YOLO ≈ index 9/10).
- The fork's `ConvNormLayer` may need the patch-#6 fuse guard to survive `model.eval()`.

### Cost
Inference only, **~5-10 min** once images and layers are chosen (plus iteration on layer/target).

---

## Recommended order
1. ~~Grad-CAM~~ ✅ DONE (in `figures/`).
2. Ablation, still pending (12-15 h GPU, needs careful YAML authoring + validation).
Both are *optional extensions*, the core 4-model comparison (REPORT.md §6/§7) is already complete and valid without them.

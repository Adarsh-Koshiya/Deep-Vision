# Cross-Domain Evaluation of Diamond-DETR for Surface-Defect Detection

*Transformers versus CNNs on the Magnetic Tile Defect Dataset*

**Author:** Adarsh Koshiya (a.koshiya@oth-aw.de)
**Course:** Deep Vision — Comparison of Transformers and CNNs
**University:** Ostbayerische Technische Hochschule (OTH) Amberg-Weiden
**Term:** Sommersemester 2026

---

## Project Overview

This project tests whether **Diamond-DETR** — a lightweight RT-DETR variant from Yan et al.,
*Scientific Reports* **16**, 10711 (2026), designed for synthetic-diamond surface inspection — transfers
to a different industrial domain: the **Magnetic Tile Surface Defect** dataset (five classes: blowhole,
break, crack, fray, uneven). It is benchmarked against its own RT-DETR-r18 baseline and two compact CNN
detectors (YOLOv8n, YOLO11n) under matched conditions, and its three component swaps are ablated as a
complete 2³ factorial.

**Key results:**
- Both transformers reach ~0.76 mAP50 against ~0.40 for the best CNN (YOLOv8n), and — the headline of the
  study — they keep every class at or above 0.36 AP where the CNNs collapse on the rare *break* and *fray*
  defects. Recall, not precision, is where the transformers win (0.82 vs 0.23–0.35).
- Under matched **from-scratch** initialisation, Diamond-DETR ties RT-DETR-r18 on mAP50 at **28% fewer
  parameters** (14.3 M vs 19.9 M). This is an efficiency result, not an accuracy win — the headline
  four-model table is *not* initialisation-controlled (Diamond-DETR is random-init, the baseline RT-DETR
  is diamond-pretrained), so a from-scratch RT-DETR baseline was trained to compare like with like.
- The component story does **not** transfer cleanly from diamonds: the HS-FPN neck — the source paper's
  single strongest component on diamonds — hurts in most tile contexts (three of four pairings), so the
  full Diamond-DETR is no longer the best configuration. A smaller trim, **A+C+D (0.587 mAP50-95)**, beats
  the **full model (0.489)**.

Every number in this README and the report is read from `results/metrics/*.json` and
`results/metrics_ablation/*.json`; none is entered by hand. Read **`REPORT.pdf`** for the full analysis.

---

## Project Structure

    Koshiya_Adarsh_DCV2026/
    |-- REPORT.pdf                   Final report (READ THIS FIRST)
    |-- README.md                    This file
    |-- requirements.txt
    |-- code/
    |   |-- prepare_data.py          Masks -> YOLO boxes + stratified 70/20/10 split
    |   |-- train.sh                 Trains all 4 models (each family in its own process)
    |   |-- evaluate.py              Test-split eval + FPS -> metrics JSON
    |   |-- clean_fork_warnings.py   Cosmetic: silences the fork's deprecation warnings (no result change)
    |   |-- experiments/
    |   |   |-- RUNBOOK.md            How to run the ablation and multi-seed experiments
    |   |   `-- scripts/             ablation.sh, multiseed.sh, aggregate_seeds.py, validate_cfg.py
    |   `-- diamond_detr/            Third-party DETR fork (see code/diamond_detr/README.md)
    |       `-- RTDETR-main/         Vendored Ultralytics 8.0.201 + the Diamond-detr.yaml configs
    |-- results/
    |   |-- metrics/                 The 4 main models (source of every headline number)
    |   `-- metrics_ablation/        Full 2^3 ablation (8 configurations)
    |-- figures/                     The 5 report figures (PNG)
    `-- gradcam_analysis/            Grad-CAM code + 20 overlays + the comparison grid

Two things are **outside this archive** (per the submission instructions):

| What | Where |
|------|-------|
| Trained weights — the four `best.pt` files | GitLab: https://git.oth-aw.de/40a2/deep-vision |
| Dataset (public) | https://github.com/abin24/Magnetic-tile-defect-datasets |

---

## Requirements

Python 3.12 (tested). Install dependencies:

    pip install -r requirements.txt

**Important — two Ultralytics versions are needed and cannot share one Python process:**
- **YOLOv8n / YOLO11n** use stock `ultralytics>=8.4` (installed by the command above).
- **RT-DETR / Diamond-DETR** use a forked `ultralytics 8.0.201` that is **bundled** under
  `code/diamond_detr/RTDETR-main/` and imported via `PYTHONPATH` (do not pip-install it).

`train.sh` and `evaluate.py` already run each family in its own subprocess, so the two versions never
clash. The 6 compatibility patches applied to the fork (`np.trapz` → `np.trapezoid`, `weights_only=False`,
etc. — all compatibility fixes, no model change) are listed in `code/diamond_detr/README.md`.

---

## Dataset

Public — download from https://github.com/abin24/Magnetic-tile-defect-datasets , then build the split:

    python code/prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo

The dataset ships 1,344 grayscale tile images annotated with pixel masks (five defect types plus 952
defect-free tiles). `prepare_data.py` converts each connected mask region to a YOLO box and writes a
seed-42 stratified split of **938 train / 268 val / 138 test** images (zero image overlap between splits,
verified). It is the only canonical generator; re-running it clears its output tree before writing.

---

## Model Weights

The four trained models (`best.pt` for Diamond-DETR, RT-DETR-r18, YOLOv8n, YOLO11n) are in the GitLab
repository:

    git clone https://git.oth-aw.de/40a2/deep-vision.git

They are stored as plain files (no Git LFS needed).

---

## Running the Code

    # 1. Build the dataset (see Dataset above)
    python code/prepare_data.py --src Magnetic-Tile-Defect --out tile_yolo

    # 2. Train all four models (several hours on an 8 GB GPU; FORCE=1 overwrites existing metrics,
    #    EPOCHS=3 for a quick smoke test)
    bash code/train.sh

    # 3. Evaluate one model (writes <name>.json to --out)
    #    CNNs (stock Ultralytics):
    python code/evaluate.py --type yolo --weights <yolov8n.pt> \
        --data tile_yolo/tile.yaml --name YOLOv8n --out results/metrics
    #    DETRs (need the fork on PYTHONPATH):
    PYTHONPATH="code/diamond_detr/RTDETR-main" python code/evaluate.py --type rtdetr \
        --weights <diamond.pt> --data tile_yolo/tile.yaml --name Diamond-DETR --out results/metrics

    # 4. Component ablation (8 configs) — see code/experiments/RUNBOOK.md
    bash code/experiments/scripts/ablation.sh

The committed numbers live in `results/`; re-running a script writes fresh JSONs under its own `outputs/`
directory, so the committed copies are never overwritten. The experiment scripts hardcode the original
training-machine path (`/home/40a2/DV/Code`) and need adjusting to run elsewhere — see `RUNBOOK.md`.

---

## Results Summary

Test split, 138 images (read from `results/metrics/*.json`):

    Model          Type         Params   mAP50   mAP50-95   P      R      FPS
    -------------  -----------  -------  ------  ---------  -----  -----  -----
    Diamond-DETR   Transformer   14.3M    0.762     0.489   0.745  0.746   56.5
    RT-DETR-r18    Transformer   19.9M    0.763     0.446   0.733  0.819   40.2
    YOLOv8n        CNN            3.0M    0.403     0.275   0.613  0.349  204.7
    YOLO11n        CNN            2.6M    0.335     0.204   0.579  0.227  153.2

Diamond-DETR is random-init; RT-DETR-r18 here is diamond-pretrained (the initialisation-controlled,
from-scratch RT-DETR scores 0.514 mAP50-95 — see `REPORT.pdf` §6.1).

Component ablation — full 2³ factorial, from random init (read from `results/metrics_ablation/*.json`).
Components: backbone **B** (RepFasterNet), neck **C** (HS-FPN), fusion **D** (DRBC3):

    Config          Components   mAP50   mAP50-95   Params   FPS
    --------------  ----------  ------  ---------  -------  -----
    A (baseline)    none         0.763     0.514    19.9M    60.7
    A+B             B            0.864     0.577    16.8M    59.6
    A+C             C            0.712     0.421    18.7M    54.6
    A+D             D            0.838     0.566    18.2M    62.9
    A+B+C           B,C          0.666     0.411    15.6M    56.1
    A+B+D           B,D          0.790     0.541    15.1M    64.6
    A+C+D           C,D          0.854     0.587    17.4M    59.0
    A+B+C+D (full)  B,C,D        0.762     0.489    14.3M    56.5

### Key Findings

1. Transformers far outperform the CNNs (mAP50 0.763 vs 0.403, a factor of ~1.9), driven by recall:
   recall jumps from 0.23–0.35 (CNNs) to 0.82. The CNNs essentially solve only the large, high-contrast
   *blowhole* class and collapse on the rare *break* and *fray* defects (AP ≈ 0.02–0.08); both transformers
   keep every class at or above 0.36.
2. Under matched from-scratch initialisation Diamond-DETR ties RT-DETR-r18 on mAP50 at 28% fewer parameters
   — an efficiency result, not a strict accuracy gain.
3. HS-FPN (the paper's strongest component on diamonds) hurts in most tile contexts, so the published full
   model is no longer optimal: the smaller A+C+D (0.587 mAP50-95) and A+B+D (0.541) both beat the full
   model (0.489). The full model's one genuine advantage is the smallest parameter count (14.3 M).
4. The transformers required AdamW in this from-scratch, five-class setup (an initial SGD run left them at
   ~0 mAP). The source paper used SGD on its easier binary diamond task, so the sensitivity is
   regime-specific, not a universal rule.

---

## Hardware

A single 8 GB GPU (RT-DETR peaks near 6.3 GB at batch 4, which is why batch 4 is shared across all models).
Python 3.12, torch 2.8.0+cu129.

---

## References

1. Diamond-DETR — X. Yan, S. Yang, S. Zhang, X. Li, A. Li, *Scientific Reports* **16**, 10711 (2026).
   https://doi.org/10.1038/s41598-026-44951-1
2. Source code — https://github.com/bigboyliang/DIAMOND
3. RT-DETR — Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," CVPR 2024. arXiv:2304.08069
4. Magnetic Tile dataset — Y. Huang, C. Qiu, K. Yuan, "Surface defect saliency of magnetic tile,"
   *The Visual Computer* **36**, 85–96 (2020). https://doi.org/10.1007/s00371-018-1588-5

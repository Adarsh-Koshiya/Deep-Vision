# `diamond_detr/` — third-party code (vendored), with our modifications listed

**Everything in this folder is third-party code.** It is included only so the transformer models
(RT-DETR-r18 and Diamond-DETR) run out-of-the-box; we did not write the library. Our own work is in the
parent `code/` folder (scripts + `experiments/`) and elsewhere in the submission (`report/`,
`gradcam_analysis/`, `results/`).

## Where it is from

- **Source repository:** `bigboyliang/DIAMOND` — https://github.com/bigboyliang/DIAMOND
  (the official code release accompanying the Diamond-DETR paper).
- **Paper:** Yan, X., Yang, S., Zhang, S., Li, X., Li, A. *"Diamond-DETR: lightweight real-time quality
  evaluation algorithm for synthetic diamonds."* **Scientific Reports 16, 10711 (2026).**
  DOI: 10.1038/s41598-026-44951-1.
- **What it contains:** `RTDETR-main/ultralytics/` is a **fork of Ultralytics 8.0.201**
  (https://github.com/ultralytics/ultralytics) extended by the paper's authors with the custom
  Diamond-DETR modules (the RepFasterNet block, the HS-FPN encoder, and the DRBC3 fusion block live in
  `RTDETR-main/ultralytics/nn/extra_modules/block.py`).
- **Usage / license:** the upstream repository states its contents are "used for study and research"
  only; Ultralytics is distributed under AGPL-3.0. This copy is included for academic reproducibility
  under those original terms. All credit for the architecture and library belongs to the original authors.

We stripped this vendored copy down to what is needed to run: the paper's bundled sample datasets, the
`adamas/` training framework, model weights, caches, and run logs were removed. Only the importable
`ultralytics` package remains.

## What WE changed inside this third-party tree

Two kinds of edits, kept inside the library because Ultralytics requires these files to live in its own
folders (they cannot be moved out without breaking imports):

### 1. Configuration files we authored (`RTDETR-main/ultralytics/cfg/models/rt-detr/`)
- `Diamond-detr.yaml` — corrected to the paper's true architecture (see patch #5 below).
- `ablation/rtdetr_r18_paper.yaml` — the baseline **A** (plain RT-DETR-r18).
- `ablation/diamond_AB.yaml`, `diamond_AC.yaml`, `diamond_AD.yaml`, `diamond_ABC.yaml` — the
  single-/double-component ablation configurations. Together with `Diamond-detr.yaml` (= A+B+C+D) and the
  two recovered combinations, these realise the paper's full 2³ Table-2 factorial.

### 2. Six compatibility patches (so the unmaintained fork runs on modern PyTorch 2.8 / NumPy 2.x)
These are **compatibility fixes only — no change to model behaviour**:

| # | File | Change |
|---|------|--------|
| 1 | `utils/metrics.py` | `np.trapz` → `np.trapezoid` (NumPy 2.x removed `trapz`) |
| 2 | `utils/torch_utils.py` | `torch.load(..., weights_only=False)` (PyTorch 2.x default change) |
| 3 | `nn/tasks.py` | same `weights_only=False` (2 call sites) |
| 4 | `nn/tasks.py` | `attempt_load_one_weight` returns `(model, ckpt)` instead of just `model` |
| 5 | `cfg/models/rt-detr/Diamond-detr.yaml` | replaced an incorrect backbone with the paper's real architecture; kept the top-level `nc: 80` key (its absence raises `KeyError: 'nc'`) |
| 6 | `nn/tasks.py` | fuse guard `if isinstance(m, ConvNormLayer) and hasattr(m, 'norm')` — prevents a double-fuse crash during evaluation |

## How it is used by our pipeline

The DETR models import this package via `PYTHONPATH`:
```bash
PYTHONPATH="code/diamond_detr/RTDETR-main:${PYTHONPATH:-}" python code/evaluate.py --type rtdetr ...
```
The YOLO models use the stock pip-installed Ultralytics 8.4.x instead; the two versions cannot share one
Python process, which is why our `train.sh` / `evaluate.py` run each family in a separate subprocess.

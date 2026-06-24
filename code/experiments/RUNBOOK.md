# Experiments runbook — the 3 rigor extensions

Everything here is **prepared and validated but NOT yet run** (needs free GPU time). Launch each
in a `screen` session on a free 8 GB GPU (check `nvidia-smi` first — shared lab machine).

The core 4-model study (REPORT.md §6/§7) is already complete and valid. These three only *raise*
rigor — none of them are required for a defensible thesis.

```
experiments/
├── scripts/
│   ├── validate_cfg.py     # build+forward check for any ablation YAML (already used)
│   ├── ablation.sh         # Task 1 — additive component ablation (paper Table 2 shape)
│   ├── multiseed.sh        # Task 3 — re-train all 4 models under one extra seed
│   └── aggregate_seeds.py  # Task 3 — mean±std across seeds
└── RUNBOOK.md              # this file
ablation YAMLs: diamond_detr/RTDETR-main/ultralytics/cfg/models/rt-detr/ablation/
```

---

## Task 1 — Ablation study (does each Diamond component help?)
**Configs authored & validated** (all build + forward-pass; params confirmed):

| Row | File | Adds | Params |
|-----|------|------|--------|
| A | `ablation/rtdetr_r18_paper.yaml` | none (paper RT-DETR-r18) | 19.9 M |
| A+B | `ablation/diamond_AB.yaml` | RepFasterNet backbone | 17.1 M |
| A+C | `ablation/diamond_AC.yaml` | HS-FPN encoder | 18.9 M |
| A+D | `ablation/diamond_AD.yaml` | DRBC3 fusion | 18.3 M |
| A+B+C | `ablation/diamond_ABC.yaml` | RepFaster + HS-FPN | 15.6 M |
| A+B+C+D | `Diamond-detr.yaml` | full Diamond-DETR | 14.3 M (✅ reused: mAP50 0.762) |

**Additive design — matches the paper's Table 2:** start from baseline A and add ONE module at a
time. Every config trains from random init with the *same* recipe (AdamW, lr 1e-4, 200 ep, batch 4,
seed 42), so each row is directly comparable to the paper's. A and A+B+C are reused from the first
run and A+B+C+D is the already-trained full model, so only **A+B, A+C, A+D** need GPU.
(An earlier *leave-one-out* version trained A+C+D and A+B+D — not paper rows — and was corrected.)

```bash
cd /home/40a2/DV/Code
screen -dmS abl bash -c 'bash experiments/scripts/ablation.sh 2>&1 | tee experiments/ablation.log'
```
**Cost ≈ 7–8 h** (3 from-scratch DETR runs: A+B, A+C, A+D; A / A+B+C / full all reused). Results →
`outputs/metrics_ablation/` (paper-named `A.json` … `A+B+C+D.json`), read automatically by notebook §5;
then fill REPORT.md Table 2.

To re-derive a YAML or add one, edit and re-validate:
```bash
cd diamond_detr/RTDETR-main
PYTHONPATH="$(pwd)" python /home/40a2/DV/Code/experiments/scripts/validate_cfg.py <yaml>
```

---

## Task 2 — From-scratch RT-DETR baseline (init-controlled comparison)
**This is row `A` in Task 1** — `rtdetr_r18_paper.yaml` trained from random init (19.9 M).
The current RT-DETR (0.763) was *pretrained* on diamonds; Diamond-DETR was *from scratch*. Training
RT-DETR from scratch at the same recipe removes that init confound (REPORT.md §9 Limitation 4):
the clean comparison becomes **Diamond (scratch, 14.3 M) vs RT-DETR (scratch, 19.9 M)**.

No separate script needed — running `ablation.sh` produces it as `A.json`. (The paper
RT-DETR architecture was reverse-engineered from its own checkpoint, so this is the *real* r18,
not the mislabeled 79.6 M `rtdetr-r18.yaml` shipped in the fork.)

---

## Task 3 — Multi-seed runs (variance / significance)
Seed 42 results already exist in `outputs/metrics/`. Add more seeds, then aggregate:
```bash
cd /home/40a2/DV/Code
screen -dmS s1 bash -c 'SEED=1 bash experiments/scripts/multiseed.sh 2>&1 | tee experiments/seed1.log'
# ...repeat SEED=2 (and 3) when the GPU is free...
python experiments/scripts/aggregate_seeds.py      # mean±std across all seed dirs
```
Each model keeps its exact recipe; only the seed + output dir change.
**Cost ≈ 6–8 h per seed** (2 CNNs + 2 transformers). For n=3 you need seeds 1 and 2 (42 is done).

---

## Recommended order (by value-per-GPU-hour)
1. **Ablation** (Task 1) — answers the thesis's component question; gives the from-scratch RT-DETR
   baseline (Task 2) for free. Highest value.
2. **Multi-seed** (Task 3) — only if you want mean±std; most expensive, least novel.

All three are optional. Do them when a free GPU window appears; the study stands without them.

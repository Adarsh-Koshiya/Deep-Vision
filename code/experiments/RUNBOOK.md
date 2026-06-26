# Experiments runbook

Two experiments extend the core four-model study (report Section 6). The main study is
self-contained; these add component-level and statistical depth.

```
experiments/scripts/
  validate_cfg.py     build and forward-pass check for one ablation config
  ablation.sh         trains the component ablation
  multiseed.sh        re-trains all four models under an extra seed
  aggregate_seeds.py  mean and standard deviation across seeds
ablation configs:     diamond_detr/RTDETR-main/ultralytics/cfg/models/rt-detr/ablation/
```

The committed metrics live under `results/`. Re-running a script writes fresh JSONs under its own
`outputs/` directory, so a re-run never overwrites the committed copies.

## 1. Component ablation (done)

The full 2^3 factorial of Diamond-DETR's three swaps, backbone B, neck C and fusion D: all eight
configurations, matching the source paper's Table 2. Every configuration trains from random init
under the same recipe (AdamW, lr 1e-4, 200 epochs, batch 4, seed 42), so the rows compare directly.

Test-split mAP@[.5:.95]:

```
A 0.514 | A+B 0.577 | A+C 0.421 | A+D 0.566 | A+B+C 0.411 | A+B+D 0.541 | A+C+D 0.587 | full 0.489
```

Committed results: `results/metrics_ablation/*.json`. Analysis: report Section 6.2.

`ablation.sh` trains A, A+B, A+C, A+D and A+B+C and reuses the already-trained full model as
A+B+C+D; A+B+D and A+C+D complete the eight.

```bash
bash experiments/scripts/ablation.sh
```

To check that a config builds before training it:

```bash
cd diamond_detr/RTDETR-main
PYTHONPATH="$(pwd)" python ../../experiments/scripts/validate_cfg.py <config.yaml>
```

## 2. Multi-seed (provided, not run)

Seed 42 is the single seed behind the reported numbers. The scripts to repeat every model under
more seeds and aggregate them are included for completeness; they were not run for this submission,
and the report records single-seed as a limitation. Budget about 6 to 8 hours per seed.

```bash
SEED=1 bash experiments/scripts/multiseed.sh
SEED=2 bash experiments/scripts/multiseed.sh
python experiments/scripts/aggregate_seeds.py
```

## Notes

- The script paths assume the original working directory (`/home/40a2/DV/Code`); adjust them to run
  elsewhere.
- Train on a free GPU with at least 8 GB (check `nvidia-smi` first).

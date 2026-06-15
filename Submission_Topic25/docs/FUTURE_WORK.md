# Future Work, Cross-Domain Evaluation of Diamond-DETR for Surface-Defect Detection

This document lists concrete follow-ups to the magnetic-tile study (Topic 25). Each item states
what to do, why it matters given the current findings, how to run it, and a rough cost. Items are
grouped into three tiers by priority. The framing is deliberately tied to the study's own
limitations (REPORT.md §9): every direction here closes a specific gap rather than adding scope for
its own sake.

The two results that shape these priorities: (a) Diamond-DETR matches RT-DETR on mAP50 and beats it
on mAP50-95 at fewer parameters, but the comparison is **not initialisation-controlled** (Diamond
from random init vs RT-DETR from a diamond-pretrained checkpoint); and (b) the transformer advantage
is dominated by **recall on rare classes** (fray, break) where the CNNs collapse. Most high-value
follow-ups attack one of these two points.

Note on rigour: results in this study are single-seed on a 138-image test split, so several items
below are about *quantifying uncertainty* rather than chasing higher headline numbers. State effect
sizes with confidence intervals before claiming any architectural win.

---

## Tier 1, closes the main confounds (do these first)

### 1. Initialisation-controlled Diamond-DETR vs RT-DETR
**Gap.** The central architectural claim rests on a comparison where the two models started from
different initialisations, and the asymmetry favours RT-DETR (Limitation 3/4). The parity is
therefore conservative but not clean.

**Plan.** Train RT-DETR-r18 **from random init** under the identical recipe (AdamW, 200 epochs,
batch 4, seed 42, same augmentation). Compare against the existing from-scratch Diamond-DETR result.
This is the single most important experiment: it converts the headline from "matches a pretrained
baseline despite a disadvantage" into a controlled A vs A+B+C+D point. Optionally, also pretrain
Diamond-DETR on the diamond dataset to produce the symmetric pretrained-vs-pretrained comparison.

**Cost.** One from-scratch DETR run, roughly 2 to 3 GPU-hours on the 8 GB RTX 3060 Ti at batch 4.

### 2. Component ablation of Diamond-DETR (B / C / D)
**Gap.** The study attributes the mAP50-95 gain to the RepFasterNet backbone (B), HS-FPN neck (C),
and DRBC3 fusion (D), but does not isolate them. The procedure is already specified in
`ABLATION_GRADCAM_PLAN.md`.

**Plan.** Train the six-config grid from scratch under one recipe: `A` (RT-DETR-r18 baseline),
`A+B`, `A+C`, `A+D`, `A+B+C`, `A+B+C+D` (the existing Diamond-DETR is a valid `A+B+C+D` point and is
reused). Report per-config ΔmAP50 / ΔmAP50-95 and per-class AP. The HS-FPN swap is not a one-line
module change (it alters neck connectivity), so validate each YAML with a forward pass before the
full run. Keep the top-level `nc: 80` key in every YAML (fork quirk, patch #5).

**Cost.** Five from-scratch DETR runs, roughly 12 to 15 GPU-hours total (`A+B+C+D` reused).

### 3. Multi-seed runs and statistical significance
**Gap.** All results are single-seed (42) on 138 test images. With a test set that small, mAP
confidence intervals are wide, and a 0.762 vs 0.763 difference is almost certainly within noise.

**Plan.** Repeat each model for at least three to five seeds and report mean ± standard deviation.
For pairwise claims, use a bootstrap confidence interval over per-image average precision, or a
paired permutation test on per-image scores, rather than comparing point estimates. This directly
qualifies which differences in REPORT.md §6 are real and which are not.

**Cost.** Linear in seeds: roughly 3 to 5 times the full four-model training budget.

---

## Tier 2, strengthens the science and the cross-domain claim

### 4. A true multi-domain cross-evaluation
**Gap.** "Cross-domain" currently means one source domain (diamonds, via RT-DETR's init) and one
target (magnetic tiles). A genuine cross-domain study needs several industrial targets.

**Plan.** Re-run the four-model protocol on additional public surface-defect datasets and compare
the transformer-vs-CNN gap and the Diamond-DETR efficiency win across domains:
NEU-DET (six hot-rolled steel defects; Song and Yan, 2013), GC10-DET (ten metal-surface defects;
Lv et al., 2020), DAGM 2007 (synthetic textured surfaces), KolektorSDD (Tabernik et al., 2020), and
MVTec AD (Bergmann et al., 2019) for an anomaly-detection contrast. Optionally add an unsupervised
domain-adaptation baseline for DETR to test whether attention transfers better than CNN features
across domains.

**Cost.** Roughly one full four-model cycle per dataset.

### 5. Matched training protocol (decouple optimizer and epochs)
**Gap.** Families differ in optimizer (AdamW vs SGD) and epochs (200 vs 150), a deliberate but
uncontrolled choice (Limitation 5). The project's own key lesson is that transformer detectors
collapse under SGD, which is itself worth studying properly.

**Plan.** Run a small optimizer sensitivity study per family (AdamW, SGD, and AdamW with cosine
schedule plus warmup) at a fixed epoch budget with early stopping on validation mAP. This both
removes the confound and turns the anecdotal "SGD left transformers at ~0 mAP" observation into a
documented curve.

**Cost.** Several short runs; can use reduced epochs with early stopping.

### 6. Class-imbalance methods for the rare defects
**Gap.** fray (32 training images) and break are where the CNNs fail and where the transformers'
advantage is largest. The current pipeline applies no imbalance handling, so this is the most direct
lever on the headline result.

**Plan.** Compare, on the same split: focal loss (Lin et al., 2017), class-balanced re-weighting
(Cui et al., 2019), simple copy-paste augmentation of rare-class instances (Ghiasi et al., 2021),
oversampling of rare-class images, and tuned mosaic. As a separate axis, generate synthetic rare
defects with a diffusion or GAN model and measure the lift. Report per-class AP and the recall at a
fixed false-alarm rate, not just mAP.

**Cost.** Moderate; mostly re-training with modified loss or data sampling.

### 7. Segmentation framing and box-conversion fidelity
**Gap.** The mask-to-box step via connected components is lossy: touching defect regions merge into
one box, and thin elongated cracks fit an axis-aligned box poorly (Limitation 8).

**Plan.** Two sub-directions. First, train instance-segmentation heads on the original masks
(Mask R-CNN as a CNN baseline, Mask-DINO or an RT-DETR segmentation variant as the transformer
counterpart) and evaluate with mask IoU, which is the metric inspection actually cares about.
Second, for crack and fray, try oriented or rotated bounding boxes so the box hugs the defect
geometry. Quantify how much of the per-class gap is a boxing artefact versus a real detection gap.

**Cost.** Moderate; reuses the same images and masks.

---

## Tier 3, robustness, deployment, and interpretability

### 8. Resolution, tiling, and multi-scale analysis
Defects are thin and low-contrast at 640 px. Test higher input resolution (1024 or 1280) and
slicing-aided inference (SAHI; Akyon et al., 2022) for small defects, and analyse which FPN level
contributes most to rare-class recall. The Grad-CAM maps are coarse (20x20); recompute on a
shallower layer (P3/P4) for sharper, more faithful attribution.

### 9. Operating-point and cost-sensitive evaluation
Inspection is a miss-rate problem, so the recall story (0.82 vs 0.35) deserves a proper operating
point. Report per-class precision-recall curves, F2 (recall-weighted), and a miss-rate vs
false-alarm trade with an explicit cost ratio (a missed defect costs more than a re-check). Add
confidence calibration (expected calibration error; Guo et al., 2017) since a deployed threshold
depends on calibrated scores.

### 10. Efficiency and deployment Pareto front
Map a fuller accuracy-vs-latency frontier rather than four points: add RT-DETR-r34/r50, RT-DETRv2
(Lv et al., 2024), and YOLOv8s/m, then push the chosen model through INT8 post-training
quantisation or quantisation-aware training, TensorRT export, structured pruning, and
transformer-to-CNN knowledge distillation (Hinton et al., 2015) to see whether a cheap student can
recover the transformer's rare-class recall. Measure real latency on an edge device (for example
Jetson), not just desktop FPS.

### 11. Robustness and out-of-distribution behaviour
Magnetic-tile images vary in illumination and surface finish. Evaluate under synthetic corruptions
(brightness, blur, noise), with test-time augmentation, and add an out-of-distribution check for
unseen defect types. This tests whether the transformer's global-attention advantage also buys
robustness, or only clean-set accuracy.

### 12. Interpretability beyond Grad-CAM
DETR Grad-CAM is finicky because there is no single class logit. Add attention rollout (Abnar and
Zuidema, 2020) for the AIFI encoder, visualise decoder cross-attention per object query, and try
Eigen-CAM (Muhammad and Yeasin, 2020). Quantify faithfulness with deletion/insertion curves
(Petsiuk et al., 2018) so the "transformers attend more precisely" claim is measured, not asserted.

### 13. Domain-appropriate pretraining without leakage
An alternative to item 1's symmetric pretraining: self-supervised pretraining of the backbone on
unlabeled tile images (the 952 defect-free tiles plus any unlabeled extras) using masked
autoencoding (He et al., 2022) or DINO (Caron et al., 2021), then fine-tune. This gives Diamond-DETR
a fair feature prior from the *target* domain, addressing the init confound without importing
diamond-domain features.

### 14. Reproducibility hardening
Containerise the two-Ultralytics split (the fork 8.0.201 for DETR, stock 8.4.x for YOLO cannot share
a process) in a single Docker image with both environments, pin all versions, upstream or
permanently record the six fork patches (REPORT.md §8), and publish the prepared split with seeds so
the 938/268/138 partition is bit-reproducible.

---

## Suggested minimal next step
If only one experiment is run, do **Tier 1, item 1** (from-scratch RT-DETR baseline). It is cheap
(a single short run), it directly removes the only confound in the headline architectural claim, and
its outcome decides whether the rest of the ablation programme (item 2) is worth the GPU time.

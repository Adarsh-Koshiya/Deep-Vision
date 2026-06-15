#!/usr/bin/env python3
"""make_figs.py, generate the report's result figures from results/metrics/*.json.

Writes three publication-quality PNGs into this figures/ folder, used by report/REPORT.tex:
  fig_overall_map.png      aggregate mAP@0.5 and mAP@[0.5:0.95] bars
  fig_perclass_map.png     per-class AP@[0.5:0.95] grouped bars (the rare-defect gap)
  fig_speed_accuracy.png   accuracy vs FPS, bubble area proportional to params

Every number comes straight from the JSON files; nothing is hard-coded. Run from anywhere:
  python figures/make_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))      # <repo>/figures
ROOT = os.path.dirname(HERE)                           # <repo>
METRICS = os.path.join(ROOT, "results", "metrics")

ORDER = ["Diamond-DETR", "RT-DETR", "YOLOv8n", "YOLO11n"]
FAMILY = {"Diamond-DETR": "Transformer", "RT-DETR": "Transformer",
          "YOLOv8n": "CNN", "YOLO11n": "CNN"}
COL = {"Diamond-DETR": "#0072B2", "RT-DETR": "#56B4E9",      # Okabe-Ito (colourblind-safe)
       "YOLOv8n": "#E69F00", "YOLO11n": "#D55E00"}
CLASSES = ["blowhole", "break", "crack", "fray", "uneven"]

D = {m: json.load(open(os.path.join(METRICS, f"{m}.json"))) for m in ORDER}

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "figure.dpi": 200,
})

# Figure 1: overall mAP50 and mAP50-95
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(ORDER)); w = 0.38
b1 = ax.bar(x - w/2, [D[m]["mAP50"]*100 for m in ORDER], w, label="mAP@0.5",
            color="#0072B2", edgecolor="white")
b2 = ax.bar(x + w/2, [D[m]["mAP50_95"]*100 for m in ORDER], w, label="mAP@0.5:0.95",
            color="#E69F00", edgecolor="white")
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=11)
ax.set_ylabel("Test AP (%)"); ax.set_ylim(0, 100)
ax.set_title("Detection accuracy on the Magnetic Tile test split", fontweight="bold")
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_overall_map.png"), bbox_inches="tight")
plt.close(fig)

# Figure 2: per-class mAP50-95
fig, ax = plt.subplots(figsize=(9, 4.4))
xc = np.arange(len(CLASSES)); w = 0.2
for i, m in enumerate(ORDER):
    ax.bar(xc + (i-1.5)*w, [D[m]["per_class_mAP50_95"][c]*100 for c in CLASSES], w,
           label=m, color=COL[m], edgecolor="white")
ax.set_xticks(xc); ax.set_xticklabels([c.capitalize() for c in CLASSES])
ax.set_ylabel("Per-class AP@0.5:0.95 (%)"); ax.set_ylim(0, 80)
ax.set_title("Per-class accuracy: the rare-defect gap (break, crack, fray)", fontweight="bold")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_perclass_map.png"), bbox_inches="tight")
plt.close(fig)

# Figure 3: speed vs accuracy, bubble area = params
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for m in ORDER:
    ax.scatter(D[m]["fps"], D[m]["mAP50"]*100, s=D[m]["params_M"]*60, color=COL[m],
               alpha=0.85, edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(f"{m}\n({FAMILY[m]}, {D[m]['params_M']:.1f}M)",
                (D[m]["fps"], D[m]["mAP50"]*100), xytext=(8, -4),
                textcoords="offset points", fontsize=9)
ax.set_xlabel("Inference speed (FPS, higher is faster)")
ax.set_ylabel("Test mAP@0.5 (%)")
ax.set_title("Accuracy vs speed (bubble area proportional to parameters)", fontweight="bold")
ax.set_xlim(0, 240); ax.set_ylim(25, 85)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_speed_accuracy.png"), bbox_inches="tight")
plt.close(fig)

print("wrote fig_overall_map.png, fig_perclass_map.png, fig_speed_accuracy.png to", HERE)

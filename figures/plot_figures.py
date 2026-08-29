"""
plot_figures.py

Generate publication-quality VECTOR (PDF) figures for the final benchmark results.
Native Matplotlib (no Seaborn dependency).
- fig_pipeline.pdf   : four-stage generation-pipeline schematic.
- fig_perscenario.pdf: 6-axis radar of per-attack recall (GBM / TCN / Semantic).
- fig_metrics.pdf    : lollipop/dot plot of overall F1 across splits (Y 0.85--1.00).
- fig_ablation.pdf   : heatmap of per-attack TPR at fixed 1% FPR.
Run:  python3 figures/plot_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# per-attack recall (run-independent split, global train threshold)
ATTACKS = ["DoS", "Drop", "Fuzz", "Low-slow", "Tamper", "Context"]
RECALL = {
    "GBM":     [0.998, 1.000, 1.000, 1.000, 0.994, 0.980],
    "TCN":     [0.143, 0.156, 0.997, 0.875, 0.990, 1.000],
    "Semantic":[0.020, 0.018, 0.104, 0.054, 0.915, 0.003],
}
ORDER = ["GBM", "TCN", "Semantic"]
PALETTE = {"GBM": "#4C72B0", "TCN": "#C44E52", "Semantic": "#E377C2"}   # Semantic = high-contrast magenta
LS = {"GBM": "--", "TCN": "-", "Semantic": ":"}
MK = {"GBM": "o", "TCN": "d", "Semantic": "s"}

# fixed-1%-FPR per-attack TPR (signal isolation, 4 detectors x 6 attacks; run-independent split)
heat = np.array([
    [0.994, 0.995, 0.032, 1.000, 0.033, 0.030],   # Timing-only
    [0.014, 0.012, 0.958, 0.107, 0.155, 0.620],   # Byte-only
    [0.992, 0.993, 0.985, 0.982, 0.145, 0.583],   # Full
    [0.085, 0.100, 1.000, 0.518, 0.992, 1.000],   # Raw-byte TCN
])
HEAT_ROWS = ["Timing-only", "Byte-only\n(stats)", "Full\n(byte+timing)", "Raw-byte TCN\n(deep)"]

# ROC-AUC per split (the discriminating metric; F1 saturates under the 91% attack imbalance)
SPLITS = ["Random", "Temporal", "Run-independent"]
ROC = {"GBM": [0.913, 0.909, 0.910], "RF": [0.897, 0.897, 0.893],
       "TCN": [0.826, 0.675, 0.862], "GRU": [0.500, 0.500, 0.500]}
METRIC_COLOR = {"GBM": "#4C72B0", "RF": "#55A868", "TCN": "#C44E52", "GRU": "#8C8C8C"}


def _round_box(ax, x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, color="#3a3a3a", lw=1.4))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    _round_box(ax, 0.1, 2.4, 2.1, 2.0,
               "Stage 1\nTestbed Topology\n\n9 ECUs\nProviders & Consumers\nSOME/IP-SD\npublish/subscribe",
               "#eef3fb", "#2b6cb0")
    _round_box(ax, 2.7, 2.4, 2.3, 2.0,
               "Stage 2\nAttack & Multi-run\n\nDoS · Fuzz · Drop\nLow-slow · Tamper\nContext-tamper\n"
               "(each run ×N)", "#fff4e5", "#b7791f")
    _round_box(ax, 5.5, 2.4, 2.2, 2.0,
               "Stage 3\nFeature & Byte Extr.\n\nEthernet frame →\nSOME/IP Header (16 B)\n"
               "+ Payload → 3×float32\n(speed, accel, yaw)", "#e6f7ef", "#2f855a")
    _round_box(ax, 8.3, 2.4, 1.7, 2.0,
               "Stage 4\nLabeled Output\n\nPCAP\nJSON Manifest\nCSV records\n(label + scenario)\n"
               "normal=0 · attack=1", "#f3e8ff", "#6b46c1")
    for i in range(3):
        _arrow(ax, 2.2 + i * 2.8, 3.4, 2.7 + i * 2.8, 3.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pipeline.pdf")


def fig_perscenario():
    N = len(ATTACKS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(ATTACKS, fontsize=10, weight="bold")
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", fontsize=9)
    ax.set_ylim(0, 1.08)
    for name in ORDER:
        vals = RECALL[name] + RECALL[name][:1]
        ax.plot(angles, vals, linewidth=2.0, linestyle=LS[name], marker=MK[name],
                markersize=6, label=name, color=PALETTE[name])
        ax.fill(angles, vals, color=PALETTE[name], alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_perscenario.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_perscenario.pdf")


def fig_metrics():
    dets = list(ROC.keys())
    x = np.arange(len(SPLITS))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, d in enumerate(dets):
        vals = ROC[d]
        offs = x + (i - (len(dets) - 1) / 2) * 0.16
        ax.vlines(offs, 0.44, vals, color=METRIC_COLOR[d], lw=1.0, alpha=0.6, zorder=2)
        ax.scatter(offs, vals, s=70, color=METRIC_COLOR[d], label=d, zorder=3)
        for j, v in enumerate(vals):
            ax.text(offs[j], v + 0.012, f"{v:.3f}", ha="center", fontsize=7, zorder=4)
    ax.text(x[2], 0.465, "GRU = majority-class\n(no ranking)", ha="center", fontsize=8,
            color="#8C8C8C", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(SPLITS, fontsize=11)
    ax.set_ylim(0.44, 1.0); ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_metrics.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_metrics.pdf")


def fig_ablation():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(heat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ATTACKS))); ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_yticks(range(len(HEAT_ROWS))); ax.set_yticklabels(HEAT_ROWS, fontsize=10)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=10, weight="bold")
    ax.set_xlabel("Attack", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("True Positive Rate @ 1% FPR", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ablation.pdf")


if __name__ == "__main__":
    fig_pipeline()
    fig_perscenario()
    fig_metrics()
    fig_ablation()
    print("done ->", OUT)

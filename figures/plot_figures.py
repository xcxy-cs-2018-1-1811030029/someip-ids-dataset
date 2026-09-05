"""
plot_figures.py

Generate publication-quality VECTOR (PDF) figures for the final benchmark results.
Native Matplotlib (no Seaborn dependency).
- fig_pipeline.pdf    : four-stage generation-pipeline schematic.
- fig_dataset.pdf     : dataset statistics (2x2: per-attack counts, per-intensity, per-run, dT distribution).
- fig_metrics.pdf     : grouped bar of ROC-AUC across splits (GBM/RF/TCN/GRU).
- fig_perscenario.pdf : per-attack detection heatmap (GBM/TCN/Semantic x 6 attacks).
- fig_ablation.pdf    : signal-isolation heatmap (timing/byte/full/TCN x 6 attacks) -- core figure.
Run:  python3 figures/plot_figures.py
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(OUT, "..", "vs_ip_gen", "generate")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

ATTACKS = ["DoS", "Drop", "Fuzz", "Low-slow", "Tamper", "Context"]
# per-attack TPR @ global 1% FPR training threshold (unified protocol, run split)
# order: DoS, Drop, Fuzz, Low-slow, Tamper, Context
RECALL = {
    "GBM":     [0.992, 0.993, 0.985, 0.982, 0.145, 0.583],
    "TCN":     [0.095, 0.104, 1.000, 0.625, 0.986, 1.000],
    "Semantic":[0.020, 0.018, 0.104, 0.054, 0.915, 0.003],
    "ST-TCN":  [0.992, 1.000, 1.000, 1.000, 0.999, 1.000],
}
# fixed-1%-FPR per-attack TPR (signal isolation, run-independent split) -> fig_ablation
heat = np.array([
    [0.994, 0.995, 0.032, 1.000, 0.033, 0.030],   # Timing-only
    [0.014, 0.012, 0.958, 0.107, 0.155, 0.620],   # Byte-only
    [0.992, 0.993, 0.985, 0.982, 0.145, 0.583],   # Full
    [0.095, 0.104, 1.000, 0.625, 0.986, 1.000],   # Raw-byte TCN
    [0.992, 1.000, 1.000, 1.000, 0.999, 1.000],   # ST-TCN (bytes+timing)
])
HEAT_ROWS = ["Timing-only", "Byte-only", "Full", "Raw-byte TCN", "ST-TCN"]
# ROC-AUC per split -> fig_metrics (grouped bar)
SPLITS = ["Random", "Temporal", "Run-indep."]
ROC = {"GBM": [0.913, 0.909, 0.910], "RF": [0.897, 0.897, 0.893],
       "TCN": [0.826, 0.675, 0.862], "GRU": [0.500, 0.500, 0.500],
       "GRU-mean": [0.811, 0.721, 0.832]}
COL = {"GBM": "#4C72B0", "RF": "#55A868", "TCN": "#C44E52", "GRU": "#8C8C8C",
       "GRU-mean": "#8172B2"}
SC_COLOR = {"normal": "#8C8C8C", "dos": "#C44E52", "fuzz": "#DD8452",
            "drop": "#55A868", "slowslow": "#8172B2", "tamper": "#4C72B0",
            "ctx_tamper": "#937860"}


# ---------------- dataset statistics (read CSVs) ----------------
def load_ds():
    per_sc, per_int, run_counts, per_sc_dt = {}, {}, [], {}
    for p in sorted(glob.glob(os.path.join(GEN, "*_cli.csv"))):
        base = os.path.basename(p)[:-8]               # strip "_cli.csv"
        base2 = re.sub(r"_run\d+$", "", base)
        m = re.search(r"_i([\d.]+)$", base2)
        it = m.group(1) if m else "1.0"
        sc = re.sub(r"_i[\d.]+$", "", base2)
        mrun = re.search(r"_run(\d+)$", base)
        run = int(mrun.group(1)) if mrun else 1
        ts = []
        with open(p) as f:
            f.readline()
            for line in f:
                if not line.strip():
                    continue
                ts.append(float(line.rstrip("\n").split(",", 4)[0]))
        n = len(ts)
        per_sc[sc] = per_sc.get(sc, 0) + n
        if sc != "normal":
            per_int[it] = per_int.get(it, 0) + n
        run_counts.append((sc, run, n))
        ts = np.sort(np.asarray(ts)); d = np.diff(ts)
        per_sc_dt.setdefault(sc, []).extend(d.tolist())
    return per_sc, per_int, run_counts, per_sc_dt


def _round_box(ax, x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, color="#3a3a3a", lw=1.4))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    _round_box(ax, 0.1, 2.5, 2.1, 2.0,
               "Stage 1\nvSomeIP testbed\n\nService + subscribers\nSOME/IP-SD\npublish/subscribe",
               "#eef3fb", "#2b6cb0")
    _round_box(ax, 2.7, 2.5, 2.6, 2.0,
               "Stage 2\nAttacks & runs\n\nNormal + 6 attacks\nx 3 intensities\nx 3 independent runs",
               "#fff4e5", "#b7791f")
    _round_box(ax, 5.9, 2.5, 2.2, 2.0,
               "Stage 3\nCapture & parse\n\nSOME/IP header\n+ payload -> 3xfloat32\n(speed, accel, yaw)",
               "#e6f7ef", "#2f855a")
    _round_box(ax, 8.6, 2.5, 1.4, 2.0,
               "Stage 4\nLabeled out\n\nCSV + manifest\nlabel / scenario",
               "#f3e8ff", "#6b46c1")
    for i in range(3):
        _arrow(ax, 2.2 + i * 3.2, 3.5, 2.7 + i * 3.2, 3.5)
    ax.text(5.0, 1.0, "190k messages / 57 runs / 7 scenarios / 3 intensity levels",
            ha="center", fontsize=10, style="italic", color="#3a3a3a")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig); print("wrote fig_pipeline.pdf")


def fig_dataset():
    import matplotlib.gridspec as gridspec
    per_sc, per_int, run_counts, per_sc_dt = load_ds()

    ORDER = ["normal", "dos", "fuzz", "drop", "slowslow", "tamper", "ctx_tamper"]
    LBL = {"normal": "Normal", "dos": "DoS", "fuzz": "Fuzz", "drop": "Drop",
           "slowslow": "Low-slow", "tamper": "Tamper", "ctx_tamper": "Ctx. tamper"}

    C_PALETTE = {
        "normal": "#94A3B8",     # 冷灰
        "dos": "#E11D48",        # 胭脂红
        "fuzz": "#F59E0B",       # 琥珀黄
        "drop": "#059669",       # 翡翠绿
        "slowslow": "#7C3AED",   # 亮紫
        "tamper": "#2563EB",     # 皇家蓝
        "ctx_tamper": "#0F766E"  # 深青
    }

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], wspace=0.2, hspace=0.35)
    total_all = sum(per_sc.values())

    # ---------------- (a) Bar + Swarm ----------------
    ax_scen = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(ORDER))
    for i, s in enumerate(ORDER):
        runs = [n for (sc, r, n) in run_counts if sc == s]
        if not runs:
            continue
        color = C_PALETTE[s]
        ax_scen.bar(i, per_sc[s], color=color, alpha=0.15, edgecolor=color, lw=1.5,
                    width=0.6, zorder=2)
        jit = np.random.RandomState(0).uniform(-0.13, 0.13, len(runs))
        ax_scen.scatter(np.asarray([i] * len(runs)) + jit, runs, color=color, s=42,
                        edgecolor="white", lw=0.8, zorder=3)
        ax_scen.text(i, per_sc[s] * 1.1, f"{per_sc[s]/1000:.1f}k", ha="center", va="bottom",
                     fontsize=9, weight="bold", color="#334155")
    ax_scen.set_yscale("log")
    ax_scen.set_xticks(xs)
    ax_scen.tick_params(axis="x", length=0, pad=14)
    ax_scen.set_xticklabels([LBL[s] for s in ORDER], fontsize=9, rotation=18, ha="right", weight="medium")
    ax_scen.set_ylabel("Messages (log scale)", fontsize=11, weight="medium")
    ax_scen.set_title("(a) Scenario volume & run-to-run variance", loc="left", fontsize=12,
                      weight="bold", pad=12, color="#1E293B")
    ax_scen.grid(axis="y", ls="--", alpha=0.3, zorder=1)
    ax_scen.set_ylim(bottom=10, top=per_sc["dos"] * 2.0)

    # ---------------- (b) Nested Donut ----------------
    ax_donut = fig.add_subplot(gs[0, 1])
    total_normal = per_sc.get("normal", 0)
    total_attack = sum(v for k, v in per_sc.items() if k != "normal")
    ints = sorted(per_int); int_vals = [per_int[i] for i in ints]
    outer_colors = [C_PALETTE["normal"], "#E11D48"]
    inner_colors = [C_PALETTE["normal"], "#FDA4AF", "#F43F5E", "#9F1239"]
    ax_donut.pie([total_normal, total_attack], radius=1.0, colors=outer_colors,
                 startangle=90, counterclock=False, wedgeprops=dict(width=0.3, edgecolor="white", lw=2))
    ax_donut.pie([total_normal] + int_vals, radius=0.7, colors=inner_colors,
                 startangle=90, counterclock=False, wedgeprops=dict(width=0.2, edgecolor="white", lw=2))
    ax_donut.set_xlim(-1.7, 1.7); ax_donut.set_ylim(-1.7, 1.7)
    ax_donut.text(0, 0, f"{total_all/1000:.0f}k\nTotal", ha="center", va="center",
                  fontsize=11, weight="bold", color="#1E293B")
    ax_donut.annotate("Normal\n(8.5%)", xy=(0.3, 0.85), xytext=(0.95, 1.35),
                      arrowprops=dict(arrowstyle="-", color="#64748B"), fontsize=10, weight="bold", color="#475569")
    ax_donut.annotate("Attack Imbalance\n(91.5%)", xy=(-0.7, -0.7), xytext=(-1.72, -1.42),
                      arrowprops=dict(arrowstyle="-", color="#64748B", lw=1.0),
                      fontsize=10, weight="bold", color="#BE123C")
    import matplotlib.patches as mpatches
    ax_donut.legend(handles=[mpatches.Patch(facecolor=inner_colors[1], label="Intensity 0.3"),
                             mpatches.Patch(facecolor=inner_colors[2], label="Intensity 0.7"),
                             mpatches.Patch(facecolor=inner_colors[3], label="Intensity 1.0")],
                    loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    ax_donut.set_title("(b) Extreme class imbalance & attack intensity", loc="left", fontsize=12,
                       weight="bold", pad=12, color="#1E293B")

    # ---------------- (c) Wide density plot ----------------
    ax_dt = fig.add_subplot(gs[1, :])
    dist_styles = [("normal", C_PALETTE["normal"], "Normal"), ("dos", C_PALETTE["dos"], "DoS"),
                   ("drop", C_PALETTE["drop"], "Message-drop"), ("slowslow", C_PALETTE["slowslow"], "Low-and-slow")]
    for sc, c, lbl in dist_styles:
        d = np.asarray(per_sc_dt.get(sc, [])); d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            ax_dt.hist(np.log10(d), bins=80, density=True, histtype="step", color=c, lw=2.0, zorder=4)
            ax_dt.hist(np.log10(d), bins=80, density=True, histtype="stepfilled", color=c, alpha=0.15, zorder=3)
    ax_dt.axvline(np.log10(0.01), color="#334155", ls=":", lw=2, zorder=2)
    ax_dt.text(np.log10(0.02), ax_dt.get_ylim()[1] * 0.93, "10 ms Period",
               fontsize=10, weight="bold", color="#334155", ha="left",
               bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax_dt.annotate("DoS burst\n(high freq.)", xy=(-2.9, 4), xytext=(-3.7, 11),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=C_PALETTE["dos"]),
                   fontsize=9.5, weight="bold", color=C_PALETTE["dos"])
    ax_dt.annotate("Low-and-slow\n(sparse timing)", xy=(0.4, 18), xytext=(1.15, 9.5),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color=C_PALETTE["slowslow"]),
                   fontsize=9.5, weight="bold", color=C_PALETTE["slowslow"])
    ax_dt.set_xlabel(r"Inter-arrival $\Delta t$ (seconds, $\log_{10}$ scale)", fontsize=11, weight="medium")
    ax_dt.set_ylabel("Density", fontsize=11, weight="medium")
    ax_dt.set_title("(c) Distinct temporal manifolds of timing-based attacks", loc="left", fontsize=12,
                    weight="bold", pad=12, color="#1E293B")
    ax_dt.grid(axis="y", ls="--", color="#E2E8F0", zorder=1)

    fig.savefig(os.path.join(OUT, "fig_dataset.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote premium fig_dataset.pdf")


def fig_metrics():
    import matplotlib.patches as mpatches
    dets = ["GBM", "RF", "TCN", "GRU", "GRU-mean"]
    ROCM = np.array([[0.913, 0.909, 0.910],
                     [0.897, 0.897, 0.893],
                     [0.826, 0.675, 0.862],
                     [0.500, 0.500, 0.500],
                     [0.811, 0.721, 0.832]])
    MCCM = np.array([[0.324, 0.166, 0.255],
                     [0.246, 0.109, 0.175],
                     [0.000, 0.000, 0.000],
                     [0.000, 0.000, 0.000],
                     [0.000, 0.029, 0.000]])
    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    im = ax.imshow(ROCM, cmap="Blues", vmin=0.45, vmax=0.95, aspect="auto")
    for i in range(ROCM.shape[0]):
        for j in range(ROCM.shape[1]):
            ax.text(j, i, f"{ROCM[i, j]:.3f}\n{MCCM[i, j]:.3f}", ha="center", va="center",
                    fontsize=8.5, color="white" if ROCM[i, j] > 0.72 else "#333", linespacing=1.25)
    # dashed box highlighting the collapsed GRU row
    ax.add_patch(mpatches.Rectangle((-0.5, 2.5), 3.0, 1.0, fill=False, ec="#8a8a8a",
                                    ls="--", lw=1.4))
    ax.text(2.62, 3.0, "random-guessing\nbaseline (0.50)", color="#8a8a8a", fontsize=8,
            ha="left", va="center", style="italic")
    # solid box highlighting the corrected GRU-mean row
    ax.add_patch(mpatches.Rectangle((-0.5, 3.5), 3.0, 1.0, fill=False, ec="#8172B2",
                                    ls="-.", lw=1.4))
    ax.text(2.62, 4.0, "mean-pooled\nreadout (fixed)", color="#8172B2", fontsize=8,
            ha="left", va="center", style="italic")
    ax.set_xticks(range(3)); ax.set_xticklabels(SPLITS, fontsize=11)
    ax.set_yticks(range(5)); ax.set_yticklabels(dets, fontsize=11)
    ax.set_xlim(-0.6, 3.5); ax.set_ylim(-0.55, 4.5)
    ax.set_xlabel("Split", fontsize=11)
    ax.set_title("Overall ROC-AUC / MCC across splits (cell: ROC-AUC, top; MCC, bottom)",
                 loc="left", fontsize=9.5, pad=8, color="#2d3748")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ROC-AUC", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_metrics.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_metrics.pdf")


def fig_perscenario():
    N = len(ATTACKS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    pal = {"GBM": "#2b6cb0", "TCN": "#c05621", "Semantic": "#805ad5", "ST-TCN": "#2f855a"}
    mk = {"GBM": "o", "TCN": "D", "Semantic": "s", "ST-TCN": "^"}
    fig, ax = plt.subplots(figsize=(6.6, 6.2), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(ATTACKS, fontsize=10, weight="bold")
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", fontsize=9)
    ax.set_ylim(0, 1.06)
    for name in ["GBM", "TCN", "Semantic", "ST-TCN"]:
        vals = RECALL[name] + RECALL[name][:1]
        ax.plot(angles, vals, lw=2.2, color=pal[name], marker=mk[name], markersize=5,
                label={"GBM": "GBM (behav.)", "TCN": "TCN (bytes)", "Semantic": "Semantic",
                       "ST-TCN": "ST-TCN (bytes+timing)"}[name])
        ax.fill(angles, vals, color=pal[name], alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=True, fontsize=9)
    ax.set_title("Per-attack TPR @ global 1% FPR threshold (unified protocol)",
                 fontsize=11, pad=26, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_perscenario.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_perscenario.pdf")


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
    plt.close(fig); print("wrote fig_ablation.pdf")


def fig_contextual():
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    s1n, s2n, s1c, s2c = [], [], [], []
    for p in sorted(glob.glob(os.path.join(GEN, "*_cli.csv"))):
        base = os.path.basename(p)[:-8]
        base = re.sub(r"_run\d+$", "", base)     # strip run first
        sc = re.sub(r"_i[\d.]+$", "", base)      # strip intensity -> scenario
        if sc not in ("normal", "ctx_tamper"):
            continue
        with open(p) as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split(",", 4)
                b = bytes(int(x, 16) for x in parts[-1].split()) if parts[-1].split() else b""
                if len(b) >= 12 and len(b) % 4 == 0:
                    v = np.frombuffer(b, dtype=np.float32)[:3]
                    if sc == "normal":
                        s1n.append(float(v[0])); s2n.append(float(v[1]))
                    else:
                        s1c.append(float(v[0])); s2c.append(float(v[1]))
    s1n, s2n, s1c, s2c = map(np.array, (s1n, s2n, s1c, s2c))
    rs = np.random.RandomState(0)
    def sub2(a, b, k=4000):
        if len(a) > k:
            idx = rs.choice(len(a), k, replace=False)
            return a[idx], b[idx]
        return a, b
    s1n, s2n = sub2(s1n, s2n)
    s1c, s2c = sub2(s1c, s2c)

    C_NORM = "#2563EB"   # 科技宝蓝 (Normal)
    C_CTX = "#E11D48"    # 警示胭脂红 (Contextual Tampering)

    fig = plt.figure(figsize=(8.0, 7.0))
    gs = gridspec.GridSpec(4, 4, wspace=0.08, hspace=0.08)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # 主散点：底层攻击噪点云，顶层正常流形
    ax_main.scatter(s1c, s2c, s=8, c=C_CTX, alpha=0.15, linewidths=0, zorder=2)
    ax_main.scatter(s1n, s2n, s=8, c=C_NORM, alpha=0.5, linewidths=0, zorder=3)

    # 顶部 X 边缘直方图 (Speed)
    ax_top.hist(s1c, bins=45, density=True, color=C_CTX, alpha=0.2, histtype="stepfilled")
    ax_top.hist(s1c, bins=45, density=True, color=C_CTX, lw=1.5, histtype="step")
    ax_top.hist(s1n, bins=45, density=True, color=C_NORM, alpha=0.4, histtype="stepfilled")
    ax_top.hist(s1n, bins=45, density=True, color=C_NORM, lw=2.0, histtype="step")
    ax_top.axis("off")

    # 右侧 Y 边缘直方图 (Acceleration)
    ax_right.hist(s2c, bins=45, density=True, color=C_CTX, alpha=0.2, histtype="stepfilled",
                  orientation="horizontal")
    ax_right.hist(s2c, bins=45, density=True, color=C_CTX, lw=1.5, histtype="step", orientation="horizontal")
    ax_right.hist(s2n, bins=45, density=True, color=C_NORM, alpha=0.4, histtype="stepfilled",
                  orientation="horizontal")
    ax_right.hist(s2n, bins=45, density=True, color=C_NORM, lw=2.0, histtype="step", orientation="horizontal")
    ax_right.axis("off")

    ax_main.set_xlabel("Signal 1: Speed", fontsize=12, weight="medium")
    ax_main.set_ylabel("Signal 2: Acceleration", fontsize=12, weight="medium")
    ax_main.set_xlim(40, 160); ax_main.set_ylim(10, 90)
    ax_main.grid(ls="--", color="#CBD5E1", alpha=0.6, zorder=1)

    # 语义合法边界框
    min_x, max_x = float(np.min(s1n)), float(np.max(s1n))
    min_y, max_y = float(np.min(s2n)), float(np.max(s2n))
    ax_main.add_patch(patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                        linewidth=2.0, edgecolor="#475569", facecolor="none",
                                        linestyle="--", zorder=4))

    ax_main.annotate("Normal Joint Manifold\n(benign traffic)", xy=(95, 78), xytext=(52, 84),
                     arrowprops=dict(arrowstyle="-|>", color=C_NORM, lw=1.5,
                                     connectionstyle="arc3,rad=0.2"), fontsize=10, weight="bold",
                     color=C_NORM, zorder=5)
    ax_main.annotate("Joint Inconsistency\n(Detectable by TCN)", xy=(130, 35), xytext=(133, 12),
                     arrowprops=dict(arrowstyle="-|>", color=C_CTX, lw=1.5,
                                     connectionstyle="arc3,rad=-0.2"), fontsize=10, weight="bold",
                     color=C_CTX, zorder=5)
    ax_main.annotate("Per-signal Valid Range\n(Semantic checks pass)", xy=(140, 80), xytext=(100, 88),
                     arrowprops=dict(arrowstyle="-", color="#475569", lw=1.5), fontsize=9.5,
                     weight="bold", color="#475569", ha="left", va="top", zorder=5)
    ax_top.text(0.5, 0.55, "Overlapping 1D ranges blind per-signal checks",
                transform=ax_top.transAxes, ha="center", fontsize=9.5, style="italic", color="#475569")

    fig.text(0.08, 0.95, "Contextual Tampering vs. Normal Joint Manifold", fontsize=14, weight="bold",
             color="#0F172A", ha="left")
    fig.text(0.08, 0.905, "Joint distribution exposes what 1D semantic-validity thresholds completely miss.",
             fontsize=11, color="#475569", ha="left")

    fig.savefig(os.path.join(OUT, "fig_contextual.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote premium fig_contextual.pdf")


if __name__ == "__main__":
    # fig_pipeline() is superseded by the released overview figure (fig_pipeline.pdf) and is not
    # regenerated here to avoid clobbering it.
    fig_dataset()
    fig_metrics()
    fig_perscenario()
    fig_ablation()
    fig_contextual()
    print("done ->", OUT)

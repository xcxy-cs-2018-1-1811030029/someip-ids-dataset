"""
plot_new_experiments.py -- figures for the reviewer-response experiments.

Inputs (from vs_ip_gen/results/):
  jitter_rtt_gateway.txt / jitter_rtt_external.txt   measured ICMP RTT (ms)
  jitter_robustness.txt                              detector metrics per injection level
  balanced_ablation.txt                              GRU/TCN ROC vs training prevalence
  latency.txt                                        fair CPU batch-1 latencies
  intensity_invariance.txt                           per-scenario intensity AUC table
  st_tcn_eval.txt                                    fixed-FPR TPR rows + torch/ONNX latency

Outputs (figures/):
  fig_jitter_dist.pdf        measured + injected jitter distributions
  fig_jitter_robustness.pdf  detector ROC vs injection level + timing per-scenario recall
  fig_balanced_ablation.pdf  ROC vs training prevalence (GRU vs TCN)
  fig_latency.pdf            fair per-packet latency bars
  fig_intensity.pdf          AUC heatmap across intensities (with design annotation)
  fig_st_tcn.pdf             ST-TCN fixed-FPR TPR vs TCN/timing-only + torch vs ONNX latency
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "vs_ip_gen", "results")
FIG = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150,
})

PC = {
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
}
BLUE, RED, GREEN, AMBER, GREY = "#4575B4", "#D6604D", "#6BA25B", "#E08214", "#8C8C8C"


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color="#E8E8E8", lw=0.6, zorder=0)


def load_jitter_files():
    gw = np.loadtxt(os.path.join(RES, "jitter_rtt_gateway.txt"))
    ext = np.loadtxt(os.path.join(RES, "jitter_rtt_external.txt"))
    return gw, ext


def parse_robustness(path):
    """Return (levels, per-level dict) from jitter_robustness.txt."""
    levels, rows = [], {}
    scen_names, scen_mode = None, None
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^(\S+)\s+([\d.]+)u\s+([\d.]+)u\s+\|\s+"
                     r"([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+\|\s+"
                     r"([\d.]+)\s+(-?[\d.]+)\s+\|\s+"
                     r"([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)$", line)
        if m:
            lvl = m.group(1)
            levels.append(lvl)
            rows[lvl] = dict(
                timing_roc=float(m.group(4)), timing_pr=float(m.group(5)),
                timing_mcc=float(m.group(6)),
                byte_roc=float(m.group(7)), byte_mcc=float(m.group(8)),
                full_roc=float(m.group(9)), full_pr=float(m.group(10)),
                full_mcc=float(m.group(11)))
            continue
        if "per-scenario recall" in line:
            scen_mode = "timing" if "timing-only" in line else "full"
            continue
        if scen_names is None and scen_mode:
            m2 = re.match(r"^level\s+(.+)$", line)
            if m2:
                scen_names = m2.group(1).split()
                continue
        if scen_names and scen_mode:
            toks = line.split()
            if toks and toks[0] in rows and len(toks) == len(scen_names) + 1:
                vals = [float(x) for x in toks[1:]]
                rows[toks[0]][f"{scen_mode}_rec"] = dict(zip(scen_names, vals))
    return levels, rows


def parse_ablation(path):
    recs = []
    split = None
    for line in open(path):
        m = re.match(r"===== split=(\S+)", line)
        if m:
            split = m.group(1)
            continue
        parts = line.split()
        # random-split rows: "GRU  0.915  0.500  0.915  0.000  1.000  0.915  1.000  5m"
        if (len(parts) == 9 and parts[0] in ("GRU", "TCN") and split
                and parts[1].replace(".", "").isdigit()):
            recs.append(dict(split=split, model=parts[0],
                             prev=float(parts[1]), roc=float(parts[2]),
                             pr=float(parts[3]), mcc=float(parts[4]),
                             posrate=float(parts[5])))
        # run-split rows: "run GRU orig 0.915 0.500 0.915 0.000 1.000 0.915 1.000"
        elif (len(parts) == 10 and parts[0] == "run"
                and parts[1] in ("GRU", "TCN")
                and parts[3].replace(".", "").isdigit()):
            recs.append(dict(split="run", model=parts[1],
                             prev=float(parts[3]), roc=float(parts[4]),
                             pr=float(parts[5]), mcc=float(parts[6]),
                             posrate=float(parts[7])))
    return recs


def fig_jitter_dist(gw, ext):
    with plt.rc_context(PC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        for ax, (name, rtt) in zip(axes, [("(a) campus path (gateway)", gw),
                                          ("(b) campus path (external)", ext)]):
            ow = rtt / 2.0
            xlim = (0, float(np.percentile(ow, 99.9)) * 1.05)
            bins = np.linspace(xlim[0], xlim[1], 70)
            ax.hist(ow, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.75, color=BLUE, edgecolor="white", lw=0.2,
                    label="one-way delay (RTT/2)", zorder=3)
            ax.hist(np.abs(np.diff(rtt)), bins=bins, density=True,
                    histtype="stepfilled", alpha=0.6, color=AMBER,
                    edgecolor="white", lw=0.2, label="|RTT jitter|", zorder=3)
            style_axes(ax)
            ax.set_title(name, loc="left", fontsize=10.5)
            ax.set_xlabel("delay (ms)")
            ax.set_ylabel("density")
            ax.legend(loc="upper right", frameon=False)
            ax.text(0.02, 0.97, f"median {np.median(ow):.2f} ms\n"
                    f"p95 {np.percentile(ow, 95):.2f} ms",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white",
                              ec="#CCCCCC", lw=0.6))
        fig.tight_layout()
        return fig


def fig_robustness(levels, rows):
    x = np.arange(len(levels))
    with plt.rc_context(PC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        ax = axes[0]
        for key, lab, col, mk, lw_ in [
                ("timing_roc", "timing-only GBM", RED, "o", 1.6),
                ("byte_roc", "byte-only GBM (jitter-immune)", GREEN, "^", 1.6),
                ("full_roc", "full GBM", BLUE, "s", 2.0)]:
            ax.plot(x, [rows[l][key] for l in levels], marker=mk, ms=5,
                    lw=lw_, color=col, markeredgecolor="white",
                    markeredgewidth=0.7, label=lab, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.40, 1.02)
        style_axes(ax)
        ax.legend(loc="lower left", frameon=False)
        ax.set_title("(a) detector ROC vs injected jitter", loc="left",
                     fontsize=10.5)

        ax = axes[1]
        scen = sorted(rows[levels[0]].get("timing_rec", {}).keys())
        scen = [s for s in scen if s != "normal"]
        if scen:
            mat = np.array([[rows[l]["timing_rec"][s] for s in scen]
                            for l in levels])
            im = ax.imshow(mat.T, aspect="auto", cmap="YlGnBu",
                           vmin=0.99, vmax=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=20, ha="right", fontsize=8.5)
            ax.set_yticks(range(len(scen)))
            ax.set_yticklabels(scen, fontsize=8.5)
            for i in range(mat.shape[1]):
                for j in range(mat.shape[0]):
                    ax.text(j, i, f"{mat[j, i]:.3f}", ha="center", va="center",
                            fontsize=6.8, color="white",
                            path_effects=[pe.withStroke(linewidth=1.1,
                                                        foreground="#1B1B1B")],
                            zorder=3)
            ax.set_title("(b) timing-only per-scenario recall", loc="left",
                         fontsize=10.5)
            cbar = fig.colorbar(im, ax=ax, fraction=0.045)
            cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        return fig


def fig_ablation(recs):
    GREEN_TXT, GREEN_EDGE, GREEN_FILL = "#4F7B3F", "#6BA25B", "#D9EFD4"
    with plt.rc_context({**PC, "axes.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for split, mk, lstyle in (("random", "o", "-"), ("run", "s", "--")):
            sub = [r for r in recs if r["model"] == "TCN" and r["split"] == split]
            sub.sort(key=lambda r: r["prev"])
            if not sub:
                continue
            ax.plot([r["prev"] for r in sub], [r["roc"] for r in sub],
                    lstyle, marker=mk, ms=5, lw=1.8, color=BLUE,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=f"TCN ({split} split)", zorder=3)
        gru = [r for r in recs if r["model"] == "GRU"]
        gru.sort(key=lambda r: r["prev"])
        ax.plot([r["prev"] for r in gru], [r["roc"] for r in gru],
                "-", marker="o", ms=5, lw=1.8, color=RED,
                markeredgecolor="white", markeredgewidth=0.8,
                label="GRU readout (random & run)", zorder=3)
        ax.set_xticks([0.10, 0.30, 0.50, 0.915])
        ax.set_xticklabels(["10%", "30%", "50%", "91.5%\n(released corpus)"])
        ax.set_xlim(0.02, 1.0)
        ax.set_ylim(0.42, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel("attack prevalence in training set")
        ax.set_ylabel("ROC-AUC (fixed test split)")
        style_axes(ax)
        ax.axhline(0.5, color="#9AA0A6", ls=":", lw=1.1, zorder=1)
        ax.text(0.985, 0.512, "chance", ha="right", va="bottom",
                fontsize=8, color=GREY)
        ax.axvline(0.915, color="#9AA0A6", ls=":", lw=1.0, zorder=1)
        ax.annotate("GRU last-state readout:\npinned at chance at every prevalence",
                    xy=(0.335, 0.503), xytext=(0.05, 0.60),
                    fontsize=8, color="#B03A2E", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#B03A2E", lw=0.7))
        x0 = (0.925 - 0.02) / 0.98
        x1 = (0.985 - 0.02) / 0.98
        ax.axhspan(0.721, 0.832, xmin=x0, xmax=x1, facecolor=GREEN_FILL,
                   edgecolor=GREEN_EDGE, lw=0.8, ls="--", zorder=1.5)
        ax.annotate("GRU-mean readout 0.72-0.83 (Table 4)", xy=(0.928, 0.752),
                    xytext=(0.895, 0.750), ha="right", va="center",
                    fontsize=8, color=GREEN_TXT,
                    arrowprops=dict(arrowstyle="-", color=GREEN_EDGE, lw=0.7))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
                  frameon=False, fontsize=8.5, columnspacing=1.0,
                  handletextpad=0.5, handlelength=1.6)
        ax.set_title("Balanced-prevalence ablation (fixed test split)",
                     loc="left", fontsize=10.5)
        fig.tight_layout()
        return fig


def fig_latency():
    txt = open(os.path.join(RES, "latency.txt")).read()
    lat = {m[0].strip(): float(m[1]) for m in
           re.findall(r"^(\S[^:]{0,40}?):\s+med=\s*([\d.]+)\s+us", txt, re.M)}
    st_path = os.path.join(RES, "st_tcn_eval.txt")
    eng = {}
    if os.path.exists(st_path):
        for line in open(st_path):
            m = re.match(r"LATENCY_(TORCH|ONNX)\s+(\S+)\s+med=\s*([\d.]+)\s+us", line)
            if m:
                eng.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
    rows = [("semantic decoding", lat["semantic decode+sc"]),
            ("behavioral features", lat["behavioral features"]),
            ("GBM pipeline (features + 300 trees)", lat["GBM (feat+300 trees)"]),
            ("TCN (PyTorch)", lat["TCN batch-1 CPU"]),
            ("GRU (PyTorch)", lat["GRU batch-1 CPU"])]
    for n in ["TCN", "ST-TCN", "GRU-mean"]:
        if n in eng and "ONNX" in eng[n]:
            rows.append((f"{n} (ONNX Runtime)", eng[n]["ONNX"]))
    if "ST-TCN" in eng and "TORCH" in eng["ST-TCN"]:
        rows.append(("ST-TCN (PyTorch)", eng["ST-TCN"]["TORCH"]))

    def bar_color(name):
        if "ONNX" in name:
            return GREEN
        if "PyTorch" in name:
            return BLUE
        return "#9AA0A6"

    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    meds = [r[1] for r in rows]
    with plt.rc_context(PC):
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        bars = ax.barh(range(len(names)), meds,
                       color=[bar_color(n) for n in names], height=0.62,
                       edgecolor="white", lw=0.6, zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(3.5, 4000)
        ax.set_xlabel("per-packet latency (µs, CPU batch-1)")
        for b, v in zip(bars, meds):
            ax.text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                    va="center", fontsize=8, color="#444444")
        style_axes(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        handles = [Patch(facecolor="#9AA0A6", label="pipeline component"),
                   Patch(facecolor=BLUE, label="PyTorch (CPU)"),
                   Patch(facecolor=GREEN, label="ONNX Runtime (CPU)")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                  ncol=3, frameon=False, fontsize=8.5, columnspacing=1.2,
                  handlelength=1.0)
        ax.set_title("Fair per-packet latency (single CPU thread, batch-1)",
                     loc="left", fontsize=10.5)
        fig.tight_layout()
        return fig


def parse_st_tcn(path):
    """Parse st_tcn_eval.txt -> (detectors, scenarios, tpr dict, latency rows)."""
    dets, scen = [], []
    tpr = {}
    lat = {}
    for line in open(path):
        if line.startswith("FPR_TPR detector"):
            scen = line.replace("FPR_TPR detector", "").split()
        elif line.startswith("FPR_TPR "):
            parts = line.split("|")
            det = parts[0].split()[1]
            dets.append(det)
            tpr[det] = [float(x) for x in " ".join(parts[1:]).split()]
        elif line.startswith("LATENCY_TORCH ") or line.startswith("LATENCY_ONNX "):
            m = re.match(r"LATENCY_(TORCH|ONNX)\s+(\S+)\s+med=\s*([\d.]+)\s+us", line)
            if m:
                lat.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
    return dets, scen, tpr, lat


def fig_st_tcn():
    path = os.path.join(RES, "st_tcn_eval.txt")
    dets, scen_all, tpr, lat = parse_st_tcn(path)
    keep = [i for i, s in enumerate(scen_all) if s != "normal"]
    scen = [scen_all[i] for i in keep]
    with plt.rc_context(PC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
        ax = axes[0]
        x = np.arange(len(scen))
        w = 0.26
        series = [("timing-only", "timing-only GBM", "#9AA0A6"),
                  ("TCN", "TCN (bytes only)", GREEN),
                  ("ST-TCN", "ST-TCN (input-level fusion)", BLUE)]
        for i, (key, lab, col) in enumerate(series):
            if key not in tpr:
                continue
            ax.bar(x + (i - 1) * w, [tpr[key][j] for j in keep], w,
                   label=lab, color=col, edgecolor="white", lw=0.4, zorder=3)
        ax.axhline(0.5, color=GREY, ls=":", lw=1.0, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("ctx_tamper", "ctx-tamper")
                            .replace("slowslow", "low-slow")
                            for s in scen], rotation=20, ha="right",
                           fontsize=8.5)
        ax.set_ylabel("per-attack TPR @ 1% FPR")
        ax.set_ylim(0, 1.1)
        style_axes(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
                  frameon=False, fontsize=8.5, columnspacing=1.0,
                  handlelength=1.4, handletextpad=0.55)
        ax.set_title("(a) per-attack TPR at global 1% FPR threshold",
                     loc="left", fontsize=10.5)

        ax = axes[1]
        names, meds, labs = [], [], []
        for name in ["TCN", "ST-TCN", "GRU-mean"]:
            for eng, lab in [("TORCH", "PyTorch"), ("ONNX", "ONNX Runtime")]:
                if name in lat and eng in lat[name]:
                    names.append(f"{name}\n({lab})")
                    meds.append(lat[name][eng])
                    labs.append(eng)
        order = np.argsort(meds)[::-1]
        cols = [BLUE if labs[i] == "TORCH" else GREEN for i in order]
        bars = ax.barh(np.arange(len(order)), [meds[i] for i in order],
                       color=cols, height=0.55, edgecolor="white", lw=0.5,
                       zorder=3)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(40, 2600)
        ax.set_xlabel("CPU batch-1 latency (µs)")
        for b, v in zip(bars, [meds[i] for i in order]):
            ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                    va="center", fontsize=8, color="#444444")
        style_axes(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_title("(b) engine comparison, single thread", loc="left",
                     fontsize=10.5)
        fig.tight_layout()
        return fig


def fig_intensity():
    txt = open(os.path.join(RES, "intensity_invariance.txt")).read()
    scen, ks, aucs = [], [], []
    in_det = False
    for line in txt.splitlines():
        if "DETECTOR-LEVEL" in line:
            in_det = True
            continue
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+", line)
        if not m or m.group(1) not in ("ctx_tamper", "dos", "drop", "fuzz",
                                       "slowslow", "tamper"):
            continue
        vals = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
        if in_det:
            scen.append(m.group(1))
            aucs.append(vals)
        else:
            ks.append(vals)
    scen_ks = ["ctx_tamper", "dos", "drop", "fuzz", "slowslow", "tamper"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, mat, title, vmin, vmax, cols in [
            (axes[0], np.array(ks), "KS distance i0.3 vs i1.0 per feature (data level)", 0.0, 1.0,
             ["dt", "byte mean", "entropy"]),
            (axes[1], np.array(aucs), "GBM AUC vs attack intensity", 0.75, 1.0,
             ["i=0.3", "i=0.7", "i=1.0"])]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(scen_ks)))
        ax.set_yticklabels(scen_ks)
        for i in range(mat.shape[0]):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] < 0.85 else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def main():
    gw, ext = load_jitter_files()
    fig_jitter_dist(gw, ext).savefig(os.path.join(FIG, "fig_jitter_dist.pdf"),
                                     bbox_inches="tight")
    print("fig_jitter_dist.pdf")

    levels, rows = parse_robustness(os.path.join(RES, "jitter_robustness.txt"))
    if rows:
        fig_robustness(levels, rows).savefig(
            os.path.join(FIG, "fig_jitter_robustness.pdf"), bbox_inches="tight")
        print("fig_jitter_robustness.pdf")

    recs = parse_ablation(os.path.join(RES, "balanced_ablation.txt"))
    run_path = os.path.join(RES, "balanced_ablation_run.txt")
    if os.path.exists(run_path):
        recs += parse_ablation(run_path)
    if recs:
        fig_ablation(recs).savefig(os.path.join(FIG, "fig_balanced_ablation.pdf"),
                                   bbox_inches="tight")
        print("fig_balanced_ablation.pdf")

    fig_latency().savefig(os.path.join(FIG, "fig_latency.pdf"), bbox_inches="tight")
    print("fig_latency.pdf")

    fig_intensity().savefig(os.path.join(FIG, "fig_intensity.pdf"), bbox_inches="tight")
    print("fig_intensity.pdf")

    st_path = os.path.join(RES, "st_tcn_eval.txt")
    if os.path.exists(st_path):
        fig_st_tcn().savefig(os.path.join(FIG, "fig_st_tcn.pdf"), bbox_inches="tight")
        print("fig_st_tcn.pdf")


if __name__ == "__main__":
    main()

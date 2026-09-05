"""Create the per-scenario dataset composition figure shown in the manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DATA = [
    ("Normal", 16_057, "#8ABB70"),
    ("DoS", 57_245, "#FC7267"),
    ("Message Drop", 5_046, "#FFC75F"),
    ("Fuzzing", 14_860, "#A487D0"),
    ("Low-and-slow", 169, "#F5D9D5"),
    ("Value Tampering", 48_060, "#80CAD2"),
    ("Contextual", 48_159, "#ED8ADE"),
]


def create_figure() -> None:
    output_dir = Path(__file__).resolve().parent
    total = sum(count for _, count, _ in DATA)

    fig = plt.figure(figsize=(9.6, 4.8), facecolor="white")
    pie_ax = fig.add_axes([0.05, 0.14, 0.40, 0.73])
    info_ax = fig.add_axes([0.48, 0.10, 0.48, 0.79])

    values = [count for _, count, _ in DATA]
    colors = [color for _, _, color in DATA]
    pie_ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 0.75},
    )
    pie_ax.set(aspect="equal")
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])

    info_ax.set_axis_off()
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.text(
        0.0,
        0.99,
        "By Scenario",
        color="#163CC0",
        fontsize=16,
        fontweight="semibold",
        va="top",
    )

    row_top = 0.80
    row_step = 0.105
    for index, (label, count, color) in enumerate(DATA):
        y = row_top - index * row_step
        percentage = 100 * count / total
        info_ax.add_patch(
            Rectangle((0.00, y - 0.025), 0.035, 0.043, facecolor=color, edgecolor="#CFCFCF", linewidth=0.5)
        )
        info_ax.text(0.06, y, label, fontsize=12.5, va="center", color="#222222")
        info_ax.text(0.74, y, f"{count:,}", fontsize=12.5, va="center", ha="right", color="#222222")
        info_ax.text(0.98, y, f"({percentage:.1f}%)", fontsize=12.5, va="center", ha="right", color="#222222")

    divider_y = 0.082
    info_ax.plot([0, 0.98], [divider_y, divider_y], color="#8E8E8E", linewidth=1.2)
    info_ax.text(0.06, 0.01, "Total", fontsize=12.5, va="center", color="#222222")
    info_ax.text(0.74, 0.01, f"{total:,}", fontsize=12.5, va="center", ha="right", color="#222222")
    info_ax.text(0.98, 0.01, "(100%)", fontsize=12.5, va="center", ha="right", color="#222222")

    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(
            output_dir / f"fig_scenario_composition.{suffix}",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    create_figure()

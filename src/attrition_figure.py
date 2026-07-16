"""
attrition_figure.py — Generates the SAFE filter attrition funnel chart.

Reads attrition_aggregated.json and produces figures/10_safe_attrition.png.
This figure is the core empirical contribution of the SAFE filter analysis.

Usage:
    python attrition_figure.py
    or called from visualize.generate_attrition_figure()
"""

import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from config import CFG


STAGE_LABELS = [
    "Generated\n(Raw)",
    "Stage 1\nRDKit Valid",
    "Stage 2\nGeometric",
    "Stage 3\nPAINS Clean",
    "Stage 4\nLipinski/Veber",
    "Final\nRetained",
]

STAGE_KEYS = [
    "stage_0_generated",
    "stage_1_rdkit_valid",
    "stage_2_geometric",
    "stage_3_pains",
    "stage_4_lipinski_veber",
    "stage_5_unique_final",
]

TEMP_COLORS = {
    "0.2": "#4C72B0",
    "0.5": "#DD8452",
    "0.7": "#55A868",
    "1.0": "#C44E52",
}


def generate_attrition_figure(avg_attrition: dict, save_path: str = None):
    """
    Generate publication-quality attrition funnel figure.

    Args:
        avg_attrition: dict of {temp_str: {stage_key: count, ...}}
        save_path: output path; defaults to figures/10_safe_attrition.png
    """
    if save_path is None:
        save_path = os.path.join(CFG.FIGURES_DIR, "11_safe_attrition.png")

    temps = sorted(avg_attrition.keys(), key=float)
    n_stages = len(STAGE_LABELS)
    n_temps  = len(temps)
    x        = np.arange(n_stages)
    bar_w    = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "SAFE Filter Attrition: Per-Stage Molecule Counts Across Temperatures",
        fontsize=14, fontweight="bold", y=1.02
    )

    # --- Left: Grouped bar chart (absolute counts) ---
    ax = axes[0]
    for i, temp in enumerate(temps):
        counts = [avg_attrition[temp].get(k, 0) for k in STAGE_KEYS]
        offset = (i - n_temps / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, counts, bar_w,
                        label=f"T={temp}",
                        color=TEMP_COLORS.get(temp, "#888888"),
                        alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("SAFE Filter Stage", fontsize=11)
    ax.set_ylabel("Number of Molecules", fontsize=11)
    ax.set_title("Absolute Molecule Counts per SAFE Stage", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.legend(title="Temperature", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right: Funnel / area chart (retention rate relative to generated) ---
    ax2 = axes[1]
    for temp in temps:
        n_gen    = avg_attrition[temp].get("stage_0_generated", 1000)
        pct_vals = [avg_attrition[temp].get(k, 0) / n_gen * 100
                    for k in STAGE_KEYS]
        ax2.plot(range(n_stages), pct_vals, "-o",
                 label=f"T={temp}",
                 color=TEMP_COLORS.get(temp, "#888888"),
                 linewidth=2, markersize=7)
        # Annotate final retained %
        ax2.annotate(
            f"{pct_vals[-1]:.1f}%",
            xy=(n_stages - 1, pct_vals[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=8, color=TEMP_COLORS.get(temp, "#888888")
        )

    ax2.set_xlabel("SAFE Filter Stage", fontsize=11)
    ax2.set_ylabel("% of Generated Molecules Retained", fontsize=11)
    ax2.set_title("Retention Rate Through Each SAFE Stage", fontsize=12)
    ax2.set_xticks(range(n_stages))
    ax2.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.legend(title="Temperature", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Attrition] Figure saved -> {save_path}")
    return save_path


def generate_attrition_from_file(
    json_path: str = None,
    save_path: str = None
):
    """Load avg_attrition from JSON file and generate figure."""
    if json_path is None:
        json_path = os.path.join(CFG.RESULTS_DIR, "attrition_aggregated.json")
    with open(json_path) as f:
        avg_attrition = json.load(f)
    return generate_attrition_figure(avg_attrition, save_path)


if __name__ == "__main__":
    generate_attrition_from_file()

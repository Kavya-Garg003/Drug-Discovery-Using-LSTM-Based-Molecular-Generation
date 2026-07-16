"""
visualize.py — All publication-quality figures for the paper.

Figures produced:
  01_training_loss_curve.png       — epoch vs train/val loss
  02_metrics_comparison.png        — model vs baseline bar chart
  03_temperature_sweep.png         — metrics across temperatures (ablation)
  04_property_distributions.png    — MW, LogP, TPSA, QED with Lipinski lines
  05_toxicity_breakdown.png        — PAINS % + ADMET component scores
  06_top12_molecules.png           — RDKit structure grid
  07_radar_top5.png                — multi-property radar chart
  08_correlation_heatmap.png       — Pearson correlation with annotation
  09_novelty_tanimoto_hist.png     — distribution of max sim to training set
  10_scaffold_diversity_pie.png    — scaffold vs non-scaffold
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Chem import AllChem, DataStructs

from config import CFG

# -- Global style --------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
DPI = 180


def _save(fig, name: str):
    path = os.path.join(CFG.FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] Saved -> {path}")


# ----------------------------------------------------------------------------─
# 01 Training loss curve
# ----------------------------------------------------------------------------─

def plot_training_history(history_csv: str):
    df  = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df.index + 1, df["loss"],     label="Train",      color=PALETTE[0])
    ax.plot(df.index + 1, df["val_loss"], label="Validation", color=PALETTE[1], linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Sparse Categorical CE)")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend()
    ax.annotate(
        f'Best val loss: {df["val_loss"].min():.4f}\n(epoch {df["val_loss"].idxmin()+1})',
        xy=(df["val_loss"].idxmin() + 1, df["val_loss"].min()),
        xytext=(len(df) * 0.6, df["val_loss"].max() * 0.85),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    ax = axes[1]
    if "accuracy" in df.columns:
        ax.plot(df.index + 1, df["accuracy"],     label="Train",      color=PALETTE[0])
        ax.plot(df.index + 1, df["val_accuracy"], label="Validation", color=PALETTE[1], linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Token Accuracy")
        ax.set_title("Training & Validation Accuracy", fontweight="bold")
        ax.legend()

    fig.suptitle("SELFIES-LSTM Training Convergence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_training_loss_curve.png")


# ----------------------------------------------------------------------------─
# 02 Model vs. baseline bar chart
# ----------------------------------------------------------------------------─

def plot_model_vs_baseline(model_summary: dict, baseline_summary: dict):
    metrics = ["validity_%", "uniqueness_%", "novelty_%",
               "drug_like_%", "pains_clean_%"]
    labels  = ["Validity", "Uniqueness", "Novelty", "Drug-Like", "PAINS-Clean"]

    def _pct(s, key):
        v = s.get(key, "0")
        return float(str(v).replace("%", ""))

    model_vals    = [_pct(model_summary,    m) for m in metrics]
    baseline_vals = [_pct(baseline_summary, m) for m in metrics]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, model_vals,    w, label="SELFIES-LSTM (Ours)",  color=PALETTE[0])
    b2 = ax.bar(x + w/2, baseline_vals, w, label="Random Baseline",      color=PALETTE[1])

    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 115)
    ax.set_title("SELFIES-LSTM vs. Random Baseline on Key Metrics",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "02_metrics_comparison.png")


# ----------------------------------------------------------------------------─
# 03 Temperature sweep (ablation)
# ----------------------------------------------------------------------------─

def plot_temperature_sweep(summaries: list[dict]):
    """
    summaries: list of summary dicts, one per temperature.
    Each must have 'label' = temperature value.
    """
    metrics  = ["validity_%", "uniqueness_%", "novelty_%",
                "drug_like_%", "avg_QED", "internal_diversity"]
    ylabels  = ["Validity (%)", "Uniqueness (%)", "Novelty (%)",
                "Drug-Like (%)", "Avg QED", "Int. Diversity"]

    temps = [s["label"] for s in summaries]

    def _val(s, k):
        v = s.get(k, 0)
        try:
            return float(str(v).replace("%", ""))
        except Exception:
            return 0.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (m, yl) in enumerate(zip(metrics, ylabels)):
        vals = [_val(s, m) for s in summaries]
        axes[i].plot(temps, vals, marker="o", color=PALETTE[i], linewidth=2, markersize=8)
        axes[i].set_xlabel("Temperature")
        axes[i].set_ylabel(yl)
        axes[i].set_title(yl, fontweight="bold")
        axes[i].set_xticks(temps)
        for x_, y_ in zip(temps, vals):
            axes[i].annotate(f"{y_:.1f}", (x_, y_), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle("Temperature Ablation Study — Effect on Generation Quality",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_temperature_sweep.png")


# ----------------------------------------------------------------------------─
# 04 Property distributions
# ----------------------------------------------------------------------------─

def plot_property_distributions(df: pd.DataFrame):
    props   = ["MolWeight", "LogP", "TPSA", "HBD", "HBA", "QED"]
    limits  = {"MolWeight": 500, "LogP": 5, "HBD": 5, "HBA": 10, "TPSA": 140}
    lim_lbl = {"MolWeight": "Lipinski limit (500 Da)",
                "LogP"     : "Lipinski limit (5)",
                "HBD"      : "Lipinski limit (5)",
                "HBA"      : "Lipinski limit (10)",
                "TPSA"     : "Veber limit (140 Å²)"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, prop in enumerate(props):
        if prop not in df.columns:
            continue
        ax = axes[i]
        ax.hist(df[prop], bins=25, color=PALETTE[i], edgecolor="white",
                alpha=0.85, density=False)
        if prop in limits:
            ax.axvline(limits[prop], color="red", linestyle="--", linewidth=1.8,
                       label=lim_lbl[prop])
            ax.legend(fontsize=8)
        ax.set_xlabel(prop)
        ax.set_ylabel("Count")
        # Annotate with mean
        mu = df[prop].mean()
        ax.axvline(mu, color="navy", linestyle=":", linewidth=1.5,
                   label=f"Mean={mu:.2f}")
        ax.set_title(
            f"{prop} — mean={mu:.2f}, σ={df[prop].std():.2f}",
            fontweight="bold", fontsize=10
        )
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Molecular Property Distributions of Generated Drug Candidates\n"
                 "(dashed red = Lipinski/Veber thresholds, dotted navy = mean)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_property_distributions.png")


# ----------------------------------------------------------------------------─
# 05 Toxicity breakdown
# ----------------------------------------------------------------------------─

def plot_toxicity_breakdown(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: toxicity score histogram
    ax = axes[0]
    ax.hist(df["Toxicity"], bins=20, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.axvline(df["Toxicity"].mean(), color="navy", linestyle="--", linewidth=2,
               label=f'Mean={df["Toxicity"].mean():.3f}')
    ax.set_xlabel("ADMET Toxicity Score (lower = safer)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ADMET Toxicity Scores\n"
                 "(TPSA + logP + MW + PAINS/Brenk weighted composite)",
                 fontweight="bold")
    ax.legend()

    # Right: PAINS pass/fail pie
    ax = axes[1]
    pains_ok  = df["PassesPAINS"].sum()
    pains_bad = len(df) - pains_ok
    ax.pie([pains_ok, pains_bad],
           labels=[f"PAINS-clean\n({pains_ok})", f"PAINS-flagged\n({pains_bad})"],
           colors=[PALETTE[2], PALETTE[3]],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("PAINS Filter — Pan-Assay\nInterference Compounds",
                 fontweight="bold")

    fig.suptitle("Toxicity & Drug-Safety Profile of Generated Molecules",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_toxicity_breakdown.png")


# ----------------------------------------------------------------------------─
# 06 Top-12 molecule grid
# ----------------------------------------------------------------------------─

def plot_top_molecules(df: pd.DataFrame, n: int = 12):
    top = df.head(n)
    mols, legs = [], []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol:
            mols.append(mol)
            legs.append(
                f"QED={row['QED']:.2f} | Tox={row['Toxicity']:.2f}\nScore={row['FinalScore']:.2f}"
            )
    if not mols:
        print("[Fig] No molecules for top-grid.")
        return
    img = Draw.MolsToGridImage(
        mols, molsPerRow=4, subImgSize=(350, 300),
        legends=legs, returnPNG=False,
    )
    path = os.path.join(CFG.FIGURES_DIR, "06_top12_molecules.png")
    img.save(path)
    print(f"  [Fig] Saved -> {path}")


# ----------------------------------------------------------------------------─
# 07 Radar chart — top 5
# ----------------------------------------------------------------------------─

def plot_radar(df: pd.DataFrame, n: int = 5):
    top = df.head(n)
    cats = ["QED", "DrugScore", "LowTox", "LowTPSA", "LowMW"]

    def _norm(row):
        low_tox  = 1 - row["Toxicity"]                      # invert (lower=better)
        low_tpsa = 1 - min(1, row["TPSA"] / 140)
        low_mw   = 1 - min(1, (row["MolWeight"] - 150) / 450)
        return [row["QED"], row["DrugScore"], low_tox, low_tpsa, low_mw]

    N   = len(cats)
    ang = [n_ / N * 2 * pi for n_ in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(top.iterrows()):
        vals = _norm(row) + [_norm(row)[0]]
        ax.plot(ang, vals, linewidth=2, color=PALETTE[i],
                label=f"{row['SMILES'][:18]}... (Score={row['FinalScore']:.2f})")
        ax.fill(ang, vals, alpha=0.10, color=PALETTE[i])

    ax.set_thetagrids(np.degrees(ang[:-1]), cats)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart — Top 5 Generated Molecules\n"
                 "(Higher = Better on All Axes)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)
    plt.tight_layout()
    _save(fig, "07_radar_top5.png")


# ----------------------------------------------------------------------------─
# 08 Correlation heatmap
# ----------------------------------------------------------------------------─

def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["MolWeight", "LogP", "TPSA", "HBD", "HBA",
            "QED", "DrugScore", "Toxicity", "FinalScore"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        vmin=-1, vmax=1,
    )
    ax.set_title(
        "Pearson Correlation Matrix — Molecular Properties & Evaluation Metrics\n"
        "(lower triangle; values from −1 to +1)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "08_correlation_heatmap.png")


# ----------------------------------------------------------------------------─
# 09 Tanimoto novelty distribution
# ----------------------------------------------------------------------------─

def plot_novelty_tanimoto(gen_smiles: list[str], train_smiles: list[str], n_train: int = 2000):
    from rdkit.Chem import AllChem, DataStructs
    print("[Fig09] Computing Tanimoto novelty histogram ...")
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles[:n_train] if Chem.MolFromSmiles(s)]
    train_fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in train_mols]

    max_sims = []
    for smi in gen_smiles[:500]:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        if sims:
            max_sims.append(max(sims))

    if not max_sims:
        print("[Fig09] No data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_sims, bins=30, color=PALETTE[4], edgecolor="white", alpha=0.85)
    ax.axvline(0.4, color="red", linestyle="--", linewidth=2,
               label="Novelty threshold (Tanimoto < 0.4)")
    novel_pct = 100 * sum(s < 0.4 for s in max_sims) / len(max_sims)
    ax.axvline(np.mean(max_sims), color="navy", linestyle=":", linewidth=2,
               label=f"Mean similarity = {np.mean(max_sims):.3f}")
    ax.set_xlabel("Max Tanimoto Similarity to Training Set")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Tanimoto Novelty Distribution\n"
        f"{novel_pct:.1f}% of generated molecules are novel (Tanimoto < 0.4 to any training molecule)",
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "09_novelty_tanimoto_hist.png")


# ----------------------------------------------------------------------------─
# Convenience: generate all figures
# ----------------------------------------------------------------------------─

def generate_all_figures(
    history_csv: str,
    df_model: pd.DataFrame,
    df_baseline: pd.DataFrame,
    summaries_by_temp: list,
    model_summary: dict,
    baseline_summary: dict,
    train_smiles: list,
):
    print("\n[Visualizer] Generating all figures ...")

    plot_training_history(history_csv)
    plot_model_vs_baseline(model_summary, baseline_summary)
    plot_temperature_sweep(summaries_by_temp)
    plot_property_distributions(df_model)
    plot_toxicity_breakdown(df_model)
    plot_top_molecules(df_model)
    plot_radar(df_model)
    plot_correlation_heatmap(df_model)
    plot_novelty_tanimoto(df_model["SMILES"].tolist(), train_smiles)

    print(f"\n[Visualizer] All figures saved to -> {CFG.FIGURES_DIR}")


# ----------------------------------------------------------------------------─
# Attrition figure (called separately with avg_attrition dict)
# ----------------------------------------------------------------------------─

def generate_attrition_figure(avg_attrition: dict, save_path: str = None):
    """Delegate to dedicated attrition_figure module."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from attrition_figure import generate_attrition_figure as _gen
    return _gen(avg_attrition, save_path)

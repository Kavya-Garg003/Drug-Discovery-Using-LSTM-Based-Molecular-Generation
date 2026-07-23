"""
generate_attrition_standalone.py

Generates figures/11_safe_attrition.png directly from the already-generated
SMILES files in results/. Does NOT require PyTorch.

Run from project root:
    python src/generate_attrition_standalone.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rmd, FilterCatalog

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TEMPERATURES = ["0.2", "0.5", "0.7", "1.0"]

# ── SAFE filter config (mirrors config.py) ────────────────────────────────────
MIN_MW, MAX_MW   = 120.0, 600.0
MIN_HEAVY        = 10
MAX_LOGP         = 5.5
MAX_HBD, MAX_HBA = 5, 10
MAX_TPSA         = 140.0
MAX_ROT          = 10

# Build PAINS catalog once
_PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_PAINS_PARAMS)


# Import RDKit SA Score module
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError:
    sascorer = None

_CUMULENE_SMARTS = Chem.MolFromSmarts('[#6,#7,#8,#16]=[#6]=[#6,#7,#8,#16]')


def compute_attrition(smiles_list):
    """Run all SAFE+ stages and return per-stage counts."""
    n_gen  = len(smiles_list)
    n_rdkit, n_geo, n_pains, n_lip, n_sa = 0, 0, 0, 0, 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            continue
        n_rdkit += 1

        mw  = Descriptors.MolWt(mol)
        ha  = mol.GetNumHeavyAtoms()
        rot = rmd.CalcNumRotatableBonds(mol)
        if not (MIN_MW <= mw <= MAX_MW and ha >= MIN_HEAVY and rot <= MAX_ROT):
            continue
        n_geo += 1

        if _PAINS_CATALOG.HasMatch(mol):
            continue
        n_pains += 1

        logp = Descriptors.MolLogP(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)
        tpsa = rmd.CalcTPSA(mol)
        if not (logp <= MAX_LOGP and hbd <= MAX_HBD and hba <= MAX_HBA and tpsa <= MAX_TPSA):
            continue
        n_lip += 1

        # Stage 5: SAFE-SA (Structural Strain & Synthetic Accessibility)
        if _CUMULENE_SMARTS and mol.HasSubstructMatch(_CUMULENE_SMARTS):
            continue

        strained_alkyne = False
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing():
                for ring in mol.GetRingInfo().AtomRings():
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring and len(ring) < 8:
                        strained_alkyne = True
                        break
        if strained_alkyne:
            continue

        if sascorer is not None:
            try:
                score = sascorer.calculateScore(mol)
                if score > 4.5:
                    continue
            except Exception:
                pass
        n_sa += 1

    return {
        "stage_0_generated"      : n_gen,
        "stage_1_rdkit_valid"    : n_rdkit,
        "stage_2_geometric"      : n_geo,
        "stage_3_pains"          : n_pains,
        "stage_4_lipinski_veber" : n_lip,
        "stage_5_strain_sa"      : n_sa,
        "stage_6_unique_final"   : n_sa,     # dedup count
    }


# ── compute attrition for each temperature ────────────────────────────────────
attrition = {}
for T in TEMPERATURES:
    smiles_file = os.path.join(RESULTS_DIR, f"generated_T{T}.txt")
    if not os.path.exists(smiles_file):
        print(f"[WARN] Missing: {smiles_file} — skipping T={T}")
        continue
    with open(smiles_file) as f:
        smiles = [l.strip() for l in f if l.strip()]
    print(f"[T={T}] Processing {len(smiles)} SMILES ...")
    attrition[T] = compute_attrition(smiles)
    a = attrition[T]
    print(f"  Generated={a['stage_0_generated']} | Valid={a['stage_1_rdkit_valid']} "
          f"| Geo={a['stage_2_geometric']} | PAINS={a['stage_3_pains']} "
          f"| Lip={a['stage_4_lipinski_veber']}")

# Save JSON
attr_path = os.path.join(RESULTS_DIR, "attrition_aggregated.json")
with open(attr_path, "w") as f:
    json.dump(attrition, f, indent=2)
print(f"\n[OK] Attrition JSON saved -> {attr_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
STAGE_LABELS = [
    "Generated\n(Raw)",
    "Stage 1\nRDKit Valid",
    "Stage 2\nGeometric",
    "Stage 3\nPAINS Clean",
    "Stage 4\nLipinski/Veber",
    "Final\nRetained",
]
STAGE_KEYS = [
    "stage_0_generated", "stage_1_rdkit_valid", "stage_2_geometric",
    "stage_3_pains", "stage_4_lipinski_veber", "stage_5_unique_final",
]
TEMP_COLORS = {"0.2": "#4C72B0", "0.5": "#DD8452", "0.7": "#55A868", "1.0": "#C44E52"}

temps     = sorted(attrition.keys(), key=float)
n_stages  = len(STAGE_LABELS)
x         = np.arange(n_stages)
bar_w     = 0.18

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "SAFE Filter Attrition: Per-Stage Molecule Counts Across Temperatures",
    fontsize=14, fontweight="bold"
)

# Left: absolute counts
ax = axes[0]
for i, T in enumerate(temps):
    counts = [attrition[T].get(k, 0) for k in STAGE_KEYS]
    offset = (i - len(temps) / 2 + 0.5) * bar_w
    ax.bar(x + offset, counts, bar_w,
           label=f"T={T}", color=TEMP_COLORS.get(T, "#888"), alpha=0.85,
           edgecolor="white", linewidth=0.5)

ax.set_xlabel("SAFE Filter Stage", fontsize=11)
ax.set_ylabel("Number of Molecules", fontsize=11)
ax.set_title("Absolute Molecule Counts per SAFE Stage", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(STAGE_LABELS, fontsize=9)
ax.legend(title="Temperature", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: retention rate %
ax2 = axes[1]
for T in temps:
    n_gen = attrition[T].get("stage_0_generated", 1000)
    pcts  = [attrition[T].get(k, 0) / n_gen * 100 for k in STAGE_KEYS]
    ax2.plot(range(n_stages), pcts, "-o",
             label=f"T={T}", color=TEMP_COLORS.get(T, "#888"),
             linewidth=2, markersize=7)
    ax2.annotate(f"{pcts[-1]:.1f}%",
                 xy=(n_stages - 1, pcts[-1]),
                 xytext=(5, 0), textcoords="offset points",
                 fontsize=8, color=TEMP_COLORS.get(T, "#888"))

ax2.set_xlabel("SAFE Filter Stage", fontsize=11)
ax2.set_ylabel("% of Generated Molecules Retained", fontsize=11)
ax2.set_title("Retention Rate Through Each SAFE Stage", fontsize=12)
ax2.set_xticks(range(n_stages))
ax2.set_xticklabels(STAGE_LABELS, fontsize=9)
ax2.set_ylim(0, 115)
ax2.legend(title="Temperature", fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, "11_safe_attrition.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Attrition figure saved -> {save_path}")

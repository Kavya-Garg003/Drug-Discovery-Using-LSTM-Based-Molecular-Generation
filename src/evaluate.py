"""
evaluate.py — Full multi-metric evaluation suite for generated molecules.

Implements:
  1. Validity      — % that parse as valid RDKit molecules (should be ~100% with SELFIES)
  2. Uniqueness    — % distinct molecules among candidates
  3. Novelty       — % not in training set (Tanimoto similarity < 0.4 threshold)
  4. Drug-Likeness — Lipinski Rule of 5 compliance (strict)
  5. QED           — Quantitative Estimate of Drug-likeness (RDKit built-in)
  6. REAL Toxicity — ADMET proxy: TPSA + PAINS + logP + MW (NOT length-based!)
  7. Internal Diversity — avg pairwise Tanimoto distance (MOSES metric)
  8. Scaffold Diversity — unique Murcko scaffolds / total valid
  9. Final Score   — composite (drug_score + qed) / 2 − toxicity
  10. Attrition    — per-stage SAFE filter attrition counts (new)

CRITICAL FIX vs original notebook:
  - Toxicity is now based on validated ADMET descriptors, not string length
  - Minimum MW/atom filter applied so trivial molecules never top the ranking
  - Attrition tracking added per reviewer request
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, QED,
    FilterCatalog, rdMolDescriptors as rmd, RDConfig,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from tqdm import tqdm
from config import CFG

warnings.filterwarnings("ignore")

# Import RDKit SA Score module
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError:
    sascorer = None

_CUMULENE_SMARTS = Chem.MolFromSmarts('[#6,#7,#8,#16]=[#6]=[#6,#7,#8,#16]')


# ----------------------------------------------------------------------------─
# ADMET / Toxicity — REPLACES the naive len(smiles)/50 proxy
# ----------------------------------------------------------------------------─

# Build PAINS filter catalog once
_PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_PAINS_PARAMS)
# Brenk structural alerts
_BRENK_PARAMS = FilterCatalog.FilterCatalogParams()
_BRENK_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK_CATALOG = FilterCatalog.FilterCatalog(_BRENK_PARAMS)


def admet_toxicity(mol) -> float:
    """
    Evidence-based ADMET toxicity / undesirability proxy.
    Returns a score in [0, 1] where:
        0.0 = excellent ADMET profile (low toxicity)
        1.0 = poor ADMET profile (high toxicity)

    Components:
      - TPSA penalty  : TPSA > 140 Å² -> poor oral absorption (Veber rule)
      - logP penalty  : logP > 5      -> poor solubility
      - MW penalty    : MW > 500      -> poor absorption (Lipinski)
      - PAINS penalty : Pan-Assay Interference — known false-positive alerts
      - Brenk penalty : Structural alerts for toxic motifs
    """
    tpsa  = rmd.CalcTPSA(mol)
    logp  = Descriptors.MolLogP(mol)
    mw    = Descriptors.MolWt(mol)

    # Continuous penalties normalised to [0,1]
    tpsa_pen  = min(1.0, max(0.0, (tpsa  - 140) / 60))   # 0 if ≤140, 1 if ≥200
    logp_pen  = min(1.0, max(0.0, (logp  -   5) / 3))    # 0 if ≤5,   1 if ≥8
    mw_pen    = min(1.0, max(0.0, (mw    - 500) / 200))  # 0 if ≤500, 1 if ≥700

    pains_pen = 1.0 if _PAINS_CATALOG.HasMatch(mol) else 0.0
    brenk_pen = 0.5 if _BRENK_CATALOG.HasMatch(mol) else 0.0

    score = (
        CFG.TOX_TPSA_WEIGHT  * tpsa_pen  +
        CFG.TOX_LOGP_WEIGHT  * logp_pen  +
        CFG.TOX_MW_WEIGHT    * mw_pen    +
        CFG.TOX_PAINS_WEIGHT * min(1.0, pains_pen + brenk_pen)
    )
    return round(float(score), 4)


# ----------------------------------------------------------------------------─
# Drug-likeness
# ----------------------------------------------------------------------------─

def lipinski_score(mol) -> float:
    """
    Lipinski Rule of 5 compliance score ∈ {0.0, 0.25, 0.50, 0.75, 1.0}.
    Each criterion passed = 0.25 added.
    """
    mw  = Descriptors.MolWt(mol)
    lp  = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return round(
        sum([mw <= 500, lp <= 5, hbd <= 5, hba <= 10]) / 4, 4
    )


# ----------------------------------------------------------------------------─
# Filters (SAFE stages — now individually instrumented)
# ----------------------------------------------------------------------------─

def passes_geometric_filter(mol) -> bool:
    """SAFE Stage 1: Geometric hard filter (MW, heavy atoms)."""
    mw          = Descriptors.MolWt(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    return (CFG.MIN_MW <= mw <= CFG.MAX_MW and
            heavy_atoms >= CFG.MIN_HEAVY_ATOMS)


def passes_pains_filter(mol) -> bool:
    """SAFE Stage 2: PAINS substructure screen."""
    return not _PAINS_CATALOG.HasMatch(mol)


def passes_lipinski_veber(mol) -> bool:
    """SAFE Stage 3: Lipinski/Veber oral bioavailability rules."""
    logp = Descriptors.MolLogP(mol)
    tpsa = rmd.CalcTPSA(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rot  = rmd.CalcNumRotatableBonds(mol)
    return (logp <= CFG.MAX_LOGP and
            hbd  <= CFG.MAX_HBD  and
            hba  <= CFG.MAX_HBA  and
            tpsa <= CFG.MAX_TPSA and
            rot  <= CFG.MAX_ROT_BONDS)


def passes_strain_and_sa_filter(mol) -> bool:
    """SAFE+ Stage 4: Structural Strain & Synthetic Accessibility (SA Score)."""
    if not CFG.ALLOW_CUMULENES and _CUMULENE_SMARTS and mol.HasSubstructMatch(_CUMULENE_SMARTS):
        return False

    if not CFG.ALLOW_STRAINED_RING:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing():
                for ring in mol.GetRingInfo().AtomRings():
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring and len(ring) < 8:
                        return False

    if sascorer is not None:
        try:
            score = sascorer.calculateScore(mol)
            if score > CFG.MAX_SA_SCORE:
                return False
        except Exception:
            pass

    return True


from rdkit.Chem import AllChem, rdForceFieldHelpers
from scipy.stats import wasserstein_distance


def compute_mmff94_strain(mol) -> float:
    """Computes 3D MMFF94 force field conformational strain energy (kcal/mol)."""
    try:
        m_h = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(m_h, randomSeed=CFG.SEED, maxAttempts=10)
        if res != 0:
            return 999.0
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(m_h)
        if props is None:
            return 999.0
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m_h, props)
        if ff is None:
            return 999.0
        e_init = ff.CalcEnergy()
        ff.Minimize(maxIts=200)
        e_min = ff.CalcEnergy()
        return round(float(e_init - e_min), 2)
    except Exception:
        return 999.0


def compute_scaffold_shannon_entropy(mols: list) -> float:
    """Calculates the Shannon Entropy H = -sum(p_i * log2(p_i)) of Bemis-Murcko scaffolds."""
    from collections import Counter
    scaffolds = []
    for m in mols:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.append(Chem.MolToSmiles(sc))
        except Exception:
            pass
    if not scaffolds:
        return 0.0
    counts = Counter(scaffolds)
    total = len(scaffolds)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs)
    return round(float(entropy), 4)


def compute_5d_wasserstein(df_gen: pd.DataFrame, df_ref: pd.DataFrame) -> float:
    """Calculates average Wasserstein-1 Distance across 5 normalized property distributions."""
    cols = ["MolWeight", "LogP", "TPSA", "QED", "SAScore"]
    dists = []
    for col in cols:
        if col in df_gen.columns and col in df_ref.columns:
            min_val = min(df_gen[col].min(), df_ref[col].min())
            max_val = max(df_gen[col].max(), df_ref[col].max())
            rng = max(max_val - min_val, 1e-5)
            v1 = (df_gen[col] - min_val) / rng
            v2 = (df_ref[col] - min_val) / rng
            dists.append(wasserstein_distance(v1, v2))
    return round(float(np.mean(dists)), 4) if dists else 0.0


def passes_basic_filters(mol) -> bool:
    """Combined hard filter: SAFE+ (geometric + PAINS + Lipinski/Veber + Strain/SA)."""
    return (passes_geometric_filter(mol) and
            passes_pains_filter(mol) and
            passes_lipinski_veber(mol) and
            passes_strain_and_sa_filter(mol))


# ----------------------------------------------------------------------------─
# Tanimoto diversity
# ----------------------------------------------------------------------------─

def morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def internal_diversity(mols: list, sample: int = 200) -> float:
    """
    Average pairwise Tanimoto distance among generated molecules.
    1.0 = maximally diverse, 0.0 = all identical.
    Sampled for speed when > `sample` molecules.
    """
    if len(mols) < 2:
        return 0.0
    fps = [morgan_fp(m) for m in mols]
    if len(fps) > sample:
        rng = np.random.default_rng(CFG.SEED)
        idx = rng.choice(len(fps), sample, replace=False)
        fps = [fps[i] for i in idx]
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return round(float(np.mean(dists)), 4) if dists else 0.0


def novelty_tanimoto(gen_mol, train_fps: list, threshold: float = 0.4) -> bool:
    """
    A generated molecule is novel if its max Tanimoto similarity to any
    training molecule is below `threshold` (i.e. not memorised).
    """
    fp = morgan_fp(gen_mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims) < threshold if sims else True


def scaffold_diversity(mols: list) -> float:
    """Fraction of unique Murcko scaffolds among valid molecules."""
    scaffolds = set()
    for mol in mols:
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(sc))
        except Exception:
            pass
    return round(len(scaffolds) / len(mols), 4) if mols else 0.0


# ----------------------------------------------------------------------------─
# Full evaluation pipeline (with attrition tracking)
# ----------------------------------------------------------------------------─

def evaluate(
    smiles_list: list,
    train_smiles: list,
    label: str = "model",
    save_csv: bool = True,
) -> dict:
    """
    Complete evaluation of a list of generated SMILES strings.

    Args:
        smiles_list  : list of generated SMILES (may include invalids)
        train_smiles : SMILES from training set (for novelty check)
        label        : identifier for saving results (e.g. "temp_0.7")
        save_csv     : whether to save per-molecule CSV

    Returns:
        dict with keys: "summary", "df", "attrition"
    """
    print(f"\n[Evaluator] Evaluating {len(smiles_list):,} candidates (label={label}) ...")

    # -- Pre-compute training fingerprints (sample 5K for speed) ----------
    print("[Evaluator] Building training fingerprints ...")
    train_sample = train_smiles[:5_000]
    train_fps = []
    for smi in tqdm(train_sample, desc="train fps"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(morgan_fp(mol))

    # -- Attrition counters -----------------------------------------------
    n_generated      = len(smiles_list)
    n_rdkit_valid    = 0   # pass RDKit parse
    n_geometric      = 0   # pass Stage 1: geometric (MW, heavy atoms)
    n_pains          = 0   # pass Stage 2: PAINS screen
    n_lipinski_veber = 0   # pass Stage 3: Lipinski/Veber
    n_strain_sa      = 0   # pass Stage 4: Strain & SA Score (SAFE+)
    n_unique         = 0   # after deduplication

    # -- Parse + filter (staged) ------------------------------------------
    rows = []
    valid_mols = []
    seen_smiles = set()

    for smi in tqdm(smiles_list, desc="evaluating"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_rdkit_valid += 1

        # Stage 1: Geometric filter
        if not passes_geometric_filter(mol):
            continue
        n_geometric += 1

        # Stage 2: PAINS screen
        pains_pass = passes_pains_filter(mol)
        if not pains_pass:
            continue
        n_pains += 1

        # Stage 3: Lipinski/Veber
        if not passes_lipinski_veber(mol):
            continue
        n_lipinski_veber += 1

        # Stage 4: Structural Strain & SA Score (SAFE+)
        if not passes_strain_and_sa_filter(mol):
            continue
        n_strain_sa += 1

        # Compute all properties
        canonical = Chem.MolToSmiles(mol)
        mw     = round(Descriptors.MolWt(mol), 2)
        logp   = round(Descriptors.MolLogP(mol), 2)
        tpsa   = round(rmd.CalcTPSA(mol), 2)
        hbd    = Descriptors.NumHDonors(mol)
        hba    = Descriptors.NumHAcceptors(mol)
        rot    = rmd.CalcNumRotatableBonds(mol)
        arom   = rmd.CalcNumAromaticRings(mol)
        ha     = mol.GetNumHeavyAtoms()
        qed_sc = round(QED.qed(mol), 4)
        lip    = lipinski_score(mol)
        tox    = admet_toxicity(mol)
        sa_sc  = round(sascorer.calculateScore(mol), 2) if sascorer is not None else 3.0
        novel  = novelty_tanimoto(mol, train_fps) if train_fps else True
        final  = round((lip + qed_sc) / 2 - tox, 4)
        mmff_e = compute_mmff94_strain(mol)

        valid_mols.append(mol)
        rows.append({
            "SMILES"       : canonical,
            "MolWeight"    : mw,
            "LogP"         : logp,
            "TPSA"         : tpsa,
            "HBD"          : hbd,
            "HBA"          : hba,
            "RotBonds"     : rot,
            "AromaticRings": arom,
            "HeavyAtoms"   : ha,
            "QED"          : qed_sc,
            "SAScore"      : sa_sc,
            "MMFF_Strain"  : mmff_e,
            "DrugScore"    : lip,
            "Toxicity"     : tox,
            "FinalScore"   : final,
            "IsNovel"      : novel,
            "IsDrugLike"   : lip >= 0.75,
            "PassesPAINS"  : True,  # all molecules here have passed PAINS
        })

    df = pd.DataFrame(rows).drop_duplicates("SMILES").sort_values(
        "FinalScore", ascending=False
    ).reset_index(drop=True)
    n_unique = len(df)

    # -- Attrition report --------------------------------------------------
    attrition = {
        "stage_0_generated"      : n_generated,
        "stage_1_rdkit_valid"    : n_rdkit_valid,
        "stage_2_geometric"      : n_geometric,
        "stage_3_pains"          : n_pains,
        "stage_4_lipinski_veber" : n_lipinski_veber,
        "stage_5_strain_sa"      : n_strain_sa,
        "stage_6_unique_final"   : n_unique,
        # Pass rates
        "pct_rdkit_valid"        : round(n_rdkit_valid / n_generated * 100, 1) if n_generated else 0,
        "pct_geometric"          : round(n_geometric / max(n_rdkit_valid, 1) * 100, 1),
        "pct_pains"              : round(n_pains / max(n_geometric, 1) * 100, 1),
        "pct_lipinski_veber"     : round(n_lipinski_veber / max(n_pains, 1) * 100, 1),
        "pct_strain_sa"          : round(n_strain_sa / max(n_lipinski_veber, 1) * 100, 1),
        "pct_overall_retained"   : round(n_unique / n_generated * 100, 1) if n_generated else 0,
    }

    print("\n-- SAFE+ Attrition Report ---------------------------")
    print(f"   Stage 0 (Generated raw)       : {n_generated:,}")
    print(f"   Stage 1 (RDKit valid)          : {n_rdkit_valid:,}  ({attrition['pct_rdkit_valid']:.1f}%)")
    print(f"   Stage 2 (Geometric filter)     : {n_geometric:,}  ({attrition['pct_geometric']:.1f}% of valid)")
    print(f"   Stage 3 (PAINS screen)         : {n_pains:,}  ({attrition['pct_pains']:.1f}% of geometric)")
    print(f"   Stage 4 (Lipinski/Veber)       : {n_lipinski_veber:,}  ({attrition['pct_lipinski_veber']:.1f}% of PAINS-clean)")
    print(f"   Stage 5 (Strain & SA Filter)   : {n_strain_sa:,}  ({attrition['pct_strain_sa']:.1f}% of Lipinski-clean)")
    print(f"   Stage 6 (Unique final)         : {n_unique:,}  ({attrition['pct_overall_retained']:.1f}% of raw)")
    print("-----------------------------------------------------")

    # -- Summary metrics --------------------------------------------------─
    n_valid    = n_rdkit_valid
    n_filtered = n_unique
    validity   = round(n_valid / n_generated, 4)       if n_generated else 0
    uniqueness = round(n_unique / max(n_lipinski_veber, 1), 4)
    novelty    = round(df["IsNovel"].mean(), 4)    if len(df) else 0
    drug_like  = round(df["IsDrugLike"].mean(), 4) if len(df) else 0
    avg_qed    = round(df["QED"].mean(), 4)         if len(df) else 0
    avg_tox    = round(df["Toxicity"].mean(), 4)    if len(df) else 0
    avg_final  = round(df["FinalScore"].mean(), 4)  if len(df) else 0
    pains_ok   = round(df["PassesPAINS"].mean(), 4) if len(df) else 0
    int_div    = internal_diversity(valid_mols)
    scaf_div   = scaffold_diversity(valid_mols)
    scaf_ent   = compute_scaffold_shannon_entropy(valid_mols)

    summary = {
        "label"            : label,
        "n_generated"      : n_generated,
        "n_valid"          : n_valid,
        "validity_%"       : f"{validity*100:.1f}",
        "uniqueness_%"     : f"{uniqueness*100:.1f}",
        "novelty_%"        : f"{novelty*100:.1f}",
        "drug_like_%"      : f"{drug_like*100:.1f}",
        "pains_clean_%"    : f"{pains_ok*100:.1f}",
        "avg_QED"          : avg_qed,
        "avg_toxicity"     : avg_tox,
        "avg_final_score"  : avg_final,
        "internal_diversity": int_div,
        "scaffold_diversity": scaf_div,
        "scaffold_entropy"  : scaf_ent,
    }

    print("\n-- Evaluation Summary ------------------------------")
    for k, v in summary.items():
        print(f"   {k:<25}: {v}")
    print("----------------------------------------------------")

    if save_csv and len(df) > 0:
        out_path = os.path.join(CFG.RESULTS_DIR, f"molecules_{label}.csv")
        df.to_csv(out_path, index=False)
        print(f"[Evaluator] Per-molecule results -> {out_path}")

    # Save summary + attrition
    summ_path = os.path.join(CFG.RESULTS_DIR, f"summary_{label}.json")
    with open(summ_path, "w") as f:
        json.dump({"summary": summary, "attrition": attrition}, f, indent=2)

    attr_path = os.path.join(CFG.RESULTS_DIR, f"attrition_{label}.json")
    with open(attr_path, "w") as f:
        json.dump(attrition, f, indent=2)

    return {"summary": summary, "df": df, "attrition": attrition}


# ----------------------------------------------------------------------------─
# MOSES-style benchmark table
# ----------------------------------------------------------------------------─

def build_benchmark_table(results: list) -> pd.DataFrame:
    """
    Combine multiple evaluation summaries into a single comparison table
    (e.g. model at different temperatures vs. random baseline).
    """
    rows = [r["summary"] for r in results]
    df = pd.DataFrame(rows).set_index("label")
    return df


# ----------------------------------------------------------------------------─
# Multi-seed aggregation utility
# ----------------------------------------------------------------------------─

def aggregate_seed_results(seed_results: list) -> dict:
    """
    Given a list of summary dicts from multiple seeds, compute mean ± std
    for each numeric metric. Returns a dict ready for paper table.

    Args:
        seed_results: list of summary dicts (one per seed)
    Returns:
        dict with keys like "novelty_%_mean", "novelty_%_std", etc.
    """
    numeric_keys = [
        "avg_QED", "avg_toxicity", "avg_final_score",
        "internal_diversity", "scaffold_diversity"
    ]
    pct_keys = [
        "uniqueness_%", "novelty_%", "drug_like_%", "pains_clean_%"
    ]

    agg = {}
    for k in numeric_keys:
        vals = [float(r[k]) for r in seed_results if k in r]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"]  = round(float(np.std(vals)), 4)

    for k in pct_keys:
        vals = [float(r[k]) for r in seed_results if k in r]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 2)
        agg[f"{k}_std"]  = round(float(np.std(vals)), 2)

    return agg

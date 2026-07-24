# 🧬 SAFE+: An Empirical Benchmark of Multi-Stage ADMET, Synthetizability, and Structural Strain Attrition in SELFIES-LSTM Molecular Generation

A research pipeline combining a **2-Layer Stacked SELFIES-LSTM** generator with **SAFE+**, a five-stage sequential pharmacological and structural filter, to generate and rigorously evaluate novel drug-like molecules.

Submitted to **ACS Omega** (transferred from *Journal of Chemical Information and Modeling*, July 2026).

---

## 📋 Table of Contents
- [Overview](#overview)
- [SAFE+ Pipeline Architecture](#safe-pipeline-architecture)
- [Dataset](#dataset)
- [Five-Stage SAFE+ Filtering](#five-stage-safe-filtering)
- [Advanced Metrics](#advanced-metrics)
- [Results](#results)
- [Figures](#figures)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)

---

## 🔬 Overview

Most generative molecular design papers stop at syntactic validity and novelty. This work shows that is not enough.

Standard 2D drug-likeness filters (Lipinski, Veber) pass molecules that are geometrically strained in 3D or synthetically inaccessible. At high sampling temperature (`T=1.0`), **73% of Lipinski-passing candidates fail Stage 5 structural screening** — a 3D force-field filter that 2D rules cannot replicate.

We introduce **SAFE+**, a five-stage sequential benchmark pipeline that:
1. Instruments every filter stage to quantify per-stage attrition
2. Reveals a novel **temperature-dependent 3D strain liability** not previously reported for SELFIES-LSTM systems
3. Provides information-theoretic diversity metrics (5D Wasserstein distance, scaffold Shannon entropy)
4. Delivers proof-of-concept target docking with residue-level SAR

---

## 🏗️ SAFE+ Pipeline Architecture

```
Input: ZINC250K Dataset (249,455 SMILES strings)
         │
         ▼
┌─────────────────────────────────┐
│  Data Preprocessing             │  → SMILES → SELFIES conversion
│  & Vocabulary Build             │  → Zero-pad to fixed max length
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2-Layer Stacked LSTM           │  → 2× 512 units, 20% Dropout
│  Training (30 epochs)           │  → Best val loss: 1.156, Token acc: ~64%
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Temperature-Scaled Sampling    │  → T ∈ {0.2, 0.5, 0.7, 1.0}
│  SELFIES → SMILES decoding      │  → 100% bond-topological validity
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  SAFE+ Five-Stage Sequential Filter                         │
│                                                             │
│  Stage 1: Geometric hard filter  (MW ∈ [120,600] Da)        │
│  Stage 2: PAINS screen           (Baell & Holloway)         │
│  Stage 3: Lipinski / Veber 2D    (logP, HBD, HBA, TPSA)    │
│  Stage 4: SA Score + SMARTS      (SA ≤ 4.5, no cumulenes)  │
│  Stage 5: 3D MMFF94 strain       (ΔE_strain ≤ 35 kcal/mol) │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Advanced Metric Computation    │  → WD_5D, H_scaffold, MMFF94 strain
│  + Target Docking (EGFR 1M17)   │  → AutoDock Vina / PyRx
└─────────────────────────────────┘
```

---

## 📊 Dataset

- **Name:** ZINC250K (Clean subset)
- **Source:** [Kaggle — ZINC250K](https://www.kaggle.com/datasets/basu369victor/zinc250k)
- **File:** `data/250k_rndm_zinc_drugs_clean_3.csv`
- **Size:** ~249,455 valid SMILES strings
- **Split:** 90% training / 10% held-out validation

---

## 🧪 Five-Stage SAFE+ Filtering

| Stage | Filter | Criterion |
|:-----:|:-------|:----------|
| 1 | Geometric hard filter | MW ∈ [120, 600] Da; ≥10 heavy atoms; rotatable bonds ≤ 10 |
| 2 | PAINS screen | RDKit PAINS (Baell & Holloway) — discard on any match |
| 3 | Lipinski / Veber 2D | logP ≤ 5, HBD ≤ 5, HBA ≤ 10, TPSA ≤ 140 Å² |
| 4 | SA Score + SMARTS strain | Ertl SA Score ≤ 4.5; exclude cumulenes (C=C=C), allenes, strained ring alkynes (<8 atoms) |
| 5 | 3D MMFF94 force field | Embed 3D conformer → MMFF94 minimize → purge if ΔE_strain > 35.0 kcal/mol |

### Attrition at T=0.5 (999 raw → 199 unique)

| Stage | Count | Pass Rate |
|:------|------:|----------:|
| Raw generated | 999 | — |
| Stage 1: RDKit valid | 999 | 100.0% |
| Stage 2: Geometric filter | 417 | 41.7% |
| Stage 3: PAINS screen | 402 | 96.4% of geometric |
| Stage 4: Lipinski/Veber 2D | 369 | 91.8% of PAINS-clean |
| Stage 5: Strain & SA filter | 205 | 55.6% of Lipinski-clean |
| **Unique final** | **199** | **19.9% of raw** |

---

## 📐 Advanced Metrics

Three information-theoretic and physical metrics beyond standard validity/novelty:

### 1. 3D MMFF94 Conformational Strain Energy
Measures physical 3D geometric stability of each generated candidate after force-field minimization:

> ΔE_strain = E(x_initial) − E(x_min)

Candidates with ΔE_strain > 35.0 kcal/mol are purged.

### 2. 5D Joint Property Manifold Wasserstein Distance (WD_5D)
Measures divergence between generated and ZINC250K reference distributions across five descriptors (MW, LogP, TPSA, QED, SA Score) using the Wasserstein-1 (Earth Mover's) distance.

### 3. Bemis-Murcko Scaffold Shannon Entropy (H_scaffold)
Formal information-theoretic measurement of scaffold-level structural diversity. Higher bits = more uniform scaffold exploration.

---

## 📈 Results

### Core Metrics at T=0.5 (SAFE+ retained set)

| Metric | Value |
|:-------|------:|
| Unique candidates retained | 199 |
| Novelty | 100.0% |
| Mean SA Score | 3.38 |
| Mean QED | 0.559 |
| Mean MMFF94 Strain | 50.43 kcal/mol |
| 5D Wasserstein (WD_5D) | 0.1770 |
| Scaffold Shannon Entropy | 6.31 bits |

### Advanced Metrics Across Temperatures

| Temperature | WD_5D | Mean SA | MMFF94 Strain (kcal/mol) | H_scaffold |
|:-----------:|:-----:|:-------:|:------------------------:|:----------:|
| T = 0.2 | 0.1933 | 3.42 | 52.54 | 6.08 bits |
| **T = 0.5** *(Pareto-optimal)* | **0.1770** | **3.38** | **50.43** | 6.31 bits |
| T = 0.7 | 0.1553 | 3.46 | 66.94 | **6.54 bits** |
| T = 1.0 | 0.2148 | 3.60 | 108.24 | 5.56 bits |

**Key finding:** Mean MMFF94 strain among SAFE+-passing candidates rises **2.1×** from T=0.5 to T=1.0 — a temperature-dependent 3D strain liability not previously reported for SELFIES-LSTM systems.

### Comparison with CharRNN Baseline (MOSES)

| Config | Valid | Unique | Novelty | Avg QED | Int. Div. |
|:-------|:-----:|:------:|:-------:|:-------:|:---------:|
| T=0.5 (SELFIES†) | **100.0%** | 98.4% | 98.7% | 0.559 | 0.896 |
| MOSES CharRNN | 97.3% | 99.9% | 84.2% | 0.600 | 0.856 |

†SELFIES guarantees 100% bond-topological validity by construction.

### Top Docking Result
- **Molecule:** *N*-(2-(2-(2-fluorophenyl)cyclopropyl)ethyl)acetamide
- **SA Score:** 2.87 | **QED:** 0.831 | **Toxicity:** 0.00 | **FinalScore:** 0.915
- **Target:** EGFR kinase domain (PDB: 1M17)
- **Binding affinity:** −7.6 kcal/mol with explicit Met793 hinge H-bonding

---

## 🖼️ Figures

All figures are generated automatically in `figures/`:

| File | Description |
|:-----|:------------|
| `01_training_loss_curve.png` | Training & validation loss + token accuracy over 30 epochs |
| `03_temperature_sweep.png` | Six metrics across four sampling temperatures |
| `04_property_distributions.png` | MW, LogP, TPSA, QED, SA Score distributions (T=0.5 retained set) |
| `05_toxicity_breakdown.png` | Composite ADMET toxicity score breakdown |
| `06_top12_molecules.png` | Top-12 candidates ranked by composite score |
| `08_correlation_heatmap.png` | Pearson correlation matrix (9 properties) |
| `09_novelty_tanimoto_hist.png` | Tanimoto similarity to nearest ZINC250K training neighbour |
| `10_docking_pose.png` | Best docking pose of top candidate into EGFR 1M17 |
| `11_safe_attrition.png` | Per-stage SAFE+ attrition funnel visualisation |

---

## 📁 File Structure

```
├── data/
│   └── 250k_rndm_zinc_drugs_clean_3.csv    # ZINC250K training dataset
├── figures/                                # Publication-ready charts
├── models/                                 # Saved PyTorch LSTM checkpoints (.pt)
├── results/                                # CSV outputs (molecules_T*_SAFEplus.csv)
├── legacy_results/                         # Previous notebook outputs
├── src/                                    # Source code pipeline
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── paper.tex                               # Main manuscript (IEEEtran format)
├── cover_letter.tex                        # Submission cover letter
├── run.py                                  # Root execution file
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Full Pipeline
Runs all stages: preprocessing → training → generation → SAFE+ filtering → advanced metrics → figures.
```bash
python run.py
```

### 3. Quick Test Run
5K molecules, 5 epochs — for verifying your environment.
```bash
python run.py --quick
```

---

## 📄 Paper

**Title:** SAFE+: An Empirical Benchmark of Multi-Stage ADMET, Synthetizability, and Structural Strain Attrition in SELFIES-LSTM Molecular Generation

**Authors:** Kavya Garg, Thanyaa S, Harshitha M, Madderla Chiranjeevi

**Affiliation:** School of Computer Science Engineering, RV University, Bangalore, India

---

**Corresponding Author:** Kavya Garg — kavyag.btech1@rvu.edu.in

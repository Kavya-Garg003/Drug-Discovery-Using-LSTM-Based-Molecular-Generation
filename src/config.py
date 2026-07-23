"""
config.py — Central hyperparameter & path configuration.
All constants live here so any script just does: from config import CFG
"""
import os

class CFG:
    # -- Paths --------------------------------------------------------------
    # src/config.py lives in src/, so go up one level for project root
    BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH      = os.path.join(BASE_DIR, "data",    "250k_rndm_zinc_drugs_clean_3.csv")
    RESULTS_DIR    = os.path.join(BASE_DIR, "results")
    MODEL_DIR      = os.path.join(BASE_DIR, "models")
    FIGURES_DIR    = os.path.join(BASE_DIR, "figures")

    # -- Data --------------------------------------------------------------─
    SMILES_COL     = "smiles"
    MAX_MOLECULES  = 50_000      # subset for faster training (use None for all 250K)
    TRAIN_SPLIT    = 0.90        # 90% train, 10% held-out for novelty check

    # -- SELFIES tokenisation ----------------------------------------------─
    MAX_LEN        = 80          # max token length (SELFIES tokens, not characters)
    PAD_TOKEN      = "[nop]"     # SELFIES no-operation (padding token)

    # -- Model --------------------------------------------------------------
    EMBED_DIM      = 128
    LSTM_UNITS_1   = 256
    LSTM_UNITS_2   = 256
    DROPOUT        = 0.2
    RECURRENT_DROP = 0.1

    # -- Training ----------------------------------------------------------─
    EPOCHS         = 30
    BATCH_SIZE     = 256
    LEARNING_RATE  = 1e-3
    LR_PATIENCE    = 5           # ReduceLROnPlateau patience
    EARLY_STOP     = 8           # EarlyStopping patience

    # -- Generation --------------------------------------------------------─
    N_GENERATE     = 1_000       # total candidates to generate
    TEMPERATURES   = [0.2, 0.5, 0.7, 1.0]   # sweep for ablation
    DEFAULT_TEMP   = 0.7

    # -- Molecule filters --------------------------------------------------─
    MIN_MW         = 120.0       # exclude trivial fragments (was missing before!)
    MAX_MW         = 600.0       # exclude very large molecules
    MIN_HEAVY_ATOMS= 10          # exclude trivially small SMILES
    MAX_LOGP       = 5.5         # slightly relaxed Lipinski
    MAX_HBD        = 5
    MAX_HBA        = 10
    MAX_TPSA       = 140.0       # Veber oral bioavailability criterion
    MAX_ROT_BONDS  = 10          # Veber criterion

    # -- SAFE+ Stage 4 & 5: Structural Strain & Synthetic Accessibility ----
    MAX_SA_SCORE       = 4.5     # Ertl Synthetic Accessibility Score (<= 4.5 is drug-like)
    ALLOW_CUMULENES    = False   # Filter out C=C=C / X=C=Y cumulene artifacts
    ALLOW_STRAINED_RING= False   # Filter out alkynes in rings < 8 atoms & bridgehead violations
    MAX_MMFF_STRAIN    = 35.0    # Max MMFF94 conformational strain energy (kcal/mol)

    # -- Toxicity (ADMET proxy) --------------------------------------------─
    # Score = weighted combo of TPSA, logP, MW, PAINS; lower = safer
    TOX_TPSA_WEIGHT  = 0.30
    TOX_LOGP_WEIGHT  = 0.25
    TOX_MW_WEIGHT    = 0.20
    TOX_PAINS_WEIGHT = 0.25

    # -- Reproducibility ----------------------------------------------------
    SEED           = 42


# Create output directories on import
for _d in [CFG.RESULTS_DIR, CFG.MODEL_DIR, CFG.FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

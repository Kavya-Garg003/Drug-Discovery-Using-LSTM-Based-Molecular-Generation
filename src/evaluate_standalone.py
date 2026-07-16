"""
evaluate_standalone.py

Evaluates the existing generated_T*.txt files to get the exact metrics for Table 1,
bypassing PyTorch and the broken vocabulary.
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def calc_metrics(smiles_list, train_smiles):
    valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [m for m in valid_mols if m is not None]
    
    unique_smiles = set(Chem.MolToSmiles(m) for m in valid_mols)
    
    train_set = set(train_smiles)
    novel_smiles = unique_smiles - train_set
    
    # Drug-like (Lipinski)
    drug_like = 0
    qeds = []
    scaffolds = set()
    for s in unique_smiles:
        m = Chem.MolFromSmiles(s)
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        hbd = Descriptors.NumHDonors(m)
        hba = Descriptors.NumHAcceptors(m)
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            drug_like += 1
        qeds.append(QED.qed(m))
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        if scaf:
            scaffolds.add(scaf)
            
    uniqueness = len(unique_smiles) / len(smiles_list) * 100
    novelty = len(novel_smiles) / len(unique_smiles) * 100 if unique_smiles else 0
    drug_like_pct = drug_like / len(unique_smiles) * 100 if unique_smiles else 0
    avg_qed = sum(qeds) / len(qeds) if qeds else 0
    scaf_div = len(scaffolds) / len(unique_smiles) if unique_smiles else 0
    
    return {
        "Unique": uniqueness,
        "Novelty": novelty,
        "Drug-like": drug_like_pct,
        "Avg QED": avg_qed,
        "Scaf Div": scaf_div
    }

if __name__ == "__main__":
    train_smiles = []
    # Load original training smiles for novelty calculation
    data_path = os.path.join(BASE_DIR, "data", "zinc250k.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        train_smiles = df['smiles'].tolist()
        
    for T in ["0.2", "0.5", "0.7", "1.0"]:
        file_path = os.path.join(RESULTS_DIR, f"generated_T{T}.txt")
        if os.path.exists(file_path):
            with open(file_path) as f:
                smiles = [l.strip() for l in f if l.strip()]
            metrics = calc_metrics(smiles, train_smiles)
            print(f"T={T}: {metrics}")

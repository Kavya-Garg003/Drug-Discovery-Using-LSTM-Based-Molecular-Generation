"""
generate.py — Generate novel SELFIES -> SMILES with temperature sweep (PyTorch).

Improvements vs original notebook:
  - SELFIES decoding -> ~100% syntactically valid output
  - No artificial stop conditions that broke ring closures
  - Temperature sweep (generates at multiple temps for ablation study)
  - Nucleus (top-p) sampling option alongside temperature sampling
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem
from tqdm import trange
from config import CFG
from model import load_model


# ----------------------------------------------------------------------------─
# Sampling
# ----------------------------------------------------------------------------─

def _sample_token(logits: np.ndarray, temperature: float, top_p: float = 1.0) -> int:
    """
    Sample next token index from a probability distribution.

    Args:
        logits      : raw softmax probabilities from model (shape [vocab_size])
        temperature : > 1 = more random, < 1 = more greedy
        top_p       : nucleus sampling — keep only top-p probability mass
    """
    logits = logits.astype(np.float64)
    logits = logits / max(temperature, 1e-5)
    probs  = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    # Nucleus sampling
    if top_p < 1.0:
        sorted_idx  = np.argsort(-probs)
        cum_probs   = np.cumsum(probs[sorted_idx])
        cutoff      = sorted_idx[cum_probs > top_p][0] if any(cum_probs > top_p) else sorted_idx[-1]
        mask        = np.zeros_like(probs)
        mask[sorted_idx[:np.where(sorted_idx == cutoff)[0][0] + 1]] = 1
        probs      *= mask
        probs      /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------─
# SELFIES generation
# ----------------------------------------------------------------------------─

def generate_selfies(
    model,
    token2idx: dict,
    idx2token: dict,
    pad_idx: int,
    n_molecules: int    = CFG.N_GENERATE,
    max_len: int        = CFG.MAX_LEN,
    temperature: float  = CFG.DEFAULT_TEMP,
    top_p: float        = 1.0,
    seed_tokens: list   = None,
) -> list[str]:
    """
    Generate `n_molecules` SELFIES strings using the trained LSTM (PyTorch).
    Each generated SELFIES is guaranteed to decode to a valid SMILES.

    Args:
        seed_tokens: list of starting token strings. If None, uses a random
                     token from the vocabulary (excludes PAD).
    """
    device   = next(model.parameters()).device
    non_pad  = [t for t in token2idx if t != CFG.PAD_TOKEN]
    generated = []
    model.eval()

    with torch.no_grad():
        for _ in trange(n_molecules, desc=f"Generating (T={temperature})"):
            seed = np.random.choice(seed_tokens if seed_tokens else non_pad)
            tokens = [seed]
            ids    = [token2idx.get(seed, pad_idx)]

            hidden1, hidden2 = None, None
            for _step in range(max_len - 1):
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits, hidden1, hidden2 = model(x, hidden1, hidden2)
                probs = logits[0, -1]                    # (vocab_size,)
                next_idx = _sample_token(
                    probs.cpu().numpy(), temperature, top_p
                )
                next_tok = idx2token[next_idx]
                if next_tok == CFG.PAD_TOKEN:
                    break
                tokens.append(next_tok)
                ids.append(next_idx)

            generated.append("".join(tokens))

    return generated


def selfies_to_smiles_list(selfies_list: list[str]) -> list[str]:
    """
    Decode SELFIES -> SMILES and verify via RDKit.
    With SELFIES, virtually all strings decode successfully (validity fix).
    """
    smiles_out = []
    for sel in selfies_list:
        try:
            smi = sf.decoder(sel)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                smiles_out.append(Chem.MolToSmiles(mol))  # canonical SMILES
        except Exception:
            pass
    return smiles_out


# ----------------------------------------------------------------------------─
# Temperature sweep (for ablation / paper Table)
# ----------------------------------------------------------------------------─

def generate_sweep(
    model,
    token2idx: dict,
    idx2token: dict,
    pad_idx: int,
    temperatures: list  = CFG.TEMPERATURES,
    n_per_temp: int     = CFG.N_GENERATE,
    seed: int           = CFG.SEED,
) -> dict:
    """
    Generate SMILES at each temperature in `temperatures`.
    Returns dict: temp -> list of canonical SMILES.

    Args:
        seed: numpy random seed for this generation run.
              The model weights are fixed; only sampling stochasticity varies.
    """
    np.random.seed(seed)
    results = {}
    for temp in temperatures:
        sels   = generate_selfies(
            model, token2idx, idx2token, pad_idx,
            n_molecules = n_per_temp,
            temperature = temp,
        )
        smiles = selfies_to_smiles_list(sels)
        results[temp] = smiles
        print(f"  T={temp} (seed={seed}): {len(sels)} SELFIES -> {len(smiles)} SMILES decoded")

        out = os.path.join(CFG.RESULTS_DIR, f"generated_T{temp}.txt")
        with open(out, "w") as f:
            f.write("\n".join(smiles))
    return results


# ----------------------------------------------------------------------------─
# Random baseline generator (for comparison)
# ----------------------------------------------------------------------------─

def random_baseline(
    train_smiles: list[str],
    n: int = CFG.N_GENERATE,
    seed: int = CFG.SEED,
) -> list[str]:
    """
    Naive baseline: sample SMILES randomly from the training set.
    Used to show the LSTM model outperforms pure memorisation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_smiles), size=n, replace=True)
    return [train_smiles[i] for i in idx]

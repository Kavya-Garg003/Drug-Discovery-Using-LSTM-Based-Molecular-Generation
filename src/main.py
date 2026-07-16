"""
main.py — Full pipeline orchestrator with multi-seed support.

Run this single file to reproduce all results:
    python main.py                  # full run, 3 seeds
    python main.py --skip-train     # skip training, load existing weights
    python main.py --n-seeds 1      # single run (original behaviour)
    python main.py --quick          # fast test with reduced data

Stages:
  1. Data loading & SELFIES preprocessing
  2. Model training (or load if already trained)
  3. Temperature sweep generation (multi-seed)
  4. Random baseline generation
  5. Full evaluation with SAFE attrition tracking
  6. All publication figures
  7. Summary CSV, benchmark table, and seed-aggregated stats
"""

import os, json, argparse
import numpy as np
import pandas as pd
from config import CFG


def parse_args():
    p = argparse.ArgumentParser(description="FGAI Drug Molecule Generation Pipeline")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training and load existing weights")
    p.add_argument("--quick",       action="store_true",
                   help="Use reduced dataset (5K molecules, 5 epochs) for quick test")
    p.add_argument("--n-seeds",     type=int, default=3,
                   help="Number of random seeds for generation (default: 3 for statistical rigour)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        CFG.MAX_MOLECULES = 5_000
        CFG.EPOCHS        = 5
        CFG.N_GENERATE    = 200
        CFG.TEMPERATURES  = [0.5, 0.7]
        print("[Main] QUICK MODE: reduced dataset + epochs for testing.")

    # Generation seeds (different from training seed — model weights are fixed)
    gen_seeds = [42, 123, 456, 789, 999][:args.n_seeds]
    print(f"[Main] Multi-seed generation: {gen_seeds}")

    # -- 1. Load & preprocess data ----------------------------------------─
    print("\n" + "="*60)
    print("STAGE 1: Data Loading & SELFIES Preprocessing")
    print("="*60)
    from data_loader import load_and_prepare
    data = load_and_prepare()
    vocab     = data["vocab"]
    token2idx = data["token2idx"]
    idx2token = data["idx2token"]
    pad_idx   = data["pad_idx"]

    # -- 2. Build / train model --------------------------------------------
    print("\n" + "="*60)
    print("STAGE 2: Model Training")
    print("="*60)
    from model import build_model, load_model

    weights_path = os.path.join(CFG.MODEL_DIR, "selfies_lstm_best.pt")

    if args.skip_train and os.path.exists(weights_path):
        print(f"[Main] Loading existing weights from {weights_path}")
        model = load_model(len(vocab), weights_path)
        history_csv = os.path.join(CFG.RESULTS_DIR, "training_history.csv")
        if not os.path.exists(history_csv):
            pd.DataFrame({"loss":[0], "val_loss":[0]}).to_csv(history_csv, index=False)
    else:
        from train import train
        history = train(data)
        model   = load_model(len(vocab), weights_path)
        history_csv = os.path.join(CFG.RESULTS_DIR, "training_history.csv")

    # -- 3. Multi-seed temperature sweep generation -----------------------─
    print("\n" + "="*60)
    print(f"STAGE 3: Multi-Seed Molecule Generation ({len(gen_seeds)} seeds × {len(CFG.TEMPERATURES)} temps)")
    print("="*60)
    from generate import generate_sweep, selfies_to_smiles_list, random_baseline

    # seed_sweep_smiles[seed][temp] = list of SMILES
    seed_sweep_smiles = {}
    for seed in gen_seeds:
        np.random.seed(seed)
        print(f"\n  >> Running generation with seed={seed}")
        sweep = generate_sweep(
            model, token2idx, idx2token, pad_idx,
            temperatures = CFG.TEMPERATURES,
            n_per_temp   = CFG.N_GENERATE,
            seed         = seed,
        )
        seed_sweep_smiles[seed] = sweep

    # -- 4. Random baseline ------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 4: Random Baseline Generation")
    print("="*60)
    baseline_smiles = random_baseline(data["train_smiles"], n=CFG.N_GENERATE)
    baseline_path   = os.path.join(CFG.RESULTS_DIR, "generated_baseline.txt")
    with open(baseline_path, "w") as f:
        f.write("\n".join(baseline_smiles))

    # -- 5. Evaluate (multi-seed + aggregate) ----------------------------─
    print("\n" + "="*60)
    print("STAGE 5: Multi-Seed Evaluation with SAFE Attrition")
    print("="*60)
    from evaluate import evaluate, build_benchmark_table, aggregate_seed_results

    # Aggregate results across seeds per temperature
    # Structure: temp -> list of per-seed summaries
    temp_seed_summaries   = {str(t): [] for t in CFG.TEMPERATURES}
    temp_seed_attritions  = {str(t): [] for t in CFG.TEMPERATURES}
    all_results           = []

    for seed in gen_seeds:
        for temp in CFG.TEMPERATURES:
            smiles = seed_sweep_smiles[seed][temp]
            label  = f"T{temp}_seed{seed}"
            res    = evaluate(
                smiles_list  = smiles,
                train_smiles = data["train_smiles"],
                label        = label,
                save_csv     = (seed == gen_seeds[0]),  # save CSV only for first seed
            )
            temp_seed_summaries[str(temp)].append(res["summary"])
            temp_seed_attritions[str(temp)].append(res["attrition"])
            all_results.append(res)

    # Compute mean ± std across seeds for each temperature
    aggregated = {}
    for temp in CFG.TEMPERATURES:
        agg = aggregate_seed_results(temp_seed_summaries[str(temp)])
        agg["label"] = f"T{temp}"
        agg["n_seeds"] = len(gen_seeds)
        aggregated[str(temp)] = agg

    agg_path = os.path.join(CFG.RESULTS_DIR, "aggregated_seed_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n[Main] Aggregated seed results -> {agg_path}")

    # Attrition averaged across seeds (for the funnel figure)
    avg_attrition = {}
    for temp in CFG.TEMPERATURES:
        attrs = temp_seed_attritions[str(temp)]
        avg = {}
        for key in attrs[0]:
            vals = [a[key] for a in attrs]
            avg[key] = round(float(np.mean(vals)), 1)
        avg_attrition[str(temp)] = avg

    attr_agg_path = os.path.join(CFG.RESULTS_DIR, "attrition_aggregated.json")
    with open(attr_agg_path, "w") as f:
        json.dump(avg_attrition, f, indent=2)

    # Best temperature (highest avg_QED across seeds)
    best_temp = max(
        CFG.TEMPERATURES,
        key=lambda t: float(aggregated[str(t)]["avg_QED_mean"])
    )
    best_label = f"T{best_temp}"
    best_df    = pd.read_csv(os.path.join(CFG.RESULTS_DIR, f"molecules_{best_label}_seed{gen_seeds[0]}.csv"))
    best_summary = temp_seed_summaries[str(best_temp)][0]

    # Baseline evaluation
    base_res = evaluate(
        smiles_list  = baseline_smiles,
        train_smiles = data["train_smiles"],
        label        = "baseline",
        save_csv     = True,
    )
    base_summary = base_res["summary"]
    base_df      = base_res["df"]

    # -- Build benchmark table (first-seed results for comparison) --------
    first_seed_results = []
    for temp in CFG.TEMPERATURES:
        s = temp_seed_summaries[str(temp)][0].copy()
        s["label"] = f"T{temp}"
        first_seed_results.append({"summary": s, "df": pd.DataFrame()})

    benchmark_df = build_benchmark_table(first_seed_results + [base_res])
    bench_path   = os.path.join(CFG.RESULTS_DIR, "benchmark_table.csv")
    benchmark_df.to_csv(bench_path)
    print(f"\n[Main] Benchmark table -> {bench_path}")
    print(benchmark_df.to_string())

    # -- 6. All figures ----------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 6: Figure Generation")
    print("="*60)
    from visualize import generate_all_figures
    generate_all_figures(
        history_csv        = history_csv,
        df_model           = best_df,
        df_baseline        = base_df,
        summaries_by_temp  = [temp_seed_summaries[str(t)][0] for t in CFG.TEMPERATURES],
        model_summary      = best_summary,
        baseline_summary   = base_summary,
        train_smiles       = data["train_smiles"],
    )

    # Generate attrition figure
    from visualize import generate_attrition_figure
    generate_attrition_figure(avg_attrition)

    # -- 7. Final summary --------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPLETE ✓")
    print("="*60)
    print(f"  Results dir : {CFG.RESULTS_DIR}")
    print(f"  Figures dir : {CFG.FIGURES_DIR}")
    print(f"  Model dir   : {CFG.MODEL_DIR}")
    print(f"  Seeds used  : {gen_seeds}")
    print(f"\n  Best config : {best_label}")
    print(f"\n  Aggregated metrics (mean ± std across {len(gen_seeds)} seeds):")
    agg_best = aggregated[str(best_temp)]
    for k in ["novelty_%", "uniqueness_%", "drug_like_%", "pains_clean_%",
              "avg_QED", "avg_toxicity", "internal_diversity", "scaffold_diversity"]:
        mean_k = f"{k}_mean"
        std_k  = f"{k}_std"
        if mean_k in agg_best:
            print(f"    {k:<25}: {agg_best[mean_k]} ± {agg_best[std_k]}")


if __name__ == "__main__":
    main()

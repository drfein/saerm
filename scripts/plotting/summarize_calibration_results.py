#!/usr/bin/env python3
"""
Summarize calibration alpha sweep results in a table.
"""

import json
from pathlib import Path
import pandas as pd

MODEL_NAMES = {
    "skywork": "Skywork Llama 3.1-8B",
    "allen": "Allen AI Llama 3.1-8B",
    "skywork_qwen3": "Skywork Qwen 3-8B",
    "skywork_qwen-smallest": "Skywork Qwen 3-0.6B",
    "deberta": "DeBERTa v3 Large",
}

ALPHA_VALUES = [0.1, 0.5, 0.75, 1.0, 1.5]


def load_and_summarize(results_dir: Path):
    """Load results and create summary tables."""
    
    # Table 1: Overconfidence bias by alpha
    print("=" * 100)
    print("CALIBRATION RESULTS: Overconfidence Bias (%) - Lower is Better")
    print("=" * 100)
    print()
    
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        result_file = results_dir / f"calibration_{model_key}_math500_alpha_sweep.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        row = {"Model": model_name}
        
        # Baseline
        baseline = data["results"]["baseline"]["metrics"]["overconfidence_bias_pct"] * 100
        row["Baseline"] = f"{baseline:.1f}"
        
        # Each alpha
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = data["results"][alpha_key]["metrics"]["overconfidence_bias_pct"] * 100
            improvement = baseline - val
            row[f"α={alpha}"] = f"{val:.1f} ({improvement:+.1f})"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    
    # Table 2: Correct > Incorrect at Conf:10
    print("=" * 100)
    print("CALIBRATION RESULTS: Correct > Incorrect at Confidence:10 (%) - Higher is Better")
    print("=" * 100)
    print()
    
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        result_file = results_dir / f"calibration_{model_key}_math500_alpha_sweep.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        row = {"Model": model_name}
        
        # Baseline
        baseline = data["results"]["baseline"]["metrics"]["correct_over_incorrect_conf10_pct"] * 100
        row["Baseline"] = f"{baseline:.1f}"
        
        # Each alpha
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = data["results"][alpha_key]["metrics"]["correct_over_incorrect_conf10_pct"] * 100
            improvement = val - baseline
            row[f"α={alpha}"] = f"{val:.1f} ({improvement:+.1f})"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    
    # Table 3: Modest Correct > Overconfident Incorrect
    print("=" * 100)
    print("CALIBRATION RESULTS: Correct(Conf:5) > Incorrect(Conf:10) (%) - Higher is Better")
    print("=" * 100)
    print()
    
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        result_file = results_dir / f"calibration_{model_key}_math500_alpha_sweep.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        row = {"Model": model_name}
        
        # Baseline
        baseline = data["results"]["baseline"]["metrics"]["modest_correct_over_overconfident_incorrect_pct"] * 100
        row["Baseline"] = f"{baseline:.1f}"
        
        # Each alpha
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = data["results"][alpha_key]["metrics"]["modest_correct_over_overconfident_incorrect_pct"] * 100
            improvement = val - baseline
            row[f"α={alpha}"] = f"{val:.1f} ({improvement:+.1f})"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    
    # Table 4: Probe quality
    print("=" * 100)
    print("PROBE QUALITY METRICS")
    print("=" * 100)
    print()
    
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        result_file = results_dir / f"calibration_{model_key}_math500_alpha_sweep.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        probe_meta = data["probe_metadata"]
        
        row = {
            "Model": model_name,
            "Probe Accuracy (%)": f"{probe_meta['probe_accuracy'] * 100:.1f}",
            "Separation": f"{probe_meta['separation']:.2f}",
            "Hidden Dim": probe_meta["hidden_dim"],
            "N Pairs": probe_meta["n_positive"],
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    
    # Summary: Best alpha per model
    print("=" * 100)
    print("OPTIMAL ALPHA PER MODEL (Based on Overconfidence Reduction)")
    print("=" * 100)
    print()
    
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        result_file = results_dir / f"calibration_{model_key}_math500_alpha_sweep.json"
        if not result_file.exists():
            continue
        
        with open(result_file) as f:
            data = json.load(f)
        
        baseline = data["results"]["baseline"]["metrics"]["overconfidence_bias_pct"] * 100
        
        best_alpha = None
        best_val = float('inf')
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = data["results"][alpha_key]["metrics"]["overconfidence_bias_pct"] * 100
            if val < best_val:
                best_val = val
                best_alpha = alpha
        
        improvement = baseline - best_val
        
        row = {
            "Model": model_name,
            "Best Alpha": best_alpha,
            "Baseline Overconf (%)": f"{baseline:.1f}",
            "Best Overconf (%)": f"{best_val:.1f}",
            "Improvement (pp)": f"{improvement:.1f}",
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()


def main():
    results_dir = Path("artifacts/results/calibration_alpha_sweep")
    load_and_summarize(results_dir)


if __name__ == "__main__":
    main()

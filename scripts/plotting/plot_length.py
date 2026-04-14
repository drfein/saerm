#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from utils import (
    load_results, parse_experiment_name, COLORS, MODEL_NAMES, DATASET_NAMES, EXCLUDED_MODELS,
    binomial_ci_2sigma, sem_2sigma, save_plot, plot_heatmap
)

def plot_length_aggregate(results: Dict[str, Dict], output_dir: Path):
    """Create per-model length bias plots showing traditional vs inverse bias patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect per-model data
    model_data = {}
    for name, data in results.items():
        model, dataset, _ = parse_experiment_name(name, "length")
        if model in EXCLUDED_MODELS:
            continue
        
        if model not in model_data:
            model_data[model] = {
                "concise_base": [], "concise_null": [],
                "verbose_base": [], "verbose_null": [],
                "n_concise": [], "n_verbose": []
            }
        
        # Concise: P(incorrect > correct_concise)
        model_data[model]["concise_base"].append(data["baseline"].get("incorrect_beats_correct_pct", 0) * 100)
        model_data[model]["concise_null"].append(data["nulled"].get("incorrect_beats_correct_pct", 0) * 100)
        model_data[model]["n_concise"].append(data["baseline"].get("n_examples", 1000))
        
        # Verbose: P(incorrect > correct_verbose)  
        model_data[model]["verbose_base"].append(data["baseline"].get("incorrect_beats_correct_verbose_pct", 0) * 100)
        model_data[model]["verbose_null"].append(data["nulled"].get("incorrect_beats_correct_verbose_pct", 0) * 100)
        model_data[model]["n_verbose"].append(data["baseline"].get("n_verbose_examples", 1000))
    
    if not model_data:
        return
    
    # Create one subplot per model
    models = sorted(model_data.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), squeeze=False)
    axes = axes[0]
    
    # Colors
    concise_color = "#6C757D"
    verbose_color = "#E85D75"
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]
        
        # Average across datasets
        concise_base_avg = 100 - np.mean(data["concise_base"])
        concise_null_avg = 100 - np.mean(data["concise_null"])
        verbose_base_avg = 100 - np.mean(data["verbose_base"])
        verbose_null_avg = 100 - np.mean(data["verbose_null"])
        
        # Error bars (stay same as they are symmetric)
        concise_base_err = sem_2sigma(data["concise_base"])
        concise_null_err = sem_2sigma(data["concise_null"])
        verbose_base_err = sem_2sigma(data["verbose_base"])
        verbose_null_err = sem_2sigma(data["verbose_null"])
        
        x = np.array([0, 1])
        width = 0.35
        
        # Baseline group
        ax.bar(x[0] - width/2, concise_base_avg, width, yerr=concise_base_err,
               label="Concise Correct", color=concise_color, edgecolor="white", 
               linewidth=1.5, capsize=4)
        ax.bar(x[0] + width/2, verbose_base_avg, width, yerr=verbose_base_err,
               label="Verbose Correct", color=verbose_color, edgecolor="white",
               linewidth=1.5, capsize=4)
        
        # Nulled group
        ax.bar(x[1] - width/2, concise_null_avg, width, yerr=concise_null_err,
               color=concise_color, edgecolor="white", linewidth=1.5, capsize=4)
        ax.bar(x[1] + width/2, verbose_null_avg, width, yerr=verbose_null_err,
               color=verbose_color, edgecolor="white", linewidth=1.5, capsize=4)
        
        ax.set_title(MODEL_NAMES.get(model, model), fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Nulled"], fontsize=12)
        ax.set_ylim(0, 105)
        
        if idx == 0:
            ax.set_ylabel("Accuracy Rate (%)", fontsize=12)
        
        for xi, (c_val, v_val) in enumerate([(concise_base_avg, verbose_base_avg), 
                                               (concise_null_avg, verbose_null_avg)]):
            ax.text(xi - width/2, c_val + 2, f"{c_val:.0f}%", ha="center", va="bottom", fontsize=9, color=concise_color)
            ax.text(xi + width/2, v_val + 2, f"{v_val:.0f}%", ha="center", va="bottom", fontsize=9, color=verbose_color)
    
    # Legend at the top row under the title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
               ncol=2, fontsize=12, frameon=False)

    plt.suptitle("Reward Model Length Bias on Grade School Math", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_plot(fig, output_dir / "length_bias_per_model.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots/aggregate/length")
    args = parser.parse_args()
    
    results = load_results(Path(args.results_dir), "length")
    if not results:
        return
        
    output_dir = Path(args.output_dir)
    plot_length_aggregate(results, output_dir)
    plot_heatmap(results, "length", "incorrect_beats_correct_pct", output_dir, "Length Bias (P(I > C))", vmin=0, vmax=0.6, cmap="RdYlGn_r")

if __name__ == "__main__":
    main()

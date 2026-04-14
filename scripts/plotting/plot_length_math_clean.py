#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from utils import (
    load_results, parse_experiment_name, MODEL_NAMES, EXCLUDED_MODELS,
    binomial_ci_2sigma, bootstrap_ci_2sigma, save_plot
)

def plot_length_math_clean(results: Dict[str, Dict], output_dir: Path):
    """Create a clean plot showing only the two extreme length bias comparisons for MATH dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for MATH dataset only
    data_map = {}
    models = sorted([m for m in MODEL_NAMES.keys() if m not in EXCLUDED_MODELS])
    
    for name, data in results.items():
        model, dataset, _ = parse_experiment_name(name, "length")
        if model in EXCLUDED_MODELS:
            continue
        
        # Only process MATH dataset
        if "math" not in dataset.lower() or "gsm8k_to_math" in name:
            continue
        
        if model not in data_map:
            data_map[model] = {}
        
        n = data.get("n_eval_examples", data["baseline"].get("n_examples", 100))
        baseline_per_ex = data.get("baseline_per_example", {})
        nulled_per_ex = data.get("nulled_per_example", {})
        
        def get_acc_stats(metric_key, per_ex_key):
            base_pct = data["baseline"].get(metric_key)
            if base_pct is None:
                return None, 0, None, 0
            
            if baseline_per_ex.get(per_ex_key):
                base_ci = bootstrap_ci_2sigma(baseline_per_ex[per_ex_key]) * 100
            else:
                base_ci = binomial_ci_2sigma(base_pct, n) * 100
            
            null_pct = data["nulled"].get(metric_key, 0)
            if nulled_per_ex.get(per_ex_key):
                null_ci = bootstrap_ci_2sigma(nulled_per_ex[per_ex_key]) * 100
            else:
                null_ci = binomial_ci_2sigma(null_pct, n) * 100
                
            return (1.0 - base_pct) * 100, base_ci, (1.0 - null_pct) * 100, null_ci

        # Get the two key comparisons
        acc_sl_base, ci_sl_base, acc_sl_null, ci_sl_null = get_acc_stats(
            "short_inc_beats_long_corr_pct", "short_inc_beats_long_corr"
        )
        acc_ls_base, ci_ls_base, acc_ls_null, ci_ls_null = get_acc_stats(
            "long_inc_beats_short_corr_pct", "long_inc_beats_short_corr"
        )

        if acc_sl_base is not None and acc_ls_base is not None:
            data_map[model] = {
                "sl": (acc_sl_base, ci_sl_base, acc_sl_null, ci_sl_null),
                "ls": (acc_ls_base, ci_ls_base, acc_ls_null, ci_ls_null),
                "n": n,
            }

    # Colors
    color_easy = "#2D6A4F"  # Dark Green - Easier to resist
    color_trap = "#E85D75"  # Red - Harder to resist (the trap)
    
    # Create plot
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    
    for col_idx, model in enumerate(models):
        ax = axes[col_idx]
        data = data_map.get(model)
        
        if not data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
            
        x = np.array([0, 1])
        width = 0.35
        
        # Two comparisons
        comparisons = [
            ("sl", color_easy, "Short incorrect > Long correct"),
            ("ls", color_trap, "Long incorrect > Short correct"),
        ]
        
        for i, (key, color, label) in enumerate(comparisons):
            stats = data[key]
            offset = (i - 0.5) * width
            
            # Base condition
            ax.bar(x[0] + offset, stats[0], width * 0.9, yerr=stats[1], 
                   color=color, capsize=3, linewidth=0, label=label)
            
            # Nulled condition (lighter)
            ax.bar(x[1] + offset, stats[2], width * 0.9, yerr=stats[3], 
                   color=color, alpha=0.5, capsize=3, linewidth=0)
        
        # Add sample size annotation
        ax.text(0.5, -0.08, f"n = {data['n']}", ha="center", va="top", 
                fontsize=9, transform=ax.transAxes, style='italic')

        # Title
        ax.set_title(MODEL_NAMES.get(model, model), fontsize=13, fontweight="bold", pad=10)
        
        # Y-axis
        if col_idx == 0:
            ax.set_ylabel("Model Accuracy (%)", fontsize=12, fontweight="bold")
        
        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Length Probe\nRemoved"], fontsize=11)
        ax.set_ylim(0, 105)
        
        # Styling
        ax.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    # Add legend with better positioning
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=2, fontsize=11, frameon=True, fancybox=True, shadow=False,
               edgecolor='gray', framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save both PDF and PNG
    save_path_pdf = output_dir / "length_math_clean.pdf"
    save_path_png = output_dir / "length_math_clean.png"
    fig.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(save_path_png, dpi=300, bbox_inches="tight")
    print(f"Saved clean MATH length bias plot to {save_path_pdf} and {save_path_png}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="artifacts/results")
    parser.add_argument("--output-dir", type=str, default="plots/aggregate/length")
    args = parser.parse_args()
    
    results = load_results(Path(args.results_dir), "length")
    if not results:
        print("No results found!")
        return
        
    output_dir = Path(args.output_dir)
    plot_length_math_clean(results, output_dir)

if __name__ == "__main__":
    main()

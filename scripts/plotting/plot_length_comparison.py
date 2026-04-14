#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from utils import (
    load_results, parse_experiment_name, COLORS, MODEL_NAMES, DATASET_NAMES, EXCLUDED_MODELS,
    binomial_ci_2sigma, bootstrap_ci_2sigma, sem_2sigma, save_plot, plot_heatmap
)

def plot_length_aggregate_two_panel(results: Dict[str, Dict], output_dir: Path):
    """Create per-model length bias plots showing GSM8K vs MATH results with bootstrapped error bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # model -> dataset -> {metrics}
    data_map = {}
    datasets = ["gsm8k", "math", "gsm8k_to_math"]
    models = sorted([m for m in MODEL_NAMES.keys() if m not in EXCLUDED_MODELS])
    
    for name, data in results.items():
        model, dataset, _ = parse_experiment_name(name, "length")
        if model in EXCLUDED_MODELS:
            continue
        
        # Standardize dataset names
        if "gsm8k_to_math" in name:
            ds_key = "gsm8k_to_math"
        elif "gsm8k" in dataset.lower():
            ds_key = "gsm8k"
        elif "math" in dataset.lower():
            ds_key = "math"
        else:
            continue
        
        if ds_key not in data_map: data_map[ds_key] = {}
        if model not in data_map[ds_key]: data_map[ds_key][model] = {}
        
        n = data.get("n_eval_examples", data["baseline"].get("n_examples", 100))
        baseline_per_ex = data.get("baseline_per_example", {})
        nulled_per_ex = data.get("nulled_per_example", {})
        
        # Helper to get Accuracy % and CI from metrics
        def get_acc_stats(metric_key, per_ex_key):
            # Check baseline
            base_pct = data["baseline"].get(metric_key)
            if base_pct is None: return None, 0, None, 0
            
            # Check if we have per-example data for bootstrapping
            if baseline_per_ex.get(per_ex_key):
                base_ci = bootstrap_ci_2sigma(baseline_per_ex[per_ex_key]) * 100
            else:
                base_ci = binomial_ci_2sigma(base_pct, n) * 100
            
            # Check nulled
            null_pct = data["nulled"].get(metric_key, 0)
            if nulled_per_ex.get(per_ex_key):
                null_ci = bootstrap_ci_2sigma(nulled_per_ex[per_ex_key]) * 100
            else:
                null_ci = binomial_ci_2sigma(null_pct, n) * 100
                
            return (1.0 - base_pct) * 100, base_ci, (1.0 - null_pct) * 100, null_ci

        # Standard metrics (for backward compatibility)
        acc_concise_base, ci_concise_base, acc_concise_null, ci_concise_null = get_acc_stats("incorrect_beats_correct_pct", "incorrect_beats_correct")
        acc_verbose_base, ci_verbose_base, acc_verbose_null, ci_verbose_null = get_acc_stats("incorrect_beats_correct_verbose_pct", "incorrect_beats_correct_verbose")

        # 1. Short Incorrect vs Long Correct (Easiest)
        acc_sl_base, ci_sl_base, acc_sl_null, ci_sl_null = get_acc_stats("short_inc_beats_long_corr_pct", "short_inc_beats_long_corr")
        
        # 2. Short Incorrect vs Short Correct (Logic)
        acc_ss_base, ci_ss_base, acc_ss_null, ci_ss_null = get_acc_stats("short_inc_beats_short_corr_pct", "short_inc_beats_short_corr")
        
        # 3. Long Incorrect vs Long Correct (Consistency)
        acc_ll_base, ci_ll_base, acc_ll_null, ci_ll_null = get_acc_stats("long_inc_beats_long_corr_pct", "long_inc_beats_long_corr")
        
        # 4. Long Incorrect vs Short Correct (The Trap)
        acc_ls_base, ci_ls_base, acc_ls_null, ci_ls_null = get_acc_stats("long_inc_beats_short_corr_pct", "long_inc_beats_short_corr")

        data_map[ds_key][model] = {
            "sl": (acc_sl_base, ci_sl_base, acc_sl_null, ci_sl_null),
            "ss": (acc_ss_base, ci_ss_base, acc_ss_null, ci_ss_null),
            "ll": (acc_ll_base, ci_ll_base, acc_ll_null, ci_ll_null),
            "ls": (acc_ls_base, ci_ls_base, acc_ls_null, ci_ls_null),
            "concise": (acc_concise_base, ci_concise_base, acc_concise_null, ci_concise_null),
            "verbose": (acc_verbose_base, ci_verbose_base, acc_verbose_null, ci_verbose_null),
            "n": n,
        }

    # Colors
    color_sl = "#2D6A4F"  # Dark Green (Easiest)
    color_ss = "#74C69D"  # Light Green (Logic)
    color_ll = "#FFB4A2"  # Peach (Consistency)
    color_ls = "#E85D75"  # Pink/Red (The Trap)
    concise_color = "#6C757D" # Grey
    verbose_color = "#E85D75" # Pink
    
    row_labels = {
        "gsm8k": "GSM8K",
        "math": "MATH",
        "gsm8k_to_math": "GSM8K Probe\non MATH"
    }
    
    # Create plot
    fig, axes = plt.subplots(3, len(models), figsize=(3 * len(models), 12), sharey=True)
    
    row_labels_list = ["a)", "b)", "c)"]
    
    for row_idx, ds in enumerate(datasets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            data = data_map.get(ds, {}).get(model)
            
            if not data:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.axis('off')
                continue
                
            x = np.array([0, 1])
            width = 0.18
            
            # Plot the four comparisons if available, else fallback to standard two
            if data["sl"][0] is not None:
                comparisons = [
                    ("sl", color_sl, "Short Inc > Long Corr (Easy)"),
                    ("ss", color_ss, "Short Inc > Short Corr (Logic)"),
                    ("ll", color_ll, "Long Inc > Long Corr (Const)"),
                    ("ls", color_ls, "Long Inc > Short Corr (Trap)")
                ]
                for i, (key, color, label) in enumerate(comparisons):
                    stats = data[key]
                    offset = (i - 1.5) * width
                    # Base - No borders
                    ax.bar(x[0] + offset, stats[0], width, yerr=stats[1], color=color, capsize=2, linewidth=0,
                           label=label if row_idx==0 and col_idx==0 else "")
                    # Null - No borders
                    ax.bar(x[1] + offset, stats[2], width, yerr=stats[3], color=color, alpha=0.6, capsize=2, linewidth=0)
            else:
                # Fallback to 2-bar version
                width_2 = 0.35
                c_stats = data["concise"]
                v_stats = data["verbose"]
                # Concise
                ax.bar(x[0] - width_2/2, c_stats[0], width_2, yerr=c_stats[1], color=concise_color, capsize=2, linewidth=0,
                       label="Concise Correct" if row_idx==0 and col_idx==0 else "")
                ax.bar(x[1] - width_2/2, c_stats[2], width_2, yerr=c_stats[3], color=concise_color, alpha=0.6, capsize=2, linewidth=0)
                # Verbose
                ax.bar(x[0] + width_2/2, v_stats[0], width_2, yerr=v_stats[1], color=verbose_color, capsize=2, linewidth=0,
                       label="Verbose Correct" if row_idx==0 and col_idx==0 else "")
                ax.bar(x[1] + width_2/2, v_stats[2], width_2, yerr=v_stats[3], color=verbose_color, alpha=0.6, capsize=2, linewidth=0)
            
            ax.text(0.5, -8, f"n={data['n']}", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())

            if row_idx == 0:
                ax.set_title(MODEL_NAMES.get(model, model), fontsize=11, fontweight="bold")
            
            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[ds]}\nAcc (%)", fontsize=10, fontweight="bold")
                # Add row labels a), b), c)
                ax.text(-0.35, 1.15, row_labels_list[row_idx], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
                
            ax.set_xticks(x)
            ax.set_xticklabels(["Base", "Null"], fontsize=9)
            ax.set_ylim(0, 110)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=9, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = output_dir / "length_bias_comparison.pdf"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved three-panel length bias plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="artifacts/results")
    parser.add_argument("--output-dir", type=str, default="plots/aggregate/length")
    args = parser.parse_args()
    
    results = load_results(Path(args.results_dir), "length")
    if not results:
        return
        
    output_dir = Path(args.output_dir)
    plot_length_aggregate_two_panel(results, output_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot calibration alpha sweep results.

Creates visualizations showing:
1. Calibration metrics vs alpha for each model
2. Model comparison at optimal alpha
3. Confidence vs uncertainty probe comparison
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List

# Model display names
MODEL_NAMES = {
    "skywork": "Skywork Llama 3.1-8B",
    "allen": "Allen AI Llama 3.1-8B",
    "skywork_qwen3": "Skywork Qwen 3-8B",
    "skywork_qwen-smallest": "Skywork Qwen 3-0.6B",
    "deberta": "DeBERTa v3 Large",
}

# Colors for models
MODEL_COLORS = {
    "skywork": "#2E86AB",
    "allen": "#A23B72",
    "skywork_qwen3": "#F18F01",
    "skywork_qwen-smallest": "#C73E1D",
    "deberta": "#6A994E",
}

# Alpha values tested
ALPHA_VALUES = [0.1, 0.5, 0.75, 1.0, 1.5]


def load_results(results_dir: Path, probe_type: str = "calibration") -> Dict[str, Dict]:
    """Load alpha sweep results for all models.
    
    Args:
        results_dir: Directory containing results
        probe_type: "calibration" or "uncertainty"
    
    Returns:
        Dict mapping model name to results
    """
    results = {}
    
    for model_key in MODEL_NAMES.keys():
        if probe_type == "calibration":
            pattern = f"calibration_{model_key}_math500_alpha_sweep.json"
        else:
            pattern = f"uncertainty_{model_key}_plausibleqa_alpha_sweep.json"
        
        result_file = results_dir / pattern
        if result_file.exists():
            with open(result_file) as f:
                results[model_key] = json.load(f)
    
    return results


def extract_metric_by_alpha(results: Dict, metric_name: str) -> Dict[str, List[float]]:
    """Extract a specific metric across alpha values for all models.
    
    Returns:
        Dict mapping model_key to list of metric values (one per alpha)
    """
    metric_by_model = {}
    
    for model_key, result in results.items():
        values = []
        
        # Baseline
        baseline_val = result["results"]["baseline"]["metrics"].get(metric_name, np.nan)
        
        # Each alpha
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = result["results"][alpha_key]["metrics"].get(metric_name, np.nan)
            values.append(val)
        
        metric_by_model[model_key] = {
            "baseline": baseline_val,
            "alphas": values,
        }
    
    return metric_by_model


def plot_calibration_by_alpha(results: Dict, output_dir: Path):
    """Plot key calibration metrics vs alpha for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Key metrics to plot
    metrics = [
        ("overconfidence_bias_pct", "Overconfidence Bias (%)", "Lower is better", False),
        ("modest_correct_over_overconfident_incorrect_pct", "Correct (Conf:5) > Incorrect (Conf:10) (%)", "Higher is better", True),
        ("correct_over_incorrect_conf10_pct", "Correct > Incorrect at Conf:10 (%)", "Higher is better", True),
        ("correct_over_incorrect_conf1_pct", "Correct > Incorrect at Conf:1 (%)", "Higher is better", True),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (metric_key, title, direction, higher_better) in enumerate(metrics):
        ax = axes[idx]
        
        metric_data = extract_metric_by_alpha(results, metric_key)
        
        for model_key in MODEL_NAMES.keys():
            if model_key not in metric_data:
                continue
            
            data = metric_data[model_key]
            baseline = data["baseline"] * 100  # Convert to percentage
            alphas_vals = [v * 100 for v in data["alphas"]]  # Convert to percentage
            
            # Plot line for alpha values
            ax.plot(ALPHA_VALUES, alphas_vals, 'o-', linewidth=2.5, markersize=8,
                   color=MODEL_COLORS[model_key], label=MODEL_NAMES[model_key], alpha=0.8)
            
            # Plot baseline as horizontal line
            ax.axhline(y=baseline, color=MODEL_COLORS[model_key], linestyle='--', 
                      alpha=0.4, linewidth=1.5)
        
        ax.set_xlabel('Alpha (Nulling Strength)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f"{title}\n({direction})", fontsize=12, fontweight='bold', pad=10)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_xticks(ALPHA_VALUES)
        
        # Set y-axis limits
        if not higher_better:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, 105)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_alpha_sweep.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "calibration_alpha_sweep.png", dpi=300, bbox_inches='tight')
    print(f"Saved calibration alpha sweep plot to {output_dir}/calibration_alpha_sweep.pdf")
    plt.close()


def plot_optimal_alpha_comparison(results: Dict, output_dir: Path):
    """Plot comparison of baseline vs best alpha for each model."""
    
    # Key metric: overconfidence bias (want to minimize)
    metric_key = "overconfidence_bias_pct"
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(MODEL_NAMES.keys())
    x = np.arange(len(models))
    width = 0.25
    
    baseline_vals = []
    best_alpha_vals = []
    best_alphas = []
    
    for model_key in models:
        if model_key not in results:
            baseline_vals.append(np.nan)
            best_alpha_vals.append(np.nan)
            best_alphas.append(np.nan)
            continue
        
        result = results[model_key]
        baseline = result["results"]["baseline"]["metrics"].get(metric_key, np.nan)
        baseline_vals.append(baseline)
        
        # Find best alpha (lowest overconfidence)
        best_val = float('inf')
        best_a = None
        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha_{alpha}"
            val = result["results"][alpha_key]["metrics"].get(metric_key, np.nan)
            if val < best_val:
                best_val = val
                best_a = alpha
        
        best_alpha_vals.append(best_val)
        best_alphas.append(best_a)
    
    # Plot bars
    bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline (No Probe)', 
                   color='#95190C', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, best_alpha_vals, width, label='Best Alpha (Probe Removed)', 
                   color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add alpha labels on bars
    for i, (bar, alpha) in enumerate(zip(bars2, best_alphas)):
        if not np.isnan(alpha):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'α={alpha}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement arrows
    for i in range(len(models)):
        if not np.isnan(baseline_vals[i]) and not np.isnan(best_alpha_vals[i]):
            improvement = baseline_vals[i] - best_alpha_vals[i]
            if improvement > 0:
                ax.annotate('', xy=(x[i] + width/2, best_alpha_vals[i]), 
                          xytext=(x[i] - width/2, baseline_vals[i]),
                          arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
                # Add percentage improvement
                mid_y = (baseline_vals[i] + best_alpha_vals[i]) / 2
                ax.text(x[i] + width*1.5, mid_y, f'-{improvement:.1f}%', 
                       fontsize=9, color='green', fontweight='bold')
    
    ax.set_ylabel('Overconfidence Bias (%)', fontsize=13, fontweight='bold')
    ax.set_title('Overconfidence Bias: Baseline vs Optimal Alpha\n(Lower is Better)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim(0, max(baseline_vals + [1]) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_optimal_alpha.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "calibration_optimal_alpha.png", dpi=300, bbox_inches='tight')
    print(f"Saved optimal alpha comparison to {output_dir}/calibration_optimal_alpha.pdf")
    plt.close()


def plot_probe_comparison(calib_results: Dict, uncert_results: Dict, output_dir: Path):
    """Compare confidence probe vs uncertainty probe effectiveness."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(MODEL_NAMES.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Metric: Overconfidence bias reduction (only calibration has this)
    calib_reductions = []
    
    for model_key in models:
        # Calibration probe
        if model_key in calib_results:
            baseline = calib_results[model_key]["results"]["baseline"]["metrics"]["overconfidence_bias_pct"]
            # Use alpha=1.0 for comparison
            nulled = calib_results[model_key]["results"]["alpha_1.0"]["metrics"]["overconfidence_bias_pct"]
            calib_reductions.append((baseline - nulled) * 100)  # Convert to percentage points
        else:
            calib_reductions.append(0)
    
    bars = ax.bar(x, calib_reductions, width*1.5, 
                  color=[MODEL_COLORS[m] for m in models], 
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}pp', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Overconfidence Bias Reduction (percentage points)', fontsize=13, fontweight='bold')
    ax.set_title('Confidence Probe Effectiveness: Overconfidence Reduction\n(Higher is Better, α=1.0)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylim(min(calib_reductions) * 1.2 if min(calib_reductions) < 0 else 0, 
                max(calib_reductions) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "probe_effectiveness.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "probe_effectiveness.png", dpi=300, bbox_inches='tight')
    print(f"Saved probe effectiveness plot to {output_dir}/probe_effectiveness.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, 
                       default="artifacts/results/calibration_alpha_sweep")
    parser.add_argument("--output-dir", type=str, 
                       default="plots/aggregate/calibration")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Load results
    print("Loading calibration results...")
    calib_results = load_results(results_dir, "calibration")
    print(f"  Loaded {len(calib_results)} models")
    
    print("Loading uncertainty results...")
    uncert_results = load_results(results_dir, "uncertainty")
    print(f"  Loaded {len(uncert_results)} models")
    
    # Create plots
    print("\nCreating plots...")
    plot_calibration_by_alpha(calib_results, output_dir)
    plot_optimal_alpha_comparison(calib_results, output_dir)
    plot_probe_comparison(calib_results, uncert_results, output_dir)
    
    print("\n✓ All plots created successfully!")


if __name__ == "__main__":
    main()

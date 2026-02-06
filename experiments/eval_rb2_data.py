"""
Analzye OOD RewardBench2 results with and without probe debiasing

Run with
uv run experiments/eval_rb2_data.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from scipy.stats import spearmanr

FONTSIZE = 16

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def bootstrap_ci(data, num_samples=2000, ci=95):
    bootstrapped_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

def print_table_allprobes(res_table: Dict):
    """Ez printing for latex tables (keep it legible here and make final formatting with LM)"""

    print("Probe\tMean Chosen \t95% CI Chosen \tMean Rejected \t95% CI Rejected \tMean Diff \t95% CI Diff")
    for model in res_table:
        print(f"{model}")
        for probe in res_table[model]:
            stats = res_table[model][probe]
            mean_chosen = stats["mean_chosen"]
            ci_chosen = stats["ci_chosen"]
            mean_rejected = stats["mean_rejected"]
            ci_rejected = stats["ci_rejected"]
            mean_diff = stats["mean_diff"]
            ci_diff = stats["ci_diff"]
            print(f"{probe} \t{mean_chosen:.3f} \t({mean_chosen - ci_chosen[0]:.3f}, {ci_chosen[1] - mean_chosen:.3f}) \t"
                f"{mean_rejected:.3f} \t({mean_rejected - ci_rejected[0]:.3f}, {ci_rejected[1] - mean_rejected:.3f}) \t"
                f"{mean_diff:.3f} \t({mean_diff - ci_diff[0]:.3f}, {ci_diff[1] - mean_diff:.3f})")


def load_and_process_json(filepath: str, print_table: bool = True):

    data = load_json(filepath)

    rm_models = data["models"]
    probes = data["probes"]
    # rm_models = data["examples"][0]["models"].keys()
    # probes = data["examples"][0]["models"]["allen"].keys()

    res_dict_c: Dict = {model: {probe: [] for probe in probes} for model in rm_models}
    res_dict_r: Dict = {model: {probe: [] for probe in probes} for model in rm_models}
    res_table: Dict = {model: {probe: {} for probe in probes} for model in rm_models}
    res_length_corr: Dict = {model: {probe: {"r": [], "l": []} for probe in probes} for model in rm_models}

    for model in rm_models:
        for probe in probes:
            for ex in data["examples"]:
                res_dict_c[model][probe].append(ex["models"][model][probe]["chosen"])
                res_dict_r[model][probe].append(ex["models"][model][probe]["rejected"])
                r_vals = []
                lengths = []
                r_vals += [ex["models"][model][probe]["chosen"]]
                lengths += [len(ex["chosen"])]
                r_vals += ex["models"][model][probe]["rejected"]
                lengths += [len(e) for e in ex["rejected"]]
                res_length_corr[model][probe]["r"] += r_vals
                res_length_corr[model][probe]["l"] += lengths

        # print(f"\nModel: {model}")
        for probe in probes:
        # for probe in ["baseline", "all_combined"]:
            r_chosen = np.array(res_dict_c[model][probe])
            r_rejected = np.array([np.mean(res_j) for res_j in res_dict_r[model][probe]])
            r_diff = r_chosen - r_rejected

            chosen_ci = bootstrap_ci(r_chosen)
            rejected_ci = bootstrap_ci(r_rejected)
            diff_ci = bootstrap_ci(r_diff)

            mean_chosen = np.mean(r_chosen)
            mean_rejected = np.mean(r_rejected)
            mean_diff = np.mean(r_diff)

            lower_chosen, upper_chosen = chosen_ci
            lower_rejected, upper_rejected = rejected_ci
            lower_diff, upper_diff = diff_ci

            res_table[model][probe] = {
                "mean_chosen": mean_chosen,
                "ci_chosen": (lower_chosen, upper_chosen),
                "mean_rejected": mean_rejected,
                "ci_rejected": (lower_rejected, upper_rejected),
                "mean_diff": mean_diff,
                "ci_diff": (lower_diff, upper_diff),
            }

    if print_table:
        print_table_allprobes(res_table)

    analyze_length_correlation(res_length_corr)

    return res_table, res_dict_c, res_dict_r, res_length_corr

def binned_median_line(x, y, bins=30):
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    xm, ym = [], []
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i+1] if i < bins-1 else x <= edges[i+1])
        if m.sum() >= 3:
            xm.append(np.median(x[m]))
            ym.append(np.median(y[m]))
    return np.array(xm), np.array(ym)

def analyze_length_correlation(res_length_corr: Dict, plot: bool = True, n_bootstrap: int = 2000):
    """Analyze and optionally plot length-reward correlation for each model."""
    probes_to_analyze = ["baseline", "length"]
    
    def bootstrap_corr(rewards, lengths, n_samples=2000):
        """Bootstrap confidence interval for Spearman correlation."""
        n = len(rewards)
        boot_corrs = []
        for _ in range(n_samples):
            idx = np.random.choice(n, size=n, replace=True)
            corr, _ = spearmanr(rewards[idx], lengths[idx])
            boot_corrs.append(corr)
        lower = np.percentile(boot_corrs, 2.5)
        upper = np.percentile(boot_corrs, 97.5)
        return lower, upper
    
    for model in res_length_corr:
        print(f"\nModel: {model}")
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        baseline_rewards = None
        length_rewards = None
        baseline_lengths = None  # Store lengths separately
        
        for probe in probes_to_analyze:
            if probe not in res_length_corr[model]:
                continue
            lengths = np.array(res_length_corr[model][probe]["l"])
            rewards = np.array(res_length_corr[model][probe]["r"])
            
            if probe == "baseline":
                baseline_rewards = rewards
                baseline_lengths = lengths  # Save baseline lengths
            elif probe == "length":
                length_rewards = rewards
            
            corr, p_val = spearmanr(rewards, lengths)
            ci_lower, ci_upper = bootstrap_corr(rewards, lengths, n_samples=n_bootstrap)
            print(f"Probe: {probe} - Length Correlation (Spearman): {corr:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], p={p_val:.4f})")
            
            if plot:
                if probe == "baseline":
                    label = "Uncorrected"
                    c = "#777777" #777777 (Dark Gray)Debiased: #7D83FF (Periwinkle)
                elif probe == "length":
                    label = "Length-corrected"
                    c = "#7D83FF"
                else:
                    NotImplementedError()
                ax1.scatter(lengths, rewards, alpha=1.0, label=label, s=10, color=c)
                # xm, ym = binned_median_line(lengths, rewards, bins=25)
                # ax1.plot(xm, ym, linewidth=2, label=f'{label} median (ρ={corr:.3f})')
        
        if plot:
            ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
            ax1.set_xlabel('Response Length [#char]', fontsize=FONTSIZE)
            ax1.set_ylabel('Reward Score [ ]', fontsize=FONTSIZE)
            # ax1.set_title(f'Length vs Reward - {model}')
            ax1.legend(loc='best', fontsize=FONTSIZE-2)
            
            # Plot diff
            if baseline_rewards is not None and length_rewards is not None and baseline_lengths is not None:
                if len(baseline_rewards) != len(length_rewards):
                    print(f"Warning: Array length mismatch for {model}")
                    continue
                diff_rewards = baseline_rewards - length_rewards
                corr_diff, p_val = spearmanr(diff_rewards, baseline_lengths)
                ci_lower, ci_upper = bootstrap_corr(diff_rewards, baseline_lengths, n_samples=n_bootstrap)
                print(f"Probe: baseline - length - Length Correlation (Spearman): {corr_diff:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], p={p_val:.4f})")
                ax2.scatter(baseline_lengths, diff_rewards, alpha=1.0, label='Uncorrected - Length-corrected', s=10, color='#06070E')
                ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
                ax2.set_xlabel('Response Length [#char]', fontsize=FONTSIZE)
                ax2.set_ylabel('Reward Difference [ ]', fontsize=FONTSIZE)
                # ax2.set_title(f'Length vs Reward Difference - {model}')
                ax2.legend(loc='best', fontsize=FONTSIZE-2)
            
            plt.tight_layout()
            model_filename = model.replace("/", "_").replace(" ", "_")
            plt.savefig(f'length_correlation_{model_filename}.pdf', format='pdf', bbox_inches='tight')
            # plt.savefig(f'length_correlation_{model_filename}.png', dpi=300)
            plt.close(fig)

def main():
    filepath = Path(__file__).parent.parent / 'rb2_data' / 'rewardbench2_multiprobe_combined.json'
    # filepath = Path(__file__).parent.parent / 'rb2_data' / 'rewardbench2_multiprobe_combined_update.json'

    res_table, res_dict_c, res_dict_r, res_length_corr = load_and_process_json(filepath)

if __name__ == "__main__":
    main()

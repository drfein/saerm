#!/usr/bin/env python3
"""
Evaluate RewardBench 2 with multiple bias probes applied simultaneously.

This script:
1. Loads all available probes from artifacts/probes/
2. Orthogonalizes them using Gram-Schmidt
3. Projects out the entire bias subspace when computing rewards
4. Reports RewardBench 2 accuracy with and without nullification

RewardBench 2 has multiple rejected responses per example.
The model must score chosen higher than ALL rejected to be correct.

Usage:
    python experiments/run_rewardbench_multiprobe.py \
        --model Skywork/Skywork-Reward-V2-Llama-3.1-8B \
        --probes-dir artifacts/probes \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.nullbias.probe import (
    get_base_model, 
    get_score_head, 
    tokenize_inputs,
    gram_schmidt,
    project_to_null_space,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_probes(probes_dir: Path) -> Dict[str, torch.Tensor]:
    """Load all probes from the probes directory.
    
    Returns:
        Dictionary mapping probe name to probe tensor
        Probes can be 1D vectors [hidden_dim] or 2D matrices [n_basis, hidden_dim]
    """
    probes = {}
    
    for bias_type_dir in probes_dir.iterdir():
        if not bias_type_dir.is_dir():
            continue
        
        for probe_dir in bias_type_dir.iterdir():
            probe_path = probe_dir / "probe.pt"
            if probe_path.exists():
                probe = torch.load(probe_path, map_location="cpu")
                name = f"{bias_type_dir.name}/{probe_dir.name}"
                probes[name] = probe
                if probe.dim() == 1:
                    logger.info("Loaded probe: %s (shape=%s, dim=%d)", name, list(probe.shape), probe.shape[0])
                elif probe.dim() == 2:
                    logger.info("Loaded probe: %s (shape=%s, n_basis=%d, hidden_dim=%d)", 
                               name, list(probe.shape), probe.shape[0], probe.shape[1])
                else:
                    logger.warning("Loaded probe: %s (unexpected shape=%s)", name, list(probe.shape))
    
    return probes


def get_rewards_multi(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probes: Dict[str, torch.Tensor],
    combined_basis: Optional[torch.Tensor] = None,
    null_alpha: float = 1.0,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute baseline, individual probe, and combined rewards in a single forward pass.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer
        texts: List of formatted texts
        probes: Dict mapping probe name to probe tensor
        combined_basis: [n_probes, hidden_dim] orthonormal basis for all probes combined
        null_alpha: Nullification strength
        batch_size: Batch size
        device: Device for inputs
        max_length: Max sequence length
        show_progress: Show progress bar
        
    Returns:
        Dict mapping condition name to rewards tensor:
        - "baseline": no nulling
        - "all_probes": all probes combined
        - "{probe_name}": each individual probe
    """
    model.eval()
    base_model = get_base_model(model)
    score_head = get_score_head(model)
    
    # Prepare probes - handle both 1D vectors and 2D matrices
    probe_vecs = {}
    for name, probe in probes.items():
        probe = probe.to(device).float()
        if probe.dim() == 1:
            # Single vector probe
            probe_vecs[name] = probe / (probe.norm() + 1e-8)
        elif probe.dim() == 2:
            # Matrix probe (e.g., position probe with multiple basis vectors)
            # Use the first basis vector as the probe direction for individual evaluation
            v = probe[0]
            probe_vecs[name] = v / (v.norm() + 1e-8)
        else:
            raise ValueError(f"Unexpected probe shape for {name}: {probe.shape}")
    
    if combined_basis is not None and combined_basis.shape[0] > 0:
        combined_basis = combined_basis.to(device).float()
    
    # Pre-tokenize all texts (handles both strings and pairs)
    logger.info("Tokenizing %d texts...", len(texts))
    all_encodings = tokenize_inputs(
        tokenizer,
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Initialize score collectors
    all_scores = {name: [] for name in ["baseline", "all_probes"] + list(probes.keys())}
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing rewards", total=n_batches)
    
    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            
            inputs = {k: v[start:end].to(device) for k, v in all_encodings.items()}
            
            # Get hidden states once
            outputs = base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=device),
                last_token_indices,
            ].float()
            
            # Baseline score (no nulling)
            baseline_scores = score_head(last_hidden.to(hidden_states.dtype)).squeeze(-1)
            all_scores["baseline"].append(baseline_scores.cpu())
            
            # All probes combined
            if combined_basis is not None and combined_basis.shape[0] > 0:
                nulled_hidden = project_to_null_space(last_hidden, combined_basis)
                combined_scores = score_head(nulled_hidden.to(hidden_states.dtype)).squeeze(-1)
            else:
                combined_scores = baseline_scores
            all_scores["all_probes"].append(combined_scores.cpu())
            
            # Each probe individually
            for name, probe_vec in probe_vecs.items():
                proj = (last_hidden @ probe_vec).unsqueeze(-1)
                nulled = last_hidden - null_alpha * proj * probe_vec.unsqueeze(0)
                scores = score_head(nulled.to(hidden_states.dtype)).squeeze(-1)
                all_scores[name].append(scores.cpu())
    
    return {name: torch.cat(scores_list, dim=0) for name, scores_list in all_scores.items()}


def load_rewardbench(split: str = "test") -> List[Dict]:
    """Load RewardBench 2 dataset.
    
    RewardBench 2 has multiple rejected responses per example.
    The model must score chosen higher than ALL rejected to be correct.
    
    The prompt field is a list of messages (conversation history).
    """
    logger.info("Loading RewardBench 2 (split=%s)", split)
    dataset = load_dataset("allenai/reward-bench-2", split=split)
    
    examples = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", [])
        chosen_raw = row.get("chosen", "")
        rejected_list = row.get("rejected", [])
        
        # prompt is a list of messages in RewardBench 2
        # Convert to string if it's a list
        if isinstance(prompt, list):
            # It's a conversation - extract the text content
            if prompt and isinstance(prompt[0], dict):
                # List of message dicts
                prompt_text = "\n\n".join(
                    m.get("content", "") for m in prompt if m.get("content")
                )
            else:
                # List of strings
                prompt_text = "\n\n".join(str(p) for p in prompt)
        else:
            prompt_text = str(prompt)
        
        # chosen is a list in RewardBench 2, extract first element
        if isinstance(chosen_raw, list):
            chosen = chosen_raw[0] if chosen_raw else ""
        else:
            chosen = chosen_raw
        
        # Ensure rejected is a list
        if isinstance(rejected_list, str):
            rejected_list = [rejected_list]
        
        if prompt_text and chosen and rejected_list:
            examples.append({
                "idx": idx,
                "prompt": prompt_text,
                "prompt_raw": prompt,  # Keep original for proper formatting
                "chosen": chosen,
                "rejected": rejected_list,  # List of rejected responses
                "subset": row.get("subset", "unknown"),
            })
    
    logger.info("Loaded %d RewardBench 2 examples", len(examples))
    return examples


def format_rb2_conversation(tokenizer: Any, prompt_raw: Any, response: str):
    """Format RewardBench 2 conversation with response.
    
    Handles the case where prompt is a list of messages.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_raw: Raw prompt (list of messages or string)
        response: Assistant response to append
        
    Returns:
        Formatted conversation string OR tuple (prompt, response) for pair-format models
    """
    # Extract prompt text first
    if isinstance(prompt_raw, list):
        if prompt_raw and isinstance(prompt_raw[0], dict):
            prompt_text = "\n\n".join(m.get("content", "") for m in prompt_raw)
        else:
            prompt_text = "\n\n".join(str(p) for p in prompt_raw)
    else:
        prompt_text = str(prompt_raw)
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # Build conversation from prompt messages
        if isinstance(prompt_raw, list) and prompt_raw and isinstance(prompt_raw[0], dict):
            # Already in message format
            conv = list(prompt_raw)
        elif isinstance(prompt_raw, list):
            # List of strings - treat as user messages
            conv = [{"role": "user", "content": prompt_text}]
        else:
            conv = [{"role": "user", "content": prompt_text}]
        
        # Add assistant response
        conv.append({"role": "assistant", "content": response})
        
        formatted = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        # Remove BOS token - it will be added back during tokenization
        if tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token):]
        return formatted
    else:
        # Pair format for models without chat template (e.g., DeBERTa)
        # Return tuple for tokenizer(prompt, response) pair encoding
        return (prompt_text, response)


def evaluate_rewardbench_multi(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    examples: List[Dict],
    probes: Dict[str, torch.Tensor],
    probe_basis: Optional[torch.Tensor] = None,
    null_alpha: float = 1.0,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Dict[str, Dict[str, float]]:
    """Evaluate on RewardBench 2 with baseline, individual probes, and all combined.
    
    All computed in a single forward pass for efficiency.
    
    For each example, chosen must score higher than ALL rejected responses.
    
    Returns:
        Dict mapping condition name to results dict:
        - "baseline": no nulling
        - "all_probes": all probes combined  
        - "{probe_name}": each individual probe
    """
    # Prepare all texts - flatten rejected lists for batch scoring
    all_chosen = []
    all_rejected = []  # Flat list of all rejected texts
    rejected_counts = []  # Number of rejected per example
    subsets = []
    
    for ex in examples:
        prompt_raw = ex.get("prompt_raw", ex["prompt"])
        all_chosen.append(format_rb2_conversation(tokenizer, prompt_raw, ex["chosen"]))
        rejected_list = ex["rejected"]
        rejected_counts.append(len(rejected_list))
        for rej in rejected_list:
            all_rejected.append(format_rb2_conversation(tokenizer, prompt_raw, rej))
        subsets.append(ex["subset"])
    
    # Get rewards for chosen (all conditions)
    logger.info("Computing chosen rewards (%d texts)...", len(all_chosen))
    chosen_rewards = get_rewards_multi(
        model, tokenizer, all_chosen,
        probes=probes,
        combined_basis=probe_basis,
        null_alpha=null_alpha,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    # Get rewards for all rejected (all conditions)
    logger.info("Computing rejected rewards (%d texts)...", len(all_rejected))
    rejected_rewards = get_rewards_multi(
        model, tokenizer, all_rejected,
        probes=probes,
        combined_basis=probe_basis,
        null_alpha=null_alpha,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    def compute_accuracy(chosen_r, rejected_r):
        """Compute accuracy from reward tensors."""
        correct = []
        rej_idx = 0
        for i, count in enumerate(rejected_counts):
            ch = chosen_r[i]
            rej = rejected_r[rej_idx : rej_idx + count]
            rej_idx += count
            is_correct = (ch > rej).all().item()
            correct.append(float(is_correct))
        return torch.tensor(correct)
    
    def build_results(correct):
        """Build results dict from correctness tensor."""
        results = {
            "accuracy": correct.mean().item(),
            "n_examples": len(examples),
            "total_rejected": len(all_rejected),
            "avg_rejected_per_example": len(all_rejected) / len(examples),
        }
        
        subset_correct = {}
        subset_total = {}
        for i, subset in enumerate(subsets):
            if subset not in subset_correct:
                subset_correct[subset] = 0
                subset_total[subset] = 0
            subset_correct[subset] += correct[i].item()
            subset_total[subset] += 1
        
        for subset in sorted(subset_correct.keys()):
            results[f"accuracy_{subset}"] = subset_correct[subset] / subset_total[subset]
            results[f"n_{subset}"] = subset_total[subset]
        
        return results
    
    # Compute results for all conditions
    all_results = {}
    for condition in chosen_rewards.keys():
        correct = compute_accuracy(chosen_rewards[condition], rejected_rewards[condition])
        all_results[condition] = build_results(correct)
    
    return all_results


# Multiplier for 95% confidence interval
CI_95_MULTIPLIER = 1.96


def binomial_ci95(p: float, n: int) -> float:
    """Compute 95% confidence interval half-width for a proportion."""
    if n <= 0:
        return 0.0
    se = np.sqrt(p * (1 - p) / n)
    return CI_95_MULTIPLIER * se


def create_rewardbench_multiprobe_plot(
    baseline: Dict[str, float],
    combined: Dict[str, float],
    individual: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "RewardBench 2 Results",
) -> None:
    """Create bar plot showing baseline, individual probes, and combined accuracy.
    
    Args:
        baseline: Baseline results dict
        combined: Combined (all probes) results dict
        individual: Dict mapping probe name to results dict
        output_path: Path to save the plot
        title: Plot title
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data: baseline, individual probes, combined
    conditions = ["Baseline"]
    accuracies = [baseline["accuracy"] * 100]
    n = baseline["n_examples"]
    
    # Add individual probes
    for name in sorted(individual.keys()):
        # Shorten name
        short = name.split("/")[-1] if "/" in name else name
        # Further shorten if needed
        if len(short) > 20:
            short = short[:17] + "..."
        conditions.append(short)
        accuracies.append(individual[name]["accuracy"] * 100)
    
    # Add combined
    conditions.append("All Combined")
    accuracies.append(combined["accuracy"] * 100)
    
    # Compute error bars
    errors = [binomial_ci95(acc/100, n) * 100 for acc in accuracies]
    
    # Compute deltas from baseline
    baseline_acc = accuracies[0]
    deltas = [acc - baseline_acc for acc in accuracies]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 0.9), 6))
    
    x = np.arange(len(conditions))
    
    # Color scheme: baseline is blue, individual are grey, combined is green
    colors = ["#4C72B0"]  # Baseline
    colors.extend(["#888888"] * len(individual))  # Individual probes
    colors.append("#55A868")  # Combined
    
    bars = ax.bar(x, accuracies, yerr=errors, capsize=3, 
                  color=colors, edgecolor="#333", linewidth=0.5)
    
    # Add value labels and delta on bars
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        height = bar.get_height()
        # Value label
        ax.annotate(f"{height:.1f}%",
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8)
        # Delta label (skip baseline)
        if i > 0:
            delta_color = "green" if delta >= 0 else "red"
            ax.annotate(f"({delta:+.1f})",
                       xy=(bar.get_x() + bar.get_width()/2, height + errors[i] + 3),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=7, color=delta_color)
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, max(accuracies) + 15)
    
    # Title
    full_title = f"{title}\n(n={n}, {len(individual)} probes)"
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    
    # Add reference line at baseline
    ax.axhline(y=baseline_acc, color="#4C72B0", linestyle="--", alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--probes-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/rewardbench2_multiprobe.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots/rewardbench"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null-alpha", type=float, default=1.0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--probes", type=str, nargs="*", help="Specific probes to use (default: all)")
    parser.add_argument("--probe-model", type=str, help="Only use probes trained on this model (e.g., 'skywork', 'gemma')")
    args = parser.parse_args()
    
    # Load probes
    all_probes = load_probes(args.probes_dir)
    
    if not all_probes:
        logger.error("No probes found in %s", args.probes_dir)
        logger.error("Run bias experiments first to generate probes:")
        logger.error("  python experiments/run_experiment.py --config experiments/configs/length_skywork.yaml")
        logger.error("  python experiments/run_experiment.py --config experiments/configs/position_skywork_gsm8k.yaml")
        logger.error("  python experiments/run_experiment.py --config experiments/configs/sycophancy_skywork_gsm8k_mc.yaml")
        logger.error("  python experiments/run_experiment.py --config experiments/configs/uncertainty_skywork_gsm8k_mc.yaml")
        sys.exit(1)
    
    # Filter probes by model name if specified
    if args.probe_model:
        filtered = {k: v for k, v in all_probes.items() if args.probe_model.lower() in k.lower()}
        if not filtered:
            logger.error("No probes found for model '%s'. Available probes:", args.probe_model)
            for name in all_probes:
                logger.error("  - %s", name)
            sys.exit(1)
        all_probes = filtered
        logger.info("Filtered to probes for model: %s", args.probe_model)
    
    # Filter probes by specific names if specified
    if args.probes:
        filtered = {}
        for name in args.probes:
            matches = [k for k in all_probes if name in k]
            for m in matches:
                filtered[m] = all_probes[m]
        all_probes = filtered
    
    # Load model first to get hidden dimension
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(args.device)
    
    # Get model's hidden dimension
    base_model = get_base_model(model)
    hidden_dim = base_model.config.hidden_size
    logger.info("Model hidden dimension: %d", hidden_dim)
    
    # Filter probes to only those compatible with model's hidden dimension
    # and expand matrices into individual vectors
    compatible_probes = {}
    for name, probe in all_probes.items():
        if probe.dim() == 1:
            # 1D vector: check if dimension matches
            if probe.shape[0] == hidden_dim:
                compatible_probes[name] = probe
            else:
                logger.warning("Skipping probe %s: dimension mismatch (%d != %d)", 
                             name, probe.shape[0], hidden_dim)
        elif probe.dim() == 2:
            # 2D matrix: check if hidden_dim matches
            if probe.shape[1] == hidden_dim:
                # For matrices, we'll use all basis vectors separately
                # Add each basis vector as a separate probe
                for i in range(probe.shape[0]):
                    vec_name = f"{name}_basis{i}" if probe.shape[0] > 1 else name
                    compatible_probes[vec_name] = probe[i]
            else:
                logger.warning("Skipping probe %s: hidden_dim mismatch (%d != %d)", 
                             name, probe.shape[1], hidden_dim)
        else:
            logger.warning("Skipping probe %s: unexpected shape %s", name, list(probe.shape))
    
    all_probes = compatible_probes
    
    logger.info("Using %d compatible probes:", len(all_probes))
    for name in all_probes:
        logger.info("  - %s", name)
    
    # Group probes by category and average within each category
    # Categories: sycophancy, uncertainty, position, length
    category_probes = {}  # category -> list of normalized probes
    for name, probe in all_probes.items():
        # Extract category from probe name (e.g., "sycophancy/sycophancy_gemma2_mmlu" -> "sycophancy")
        category = name.split("/")[0] if "/" in name else "other"
        probe = probe.float()
        probe = probe / (probe.norm() + 1e-8)  # Normalize
        if category not in category_probes:
            category_probes[category] = []
        category_probes[category].append(probe)
    
    # Average probes within each category, then normalize
    averaged_probes = {}
    for category, probes in category_probes.items():
        if len(probes) == 1:
            averaged = probes[0]
        else:
            # Stack and average
            stacked = torch.stack(probes, dim=0)  # [n, hidden_dim]
            averaged = stacked.mean(dim=0)  # [hidden_dim]
        # Re-normalize the averaged probe
        averaged = averaged / (averaged.norm() + 1e-8)
        averaged_probes[category] = averaged
        logger.info("Category '%s': averaged %d probes into 1 direction", category, len(probes))
    
    # Stack averaged category probes for combined nulling
    probe_list = list(averaged_probes.values())
    probe_basis = torch.stack(probe_list, dim=0)  # [n_categories, hidden_dim]
    logger.info("Combined basis: %d category directions (averaged)", probe_basis.shape[0])
    
    # Load RewardBench 2
    examples = load_rewardbench(args.split)
    
    # Evaluate all conditions in single pass
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING (baseline + %d probes individually + all combined)", len(all_probes))
    logger.info("=" * 60)
    all_results = evaluate_rewardbench_multi(
        model, tokenizer, examples,
        probes=all_probes,
        probe_basis=probe_basis,
        null_alpha=args.null_alpha,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    
    baseline_results = all_results["baseline"]
    combined_results = all_results["all_probes"]
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL ACCURACY BY CONDITION")
    logger.info("=" * 80)
    logger.info("%-50s %10s %10s", "Condition", "Accuracy", "Delta")
    logger.info("-" * 80)
    
    baseline_acc = baseline_results["accuracy"]
    logger.info("%-50s %9.2f%% %10s", "Baseline (no nulling)", 100 * baseline_acc, "-")
    
    # Individual probes
    individual_results = {}
    for name in sorted(all_probes.keys()):
        if name in all_results:
            acc = all_results[name]["accuracy"]
            delta = acc - baseline_acc
            individual_results[name] = all_results[name]
            # Shorten probe name for display
            short_name = name.split("/")[-1] if "/" in name else name
            logger.info("%-50s %9.2f%% %+9.2f%%", f"  {short_name}", 100 * acc, 100 * delta)
    
    # Combined
    combined_acc = combined_results["accuracy"]
    delta_combined = combined_acc - baseline_acc
    logger.info("-" * 80)
    logger.info("%-50s %9.2f%% %+9.2f%%", "All probes combined", 100 * combined_acc, 100 * delta_combined)
    
    # Print per-subset breakdown for baseline vs combined
    logger.info("\n" + "=" * 80)
    logger.info("SUBSET BREAKDOWN: Baseline vs All Probes Combined")
    logger.info("=" * 80)
    logger.info("%-30s %10s %10s %10s", "Subset", "Baseline", "Combined", "Delta")
    logger.info("-" * 80)
    
    logger.info("%-30s %9.2f%% %9.2f%% %+9.2f%%", 
                "OVERALL", 100 * baseline_acc, 100 * combined_acc, 100 * delta_combined)
    
    for key in sorted(baseline_results.keys()):
        if key.startswith("accuracy_") and not key.startswith("accuracy_n"):
            subset = key[9:]
            b_acc = baseline_results[key]
            c_acc = combined_results.get(key, 0)
            delta = c_acc - b_acc
            logger.info("%-30s %9.2f%% %9.2f%% %+9.2f%%",
                        subset, 100 * b_acc, 100 * c_acc, 100 * delta)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "probes_used": list(all_probes.keys()),
        "n_probes": len(all_probes),
        "null_alpha": args.null_alpha,
        "baseline": baseline_results,
        "all_probes": combined_results,
        "individual_probes": individual_results,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", args.output)
    
    # Generate plot with individual probes
    plot_path = args.plots_dir / f"rewardbench2_{args.probe_model or 'all'}_plot.png"
    create_rewardbench_multiprobe_plot(
        baseline_results, 
        combined_results,
        individual_results,
        output_path=plot_path, 
        title=f"RewardBench 2: {args.probe_model or 'All'} Probes",
    )
    logger.info("Plot saved to %s", plot_path)


if __name__ == "__main__":
    main()


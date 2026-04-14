#!/usr/bin/env python3
"""
Run calibration experiments with multiple alpha values.

Tests both:
1. Confidence probe (manually manipulated confidence scores)
2. Uncertainty probe (hedging language)

For each probe, evaluates with alpha = [0.1, 0.5, 0.75, 1.0, 1.5]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch

from src.nb.experiments.calibration import CalibrationBiasExperiment
from src.nb.experiments.uncertainty import UncertaintyBiasExperiment
from src.nb.experiments.base import ExperimentConfig
from src.nb.nullbias.probe import get_rewards_both

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiment_with_alpha_sweep(
    config_path: Path,
    alpha_values: list[float],
    experiment_class,
) -> Dict[str, Any]:
    """Run experiment with multiple alpha values.
    
    Args:
        config_path: Path to experiment config
        alpha_values: List of alpha values to test
        experiment_class: CalibrationBiasExperiment or UncertaintyBiasExperiment
    
    Returns:
        Dict with results for each alpha value
    """
    # Load config
    config = ExperimentConfig.from_yaml(config_path)
    
    # Create experiment
    experiment = experiment_class(config)
    
    # Load model and dataset
    logger.info("Loading model and dataset...")
    experiment.load_model()
    experiment.load_dataset()
    
    # Build probe
    logger.info("Building probe...")
    probe_metadata = experiment.build_probe()
    
    # Get evaluation examples
    eval_examples = experiment.dataset.get_eval_examples(experiment.tokenizer)
    n_eval = len(eval_examples)
    logger.info("Evaluating on %d examples", n_eval)
    
    # Get all texts
    all_texts, text_meta = experiment._get_all_texts_and_variants(eval_examples)
    
    # Run baseline once (alpha doesn't matter for baseline)
    logger.info("Running baseline evaluation...")
    baseline_rewards, _ = get_rewards_both(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        texts=all_texts,
        probe=None,  # No probe for baseline
        alpha=1.0,
        batch_size=config.batch_size,
        device=config.device,
        max_length=config.max_length,
    )
    baseline_organized = experiment._organize_rewards(baseline_rewards, text_meta, n_eval)
    baseline_metrics, baseline_per_example = experiment._compute_metrics(
        baseline_organized, eval_examples
    )
    
    # Run nulled for each alpha value
    results_by_alpha = {
        "baseline": {
            "metrics": baseline_metrics,
            "per_example": baseline_per_example,
        }
    }
    
    for alpha in alpha_values:
        logger.info(f"Running evaluation with alpha={alpha}...")
        
        # Compute nulled rewards with this alpha
        _, nulled_rewards = get_rewards_both(
            model=experiment.model,
            tokenizer=experiment.tokenizer,
            texts=all_texts,
            probe=experiment.probe,
            alpha=alpha,
            batch_size=config.batch_size,
            device=config.device,
            max_length=config.max_length,
        )
        
        nulled_organized = experiment._organize_rewards(nulled_rewards, text_meta, n_eval)
        nulled_metrics, nulled_per_example = experiment._compute_metrics(
            nulled_organized, eval_examples
        )
        
        results_by_alpha[f"alpha_{alpha}"] = {
            "metrics": nulled_metrics,
            "per_example": nulled_per_example,
        }
    
    # Compile full results
    full_results = {
        "config": {
            "name": config.name,
            "bias_type": config.bias_type,
            "model_path": config.model_path,
            "dataset_source": config.dataset_source,
            "probe_size": config.probe_size,
            "n_eval_examples": n_eval,
        },
        "probe_metadata": probe_metadata,
        "alpha_values": alpha_values,
        "results": results_by_alpha,
    }
    
    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Run calibration experiments with alpha sweep"
    )
    parser.add_argument(
        "--calibration-config",
        type=str,
        required=True,
        help="Path to calibration experiment config (confidence probe)",
    )
    parser.add_argument(
        "--uncertainty-config",
        type=str,
        default=None,
        help="Path to uncertainty experiment config (hedging probe)",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.75, 1.0, 1.5],
        help="Alpha values to test (default: 0.1 0.5 0.75 1.0 1.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results/calibration_alpha_sweep",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run calibration experiment (confidence probe)
    logger.info("=" * 80)
    logger.info("Running CALIBRATION experiment (confidence probe)")
    logger.info("=" * 80)
    
    calibration_results = run_experiment_with_alpha_sweep(
        config_path=Path(args.calibration_config),
        alpha_values=args.alpha_values,
        experiment_class=CalibrationBiasExperiment,
    )
    
    # Save calibration results
    calibration_output = output_dir / f"{calibration_results['config']['name']}_alpha_sweep.json"
    with open(calibration_output, "w") as f:
        # Convert per-example lists to summary stats to keep file size manageable
        for alpha_key, alpha_results in calibration_results["results"].items():
            if "per_example" in alpha_results:
                del alpha_results["per_example"]
        json.dump(calibration_results, f, indent=2)
    
    logger.info(f"Saved calibration results to {calibration_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CALIBRATION RESULTS (Confidence Probe)")
    print("=" * 80)
    print(f"\nBaseline metrics:")
    for key, val in calibration_results["results"]["baseline"]["metrics"].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    print(f"\nNulled metrics by alpha:")
    for alpha in args.alpha_values:
        alpha_key = f"alpha_{alpha}"
        print(f"\n  Alpha = {alpha}:")
        for key, val in calibration_results["results"][alpha_key]["metrics"].items():
            if isinstance(val, float) and key != "n_examples":
                baseline_val = calibration_results["results"]["baseline"]["metrics"].get(key, 0)
                delta = val - baseline_val
                print(f"    {key}: {val:.4f} (Δ {delta:+.4f})")
    
    # Run uncertainty experiment if config provided
    if args.uncertainty_config:
        logger.info("\n" + "=" * 80)
        logger.info("Running UNCERTAINTY experiment (hedging probe)")
        logger.info("=" * 80)
        
        uncertainty_results = run_experiment_with_alpha_sweep(
            config_path=Path(args.uncertainty_config),
            alpha_values=args.alpha_values,
            experiment_class=UncertaintyBiasExperiment,
        )
        
        # Save uncertainty results
        uncertainty_output = output_dir / f"{uncertainty_results['config']['name']}_alpha_sweep.json"
        with open(uncertainty_output, "w") as f:
            # Convert per-example lists to summary stats
            for alpha_key, alpha_results in uncertainty_results["results"].items():
                if "per_example" in alpha_results:
                    del alpha_results["per_example"]
            json.dump(uncertainty_results, f, indent=2)
        
        logger.info(f"Saved uncertainty results to {uncertainty_output}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("UNCERTAINTY RESULTS (Hedging Probe)")
        print("=" * 80)
        print(f"\nBaseline metrics:")
        for key, val in uncertainty_results["results"]["baseline"]["metrics"].items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
        
        print(f"\nNulled metrics by alpha:")
        for alpha in args.alpha_values:
            alpha_key = f"alpha_{alpha}"
            print(f"\n  Alpha = {alpha}:")
            for key, val in uncertainty_results["results"][alpha_key]["metrics"].items():
                if isinstance(val, float) and key != "n_examples":
                    baseline_val = uncertainty_results["results"]["baseline"]["metrics"].get(key, 0)
                    delta = val - baseline_val
                    print(f"    {key}: {val:.4f} (Δ {delta:+.4f})")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

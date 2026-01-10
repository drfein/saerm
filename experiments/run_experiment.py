#!/usr/bin/env python3
"""
Run bias evaluation experiments.

Usage:
    # From config file:
    python experiments/run_experiment.py --config experiments/configs/length_skywork.yaml
    
    # From command line args:
    python experiments/run_experiment.py \\
        --bias-type length \\
        --name length_skywork_gsm8k \\
        --model Skywork/Skywork-Reward-V2-Llama-3.1-8B \\
        --dataset-source data/gsm8k_soln.json \\
        --artifacts-dir artifacts \\
        --plots-dir plots
    
    # Cross-dataset generalization (train probe on one, eval on another):
    python experiments/run_experiment.py \\
        --bias-type position \\
        --name position_skywork_gsm8k_to_mmlu \\
        --model Skywork/Skywork-Reward-V2-Llama-3.1-8B \\
        --dataset-source cais/mmlu \\
        --probe-source guipenedo/gsm8k-mc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Type

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.length import LengthBiasExperiment
from src.nb.experiments.sycophancy import SycophancyBiasExperiment
from src.nb.experiments.uncertainty import UncertaintyBiasExperiment
from src.nb.experiments.position import (
    PositionBiasExperiment, 
    BinaryPositionBiasExperiment,
    FreeformPositionBiasExperiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT_CLASSES: dict[str, Type[BiasExperiment]] = {
    "length": LengthBiasExperiment,
    "sycophancy": SycophancyBiasExperiment,
    "uncertainty": UncertaintyBiasExperiment,
    "position": PositionBiasExperiment,
}


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from command line arguments."""
    extra = {}
    
    # Collect --extra-* arguments
    for key, value in vars(args).items():
        if key.startswith("extra_") and value is not None:
            extra_key = key[6:]  # Remove "extra_" prefix
            # Try to parse as appropriate type
            if value.lower() in ("true", "false"):
                extra[extra_key] = value.lower() == "true"
            else:
                try:
                    extra[extra_key] = int(value)
                except ValueError:
                    try:
                        extra[extra_key] = float(value)
                    except ValueError:
                        extra[extra_key] = value
    
    # Dataset class may be used both:
    # - at the top-level (to select experiment class in this runner)
    # - inside config.extra (for dataset builders that look up extra["dataset_class"])
    # Keep it in `extra` and also copy to the top-level field.
    dataset_class = extra.get("dataset_class", "")
    
    return ExperimentConfig(
        name=args.name,
        bias_type=args.bias_type,
        model_path=args.model,
        trust_remote_code=args.trust_remote_code,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset,
        dataset_class=dataset_class,
        probe_size=args.probe_size,
        max_test_examples=args.max_examples,
        split_seed=args.split_seed,
        null_alpha=args.null_alpha,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        artifacts_dir=args.artifacts_dir,
        plots_dir=args.plots_dir,
        save_probe=True,
        extra=extra,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file mode
    parser.add_argument("--config", type=Path, help="Path to config YAML file")
    
    # Direct specification mode
    parser.add_argument("--bias-type", choices=list(EXPERIMENT_CLASSES.keys()), help="Bias type")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--model", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--dataset-source", type=str, help="Dataset source path or ID")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for output organization (e.g., gsm8k_mc, mmlu)")
    
    # Cross-dataset generalization
    parser.add_argument("--probe-source", type=str, help="Probe training dataset (for cross-dataset)")
    
    # Common options
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory for results/probes")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory for plots")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--null-alpha", type=float, default=1.0, help="Nullification strength")
    parser.add_argument("--probe-size", type=int, default=500, help="Probe training size")
    parser.add_argument("--max-examples", type=int, default=None, help="Max test examples")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    
    # Extra parameters (passed through to dataset)
    parser.add_argument("--extra-format", type=str, help="Dataset format (e.g., 'plausibleqa', 'mcq')")
    parser.add_argument("--extra-dataset_class", type=str, help="Dataset class override")
    parser.add_argument("--extra-dataset_id", type=str, help="Dataset ID override")
    parser.add_argument("--extra-train_split", type=str, help="Train split name")
    parser.add_argument("--extra-eval_split", type=str, help="Eval split name")
    parser.add_argument("--extra-subset", type=str, help="Dataset subset/config")
    parser.add_argument("--extra-split", type=str, help="Split name")
    parser.add_argument("--extra-min_plaus_gap", type=str, help="Min plausibility gap")
    parser.add_argument("--extra-clean_with_correctness", type=str, help="Clean probe with correctness (true/false)")
    
    # Probe extra parameters
    parser.add_argument("--probe-extra-format", type=str)
    parser.add_argument("--probe-extra-dataset_class", type=str)
    parser.add_argument("--probe-extra-dataset_id", type=str)
    parser.add_argument("--probe-extra-train_split", type=str)
    parser.add_argument("--probe-extra-eval_split", type=str)
    parser.add_argument("--probe-extra-subset", type=str)
    
    args = parser.parse_args()
    
    # Load config from file or build from args
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        # Apply overrides
        if args.model:
            config.model_path = args.model
        if args.artifacts_dir != "artifacts":
            config.artifacts_dir = args.artifacts_dir
        if args.plots_dir != "plots":
            config.plots_dir = args.plots_dir
        if args.device != "cuda":
            config.device = args.device
        if args.null_alpha != 1.0:
            config.null_alpha = args.null_alpha
        if args.probe_size != 500:
            config.probe_size = args.probe_size
        if args.max_examples is not None:
            config.max_test_examples = args.max_examples
    else:
        if not all([args.bias_type, args.name, args.model, args.dataset_source]):
            parser.error("Either --config or (--bias-type, --name, --model, --dataset-source) required")
        config = build_config_from_args(args)
    
    logger.info("Experiment: %s", config.name)
    logger.info("Bias type: %s", config.bias_type)
    logger.info("Model: %s", config.model_path)
    logger.info("Dataset: %s", config.dataset_source)
    
    # Get experiment class
    exp_cls = EXPERIMENT_CLASSES.get(config.bias_type)
    if exp_cls is None:
        logger.error("Unknown bias type: %s", config.bias_type)
        sys.exit(1)
    
    # Use appropriate position experiment based on dataset
    if config.bias_type == "position":
        if config.dataset_class in ("position_freeform", "position_freeform_bigbench", "position_freeform_plausibleqa"):
            exp_cls = FreeformPositionBiasExperiment
            logger.info("Using freeform position bias experiment")
        elif config.dataset_class == "position_plausibleqa" or ("plausibleqa" in config.dataset_source.lower() and not config.dataset_class):
            exp_cls = BinaryPositionBiasExperiment
            logger.info("Using binary position bias experiment for PlausibleQA")
    
    # Handle cross-dataset generalization
    probe_source = args.probe_source if hasattr(args, 'probe_source') else None
    if probe_source:
        logger.info("Cross-dataset: probe from %s, eval on %s", probe_source, config.dataset_source)
        # Store probe source in extra for experiment to use
        config.extra["probe_source"] = probe_source
        # Collect probe extras
        probe_extra = {}
        for key, value in vars(args).items():
            if key.startswith("probe_extra_") and value is not None:
                probe_extra[key[12:]] = value
        if probe_extra:
            config.extra["probe_extra"] = probe_extra
    
    # Run experiment
    experiment = exp_cls(config)
    results = experiment.run()
    
    logger.info("\nExperiment complete!")
    logger.info("Artifacts: %s | Plots: %s", config.artifacts_dir, config.plots_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run all bias evaluation experiments across all models and datasets.

Usage:
    python experiments/run_all.py                    # Run everything
    python experiments/run_all.py --filter position  # Only position bias
    python experiments/run_all.py --filter skywork   # Only Skywork model
    python experiments/run_all.py --list             # List all experiments
    python experiments/run_all.py --cross            # Include cross-dataset generalization
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_CONFIG = PROJECT_ROOT / "experiments" / "models.yaml"


def load_registry() -> Dict[str, Any]:
    """Load models and datasets registry."""
    with open(MODELS_CONFIG) as f:
        return yaml.safe_load(f)


def generate_experiment_configs(
    registry: Dict[str, Any],
    filter_str: Optional[str] = None,
    include_cross: bool = False,
) -> List[Dict[str, Any]]:
    """Generate all experiment configurations.
    
    Args:
        registry: Models and datasets registry
        filter_str: Optional filter substring
        include_cross: Whether to include cross-dataset generalization experiments
        
    Returns:
        List of experiment config dicts
    """
    models = registry["models"]
    datasets_by_bias = registry["datasets"]
    
    experiments = []
    
    def _matches_filter(exp_name: str, filt: Optional[str]) -> bool:
        """Return True if exp_name matches the provided filter string.
        
        Supports:
        - Plain substring (legacy): "--filter truthfulqa"
        - Multi-term AND: "--filter 'position truthfulqa'" or "--filter 'position,truthfulqa'"
        - Bias+dataset shorthand: "--filter position_truthfulqa" matches "position_*_truthfulqa"
        """
        if not filt:
            return True
        
        f = filt.strip().lower()
        name = exp_name.lower()
        
        # Multi-term AND (space or comma separated)
        terms = [t for t in f.replace(",", " ").split() if t]
        if len(terms) > 1:
            return all(t in name for t in terms)
        
        # Bias+dataset shorthand: "<bias>_<dataset>" matches "<bias>_*_<dataset>"
        if "_" in f:
            parts = [p for p in f.split("_") if p]
            if len(parts) == 2:
                bias, dataset = parts
                if bias in datasets_by_bias:
                    return name.startswith(bias + "_") and name.endswith("_" + dataset)
        
        # Legacy substring match
        return f in name
    
    for bias_type, datasets in datasets_by_bias.items():
        for model in models:
            for dataset in datasets:
                exp_name = f"{bias_type}_{model['name']}_{dataset['name']}"
                
                # Apply filter
                if not _matches_filter(exp_name, filter_str):
                    continue
                
                experiments.append({
                    "name": exp_name,
                    "bias_type": bias_type,
                    "model_name": model["name"],
                    "model_path": model["path"],
                    "trust_remote_code": model.get("trust_remote_code", True),
                    "dataset_name": dataset["name"],
                    "dataset_source": dataset["source"],
                    "extra": dataset.get("extra", {}),
                    "probe_dataset": dataset["name"],  # Same as eval for standard runs
                })
                
                # Cross-dataset generalization: train probe on one, eval on others
                if include_cross and len(datasets) > 1:
                    for other_dataset in datasets:
                        if other_dataset["name"] != dataset["name"]:
                            cross_name = f"{bias_type}_{model['name']}_{dataset['name']}_to_{other_dataset['name']}"
                            
                            if not _matches_filter(cross_name, filter_str):
                                continue
                            
                            experiments.append({
                                "name": cross_name,
                                "bias_type": bias_type,
                                "model_name": model["name"],
                                "model_path": model["path"],
                                "trust_remote_code": model.get("trust_remote_code", True),
                                "dataset_name": other_dataset["name"],
                                "dataset_source": other_dataset["source"],
                                "extra": other_dataset.get("extra", {}),
                                "probe_dataset": dataset["name"],
                                "probe_source": dataset["source"],
                                "probe_extra": dataset.get("extra", {}),
                                "is_cross": True,
                            })
    
    return experiments


def run_experiment(exp: Dict[str, Any], device: str = "cuda", dry_run: bool = False) -> Dict[str, Any]:
    """Run a single experiment."""
    logger.info("=" * 60)
    logger.info("Running: %s", exp["name"])
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_experiment.py"),
        "--bias-type", exp["bias_type"],
        "--name", exp["name"],
        "--model", exp["model_path"],
        "--dataset-source", exp["dataset_source"],
        "--artifacts-dir", "artifacts",
        "--plots-dir", "plots",
        "--device", device,
    ]
    
    if exp.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    
    # Add extra settings
    for key, value in exp.get("extra", {}).items():
        cmd.extend([f"--extra-{key}", str(value)])
    
    # Cross-dataset: specify probe source
    if exp.get("is_cross"):
        cmd.extend(["--probe-source", exp["probe_source"]])
        for key, value in exp.get("probe_extra", {}).items():
            cmd.extend([f"--probe-extra-{key}", str(value)])
    
    if dry_run:
        logger.info("DRY RUN: %s", " ".join(cmd))
        return {"name": exp["name"], "success": True, "dry_run": True}
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
        return {
            "name": exp["name"],
            "success": result.returncode == 0,
            "returncode": result.returncode,
        }
    except Exception as e:
        logger.error("Failed %s: %s", exp["name"], e)
        return {"name": exp["name"], "success": False, "error": str(e)}


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of all runs."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    n_success = sum(1 for r in results if r.get("success", False))
    n_total = len(results)
    
    logger.info("Total: %d | Success: %d | Failed: %d", n_total, n_success, n_total - n_success)
    
    if n_success < n_total:
        logger.info("\nFailed:")
        for r in results:
            if not r.get("success"):
                logger.info("  - %s", r["name"])
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", type=str, help="Filter experiments by name substring")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--list", action="store_true", help="List experiments without running")
    parser.add_argument("--cross", action="store_true", help="Include cross-dataset generalization")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    registry = load_registry()
    experiments = generate_experiment_configs(registry, args.filter, args.cross)
    
    if not experiments:
        logger.error("No experiments matched filter '%s'", args.filter)
        sys.exit(1)
    
    logger.info("Found %d experiments:", len(experiments))
    for exp in experiments:
        cross_marker = " [CROSS]" if exp.get("is_cross") else ""
        logger.info("  - %s%s", exp["name"], cross_marker)
    
    if args.list:
        return
    
    results = []
    for i, exp in enumerate(experiments, 1):
        logger.info("\n[%d/%d] %s", i, len(experiments), exp["name"])
        result = run_experiment(exp, args.device, args.dry_run)
        results.append(result)
    
    print_summary(results)


if __name__ == "__main__":
    main()

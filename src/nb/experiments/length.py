"""
Length/verbosity bias experiment.

Reports how often a reward model prefers an incorrect response over:
- the concise correct answer
- a verbose correct answer (if available)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.nb.datasets.base import EvalExample
from src.nb.datasets.length import LengthBiasDataset, compute_length_bias_metrics
from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.plotting import create_length_bias_plot

logger = logging.getLogger(__name__)


class LengthBiasExperiment(BiasExperiment):
    """Experiment for evaluating length/verbosity bias.
    
    Compares RM preference between incorrect vs. correct and incorrect vs.
    correct_verbose (when the verbose variant exists).
    """
    
    @property
    def bias_type(self) -> str:
        return "length"
    
    def _create_dataset(self) -> LengthBiasDataset:
        """Create length bias dataset."""
        return LengthBiasDataset(
            source=self.config.dataset_source,
            probe_size=self.config.probe_size,
            split_seed=self.config.split_seed,
            max_test_examples=self.config.max_test_examples,
        )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Compute length bias metrics."""
        return compute_length_bias_metrics(
            rewards=rewards,
            n_examples=len(eval_examples),
        )
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create length bias plot."""
        create_length_bias_plot(
            baseline=results.baseline_metrics,
            nulled=results.nulled_metrics,
            output_path=output_path,
            title=f"Length/Verbosity Bias: {self.config.name}",
            n_examples=results.n_eval_examples,
        )


def run_length_experiment(config_path: Path) -> ExperimentResults:
    """Run length bias experiment from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Experiment results
    """
    config = ExperimentConfig.from_yaml(config_path)
    experiment = LengthBiasExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run length bias experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_length_experiment(args.config)





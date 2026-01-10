"""
Uncertainty/hedging bias experiment.

Tests whether reward models handle uncertainty expressions appropriately.

Supports multiple dataset types:
- uncertainty: PlausibleQA with plausibility-scored distractors
- uncertainty_mcq: MCQ datasets (GSM8K-MC, MMLU) with uniform plausibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from src.nb.datasets.base import EvalExample, ProbeDataset
from src.nb.datasets.uncertainty import (
    UncertaintyBiasDataset,
    UncertaintyMCQDataset,
    compute_uncertainty_metrics,
)
from src.nb.datasets.bigbench import UncertaintyBigBenchDataset
from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.plotting import create_comparison_plot

logger = logging.getLogger(__name__)


# Metric labels for uncertainty plot
UNCERTAINTY_METRIC_LABELS = [
    ("A_C_gt_I", "A: P[C > I]"),
    ("B_C_gt_CU", "B: P[C > C+U]"),
    ("C_IU_gt_I", "C: P[I+U > I]"),
    ("E_CU_gt_I", "E: P[C+U > I]"),
]

# Map dataset class names to classes
UNCERTAINTY_DATASET_CLASSES = {
    "uncertainty": UncertaintyBiasDataset,
    "uncertainty_mcq": UncertaintyMCQDataset,
    "uncertainty_bigbench": UncertaintyBigBenchDataset,
}


class UncertaintyBiasExperiment(BiasExperiment):
    """Experiment for evaluating uncertainty/hedging bias.
    
    Tests preference ordering for response variants:
    - C: Direct correct
    - C+U: Hedged correct  
    - I: Direct incorrect
    - I+U: Hedged incorrect
    
    Ideal ordering: C > C+U > I+U > I
    
    Key metrics:
    - B: P[C > C+U] - does RM penalize uncertainty on correct answers?
    - E: P[C+U > I] - does hedged correct beat confident incorrect?
    """
    
    @property
    def bias_type(self) -> str:
        return "uncertainty"
    
    def _create_dataset(self) -> ProbeDataset:
        """Create uncertainty bias dataset based on config."""
        extra = self.config.extra
        dataset_class_name = extra.get("dataset_class", "uncertainty")
        
        # Also check top-level config for dataset_class
        if hasattr(self.config, "dataset_class") and self.config.dataset_class:
            dataset_class_name = self.config.dataset_class
        
        # If format is "mcq", use the MCQ dataset class
        if extra.get("format") == "mcq":
            dataset_class_name = "uncertainty_mcq"
        
        dataset_cls = UNCERTAINTY_DATASET_CLASSES.get(dataset_class_name, UncertaintyBiasDataset)
        
        logger.info("Using dataset class: %s", dataset_cls.__name__)
        
        if dataset_cls == UncertaintyMCQDataset:
            return UncertaintyMCQDataset(
                source=self.config.dataset_source,
                dataset_id=extra.get("dataset_id", self.config.dataset_source),
                train_split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                subset=extra.get("subset"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_cls == UncertaintyBigBenchDataset:
            return UncertaintyBigBenchDataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            return UncertaintyBiasDataset(
                source=self.config.dataset_source,
                split=extra.get("split", "train"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
                min_plaus_gap=extra.get("min_plaus_gap", 0.0),
            )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Compute uncertainty metrics for low-plausibility incorrect answers."""
        # Map variant names to expected format
        mapped_rewards = {
            "C": rewards.get("C", []),
            "C_U": rewards.get("C_U", []),
            "I_low": rewards.get("I_low", []),
            "I_U_low": rewards.get("I_U_low", []),
        }
        
        return compute_uncertainty_metrics(
            rewards=mapped_rewards,
            n_examples=len(eval_examples),
            plausibility="low",
        )
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create uncertainty bias plot."""
        create_comparison_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics,
            metric_labels=UNCERTAINTY_METRIC_LABELS,
            output_path=output_path,
            title=f"Uncertainty bias: {self.config.name}",
            ylabel="Proportion",
            ylim=(0.0, 1.05),
            n_examples=results.n_eval_examples,
            null_alpha=self.config.null_alpha,
        )
    
    def evaluate_both_plausibilities(self) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics for both high and low plausibility incorrect answers.
        
        Returns:
            Dictionary with 'high' and 'low' keys containing metrics
        """
        if self.model is None:
            self.load_model()
        if self.dataset is None:
            self.load_dataset()
        
        eval_examples = self.dataset.get_eval_examples(self.tokenizer)
        all_texts, text_meta = self._get_all_texts_and_variants(eval_examples)
        
        # Get rewards
        from src.nb.nullbias.probe import get_rewards_with_nulling
        
        baseline_rewards = get_rewards_with_nulling(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=all_texts,
            probe=None,
            batch_size=self.config.batch_size,
            device=self.config.device,
        )
        baseline_organized = self._organize_rewards(baseline_rewards, text_meta, len(eval_examples))
        
        # Compute for both plausibilities
        results = {}
        for plaus in ["low", "high"]:
            mapped_rewards = {
                "C": baseline_organized.get("C", []),
                "C_U": baseline_organized.get("C_U", []),
                f"I_{plaus}": baseline_organized.get(f"I_{plaus}", []),
                f"I_U_{plaus}": baseline_organized.get(f"I_U_{plaus}", []),
            }
            results[plaus] = compute_uncertainty_metrics(
                rewards=mapped_rewards,
                n_examples=len(eval_examples),
                plausibility=plaus,
            )
        
        return results


def run_uncertainty_experiment(config_path: Path) -> ExperimentResults:
    """Run uncertainty experiment from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Experiment results
    """
    config = ExperimentConfig.from_yaml(config_path)
    experiment = UncertaintyBiasExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run uncertainty experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_uncertainty_experiment(args.config)



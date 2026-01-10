"""
RewardBench evaluation experiment.

Bias type: rewardbench

Evaluates accuracy on allenai/reward-bench-2 where an example is counted
correct only if the reward for the correct answer exceeds ALL incorrect
answers. Supports null-space projection to see effect of probes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.nb.datasets.base import EvalExample
from src.nb.datasets.rewardbench import RewardBenchDataset
from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.plotting import create_comparison_plot

logger = logging.getLogger(__name__)


class RewardBenchExperiment(BiasExperiment):
    """Evaluate reward models on allenai/reward-bench-2 with nulling."""

    @property
    def bias_type(self) -> str:
        return "rewardbench"

    def _create_dataset(self) -> RewardBenchDataset:
        extra = self.config.extra
        return RewardBenchDataset(
            source=self.config.dataset_source or "allenai/reward-bench-2",
            split=extra.get("split", "test"),
            probe_size=self.config.probe_size,
            split_seed=self.config.split_seed,
            max_test_examples=self.config.max_test_examples,
        )

    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Accuracy: correct reward must exceed all incorrect rewards."""
        correct = rewards["correct"]
        n = len(correct)
        successes = 0
        for i in range(n):
            # Collect incorrect_j rewards for this example
            example_incorrect = []
            j = 0
            while f"incorrect_{j}" in rewards:
                example_incorrect.append(rewards[f"incorrect_{j}"][i])
                j += 1
            if all(correct[i] > r for r in example_incorrect):
                successes += 1
        return {
            "accuracy": successes / n if n > 0 else 0.0,
            "n_examples": n,
        }

    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Plot baseline vs nulled accuracy."""
        metric_labels = [("accuracy", "Accuracy")]
        create_comparison_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics,
            metric_labels=metric_labels,
            output_path=output_path,
            title=f"RewardBench: {self.config.name}",
            ylabel="Accuracy",
            ylim=(0.0, 1.05),
            n_examples=results.n_eval_examples,
            null_alpha=self.config.null_alpha,
        )


def run_rewardbench_experiment(config_path: Path) -> ExperimentResults:
    config = ExperimentConfig.from_yaml(config_path)
    experiment = RewardBenchExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rewardbench experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_rewardbench_experiment(args.config)



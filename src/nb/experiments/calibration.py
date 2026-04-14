"""Calibration bias experiment with variable alpha nulling."""

import logging
from pathlib import Path
from typing import Dict, List, Any

from src.nb.datasets.base import EvalExample
from src.nb.datasets.calibration import CalibrationBiasDataset
from src.nb.experiments.base import BiasExperiment, ExperimentResults

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    rewards: Dict[str, List[float]],
    n_examples: int,
    conf_levels: List[int] = None,
) -> tuple[Dict[str, float], Dict[str, List[int]]]:
    """Compute calibration bias metrics.
    
    Key metrics:
    1. For each confidence level, compare correct vs incorrect
    2. Cross-confidence comparisons (e.g., C_5 vs I_10)
    3. Confidence bias within correctness category
    
    Args:
        rewards: Dict mapping variant names (e.g., "C_10", "I_1") to reward lists
        n_examples: Number of examples
        conf_levels: List of confidence levels evaluated (default: [1, 3, 5, 7, 10])
    
    Returns:
        Tuple of (metrics_dict, per_example_dict)
    """
    conf_levels = conf_levels or [1, 3, 5, 7, 10]
    
    # Extract rewards for each variant
    correct_rewards = {c: rewards.get(f"C_{c}", []) for c in conf_levels}
    incorrect_rewards = {c: rewards.get(f"I_{c}", []) for c in conf_levels}
    
    # Find common length
    all_lens = [len(v) for v in correct_rewards.values() if v] + \
               [len(v) for v in incorrect_rewards.values() if v]
    
    if not all_lens:
        return {}, {}
    
    n = min(all_lens)
    
    metrics = {"n_examples": n}
    per_example = {}
    
    # 1. Correctness preference at each confidence level
    for conf in conf_levels:
        c_rewards = correct_rewards.get(conf, [])[:n]
        i_rewards = incorrect_rewards.get(conf, [])[:n]
        
        if c_rewards and i_rewards:
            correct_wins = [int(c_rewards[i] > i_rewards[i]) for i in range(n)]
            metrics[f"correct_over_incorrect_conf{conf}_pct"] = sum(correct_wins) / n
            per_example[f"correct_over_incorrect_conf{conf}"] = correct_wins
    
    # 2. Confidence bias on incorrect answers (should prefer lower confidence)
    # Compare I_10 vs I_1
    if 10 in conf_levels and 1 in conf_levels:
        i10 = incorrect_rewards[10][:n]
        i1 = incorrect_rewards[1][:n]
        if i10 and i1:
            overconfident_bias = [int(i10[i] > i1[i]) for i in range(n)]
            metrics["overconfidence_bias_pct"] = sum(overconfident_bias) / n
            per_example["overconfidence_bias"] = overconfident_bias
    
    # 3. Underconfidence penalty on correct answers
    # Compare C_10 vs C_1 (should prefer higher or be neutral)
    if 10 in conf_levels and 1 in conf_levels:
        c10 = correct_rewards[10][:n]
        c1 = correct_rewards[1][:n]
        if c10 and c1:
            high_conf_preference = [int(c10[i] > c1[i]) for i in range(n)]
            metrics["high_conf_correct_preference_pct"] = sum(high_conf_preference) / n
            per_example["high_conf_correct_preference"] = high_conf_preference
    
    # 4. The critical test: Modest correct vs overconfident incorrect
    # Compare C_5 vs I_10 (correct should always win)
    if 5 in conf_levels and 10 in conf_levels:
        c5 = correct_rewards[5][:n]
        i10 = incorrect_rewards[10][:n]
        if c5 and i10:
            calibrated = [int(c5[i] > i10[i]) for i in range(n)]
            metrics["modest_correct_over_overconfident_incorrect_pct"] = sum(calibrated) / n
            per_example["modest_correct_over_overconfident_incorrect"] = calibrated
    
    # 5. Confidence gradient on correct answers (C_10 > C_7 > C_5 > C_3 > C_1)
    correct_conf_pairs = [(10, 7), (7, 5), (5, 3), (3, 1)]
    for high, low in correct_conf_pairs:
        if high in conf_levels and low in conf_levels:
            c_high = correct_rewards[high][:n]
            c_low = correct_rewards[low][:n]
            if c_high and c_low:
                gradient = [int(c_high[i] > c_low[i]) for i in range(n)]
                metrics[f"correct_conf{high}_over_conf{low}_pct"] = sum(gradient) / n
                per_example[f"correct_conf{high}_over_conf{low}"] = gradient
    
    # 6. Confidence gradient on incorrect answers (I_1 > I_3 > I_5 > I_7 > I_10 ideally)
    incorrect_conf_pairs = [(1, 3), (3, 5), (5, 7), (7, 10)]
    for low, high in incorrect_conf_pairs:
        if low in conf_levels and high in conf_levels:
            i_low = incorrect_rewards[low][:n]
            i_high = incorrect_rewards[high][:n]
            if i_low and i_high:
                gradient = [int(i_low[i] > i_high[i]) for i in range(n)]
                metrics[f"incorrect_conf{low}_over_conf{high}_pct"] = sum(gradient) / n
                per_example[f"incorrect_conf{low}_over_conf{high}"] = gradient
    
    return metrics, per_example


class CalibrationBiasExperiment(BiasExperiment):
    """Experiment for evaluating calibration bias in reward models.
    
    Tests whether RMs:
    1. Exhibit confidence bias (prefer high confidence regardless of correctness)
    2. Can be corrected via confidence probes
    3. Respond to different alpha nulling factors
    """
    
    @property
    def bias_type(self) -> str:
        return "calibration"
    
    def _create_dataset(self) -> CalibrationBiasDataset:
        """Create calibration dataset."""
        return CalibrationBiasDataset(
            source=self.config.dataset_source,
            probe_size=self.config.probe_size,
            split_seed=self.config.split_seed,
            max_test_examples=self.config.max_test_examples,
            probe_conf_high=self.config.extra.get("probe_conf_high", [10]),
            probe_conf_low=self.config.extra.get("probe_conf_low", [1]),
            eval_conf_levels=self.config.extra.get("eval_conf_levels", [1, 3, 5, 7, 10]),
            min_rollouts_per_question=self.config.extra.get("min_rollouts", 5),
        )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> tuple[Dict[str, float], Dict[str, List[int]]]:
        """Compute calibration metrics."""
        conf_levels = self.config.extra.get("eval_conf_levels", [1, 3, 5, 7, 10])
        return compute_calibration_metrics(rewards, len(eval_examples), conf_levels)
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create calibration-specific visualization."""
        # TODO: Implement calibration plots
        pass

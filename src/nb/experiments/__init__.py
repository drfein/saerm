"""
Experiment orchestration for bias evaluation.

Each experiment follows the same pattern:
1. Load dataset (probe + eval splits)
2. Build probe direction from probe split
3. Evaluate baseline (no projection)
4. Evaluate with null-space projection
5. Plot results with error bars
"""

from src.nb.experiments.base import BiasExperiment, ExperimentConfig
from src.nb.experiments.plotting import PlotStyle, create_comparison_plot, create_position_bias_plot
from src.nb.experiments.rewardbench import RewardBenchExperiment

__all__ = [
    "BiasExperiment",
    "ExperimentConfig",
    "PlotStyle",
    "create_comparison_plot",
    "create_position_bias_plot",
    "RewardBenchExperiment",
]





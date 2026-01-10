"""
Dataset definitions for bias evaluation experiments.

Each dataset provides:
- probe_train: 500 examples for building the probe direction
- probe_test: remaining examples for evaluation

Supports multiple bias types and dataset sources:
- Length: GSM8K solutions
- Sycophancy: Preference pairs, PlausibleQA, GSM8K-MC, MMLU, BigBench
- Uncertainty: PlausibleQA, GSM8K-MC, MMLU, BigBench
- Position: GSM8K-MC, MMLU, BigBench
"""

from src.nb.datasets.base import DatasetRegistry, ProbeDataset
from src.nb.datasets.length import LengthBiasDataset
from src.nb.datasets.sycophancy import (
    SycophancyBiasDataset,
    SycophancyMCQDataset,
    SycophancyPlausibleQADataset,
)
from src.nb.datasets.uncertainty import UncertaintyBiasDataset, UncertaintyMCQDataset
from src.nb.datasets.position import PositionBiasDataset
from src.nb.datasets.rewardbench import RewardBenchDataset
from src.nb.datasets.bigbench import (
    SycophancyBigBenchDataset,
    UncertaintyBigBenchDataset,
    PositionBigBenchDataset,
    CorrectnessBigBenchDataset,
    CorrectnessPositionBigBenchDataset,
)

__all__ = [
    "DatasetRegistry",
    "ProbeDataset",
    # Length
    "LengthBiasDataset",
    # Sycophancy
    "SycophancyBiasDataset",
    "SycophancyMCQDataset",
    "SycophancyPlausibleQADataset",
    # Uncertainty
    "UncertaintyBiasDataset",
    "UncertaintyMCQDataset",
    # Position
    "PositionBiasDataset",
    # RewardBench
    "RewardBenchDataset",
    # BigBench
    "SycophancyBigBenchDataset",
    "UncertaintyBigBenchDataset",
    "PositionBigBenchDataset",
    "CorrectnessBigBenchDataset",
    "CorrectnessPositionBigBenchDataset",
]

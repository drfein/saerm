"""
Nullbias (NB) module for reward model bias evaluation and mitigation.

This module provides a unified interface for:
- Building probe directions from contrastive datasets
- Null-space projection to debias reward model hidden states
- Evaluating multiple bias types: length, sycophancy, uncertainty, position
"""

from src.nb.nullbias import build_probe_direction
from src.nb.datasets import DatasetRegistry, ProbeDataset

__all__ = [
    "build_probe_direction",
    "DatasetRegistry",
    "ProbeDataset",
]

"""
Null-space projection for reward model debiasing.

Provides efficient projection of hidden states onto the null space of probe directions.
"""

from src.nb.nullbias.probe import build_probe_direction, get_embeddings

__all__ = [
    "build_probe_direction",
    "get_embeddings",
]

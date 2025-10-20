from __future__ import annotations

import torch
from torch.utils.data import Dataset


class EmbeddingTensorDataset(Dataset):
    """Wraps a tensor of token embeddings for SAE training."""

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise ValueError("Expected 2D tensor [examples, features]")
        self._tensor = tensor

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._tensor[index]

from __future__ import annotations

import torch
from torch import nn


class BatchTopKSAE(nn.Module):
    """Batch Top-K sparse autoencoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k_active: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if k_active <= 0:
            raise ValueError("k_active must be positive")
        if k_active > hidden_dim:
            raise ValueError("k_active cannot exceed hidden_dim")
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.activation = _build_activation(activation)
        self.k_active = k_active

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_activations = self.encoder(x)
        activated = self.activation(pre_activations)
        sparse_codes = self._batch_topk(activated)
        reconstruction = self.decoder(sparse_codes)
        return reconstruction, sparse_codes

    def _batch_topk(self, activations: torch.Tensor) -> torch.Tensor:
        if self.k_active >= activations.shape[1]:
            return activations
        if activations.requires_grad:
            topk = activations.topk(self.k_active, dim=1).indices
            mask = torch.zeros_like(activations)
            mask.scatter_(1, topk, 1.0)
            return activations * mask
        # eval path: reuse same logic but avoid building graph
        with torch.no_grad():
            topk = activations.topk(self.k_active, dim=1).indices
            mask = torch.zeros_like(activations)
            mask.scatter_(1, topk, 1.0)
        return activations * mask


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation {name}")

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .models import BatchTopKSAE


class SAEFeatureExtractor:
    """Load a trained Batch-Top-K SAE checkpoint to produce sparse codes."""

    def __init__(
        self,
        checkpoint_path: str | bytes,
        *,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k_active: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        state = torch.load(checkpoint_path, map_location=self._device)
        state_dict, meta = _extract_state_and_meta(state)

        inferred_input = input_dim or meta.get("input_dim")
        inferred_hidden = hidden_dim or meta.get("hidden_dim") or meta.get("num_neurons")
        inferred_k = k_active or meta.get("k_active")

        if inferred_input is None or inferred_hidden is None or inferred_k is None:
            raise ValueError("Checkpoint does not contain sufficient metadata to instantiate the SAE")

        self._model = BatchTopKSAE(inferred_input, inferred_hidden, inferred_k)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def transform(self, tensor: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for idx in range(0, tensor.shape[0], batch_size):
                batch = tensor[idx: idx + batch_size].to(self._device)
                _, codes = self._model(batch)
                outputs.append(codes.cpu())
        return torch.cat(outputs, dim=0)


def _extract_state_and_meta(state: Any) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        meta = {k: v for k, v in state.items() if k != "state_dict"}
        return state_dict, meta
    if isinstance(state, dict):
        return state, {}
    raise TypeError("Unexpected checkpoint format")

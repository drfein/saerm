from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib


@dataclass
class _BTLinearState:
    weights: np.ndarray


class LinearBTHead(PredictionHead):
    """Linear reward head trained with Bradley-Terry (pairwise logistic) loss."""

    def __init__(
        self,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        max_epochs: int = 1000,
        tol: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.tol = tol
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ PredictionHead overrides
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError("LinearBTHead.fit is not supported. Use fit_pairwise instead.")

    def fit_pairwise(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        weight_chosen: float = 1.0,
        weight_rejected: float = 1.0,
    ) -> Dict[str, float]:
        chosen = torch.as_tensor(chosen_features, dtype=torch.float32, device=self.device)
        rejected = torch.as_tensor(rejected_features, dtype=torch.float32, device=self.device)
        if chosen.shape != rejected.shape:
            raise ValueError("Chosen and rejected features must share the same shape for BT training")
        diff = weight_chosen * chosen - weight_rejected * rejected

        weights = torch.zeros(diff.shape[1], dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights], lr=self.lr, weight_decay=self.weight_decay)

        if sample_weights is not None:
            weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)
        else:
            weight_tensor = None

        previous_loss: Optional[float] = None
        current_loss: float = float("nan")
        epochs_run = 0
        for epoch in range(self.max_epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = diff @ weights
            loss = F.softplus(-logits)  # -log(sigmoid(logits))
            if weight_tensor is not None:
                weight_sum = torch.clamp(weight_tensor.sum(), min=1e-12)
                loss = (loss * weight_tensor).sum() / weight_sum
            else:
                loss = loss.mean()
            loss.backward()
            optimizer.step()

            current_loss = float(loss.item())
            epochs_run = epoch + 1
            if previous_loss is not None and abs(previous_loss - current_loss) < self.tol:
                break
            previous_loss = current_loss

        self._weights = weights.detach().cpu().numpy()
        return {"epochs": float(epochs_run), "loss": float(previous_loss or current_loss)}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Head must be trained before calling predict()")
        return np.asarray(features @ self._weights, dtype=np.float32)

    def save(self, path: str) -> None:
        if self._weights is None:
            raise RuntimeError("Cannot save an untrained LinearBTHead")
        save_joblib(_BTLinearState(weights=self._weights), path)

    @classmethod
    def load(cls, path: str) -> "LinearBTHead":
        state: _BTLinearState = load_joblib(path)
        instance = cls()
        instance._weights = state.weights
        return instance


HeadFactory.register("linear_bt", LinearBTHead)

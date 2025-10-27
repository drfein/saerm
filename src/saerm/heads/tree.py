from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib


class DecisionTreeHead(PredictionHead):
    """Decision tree head supporting both supervised and pairwise training."""

    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 1) -> None:
        self._tree_kwargs = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
        self._model: DecisionTreeClassifier | DecisionTreeRegressor | None = None
        self._mode: str | None = None
        self._reference: np.ndarray | None = None
        self._pairwise_weights: tuple[float, float] = (1.0, 1.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features = self._ensure_2d(features)
        targets = np.asarray(targets, dtype=np.float32)
        regressor = DecisionTreeRegressor(**self._tree_kwargs)
        regressor.fit(features, targets)
        self._model = regressor
        self._mode = "supervised"
        self._reference = None

    def fit_pairwise(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        weight_chosen: float = 1.0,
        weight_rejected: float = 1.0,
    ) -> dict[str, float] | None:
        chosen = self._ensure_2d(chosen_features)
        rejected = self._ensure_2d(rejected_features)
        if chosen.shape != rejected.shape:
            raise ValueError("Chosen and rejected feature tensors must share the same shape")
        num_pairs = chosen.shape[0]
        if num_pairs == 0:
            raise ValueError("Received empty feature batches for pairwise training")

        base_weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else np.ones(num_pairs, dtype=np.float32)
        )
        base_weights = np.clip(base_weights, a_min=0.0, a_max=None)
        if not np.any(base_weights):
            base_weights = np.ones_like(base_weights)

        chosen_weights = base_weights * float(weight_chosen)
        rejected_weights = base_weights * float(weight_rejected)
        self._reference = self._compute_reference(chosen, rejected, chosen_weights, rejected_weights)

        deltas = float(weight_chosen) * chosen - float(weight_rejected) * rejected
        X = np.concatenate([deltas, -deltas], axis=0)
        y = np.concatenate(
            [np.ones(num_pairs, dtype=np.int64), np.zeros(num_pairs, dtype=np.int64)],
            axis=0,
        )
        doubled_weights = np.concatenate([base_weights, base_weights], axis=0)

        classifier = DecisionTreeClassifier(**self._tree_kwargs)
        classifier.fit(X, y, sample_weight=doubled_weights)
        self._model = classifier
        self._mode = "pairwise"
        self._pairwise_weights = (float(weight_chosen), float(weight_rejected))

        predictions = classifier.predict(X)
        accuracy = float(np.mean(predictions == y))
        return {"pairwise_accuracy": accuracy}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None or self._mode is None:
            raise RuntimeError("DecisionTreeHead must be fitted before calling predict")
        features = self._ensure_2d(features)
        if self._mode == "pairwise":
            if self._reference is None:
                raise RuntimeError("Pairwise head missing reference features for prediction")
            deltas = (
                self._pairwise_weights[0] * features
                - self._pairwise_weights[1] * self._reference[np.newaxis, :]
            )
            probs = self._model.predict_proba(deltas)  # type: ignore[arg-type]
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1]
            # Fall back to hard predictions if probability estimates are unavailable.
            return self._model.predict(deltas)  # type: ignore[arg-type]
        return self._model.predict(features)  # type: ignore[arg-type]

    def save(self, path: str) -> None:
        state = {
            "tree_kwargs": self._tree_kwargs,
            "mode": self._mode,
            "model": self._model,
            "reference": self._reference,
            "pairwise_weights": self._pairwise_weights,
        }
        save_joblib(state, path)

    @classmethod
    def load(cls, path: str) -> "DecisionTreeHead":
        payload = load_joblib(path)
        instance = cls(**payload.get("tree_kwargs", {})) if isinstance(payload, dict) else cls()
        if isinstance(payload, dict):
            instance._model = payload.get("model")
            instance._mode = payload.get("mode")
            instance._reference = payload.get("reference")
            instance._pairwise_weights = tuple(payload.get("pairwise_weights", (1.0, 1.0)))  # type: ignore[arg-type]
        else:
            # Backwards compatibility: assume a supervised regressor was stored directly.
            instance._model = payload
            instance._mode = "supervised"
            instance._reference = None
            instance._pairwise_weights = (1.0, 1.0)
        return instance

    @staticmethod
    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        return arr

    @staticmethod
    def _compute_reference(
        chosen: np.ndarray,
        rejected: np.ndarray,
        chosen_weights: np.ndarray,
        rejected_weights: np.ndarray,
    ) -> np.ndarray:
        chosen_sum = (chosen * chosen_weights[:, None]).sum(axis=0)
        rejected_sum = (rejected * rejected_weights[:, None]).sum(axis=0)
        total_weight = float(np.sum(chosen_weights) + np.sum(rejected_weights))
        if total_weight <= 0.0:
            return np.mean(np.concatenate([chosen, rejected], axis=0), axis=0)
        return (chosen_sum + rejected_sum) / total_weight


HeadFactory.register("decision_tree", DecisionTreeHead)

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import xgboost as xgb

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib


class XGBoostHead(PredictionHead):
    """XGBoost-based head supporting supervised and pairwise training."""

    def __init__(self, params: dict | None = None, num_rounds: int = 200) -> None:
        defaults: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "lambda": 1.0,
        }
        if params:
            defaults.update(params)
        self._base_params = defaults
        self._num_rounds = num_rounds
        self._booster: xgb.Booster | None = None
        self._mode: str | None = None
        self._reference: np.ndarray | None = None
        self._pairwise_weights: tuple[float, float] = (1.0, 1.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        features = self._ensure_2d(features)
        targets = np.asarray(targets, dtype=np.float32)
        params = dict(self._base_params)
        params.setdefault("objective", "reg:squarederror")
        dtrain = xgb.DMatrix(features, label=targets)
        self._booster = xgb.train(params, dtrain, num_boost_round=self._num_rounds)
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
            raise ValueError("Chosen and rejected features must have the same shape")
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

        params = dict(self._base_params)
        params.setdefault("objective", "binary:logistic")
        dtrain = xgb.DMatrix(X, label=y, weight=doubled_weights)
        self._booster = xgb.train(params, dtrain, num_boost_round=self._num_rounds)
        self._mode = "pairwise"
        self._pairwise_weights = (float(weight_chosen), float(weight_rejected))

        predictions = (self._booster.predict(dtrain) > 0.5).astype(np.int64)
        accuracy = float(np.mean(predictions == y))
        return {"pairwise_accuracy": accuracy}

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._booster is None or self._mode is None:
            raise RuntimeError("XGBoost head must be trained before prediction")
        features = self._ensure_2d(features)
        if self._mode == "pairwise":
            if self._reference is None:
                raise RuntimeError("Pairwise mode requires stored reference features")
            deltas = (
                self._pairwise_weights[0] * features
                - self._pairwise_weights[1] * self._reference[np.newaxis, :]
            )
            dmatrix = xgb.DMatrix(deltas)
            return self._booster.predict(dmatrix)
        dmatrix = xgb.DMatrix(features)
        return self._booster.predict(dmatrix)

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("Head not trained")
        state = {
            "params": self._base_params,
            "num_rounds": self._num_rounds,
            "mode": self._mode,
            "reference": self._reference,
            "pairwise_weights": self._pairwise_weights,
            "booster": self._booster.save_raw(),
        }
        save_joblib(state, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostHead":
        state: dict | None = None
        try:
            payload = load_joblib(path)
            if isinstance(payload, dict) and "booster" in payload:
                state = payload
        except Exception:
            state = None

        if state is None:
            # Fallback to legacy format saved via Booster.save_model.
            instance = cls()
            booster = xgb.Booster()
            booster.load_model(path)
            instance._booster = booster
            instance._mode = "supervised"
            return instance

        instance = cls(params=state.get("params"), num_rounds=state.get("num_rounds", 200))
        booster = xgb.Booster()
        booster.load_model(bytearray(state["booster"]))
        instance._booster = booster
        instance._mode = state.get("mode")
        instance._reference = state.get("reference")
        pairwise_weights = state.get("pairwise_weights", (1.0, 1.0))
        instance._pairwise_weights = (float(pairwise_weights[0]), float(pairwise_weights[1]))
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


HeadFactory.register("xgboost", XGBoostHead)

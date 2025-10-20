from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib


class LinearHead(PredictionHead):
    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=alpha)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def save(self, path: str) -> None:
        save_joblib(self._model, path)

    @classmethod
    def load(cls, path: str) -> "LinearHead":
        instance = cls()
        instance._model = load_joblib(path)
        return instance


HeadFactory.register("linear", LinearHead)

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib


class DecisionTreeHead(PredictionHead):
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 1) -> None:
        self._model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def save(self, path: str) -> None:
        save_joblib(self._model, path)

    @classmethod
    def load(cls, path: str) -> "DecisionTreeHead":
        instance = cls()
        instance._model = load_joblib(path)
        return instance


HeadFactory.register("decision_tree", DecisionTreeHead)

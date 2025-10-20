from __future__ import annotations

import numpy as np
import xgboost as xgb

from .base import HeadFactory, PredictionHead


class XGBoostHead(PredictionHead):
    def __init__(self, params: dict | None = None, num_rounds: int = 200) -> None:
        default_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "lambda": 1.0,
        }
        if params:
            default_params.update(params)
        self._params = default_params
        self._num_rounds = num_rounds
        self._booster: xgb.Booster | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        dtrain = xgb.DMatrix(features, label=targets)
        self._booster = xgb.train(self._params, dtrain, num_boost_round=self._num_rounds)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Head not trained")
        dmatrix = xgb.DMatrix(features)
        return self._booster.predict(dmatrix)

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("Head not trained")
        self._booster.save_model(path)

    @classmethod
    def load(cls, path: str) -> "XGBoostHead":
        instance = cls()
        booster = xgb.Booster()
        booster.load_model(path)
        instance._booster = booster
        return instance


HeadFactory.register("xgboost", XGBoostHead)

from __future__ import annotations

import numpy as np

from .base import HeadFactory, PredictionHead, load_joblib, save_joblib

try:
    from pygam import LinearGAM
except ImportError as exc:  # pragma: no cover
    LinearGAM = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GAMHead(PredictionHead):
    def __init__(self, lam: float = 0.6, max_iter: int = 200) -> None:
        if LinearGAM is None:
            raise ImportError("pygam is required for GAMHead") from _IMPORT_ERROR
        self._model = LinearGAM(lam=lam, max_iter=max_iter)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._model.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def save(self, path: str) -> None:
        save_joblib(self._model, path)

    @classmethod
    def load(cls, path: str) -> "GAMHead":
        instance = cls.__new__(cls)
        instance._model = load_joblib(path)
        return instance


HeadFactory.register("gam", GAMHead)

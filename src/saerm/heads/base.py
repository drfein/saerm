from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Type

import joblib
import numpy as np


class PredictionHead(abc.ABC):
    """Interface for downstream reward prediction heads."""

    @abc.abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "PredictionHead":
        raise NotImplementedError


class HeadFactory:
    _registry: Dict[str, Type[PredictionHead]] = {}

    @classmethod
    def register(cls, name: str, head_cls: Type[PredictionHead]) -> None:
        cls._registry[name] = head_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> PredictionHead:
        if name not in cls._registry:
            raise KeyError(f"Unknown head type {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def load(cls, name: str, path: str) -> PredictionHead:
        if name not in cls._registry:
            raise KeyError(f"Unknown head type {name}")
        loader = getattr(cls._registry[name], "load")
        return loader(path)

    @classmethod
    def available(cls) -> Dict[str, Type[PredictionHead]]:
        return dict(cls._registry)


def save_joblib(obj: Any, path: str) -> None:
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    return joblib.load(path)

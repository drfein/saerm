from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Mapping, MutableMapping, Optional, Type, TypeVar


class BaseRewardModel(ABC):
    """Abstract base class for reward models.

    Subclasses should implement `reward`, `train`, and `load`.
    """

    # Optional human-readable model type identifier (e.g., "btrm").
    model_type: ClassVar[Optional[str]] = None

    @abstractmethod
    def reward(self, body: str, prompt: Optional[str] = None) -> float:
        """Compute a reward for a given body with an optional prompt.

        Args:
            body: The main text to evaluate.
            prompt: Optional prompt or context used to condition the reward.

        Returns:
            A scalar reward value.
        """

    @abstractmethod
    def train(self, dataset: Any) -> None:
        """Train the reward model on the provided dataset.

        Args:
            dataset: An object providing training samples. The exact type is
                left to concrete implementations.
        """

    @classmethod
    @abstractmethod
    def load(cls: Type["BaseRewardModel"], path: str, config: Mapping[str, Any]) -> "BaseRewardModel":
        """Load a model instance from a path and configuration.

        Implementations may load weights, tokenizers, or other artifacts as
        needed.

        Args:
            path: Filesystem path to model artifacts.
            config: Configuration mapping. Must include fields required by the
                implementation; the factory also expects a "type" key.

        Returns:
            A loaded model instance.
        """


# Global registry mapping model type -> subclass
_REWARD_MODEL_REGISTRY: MutableMapping[str, Type[BaseRewardModel]] = {}


def register_reward_model(model_type: str):
    """Class decorator to register a reward model subclass by type name."""

    def _decorator(cls: Type[BaseRewardModel]) -> Type[BaseRewardModel]:
        if not issubclass(cls, BaseRewardModel):
            raise TypeError("Registered class must inherit from BaseRewardModel")
        if model_type in _REWARD_MODEL_REGISTRY:
            raise ValueError(f"Reward model type already registered: {model_type}")
        _REWARD_MODEL_REGISTRY[model_type] = cls
        # Optionally set the class-level identifier if not set
        if getattr(cls, "model_type", None) in (None, ""):
            try:
                cls.model_type = model_type  # type: ignore[assignment]
            except Exception:
                # Best-effort assignment; non-fatal if class forbids it
                pass
        return cls

    return _decorator


def load_reward_model(path: str, config: Mapping[str, Any]) -> BaseRewardModel:
    """Factory: load a reward model by type.

    The `config` mapping must contain a `type` key that selects the registered
    implementation. Remaining config entries are passed directly to the
    subclass `load` method.
    """

    model_type = config.get("type")
    if not isinstance(model_type, str) or model_type == "":
        raise ValueError("config['type'] must be a non-empty string")

    cls = _REWARD_MODEL_REGISTRY.get(model_type)
    if cls is None:
        available = ", ".join(sorted(_REWARD_MODEL_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown reward model type '{model_type}'. Available: {available}")

    return cls.load(path=path, config=config)


__all__ = [
    "BaseRewardModel",
    "load_reward_model",
    "register_reward_model",
]



from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PreferencePair:
    """A single preference training example with chosen and rejected texts.

    This minimal container is intentionally framework-agnostic so that it can
    be adapted for different model/training stacks without coupling.
    """

    chosen: str
    rejected: str


class BasePreferenceDataset(ABC):
    """Abstract base for pairwise preference datasets.

    Concrete subclasses must load pairs and expose train/test splits based on
    construction arguments.
    """

    @abstractmethod
    def train(self) -> Sequence[PreferencePair]:
        """Return the training split as a sequence of pairs."""

    @abstractmethod
    def test(self) -> Sequence[PreferencePair]:
        """Return the test/evaluation split as a sequence of pairs."""

    def as_text_tuples(self, split: str = "train") -> List[Tuple[str, str]]:
        """Convenience: get (chosen, rejected) tuples for a split.

        Args:
            split: "train" or "test".
        """

        data = self.train() if split == "train" else self.test()
        return [(p.chosen, p.rejected) for p in data]


__all__ = [
    "PreferencePair",
    "BasePreferenceDataset",
]



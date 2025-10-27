from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List

from datasets import Dataset, IterableDataset


@dataclass
class PreferencePair:
    dataset: str
    chosen: List[dict]
    rejected: List[dict]


def iter_preference_pairs(
    dataset: Dataset | IterableDataset,
    dataset_name: str,
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
) -> Iterator[PreferencePair]:
    for row in dataset:
        yield PreferencePair(
            dataset=dataset_name,
            chosen=_ensure_messages(row[chosen_field]),
            rejected=_ensure_messages(row[rejected_field]),
        )


def _ensure_messages(value: Any) -> List[dict]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return [{"role": "assistant", "content": str(value)}]

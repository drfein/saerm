from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import Dataset, IterableDataset


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, str]


def iter_preference_pairs(
    dataset: Dataset | IterableDataset,
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    metadata_fields: Optional[List[str]] = None,
) -> Iterator[PreferencePair]:
    metadata_fields = metadata_fields or []
    for row in dataset:
        metadata = {field: row[field] for field in metadata_fields if field in row}
        yield PreferencePair(
            prompt=row[prompt_field],
            chosen=row[chosen_field],
            rejected=row[rejected_field],
            metadata=metadata,
        )

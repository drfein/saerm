from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import Dataset, IterableDataset

from .preferences import PreferencePair


@dataclass
class SkyworkFields:
    prompt: str = "prompt"
    chosen: str = "chosen"
    rejected: str = "rejected"
    metadata: tuple[str, ...] = ("category", "source")


def iter_skywork_pairs(
    dataset: Dataset | IterableDataset,
    fields: SkyworkFields | None = None,
) -> Iterator[PreferencePair]:
    mapping = fields or SkyworkFields()
    for row in dataset:
        metadata = {field: row[field] for field in mapping.metadata if field in row}
        yield PreferencePair(
            prompt=row[mapping.prompt],
            chosen=row[mapping.chosen],
            rejected=row[mapping.rejected],
            metadata=metadata,
        )

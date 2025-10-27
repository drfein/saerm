from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import Dataset, IterableDataset

from .preferences import PreferencePair, _ensure_messages


@dataclass
class SkyworkFields:
    dataset: str = "skywork"
    chosen: str = "chosen"
    rejected: str = "rejected"


def iter_skywork_pairs(
    dataset: Dataset | IterableDataset,
    fields: SkyworkFields | None = None,
) -> Iterator[PreferencePair]:
    mapping = fields or SkyworkFields()
    for row in dataset:
        yield PreferencePair(
            dataset=mapping.dataset,
            chosen=_ensure_messages(row[mapping.chosen]),
            rejected=_ensure_messages(row[mapping.rejected]),
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from datasets import Dataset, IterableDataset


@dataclass
class RewardBenchExample:
    prompt: str
    responses: Dict[str, str]
    label: str
    metadata: Dict[str, str]


def iter_reward_bench_examples(
    dataset: Dataset | IterableDataset,
    prompt_field: str = "prompt",
    response_fields: Optional[Dict[str, str]] = None,
    label_field: str = "label",
    metadata_fields: Optional[list[str]] = None,
) -> Iterator[RewardBenchExample]:
    response_fields = response_fields or {
        "response_a": "response_a",
        "response_b": "response_b",
    }
    metadata_fields = metadata_fields or []
    for row in dataset:
        responses = {alias: row[source] for alias, source in response_fields.items() if source in row}
        metadata = {field: row[field] for field in metadata_fields if field in row}
        yield RewardBenchExample(
            prompt=row[prompt_field],
            responses=responses,
            label=row[label_field],
            metadata=metadata,
        )

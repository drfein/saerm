from __future__ import annotations

from typing import Dict, Optional

from datasets import Dataset, IterableDataset, load_dataset

from ..config import DatasetConfig, ExperimentConfig

DatasetLike = Dataset | IterableDataset


def _load_dataset(config: DatasetConfig, split: str) -> DatasetLike:
    args = [config.name]
    if config.subset:
        args.append(config.subset)
    kwargs = {
        "split": split,
        "revision": config.revision,
        "streaming": config.streaming,
        "cache_dir": config.cache_dir,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    ds = load_dataset(*args, **kwargs)
    return _rename_fields(ds, config.field_mapping)


def _rename_fields(ds: DatasetLike, mapping: Dict[str, str]) -> DatasetLike:
    if not mapping:
        return ds
    if isinstance(ds, Dataset):
        applicable = {src: dest for src, dest in mapping.items() if src in ds.column_names}
        return ds.rename_columns(applicable) if applicable else ds
    if hasattr(ds, "column_names"):
        applicable = {src: dest for src, dest in mapping.items() if src in ds.column_names}
        return ds.rename_columns(applicable) if applicable else ds
    return ds


class DatasetManager:
    """Loads datasets defined in the experiment configuration."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._cache: Dict[str, Dict[str, DatasetLike]] = {}

    def get(self, dataset_key: str, split: Optional[str] = None) -> DatasetLike:
        if dataset_key not in self._config.datasets:
            raise KeyError(f"Unknown dataset key {dataset_key}")
        cfg = self._config.datasets[dataset_key]
        target_split = split or cfg.split
        dataset_store = self._cache.setdefault(dataset_key, {})
        if target_split not in dataset_store:
            dataset_store[target_split] = _load_dataset(cfg, target_split)
        return dataset_store[target_split]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StorageConfig:
    """Filesystem storage configuration."""

    root_url: str
    embeddings_dir: str = "embeddings"
    sae_dir: str = "sae"
    heads_dir: str = "heads"

    def root_path(self) -> Path:
        return Path(self.root_url).expanduser()


@dataclass
class DatasetConfig:
    """Represents a dataset to be loaded via datasets.load_dataset."""

    name: str
    subset: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    streaming: bool = False
    cache_dir: Optional[str] = None
    field_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingJobConfig:
    """Configuration for generating and caching embeddings."""

    job_id: str
    model: str
    layer: str | int
    dataset: str
    batch_size: int = 16
    max_examples: Optional[int] = None
    tokenizer: Optional[str] = None
    prompt_field: str = "prompt"
    response_field: Optional[str] = None


@dataclass
class SAETrainingConfig:
    """Configuration for training an SAE on a cached embedding dataset."""

    job_id: str
    embedding_job: str
    hidden_size: int
    k_active: int
    l1_coef: float
    learning_rate: float
    steps: int
    batch_size: int
    device: str = "cuda"
    checkpoint_interval: int = 1000
    log_interval: int = 50
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    min_lr_scale: float = 0.05
    use_cosine_decay: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None


@dataclass
class HeadTrainingConfig:
    """Configuration for fitting a prediction head on top of SAE features."""

    job_id: str
    embedding_job: str
    sae_job: Optional[str]
    dataset: str
    target_field: str
    head_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    train_split: str = "train"
    eval_split: Optional[str] = "validation"


@dataclass
class ExperimentConfig:
    storage: StorageConfig
    datasets: Dict[str, DatasetConfig]
    embedding_jobs: List[EmbeddingJobConfig] = field(default_factory=list)
    sae_jobs: List[SAETrainingConfig] = field(default_factory=list)
    head_jobs: List[HeadTrainingConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        storage = StorageConfig(**data["storage"])
        datasets: Dict[str, DatasetConfig] = {}
        for name, cfg in data.get("datasets", {}).items():
            if isinstance(cfg, dict):
                payload = dict(cfg)
                payload.setdefault("name", name)
                datasets[name] = DatasetConfig(**payload)
            elif cfg is None:
                datasets[name] = DatasetConfig(name=name)
            else:
                raise TypeError(f"Dataset config for {name} must be a mapping")

        embedding_jobs = [EmbeddingJobConfig(**job) for job in data.get("embedding_jobs", [])]
        sae_jobs = [SAETrainingConfig(**job) for job in data.get("sae_jobs", [])]
        head_jobs = [HeadTrainingConfig(**job) for job in data.get("head_jobs", [])]

        return cls(
            storage=storage,
            datasets=datasets,
            embedding_jobs=embedding_jobs,
            sae_jobs=sae_jobs,
            head_jobs=head_jobs,
        )


def load_experiment_config(path: Optional[Path | str] = None) -> ExperimentConfig:
    """Load an experiment configuration from disk."""

    candidate_paths: List[Path]
    if path is not None:
        candidate_paths = [Path(path)]
    else:
        candidate_paths = [
            Path("config.yaml"),
            Path.cwd() / "config.yaml",
        ]

    for candidate in candidate_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            if not isinstance(raw, dict):
                raise ValueError(f"Expected mapping at top level of {candidate}, found {type(raw)}")
            return ExperimentConfig.from_dict(raw)

    search_str = ", ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(f"No configuration file found in {search_str}")

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_AUTOINTERP_ASSISTANT_PROMPT = """Background
We are analyzing the activation levels of features in a language model, where each feature activates certain sequences at the end of it.
Each sequence's activation value indicates its relevance to the feature, with higher values showing stronger association.
Task description
Your task is to give this feature a monosemanticity score based on the following scoring rubric:
Activation Consistency
5: Clear pattern with no deviating examples
4: Clear pattern with one or two deviating examples
3: Clear overall pattern but quite a few examples not fitting that pattern
2: Broad consistent theme but lacking structure
1: No discernible pattern
Consider the following activations for a feature in the language model.
Activation: ... Context: ...
Question
Provide your response in the following fixed format:
Explanation: [Your brief explanation]
Score: [5/4/3/2/1]
Now provide your two-line answer."""


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
    dataset: str
    batch_size: int = 16
    max_examples: Optional[int] = None
    tokenizer: Optional[str] = None
    prompt_field: Optional[str] = "prompt"
    response_field: Optional[str] = None
    text_field: Optional[str] = None
    chat_messages_field: Optional[str] = None
    paired_chat_messages_field: Optional[str] = None
    paired_text_field: Optional[str] = None
    chosen_field: Optional[str] = "chosen"
    rejected_field: Optional[str] = "rejected"
    chat_add_generation_prompt: bool = False
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAETrainingConfig:
    """Configuration for training an SAE on a cached embedding dataset."""

    job_id: str
    embedding_job: str
    hidden_size: int
    k_active: int
    learning_rate: float
    batch_size: int
    steps: Optional[int] = None
    epochs: Optional[int] = None
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
    top_k_feature_examples: int = 10
    activation: str = "topk"
    aux_k: Optional[int] = None
    aux_loss_coef: float = 1.0 / 32.0
    prefix_lengths: Optional[List[int]] = None
    batch_topk_threshold_lr: float = 1e-2
    dead_neuron_threshold_steps: int = 256
    normalize_decoder: bool = True
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class AutoInterpJobConfig:
    """Configuration for running LLM-based feature interpretations."""

    job_id: str
    sae_job: str
    model_name: str = "gemini-1.5-pro"
    assistant_prompt: str = DEFAULT_AUTOINTERP_ASSISTANT_PROMPT
    max_examples_per_feature: int = 10
    batch_size: int = 20
    feature_ids: Optional[List[int]] = None
    head_job: Optional[str] = None
    head_top_n: int = 20
    api_key: Optional[str] = None
    max_concepts: Optional[int] = None


@dataclass
class HeadTrainingConfig:
    """Configuration for fitting a prediction head on top of SAE features."""

    job_id: str
    embedding_job: str
    sae_job: Optional[str]
    dataset: str
    head_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    target_field: Optional[str] = None
    rejected_embedding_job: Optional[str] = None
    preference_chosen_field: str = "chosen"
    preference_rejected_field: str = "rejected"
    preference_weight_field: Optional[str] = None


@dataclass
class HeadEvalConfig:
    """Configuration for evaluating a trained head on a particular embedding job/dataset."""

    job_id: str
    head_job: str
    embedding_job: str
    dataset: str
    split: Optional[str] = None


@dataclass
class BaselineConfig:
    """Configuration for baseline evaluations using HF sequence classification models."""

    batch_size: int = 16


@dataclass
class ExperimentConfig:
    storage: StorageConfig
    datasets: Dict[str, DatasetConfig]
    embedding_jobs: List[EmbeddingJobConfig] = field(default_factory=list)
    sae_jobs: List[SAETrainingConfig] = field(default_factory=list)
    head_jobs: List[HeadTrainingConfig] = field(default_factory=list)
    head_eval_jobs: List[HeadEvalConfig] = field(default_factory=list)
    autointerp_jobs: List[AutoInterpJobConfig] = field(default_factory=list)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

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

        # Backward compatibility: drop deprecated 'layer' key if present
        embedding_jobs: List[EmbeddingJobConfig] = []
        for job in data.get("embedding_jobs", []):
            if isinstance(job, dict) and "layer" in job:
                job = {k: v for k, v in job.items() if k != "layer"}
            embedding_jobs.append(EmbeddingJobConfig(**job))
        sae_jobs = [SAETrainingConfig(**job) for job in data.get("sae_jobs", [])]
        head_jobs = [HeadTrainingConfig(**job) for job in data.get("head_jobs", [])]
        head_eval_jobs = [HeadEvalConfig(**job) for job in data.get("head_eval_jobs", [])]
        autointerp_jobs = [AutoInterpJobConfig(**job) for job in data.get("autointerp_jobs", [])]
        baseline_cfg = BaselineConfig(**data.get("baseline", {}))

        return cls(
            storage=storage,
            datasets=datasets,
            embedding_jobs=embedding_jobs,
            sae_jobs=sae_jobs,
            head_jobs=head_jobs,
            head_eval_jobs=head_eval_jobs,
            autointerp_jobs=autointerp_jobs,
            baseline=baseline_cfg,
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

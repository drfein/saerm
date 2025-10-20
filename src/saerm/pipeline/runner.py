from __future__ import annotations

import logging
from typing import Iterable

from ..config import EmbeddingJobConfig, ExperimentConfig, HeadTrainingConfig, SAETrainingConfig
from ..data.datasets import DatasetManager
from ..embeddings.cache import EmbeddingCacheManager
from ..heads.trainer import HeadTrainer
from ..logging import configure_logging
from ..sae.trainer import SAETrainer
from ..storage import StorageManager

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs embedding caching, SAE training, and head fitting sequentially."""

    def __init__(self, config: ExperimentConfig, log_level: str = "INFO") -> None:
        configure_logging(log_level, freeze_existing=False)
        self._config = config
        self._storage = StorageManager(config.storage)
        self._storage.prepare()
        self._datasets = DatasetManager(config)
        self._cache = EmbeddingCacheManager(self._storage)

    def run_all(self) -> None:
        self.run_embedding_jobs(self._config.embedding_jobs)
        self.run_sae_jobs(self._config.sae_jobs)
        self.run_head_jobs(self._config.head_jobs)

    def run_embedding_jobs(self, jobs: Iterable[EmbeddingJobConfig]) -> None:
        for job in jobs:
            dataset = self._datasets.get(job.dataset, None)
            self._cache.run_job(job, dataset)

    def run_sae_jobs(self, jobs: Iterable[SAETrainingConfig]) -> None:
        for job in jobs:
            trainer = SAETrainer(self._storage, self._cache, job)
            trainer.train()

    def run_head_jobs(self, jobs: Iterable[HeadTrainingConfig]) -> None:
        for job in jobs:
            trainer = HeadTrainer(self._storage, self._cache, self._datasets, job)
            trainer.train()

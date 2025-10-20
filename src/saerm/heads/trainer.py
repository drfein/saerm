from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from ..config import HeadTrainingConfig
from ..data.datasets import DatasetManager
from ..embeddings.cache import EmbeddingCacheManager
from ..sae.inference import SAEFeatureExtractor
from ..storage import StorageManager
from .base import HeadFactory

logger = logging.getLogger(__name__)


class HeadTrainer:
    """Train configurable prediction heads on top of embeddings or SAE features."""

    def __init__(
        self,
        storage: StorageManager,
        cache: EmbeddingCacheManager,
        datasets: DatasetManager,
        job: HeadTrainingConfig,
    ) -> None:
        self._storage = storage
        self._cache = cache
        self._datasets = datasets
        self._job = job

    def train(self) -> Dict[str, float]:
        features = self._prepare_features()
        targets = self._load_targets(features.shape[0])
        head = HeadFactory.create(self._job.head_type, **self._job.params)
        head.fit(features, targets)

        predictions = head.predict(features)
        mse = float(np.mean((predictions - targets) ** 2))
        path = self._storage.head_model_path(self._job.job_id)
        head.save(str(path))
        metadata_path = self._storage.head_metadata_path(self._job.job_id)
        self._storage.write_metadata(metadata_path, {
            "job_id": self._job.job_id,
            "embedding_job": self._job.embedding_job,
            "sae_job": self._job.sae_job,
            "dataset": self._job.dataset,
            "train_split": self._job.train_split,
            "eval_split": self._job.eval_split,
            "target_field": self._job.target_field,
            "metrics": {
                "mse": mse,
            },
            "head_type": self._job.head_type,
            "params": self._job.params,
        })
        logger.info("Trained head %s with MSE %.4f", self._job.job_id, mse)
        return {"mse": mse}

    def _prepare_features(self) -> np.ndarray:
        payload = self._cache.load_embeddings(self._job.embedding_job)
        embeddings = payload["embeddings"].float()
        features = embeddings
        if self._job.sae_job:
            metadata_path = self._storage.sae_metadata_path(self._job.sae_job)
            metadata = self._storage.read_metadata(metadata_path)
            checkpoint = self._storage.sae_checkpoint_path(self._job.sae_job)
            hidden_size = metadata["hidden_size"]
            k_active = metadata["k_active"]
            input_dim = metadata.get("input_dim", embeddings.shape[1])
            extractor = SAEFeatureExtractor(
                checkpoint_path=str(checkpoint),
                input_dim=input_dim,
                hidden_dim=hidden_size,
                k_active=k_active,
                device="cpu",
            )
            features = extractor.transform(embeddings)
        return features.numpy()

    def _load_targets(self, size: int) -> np.ndarray:
        dataset = self._datasets.get(self._job.dataset, self._job.train_split)
        column = dataset[self._job.target_field]
        if len(column) < size:
            logger.warning("Target dataset shorter (%s) than features (%s); truncating", len(column), size)
        truncated = column[:size]
        return np.asarray(truncated, dtype=np.float32)

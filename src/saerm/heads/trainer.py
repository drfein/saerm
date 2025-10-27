from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
        self._sae_extractor: Optional[SAEFeatureExtractor] = None

    def train(self) -> Dict[str, float]:
        dataset = self._datasets.get(self._job.dataset, self._job.train_split)
        if self._should_use_supervised(dataset):
            return self._train_supervised(dataset)
        return self._train_pairwise(dataset)

    def _should_use_supervised(self, dataset) -> bool:
        if not self._job.target_field:
            return False
        if hasattr(dataset, "column_names"):
            return self._job.target_field in dataset.column_names
        # Fallback: attempt to access the column lazily
        try:
            dataset[self._job.target_field]  # type: ignore[index]
            return True
        except Exception:
            return False

    def _train_supervised(self, dataset) -> Dict[str, float]:
        payload = self._cache.load_embeddings(self._job.embedding_job)
        supervised_choice = self._job.params.get("supervised_choice", "chosen")
        features, _ = self._extract_features(payload, choice=str(supervised_choice) if supervised_choice else None)
        targets = self._load_targets(dataset, features.shape[0])
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
            "rejected_embedding_job": self._job.rejected_embedding_job,
            "sae_job": self._job.sae_job,
            "dataset": self._job.dataset,
            "train_split": self._job.train_split,
            "eval_split": self._job.eval_split,
            "target_field": self._job.target_field,
            "preference_chosen_field": self._job.preference_chosen_field,
            "preference_rejected_field": self._job.preference_rejected_field,
            "preference_weight_field": self._job.preference_weight_field,
            "metrics": {
                "mse": mse,
            },
            "head_type": self._job.head_type,
            "params": self._job.params,
        })
        logger.info("Trained head %s with MSE %.4f", self._job.job_id, mse)
        return {"mse": mse}

    def _train_pairwise(self, dataset) -> Dict[str, float]:
        payload = self._cache.load_embeddings(self._job.embedding_job)
        chosen_features, chosen_ids = self._extract_features(payload, choice="chosen")

        if self._job.rejected_embedding_job:
            rejected_payload = self._cache.load_embeddings(self._job.rejected_embedding_job)
            rejected_features, rejected_ids = self._extract_features(rejected_payload, choice="rejected")
        else:
            rejected_features, rejected_ids = self._extract_features(payload, choice="rejected")

        chosen_features, rejected_features, aligned_ids = self._align_features(chosen_features, chosen_ids, rejected_features, rejected_ids)
        if chosen_features.size == 0 or rejected_features.size == 0:
            raise ValueError(f"No overlapping samples between chosen and rejected embeddings for head job {self._job.job_id}")

        sample_weights = self._load_preference_weights(dataset, chosen_features.shape[0], aligned_ids)
        head_params = dict(self._job.params)
        weight_chosen = float(head_params.pop("weight_chosen", 1.0))
        weight_rejected = float(head_params.pop("weight_rejected", 1.0))

        head = HeadFactory.create(self._job.head_type, **head_params)
        if hasattr(head, "fit_pairwise"):
            fit_metrics = head.fit_pairwise(  # type: ignore[attr-defined]
                chosen_features,
                rejected_features,
                sample_weights=sample_weights,
                weight_chosen=weight_chosen,
                weight_rejected=weight_rejected,
            ) or {}
        else:
            # Fallback: train supervised heads by classifying chosen (1) vs rejected (0)
            if sample_weights is not None:
                logger.warning(
                    "Head type %s lacks pairwise training; ignoring pair weights for supervised fallback",
                    self._job.head_type,
                )
            X = np.vstack([chosen_features, rejected_features])
            y = np.concatenate([
                np.ones(chosen_features.shape[0], dtype=np.float32),
                np.zeros(rejected_features.shape[0], dtype=np.float32),
            ])
            head.fit(X, y)
            fit_metrics = {}

        chosen_scores = head.predict(chosen_features)
        rejected_scores = head.predict(rejected_features)
        bt_loss = self._bt_loss(chosen_scores, rejected_scores, sample_weights, weight_chosen, weight_rejected)
        bt_accuracy = self._bt_accuracy(chosen_scores, rejected_scores, sample_weights, weight_chosen, weight_rejected)
        margin = self._bt_margin(chosen_scores, rejected_scores, weight_chosen, weight_rejected)

        path = self._storage.head_model_path(self._job.job_id)
        head.save(str(path))
        metadata_path = self._storage.head_metadata_path(self._job.job_id)
        metrics = {
            "bt_loss": bt_loss,
            "bt_accuracy": bt_accuracy,
            "margin_mean": margin,
        }
        metrics.update({f"fit_{key}": value for key, value in fit_metrics.items()})

        self._storage.write_metadata(metadata_path, {
            "job_id": self._job.job_id,
            "embedding_job": self._job.embedding_job,
            "rejected_embedding_job": self._job.rejected_embedding_job,
            "sae_job": self._job.sae_job,
            "dataset": self._job.dataset,
            "train_split": self._job.train_split,
            "eval_split": self._job.eval_split,
            "target_field": self._job.target_field,
            "preference_chosen_field": self._job.preference_chosen_field,
            "preference_rejected_field": self._job.preference_rejected_field,
            "preference_weight_field": self._job.preference_weight_field,
            "bt_weights": {
                "chosen": weight_chosen,
                "rejected": weight_rejected,
            },
            "metrics": metrics,
            "head_type": self._job.head_type,
            "params": self._job.params,
        })
        logger.info(
            "Trained head %s with BT loss %.4f, accuracy %.4f, margin %.4f",
            self._job.job_id,
            bt_loss,
            bt_accuracy,
            margin,
        )
        return metrics

    def _prepare_features(self, embedding_job_id: str) -> np.ndarray:
        # Deprecated: retained for backward compatibility if needed elsewhere.
        payload = self._cache.load_embeddings(embedding_job_id)
        features, _ = self._extract_features(payload, choice=None)
        return features

    def _get_sae_extractor(self, input_dim: int) -> Optional[SAEFeatureExtractor]:
        if self._job.sae_job is None:
            return None
        if self._sae_extractor is not None:
            return self._sae_extractor
        metadata_path = self._storage.sae_metadata_path(self._job.sae_job)
        metadata = self._storage.read_metadata(metadata_path)
        checkpoint = self._storage.sae_checkpoint_path(self._job.sae_job)
        hidden_size = metadata["hidden_size"]
        k_active = metadata["k_active"]
        inferred_input = metadata.get("input_dim", input_dim)
        self._sae_extractor = SAEFeatureExtractor(
            checkpoint_path=str(checkpoint),
            input_dim=inferred_input,
            hidden_dim=hidden_size,
            k_active=k_active,
            device="cpu",
        )
        return self._sae_extractor

    def _load_targets(self, dataset, size: int) -> np.ndarray:
        if not self._job.target_field:
            raise ValueError("target_field must be set for supervised training")
        column = dataset[self._job.target_field]
        if len(column) < size:
            logger.warning("Target dataset shorter (%s) than features (%s); truncating", len(column), size)
        truncated = column[:size]
        return np.asarray(truncated, dtype=np.float32)

    def _load_preference_weights(self, dataset, size: int, example_ids: Optional[np.ndarray]) -> Optional[np.ndarray]:
        weight_field = self._job.preference_weight_field or self._job.params.get("sample_weight_field")
        if not weight_field:
            return None
        if hasattr(dataset, "column_names") and weight_field not in dataset.column_names:
            logger.warning("Weight field '%s' missing in dataset %s; ignoring weights", weight_field, self._job.dataset)
            return None
        column = dataset[weight_field]
        if example_ids is not None:
            if example_ids.size == 0:
                return None
            max_id = int(example_ids.max())
            if len(column) <= max_id:
                logger.warning(
                    "Weight column shorter (%s) than highest example id (%s); truncating",
                    len(column),
                    max_id,
                )
            selected = [column[i] for i in example_ids if i < len(column)]
            weights = np.asarray(selected, dtype=np.float32)
        else:
            if len(column) < size:
                logger.warning("Weight column shorter (%s) than features (%s); truncating", len(column), size)
            weights = np.asarray(column[:size], dtype=np.float32)
        weights = np.clip(weights, a_min=0.0, a_max=None)
        if weights.size == 0:
            return None
        return weights

    @staticmethod
    def _bt_loss(
        chosen_scores: np.ndarray,
        rejected_scores: np.ndarray,
        sample_weights: Optional[np.ndarray],
        weight_chosen: float,
        weight_rejected: float,
    ) -> float:
        diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
        losses = np.logaddexp(0.0, -diff)
        if sample_weights is not None:
            total = float(np.sum(sample_weights))
            if total <= 0.0:
                return float(np.mean(losses))
            return float(np.sum(losses * sample_weights) / total)
        return float(np.mean(losses))

    @staticmethod
    def _bt_accuracy(
        chosen_scores: np.ndarray,
        rejected_scores: np.ndarray,
        sample_weights: Optional[np.ndarray],
        weight_chosen: float,
        weight_rejected: float,
    ) -> float:
        diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
        correct = diff > 0.0
        if sample_weights is not None:
            total = float(np.sum(sample_weights))
            if total <= 0.0:
                return float(np.mean(correct))
            return float(np.sum(correct * sample_weights) / total)
        return float(np.mean(correct))

    @staticmethod
    def _bt_margin(
        chosen_scores: np.ndarray,
        rejected_scores: np.ndarray,
        weight_chosen: float,
        weight_rejected: float,
    ) -> float:
        diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
        return float(np.mean(diff))

    def _extract_features(self, payload: Dict[str, Any], *, choice: Optional[str]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        embeddings = payload.get("embeddings")
        if embeddings is None:
            raise KeyError("Embedding payload missing 'embeddings'")
        embeddings = embeddings.float()
        records = payload.get("records")
        indices, example_ids = self._select_indices(records, embeddings.shape[0], choice)
        if not indices:
            raise ValueError(f"No embeddings found matching choice {choice!r} for job {self._job.embedding_job}")
        index_tensor = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
        selected = embeddings.index_select(0, index_tensor)
        features = selected
        if self._job.sae_job:
            extractor = self._get_sae_extractor(selected.shape[1])
            if extractor is not None:
                features = extractor.transform(selected)
        return features.cpu().numpy().astype(np.float32, copy=False), example_ids

    def _select_indices(
        self,
        records: Optional[List[Dict[str, Any]]],
        total_rows: int,
        choice: Optional[str],
    ) -> tuple[List[int], Optional[np.ndarray]]:
        if records and len(records) == total_rows:
            if choice is None:
                indices = list(range(total_rows))
            else:
                indices = [idx for idx, entry in enumerate(records) if entry.get("choice") == choice]
            example_ids = np.asarray(
                [int(records[idx].get("example_index", idx)) for idx in indices],
                dtype=np.int64,
            ) if indices else None
            return indices, example_ids

        logger.warning(
            "Embedding payload missing aligned records; falling back to alternating split for choice %s",
            choice or "all",
        )
        if choice == "rejected":
            indices = list(range(1, total_rows, 2))
        elif choice == "chosen":
            indices = list(range(0, total_rows, 2))
        else:
            indices = list(range(total_rows))
        example_ids = np.arange(len(indices), dtype=np.int64) if indices else None
        return indices, example_ids

    def _align_features(
        self,
        chosen_features: np.ndarray,
        chosen_ids: Optional[np.ndarray],
        rejected_features: np.ndarray,
        rejected_ids: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if chosen_ids is None or rejected_ids is None:
            size = min(len(chosen_features), len(rejected_features))
            if len(chosen_features) != len(rejected_features):
                logger.warning(
                    "Feature counts differ (%s vs %s); truncating to %s",
                    len(chosen_features),
                    len(rejected_features),
                    size,
                )
            return chosen_features[:size], rejected_features[:size], None

        chosen_ids = chosen_ids.astype(np.int64, copy=False)
        rejected_ids = rejected_ids.astype(np.int64, copy=False)
        chosen_len = chosen_features.shape[0]
        rejected_len = rejected_features.shape[0]
        chosen_ids = chosen_ids[:chosen_len]
        rejected_ids = rejected_ids[:rejected_len]

        if np.array_equal(chosen_ids, rejected_ids):
            size = min(chosen_len, rejected_len)
            return chosen_features[:size], rejected_features[:size], chosen_ids[:size]

        chosen_index = {int(idx): pos for pos, idx in enumerate(chosen_ids)}
        alignment: List[tuple[int, int]] = []
        for pos, idx in enumerate(rejected_ids):
            mapped = chosen_index.get(int(idx))
            if mapped is not None:
                alignment.append((mapped, pos))
        if not alignment:
            raise ValueError("No overlapping example ids between chosen and rejected embeddings")

        chosen_positions = np.asarray([pair[0] for pair in alignment], dtype=np.int64)
        rejected_positions = np.asarray([pair[1] for pair in alignment], dtype=np.int64)
        aligned_ids = chosen_ids[chosen_positions]

        if len(alignment) < min(chosen_len, rejected_len):
            logger.warning(
                "Only %s aligned examples between chosen (%s) and rejected (%s)",
                len(alignment),
                chosen_len,
                rejected_len,
            )

        return chosen_features[chosen_positions], rejected_features[rejected_positions], aligned_ids

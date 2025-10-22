#!/usr/bin/env python3
"""Evaluate a trained reward model head against a labeled dataset split."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import torch

from saerm.config import load_experiment_config
from saerm.data import DatasetManager
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.heads import HeadFactory
from saerm.logging import configure_logging
from saerm.sae.inference import SAEFeatureExtractor
from saerm.storage import StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--job-id", required=True, help="Head job id to evaluate")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--split", help="Override dataset split for evaluation")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    datasets = DatasetManager(config)
    cache = EmbeddingCacheManager(storage)

    metadata_path = storage.head_metadata_path(args.job_id)
    metadata = storage.read_metadata(metadata_path)

    head_type = metadata["head_type"]
    model_path = storage.head_model_path(args.job_id)
    head = HeadFactory.load(head_type, str(model_path))

    dataset_key = metadata["dataset"]
    split = args.split or metadata.get("eval_split") or metadata.get("train_split")
    dataset = datasets.get(dataset_key, split)

    target_field = metadata.get("target_field")
    rejected_job = metadata.get("rejected_embedding_job")

    if target_field:
        payload = cache.load_embeddings(metadata["embedding_job"])
        features_np, _ = _extract_features(payload, metadata, storage, use_paired=False)
        targets = np.asarray(dataset[target_field][: features_np.shape[0]], dtype=np.float32)

        preds = head.predict(features_np)
        mse = float(np.mean((preds - targets) ** 2))
        corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else float("nan")

        logging.info("Evaluation results for %s", args.job_id)
        logging.info("MSE: %.6f", mse)
        logging.info("Corr: %.6f", corr)
    else:
        chosen_payload = cache.load_embeddings(metadata["embedding_job"])
        chosen_np, chosen_ids = _extract_features(chosen_payload, metadata, storage, use_paired=False)

        if rejected_job:
            rejected_payload = cache.load_embeddings(rejected_job)
            rejected_np, rejected_ids = _extract_features(rejected_payload, metadata, storage, use_paired=False)
        else:
            if "embeddings_paired" not in chosen_payload:
                raise ValueError(
                    f"Head {args.job_id} metadata missing paired embeddings. "
                    "Re-run cache_embeddings with paired output or specify rejected_embedding_job."
                )
            rejected_np, rejected_ids = _extract_features(chosen_payload, metadata, storage, use_paired=True)

        chosen_np, rejected_np, aligned_ids = _align_features(chosen_np, chosen_ids, rejected_np, rejected_ids)
        if chosen_np.size == 0 or rejected_np.size == 0:
            raise ValueError(f"No overlapping samples available to evaluate head {args.job_id}")

        weights = _load_pairwise_weights(dataset, metadata.get("preference_weight_field"), aligned_ids, chosen_np.shape[0])

        weight_chosen = float(metadata.get("bt_weights", {}).get("chosen", 1.0))
        weight_rejected = float(metadata.get("bt_weights", {}).get("rejected", 1.0))

        chosen_scores = head.predict(chosen_np)
        rejected_scores = head.predict(rejected_np)
        bt_loss = _bt_loss(chosen_scores, rejected_scores, weights, weight_chosen, weight_rejected)
        bt_accuracy = _bt_accuracy(chosen_scores, rejected_scores, weights, weight_chosen, weight_rejected)
        margin = _bt_margin(chosen_scores, rejected_scores, weight_chosen, weight_rejected)

        logging.info("Pairwise evaluation results for %s", args.job_id)
        logging.info("BT loss: %.6f", bt_loss)
        logging.info("BT accuracy: %.6f", bt_accuracy)
        logging.info("Margin mean: %.6f", margin)


def _build_sae_extractor(storage: StorageManager, metadata: dict, input_dim: int) -> Optional[SAEFeatureExtractor]:
    if not metadata.get("sae_job"):
        return None
    sae_meta = storage.read_metadata(storage.sae_metadata_path(metadata["sae_job"]))
    return SAEFeatureExtractor(
        checkpoint_path=str(storage.sae_checkpoint_path(metadata["sae_job"])),
        input_dim=sae_meta.get("input_dim", input_dim),
        hidden_dim=sae_meta["hidden_size"],
        k_active=sae_meta.get("k_active"),
        device="cpu",
    )


def _bt_loss(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    losses = np.logaddexp(0.0, -diff)
    if sample_weights is not None and sample_weights.size > 0:
        total = float(np.sum(sample_weights))
        if total > 0.0:
            return float(np.sum(losses * sample_weights) / total)
    return float(np.mean(losses))


def _bt_accuracy(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    correct = diff > 0.0
    if sample_weights is not None and sample_weights.size > 0:
        total = float(np.sum(sample_weights))
        if total > 0.0:
            return float(np.sum(correct * sample_weights) / total)
    return float(np.mean(correct))


def _bt_margin(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    return float(np.mean(diff))


def _extract_features(
    payload: dict,
    metadata: dict,
    storage: StorageManager,
    *,
    use_paired: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    key = "embeddings_paired" if use_paired else "embeddings"
    if key not in payload:
        raise KeyError(f"Embedding payload missing key '{key}'")
    embeddings = payload[key].float()
    extractor = _build_sae_extractor(storage, metadata, embeddings.shape[1])
    if extractor:
        features = extractor.transform(embeddings)
    else:
        features = embeddings
    ids_key = "paired_example_ids" if use_paired else "example_ids"
    ids = payload.get(ids_key)
    ids_array = None
    if ids is not None:
        if isinstance(ids, torch.Tensor):
            ids_array = ids.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            ids_array = np.asarray(ids, dtype=np.int64)
    return features.cpu().numpy().astype(np.float32, copy=False), ids_array


def _align_features(
    chosen_features: np.ndarray,
    chosen_ids: Optional[np.ndarray],
    rejected_features: np.ndarray,
    rejected_ids: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if chosen_ids is None or rejected_ids is None:
        size = min(len(chosen_features), len(rejected_features))
        if len(chosen_features) != len(rejected_features):
            logging.warning(
                "Feature counts differ (%s vs %s); truncating to %s",
                len(chosen_features),
                len(rejected_features),
                size,
            )
        return chosen_features[:size], rejected_features[:size], None

    chosen_ids = chosen_ids.astype(np.int64, copy=False)
    rejected_ids = rejected_ids.astype(np.int64, copy=False)
    chosen_ids = chosen_ids[: chosen_features.shape[0]]
    rejected_ids = rejected_ids[: rejected_features.shape[0]]

    if np.array_equal(chosen_ids, rejected_ids):
        size = min(chosen_features.shape[0], rejected_features.shape[0])
        return chosen_features[:size], rejected_features[:size], chosen_ids[:size]

    chosen_lookup = {int(idx): pos for pos, idx in enumerate(chosen_ids)}
    alignment = [(chosen_lookup[int(idx)], pos) for pos, idx in enumerate(rejected_ids) if int(idx) in chosen_lookup]
    if not alignment:
        raise ValueError("No overlapping example ids between chosen and rejected embeddings")

    chosen_positions = np.asarray([item[0] for item in alignment], dtype=np.int64)
    rejected_positions = np.asarray([item[1] for item in alignment], dtype=np.int64)
    aligned_ids = chosen_ids[chosen_positions]

    if len(alignment) < min(chosen_features.shape[0], rejected_features.shape[0]):
        logging.warning(
            "Only %s aligned examples between chosen (%s) and rejected (%s)",
            len(alignment),
            chosen_features.shape[0],
            rejected_features.shape[0],
        )

    return chosen_features[chosen_positions], rejected_features[rejected_positions], aligned_ids


def _load_pairwise_weights(
    dataset,
    weight_field: Optional[str],
    example_ids: Optional[np.ndarray],
    fallback_size: int,
) -> Optional[np.ndarray]:
    if not weight_field:
        return None
    if hasattr(dataset, "column_names") and weight_field not in dataset.column_names:
        logging.warning("Weight field '%s' missing in dataset; ignoring weights", weight_field)
        return None
    column = dataset[weight_field]
    if example_ids is not None and example_ids.size > 0:

        max_id = int(example_ids.max())
        if len(column) <= max_id:
            logging.warning(
                "Weight column shorter (%s) than highest example id (%s); truncating",
                len(column),
                max_id,
            )
        values = [column[i] for i in example_ids if i < len(column)]
    else:
        if len(column) < fallback_size:
            logging.warning("Weight column shorter (%s) than features (%s); truncating", len(column), fallback_size)
        values = column[:fallback_size]
    weights = np.asarray(values, dtype=np.float32)
    if weights.size == 0:
        return None
    return np.clip(weights, a_min=0.0, a_max=None)


if __name__ == "__main__":
    main()
